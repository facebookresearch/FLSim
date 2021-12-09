#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import logging
import sys
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

import torch
from flsim.channels.base_channel import FLChannelConfig
from flsim.channels.communication_stats import (
    ChannelDirection,
)
from flsim.clients.base_client import ClientConfig
from flsim.common.logger import Logger
from flsim.common.timeline import Timeline
from flsim.common.timeout_simulator import (
    TimeOutSimulatorConfig,
    NeverTimeOutSimulatorConfig,
)
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.metrics_reporter import IFLMetricsReporter, Metric, TrainingStage
from flsim.interfaces.model import IFLModel
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.cuda import CudaTransferMinimizer
from hydra.utils import instantiate
from omegaconf import MISSING
from omegaconf import OmegaConf


class FLTrainer(abc.ABC):
    """Base class for FederatedLearning Training"""

    logger: logging.Logger = Logger.get_logger(__name__)

    def __init__(
        self,
        *,
        model: IFLModel,
        cuda_enabled: bool = False,
        **kwargs,
    ):

        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=FLTrainerConfig,
            **kwargs,
        )

        assert (
            # pyre-fixme[16]: `FLTrainer` has no attribute `cfg`.
            self.cfg.eval_epoch_frequency
            <= self.cfg.epochs
        ), "We expect to do at least one eval. However, eval_epoch_frequency:"
        f" {self.cfg.eval_epoch_frequency} > total epochs: {self.cfg.epochs}"

        self._cuda_state_manager = CudaTransferMinimizer(cuda_enabled)
        self._cuda_state_manager.on_trainer_init(model)
        self.cuda_enabled = cuda_enabled

        self._timeout_simulator = instantiate(self.cfg.timeout_simulator)
        self.channel = instantiate(self.cfg.channel)
        self.data_provider = None
        self.num_total_users: int = -1
        # Initialize tracker for measuring communication between the clients and
        # the server if communication metrics are enabled

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        if OmegaConf.is_missing(cfg.timeout_simulator, "_target_"):
            cfg.timeout_simulator = NeverTimeOutSimulatorConfig()
        if OmegaConf.is_missing(cfg.client, "_target_"):
            cfg.client = ClientConfig()
        if OmegaConf.is_missing(cfg.channel, "_target_"):
            cfg.channel = FLChannelConfig()

    @abc.abstractmethod
    def train(
        self,
        data_provider: IFLDataProvider,
        metric_reporter: IFLMetricsReporter,
        num_total_users: int,
        distributed_world_size: int = 1,
        rank: int = 0,
    ) -> Tuple[IFLModel, Any]:
        pass

    def test(
        self, data_iter: Iterable[Any], metric_reporter: IFLMetricsReporter
    ) -> Any:
        return self._test(
            timeline=Timeline(global_round=1),
            data_iter=data_iter,
            model=self.global_model(),
            metric_reporter=metric_reporter,
        )

    def global_model(self) -> IFLModel:
        pass

    def _maybe_run_evaluation(
        self,
        model: IFLModel,
        timeline: Timeline,
        eval_iter: Iterable[Any],
        metric_reporter: IFLMetricsReporter,
        best_metric,
        best_model_state,
    ):
        # pyre-fixme[16]: `FLTrainer` has no attribute `cfg`.
        if not self.cfg.do_eval:
            return best_metric, best_model_state
        if not timeline.tick(self.cfg.eval_epoch_frequency):
            return best_metric, best_model_state
        eval_metric, eval_metric_better_than_prev = self._evaluate(
            timeline=timeline,
            data_iter=eval_iter,
            model=model,
            metric_reporter=metric_reporter,
        )

        # 1) keep the best model so far if metric_reporter.compare_metrics is specified
        # 2) if self.always_keep_trained_model is set as true, ignore the metricsa and
        #    keep the trained model for each epoch
        if self.cfg.always_keep_trained_model or eval_metric_better_than_prev:
            # last_best_epoch = epoch
            self.logger.info(
                f"Found a better model!, current_eval_metric:{eval_metric}"
            )
            best_metric = eval_metric
            model_state = model.fl_get_module().state_dict()
            best_model_state = model_state
        sys.stdout.flush()
        return best_metric, best_model_state

    def _print_training_stats(self, timeline: Timeline) -> None:
        print(f"Train finished Global Round: {timeline.global_round_num()}")

    def _report_train_metrics(
        self,
        model: IFLModel,
        timeline: Timeline,
        metric_reporter: Optional[IFLMetricsReporter] = None,
        extra_metrics: Optional[List[Metric]] = None,
    ) -> None:
        if (
            # pyre-fixme[16]: `FLTrainer` has no attribute `cfg`.
            self.cfg.report_train_metrics
            and metric_reporter is not None
            and timeline.tick(1.0 / self.cfg.train_metrics_reported_per_epoch)
        ):
            self._print_training_stats(timeline)
            metric_reporter.report_metrics(
                model=model,
                reset=True,
                stage=TrainingStage.TRAINING,
                timeline=timeline,
                epoch=timeline.global_round_num(),  # for legacy
                print_to_channels=True,
                extra_metrics=extra_metrics,
            )

    def _calc_post_epoch_communication_metrics(
        self, timeline: Timeline, metric_reporter: Optional[IFLMetricsReporter]
    ):

        if (
            metric_reporter is not None
            and self.channel.cfg.report_communication_metrics
            # pyre-fixme[16]: `FLTrainer` has no attribute `cfg`
            and timeline.tick(1.0 / self.cfg.train_metrics_reported_per_epoch)
        ):
            extra_metrics = [
                Metric(
                    "Client to Server Bytes Sent"
                    if name == ChannelDirection.CLIENT_TO_SERVER
                    else "Server to Client Bytes Sent",
                    tracker.mean(),
                )
                for name, tracker in self.channel.stats_collector.get_channel_stats().items()
            ]
            metric_reporter.report_metrics(
                model=None,
                reset=False,
                stage=TrainingStage.TRAINING,
                timeline=timeline,
                epoch=timeline.global_round_num(),  # for legacy
                print_to_channels=True,
                extra_metrics=extra_metrics,
            )
            self.channel.stats_collector.reset_channel_stats()

    def _evaluate(
        self,
        timeline: Timeline,
        data_iter: Iterable[Any],
        model: IFLModel,
        metric_reporter: IFLMetricsReporter,
    ) -> Tuple[Any, bool]:
        with torch.no_grad():
            self._cuda_state_manager.before_train_or_eval(model)
            model.fl_get_module().eval()
            print(f"Running {timeline} for {TrainingStage.EVAL.name.title()}")

            for _, batch in enumerate(data_iter):
                batch_metrics = model.get_eval_metrics(batch)
                metric_reporter.add_batch_metrics(batch_metrics)

            metrics, found_best_model = metric_reporter.report_metrics(
                model=model,
                reset=True,
                stage=TrainingStage.EVAL,
                timeline=timeline,
                epoch=timeline.global_round_num(),  # for legacy
                print_to_channels=True,
            )
            self._cuda_state_manager.after_train_or_eval(model)
            return metrics, found_best_model

    def _test(
        self,
        timeline: Timeline,
        data_iter: Iterable[Any],
        model: IFLModel,
        metric_reporter: IFLMetricsReporter,
    ) -> Any:
        with torch.no_grad():
            self._cuda_state_manager.before_train_or_eval(model)
            model.fl_get_module().eval()
            print(f"Running {timeline} for {TrainingStage.TEST.name.title()}")

            for _, batch in enumerate(data_iter):
                batch_metrics = model.get_eval_metrics(batch)
                metric_reporter.add_batch_metrics(batch_metrics)

            metrics, _ = metric_reporter.report_metrics(
                model=model,
                reset=True,
                stage=TrainingStage.TEST,
                timeline=timeline,
                epoch=timeline.global_round_num(),  # for legacy
                print_to_channels=True,
            )
            self._cuda_state_manager.after_train_or_eval(model)
            return metrics


@dataclass
class FLTrainerConfig:
    _target_: str = MISSING
    _recursive_: bool = False
    # Training epochs
    epochs: float = 10.0
    # Whether to do evaluation and model selection based on it.
    do_eval: bool = True
    # don't use metric reporter to choose: always keep trained model
    always_keep_trained_model: bool = False
    # client training timeout
    timeout_simulator: TimeOutSimulatorConfig = TimeOutSimulatorConfig()
    # how many times per epoch should we report training metrics
    # numbers greater than 1 help with plotting more precise training curves
    train_metrics_reported_per_epoch: int = 1
    # perform eval to do model selection in every eval_epoch_frequency epochs
    eval_epoch_frequency: float = 1.0
    # Whether metrics on training data should be computed and reported.
    report_train_metrics: bool = True
    report_train_metrics_after_aggregation: bool = False
    use_train_clients_for_aggregation_metrics: bool = True
    # config for the clients
    client: ClientConfig = ClientConfig()
    # config for the channels
    channel: FLChannelConfig = FLChannelConfig()
