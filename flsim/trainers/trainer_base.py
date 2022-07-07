#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import logging
import sys
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
from flsim.channels.base_channel import FLChannelConfig
from flsim.channels.communication_stats import ChannelDirection
from flsim.clients.base_client import ClientConfig
from flsim.common.fine_tuner import FineTuner
from flsim.common.logger import Logger
from flsim.common.timeline import Timeline
from flsim.common.timeout_simulator import (
    NeverTimeOutSimulatorConfig,
    TimeOutSimulatorConfig,
)
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.metrics_reporter import IFLMetricsReporter, Metric, TrainingStage
from flsim.interfaces.model import IFLModel
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.cuda import CudaTransferMinimizer
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf


class FLTrainer(abc.ABC):
    """Base class for Federated Learning Training"""

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
        self.clients = {}
        self.eval_clients = {}

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        """Set default config if missing.
        Sub-classes may further set default for other components.
        """
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
        metrics_reporter: IFLMetricsReporter,
        num_total_users: int,
        distributed_world_size: int = 1,
        rank: int = 0,
    ) -> Tuple[IFLModel, Any]:
        pass

    def test(
        self, data_provider: IFLDataProvider, metrics_reporter: IFLMetricsReporter
    ) -> Any:
        return self._test(
            timeline=Timeline(global_round=1),
            data_provider=data_provider,
            model=self.global_model(),
            metrics_reporter=metrics_reporter,
        )

    def global_model(self) -> IFLModel:
        pass

    def _maybe_run_evaluation(
        self,
        timeline: Timeline,
        data_provider,
        metrics_reporter: IFLMetricsReporter,
        best_metric,
        best_model_state,
    ):

        # pyre-fixme[16]: `FLTrainer` has no attribute `cfg`.
        # Skip evaluation
        if not self.cfg.do_eval:
            return best_metric, best_model_state
        if not timeline.tick(self.cfg.eval_epoch_frequency):
            return best_metric, best_model_state

        personalized_metrics = {}
        if self.cfg.personalized:
            # Personalized eval on eval users (i.e. finetune for each eval user first)
            # TODO: Fix metrics reporting in order to report personalized metrics
            personalized_metrics = self._evaluate_personalized_eval_users(  # noqa
                timeline=timeline,
                data_provider=data_provider,
                global_model=self.global_model(),
                metrics_reporter=metrics_reporter,
            )

        # Evaluate global model on eval users
        eval_metric, eval_metric_better_than_prev = self._evaluate(
            timeline=timeline,
            data_provider=data_provider,
            global_model=self.global_model(),
            metrics_reporter=metrics_reporter,
        )

        # 1) Keep the best model so far if metrics_reporter.compare_metrics is specified.
        # 2) If self.always_keep_trained_model is True, ignore the metrics and keep the
        #    trained model for each epoch.
        if self.cfg.always_keep_trained_model or eval_metric_better_than_prev:
            best_metric = eval_metric
            model_state = self.global_model().fl_get_module().state_dict()
            best_model_state = model_state
        sys.stdout.flush()
        return best_metric, best_model_state

    def _print_training_stats(self, timeline: Timeline) -> None:
        print(f"Train finished Global Round: {timeline.global_round_num()}")

    def _report_train_metrics(
        self,
        model: IFLModel,
        timeline: Timeline,
        metrics_reporter: Optional[IFLMetricsReporter] = None,
        extra_metrics: Optional[List[Metric]] = None,
    ) -> None:
        """Reports train metrics (e.g. loss and accuracy) after one round.
        Args:
            model: Model to calculate train metrics. Can either be global or client-side
                model.
            timeline: Timeline object that keeps track of current point of time, such as
                current round and epoch.
            metrics_reporter: Metrics reporter object. If none, do not report metrics.
            extra_metrics: Miscellaneous metrics in addition to train loss and scores.
        """
        if (
            # pyre-fixme[16]: `FLTrainer` has no attribute `cfg`.
            self.cfg.report_train_metrics
            and metrics_reporter is not None
            and timeline.tick(1.0 / self.cfg.train_metrics_reported_per_epoch)
        ):
            self._print_training_stats(timeline)
            metrics_reporter.report_metrics(
                model=model,
                reset=True,
                stage=TrainingStage.TRAINING,
                timeline=timeline,
                epoch=timeline.global_round_num(),  # for legacy
                print_to_channels=True,
                extra_metrics=extra_metrics,
            )

    def _calc_post_epoch_communication_metrics(
        self, timeline: Timeline, metrics_reporter: Optional[IFLMetricsReporter]
    ):
        """Calculates communication metrics after an epoch ends, such as amount of data
        communicated bewteen server and client.
        TODO: This should really be called `post_round` instead of `post_epoch`.
        """

        if (
            metrics_reporter is not None
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
            metrics_reporter.report_metrics(
                model=None,
                reset=False,
                stage=TrainingStage.TRAINING,
                timeline=timeline,
                epoch=timeline.global_round_num(),  # for legacy
                print_to_channels=True,
                extra_metrics=extra_metrics,
            )
            self.channel.stats_collector.reset_channel_stats()

    def _evaluate_personalized_eval_users(
        self,
        timeline: Timeline,
        data_provider,
        global_model: IFLModel,  # a global model
        metrics_reporter: IFLMetricsReporter,
    ) -> Any:
        """Perform personalized evaluation on evaluation users.
        Specifically, finetune global model for each eval user on their training data
        first and then evaluate performance on eval users' local eval data.
        """
        print(
            f"{timeline}: \t Evaluate global model w/ finetune on validation data of eval users"
        )
        personalized_metrics, _ = FineTuner.fine_tune_and_evaluate(
            data=self.data_provider.eval_users(),
            global_model=global_model,
            # pyre-ignore[16]
            client_config=self.cfg.client,
            metrics_reporter=metrics_reporter,
            cuda_state_manager=self._cuda_state_manager,
            training_stage=TrainingStage.PERSONALIZED_EVAL,
            timeline=timeline,
            epochs=self.cfg.personalized_epochs,
        )
        return personalized_metrics

    def _evaluate(
        self,
        timeline: Timeline,
        data_provider: IFLDataProvider,
        global_model: IFLModel,
        metrics_reporter: IFLMetricsReporter,
    ) -> Tuple[Any, bool]:
        """
        Evaluate global model on eval users
        """
        with torch.no_grad():
            self._cuda_state_manager.before_train_or_eval(global_model)
            global_model.fl_get_module().eval()
            print(f"{timeline}: \t Evaluates global model on all data of eval users")
            for user in data_provider.eval_users():
                for batch in user.eval_data():
                    batch_metrics = global_model.get_eval_metrics(batch)
                    metrics_reporter.add_batch_metrics(batch_metrics)

            metrics, found_best_model = metrics_reporter.report_metrics(
                model=global_model,
                reset=True,
                stage=TrainingStage.EVAL,
                timeline=timeline,
                epoch=timeline.global_round_num(),  # for legacy
                print_to_channels=True,
            )
            self._cuda_state_manager.after_train_or_eval(global_model)
            return metrics, found_best_model

    def _test(
        self,
        timeline: Timeline,
        data_provider: IFLDataProvider,
        model: IFLModel,
        metrics_reporter: IFLMetricsReporter,
    ) -> Any:

        personalized_metrics = {}
        # pyre-ignore[16]
        if self.cfg.personalized:
            personalized_metrics, _ = FineTuner.fine_tune_and_evaluate(  # noqa
                data=self.data_provider.test_users(),
                global_model=model,
                client_config=self.cfg.client,
                metrics_reporter=metrics_reporter,
                cuda_state_manager=self._cuda_state_manager,
                training_stage=TrainingStage.PERSONALIZED_TEST,
                timeline=timeline,
                epochs=self.cfg.personalized_epochs,
            )

        with torch.no_grad():
            self._cuda_state_manager.before_train_or_eval(model)
            model.fl_get_module().eval()
            print(f"Running {timeline} for {TrainingStage.TEST.name.title()}")

            for test_user in data_provider.test_users():
                for batch in test_user.eval_data():
                    batch_metrics = model.get_eval_metrics(batch)
                    metrics_reporter.add_batch_metrics(batch_metrics)

            metrics, _ = metrics_reporter.report_metrics(
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
    epochs: float = 1000.0
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
    personalized: bool = False  # flag to personalized global model by locally fine tuning before evaluation
    personalized_epochs: int = 1  # number of fine tune epochs to run
