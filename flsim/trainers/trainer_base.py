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
from typing import Any, Iterable, List, Optional, Tuple

import torch
from flsim.channels.base_channel import FLChannelConfig
from flsim.channels.communication_stats import (
    ChannelDirection,
)
from flsim.clients.base_client import ClientConfig
from flsim.clients.sarah_client import SarahClientConfig
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
from flsim.utils.config_utils import is_target
from flsim.utils.cuda import CudaTransferMinimizer
from hydra.utils import instantiate
from omegaconf import MISSING
from omegaconf import OmegaConf
from tqdm import tqdm


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
        self.clients = {}
        self.eval_clients = {}

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
        data_provider,
        metric_reporter: IFLMetricsReporter,
        best_metric,
        best_model_state,
    ):
        # pyre-fixme[16]: `FLTrainer` has no attribute `cfg`.
        if not self.cfg.do_eval:
            return best_metric, best_model_state
        if not timeline.global_round_num() % self.cfg.round_eval_frequency == 0:
            return best_metric, best_model_state

        if self.cfg.personalized:
            # personalized eval on train users
            self._evaluate_personalized_train_users(
                timeline=timeline,
                data_provider=data_provider,
                global_model=model,
                metric_reporter=metric_reporter,
            )

            # personalized eval on eval users
            self._evaluate_personalized_eval_users(
                timeline=timeline,
                data_provider=data_provider,
                global_model=model,
                metric_reporter=metric_reporter,
            )

        # evaluate global model on train users
        eval_metric_train, eval_metric_better_than_prev = self._evaluate_train(
            timeline=timeline,
            data_provider=data_provider,
            global_model=model,
            metric_reporter=metric_reporter,
        )

        # evaluate global model on eval users
        eval_metric, eval_metric_better_than_prev = self._evaluate(
            timeline=timeline,
            data_provider=data_provider,
            global_model=model,
            metric_reporter=metric_reporter,
        )

        # 1) keep the best model so far if metric_reporter.compare_metrics is specified
        # 2) if self.always_keep_trained_model is set as true, ignore the metrics and
        #    keep the trained model for each epoch
        if self.cfg.always_keep_trained_model or eval_metric_better_than_prev:
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

    def _evaluate_train(
        self,
        timeline: Timeline,
        data_provider,
        global_model: IFLModel,  # a global model
        metric_reporter: IFLMetricsReporter,
    ) -> Tuple[Any, bool]:
        """
        Eval of global model on already seen users (trained) users
        """
        print(f"{timeline}: \t Evaluate global model on validation data of train users")

        for client in self.clients.values():
            client.eval(
                model=global_model, metric_reporter=metric_reporter, fine_tune=False
            )

        metrics, found_best_model = metric_reporter.report_metrics(
            model=global_model,
            reset=True,
            stage=TrainingStage.EVAL_TRAIN,
            timeline=timeline,
            print_to_channels=True,
        )
        return metrics, found_best_model

    def _evaluate_personalized_eval_users(
        self,
        timeline: Timeline,
        data_provider,
        global_model: IFLModel,  # a global model
        metric_reporter: IFLMetricsReporter,
    ) -> Tuple[Any, bool]:
        # Finetunes global model for each eval user on their train data
        # and then evaluates performance on local eval data
        print(
            f"{timeline}: \t Evaluate global model w/ finetune on validation data of eval users"
        )

        for idx, eval_user in data_provider.eval_users.items():
            eval_client = instantiate(
                self.cfg.client,
                dataset=eval_user,
                name=f"client_{idx}",
                timeout_simulator=self._timeout_simulator,
                channel=self.channel,
                cuda_manager=self._cuda_state_manager,
            )
            eval_client.eval(
                model=global_model,
                metric_reporter=metric_reporter,
                fine_tune=True,
                personalized_epoch=self.cfg.personalized_epoch,
            )

        metrics, found_best_model = metric_reporter.report_metrics(
            model=global_model,
            reset=True,
            stage=TrainingStage.PERSONALIZED_EVAL_EVAL,
            timeline=timeline,
            print_to_channels=True,
        )
        return metrics, found_best_model

    def _evaluate_personalized_train_users(
        self,
        timeline: Timeline,
        data_provider,
        global_model: IFLModel,
        metric_reporter: IFLMetricsReporter,
    ) -> Tuple[Any, bool]:
        print(
            f"{timeline}: \t Evaluates global model on validation data of train users"
        )
        for train_client in self.clients.values():
            train_client.eval(
                model=global_model,
                metric_reporter=metric_reporter,
                fine_tune=True,
                personalized_epoch=self.cfg.personalized_epoch,
            )

        metrics, found_best_model = metric_reporter.report_metrics(
            model=global_model,
            reset=True,
            stage=TrainingStage.PERSONALIZED_EVAL_TRAIN,
            timeline=timeline,
            print_to_channels=True,
        )

        return metrics, found_best_model

    def _evaluate(
        self,
        timeline: Timeline,
        data_provider: IFLDataProvider,
        global_model: IFLModel,
        metric_reporter: IFLMetricsReporter,
    ) -> Tuple[Any, bool]:
        """
        Evaluate global model on eval users
        """
        with torch.no_grad():
            global_model.fl_get_module().eval()
            print(f"{timeline}: \t Evaluates global model on all data of eval users")

            for _, batch in enumerate(data_provider.eval_data()):
                batch_metrics = global_model.get_eval_metrics(batch)
                metric_reporter.add_batch_metrics(batch_metrics)

            metrics, found_best_model = metric_reporter.report_metrics(
                model=global_model,
                reset=True,
                stage=TrainingStage.EVAL,
                timeline=timeline,
                epoch=timeline.global_round_num(),  # for legacy
                print_to_channels=True,
            )
            return metrics, found_best_model

    def _test(
        self,
        timeline: Timeline,
        data_iter: Iterable[Any],
        model: IFLModel,
        metric_reporter: IFLMetricsReporter,
    ) -> Any:
        for idx, test_user in tqdm(
            self.data_provider.test_users.items(), desc="Personalization Test"
        ):
            test_client = instantiate(
                self.cfg.client,
                dataset=test_user,
                name=f"client_{idx}",
                timeout_simulator=self._timeout_simulator,
                channel=self.channel,
                cuda_manager=self._cuda_state_manager,
            )
            test_client.eval(
                model=model,
                metric_reporter=metric_reporter,
                fine_tune=True,
                personalized_epoch=self.cfg.personalized_epoch,
            )

        p_metrics, _ = metric_reporter.report_metrics(
            model=model,
            reset=True,
            stage=TrainingStage.PERSONALIZED_EVAL_TEST,
            timeline=timeline,
            epoch=timeline.global_round_num(),  # for legacy
            print_to_channels=True,
        )

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
            return {**metrics, **p_metrics}


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
    # frequency of eval based on number of rounds
    round_eval_frequency: int = 10
    # Whether metrics on training data should be computed and reported.
    report_train_metrics: bool = True
    report_train_metrics_after_aggregation: bool = False
    use_train_clients_for_aggregation_metrics: bool = True
    # config for the clients
    client: ClientConfig = ClientConfig()
    # config for the channels
    channel: FLChannelConfig = FLChannelConfig()
    personalized: bool = False
    personalized_epoch: int = 1
