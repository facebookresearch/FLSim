#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional

from flsim.clients.async_client import AsyncClientDevice
from flsim.common.logger import Logger
from flsim.common.timeline import Timeline
from flsim.common.training_event_handler import AsyncTrainingEventHandler
from flsim.common.training_simulator import AsyncTrainingSimulator
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.metrics_reporter import IFLMetricsReporter, Metric
from flsim.interfaces.model import IFLModel
from flsim.optimizers.async_aggregators import (
    AsyncAggregatorConfig,
    FedAvgWithLRAsyncAggregatorConfig,
    AsyncAggregator,
    HybridAggregator,
)
from flsim.trainers.trainer_base import FLTrainer, FLTrainerConfig
from flsim.utils.async_trainer.async_user_selector import (
    AsyncUserSelectorFactory,
    AsyncUserSelectorType,
)
from flsim.utils.async_trainer.async_weights import AsyncWeight, AsyncWeightConfig
from flsim.utils.async_trainer.training_event_generator import (
    AsyncTrainingEventGeneratorConfig,
    EventGeneratorConfig,
)
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.cuda import GPUMemoryMinimizer, CudaTransferMinimizer
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm


class AsyncTrainer(FLTrainer, AsyncTrainingEventHandler):
    """Implements Async version of FederatedAveraging
    Implemented like AsynchronousSGD, except that we don't have real gradients,
    but we use reconstructed gradients as if they're real gradients

    Attributes:
        epochs (int): Training epochs
        report_train_metrics (bool): Whether metrics on training data should be
            computed and reported.
    Internal attributes:
    """

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
            config_class=AsyncTrainerConfig,
            **kwargs,
        )

        super().__init__(model=model, cuda_enabled=cuda_enabled, **kwargs)
        self._cuda_state_manager = (
            CudaTransferMinimizer(cuda_enabled)
            # pyre-fixme[16]: `AsyncTrainer` has no attribute `cfg`.
            if self.cfg.minimize_cuda_transfer
            else GPUMemoryMinimizer(cuda_enabled)
        )
        self._cuda_state_manager.on_trainer_init(model)
        self.aggregator: AsyncAggregator = instantiate(
            self.cfg.aggregator, global_model=model, channel=self.channel
        )
        AsyncTrainingEventHandler.__init__(self)
        self._training_simulator: Optional[AsyncTrainingSimulator] = None
        self.weight: AsyncWeight = instantiate(self.cfg.async_weight)
        self._event_generator = instantiate(
            self.cfg.training_event_generator,
        )
        # for pyre; declare instance variables (https://fburl.com/88n6i71r)
        # pyre-fixme[8]: Attribute has type `IFLMetricsReporter`; used as `None`.
        self.metric_reporter = None  # type: IFLMetricsReporter
        self.best_metric = None
        self.best_model_state = self.global_model().fl_get_module().state_dict()

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        if OmegaConf.is_missing(cfg.aggregator, "_target_"):
            cfg.aggregator = FedAvgWithLRAsyncAggregatorConfig()
        if OmegaConf.is_missing(cfg.training_event_generator, "_target_"):
            cfg.training_event_generator = AsyncTrainingEventGeneratorConfig()
        if OmegaConf.is_missing(cfg.async_weight, "_target_"):
            cfg.async_weight = AsyncWeightConfig()

    def global_model(self) -> IFLModel:
        """This function makes it explicit that self.global_model() is owned
        by the aggregator, not by ASyncTrainer
        """
        return self.aggregator.global_model

    @property
    def global_round(self):
        return self.aggregator.global_seqnum + 1  # seqnum is 0 based.

    def on_training_start(self, client: AsyncClientDevice) -> None:
        r"""
        Overrides `AsyncTrainingEventHandler.on_training_start` to mark a client starting to train

        Marks the client state as training and copies the current global model
        to the client's local model

        Note:
        -----
        The actual training process doesn't start until on_training_end
        """
        client.training_started(
            model_seqnum=self.aggregator.global_seqnum, init_model=self.global_model()
        )

    def train_and_update_global_model(self, client: AsyncClientDevice) -> None:
        r"""
        Train a single client and aggregate update into the global model
        """
        assert client.local_model is not None
        # 1. zero grad global optimizer
        self.aggregator.zero_grad()
        # 2. train client on local data
        client_delta, final_local_model, num_examples = client.train_local_model(
            self.metric_reporter
        )
        assert num_examples > 0, "Client must have more than one example"
        # 3. get client staleness
        client_staleness = self.aggregator.model_staleness(
            model_seqnum=client.model_seqnum
        )
        # 4. check participation criteria
        if self._can_participate(staleness=client_staleness):
            weight = self.weight.weight(
                num_examples=num_examples, staleness=client_staleness
            )
            # 5. maybe take a global step
            is_global_model_updated = self.aggregator.on_client_training_end(
                client_delta=client_delta,
                final_local_model=final_local_model,
                weight=weight,
            )
            if is_global_model_updated:
                self._global_update_done()

    def _can_participate(self, staleness: float):
        # pyre-fixme[16]: `AsyncTrainer` has no attribute `cfg`.
        return staleness <= self.cfg.max_staleness

    def _num_global_steps_in_epoch(self):
        if isinstance(self.aggregator, HybridAggregator):
            return math.ceil(self.num_total_users / self.aggregator.cfg.buffer_size)
        else:
            return self.num_total_users

    def _global_update_done(self) -> None:
        timeline = self._get_timeline()
        self._report_train_metrics(
            model=self.global_model(),
            timeline=timeline,
            metric_reporter=self.metric_reporter,
            extra_metrics=self._get_training_metrics(),
        )
        self._calc_post_epoch_communication_metrics(
            self._get_timeline(), self.metric_reporter
        )
        (self.best_metric, self.best_model_state,) = FLTrainer._maybe_run_evaluation(
            self,
            self.global_model(),
            timeline,
            self.data_provider.eval_data(),
            self.metric_reporter,
            self.best_metric,
            self.best_model_state,
        )

    def _get_training_metrics(self) -> List[Metric]:
        metrics = Metric.from_args(
            # pyre-fixme[16]: `Optional` has no attribute `queue_stats`.
            Concurrency_Rate=self._training_simulator.queue_stats.avg_pending_jobs(),
            Seqnum_Diff_Mean=self.aggregator.seqnum_tracker.mean(),
            Seqnum_Diff_Std=self.aggregator.seqnum_tracker.standard_deviation(),
            Weight_Mean=self.weight.stats.mean(),
            Weight_Std=self.weight.stats.standard_deviation(),
        )
        return (
            # pyre-ignore[16]: if aggregator is private then we have privacy budget
            metrics + Metric.from_dict(self.aggregator.privacy_budget._asdict())
            if self.aggregator.is_private
            else metrics
        )

    def _get_timeline(self):
        return Timeline(
            global_round=self.global_round,
            rounds_per_epoch=self._num_global_steps_in_epoch(),
        )

    def _print_training_stats(self, timeline: Timeline):
        super()._print_training_stats(timeline)
        print("Async trainer stats:")
        # pyre-fixme[16]: `Optional` has no attribute `print_stats`.
        self._training_simulator.print_stats("\t")
        self.aggregator.seqnum_tracker.print_stats()

    def train(
        self,
        data_provider: IFLDataProvider,
        metric_reporter: IFLMetricsReporter,
        num_total_users: int,
        distributed_world_size: int = 1,
        rank: int = 0,
    ) -> Tuple[IFLModel, Any]:
        """Train and eval a model, the model states will be modified.
        Args:
            train_iter (List[Iterable[Any]]): list of batch iterators of training data,
                each batch iterator represents data from a single 'user'
            eval_iter (Iterable[Any]): batch iterator of evaluation data
            model (Model): model to be trained
            metric_reporter (IFLMetricsReporter): compute metric based on training
                output and report results to console, file.. etc

        Returns:
            model, best_metric: the trained model together with the best metric
        """
        assert (
            rank == 0 and distributed_world_size == 1
        ), "Distributed training not supported yet for AsyncTrainer"

        self.best_metric = None
        self.best_model_state = self.global_model().fl_get_module().state_dict()
        self.data_provider = data_provider
        self.metric_reporter = metric_reporter
        self.num_total_users = data_provider.num_users()
        self.aggregator.set_num_total_users(self.num_total_users)
        user_selector = AsyncUserSelectorFactory.create_users_selector(
            # pyre-fixme[16]: `AsyncTrainer` has no attribute `cfg`.
            self.cfg.async_user_selector_type,
            data_provider,
        )

        self._training_simulator = AsyncTrainingSimulator(
            event_generator=self._event_generator,
            job_scheduler=self,
            user_selector=user_selector,
            num_train_end_events_per_epoch=self.num_total_users,
            shared_client_config=self.cfg.client,
            timeout_simulator=self._timeout_simulator,
            channel=self.channel,
            cuda_manager=self._cuda_state_manager,
        )
        num_int_epochs = math.ceil(self.cfg.epochs)
        for _epoch in tqdm(range(1, num_int_epochs + 1), desc="Epoch", unit="epoch"):
            # pyre-fixme[16]: `Optional` has no attribute `run_one_epoch`.
            self._training_simulator.run_one_epoch()
            # in k-async, up to (k-1) client updates may not have been aggregated
            # into the global model
            had_unaggregated_updates = self.aggregator.on_training_epoch_end()

            if had_unaggregated_updates:
                self._global_update_done()
        self.global_model().fl_get_module().load_state_dict(self.best_model_state)
        return self.global_model(), self.best_metric


@dataclass
class AsyncTrainerConfig(FLTrainerConfig):
    _target_: str = fullclassname(AsyncTrainer)
    aggregator: AsyncAggregatorConfig = AsyncAggregatorConfig()
    training_event_generator: EventGeneratorConfig = EventGeneratorConfig()
    # TODO: async_user_selector_type should be directly instantiable from json_config
    async_user_selector_type: AsyncUserSelectorType = AsyncUserSelectorType.RANDOM
    async_weight: AsyncWeightConfig = AsyncWeightConfig()
    # any client with staleness greater than this number will be rejected
    max_staleness: float = float("inf")
    # Minimize CPU<-->GPU memory bandwidth at the cost of increasing GPU memory consumption,
    # Turning this off will increase training time
    minimize_cuda_transfer: bool = True
