#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

import torch
from flsim.clients.base_client import Client
from flsim.clients.dp_client import DPClientConfig, DPClient
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.metrics_reporter import IFLMetricsReporter, Metric
from flsim.interfaces.model import IFLModel
from flsim.reducers.dp_round_reducer import DPRoundReducerConfig
from flsim.trainers.sync_trainer import SyncTrainer, SyncTrainerConfig, Timeline
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from hydra.utils import instantiate
from omegaconf import OmegaConf


class PrivateSyncTrainer(SyncTrainer):
    r"""
    A ``SyncTrainer`` that supports both sample level
    and user level differential privacy (DP).
    """

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
            config_class=PrivateSyncTrainerConfig,
            **kwargs,
        )
        raise ValueError(
            "PrivateSyncTrainer is deprecated. Please use SyncTrainer with SyncDPServerConfig for user-level dp or with DPClientConfig for sample-level DP"
        )
        super().__init__(model=model, cuda_enabled=cuda_enabled, **kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def train(
        self,
        data_provider: IFLDataProvider,
        metric_reporter: IFLMetricsReporter,
        num_total_users: int,
        distributed_world_size: int,
        rank: int = 0,
    ) -> Tuple[IFLModel, Any]:
        dp_round_reducer = instantiate(
            # pyre-fixme[16]: `PrivateSyncTrainer` has no attribute `cfg`.
            self.cfg.reducer,
            global_model=self.global_model(),
            num_users_per_round=self.cfg.users_per_round,
            total_number_of_users=num_total_users,
            channel=self.channel,
        )
        # pyre-ignore[16]
        self.aggregator.init_round(dp_round_reducer)
        return super().train(
            data_provider,
            metric_reporter,
            num_total_users,
            distributed_world_size,
            rank,
        )

    def create_or_get_client_for_data(self, dataset_id: int, datasets: Any):
        """This function is used to create clients in a round. Thus, it
        is called UPR * num_rounds times per training run. Here, we use
        <code>OmegaConf.structured</code> instead of <code>hydra.instantiate</code>
        to minimize the overhead of hydra object creation.
        """
        self.clients[dataset_id] = DPClient(
            # pyre-ignore [16]: `PrivateSyncTrainer` has no attribute `cfg`
            **OmegaConf.structured(self.cfg.client),
            dataset=datasets[dataset_id],
            name=f"client_{dataset_id}",
            timeout_simulator=self._timeout_simulator,
            store_last_updated_model=self.cfg.report_client_metrics,
            channel=self.channel,
        )
        return self.clients[dataset_id]

    def calc_post_aggregation_train_metrics(
        self,
        clients: Iterable[Client],
        model: IFLModel,
        timeline: Timeline,
        metric_reporter: Optional[IFLMetricsReporter],
    ) -> List[Metric]:

        metrics = super().calc_post_aggregation_train_metrics(
            clients, model, timeline, metric_reporter
        )

        # calculate sample level dp privacy loss statistics.
        all_client_eps = torch.Tensor(
            [c.privacy_budget.epsilon for c in clients]  # pyre-ignore
        )
        mean_client_eps = all_client_eps.mean()
        max_client_eps = all_client_eps.max()
        min_client_eps = all_client_eps.min()
        p50_client_eps = torch.median(all_client_eps)
        sample_dp_metrics = Metric.from_args(
            mean=mean_client_eps,
            min=min_client_eps,
            max=max_client_eps,
            median=p50_client_eps,
        )

        # calculate user level dp privacy loss statistics.
        # pyre-ignore[16]
        dp_round_reducer = self.aggregator.reducer
        aggr_eps = dp_round_reducer.privacy_budget.epsilon

        return metrics + [
            Metric("sample level dp (eps)", sample_dp_metrics),
            Metric("user level dp (eps)", aggr_eps),
        ]


@dataclass
class PrivateSyncTrainerConfig(SyncTrainerConfig):
    _target_: str = fullclassname(PrivateSyncTrainer)
    client: DPClientConfig = DPClientConfig()  # For sample-level DP
    # To disable sample-level DP, set DPClientConfig.clipping_value = float("inf")
    reducer: DPRoundReducerConfig = DPRoundReducerConfig()  # For user-level DP
    # To disable user-level DP, set DPRoundReducerConfig.clipping_value = float("inf")
