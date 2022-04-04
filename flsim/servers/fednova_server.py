#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from flsim.active_user_selectors.simple_user_selector import (
    ActiveUserSelectorConfig,
    UniformlyRandomActiveUserSelectorConfig,
)
from flsim.channels.base_channel import (
    IdentityChannel,
    IFLChannel,
)
from flsim.channels.message import FedNovaMessage
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.model import IFLModel
from flsim.optimizers.server_optimizers import (
    ServerOptimizerConfig,
    FedAvgOptimizerConfig,
    OptimizerType,
    FedNovaOptimizerConfig,
)
from flsim.servers.aggregator import AggregationType, Aggregator
from flsim.servers.sync_servers import ISyncServer, SyncServerConfig
from flsim.utils.config_utils import fullclassname, init_self_cfg
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate
from omegaconf import OmegaConf


class FedNovaServer(ISyncServer):
    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IFLChannel] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=SyncServerConfig,
            **kwargs,
        )
        assert (
            self.cfg.aggregation_type == AggregationType.WEIGHTED_SUM
        ), "FedNova must use WEIGHTED_SUM as aggregation type"

        self._optimizer = OptimizerType.create_optimizer(
            config=self.cfg.server_optimizer,
            model=global_model.fl_get_module(),
        )
        self._global_model: IFLModel = global_model
        self._aggregator: Aggregator = Aggregator(
            module=global_model.fl_get_module(),
            aggregation_type=self.cfg.aggregation_type,
            only_federated_params=self.cfg.only_federated_params,
        )
        self._active_user_selector = instantiate(self.cfg.active_user_selector)
        self._channel: IFLChannel = channel or IdentityChannel()
        self.prob = 0
        self.num_samples = 0
        self.messages = []

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        if OmegaConf.is_missing(cfg.active_user_selector, "_target_"):
            cfg.active_user_selector = UniformlyRandomActiveUserSelectorConfig()
        if OmegaConf.is_missing(cfg.server_optimizer, "_target_"):
            cfg.server_optimizer = FedAvgOptimizerConfig()

    @property
    def global_model(self):
        return self._global_model

    def select_clients_for_training(
        self,
        num_total_users,
        users_per_round,
        data_provider: Optional[IFLDataProvider] = None,
        epoch: Optional[int] = None,
    ):
        assert (
            data_provider is not None
        ), "Data provider must be passed into FedNovaServer"

        if self.num_samples == 0:
            self.num_samples = sum(
                [user.num_train_examples() for user in data_provider.train_users()]
            )
            self.prob = users_per_round / num_total_users

        selected_clients = self._active_user_selector.get_user_indices(
            num_total_users=num_total_users,
            users_per_round=users_per_round,
            data_provider=data_provider,
            global_model=self.global_model,
            epoch=epoch,
        )
        return selected_clients

    def init_round(self):
        self._aggregator.zero_weights()
        self._optimizer.zero_grad()
        self.messages = []

    # pyre-ignore[]
    def receive_update_from_client(self, message: FedNovaMessage):
        self.messages.append(message)

    def step(self):
        clients_num_data = torch.Tensor(
            [message.num_examples for message in self.messages]
        )
        weights_data = clients_num_data / self.num_samples
        scaling = torch.Tensor([message.num_local_steps for message in self.messages])
        weights_actual = weights_data / scaling

        weights_actual *= sum(weights_data) / sum(weights_actual)
        weights_agg = weights_actual / self.prob
        weights_agg /= sum(weights_agg)

        for message, weight in zip(self.messages, weights_agg):
            self._aggregator.apply_weight_to_update(
                delta=message.model.fl_get_module(), weight=weight
            )
            self._aggregator.add_update(
                delta=message.model.fl_get_module(), weight=weight
            )
        weighted_model = self._aggregator.aggregate()
        FLModelParamUtils.set_gradient(
            model=self._global_model.fl_get_module(),
            reference_gradient=weighted_model,
        )
        self._optimizer.step()


@dataclass
class FedNovaServerConfig(SyncServerConfig):
    _target_: str = fullclassname(FedNovaServer)
    _recursive_: bool = False
    only_federated_params: bool = True
    aggregation_type: AggregationType = AggregationType.WEIGHTED_SUM
    server_optimizer: ServerOptimizerConfig = ServerOptimizerConfig()
    active_user_selector: ActiveUserSelectorConfig = ActiveUserSelectorConfig()
