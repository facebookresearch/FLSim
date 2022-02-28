#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Algorithm 6:
    Server with partial participation (do not store every vt for every client)
    Randomized coordinate descent for personalized FL (without variance reduction)
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from flsim.active_user_selectors.simple_user_selector import (
    UniformlyRandomActiveUserSelectorConfig,
)
from flsim.channels.base_channel import (
    IdentityChannel,
    IFLChannel,
)
from flsim.channels.message import Message
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.model import IFLModel
from flsim.optimizers.server_optimizers import (
    ServerOptimizerConfig,
    FedAvgOptimizerConfig,
    OptimizerType,
)
from flsim.servers.aggregator import AggregationType, Aggregator
from flsim.servers.sync_servers import (
    ISyncServer,
    SyncServerConfig,
)
from flsim.utils.config_utils import fullclassname, init_self_cfg
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

class CDServer(ISyncServer):
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
            config_class=CDServerConfig,
            **kwargs,
        )
        self._optimizer = OptimizerType.create_optimizer(
            # pyre-fixme[16]: `SyncServer` has no attribute `cfg`.
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
        self.vt = deepcopy(global_model.fl_get_module())
        self.dt = deepcopy(global_model.fl_get_module())
        self.initialized_users = 0
        self.users_per_round = 0
        self.num_users = 0
        self.sum_weights = 0
        # Set the averaged model to be all zeros for now.
        FLModelParamUtils.zero_weights(self.vt)
        FLModelParamUtils.zero_weights(self.dt)
        # AK: setting averaged vt for non-initialized clients to current w^t might accelerate training ?

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
        self.num_users = num_total_users
        self.users_per_round = users_per_round
        self.sum_weights = sum([u.num_examples() for u in data_provider.train_users.values()])
        return self._active_user_selector.get_user_indices(
            num_total_users=num_total_users,
            users_per_round=users_per_round,
            data_provider=data_provider,
            global_model=self.global_model,
            epoch=epoch,
        )

    def init_round(self):
        # reset auxiliary dt
        FLModelParamUtils.zero_weights(self.dt)
        # FLModelParamUtils.zero_weights(self.vt)
        self.sum_weights = torch.zeros(1, device=self._aggregator.device)
        self._optimizer.zero_grad()

    def receive_update_from_client(self, message: Message):
        message = self._channel.client_to_server(message)
        # delta = delta * weight
        FLModelParamUtils.multiply_model_by_weight(
            model=message.model.fl_get_module(),
            weight=message.weight,
            model_to_save=message.model.fl_get_module(),
        )
        # dt += delta
        FLModelParamUtils.add_model(model1=message.model.fl_get_module(), model2=self.dt, model_to_save=self.dt)
        self.sum_weights += message.weight

    def step(self):
        # (1 / n) * sum(dt)
        FLModelParamUtils.multiply_model_by_weight(
            self.dt, 1.0 / self.sum_weights, self.dt
        )
        # vt+1 = vt + 1/n * sum(dt)
        FLModelParamUtils.add_model(self.vt, self.dt, self.vt)
        # print(f"VT {[p for p in self.vt.parameters()]}")

        # (1-eta*lambda)*wt
        wt_left = deepcopy(self.vt)
        FLModelParamUtils.multiply_model_by_weight(
            self._global_model.fl_get_module(),
            weight=(1 - self.cfg.server_optimizer.lr * self.cfg.lambda_),
            model_to_save=wt_left,
        )

        # eta * lambda * wt
        wt_right = deepcopy(self.vt)
        FLModelParamUtils.multiply_model_by_weight(
            self.vt,
            weight=self.cfg.server_optimizer.lr * self.cfg.lambda_,
            model_to_save=wt_right,
        )

        # wt+1 = (1-eta*lambda)*wt + (eta * lambda * wt)
        FLModelParamUtils.add_model(wt_right, wt_left, self._global_model.fl_get_module())
        # print(f"Global CD {[p for p in self._global_model.fl_get_module().parameters()]}")
        # FLModelParamUtils.set_gradient(
        #     model=self._global_model.fl_get_module(),
        #     reference_gradient=self.vt,
        # )
        # self._optimizer.step()


@dataclass
class CDServerConfig(SyncServerConfig):
    _target_: str = fullclassname(CDServer)
    aggregation_type: AggregationType = AggregationType.SUM
    server_optimizer: ServerOptimizerConfig = ServerOptimizerConfig()
    lambda_: float = 1.0
