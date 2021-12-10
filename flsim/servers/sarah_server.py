#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, List

from flsim.active_user_selectors.simple_user_selector import (
    ActiveUserSelectorConfig,
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
from flsim.utils.config_utils import fullclassname, init_self_cfg
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate
from omegaconf import OmegaConf
from flsim.servers.sync_servers import (
    ISyncServer,
    SyncServerConfig,
)
from copy import deepcopy

class SarahServer(ISyncServer):
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
            config_class=SyncDPSGDServerConfig,
            **kwargs,
        )
        self._round_number = 0
        self.previous_global_model: IFLModel = global_model
        self._global_model: IFLModel = global_model
        self._aggregator: Aggregator = Aggregator(
            module=global_model.fl_get_module(),
            aggregation_type=self.cfg.aggregation_type,
            only_federated_params=self.cfg.only_federated_params,
        )
        self._active_user_selector = instantiate(self.cfg.active_user_selector)
        self._channel: IFLChannel = channel or IdentityChannel()

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
        """
        Selects clients to participate in a round of training.

        The selection scheme depends on the underlying selector. This
        can include: random, sequential, high loss etc.

        Args:
            num_total_users ([int]): Number of total users (population size).
            users_per_round ([int]]): Number of users per round.
            data_provider (Optional[IFLDataProvider], optional): This is useful when the selection scheme
            is high loss. Defaults to None.
            epoch (Optional[int], optional): [description]. This is useful when the selection scheme
            is high loss. Defaults to None.

        Returns:
            List[int]: A list of client indicies
        """
        if self._round_number + 1 % self.cfg.large_cohort_period == 0:
            return self._active_user_selector.get_user_indices(
                num_total_users=num_total_users,
                users_per_round=self.cfg.large_cohort_clients,
                data_provider=data_provider,
                global_model=self.global_model,
                epoch=epoch,
            )
        else:
            return self._active_user_selector.get_user_indices(
                num_total_users=num_total_users,
                users_per_round=users_per_round,
                data_provider=data_provider,
                global_model=self.global_model,
                epoch=epoch,
            )

    def init_round(self):
        self._aggregator.zero_weights()
        self._optimizer.zero_grad()

    def receive_update_from_client(self, message: Message):
        message = self._channel.client_to_server(message)

        # p * d_{i}^{t} where p = client weight
        self._aggregator.apply_weight_to_update(
            delta=message.model.fl_get_module(), weight=message.weight
        )
        # running some of all d_{i}^{t}
        self._aggregator.add_update(
            delta=message.model.fl_get_module(), weight=message.weight
        )

    def step(self):
        # 1/m \sigma_{i \in S_t} (d_{i}^{t})
        aggregated_model = self._aggregator.aggregate()
        if self._round_number + 1 % self.cfg.large_cohort_period == 0:
            FLModelParamUtils.set_gradient(
                model=self._global_model.fl_get_module(),
                reference_gradient=aggregated_model,
            )
        else:
            # g^t = g^{t-1} + 1/m \sigma_{i \in S_t} (d_{i}^{t})
            FLModelParamUtils.add_model(
                model1=self._global_model.fl_get_module(), 
                model2=aggregated_model
                model_to_save=aggregated_model
            )
            FLModelParamUtils.set_gradient(
                model=self._global_model.fl_get_module(),
                reference_gradient=aggregated_model,
            )

        self.previous_global_model = deepcopy(self._global_model)
        # w^{t+1} = w^{t} - \eta g^{t}
        self._optimizer.step()
        self._round_number += 1

@dataclass
class SarahServerConfig(SyncServerConfig):
    _target_: str = fullclassname(SarahServer)
    # using #examples on client as weight
    aggregation_type: AggregationType = AggregationType.WEIGTHED_AVERAGE 
    large_cohort_period: int = 10
    large_cohort_clients: int 1000