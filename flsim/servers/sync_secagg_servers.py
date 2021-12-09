#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from flsim.active_user_selectors.simple_user_selector import (
    ActiveUserSelectorConfig,
    UniformlyRandomActiveUserSelectorConfig,
)
from flsim.channels.base_channel import (
    IFLChannel,
    IdentityChannel,
)
from flsim.channels.message import Message
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.model import IFLModel
from flsim.optimizers.server_optimizers import (
    FedAvgOptimizerConfig,
)
from flsim.secure_aggregation.secure_aggregator import (
    FixedPointConfig,
    SecureAggregator,
    utility_config_flatter,
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


class SyncSecAggServer(ISyncServer):
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
            config_class=SyncSecAggServerConfig,
            **kwargs,
        )
        self._optimizer = instantiate(
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
        self._secure_aggregator = SecureAggregator(
            utility_config_flatter(
                global_model.fl_get_module(),
                self.cfg.fixedpoint,
            )
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

        self._aggregator.apply_weight_to_update(
            delta=message.model.fl_get_module(), weight=message.weight
        )

        self._secure_aggregator.params_to_fixedpoint(message.model.fl_get_module())
        self._secure_aggregator.apply_noise_mask(
            message.model.fl_get_module().named_parameters()
        )
        self._aggregator.add_update(
            delta=message.model.fl_get_module(), weight=message.weight
        )

    def step(self):
        aggregated_model = self._aggregator.aggregate()
        self._secure_aggregator.apply_denoise_mask(aggregated_model.named_parameters())
        self._secure_aggregator.params_to_float(aggregated_model)

        FLModelParamUtils.set_gradient(
            model=self._global_model.fl_get_module(),
            reference_gradient=aggregated_model,
        )
        self._optimizer.step()


@dataclass
class SyncSecAggServerConfig(SyncServerConfig):
    """
    Contains configurations for a server with Secure Aggregation
    """

    _target_: str = fullclassname(SyncSecAggServer)
    aggregation_type: AggregationType = AggregationType.WEIGHTED_AVERAGE
    fixedpoint: Optional[FixedPointConfig] = None
    active_user_selector: ActiveUserSelectorConfig = ActiveUserSelectorConfig()
