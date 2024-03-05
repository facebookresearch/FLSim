#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from flsim.active_user_selectors.simple_user_selector import (
    UniformlyRandomActiveUserSelectorConfig,
)
from flsim.channels.base_channel import IdentityChannel, IFLChannel
from flsim.channels.message import Message
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.model import IFLModel
from flsim.optimizers.server_optimizers import FedAvgOptimizerConfig
from flsim.privacy.common import PrivacyBudget, PrivacySetting
from flsim.privacy.privacy_engine import IPrivacyEngine
from flsim.privacy.privacy_engine_factory import NoiseType, PrivacyEngineFactory
from flsim.privacy.user_update_clip import IUserClipper
from flsim.servers.aggregator import AggregationType, Aggregator
from flsim.servers.sync_servers import ISyncServer, SyncServerConfig
from flsim.utils.config_utils import fullclassname, init_self_cfg
from flsim.utils.distributed.fl_distributed import FLDistributedUtils, OperationType
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate
from omegaconf import OmegaConf


class SyncDPSGDServer(ISyncServer):
    """
    User level DP-SGD Server implementing https://arxiv.org/abs/1710.06963

    Args:
        global_model: IFLModel: Global (server model) to be updated between rounds
        users_per_round: int: User per round to calculate sampling rate
        num_total_users: int: Total users in the dataset to calculate sampling rate
        Sampling rate = users_per_round / num_total_users
        channel: Optional[IFLChannel]: Communication channel between server and clients
    """

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
        assert (
            self.cfg.aggregation_type == AggregationType.AVERAGE  # pyre-ignore[16]
        ), "DP training must be done with simple averaging and uniform weights."

        self.privacy_budget = PrivacyBudget()
        self._optimizer = instantiate(
            config=self.cfg.server_optimizer,
            model=global_model.fl_get_module(),
        )
        self._global_model: IFLModel = global_model
        self._clipping_value = self.cfg.privacy_setting.clipping.clipping_value

        self._user_update_clipper = IUserClipper.create_clipper(
            self.cfg.privacy_setting
        )
        self._aggregator: Aggregator = Aggregator(
            module=global_model.fl_get_module(),
            aggregation_type=self.cfg.aggregation_type,
            only_federated_params=self.cfg.only_federated_params,
        )
        self._privacy_engine: Optional[IPrivacyEngine] = None
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
        global_round_num: Optional[int] = None,
    ):
        if self._privacy_engine is None:
            self._privacy_engine: IPrivacyEngine = PrivacyEngineFactory.create(
                # pyre-ignore[16]
                self.cfg.privacy_setting,
                users_per_round,
                num_total_users,
                noise_type=NoiseType.GAUSSIAN,
            )
            self._privacy_engine.attach(self._global_model.fl_get_module())
        return self._active_user_selector.get_user_indices(
            num_total_users=num_total_users,
            users_per_round=users_per_round,
            data_provider=data_provider,
            global_round_num=global_round_num,
        )

    def init_round(self):
        self._aggregator.zero_weights()
        self._optimizer.zero_grad()
        self._privacy_engine.attach(self._global_model.fl_get_module())
        self._user_update_clipper.reset_clipper_stats()

    def receive_update_from_client(self, message: Message):
        message = self._channel.client_to_server(message)

        self._aggregator.apply_weight_to_update(
            delta=message.model.fl_get_module(), weight=message.weight
        )

        self._user_update_clipper.clip(message.model.fl_get_module())

        self._aggregator.add_update(
            delta=message.model.fl_get_module(), weight=message.weight
        )

    def mock_step(self):
        """
        Populates the grad attributes of the global model without performing
        an optimization step and returns the aggregated *sum* and not avearge.
        Useful for the CANIFE Empirical Privacy Measurement method.
        """

        # mock step that updates the gradients of the global model
        self.step(mock=True)

        # scale by sum_weights
        model = self._global_model.fl_get_module()
        FLModelParamUtils.multiply_gradient_by_weight(
            model,
            self._aggregator.sum_weights.item(),
            model,
        )

        return model

    def step(self, mock=False):
        assert self._privacy_engine is not None, "PrivacyEngine is not initialized"

        aggregated_model = self._aggregator.aggregate(distributed_op=OperationType.SUM)

        if FLDistributedUtils.is_master_worker():
            self._privacy_engine.add_noise(
                aggregated_model,
                self._user_update_clipper.max_norm
                / self._aggregator.sum_weights.item(),
            )

        FLDistributedUtils.synchronize_model_across_workers(
            operation=OperationType.BROADCAST,
            model=aggregated_model,
            weights=self._aggregator.sum_weights,
        )

        FLModelParamUtils.set_gradient(
            model=self._global_model.fl_get_module(),
            reference_gradient=aggregated_model,
        )

        if not mock:
            self._optimizer.step()
            self.privacy_budget = self._privacy_engine.get_privacy_spent()
            self._user_update_clipper.update_clipper_stats()


@dataclass
class SyncDPSGDServerConfig(SyncServerConfig):
    _target_: str = fullclassname(SyncDPSGDServer)
    aggregation_type: AggregationType = AggregationType.AVERAGE
    privacy_setting: PrivacySetting = PrivacySetting()
