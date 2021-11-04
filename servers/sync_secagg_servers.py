#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from flsim.channels.base_channel import IFLChannel
from flsim.channels.message import SyncServerMessage
from flsim.interfaces.model import IFLModel
from flsim.secure_aggregation.secure_aggregator import (
    FixedPointConfig,
    SecureAggregator,
    utility_config_flatter,
)
from flsim.servers.aggregator import AggregationType, Aggregator
from flsim.servers.sync_servers import (
    ISyncServer,
    SyncServerConfig,
    OptimizerType,
)
from flsim.utils.config_utils import fullclassname, init_self_cfg
from flsim.utils.fl.common import FLModelParamUtils


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
        self._optimizer: torch.optim.Optimizer = OptimizerType.create_optimizer(
            model=global_model.fl_get_module(),
            config=self.cfg.optimizer,  # pyre-ignore[16]
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

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    @property
    def global_model(self):
        return self._global_model

    def init_round(self):
        self._aggregator.zero_weights()
        self._optimizer.zero_grad()

    def receive_update_from_client(self, message: SyncServerMessage):
        self._aggregator.apply_weight_to_update(
            delta=message.delta, weight=message.weight
        )
        self._secure_aggregator.params_to_fixedpoint(message.delta)
        self._secure_aggregator.apply_noise_mask(message.delta.named_parameters())
        self._aggregator.add_update(delta=message.delta, weight=message.weight)

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
