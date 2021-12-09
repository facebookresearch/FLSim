#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from flsim.channels.base_channel import IdentityChannel
from flsim.interfaces.model import IFLModel
from flsim.reducers.base_round_reducer import RoundReducer, RoundReducerConfig
from flsim.secure_aggregation.secure_aggregator import (
    FixedPointConfig,
    SecureAggregator,
    utility_config_flatter,
)
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.fl.common import FLModelParamUtils
from torch import nn


class SecureRoundReducer(RoundReducer):
    def __init__(
        self,
        *,
        global_model: IFLModel,
        num_users_per_round: Optional[int] = None,
        total_number_of_users: Optional[int] = None,
        channel: Optional[IdentityChannel] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=SecureRoundReducerConfig,
            **kwargs,
        )

        super().__init__(
            global_model=global_model,
            num_users_per_round=num_users_per_round,
            total_number_of_users=total_number_of_users,
            channel=channel,
            name=name,
            **kwargs,
        )
        # pyre-fixme[16]: `SecureRoundReducer` has no attribute `cfg`.
        self.sec_agg_on = self.cfg.fixedpoint is not None
        if self.sec_agg_on:
            self.secure_aggregator = self._init_secure_aggregator(
                self.cfg.fixedpoint, global_model
            )

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def _init_secure_aggregator(
        self,
        fixedpoint: FixedPointConfig,
        global_model: IFLModel,
    ) -> SecureAggregator:
        """
        Initializes a secure aggregation object based on the received config

        Args:
            fixedpoint: config for fixedpoint
            global_model: the global model

        Returns:
            A secure aggregation object based on the config

        Notes:
            Once we allow per-layer config for FixedPoint, we should change the
            fixedpoint to a dictionary or a similar data structure.
        """
        return SecureAggregator(
            utility_config_flatter(
                global_model.fl_get_module(),
                fixedpoint,
            )
        )

    def reduce(self) -> Tuple[nn.Module, float]:
        if not self.sec_agg_on:
            return super().reduce()
        self._reduce_all()
        # Apply secAgg operations related to model aggregate
        self.secure_aggregator.apply_denoise_mask(
            self.reduced_module.named_parameters()
        )
        self.secure_aggregator.params_to_float(self.reduced_module)
        return self.reduced_module, float(self.sum_weights.item())

    def update_reduced_module(self, delta_module: nn.Module, weight: float) -> None:
        if not self.sec_agg_on:
            return super().update_reduced_module(delta_module, weight)
        # TODO num_samples is used as the default weight, this needs revisit
        if not self.is_weighted:
            weight = 1.0
        # Apply secAgg operations related to model updates and
        # reduced_module, which is the model aggregate.
        FLModelParamUtils.multiply_model_by_weight(
            model=delta_module,
            weight=weight,
            model_to_save=delta_module,
            # pyre-fixme[16]: `SecureRoundReducer` has no attribute `cfg`.
            only_federated_params=self.cfg.only_federated_params,
        )  # first, apply weight to delta, to get (delta * weight)

        self.secure_aggregator.params_to_fixedpoint(delta_module)
        self.secure_aggregator.apply_noise_mask(delta_module.named_parameters())
        FLModelParamUtils.add_model(
            model1=self.reduced_module,  # in FixedPoint
            model2=delta_module,  # in FixedPoint
            model_to_save=self.reduced_module,
            only_federated_params=self.cfg.only_federated_params,
        )
        self.sum_weights += weight


@dataclass
class SecureRoundReducerConfig(RoundReducerConfig):
    """
    Contains configurations for a round reducer with Secure Aggregation
    """

    _target_: str = fullclassname(SecureRoundReducer)
    fixedpoint: Optional[FixedPointConfig] = None
