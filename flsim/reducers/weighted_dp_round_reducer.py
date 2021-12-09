#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from flsim.channels.base_channel import IdentityChannel
from flsim.interfaces.model import IFLModel
from flsim.reducers.dp_round_reducer import DPRoundReducer, DPRoundReducerConfig
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.fl.common import FLModelParamUtils
from omegaconf import MISSING
from torch import nn


class EstimatorType(IntEnum):
    UNBIASED = 0
    BIASED = 1


class WeightedDPRoundReducer(DPRoundReducer):
    r"""
    A differentially private round reducer that allows client models
    to provide weighted updates.

    There are two different estimators supported `BIASED` and `UNBIASED`, which only
    differ when in average reduction. For sum reduction both sensitivities
    are the same and are equivalent to ``max_weight * clipping_value``.
    """

    def __init__(
        self,
        *,
        global_model: IFLModel,
        num_users_per_round: int,
        total_number_of_users: int,
        channel: Optional[IdentityChannel] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=WeightedDPRoundReducerConfig,
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
        assert self.is_weighted, "Please use DPRoundReducer for unweighted cases"

        # pyre-fixme[16]: `WeightedDPRoundReducer` has no attribute `cfg`.
        self.min_weight = self.cfg.min_weight
        self.max_weight = self.cfg.max_weight
        self.mean_weight = self.cfg.mean_weight
        self.estimator_type = self.cfg.estimator_type
        self._check_boundaries()

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def _check_boundaries(self):
        """
        Checks for min, max, and mean values of config
        """
        if (
            self.min_weight < 0
            or self.max_weight < self.min_weight
            or self.mean_weight < self.min_weight
            or self.mean_weight > self.max_weight
        ):
            self.logger.error("Weight boundaries in config are not defined properly")
        if self.estimator_type == EstimatorType.UNBIASED and self.mean_weight <= 0:
            self.logger.error(
                "For unbiased sensitivity estimation mean_weight needs to be positive."
            )

    def clamp_weight(self, weight: float) -> float:
        if not self.min_weight <= weight <= self.max_weight:
            self.logger.error(
                "min/max client weight boundaries are violated!"
                "Client weight is being adjusted"
            )
            weight = max(min(weight, self.max_weight), self.min_weight)
        return weight

    def update_reduced_module(self, delta_module: nn.Module, weight: float) -> None:
        weight = self.clamp_weight(weight)
        super().update_reduced_module(delta_module, weight)

    def check_total_weight(self, total_weight: float):
        r"""
        Boundary check for total weights.
        """
        lower_bound = self.num_users_per_round * self.min_weight
        upper_bound = self.num_users_per_round * self.max_weight
        is_bounded = lower_bound <= total_weight <= upper_bound
        if not is_bounded:
            self.logger.error(
                f"Summed weights {total_weight} do not fall within expected range [{lower_bound}, {upper_bound}]"
            )

    def sensitivity(self, total_weight: float):
        r"""
        Calculates the sensitivity of the final result.

        Note:
        Sensitivity for weighted averaging may modify the result to decrease
        sensitivity for BIASED case.
        """
        self.check_total_weight(total_weight)
        if not self.is_averaged:
            return self._sum_estimator()
        elif self.estimator_type == EstimatorType.UNBIASED:
            return self._unbiased_estimator()
        else:
            return self._biased_estimator(total_weight=total_weight)

    def _sum_estimator(self) -> float:
        return self.clipping_value * self.max_weight

    def _unbiased_estimator(self) -> float:
        """
        For weighted averaged reductions, the unbiased estimator calculates the true
        weighted average of the models and the sensitivity of it will be:
            (clipping_value * max_weight) / (min_weight * users_per_round)
        """
        return (
            (self.clipping_value * self.max_weight)
            / self.num_users_per_round
            * self.min_weight
        )

    def _biased_estimator(self, total_weight: float) -> float:
        """
        For the biased estimator the weighted average is biased, where the
        average is calculated by weighted sum of the models divided by
        max(num_clients_per_round * mean_weight, total_weight) and
        Sensitivity
        (clipping_value * max_weight) / (mean_weight * num_clients_per_round)
        """
        weight_modifier = total_weight / max(
            total_weight, self.mean_weight * self.num_users_per_round
        )
        FLModelParamUtils.linear_comb_models(
            self.reduced_module,
            weight_modifier,
            self.reduced_module,
            0.0,
            self.reduced_module,
            # pyre-fixme[16]: `WeightedDPRoundReducer` has no attribute `cfg`.
            only_federated_params=self.cfg.only_federated_params,
        )
        return (
            self.clipping_value
            * self.max_weight
            / (self.num_users_per_round * self.mean_weight)
        )


@dataclass
class WeightedDPRoundReducerConfig(DPRoundReducerConfig):
    r"""
    Contains configurations for a private round reducer based that
    also allows for weights.

    Note:
    Allowing weights in dp should generally be avoided unless weights
    are in the same range. If weights are extremely different, one
    might as well throw updates from clients with smaller weights
    away as they will be drawned in noise.
    """
    _target_: str = fullclassname(WeightedDPRoundReducer)
    min_weight: float = 1e-6
    max_weight: float = float("inf")
    mean_weight: float = 1e-6
    estimator_type: EstimatorType = EstimatorType.UNBIASED


@dataclass
class DeltaDirectionWeightedDPReducerConfig(WeightedDPRoundReducerConfig):
    _target_: str = MISSING
