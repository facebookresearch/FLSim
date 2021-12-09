#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import IntEnum

import torch
import torch.nn as nn
from flsim.utils.distributed.fl_distributed import FLDistributedUtils, OperationType
from flsim.utils.fl.common import FLModelParamUtils


class AggregationType(IntEnum):
    """
    Type of averaging for the aggregator
    """

    AVERAGE = 0
    SUM = 1
    WEIGHTED_AVERAGE = 2
    WEIGHTED_SUM = 3


class Aggregator:
    """
    Util class to handle aggregation logic such as
    {weighted, unweighted}_summation, {weighted, unweighted}_averaging

    Please do not extend this class
    """

    def __init__(
        self,
        module: nn.Module,
        aggregation_type: AggregationType,
        only_federated_params: bool = True,
    ):
        self._buffer_module: nn.Module = FLModelParamUtils.clone(module)
        self.device = next(self._buffer_module.parameters()).device
        self._sum_weights: torch.Tensor = torch.zeros(1, device=self.device)
        self.only_federated_params = only_federated_params
        FLModelParamUtils.zero_weights(
            self._buffer_module, only_federated_params=self.only_federated_params
        )
        self.aggregation_type = aggregation_type

    def zero_weights(self):
        FLModelParamUtils.zero_weights(
            self._buffer_module, only_federated_params=self.only_federated_params
        )
        self._sum_weights = torch.zeros(1, device=self.device)

    def add_update(self, delta: nn.Module, weight: float):
        weight = weight if self._is_weighted else 1.0
        FLModelParamUtils.add_model(delta, self._buffer_module, self._buffer_module)
        self._sum_weights += weight

    def apply_weight_to_update(self, delta: nn.Module, weight: float):
        weight = weight if self._is_weighted else 1.0
        FLModelParamUtils.multiply_model_by_weight(
            model=delta,
            weight=weight,
            model_to_save=delta,
        )

    def aggregate(
        self, distributed_op: OperationType = OperationType.SUM_AND_BROADCAST
    ) -> nn.Module:
        FLDistributedUtils.synchronize_across_ranks(
            model=self._buffer_module,
            weights=self._sum_weights,
            operation=distributed_op,
        )

        if self._is_averaged:
            FLModelParamUtils.multiply_model_by_weight(
                model=self._buffer_module,
                weight=1.0 / self.sum_weights.item(),
                model_to_save=self._buffer_module,
            )
        return self._buffer_module

    @property
    def sum_weights(self) -> torch.Tensor:
        return self._sum_weights

    @property
    def _is_weighted(self) -> bool:
        return self.aggregation_type in [
            AggregationType.WEIGHTED_AVERAGE,
            AggregationType.WEIGHTED_SUM,
        ]

    @property
    def _is_averaged(self) -> bool:
        return self.aggregation_type in [
            AggregationType.WEIGHTED_AVERAGE,
            AggregationType.AVERAGE,
        ]
