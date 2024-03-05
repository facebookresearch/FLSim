#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from enum import IntEnum

import torch
import torch.nn as nn
from flsim.utils.distributed.fl_distributed import FLDistributedUtils, OperationType
from flsim.utils.fl.common import FLModelParamUtils


class AggregationType(IntEnum):
    """Type of averaging for the aggregator."""

    AVERAGE = 0
    SUM = 1
    WEIGHTED_AVERAGE = 2
    WEIGHTED_SUM = 3


class Aggregator:
    """Util class to handle aggregation logic such as
    {weighted, unweighted}_summation, {weighted, unweighted}_averaging.

    Please do not extend this class.
    """

    def __init__(
        self,
        module: nn.Module,
        aggregation_type: AggregationType,
        only_federated_params: bool = True,
    ):
        """Initializes the aggregator.

        Args:
            module: Target module on which to apply aggregation. We don't care about the
                weights of this module, only its network architecture.
            aggregation_type: Type of aggregation.
            only_federated_params: If True, only update the federated parameters.
        """
        # Buffer to store partially completed aggregation of some of the model deltas
        self._buffer_module = FLModelParamUtils.clone(module)
        self.device = next(self._buffer_module.parameters()).device

        # Sum of aggregation weights applied to each model delta
        self._sum_weights: torch.Tensor = torch.zeros(1, device=self.device)

        self.only_federated_params = only_federated_params

        FLModelParamUtils.zero_weights(
            self._buffer_module, only_federated_params=self.only_federated_params
        )
        self.aggregation_type = aggregation_type

    def zero_weights(self):
        """Zero out the weights (i.e. parameters) of the buffer module and the sum of
        aggregation weights.
        """
        FLModelParamUtils.zero_weights(
            self._buffer_module, only_federated_params=self.only_federated_params
        )
        self._sum_weights = torch.zeros(1, device=self.device)

    def add_update(self, delta: nn.Module, weight: float):
        """Update buffer module by adding the weights of a model delta to it.

        Args:
            delta: Module that contains the model delta in its weights.
            weight: Aggregation weight to apply to this model delta.
        """
        weight = weight if self._is_weighted else 1.0
        FLModelParamUtils.add_model(delta, self._buffer_module, self._buffer_module)
        self._sum_weights += weight

    def apply_weight_to_update(self, delta: nn.Module, weight: float):
        """Add the weights (parameters) of a model delta to the buffer module.

        Args:
            delta: Module whose parameters are the deltas for updating
                `self._buffer_module`'s parameters.
            weight: Weight to apply to `delta`'s parameters.

        Modifies parameters of `delta` in-place.
        """
        weight = weight if self._is_weighted else 1.0
        FLModelParamUtils.multiply_model_by_weight(
            model=delta,
            weight=weight,
            model_to_save=delta,
        )

    def aggregate(
        self, distributed_op: OperationType = OperationType.SUM_AND_BROADCAST
    ) -> nn.Module:
        """Apply aggregation after all model deltas are added. This typically just
        returns the buffer module along with some additional post-processing.
        """
        FLDistributedUtils.synchronize_model_across_workers(
            operation=distributed_op,
            model=self._buffer_module,
            weights=self._sum_weights,
        )

        # Normalize the weights of buffer module if we want to return the average of
        # model deltas as opposed to the sum.
        if self._is_averaged and self.sum_weights.item() != 0:
            FLModelParamUtils.multiply_model_by_weight(
                model=self._buffer_module,
                weight=1.0 / self.sum_weights.item(),
                model_to_save=self._buffer_module,
            )
        return self._buffer_module

    @property
    def sum_weights(self) -> torch.Tensor:
        """Sum of aggregation weights."""
        return self._sum_weights

    @property
    def _is_weighted(self) -> bool:
        """Whether aggregation is (unevenly) weighted, as opposed to averaging."""
        return self.aggregation_type in [
            AggregationType.WEIGHTED_AVERAGE,
            AggregationType.WEIGHTED_SUM,
        ]

    @property
    def _is_averaged(self) -> bool:
        """Whether aggregation uses the average, as opposed to sum."""
        return self.aggregation_type in [
            AggregationType.WEIGHTED_AVERAGE,
            AggregationType.AVERAGE,
        ]
