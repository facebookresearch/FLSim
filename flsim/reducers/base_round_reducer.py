#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file defines the concept of a base round aggregator for a
federated learning setting. Also defines basic config,
for an FL aggregator. Aggregator is responsible for just efficient
state_dict gathering and potentially later on user-level differential
privacy. It is different from the concept of the server which modifies
updates the global model.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from enum import IntEnum
from itertools import chain
from typing import Optional, Tuple

import torch
from flsim.channels.base_channel import IdentityChannel
from flsim.channels.message import Message
from flsim.common.logger import Logger
from flsim.interfaces.model import IFLModel
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.distributed.fl_distributed import FLDistributedUtils, OperationType
from flsim.utils.fl.common import FLModelParamUtils
from omegaconf import MISSING
from torch import nn


class ReductionType(IntEnum):
    """
    Type of reduction for the RoundReducer
    """

    AVERAGE = 0
    SUM = 1
    WEIGHTED_AVERAGE = 2
    WEIGHTED_SUM = 3


class ReductionPrecision(IntEnum):
    """
    Defines the precision of the aggregated module.

    It can help a little bit if the reduced module has a double precision.
    DEFAULT keeps the precision of the reference module.
    """

    DEFAULT = 0
    FLOAT = 1
    DOUBLE = 2

    @property
    def dtype(self) -> Optional[torch.dtype]:
        return (
            torch.float64
            if self == ReductionPrecision.DOUBLE
            else torch.float32
            if self == ReductionPrecision.FLOAT
            else None  # ReductionPrecision.DEFAULT
        )


class IFLRoundReducer(abc.ABC):
    """
    Interface for RoundReducers.
    """

    logger: logging.Logger = Logger.get_logger(__name__)

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=IFLRoundReducerConfig,
            **kwargs,
        )

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    @abc.abstractmethod
    def collect_update(self, delta: IFLModel) -> None:
        """
        Given a updated model from the client, add it to this reducer.
        """
        pass

    @abc.abstractmethod
    def reduce(self):
        """
        Reduce all the updates collected thus far and return the results.
        """
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Initializes / Resets round reducers internals.
        """
        pass


@dataclass
class IFLRoundReducerConfig:
    _target_: str = MISSING
    _recursive_: bool = False
    only_federated_params: bool = True


class RoundReducer(IFLRoundReducer):
    """
    Base Class for an aggregator which gets parameters
    from different clients and aggregates them together.
    """

    logger: logging.Logger = Logger.get_logger(__name__)

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
            config_class=RoundReducerConfig,
            **kwargs,
        )
        super().__init__(**kwargs)

        # pyre-fixme[16]: `RoundReducer` has no attribute `cfg`.
        self.dtype = self.cfg.precision.dtype
        self.channel = channel or IdentityChannel()
        self.name = name or "unnamed_aggregator"
        self.num_users_per_round = num_users_per_round
        self.total_number_of_users = total_number_of_users
        # TODO these are specific to mean reducer [this implementation]
        # we will probably need a level of inheritence here and hide
        # these frome the main class.
        self.sum_weights: torch.Tensor = torch.zeros(1)
        self.ref_model: IFLModel = global_model
        self.reduced_module: nn.Module = FLModelParamUtils.clone(
            global_model.fl_get_module(), self.dtype
        )
        self._zero_weights()

    def set_num_total_users(self, num_total_users):
        self.total_number_of_users = num_total_users

    def collect_update(self, delta: IFLModel, weight: float) -> None:
        # 0. Receive delta from client through channel
        delta = self.receive_through_channel(delta)
        # 1. reduce the delta into local state
        self.update_reduced_module(delta.fl_get_module(), weight)

    def _reduce_all(self, op: OperationType = OperationType.SUM_AND_BROADCAST):
        """
        reduce models accross all workers if multi-processing is used.
        reduction type is defined by `config.reduction_type`, see
        `ReductionType`.

        Returns:
            number of the models that have been collected. For weighted
            reductions types returns sum of all model weights.

        Note:
            The weights are sum of weights only for weighted reduction types
            see 'ReductionType`, for simple reduction it is the number of models
            that have been reduced.
        """
        state_dict = FLModelParamUtils.get_state_dict(
            self.reduced_module,
            # pyre-fixme[16]: `RoundReducer` has no attribute `cfg`.
            only_federated_params=self.cfg.only_federated_params,
        )
        FLDistributedUtils.distributed_operation(
            chain([self.sum_weights], state_dict.values()), op
        )

        if self.sum_weights.item() <= 0:
            return 0.0
        total_weight = float(self.sum_weights.item())
        if self.is_averaged:
            # reduced_module = reduced_module / total_weight
            FLModelParamUtils.multiply_model_by_weight(
                model=self.reduced_module,
                weight=1 / total_weight,
                model_to_save=self.reduced_module,
                only_federated_params=self.cfg.only_federated_params,
            )

    def receive_through_channel(self, model: IFLModel) -> IFLModel:
        """
        Receives a reference to a state (referred to as model state_dict)
        over the channel. Any channel effect is applied as part of this
        receive function.
        """
        message = self.channel.client_to_server(Message(model))
        return message.model

    @property
    def current_results(self) -> Tuple[nn.Module, float]:
        return self.reduced_module, float(self.sum_weights.item())

    def reduce(self) -> Tuple[nn.Module, float]:
        self._reduce_all()
        return self.reduced_module, float(self.sum_weights.item())

    def reset(self, ref_model: IFLModel) -> None:
        """
        Initializes / Resets round reducers internals.
        """
        self.ref_model = ref_model
        self._zero_weights()

    def _zero_weights(self):
        """
        Reset parameters and weights to zero
        """
        FLModelParamUtils.zero_weights(
            self.reduced_module, only_federated_params=self.cfg.only_federated_params
        )
        device = next(self.reduced_module.parameters()).device
        self.sum_weights = torch.zeros(1, device=device, dtype=self.dtype)

    def update_reduced_module(self, delta_module: nn.Module, weight: float) -> None:
        # TODO num_samples is used as the default weight, this needs revisit
        if not self.is_weighted:
            weight = 1.0
        FLModelParamUtils.linear_comb_models(
            self.reduced_module,
            1.0,
            delta_module,
            weight,
            self.reduced_module,
            # pyre-fixme[16]: `RoundReducer` has no attribute `cfg`.
            only_federated_params=self.cfg.only_federated_params,
        )
        self.sum_weights += weight

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "L1 norm of aggregated parameters:",
                sum(p.abs().sum() for p in self.reduced_module.parameters()),
            )

    @property
    def is_weighted(self):
        return self.cfg.reduction_type in (
            ReductionType.WEIGHTED_SUM,
            ReductionType.WEIGHTED_AVERAGE,
        )

    @property
    def is_averaged(self):
        return self.cfg.reduction_type in (
            ReductionType.WEIGHTED_AVERAGE,
            ReductionType.AVERAGE,
        )


@dataclass
class RoundReducerConfig(IFLRoundReducerConfig):
    _target_: str = fullclassname(RoundReducer)
    reduction_type: ReductionType = ReductionType.WEIGHTED_AVERAGE
    precision: ReductionPrecision = ReductionPrecision.DEFAULT
