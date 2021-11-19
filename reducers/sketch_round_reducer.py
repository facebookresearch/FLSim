#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from flsim.channels.message import Message
from flsim.channels.sketch_channel import SketchChannel
from flsim.common.logger import Logger
from flsim.interfaces.model import IFLModel
from flsim.reducers.base_round_reducer import (
    IFLRoundReducerConfig,
    IFLRoundReducer,
    ReductionType,
    ReductionPrecision,
)
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.count_sketch import CountSketch


class SketchRoundReducer(IFLRoundReducer):

    logger: logging.Logger = Logger.get_logger(__name__)

    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: SketchChannel,
        num_users_per_round: Optional[int] = None,
        total_number_of_users: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        if channel is None or not isinstance(channel, SketchChannel):
            raise AssertionError(f"Expected a SketchChannel, got {channel}")

        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=SketchRoundReducerConfig,
            **kwargs,
        )
        super().__init__(**kwargs)

        device = next(global_model.fl_get_module().parameters()).device
        self.round_cs = CountSketch(
            # pyre-fixme[16]: `SketchChannel` has no attribute `cfg`.
            width=channel.cfg.num_col,
            depth=channel.cfg.num_hash,
            prime=channel.cfg.prime,
            independence=channel.cfg.independence,
            h=channel.h,
            g=channel.g,
            device=device,
        )
        self.channel = channel

        self.num_users_per_round = num_users_per_round
        self.total_number_of_users = total_number_of_users

        self.h = channel.h
        self.g = channel.g

        # pyre-fixme[16]: `SketchRoundReducer` has no attribute `cfg`.
        self.reduction_type = self.cfg.reduction_type
        self.dtype = self.cfg.precision.dtype
        self.sum_weights: torch.Tensor = torch.zeros(1, device=device)
        self.reset(global_model)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def set_num_total_users(self, num_total_users):
        self.total_number_of_users = num_total_users

    def reset(self, ref_model: IFLModel) -> None:
        """
        Initializes / Resets round reducers internals given a model.
        """
        self.round_cs.set_params(ref_model.fl_get_module().state_dict())
        self.sum_weights.fill_(0)

    def collect_update(self, delta: IFLModel, weight: float) -> None:

        # build channel message with model state dict
        # receive through channel
        message = self.channel.client_to_server(
            Message(model=delta, weight=weight, count_sketch=CountSketch())
        )

        # decode channel message, here only get the sketch
        sketch = message.count_sketch

        if not self.is_weighted:
            weight = 1.0

        self.round_cs.linear_comb(1.0, sketch, weight)
        self.sum_weights += weight

    def reduce(self):
        if torch.isclose(
            self.sum_weights, torch.tensor([0.0], device=self.sum_weights.device)
        ):
            raise AssertionError(
                "Cannot call reduce when no updates have been collected."
            )
        self.round_cs.buckets /= self.sum_weights
        return self.round_cs, self.sum_weights

    @property
    def is_weighted(self):
        return self.reduction_type in (
            ReductionType.WEIGHTED_SUM,
            ReductionType.WEIGHTED_AVERAGE,
        )


@dataclass
class SketchRoundReducerConfig(IFLRoundReducerConfig):
    _target_: str = fullclassname(SketchRoundReducer)
    only_federated_params: bool = False
    # TODO: Handle reduction_type and precision T95227332
    reduction_type: ReductionType = ReductionType.WEIGHTED_AVERAGE
    precision: ReductionPrecision = ReductionPrecision.DEFAULT
