#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from flsim.channels.sketch_channel import SketchChannel
from flsim.common.logger import Logger
from flsim.interfaces.model import IFLModel
from flsim.optimizers.sync_aggregators import SyncAggregator, SyncAggregatorConfig
from flsim.reducers.base_round_reducer import IFLRoundReducer
from flsim.reducers.sketch_round_reducer import (
    SketchRoundReducer,
    SketchRoundReducerConfig,
)
from flsim.utils.config_utils import fullclassname, init_self_cfg
from flsim.utils.count_sketch import CountSketch, clone_count_sketch
from flsim.utils.fl.common import FLModelParamUtils
from omegaconf import OmegaConf


class SketchAggregator(SyncAggregator):
    """
    Implements FetchSGD
    """

    logger: logging.Logger = Logger.get_logger("SketchAggregator")

    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: SketchChannel,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=SketchAggregatorConfig,
            **kwargs,
        )

        super().__init__(global_model=global_model, channel=channel, **kwargs)

        device = next(global_model.fl_get_module().parameters()).device
        self.grad_cs = CountSketch(
            # pyre-fixme[16]: `SketchChannel` has no attribute `cfg`.
            width=channel.cfg.num_col,
            depth=channel.cfg.num_hash,
            prime=channel.cfg.prime,
            independence=channel.cfg.independence,
            h=channel.h,
            g=channel.g,
            device=device,
        )
        self.error_cs = CountSketch(
            width=channel.cfg.num_col,
            depth=channel.cfg.num_hash,
            prime=channel.cfg.prime,
            independence=channel.cfg.independence,
            h=channel.h,
            g=channel.g,
            device=device,
        )
        self.grad_cs.set_params(global_model.fl_get_module().state_dict())
        self.error_cs.set_params(global_model.fl_get_module().state_dict())

        # pyre-fixme[16]: `SketchAggregator` has no attribute `cfg`.
        if self.cfg.top_k_ratio > 1.0 or self.cfg.top_k_ratio < 0.0:
            raise AssertionError("top k ratio must be between 0.0 and 1.0.")
        self.num_params = sum(
            [p.numel() for p in global_model.fl_get_module().parameters()]
        )
        self.top_k = int(self.cfg.top_k_ratio * self.num_params)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        if OmegaConf.is_missing(cfg.reducer, "_target_"):
            cfg.reducer = SketchRoundReducerConfig()

    def step(self) -> Optional[float]:
        """
        Implements FetchSGD steps 11-15.
        """
        reduced_sketch, sum_weights = self.reducer.reduce()

        # momentum
        # pyre-fixme[16]: `SketchAggregator` has no attribute `cfg`.
        self.grad_cs.linear_comb(self.cfg.momentum, reduced_sketch, 1.0)
        # error feedback
        self.error_cs.linear_comb(1.0, self.grad_cs, self.cfg.lr)
        # unsketch
        state_dict = self.error_cs.unsketch_model(k=self.top_k)
        # error accumulation
        topk_cs = clone_count_sketch(self.error_cs)
        topk_cs.sketch_state_dict(state_dict)
        self.error_cs.linear_comb(1.0, topk_cs, -1.0)
        # update global model
        aggregated_module = FLModelParamUtils.clone(self._global_model.fl_get_module())
        FLModelParamUtils.load_state_dict(
            model=aggregated_module, state_dict=state_dict, only_federated_params=False
        )
        FLModelParamUtils.subtract_model(
            minuend=self._global_model.fl_get_module(),
            subtrahend=aggregated_module,
            difference=self._global_model.fl_get_module(),
            only_federated_params=False,
        )

    def init_round(self, reducer: Optional[IFLRoundReducer] = None):
        if reducer is not None and not isinstance(reducer, SketchRoundReducer):
            raise AssertionError(
                f"Must use SketchRoundReducer in SketchAggregator, but got {reducer}"
            )
        if reducer is not None and reducer is not self.reducer:
            self.logger.warning("Changing the round reducer!")
            self.reducer = reducer
        self.reducer.reset(ref_model=self._global_model)


@dataclass
class SketchAggregatorConfig(SyncAggregatorConfig):
    _target_: str = fullclassname(SketchAggregator)
    momentum: float = 0.1
    lr: float = 0.1
    top_k_ratio: float = 1.0
    reducer: SketchRoundReducerConfig = SketchRoundReducerConfig()
