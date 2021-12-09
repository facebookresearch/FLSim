#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Note: Ignore pyre errors here, because we are dynamically instantiating
class attributes for our Configs and Pyre just complains that it cannot
figure out where the attributes are getting initialized. Putting Optional here
is not a choice for us, because we want to differentiate between Optional and
strictly required attributes in our Config classes.
Reference: https://fburl.com/4cdf3akr
"""
from __future__ import annotations

from dataclasses import dataclass

from flsim.utils.async_trainer.async_example_weights import (
    EqualExampleWeightConfig,
    AsyncExampleWeightConfig,
)
from flsim.utils.async_trainer.async_staleness_weights import (
    AsyncStalenessWeightConfig,
    ConstantStalenessWeightConfig,
)
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.fl.stats import RandomVariableStatsTracker
from hydra.utils import instantiate
from omegaconf import OmegaConf


class AsyncWeight:
    def __init__(self, **kwargs) -> None:
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=AsyncWeightConfig,
            **kwargs,
        )
        # pyre-fixme[16]: `AsyncWeight` has no attribute `cfg`.
        self.example_weight = instantiate(self.cfg.example_weight)
        self.staleness_weight = instantiate(self.cfg.staleness_weight)
        self.stats = RandomVariableStatsTracker()

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        if OmegaConf.is_missing(cfg.staleness_weight, "_target_"):
            cfg.staleness_weight = ConstantStalenessWeightConfig()
        if OmegaConf.is_missing(cfg.example_weight, "_target_"):
            cfg.example_weight = EqualExampleWeightConfig()

    def weight(self, num_examples: float, staleness: int) -> float:
        weight = self.example_weight.weight(
            num_examples
        ) * self.staleness_weight.weight(staleness)
        self.stats.update(weight)
        return weight


@dataclass
class AsyncWeightConfig:
    _target_: str = fullclassname(AsyncWeight)
    _recursive_: bool = False
    staleness_weight: AsyncStalenessWeightConfig = AsyncStalenessWeightConfig()
    example_weight: AsyncExampleWeightConfig = AsyncExampleWeightConfig()
