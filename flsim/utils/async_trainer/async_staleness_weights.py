#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
from dataclasses import dataclass

from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from omegaconf import MISSING


class StalenessWeight(abc.ABC):
    def __init__(self, **kwargs):
        """avg_staleness is used to 'normalize' the weight, such that
        weight=1 when staleness=avg_staleness
        """
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=AsyncStalenessWeightConfig,
            **kwargs,
        )

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    @abc.abstractmethod
    def _raw_weight(self, staleness: int) -> float:
        pass

    def weight(self, staleness: int) -> float:
        assert staleness >= 0, "Staleness must be non-negative"
        # pyre-fixme[16]: `StalenessWeight` has no attribute `cfg`.
        return self._raw_weight(staleness) / self._raw_weight(self.cfg.avg_staleness)


class ConstantStalenessWeight(StalenessWeight):
    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=ConstantStalenessWeightConfig,
            **kwargs,
        )
        super().__init__(**kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def _raw_weight(self, staleness: int) -> float:
        return 1.0


class ThresholdStalenessWeight(StalenessWeight):
    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=ThresholdStalenessWeightConfig,
            **kwargs,
        )
        super().__init__(**kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def _raw_weight(self, staleness: int) -> float:
        # pyre-fixme[16]: `ThresholdStalenessWeight` has no attribute `cfg`.
        if staleness <= self.cfg.cutoff:
            return 1.0
        else:
            return self.cfg.value_after_cutoff


class PolynomialStalenessWeight(StalenessWeight):
    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=PolynomialStalenessWeightConfig,
            **kwargs,
        )
        super().__init__(**kwargs)
        assert (
            self.cfg.exponent <= 1 and self.cfg.exponent >= 0
        ), f"PolynomialExponent must be between 0 and 1, inclusive. Got {self.cfg.exponent}"

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def _raw_weight(self, staleness: int) -> float:
        # pyre-fixme[16]: `PolynomialStalenessWeight` has no attribute `cfg`.
        denom = (1 + staleness) ** self.cfg.exponent
        return 1 / denom


@dataclass
class AsyncStalenessWeightConfig:
    _target_: str = MISSING
    _recursive_: bool = False
    avg_staleness: int = 1


@dataclass
class ConstantStalenessWeightConfig(AsyncStalenessWeightConfig):
    _target_: str = fullclassname(ConstantStalenessWeight)


@dataclass
class ThresholdStalenessWeightConfig(AsyncStalenessWeightConfig):
    _target_: str = fullclassname(ThresholdStalenessWeight)
    cutoff: int = MISSING
    value_after_cutoff: float = MISSING


@dataclass
class PolynomialStalenessWeightConfig(AsyncStalenessWeightConfig):
    _target_: str = fullclassname(PolynomialStalenessWeight)
    exponent: float = MISSING
