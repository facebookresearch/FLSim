#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pyre complains about non-default class members in config being not initialized
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from math import log10, sqrt

from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from omegaconf import MISSING


class ExampleWeight(abc.ABC):
    def __init__(self, **kwargs):
        """avg_num_examples is used to 'normalize' the weight, such that
        weight=1 for the average user
        """
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=AsyncExampleWeightConfig,
            **kwargs,
        )

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    @abc.abstractmethod
    def _raw_weight(self, num_examples: float) -> float:
        pass

    def weight(self, num_examples: float) -> float:
        assert num_examples > 0, "Num examples must be positive"
        return self._raw_weight(num_examples) / self._raw_weight(
            # pyre-fixme[16]: `ExampleWeight` has no attribute `cfg`.
            self.cfg.avg_num_examples
        )


class EqualExampleWeight(ExampleWeight):
    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=EqualExampleWeightConfig,
            **kwargs,
        )

        super().__init__(**kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def _raw_weight(self, num_examples: float) -> float:
        return 1.0


class LinearExampleWeight(ExampleWeight):
    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=LinearExampleWeightConfig,
            **kwargs,
        )

        super().__init__(**kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def _raw_weight(self, num_examples: float) -> float:
        return num_examples


class SqrtExampleWeight(ExampleWeight):
    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=SqrtExampleWeightConfig,
            **kwargs,
        )

        super().__init__(**kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def _raw_weight(self, num_examples: float) -> float:
        return sqrt(num_examples)


class Log10ExampleWeight(ExampleWeight):
    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=Log10ExampleWeightConfig,
            **kwargs,
        )

        super().__init__(**kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def _raw_weight(self, num_examples: float) -> float:
        return log10(1 + num_examples)


@dataclass
class AsyncExampleWeightConfig:
    _target_: str = MISSING
    _recursive_: bool = False
    avg_num_examples: int = 1


@dataclass
class EqualExampleWeightConfig(AsyncExampleWeightConfig):
    _target_: str = fullclassname(EqualExampleWeight)


@dataclass
class LinearExampleWeightConfig(AsyncExampleWeightConfig):
    _target_: str = fullclassname(LinearExampleWeight)


@dataclass
class SqrtExampleWeightConfig(AsyncExampleWeightConfig):
    _target_: str = fullclassname(SqrtExampleWeight)


@dataclass
class Log10ExampleWeightConfig(AsyncExampleWeightConfig):
    _target_: str = fullclassname(Log10ExampleWeight)
