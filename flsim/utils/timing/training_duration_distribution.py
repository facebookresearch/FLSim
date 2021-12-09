#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import copy
from dataclasses import dataclass, field
from typing import List

import torch
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from omegaconf import MISSING
from torch.distributions.exponential import Exponential
from torch.distributions.half_normal import HalfNormal
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform


@dataclass
class DurationInfo:
    duration: float = 0


class IDurationDistribution(abc.ABC):
    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=DurationDistributionConfig,
            **kwargs,
        )
        validate_args = False if self.cfg.training_duration_sd == 0 else True
        self.gaussian_generator: Normal = Normal(
            torch.tensor([self.cfg.training_duration_mean], dtype=torch.float),
            torch.tensor([self.cfg.training_duration_sd], dtype=torch.float),
            validate_args=validate_args,
        )

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def bounded_gaussian_sample(self) -> float:
        raw_sample = self.gaussian_generator.sample().item()
        # pyre-fixme[16]: `IDurationDistribution` has no attribute `cfg`.
        return max(raw_sample, self.cfg.training_duration_min)

    @abc.abstractmethod
    def training_duration(self, num_training_examples: int) -> float:
        pass


class PerExampleGaussianDurationDistribution(IDurationDistribution):
    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=PerExampleGaussianDurationDistributionConfig,
            **kwargs,
        )
        super().__init__(**kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def training_duration(self, num_training_examples: int) -> float:
        one_example_duration = self.bounded_gaussian_sample()
        return num_training_examples * one_example_duration


class PerUserGaussianDurationDistribution(IDurationDistribution):
    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=PerUserGaussianDurationDistributionConfig,
            **kwargs,
        )
        super().__init__(**kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def training_duration(self, num_training_examples: int) -> float:
        return self.bounded_gaussian_sample()


class PerUserHalfNormalDurationDistribution(IDurationDistribution):
    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=PerUserHalfNormalDurationDistributionConfig,
            **kwargs,
        )
        super().__init__(**kwargs)

        self.generator: HalfNormal = HalfNormal(scale=self.cfg.training_duration_sd)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def training_duration(self, num_training_examples: int) -> float:
        # pyre-fixme[16]: `PerUserHalfNormalDurationDistribution` has no attribute
        #  `cfg`.
        return self.cfg.training_duration_min + self.generator.sample()


class PerUserUniformDurationDistribution(IDurationDistribution):
    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=PerUserUniformDurationDistributionConfig,
            **kwargs,
        )
        super().__init__(**kwargs)

        assert (
            self.cfg.training_duration_sd == 0.0
        ), "Cannot set training duration sd for uniform distribution"
        self.generator: Uniform = Uniform(
            low=self.cfg.training_duration_min,
            high=2 * self.cfg.training_duration_mean - self.cfg.training_duration_min,
        )

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def training_duration(self, num_training_examples: int) -> float:
        # pyre-ignore[20]: default return shape is 1
        return self.generator.sample().item()


class PerUserExponentialDurationDistribution(IDurationDistribution):
    """
    Exponetial Duration where training_duration_mean is the rate parameter
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=PerUserExponentialDurationDistributionConfig,
            **kwargs,
        )
        super().__init__(**kwargs)

        assert (
            self.cfg.training_duration_sd == 0.0
        ), "Cannot set training duration sd for exponetial"
        self.generator: Exponential = Exponential(
            rate=1 / self.cfg.training_duration_mean
        )

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def training_duration(self, num_training_examples: int) -> float:
        # pyre-ignore[20]: default return shape is 1
        return self.generator.sample().item()


class DurationDistributionFromList(IDurationDistribution):
    """
    This class simulates IDurationDistribution
    It returns traing duration from a fixed list
    Useful for writing unit tests for components that use TrainingEventGenerator
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=DurationDistributionFromListConfig,
            **kwargs,
        )
        super().__init__(**kwargs)
        self.training_events = list(self.cfg.training_events)

        self.distr: List[DurationInfo] = copy.deepcopy(self.training_events)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def training_duration(self, num_training_examples: int) -> float:
        return self.distr.pop(0).duration


@dataclass
class DurationDistributionConfig:
    _target_: str = MISSING
    _recursive_: bool = False
    training_duration_mean: float = 0.0
    training_duration_sd: float = 0.0
    training_duration_min: float = float("-inf")


@dataclass
class PerExampleGaussianDurationDistributionConfig(DurationDistributionConfig):
    _target_: str = fullclassname(PerExampleGaussianDurationDistribution)


@dataclass
class PerUserGaussianDurationDistributionConfig(DurationDistributionConfig):
    _target_: str = fullclassname(PerUserGaussianDurationDistribution)


@dataclass
class PerUserHalfNormalDurationDistributionConfig(DurationDistributionConfig):
    _target_: str = fullclassname(PerUserHalfNormalDurationDistribution)


@dataclass
class PerUserUniformDurationDistributionConfig(DurationDistributionConfig):
    _target_: str = fullclassname(PerUserUniformDurationDistribution)


@dataclass
class PerUserExponentialDurationDistributionConfig(DurationDistributionConfig):
    _target_: str = fullclassname(PerUserExponentialDurationDistribution)


@dataclass
class DurationDistributionFromListConfig(DurationDistributionConfig):
    _target_: str = fullclassname(DurationDistributionFromList)
    training_events: List[DurationInfo] = field(default_factory=list)
