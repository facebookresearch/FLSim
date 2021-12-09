#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import copy
import math
from dataclasses import dataclass
from typing import List

import numpy as np
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.timing.training_duration_distribution import (
    PerExampleGaussianDurationDistributionConfig,
    DurationDistributionConfig,
)
from hydra.utils import instantiate
from omegaconf import MISSING
from omegaconf import OmegaConf


@dataclass
class EventTimingInfo:
    r"""
    Used only for testing
    """
    prev_event_start_to_current_start: int
    duration: int


class IAsyncTrainingStartTimeDistr(abc.ABC):
    """Abstract class for generating training-start events in AsyncFL"""

    def __init__(self, **kwargs) -> None:
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=AsyncTrainingStartTimeDistrConfig,
            **kwargs,
        )
        assert (
            # pyre-fixme[16]: `IAsyncTrainingStartTimeDistr` has no attribute `cfg`.
            self.cfg.training_rate
            > 0
        ), f"Event rate must be positive, got {self.cfg.training_rate}"

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    @abc.abstractmethod
    def time_to_next_event_start(self) -> float:
        pass


class PoissonAsyncTrainingStartTimeDistr(IAsyncTrainingStartTimeDistr):
    """Training start times are poisson distributed"""

    def __init__(self, **kwargs) -> None:
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=PoissonAsyncTrainingStartTimeDistrConfig,
            **kwargs,
        )

        super().__init__(**kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def time_to_next_event_start(self) -> float:
        # if events are poisson distributed,
        # time to next event is exponentially distributed
        # -ln(U)/lambda = time to next event (from CDF of exponential distribution)
        u = np.random.random()
        # pyre-fixme[16]: `PoissonAsyncTrainingStartTimeDistr` has no attribute `cfg`.
        return -(math.log(u)) / self.cfg.training_rate


class ConstantAsyncTrainingStartTimeDistr(IAsyncTrainingStartTimeDistr):
    """Gap between training start times is constant"""

    def __init__(self, **kwargs) -> None:
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=ConstantAsyncTrainingStartTimeDistrConfig,
            **kwargs,
        )

        super().__init__(**kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def time_to_next_event_start(self) -> float:
        # pyre-fixme[16]: `ConstantAsyncTrainingStartTimeDistr` has no attribute `cfg`.
        return 1.0 / self.cfg.training_rate


class IEventGenerator(abc.ABC):
    """Class that generates both training_start and training_duration events"""

    @abc.abstractmethod
    def time_to_next_event_start(self) -> float:
        pass

    @abc.abstractmethod
    def training_duration(self, num_training_examples: int) -> float:
        pass


class AsyncTrainingEventGenerator(IEventGenerator):
    """Class that generates both training_start and training_duration events"""

    def __init__(self, **kwargs) -> None:
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=AsyncTrainingEventGeneratorConfig,
            **kwargs,
        )
        self._validate_cfg()

        self._training_start_time_distr = instantiate(
            # pyre-fixme[16]: `AsyncTrainingEventGenerator` has no attribute `cfg`.
            self.cfg.training_start_time_distribution
        )
        self._training_duration_distr = instantiate(
            self.cfg.duration_distribution_generator
        )

    def _validate_cfg(self):
        # looping over the config fields throws incase of missing field
        for _ in self.cfg.items():
            pass

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        if OmegaConf.is_missing(cfg.training_start_time_distribution, "_target_"):
            cfg.training_start_time_distribution = (
                ConstantAsyncTrainingStartTimeDistrConfig()
            )
        if OmegaConf.is_missing(cfg.duration_distribution_generator, "_target_"):
            cfg.duration_distribution_generator = (
                PerExampleGaussianDurationDistributionConfig()
            )

    def time_to_next_event_start(self) -> float:
        return self._training_start_time_distr.time_to_next_event_start()

    def training_duration(self, num_training_examples: int) -> float:
        return self._training_duration_distr.training_duration(num_training_examples)


class AsyncTrainingEventGeneratorFromList(IEventGenerator):
    """This class simulates TrainingEventGenerator
    It returns time-to-next-event and event-duration from a fixed list
    Useful for writing unit tests for components that use TrainingEventGenerator
    """

    def __init__(self, **kwargs) -> None:
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=AsyncTrainingEventGeneratorFromListConfig,
            **kwargs,
        )
        self._validate_cfg()

        self.distr: List[EventTimingInfo] = copy.deepcopy(
            # pyre-fixme[16]: `AsyncTrainingEventGeneratorFromList` has no attribute
            #  `cfg`.
            list(self.cfg.training_events)
        )
        self.training_events = list(self.cfg.training_events)
        self.current_event: EventTimingInfo = EventTimingInfo(
            prev_event_start_to_current_start=0, duration=0
        )

    def _validate_cfg(self):
        # looping over the config fields throws incase of missing field
        for _ in self.cfg.items():
            pass

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def time_to_next_event_start(self) -> float:
        self.set_next_event()
        return self.current_event.prev_event_start_to_current_start

    def training_duration(self, num_training_examples: int) -> float:
        return self.current_event.duration

    def set_next_event(self):
        if len(self.distr) == 0:
            self.distr = copy.deepcopy(self.training_events)
        self.current_event = self.distr.pop(0)


@dataclass
class AsyncTrainingStartTimeDistrConfig:
    _target_: str = MISSING
    _recursive_: bool = False
    # Average number of devices training per unit time
    training_rate: float = 1.0


@dataclass
class PoissonAsyncTrainingStartTimeDistrConfig(AsyncTrainingStartTimeDistrConfig):
    _target_: str = fullclassname(PoissonAsyncTrainingStartTimeDistr)


@dataclass
class ConstantAsyncTrainingStartTimeDistrConfig(AsyncTrainingStartTimeDistrConfig):
    _target_: str = fullclassname(ConstantAsyncTrainingStartTimeDistr)


@dataclass
class EventGeneratorConfig:
    _target_: str = MISSING
    _recursive_: bool = False


@dataclass
class AsyncTrainingEventGeneratorFromListConfig(EventGeneratorConfig):
    _target_: str = fullclassname(AsyncTrainingEventGeneratorFromList)
    # list of (time-to-next-event-start, event-duration) tuples
    training_events: List[EventTimingInfo] = MISSING


@dataclass
class AsyncTrainingEventGeneratorConfig(EventGeneratorConfig):
    _target_: str = fullclassname(AsyncTrainingEventGenerator)
    training_start_time_distribution: AsyncTrainingStartTimeDistrConfig = (
        AsyncTrainingStartTimeDistrConfig()
    )
    duration_distribution_generator: DurationDistributionConfig = (
        DurationDistributionConfig()
    )
