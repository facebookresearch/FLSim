#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List

from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.timing.training_duration_distribution import (
    PerExampleGaussianDurationDistributionConfig,
)
from hydra.utils import instantiate
from omegaconf import MISSING


class TimeOutSimulator(abc.ABC):
    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=TimeOutSimulatorConfig,
            **kwargs,
        )

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    @abc.abstractmethod
    def simulate_per_example_training_time(self) -> float:
        """
        return avg unit of time required to process an example
        """
        pass

    @abc.abstractmethod
    def simulate_training_time(self, device_perf: float, num_samples: int) -> float:
        """
        return training time on one client
        """
        pass

    @abc.abstractmethod
    def track_training_time_distribution(self, one_user_training_time: float) -> None:
        """
        update the empirical training time statistics
        """
        pass

    @abc.abstractmethod
    def track_fl_elapsed_time(self, training_time_in_round: List[float]) -> None:
        """
        track total unit of time taken by FL so far given list of user training time
        in the round
        """
        pass

    @abc.abstractmethod
    def user_timeout(self, training_time: float) -> bool:
        """
        time out for one user
        """
        pass

    @abc.abstractmethod
    def stop_fl(self) -> bool:
        """
        stopping condition for entire FL training
        """
        pass

    @property
    @abc.abstractmethod
    def sample_mean_per_user(self) -> float:
        """
        empirical mean of training time per user
        """
        pass

    @property
    @abc.abstractmethod
    def sample_var_per_user(self) -> float:
        """
        empirical variance of training time per user
        """
        pass


class NeverTimeOutSimulator(TimeOutSimulator):
    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=NeverTimeOutSimulatorConfig,
            **kwargs,
        )

        super().__init__(**kwargs)
        self._sample_mean_per_user = 0.0
        self._sample_var_per_user = 0.0

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def simulate_per_example_training_time(self) -> float:
        """
        training finish instantaneously
        """
        return 0.0

    def simulate_training_time(self, device_perf: float, num_samples: int) -> float:
        """
        training finish instantaneously
        """
        return 0.0

    def track_training_time_distribution(self, one_user_training_time: float) -> None:
        pass

    def user_timeout(self, training_time: float) -> bool:
        return False

    @property
    def sample_mean_per_user(self) -> float:
        return 0.0

    @property
    def sample_var_per_user(self) -> float:
        return 0.0

    def track_fl_elapsed_time(self, training_time_in_round: List[float]) -> None:
        pass

    def stop_fl(self) -> bool:
        return False


class GaussianTimeOutSimulator(TimeOutSimulator):
    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=GaussianTimeOutSimulatorConfig,
            **kwargs,
        )

        super().__init__(**kwargs)

        self.duration_distribution_generator = instantiate(
            self.cfg.duration_distribution_generator
        )
        self._num_users_tracked = 0
        self._num_users_succeed = 0
        self._sample_mean_per_user = 0.0
        self._sample_sec_moment_per_user = 0.0
        self._sample_var_per_user = 0.0
        self._fl_stopping_time = self.cfg.fl_stopping_time
        self._fl_total_elapse_time = 0.0

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    @property
    def fl_stopping_time(self):
        return self._fl_stopping_time

    @property
    def sample_mean_per_user(self):
        return self._sample_mean_per_user

    @property
    def sample_var_per_user(self):
        return self._sample_var_per_user

    def simulate_per_example_training_time(self) -> float:
        """
        return an gaussian R.V. representing # unit of time per example
        """
        return self.duration_distribution_generator.bounded_gaussian_sample()

    def simulate_training_time(self, device_perf: float, num_samples: int) -> float:
        """
        return training time on one client
        """
        # pyre-fixme[16]: `GaussianTimeOutSimulator` has no attribute `cfg`.
        return min([device_perf * num_samples, self.cfg.timeout_wall_per_round])

    def track_training_time_distribution(self, one_user_training_time: float) -> None:
        self._num_users_tracked += 1
        # update sample mean
        self._sample_mean_per_user = (
            self._sample_mean_per_user * (self._num_users_tracked - 1)
            + one_user_training_time
        ) / self._num_users_tracked
        # update sample second moment
        self._sample_sec_moment_per_user = (
            self._sample_sec_moment_per_user * (self._num_users_tracked - 1)
            + one_user_training_time ** 2
        ) / self._num_users_tracked
        # update sample variance, with degree of freedom correction
        if self._num_users_tracked > 1:
            self._sample_var_per_user = (
                self._num_users_tracked / (self._num_users_tracked - 1)
            ) * (self._sample_sec_moment_per_user - self._sample_mean_per_user ** 2)

    def user_timeout(self, training_time: float) -> bool:
        """
        training time out for one user
        """
        # pyre-fixme[16]: `GaussianTimeOutSimulator` has no attribute `cfg`.
        return training_time >= self.cfg.timeout_wall_per_round

    def stop_fl(self) -> bool:
        """
        stop entire FL training when total budget exceeds
        """
        return self._fl_total_elapse_time >= self._fl_stopping_time

    def track_fl_elapsed_time(self, training_time_in_round: List[float]) -> None:
        self._fl_total_elapse_time += min(
            # pyre-fixme[16]: `GaussianTimeOutSimulator` has no attribute `cfg`.
            [self.cfg.timeout_wall_per_round, max(training_time_in_round)]
        )


@dataclass
class TimeOutSimulatorConfig:
    _target_: str = MISSING
    _recursive_: bool = False


@dataclass
class NeverTimeOutSimulatorConfig(TimeOutSimulatorConfig):
    _target_: str = fullclassname(NeverTimeOutSimulator)


@dataclass
class GaussianTimeOutSimulatorConfig(TimeOutSimulatorConfig):
    _target_: str = fullclassname(GaussianTimeOutSimulator)
    timeout_wall_per_round: float = 1.0
    fl_stopping_time: float = 1.0
    duration_distribution_generator: PerExampleGaussianDurationDistributionConfig = (
        PerExampleGaussianDurationDistributionConfig()
    )
