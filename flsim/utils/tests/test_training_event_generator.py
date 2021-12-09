#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Type

import numpy as np
import pytest
from flsim.common.pytest_helper import (
    assertEqual,
    assertTrue,
    assertAlmostEqual,
    assertGreaterEqual,
)
from flsim.utils.async_trainer.training_event_generator import (
    AsyncTrainingEventGenerator,
    AsyncTrainingEventGeneratorConfig,
    AsyncTrainingEventGeneratorFromList,
    AsyncTrainingEventGeneratorFromListConfig,
    AsyncTrainingStartTimeDistrConfig,
    ConstantAsyncTrainingStartTimeDistr,
    ConstantAsyncTrainingStartTimeDistrConfig,
    EventTimingInfo,
    IEventGenerator,
    PoissonAsyncTrainingStartTimeDistr,
    PoissonAsyncTrainingStartTimeDistrConfig,
)
from flsim.utils.timing.training_duration_distribution import (
    PerExampleGaussianDurationDistribution,
    PerExampleGaussianDurationDistributionConfig,
    PerUserGaussianDurationDistribution,
    PerUserGaussianDurationDistributionConfig,
)
from omegaconf import OmegaConf


class TestEventDistributionsUtil:
    def test_simulated_training_training_event_generator(self) -> None:
        """Check that EventDistributionFromList works correctly by inputing
        a sample distribution, and confirming that the output is correct
        """
        timing_info1 = EventTimingInfo(prev_event_start_to_current_start=1, duration=2)
        timing_info2 = EventTimingInfo(prev_event_start_to_current_start=2, duration=1)
        timing_info3 = EventTimingInfo(prev_event_start_to_current_start=2, duration=5)
        random_list = [timing_info1, timing_info2, timing_info3]
        distr = AsyncTrainingEventGeneratorFromList(
            **OmegaConf.structured(
                AsyncTrainingEventGeneratorFromListConfig(training_events=random_list)
            )
        )
        assertTrue(
            distr.time_to_next_event_start()
            == timing_info1.prev_event_start_to_current_start
        )
        assertTrue(
            distr.training_duration(num_training_examples=1) == timing_info1.duration
        )
        assertTrue(
            distr.time_to_next_event_start()
            == timing_info2.prev_event_start_to_current_start
        )
        assertTrue(
            distr.training_duration(num_training_examples=1) == timing_info2.duration
        )
        assertTrue(
            distr.time_to_next_event_start()
            == timing_info3.prev_event_start_to_current_start
        )
        assertTrue(
            distr.training_duration(num_training_examples=1) == timing_info3.duration
        )

    def _duration_normality_check(
        self,
        event_generator: IEventGenerator,
        sample_count: int,
        expected_mean: float,
        expected_sd: float,
        epsilon: float,
    ) -> None:
        durations = []
        for _ in range(sample_count):
            durations.append(event_generator.training_duration(num_training_examples=1))
        # normality check doesn't verify mean and variance
        assertAlmostEqual(np.mean(durations), expected_mean, delta=epsilon)
        assertAlmostEqual(np.std(durations), expected_sd, delta=epsilon)

    def test_poisson_training_event_generator(self) -> None:
        """Check that TrainingEventDistritubion makes sense by checking that
        generated event durations follow normal distribution.
        """
        np.random.seed(1)
        # follows Poisson
        event_rate_per_sec, duration_mean, duration_sd = 10, 1, 5
        # set training_duration_min to a very negative number, to not bound the distribution
        duration_distr = PerExampleGaussianDurationDistributionConfig(
            training_duration_mean=duration_mean, training_duration_sd=duration_sd
        )
        training_start_time_distr = PoissonAsyncTrainingStartTimeDistrConfig(
            training_rate=event_rate_per_sec
        )
        distr = AsyncTrainingEventGenerator(
            **OmegaConf.structured(
                AsyncTrainingEventGeneratorConfig(
                    training_start_time_distribution=training_start_time_distr,
                    duration_distribution_generator=duration_distr,
                )
            )
        )
        self._duration_normality_check(
            distr,
            1000,  # sample_count
            duration_mean,
            duration_sd,
            epsilon=1,
        )

    def test_constant_training_event_distribution(self) -> None:
        """Check that ConstantAsyncTrainingStartTimeDistr generates the right
        next_event_time
        """
        np.random.seed(1)
        min_mean = 0.0001
        max_mean = 10
        max_sd = 1
        event_rate_per_sec = np.random.uniform(min_mean, max_mean)
        duration_mean = np.random.uniform(0, max_mean)
        duration_sd = np.random.uniform(0, max_sd)
        training_start_time_distr = ConstantAsyncTrainingStartTimeDistrConfig(
            training_rate=event_rate_per_sec
        )
        duration_distr = PerExampleGaussianDurationDistributionConfig(
            training_duration_mean=duration_mean, training_duration_sd=duration_sd
        )
        distr = AsyncTrainingEventGenerator(
            **OmegaConf.structured(
                AsyncTrainingEventGeneratorConfig(
                    training_start_time_distribution=training_start_time_distr,
                    duration_distribution_generator=duration_distr,
                )
            )
        )
        assertEqual(distr.time_to_next_event_start(), 1 / event_rate_per_sec)
        self._duration_normality_check(
            distr,
            1000,  # sample_count
            duration_mean,
            duration_sd,
            epsilon=1,
        )

    def test_constant_training_event_distribution_zero_sd(self) -> None:
        """Check that ConstantAsyncTrainingStartTimeDistr generates the right
        constant training_duration and next_event_time when SD is zero
        """
        np.random.seed(1)
        min_mean = 0.0001
        max_mean = 10
        for _num_rand_vals in range(10):
            event_rate_per_sec = np.random.uniform(min_mean, max_mean)
            duration_mean = np.random.uniform(0, max_mean)
            training_start_time_distr = ConstantAsyncTrainingStartTimeDistrConfig(
                training_rate=event_rate_per_sec
            )
            duration_distr = PerExampleGaussianDurationDistributionConfig(
                training_duration_mean=duration_mean, training_duration_sd=0
            )
            distr = AsyncTrainingEventGenerator(
                **OmegaConf.structured(
                    AsyncTrainingEventGeneratorConfig(
                        training_start_time_distribution=training_start_time_distr,
                        duration_distribution_generator=duration_distr,
                    )
                )
            )
            # generate duration and time_to_next_event_start two times
            for _ in range(2):
                assertAlmostEqual(
                    distr.training_duration(num_training_examples=1),
                    duration_mean,
                    delta=1e-4,
                )
                assertEqual(distr.time_to_next_event_start(), 1 / event_rate_per_sec)

    def test_training_duration_min_bound(self) -> None:
        """Check that training_duration_min bound is followed"""
        np.random.seed(1)
        max_mean = 10
        max_sd = 1
        # generate 10 random event generators
        for _num_rand_vals in range(10):
            duration_mean = np.random.uniform(0, max_mean)
            duration_sd = np.random.uniform(0, max_sd)
            # choose a duration_min that is likely to be hit often
            duration_min = duration_mean
            per_user_duration_distr = PerUserGaussianDurationDistribution(
                **OmegaConf.structured(
                    PerUserGaussianDurationDistributionConfig(
                        training_duration_mean=duration_mean,
                        training_duration_sd=duration_sd,
                        training_duration_min=duration_min,
                    )
                )
            )
            per_example_duration_distr = PerExampleGaussianDurationDistribution(
                **OmegaConf.structured(
                    PerExampleGaussianDurationDistributionConfig(
                        training_duration_mean=duration_mean,
                        training_duration_sd=duration_sd,
                        training_duration_min=duration_min,
                    )
                )
            )
            # generate 100 random num_examples
            for _ in range(100):
                num_examples = np.random.randint(low=1, high=1e5)
                per_user_gaussian_duration = per_user_duration_distr.training_duration(
                    num_training_examples=num_examples
                )
                assertGreaterEqual(per_user_gaussian_duration, duration_min)
                # for per-example training duration, duration-min_bound applies to each example
                # while training_duration() returns time for each user
                # so actual bound is
                per_example_gaussian_duration = (
                    per_example_duration_distr.training_duration(
                        num_training_examples=num_examples
                    )
                )
                assertGreaterEqual(
                    per_example_gaussian_duration, duration_min * num_examples
                )

    @pytest.mark.parametrize(
        "start_time_distr_config_class, start_time_distr_class",
        [
            (
                PoissonAsyncTrainingStartTimeDistrConfig,
                PoissonAsyncTrainingStartTimeDistr,
            ),
            (
                ConstantAsyncTrainingStartTimeDistrConfig,
                ConstantAsyncTrainingStartTimeDistr,
            ),
        ],
    )
    def test_string_conversion(
        self,
        start_time_distr_config_class: AsyncTrainingStartTimeDistrConfig,
        start_time_distr_class: Type,
    ) -> None:
        """Check that strings are correctly converted to TrainingEventGenerator"""
        training_rate = 1
        duration_mean_sec = 1
        training_duration_sd = 1
        training_start_time_distr = OmegaConf.structured(start_time_distr_config_class)
        training_start_time_distr.training_rate = training_rate
        duration_distr = PerExampleGaussianDurationDistributionConfig(
            training_duration_mean=duration_mean_sec,
            training_duration_sd=training_duration_sd,
        )
        assertEqual(
            AsyncTrainingEventGenerator(
                **OmegaConf.structured(
                    AsyncTrainingEventGeneratorConfig(
                        training_start_time_distribution=training_start_time_distr,
                        duration_distribution_generator=duration_distr,
                    )
                )
            )._training_start_time_distr.__class__,
            start_time_distr_class,
        )
