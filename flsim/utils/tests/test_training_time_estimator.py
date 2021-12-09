#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from flsim.common.pytest_helper import assertEqual, assertAlmostEqual
from flsim.utils.timing.training_duration_distribution import (
    PerUserUniformDurationDistribution,
    PerUserUniformDurationDistributionConfig,
    PerUserHalfNormalDurationDistribution,
    PerUserHalfNormalDurationDistributionConfig,
    DurationDistributionFromListConfig,
    DurationDistributionFromList,
    DurationInfo,
)
from flsim.utils.timing.training_time_estimator import (
    get_training_time,
    AsyncTrainingTimeEstimator,
    SyncTrainingTimeEstimator,
)
from omegaconf import OmegaConf


class TestTrainingTimeEstimator:
    def test_time_from_list(self) -> None:
        """
        Test training time from list

        Assuming UPR = 2

        Sync would be the sum of slowest user between rounds
            round 1
            user_1: duration = 4
            user_2: duration = 3

            round 2
            user_3: duration = 2
            user_4: duration = 1

            total = 4 + 2 = 6

        Async would be the

            user_1: duration = 4, start_time = 1
            user_2: duration = 3, start_time = 1

            user_3: duration = 2, start_time = 2
            user_4: duration = 1, start_time = 3

            users training @ time 1: user 1, user 2
            users training @ time 3: user 2, user 3
            users training @ time 4: user 3, user 4
            users training @ time 5: user 4 finishes training
        """
        training_events = [
            DurationInfo(duration=4),
            DurationInfo(duration=3),
            DurationInfo(duration=2),
            DurationInfo(duration=1),
        ]
        async_start_times = [1, 1, 2, 3]
        sync_training_dist = DurationDistributionFromList(
            **OmegaConf.structured(
                DurationDistributionFromListConfig(training_events=training_events)
            )
        )
        async_training_dist = DurationDistributionFromList(
            **OmegaConf.structured(
                DurationDistributionFromListConfig(training_events=training_events)
            )
        )
        num_users = len(training_events)
        epochs = 1
        users_per_round = 2

        sync_estimator = SyncTrainingTimeEstimator(
            total_users=len(training_events),
            users_per_round=users_per_round,
            epochs=epochs,
            training_dist=sync_training_dist,
        )

        async_estimator = AsyncTrainingTimeEstimator(
            total_users=num_users,
            users_per_round=users_per_round,
            epochs=epochs,
            training_dist=async_training_dist,
            start_times=async_start_times,
        )

        async_time = async_estimator.training_time()
        sync_time = sync_estimator.training_time()
        assertEqual(sync_time, 6)
        assertEqual(async_time, 5)

    def test_uniform_training_time(self) -> None:
        """
        Test uniform training time

        Sync and Async should have the same training time if
        UPR = 1 and duration_min close to duration_mean
        """
        torch.manual_seed(0)
        num_users = 1000
        epochs = 1
        users_per_round = 1
        duration_mean = 1.00
        duration_min = 0.99999

        training_dist = PerUserUniformDurationDistribution(
            **OmegaConf.structured(
                PerUserUniformDurationDistributionConfig(
                    training_duration_mean=duration_mean,
                    training_duration_min=duration_min,
                )
            )
        )

        sync_time, async_time = get_training_time(
            num_users=num_users,
            users_per_round=users_per_round,
            epochs=epochs,
            training_dist=training_dist,
        )
        assertAlmostEqual(sync_time, async_time, delta=1e-3)

    def test_per_user_half_normal(self) -> None:
        """
        Test half normal training time

        Sync and Async should have the following training time
            sync_training_time = async_training_time = num_users * duration_min
        if UPR = 1 and duraton_std is close to 0
        """
        torch.manual_seed(0)
        num_users = 1000
        epochs = 1
        users_per_round = 1
        duration_std = 1e-6
        duration_min = 1.0

        training_dist = PerUserHalfNormalDurationDistribution(
            **OmegaConf.structured(
                PerUserHalfNormalDurationDistributionConfig(
                    training_duration_sd=duration_std,
                    training_duration_min=duration_min,
                )
            )
        )

        sync_time, async_time = get_training_time(
            num_users=num_users,
            users_per_round=users_per_round,
            epochs=epochs,
            training_dist=training_dist,
        )
        assertAlmostEqual(sync_time, async_time, delta=1e-3)
        assertAlmostEqual(sync_time, num_users * duration_min, delta=1e-3)
        assertAlmostEqual(async_time, num_users * duration_min, delta=1e-3)
