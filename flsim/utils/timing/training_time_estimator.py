#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import List, Optional

import numpy as np
from flsim.utils.timing.training_duration_distribution import (
    IDurationDistribution,
)


class TrainingTimeEstimator:
    def __init__(
        self,
        total_users: int,
        users_per_round: int,
        epochs: int,
        training_dist: IDurationDistribution,
        num_examples: Optional[List[int]] = None,
    ):
        self.total_users = total_users
        self.users_per_round = users_per_round
        self.epochs = epochs
        self.rounds = int(self.epochs * self.total_users / self.users_per_round)

        self.num_examples: Optional[List[int]] = num_examples
        self.training_dist = training_dist

    def random_select(self):
        """
        Simulate user random selection to return the number of examples for that user
        """
        return 1 if self.num_examples is None else random.choice(self.num_examples)


class SyncTrainingTimeEstimator(TrainingTimeEstimator):
    def __init__(
        self,
        total_users: int,
        users_per_round: int,
        epochs: int,
        training_dist: IDurationDistribution,
        num_examples: Optional[List[int]] = None,
    ):
        super().__init__(
            total_users=total_users,
            users_per_round=users_per_round,
            epochs=epochs,
            num_examples=num_examples,
            training_dist=training_dist,
        )

    def training_time(self):
        """
        Returns training time for SyncFL
        """
        round_completion_time = [
            self.round_completion_time(
                users_per_round=self.users_per_round,
                num_examples=self.num_examples,
                training_dist=self.training_dist,
            )
            for x in range(self.rounds)
        ]
        return sum(round_completion_time)

    def round_completion_time(
        self,
        users_per_round: int,
        num_examples: List[int],
        training_dist: IDurationDistribution,
    ):
        """
        Return the max completion time: straggler effect
        """
        training_times = [
            training_dist.training_duration(self.random_select())
            for _ in range(users_per_round)
        ]
        return max(training_times)


class AsyncTrainingTimeEstimator(TrainingTimeEstimator):
    def __init__(
        self,
        total_users: int,
        users_per_round: int,
        epochs: int,
        training_dist: IDurationDistribution,
        num_examples: Optional[List[int]] = None,
        start_times: Optional[List[int]] = None,
    ):
        super().__init__(
            total_users=total_users,
            users_per_round=users_per_round,
            epochs=epochs,
            num_examples=num_examples,
            training_dist=training_dist,
        )
        self.start_times = start_times

    def training_time(self):
        """
        Returns the training time for AsyncFL
        Assuming client starts training at a linear rate

        """
        user_training_events = self.total_users * self.epochs
        training_durations = [
            self.training_dist.training_duration(self.random_select())
            for _ in range(user_training_events)
        ]
        training_start_times = self.training_start_times(
            training_durations, user_training_events
        )
        training_end_times = self.list_sum(training_start_times, training_durations)
        return max(training_end_times)

    def training_start_times(self, training_durations, user_training_events):
        if self.start_times is None:
            training_start_delta = np.mean(training_durations) / self.users_per_round
            return [
                user_index * training_start_delta
                for user_index in range(user_training_events)
            ]
        else:
            return self.start_times

    @classmethod
    def list_sum(cls, listA: List[float], listB: List[float]):
        ret_val = map(lambda x, y: x + y, listA, listB)
        return list(ret_val)


def get_training_time(
    num_users: int,
    users_per_round: int,
    epochs: int,
    training_dist: IDurationDistribution,
    num_examples: Optional[List[int]] = None,
):
    """
    Returns the estimated training time between SyncFL and AsyncFL
    """
    sync_estimator = SyncTrainingTimeEstimator(
        total_users=num_users,
        users_per_round=users_per_round,
        epochs=epochs,
        num_examples=num_examples,
        training_dist=training_dist,
    )

    async_estimator = AsyncTrainingTimeEstimator(
        total_users=num_users,
        users_per_round=users_per_round,
        epochs=epochs,
        num_examples=num_examples,
        training_dist=training_dist,
    )
    async_time = async_estimator.training_time()
    sync_time = sync_estimator.training_time()
    print(f"Sync {sync_time} Async {async_time}")
    return sync_time, async_time
