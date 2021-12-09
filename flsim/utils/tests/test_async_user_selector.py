#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from flsim.common.pytest_helper import assertEqual, assertTrue
from flsim.data.data_provider import FLDataProviderFromList
from flsim.utils.async_trainer.async_user_selector import (
    RandomAsyncUserSelector,
    RoundRobinAsyncUserSelector,
)
from flsim.utils.sample_model import MockFLModel


@pytest.fixture
def num_users():
    return 20


@pytest.fixture
def num_trials():
    return 100


class TestAsyncUserSelectorUtils:
    def test_random_user_selector(self, num_users, num_trials):
        # users are 0....n-1
        # number of examples per user: [1, 2, 3...., n-1, n]
        num_examples_per_user = list(range(1, num_users + 1))
        data = [
            [1] * num_example
            for num_example, _ in zip(num_examples_per_user, range(num_users))
        ]
        data_provider = FLDataProviderFromList(
            train_user_list=data,
            eval_user_list=data,
            test_user_list=data,
            model=MockFLModel(),
        )
        random_user_selector = RandomAsyncUserSelector(data_provider=data_provider)
        for _ in range(0, num_trials):
            random_user_info = random_user_selector.get_random_user()
            random_user, user_index = (
                random_user_info.user_data,
                random_user_info.user_index,
            )
            assertTrue(user_index >= 0 and user_index < num_users)
            assertEqual(random_user.num_examples(), user_index + 1)

    def test_round_robin_user_selector(self, num_users, num_trials):
        # users are 0....n-1
        # number of examples per user: [10, num_users, 30...., 10*n-1, 10*n]
        multiplier = 10
        num_examples_per_user = [multiplier * i for i in list(range(1, num_users + 1))]
        round_robin_user_selector = RoundRobinAsyncUserSelector(num_examples_per_user)
        data = [
            [1] * num_example
            for num_example, _ in zip(num_examples_per_user, range(num_users))
        ]
        data_provider = FLDataProviderFromList(
            train_user_list=data,
            eval_user_list=data,
            test_user_list=data,
            model=MockFLModel(),
        )
        round_robin_user_selector = RoundRobinAsyncUserSelector(
            data_provider=data_provider
        )
        for num_trial in range(0, num_trials):
            random_user_info = round_robin_user_selector.get_random_user()
            random_user, user_index = (
                random_user_info.user_data,
                random_user_info.user_index,
            )
            assertEqual(user_index, num_trial % num_users)
            assertEqual(random_user.num_examples(), (user_index + 1) * multiplier)
