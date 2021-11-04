#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from flsim.data.data_provider import FLDataProviderFromList
from flsim.utils.async_trainer.async_user_selector import (
    RandomAsyncUserSelector,
    RoundRobinAsyncUserSelector,
)
from flsim.utils.sample_model import MockFLModel
from libfb.py import testutil


class AsyncUserSelectorUtilsTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.num_users = 20
        self.num_trials = 100

    def test_random_user_selector(self):
        # users are 0....n-1
        # number of examples per user: [1, 2, 3...., n-1, n]
        num_examples_per_user = list(range(1, self.num_users + 1))
        data = [
            [1] * num_example
            for num_example, _ in zip(num_examples_per_user, range(self.num_users))
        ]
        data_provider = FLDataProviderFromList(
            train_user_list=data,
            eval_user_list=data,
            test_user_list=data,
            model=MockFLModel(),
        )
        random_user_selector = RandomAsyncUserSelector(data_provider=data_provider)
        for _ in range(0, 100):
            random_user_info = random_user_selector.get_random_user()
            random_user, user_index = (
                random_user_info.user_data,
                random_user_info.user_index,
            )
            self.assertTrue(user_index >= 0 and user_index < self.num_users)
            self.assertEqual(random_user.num_examples(), user_index + 1)

    def test_round_robin_user_selector(self):
        # users are 0....n-1
        # number of examples per user: [10, 20, 30...., 10*n-1, 10*n]
        multiplier = 10
        num_examples_per_user = [
            multiplier * i for i in list(range(1, self.num_users + 1))
        ]
        round_robin_user_selector = RoundRobinAsyncUserSelector(num_examples_per_user)
        data = [
            [1] * num_example
            for num_example, _ in zip(num_examples_per_user, range(self.num_users))
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
        for num_trial in range(0, self.num_trials):
            random_user_info = round_robin_user_selector.get_random_user()
            random_user, user_index = (
                random_user_info.user_data,
                random_user_info.user_index,
            )
            self.assertEqual(user_index, num_trial % self.num_users)
            self.assertEqual(random_user.num_examples(), (user_index + 1) * multiplier)
