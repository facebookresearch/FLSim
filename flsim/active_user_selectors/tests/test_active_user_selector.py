#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
from collections import Counter

import torch
from flsim.active_user_selectors.simple_user_selector import (
    ActiveUserSelectorUtils,
    ImportanceSamplingActiveUserSelector,
    ImportanceSamplingActiveUserSelectorConfig,
    RandomMultiStepActiveUserSelector,
    RandomMultiStepActiveUserSelectorConfig,
    RandomRoundRobinActiveUserSelector,
    RandomRoundRobinActiveUserSelectorConfig,
    SequentialActiveUserSelector,
    SequentialActiveUserSelectorConfig,
    UniformlyRandomActiveUserSelector,
    UniformlyRandomActiveUserSelectorConfig,
)
from flsim.common.pytest_helper import (
    assertEqual,
    assertIsInstance,
    assertRaises,
    assertTrue,
)
from flsim.utils.sample_model import DummyAlphabetFLModel
from flsim.utils.tests.helpers.test_data_utils import DummyAlphabetDataset
from hydra.utils import instantiate


class TestActiveUserSelector:
    def test_uniformly_random_user_selection(self) -> None:
        # Test the uniformly random user selection in the ActiveUserSelector base class
        null_selector = instantiate(SequentialActiveUserSelectorConfig())
        assertIsInstance(null_selector, SequentialActiveUserSelector)
        users = tuple(range(5))
        torch.manual_seed(123456789)
        selections = [
            null_selector.get_users_unif_rand(
                num_total_users=len(users), users_per_round=1
            )[0]
            for _ in range(10000)
        ]
        counts = Counter(selections)
        assertTrue(min(counts.values()) > 1500 and max(counts.values()) < 2500)
        # Make sure that the users aren't being selected sequentially
        assertTrue(min(counts.values()) < max(counts.values()) - 2)
        with assertRaises(AssertionError):
            null_selector.get_user_indices()

        uniform_selector_1 = instantiate(
            SequentialActiveUserSelectorConfig(user_selector_seed=12345)
        )
        uniform_selector_2 = instantiate(
            SequentialActiveUserSelectorConfig(user_selector_seed=12345)
        )
        selection_1 = uniform_selector_1.get_users_unif_rand(
            num_total_users=10000, users_per_round=1
        )
        selection_2 = uniform_selector_2.get_users_unif_rand(
            num_total_users=10000, users_per_round=1
        )
        assertEqual(selection_1, selection_2)

    def test_uniformly_random_user_selector(self) -> None:
        null_selector = instantiate(UniformlyRandomActiveUserSelectorConfig())
        assertIsInstance(null_selector, UniformlyRandomActiveUserSelector)
        users = tuple(range(5))
        torch.manual_seed(123456789)
        selections = [
            null_selector.get_user_indices(
                num_total_users=len(users), users_per_round=1
            )[0]
            for _ in range(10000)
        ]
        counts = Counter(selections)
        assertTrue(min(counts.values()) > 1500 and max(counts.values()) < 2500)
        with assertRaises(AssertionError):
            null_selector.get_user_indices()

        uniform_selector_1 = instantiate(
            UniformlyRandomActiveUserSelectorConfig(user_selector_seed=12345)
        )
        uniform_selector_2 = instantiate(
            UniformlyRandomActiveUserSelectorConfig(user_selector_seed=12345)
        )
        selection_1 = uniform_selector_1.get_user_indices(
            num_total_users=10000, users_per_round=1
        )
        selection_2 = uniform_selector_2.get_user_indices(
            num_total_users=10000, users_per_round=1
        )
        assertEqual(selection_1, selection_2)

    def test_sequential_user_selector(self) -> None:
        # 1) test if num_users is not divisible by users_per_round
        num_total_users, users_per_round, round_num = 10, 3, 8
        selector = instantiate(SequentialActiveUserSelectorConfig())
        assertIsInstance(selector, SequentialActiveUserSelector)
        for round_index in range(round_num):
            user_indices = selector.get_user_indices(
                num_total_users=num_total_users, users_per_round=users_per_round
            )
            if round_index % 4 == 0:
                assertEqual(user_indices, [0, 1, 2])
            if round_index % 4 == 1:
                assertEqual(user_indices, [3, 4, 5])
            if round_index % 4 == 2:
                assertEqual(user_indices, [6, 7, 8])
            if round_index % 4 == 3:
                assertEqual(user_indices, [9])

        # 2) test if num_users is divisible by users_per_round
        num_total_users, round_num = 9, 6
        selector = instantiate(SequentialActiveUserSelectorConfig())
        for round_index in range(round_num):
            user_indices = selector.get_user_indices(
                num_total_users=num_total_users, users_per_round=users_per_round
            )
            if round_index % 3 == 0:
                assertEqual(user_indices, [0, 1, 2])
            if round_index % 3 == 1:
                assertEqual(user_indices, [3, 4, 5])
            if round_index % 3 == 2:
                assertEqual(user_indices, [6, 7, 8])

    def test_random_multi_step_user_selector(self) -> None:
        selector = instantiate(
            RandomMultiStepActiveUserSelectorConfig(milestones=[2, 4]), gamma=10
        )
        assertIsInstance(selector, RandomMultiStepActiveUserSelector)

        num_total_users, users_per_round, round_num = 1000, 5, 10
        for round_index in range(round_num):
            user_indices = selector.get_user_indices(
                num_total_users=num_total_users,
                users_per_round=users_per_round,
                global_round_num=round_index,
            )

            if round_index < 2:
                assertEqual(len(user_indices), users_per_round)
            if round_index >= 2 and round_index < 4:
                assertEqual(len(user_indices), users_per_round * 10)
            if round_index > 4:
                assertEqual(len(user_indices), users_per_round * 100)

    def test_random_round_robin_user_selector(self) -> None:
        # 1) test if num_users is not divisible by users_per_round
        num_total_users, users_per_round = 87, 15
        available_users = range(num_total_users)
        selector = instantiate(RandomRoundRobinActiveUserSelectorConfig())
        assertIsInstance(selector, RandomRoundRobinActiveUserSelector)
        while len(available_users) > 0:
            user_indices = selector.get_user_indices(
                num_total_users=num_total_users, users_per_round=users_per_round
            )
            user_indices_set = set(user_indices)
            available_user_set = set(available_users)
            assertEqual(len(user_indices), min(users_per_round, len(available_users)))
            assertEqual(len(user_indices_set), len(user_indices))
            assertEqual(len(available_user_set), len(available_users))
            assertTrue(user_indices_set.issubset(available_user_set))
            available_users = list(available_user_set - user_indices_set)

        # Try one more time to make sure that the user selector class reset
        user_indices = selector.get_user_indices(
            num_total_users=num_total_users, users_per_round=users_per_round
        )
        user_indices_set = set(user_indices)
        available_user_set = set(range(num_total_users))
        assertEqual(len(user_indices), users_per_round)
        assertEqual(len(user_indices_set), users_per_round)
        assertTrue(user_indices_set.issubset(available_user_set))

        # 2) test if num_users is divisible by users_per_round
        num_total_users, users_per_round = 60, 15
        selector = instantiate(RandomRoundRobinActiveUserSelectorConfig())
        while len(available_users) > 0:
            user_indices = selector.get_user_indices(
                num_total_users=num_total_users, users_per_round=users_per_round
            )
            user_indices_set = set(user_indices)
            available_user_set = set(available_users)
            assertEqual(len(user_indices), users_per_round)
            assertEqual(len(user_indices_set), len(user_indices))
            assertEqual(len(available_user_set), len(available_users))
            assertTrue(user_indices_set.issubset(available_user_set))
            available_users = list(available_user_set - user_indices_set)

        # Try one more time to make sure that the user selector class reset
        user_indices = selector.get_user_indices(
            num_total_users=num_total_users, users_per_round=users_per_round
        )
        user_indices_set = set(user_indices)
        available_user_set = set(range(num_total_users))
        assertEqual(len(user_indices), users_per_round)
        assertEqual(len(user_indices_set), users_per_round)
        assertTrue(user_indices_set.issubset(available_user_set))

    def test_importance_sampling_user_selector(self):
        num_total_users, users_per_round = 60, 5
        selector = instantiate(
            ImportanceSamplingActiveUserSelectorConfig(user_selector_seed=1234)
        )
        assertIsInstance(selector, ImportanceSamplingActiveUserSelector)

        rng = torch.Generator().manual_seed(17)
        num_samples_per_user = torch.randint(100, (num_total_users,), generator=rng)

        # Check users selected on subsequent runs
        selected_indices1 = selector.get_user_indices(
            num_total_users=num_total_users,
            users_per_round=users_per_round,
            num_samples_per_user=num_samples_per_user,
        )

        selected_indices2 = selector.get_user_indices(
            num_total_users=num_total_users,
            users_per_round=users_per_round,
            num_samples_per_user=num_samples_per_user,
        )

        assertEqual(
            selected_indices1,
            [0, 29, 41, 47, 56],
            "Selected indices does not match expected user indices",
        )
        assertEqual(
            selected_indices2,
            [12, 24, 27, 32, 35, 48, 49],
            "Selected indices does not match expected user indices",
        )

        # Check error if all clients have zero data
        try:
            selector.get_user_indices(
                num_total_users=num_total_users,
                users_per_round=users_per_round,
                num_samples_per_user=torch.zeros(num_total_users),
            )
        except AssertionError:
            pass
        else:
            assertTrue(
                False,
                "ImportanceSampling with zero data in all clients must throw an AssertionError",
            )

        # Check mismatch between num_total_uses and len(num_samples_per_user)
        try:
            selector.get_user_indices(
                num_total_users=num_total_users,
                users_per_round=users_per_round,
                num_samples_per_user=torch.zeros(num_total_users - 2),
            )
        except AssertionError:
            pass
        else:
            assertTrue(
                False,
                "ImportanceSampling with non-matching num_total_users and num_samples_per_user length",
            )

        try:
            selector.get_user_indices(
                num_total_users=num_total_users,
                users_per_round=num_total_users + 2,
                num_samples_per_user=torch.zeros(num_total_users),
            )
        except AssertionError:
            pass
        else:
            assertTrue(
                False, "ImportanceSampling with users_per_round > num_total_users"
            )


class TestActiveUserSelectorUtils:

    tolerance = 1e-5

    def test_convert_to_probability(self) -> None:
        valuations = torch.tensor([1, 1, 1, 2, 2], dtype=torch.float)
        weights = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float)

        total_prob = sum(
            ActiveUserSelectorUtils.convert_to_probability(valuations, 0, 1)
        )

        assertEqual(total_prob, 1)

        prob_1 = ActiveUserSelectorUtils.convert_to_probability(
            valuations, 0, 1, weights
        )
        prob_2 = ActiveUserSelectorUtils.convert_to_probability(valuations, 0, 1)
        assertTrue(torch.allclose(prob_1, prob_2, rtol=self.tolerance))

        prob_1 = ActiveUserSelectorUtils.convert_to_probability(valuations, 0, 1)
        prob_2 = torch.exp(valuations) / sum(torch.exp(valuations))
        assertTrue(torch.allclose(prob_1, prob_2, rtol=self.tolerance))

        weights = torch.tensor([1, 2, 1, 2, 1], dtype=torch.float)
        unnormalized_probs = torch.tensor(
            [math.exp(1), 2 * math.exp(1), math.exp(1), 2 * math.exp(2), math.exp(2)],
            dtype=torch.float,
        )
        prob_1 = unnormalized_probs / sum(unnormalized_probs)
        prob_2 = ActiveUserSelectorUtils.convert_to_probability(
            valuations, 0, 1, weights
        )
        assertTrue(torch.allclose(prob_1, prob_2, rtol=self.tolerance))

        prob_1 = ActiveUserSelectorUtils.convert_to_probability(valuations, 0, 0)
        prob_2 = torch.tensor([0.2] * 5, dtype=torch.float)
        assertTrue(torch.allclose(prob_1, prob_2, rtol=self.tolerance))

        prob_1 = ActiveUserSelectorUtils.convert_to_probability(valuations, 0, 25)
        prob_2 = torch.tensor([0, 0, 0, 0.5, 0.5], dtype=torch.float)
        assertTrue(torch.allclose(prob_1, prob_2, rtol=self.tolerance))

        prob = ActiveUserSelectorUtils.convert_to_probability(valuations, 0.5, 1)
        assertEqual(len(torch.nonzero(prob)), 3)

        with assertRaises(AssertionError):
            ActiveUserSelectorUtils.convert_to_probability(valuations, 1, 1)

    def test_normalize_by_sample_count(self) -> None:
        user_utility = torch.tensor([0, 1, 2, 3], dtype=torch.float)
        counts = torch.tensor([1, 10, 100, 1], dtype=torch.float)

        no_normalization = ActiveUserSelectorUtils.normalize_by_sample_count(
            user_utility, counts, 0
        )
        assertTrue(torch.allclose(no_normalization, user_utility, rtol=self.tolerance))

        avged = user_utility / counts
        avg_normalization = ActiveUserSelectorUtils.normalize_by_sample_count(
            user_utility, counts, 1
        )
        assertTrue(torch.allclose(avg_normalization, avged, rtol=self.tolerance))

    def test_samples_per_user(self) -> None:
        shard_size = 4
        local_batch_size = 4
        dummy_dataset = DummyAlphabetDataset()

        dummy_model = DummyAlphabetFLModel()
        data_provider, _ = DummyAlphabetDataset.create_data_provider_and_loader(
            dummy_dataset, shard_size, local_batch_size, dummy_model
        )
        samples_per_user = ActiveUserSelectorUtils.samples_per_user(data_provider)
        assertTrue(
            torch.allclose(
                samples_per_user, torch.tensor([4, 4, 4, 4, 4, 4, 2], dtype=torch.float)
            )
        )

    def test_select_users(self) -> None:
        """select_users has two mechanisms for selecting users: p proportion are
        selected according to the probabilities in probs, and (1-p) proportion are
        selected uniformly at random, where p = fraction_uniformly_random. This test
        ensures that the correct number of users are returned for different values
        of p.
        """
        probs = torch.tensor([0.1] * 10, dtype=torch.float)
        users_per_round = 3
        rng = torch.Generator()

        assertEqual(
            len(ActiveUserSelectorUtils.select_users(users_per_round, probs, 0, rng)), 3
        )
        assertEqual(
            len(ActiveUserSelectorUtils.select_users(users_per_round, probs, 1, rng)), 3
        )
        assertEqual(
            len(ActiveUserSelectorUtils.select_users(users_per_round, probs, 0.5, rng)),
            3,
        )
        assertEqual(
            sorted(
                ActiveUserSelectorUtils.select_users(
                    2, torch.tensor([0.5, 0.5, 0, 0, 0], dtype=torch.float), 0, rng
                )
            ),
            [0, 1],
        )

    def test_sample_available_users(self) -> None:
        num_total_users, users_per_round = 95, 10
        available_users = range(num_total_users)
        rng = torch.Generator()

        while len(available_users) > 0:
            prev_available_users = available_users
            user_indices = ActiveUserSelectorUtils.sample_available_users(
                users_per_round,
                # pyre-fixme[6]: Expected `List[int]` for 2nd param but got
                #  `Union[typing.List[int], range]`.
                available_users,
                rng,
            )
            user_indices_set = set(user_indices)
            available_user_set = set(available_users)

            assertEqual(len(user_indices), min(users_per_round, len(available_users)))
            assertEqual(available_users, prev_available_users)
            assertTrue(user_indices_set.issubset(available_user_set))
            assertEqual(len(user_indices_set), len(user_indices))
            assertEqual(len(available_user_set), len(available_users))

            available_users = list(available_user_set - user_indices_set)
