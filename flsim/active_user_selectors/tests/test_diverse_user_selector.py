#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import Counter

import torch
from flsim.active_user_selectors.diverse_user_selector import (
    DiverseUserSelectorUtils,
    DiversityMaximizingUserSelector,
    DiversityMaximizingUserSelectorConfig,
    DiversityStatisticsReportingUserSelector,
    DiversityStatisticsReportingUserSelectorConfig,
)
from flsim.common.diversity_metrics import DiversityMetricType
from flsim.common.pytest_helper import assertEqual, assertTrue, assertIsInstance
from flsim.utils.sample_model import DummyAlphabetFLModel, LinearFLModel
from flsim.utils.tests.helpers.test_data_utils import (
    DummyAlphabetDataset,
    NonOverlappingDataset,
    RandomDataset,
)
from hydra.utils import instantiate


class TestDiverseUserSelector:

    # Since diversity statistics reporting selector chooses uniformly randomly, replicate the tests
    # for that class. select_diverse_cohort is tested separately.
    def test_diversity_statistics_reporting_user_selector(self):
        null_selector = instantiate(
            DiversityStatisticsReportingUserSelectorConfig(num_candidate_cohorts=2)
        )
        assertIsInstance(null_selector, DiversityStatisticsReportingUserSelector)
        # This user selector class requires a global model and data provider.
        users = tuple(range(5))

        shard_size, local_batch_size = 4, 4
        dummy_dataset = DummyAlphabetDataset()
        dummy_model = DummyAlphabetFLModel()
        data_provider, _ = DummyAlphabetDataset.create_data_provider_and_loader(
            dummy_dataset, shard_size, local_batch_size, dummy_model
        )

        uniform_selector_1 = instantiate(
            DiversityStatisticsReportingUserSelectorConfig(
                user_selector_seed=12345, num_candidate_cohorts=2
            )
        )
        uniform_selector_2 = instantiate(
            DiversityStatisticsReportingUserSelectorConfig(
                user_selector_seed=12345, num_candidate_cohorts=2
            )
        )
        # Repeat, since it can't be done with many users as test_uniformly_random_user_selector
        num_trials = 25
        for _ in range(num_trials):
            selection_1 = uniform_selector_1.get_user_indices(
                num_total_users=len(users),
                users_per_round=1,
                data_provider=data_provider,
                global_model=dummy_model,
                epoch=0,
            )
            selection_2 = uniform_selector_2.get_user_indices(
                num_total_users=len(users),
                users_per_round=1,
                data_provider=data_provider,
                global_model=dummy_model,
                epoch=0,
            )
            assertEqual(selection_1, selection_2)

    def test_diversity_maximizing_user_selector(self):
        # Check that the selector actually chooses the cohort with maximum GD.
        # For cohort size 3 our of 4 users, there are 4 possible cohorts.
        # With 100 candidate cohorts we will try all 4 with very high probability.
        num_total_users, users_per_round = 4, 3
        num_data_per_user, dim_data = 6, 40
        shard_size, local_batch_size = 6, 6
        num_candidates = 100
        available_users = range(num_total_users)

        null_selector = instantiate(
            DiversityMaximizingUserSelectorConfig(num_candidate_cohorts=num_candidates)
        )
        assertIsInstance(null_selector, DiversityMaximizingUserSelector)

        linear_model = LinearFLModel(D_in=dim_data, D_out=1)
        random_dataset = RandomDataset(
            num_users=num_total_users,
            num_data_per_user=num_data_per_user,
            dim_data=dim_data,
        )
        data_provider, _ = RandomDataset.create_data_provider_and_loader(
            random_dataset, shard_size, local_batch_size, linear_model
        )

        diverse_cohort = null_selector.get_user_indices(
            num_total_users=num_total_users,
            users_per_round=users_per_round,
            data_provider=data_provider,
            global_model=linear_model,
            epoch=0,
        )

        max_diversity = DiverseUserSelectorUtils.calculate_diversity_metrics(
            data_provider=data_provider,
            global_model=linear_model,
            user_indices=diverse_cohort,
        )

        available_users_set = set(available_users)
        # Enumerate through each possible cohort, show it doesn't have greater Diversity
        for i in available_users:
            cohort = list(available_users_set.difference({i}))

            cohort_diversity = DiverseUserSelectorUtils.calculate_diversity_metrics(
                data_provider=data_provider,
                global_model=linear_model,
                user_indices=cohort,
            )

            assertTrue(cohort_diversity <= max_diversity)

        # Test epochs_before_active as is done in test_high_loss_user_selector()
        selector = instantiate(
            DiversityMaximizingUserSelectorConfig(epochs_before_active=1)
        )

        shard_size, local_batch_size = 10, 4
        dummy_dataset = DummyAlphabetDataset(num_rows=11)
        dummy_model = DummyAlphabetFLModel()
        data_provider, _ = DummyAlphabetDataset.create_data_provider_and_loader(
            dummy_dataset, shard_size, local_batch_size, dummy_model
        )

        selections = [
            selector.get_user_indices(
                num_total_users=2,
                users_per_round=1,
                data_provider=data_provider,
                global_model=dummy_model,
                epoch=0,
            )[0]
            for i in range(1000)
        ]
        counts = Counter(selections)
        assertTrue(counts[0] > 400)
        assertTrue(counts[0] < 600)

    def test_user_selector_diversity_metric_type(self):
        # Test that the created user selector uses the specified diversity metric

        metric_type_strings = [
            "gradient_diversity",
            "orthogonality",
            "delta_norm_sq",
            "sum_client_delta_norm_sq",
            "sum_client_delta_mutual_angles",
        ]
        metric_types = [
            DiversityMetricType.gradient_diversity,
            DiversityMetricType.orthogonality,
            DiversityMetricType.delta_norm_sq,
            DiversityMetricType.sum_client_delta_norm_sq,
            DiversityMetricType.sum_client_delta_mutual_angles,
        ]

        for metric_str, metric_type in zip(metric_type_strings, metric_types):
            user_selector = instantiate(
                DiversityMaximizingUserSelectorConfig(diversity_metric_type=metric_str)
            )
            assertEqual(user_selector.cfg.diversity_metric_type, metric_type)


class TestDiverseUserSelectorUtils:

    tolerence = 1e-5

    def test_calculate_diversity_metrics(self):
        # We construct nonoverlapping data for each user so that the gradient diversity will be 1
        num_total_users, users_per_round = 10, 3
        num_nonzeros_per_user, num_data_per_user = 4, 6
        shard_size, local_batch_size = 6, 6
        linear_model = LinearFLModel(
            D_in=num_total_users * num_nonzeros_per_user, D_out=1
        )
        nonoverlap_dataset = NonOverlappingDataset(
            num_users=num_total_users,
            num_nonzeros_per_user=num_nonzeros_per_user,
            num_data_per_user=num_data_per_user,
        )
        data_provider, _ = NonOverlappingDataset.create_data_provider_and_loader(
            nonoverlap_dataset, shard_size, local_batch_size, linear_model
        )

        # Check GD for all users then for random cohorts; ensure GD=1 for each
        available_users = range(num_total_users)
        rng = torch.Generator()
        diversity_metrics = DiverseUserSelectorUtils.calculate_diversity_metrics(
            data_provider, linear_model, available_users
        )
        assertTrue(
            math.isclose(diversity_metrics.gradient_diversity, 1.0, rel_tol=1e-04)
        )

        while len(available_users) > 0:
            selected_indices = torch.multinomial(
                torch.ones(len(available_users), dtype=torch.float),
                min(users_per_round, len(available_users)),
                replacement=False,
                generator=rng,
            ).tolist()
            user_indices = [available_users[idx] for idx in selected_indices]
            available_users = [
                idx for idx in available_users if idx not in user_indices
            ]
            diversity_metrics = DiverseUserSelectorUtils.calculate_diversity_metrics(
                data_provider, linear_model, user_indices
            )
            assertTrue(
                math.isclose(diversity_metrics.gradient_diversity, 1.0, rel_tol=1e-04)
            )

        # By the definition of GD, it will always be at least 1/num_users.
        num_total_users, users_per_round = 10, 3
        num_data_per_user, dim_data = 6, 40
        shard_size, local_batch_size = 6, 6
        linear_model = LinearFLModel(
            D_in=num_total_users * num_nonzeros_per_user, D_out=1
        )
        random_dataset = RandomDataset(
            num_users=num_total_users,
            num_data_per_user=num_data_per_user,
            dim_data=dim_data,
        )
        data_provider, _ = RandomDataset.create_data_provider_and_loader(
            random_dataset, shard_size, local_batch_size, linear_model
        )

        # Check GD for all users then for random cohorts; ensure GD>=1/num_users for each
        available_users = range(num_total_users)
        rng = torch.Generator()
        diversity_metrics = DiverseUserSelectorUtils.calculate_diversity_metrics(
            data_provider, linear_model, available_users
        )
        assertTrue(diversity_metrics.gradient_diversity >= 1.0 / num_total_users)

        while len(available_users) > 0:
            selected_indices = torch.multinomial(
                torch.ones(len(available_users), dtype=torch.float),
                min(users_per_round, len(available_users)),
                replacement=False,
                generator=rng,
            ).tolist()
            user_indices = [available_users[idx] for idx in selected_indices]

            diversity_metrics = DiverseUserSelectorUtils.calculate_diversity_metrics(
                data_provider, linear_model, user_indices
            )
            assertTrue(
                diversity_metrics.gradient_diversity
                >= 1.0 / min(num_total_users, len(available_users))
            )

            available_users = [
                idx for idx in available_users if idx not in user_indices
            ]

    def test_select_diverse_cohort(self):

        # Use dataset in which the gradient diversity is 1 for any cohort and
        # Check max, avg, and min GD = 1
        num_total_users, users_per_round = 10, 3
        num_nonzeros_per_user, num_data_per_user = 4, 6
        shard_size, local_batch_size = 6, 6
        num_candidate_cohorts = 5
        loss_reduction_type = "mean"
        client_gradient_scaling = "sum"
        diversity_metric_type = DiversityMetricType.gradient_diversity
        maximize_metric = True
        rel_tol = 1e-04
        available_users = range(num_total_users)
        rng = torch.Generator()

        linear_model = LinearFLModel(
            D_in=num_total_users * num_nonzeros_per_user, D_out=1
        )
        nonoverlap_dataset = NonOverlappingDataset(
            num_users=num_total_users,
            num_nonzeros_per_user=num_nonzeros_per_user,
            num_data_per_user=num_data_per_user,
        )
        data_provider, _ = NonOverlappingDataset.create_data_provider_and_loader(
            nonoverlap_dataset, shard_size, local_batch_size, linear_model
        )

        (_, diversity_statistics,) = DiverseUserSelectorUtils.select_diverse_cohort(
            data_provider=data_provider,
            global_model=linear_model,
            users_per_round=users_per_round,
            available_users=available_users,
            rng=rng,
            num_search_samples=num_candidate_cohorts,
            maximize_metric=maximize_metric,
            loss_reduction_type=loss_reduction_type,
            client_gradient_scaling=client_gradient_scaling,
            diversity_metric_type=diversity_metric_type,
        )

        assertTrue(
            math.isclose(diversity_statistics.maximum_metric, 1.0, rel_tol=rel_tol)
        )
        assertTrue(
            math.isclose(diversity_statistics.average_metric, 1.0, rel_tol=rel_tol)
        )
        assertTrue(
            math.isclose(diversity_statistics.minimum_metric, 1.0, rel_tol=rel_tol)
        )

        # On random data, check diversity statistics for all users then for random cohorts. Ensure that
        # there is separation between the max, average, and min when not all users are in the cohort.
        num_total_users, users_per_round = 10, 3
        num_data_per_user, dim_data = 6, 40
        shard_size, local_batch_size = 6, 6
        num_candidate_cohorts = 5
        loss_reduction_type = "mean"
        client_gradient_scaling = "sum"
        diversity_metric_type = DiversityMetricType.gradient_diversity
        maximize_metric = True
        available_users = range(num_total_users)

        linear_model = LinearFLModel(D_in=dim_data, D_out=1)
        random_dataset = RandomDataset(
            num_users=num_total_users,
            num_data_per_user=num_data_per_user,
            dim_data=dim_data,
        )
        data_provider, _ = RandomDataset.create_data_provider_and_loader(
            random_dataset, shard_size, local_batch_size, linear_model
        )
        rng = torch.Generator()

        (
            diverse_cohort,
            diversity_statistics,
        ) = DiverseUserSelectorUtils.select_diverse_cohort(
            data_provider=data_provider,
            global_model=linear_model,
            users_per_round=num_total_users,
            available_users=available_users,
            rng=rng,
            num_search_samples=num_candidate_cohorts,
            maximize_metric=maximize_metric,
            loss_reduction_type=loss_reduction_type,
            client_gradient_scaling=client_gradient_scaling,
            diversity_metric_type=diversity_metric_type,
        )
        assertEqual(set(diverse_cohort), set(available_users))
        assertTrue(
            math.isclose(
                diversity_statistics.maximum_metric,
                diversity_statistics.average_metric,
                rel_tol=rel_tol,
            )
        )
        assertTrue(
            math.isclose(
                diversity_statistics.average_metric,
                diversity_statistics.minimum_metric,
                rel_tol=rel_tol,
            )
        )

        while len(available_users) > 0:

            (
                diverse_cohort,
                diversity_statistics,
            ) = DiverseUserSelectorUtils.select_diverse_cohort(
                data_provider=data_provider,
                global_model=linear_model,
                users_per_round=users_per_round,
                available_users=available_users,
                rng=rng,
                num_search_samples=num_candidate_cohorts,
                maximize_metric=maximize_metric,
                loss_reduction_type=loss_reduction_type,
                client_gradient_scaling=client_gradient_scaling,
                diversity_metric_type=diversity_metric_type,
            )

            # Check that the chosen cohort is valid
            assertTrue(set(diverse_cohort) <= set(available_users))

            assertTrue(
                diversity_statistics.minimum_metric
                >= 1.0 / min(num_total_users, len(available_users))
            )

            # The gradient diversity should be equal to the maximum in the statistics
            diverse_cohort_diversity = (
                DiverseUserSelectorUtils.calculate_diversity_metrics(
                    data_provider=data_provider,
                    global_model=linear_model,
                    user_indices=diverse_cohort,
                    loss_reduction_type=loss_reduction_type,
                    client_gradient_scaling=client_gradient_scaling,
                    diversity_metric_type=diversity_metric_type,
                )
            )
            assertTrue(
                math.isclose(
                    diverse_cohort_diversity.gradient_diversity,
                    diversity_statistics.maximum_metric,
                    rel_tol=rel_tol,
                )
            )

            if len(available_users) > users_per_round:
                assertEqual(len(diverse_cohort), users_per_round)
                assertTrue(
                    diversity_statistics.maximum_metric
                    > diversity_statistics.average_metric
                )
                assertTrue(
                    diversity_statistics.average_metric
                    > diversity_statistics.minimum_metric
                )
            else:
                assertEqual(len(diverse_cohort), len(available_users))
                assertTrue(
                    math.isclose(
                        diversity_statistics.maximum_metric,
                        diversity_statistics.average_metric,
                        rel_tol=rel_tol,
                    )
                )
                assertTrue(
                    math.isclose(
                        diversity_statistics.average_metric,
                        diversity_statistics.minimum_metric,
                        rel_tol=rel_tol,
                    )
                )
            available_users = [
                idx for idx in available_users if idx not in diverse_cohort
            ]

        # Try this again with minimizing the diversity metric
        num_total_users, users_per_round = 10, 3
        num_data_per_user, dim_data = 6, 40
        shard_size, local_batch_size = 6, 6
        num_candidate_cohorts = 5
        loss_reduction_type = "mean"
        client_gradient_scaling = "sum"
        diversity_metric_type = DiversityMetricType.gradient_diversity
        maximize_metric = False
        available_users = range(num_total_users)

        linear_model = LinearFLModel(D_in=dim_data, D_out=1)
        random_dataset = RandomDataset(
            num_users=num_total_users,
            num_data_per_user=num_data_per_user,
            dim_data=dim_data,
        )
        data_provider, _ = RandomDataset.create_data_provider_and_loader(
            random_dataset, shard_size, local_batch_size, linear_model
        )
        rng = torch.Generator()

        (
            diverse_cohort,
            diversity_statistics,
        ) = DiverseUserSelectorUtils.select_diverse_cohort(
            data_provider=data_provider,
            global_model=linear_model,
            users_per_round=num_total_users,
            available_users=available_users,
            rng=rng,
            num_search_samples=num_candidate_cohorts,
            maximize_metric=maximize_metric,
            loss_reduction_type=loss_reduction_type,
            client_gradient_scaling=client_gradient_scaling,
            diversity_metric_type=diversity_metric_type,
        )
        assertEqual(set(diverse_cohort), set(available_users))
        assertTrue(
            math.isclose(
                diversity_statistics.maximum_metric,
                diversity_statistics.average_metric,
                rel_tol=rel_tol,
            )
        )
        assertTrue(
            math.isclose(
                diversity_statistics.average_metric,
                diversity_statistics.minimum_metric,
                rel_tol=rel_tol,
            )
        )

        while len(available_users) > 0:

            (
                diverse_cohort,
                diversity_statistics,
            ) = DiverseUserSelectorUtils.select_diverse_cohort(
                data_provider=data_provider,
                global_model=linear_model,
                users_per_round=users_per_round,
                available_users=available_users,
                rng=rng,
                num_search_samples=num_candidate_cohorts,
                maximize_metric=maximize_metric,
                loss_reduction_type=loss_reduction_type,
                client_gradient_scaling=client_gradient_scaling,
                diversity_metric_type=diversity_metric_type,
            )

            # Check that the chosen cohort is valid
            assertTrue(set(diverse_cohort) <= set(available_users))

            assertTrue(
                diversity_statistics.minimum_metric
                >= 1.0 / min(num_total_users, len(available_users))
            )

            # The gradient diversity should be equal to the minimium in the statistics
            diverse_cohort_diversity = (
                DiverseUserSelectorUtils.calculate_diversity_metrics(
                    data_provider=data_provider,
                    global_model=linear_model,
                    user_indices=diverse_cohort,
                    loss_reduction_type=loss_reduction_type,
                    client_gradient_scaling=client_gradient_scaling,
                    diversity_metric_type=diversity_metric_type,
                )
            )
            assertTrue(
                math.isclose(
                    diverse_cohort_diversity.gradient_diversity,
                    diversity_statistics.minimum_metric,
                    rel_tol=rel_tol,
                )
            )

            if len(available_users) > users_per_round:
                assertEqual(len(diverse_cohort), users_per_round)
                assertTrue(
                    diversity_statistics.maximum_metric
                    > diversity_statistics.average_metric
                )
                assertTrue(
                    diversity_statistics.average_metric
                    > diversity_statistics.minimum_metric
                )
            else:
                assertEqual(len(diverse_cohort), len(available_users))
                assertTrue(
                    math.isclose(
                        diversity_statistics.maximum_metric,
                        diversity_statistics.average_metric,
                        rel_tol=rel_tol,
                    )
                )
                assertTrue(
                    math.isclose(
                        diversity_statistics.average_metric,
                        diversity_statistics.minimum_metric,
                        rel_tol=rel_tol,
                    )
                )
            available_users = [
                idx for idx in available_users if idx not in diverse_cohort
            ]
