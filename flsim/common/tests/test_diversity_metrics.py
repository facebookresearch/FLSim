# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import random

from flsim.common.diversity_metrics import (
    DiversityMetrics,
    DiversityMetricType,
    DiversityStatistics,
)
from flsim.common.pytest_helper import (
    assertEqual,
    assertTrue,
)


class TestDiversityStatistics:
    def test_initialize_diversity_statistics(self):
        n_test_cases = 1000
        num_cohorts_per_test = 10
        std_dev = 500.0
        num_tol = 1e-06

        for _ in range(n_test_cases):
            cohort_diversity_metrics = []
            cohort_gradient_diversities = []
            cohort_orthogonalities = []

            for _ in range(num_cohorts_per_test):
                norm_of_sum = abs(random.normalvariate(0, std_dev))
                sum_of_norms = abs(random.normalvariate(0, std_dev))
                diversity_metrics = DiversityMetrics(norm_of_sum, sum_of_norms)

                cohort_diversity_metrics.append(diversity_metrics)
                cohort_gradient_diversities.append(diversity_metrics.gradient_diversity)
                cohort_orthogonalities.append(diversity_metrics.orthogonality)

            diversity_statistics = DiversityStatistics(cohort_diversity_metrics)
            diversity_statistics_eq = DiversityStatistics(cohort_diversity_metrics)
            assertEqual(diversity_statistics, diversity_statistics_eq)

            assertTrue(
                math.isclose(
                    diversity_statistics.maximum_metric,
                    max(cohort_gradient_diversities),
                    rel_tol=num_tol,
                )
            )
            assertTrue(
                math.isclose(
                    diversity_statistics.minimum_metric,
                    min(cohort_gradient_diversities),
                    rel_tol=num_tol,
                )
            )
            assertTrue(
                math.isclose(
                    diversity_statistics.average_metric,
                    sum(cohort_gradient_diversities) / len(cohort_gradient_diversities),
                    rel_tol=num_tol,
                )
            )

            for metric in cohort_diversity_metrics:
                metric.diversity_metric_type = DiversityMetricType.orthogonality

            diversity_statistics = DiversityStatistics(cohort_diversity_metrics)
            assertTrue(
                math.isclose(
                    diversity_statistics.maximum_metric,
                    max(cohort_orthogonalities),
                    rel_tol=num_tol,
                )
            )
            assertTrue(
                math.isclose(
                    diversity_statistics.minimum_metric,
                    min(cohort_orthogonalities),
                    rel_tol=num_tol,
                )
            )
            assertTrue(
                math.isclose(
                    diversity_statistics.average_metric,
                    sum(cohort_orthogonalities) / len(cohort_orthogonalities),
                    rel_tol=num_tol,
                )
            )


class TestDiversityMetrics:
    def test_initialize_diversity_metrics(self):
        diversity_metrics = DiversityMetrics(norm_of_sum=1.0, sum_of_norms=2.0)
        assertTrue(isinstance(diversity_metrics, DiversityMetrics))
        assertEqual(
            diversity_metrics.diversity_metric_type,
            DiversityMetricType.gradient_diversity,
        )

        # Test all comparators for both metrics of interest
        diversity_metrics_incr = DiversityMetrics(norm_of_sum=1.0, sum_of_norms=3.0)
        assertTrue(diversity_metrics_incr > diversity_metrics)
        assertTrue(diversity_metrics_incr >= diversity_metrics)
        assertTrue(diversity_metrics < diversity_metrics_incr)
        assertTrue(diversity_metrics <= diversity_metrics_incr)
        assertTrue(diversity_metrics != diversity_metrics_incr)
        assertTrue(not (diversity_metrics == diversity_metrics_incr))
        diversity_metrics.diversity_metric_type = DiversityMetricType.orthogonality
        diversity_metrics_incr.diversity_metric_type = DiversityMetricType.orthogonality
        assertTrue(diversity_metrics_incr > diversity_metrics)
        assertTrue(diversity_metrics_incr >= diversity_metrics)
        assertTrue(diversity_metrics < diversity_metrics_incr)
        assertTrue(diversity_metrics <= diversity_metrics_incr)
        assertTrue(diversity_metrics != diversity_metrics_incr)
        assertTrue(not (diversity_metrics == diversity_metrics_incr))

        n_test_cases = 1000
        std_dev = 500.0
        numerical_tol = 1e-04
        for _ in range(n_test_cases):
            norm_of_sum = abs(random.normalvariate(0, std_dev))
            sum_of_norms = abs(random.normalvariate(0, std_dev))
            diversity_metrics = DiversityMetrics(norm_of_sum, sum_of_norms)

            # Check bounds on and relationship between the two stored quantities
            assertTrue(diversity_metrics._recpr_gradient_diversity >= 0.0)
            assertTrue(diversity_metrics._recpr_orthogonality >= -1.0)
            assertTrue(
                diversity_metrics.orthogonality < 0
                or diversity_metrics.gradient_diversity
                <= diversity_metrics.orthogonality
            )
            assertTrue(
                math.isclose(
                    diversity_metrics._recpr_gradient_diversity
                    - diversity_metrics._recpr_orthogonality,
                    1.0,
                    rel_tol=numerical_tol,
                )
            )
            diversity_metrics_cpy = copy.deepcopy(diversity_metrics)
            diversity_metrics_eq = DiversityMetrics(norm_of_sum, sum_of_norms)
            assertEqual(diversity_metrics_cpy, diversity_metrics)
            assertEqual(diversity_metrics_eq, diversity_metrics)

            # Increasing sum_of_norms will increase both gradient diversity and orthogonality.
            # Increasing norm_of_sum will decrease them both.
            diversity_metrics_incr = DiversityMetrics(norm_of_sum, sum_of_norms * 1.01)
            diversity_metrics_decr = DiversityMetrics(norm_of_sum * 1.01, sum_of_norms)
            assertTrue(diversity_metrics_incr > diversity_metrics)
            assertTrue(diversity_metrics_decr < diversity_metrics)
            diversity_metrics.diversity_metric_type = DiversityMetricType.orthogonality
            diversity_metrics_incr.diversity_metric_type = (
                DiversityMetricType.orthogonality
            )
            diversity_metrics_decr.diversity_metric_type = (
                DiversityMetricType.orthogonality
            )
            assertTrue(diversity_metrics_incr > diversity_metrics)
            assertTrue(diversity_metrics_decr < diversity_metrics)
            diversity_metrics.diversity_metric_type = (
                DiversityMetricType.gradient_diversity
            )

            # Check that metric_value returns the value of the poperty of interest
            diversity_metrics.diversity_metric_type = DiversityMetricType.orthogonality
            assertTrue(
                math.isclose(
                    diversity_metrics.metric_value, diversity_metrics.orthogonality
                )
            )
            diversity_metrics.diversity_metric_type = (
                DiversityMetricType.gradient_diversity
            )
            assertTrue(
                math.isclose(
                    diversity_metrics.metric_value, diversity_metrics.gradient_diversity
                )
            )

            # Check the special case when norm_of_sum = sum_of_norms
            diversity_metrics_eq = DiversityMetrics(sum_of_norms, sum_of_norms)
            assertTrue(
                math.isclose(
                    diversity_metrics_eq._recpr_gradient_diversity,
                    1.0,
                    rel_tol=numerical_tol,
                )
            )
            assertTrue(
                math.isclose(
                    diversity_metrics_eq._recpr_orthogonality,
                    0.0,
                    rel_tol=numerical_tol,
                )
            )
            assertTrue(
                math.isclose(
                    diversity_metrics_eq.gradient_diversity, 1.0, rel_tol=numerical_tol
                )
            )
            assertTrue(math.isnan(diversity_metrics_eq.orthogonality))
