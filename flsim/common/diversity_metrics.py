# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from enum import Enum
from typing import List, Tuple


class DiversityMetricType(Enum):
    gradient_diversity = 1
    orthogonality = 2
    delta_norm_sq = 3
    sum_client_delta_norm_sq = 4
    sum_client_delta_mutual_angles = 5


class DiversityMetrics:
    """
    This class stores diversity metrics for a set of gradients -- see arXiv:1706.05699
    For a set of vectors grad f_i,

          sum || grad f_i ||_2^2          sum_of_norms
    GD =  ______________________    = ___________________
          || sum grad f_i ||_2^2          norm_of_sum

                    sum || grad f_i ||_2^2                  sum_of_norms
    ORTH =  ___________________________________ =  ____________________________
            sum{i != j} dot(grad f_i, grad f_j)     norm_of_sum - sum_of_norms


    delta_norm_sq =  norm_of_sum

    sum_client_delta_norm_sq = sum_of_norms

    sum_client_delta_mutual_angles = norm_of_sum - sum_of_norms


    For any set of n vectors, GD >= 1/n. ORTH and GD can be infinite,
    so we store the reciprocal of these quantities. This gives us good bounds:
    0 <= 1/GD <= n and -1 <= 1/ORTH <= n-1

    delta_norm_sq is the magnitude of the aggregate gradient step.
    This is a nonnegative quantity.

    sum_client_delta_norm_sq is the sum of client gradient updates.

    sum_client_delta_mutual_angles depends only on the sum of mutual angles
    between each pair client gradients. It can be positive or negative.

    The comparators will output a comparison of GD (or ORTH), not 1/GD.
    So a > b means GD(a) > GD(b), i.e. internal comparison of 1/GD(a) < 1/GD(b)

    Inputs:
    norm_of_sum >= 0
    sum_of_norms > 0
    """

    def __init__(
        self,
        norm_of_sum: float,
        sum_of_norms: float,
        diversity_metric_type: DiversityMetricType = DiversityMetricType.gradient_diversity,
    ):

        if not isinstance(diversity_metric_type, DiversityMetricType):
            raise ValueError("diversity_metric_type must be of DiversityMetricType")

        self._recpr_gradient_diversity = norm_of_sum / sum_of_norms
        self._recpr_orthogonality = (norm_of_sum - sum_of_norms) / sum_of_norms
        self._norm_of_sum = norm_of_sum
        self._sum_of_norms = sum_of_norms
        self._diversity_metric_type = diversity_metric_type

    # Provide getters but not setters, since the GD and ORTH are not actually stored.
    @property
    def gradient_diversity(self) -> float:
        if self._recpr_gradient_diversity == 0:
            return math.nan
        return 1.0 / self._recpr_gradient_diversity

    @property
    def orthogonality(self) -> float:
        if self._recpr_orthogonality == 0:
            return math.nan
        return 1.0 / self._recpr_orthogonality

    @property
    def delta_norm_sq(self) -> float:
        return self._norm_of_sum

    @property
    def sum_client_delta_norm_sq(self) -> float:
        return self._sum_of_norms

    @property
    def sum_client_delta_mutual_angles(self) -> float:
        return self._norm_of_sum - self._sum_of_norms

    @property
    def metric_value(self) -> float:
        if self._diversity_metric_type == DiversityMetricType.orthogonality:
            return self.orthogonality
        elif self._diversity_metric_type == DiversityMetricType.delta_norm_sq:
            return self.delta_norm_sq
        elif (
            self._diversity_metric_type == DiversityMetricType.sum_client_delta_norm_sq
        ):
            return self.sum_client_delta_norm_sq
        elif (
            self._diversity_metric_type
            == DiversityMetricType.sum_client_delta_mutual_angles
        ):
            return self.sum_client_delta_mutual_angles
        return self.gradient_diversity

    @property
    def diversity_metric_type(self) -> DiversityMetricType:
        return self._diversity_metric_type

    @diversity_metric_type.setter
    def diversity_metric_type(self, diversity_metric_type: DiversityMetricType):
        if not isinstance(self.diversity_metric_type, DiversityMetricType):
            raise ValueError("diversity_metric_type must be of DiversityMetricType")
        self._diversity_metric_type = diversity_metric_type

    # Metrics must not be 'None' to be equal. If LHS has a value and RHS is
    # 'None', LHS is greater. Since quantities considered are reciprocals, the
    # inequalities below are reversed from what one would expect.
    def __eq__(self, other):
        v1, v2 = self._get_metrics_of_interest(other)
        return math.isclose(v1, v2, rel_tol=1e-06)

    def __ne__(self, other):
        v1, v2 = self._get_metrics_of_interest(other)
        return v1 != v2

    def __gt__(self, other):
        v1, v2 = self._get_metrics_of_interest(other)
        return v1 < v2

    def __lt__(self, other):
        v1, v2 = self._get_metrics_of_interest(other)
        return v1 > v2

    def __ge__(self, other):
        v1, v2 = self._get_metrics_of_interest(other)
        return v1 <= v2

    def __le__(self, other):
        v1, v2 = self._get_metrics_of_interest(other)
        return v1 >= v2

    def __repr__(self):
        return "%5.5f" % self.metric_value

    def _get_metrics_of_interest(self, other) -> Tuple[float, float]:
        # For comparison, the two objects must have the same metric of interest
        assert self.diversity_metric_type == other.diversity_metric_type

        v1 = self._recpr_gradient_diversity
        v2 = other._recpr_gradient_diversity
        if self.diversity_metric_type == DiversityMetricType.orthogonality:
            v1 = self._recpr_orthogonality
            v2 = other._recpr_orthogonality
        elif self.diversity_metric_type == DiversityMetricType.delta_norm_sq:
            # Use negatives for comparison, since internal comparators are flipped
            v1 = -self.delta_norm_sq
            v2 = -other.delta_norm_sq
        elif self.diversity_metric_type == DiversityMetricType.sum_client_delta_norm_sq:
            # Use negatives for comparison, since internal comparators are flipped
            v1 = -self.sum_client_delta_norm_sq
            v2 = -other.sum_client_delta_norm_sq
        elif (
            self.diversity_metric_type
            == DiversityMetricType.sum_client_delta_mutual_angles
        ):
            # Use negatives for comparison, since internal comparators are flipped
            v1 = -self.sum_client_delta_mutual_angles
            v2 = -other.sum_client_delta_mutual_angles

        return v1, v2


class DiversityStatistics:
    def __init__(self, diversity_metrics_list: List[DiversityMetrics]):

        if len(diversity_metrics_list) == 0:
            raise ValueError("diversity_metrics_list must not be empty")

        metrics = [
            cohort_metric.metric_value for cohort_metric in diversity_metrics_list
        ]

        self.maximum_metric = max(metrics)
        self.average_metric = sum(metrics) / len(metrics)
        self.minimum_metric = min(metrics)

    def __eq__(self, other):
        close_max = math.isclose(
            self.maximum_metric, other.maximum_metric, rel_tol=1e-04
        )
        close_avg = math.isclose(
            self.average_metric, other.average_metric, rel_tol=1e-04
        )
        close_min = math.isclose(
            self.minimum_metric, other.minimum_metric, rel_tol=1e-04
        )
        return close_max and close_avg and close_min

    def __repr__(self):
        return "%5.5f,%5.5f,%5.5f" % (
            self.maximum_metric,
            self.average_metric,
            self.minimum_metric,
        )
