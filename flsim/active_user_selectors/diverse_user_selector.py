#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Tuple

import torch
from flsim.active_user_selectors.simple_user_selector import (
    ActiveUserSelector,
    ActiveUserSelectorConfig,
)
from flsim.common.diversity_metrics import (
    DiversityMetrics,
    DiversityMetricType,
    DiversityStatistics,
)
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.model import IFLModel
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg


class DiverseUserSelectorUtils:
    @staticmethod
    def calculate_diversity_metrics(
        data_provider: IFLDataProvider,
        global_model: IFLModel,
        user_indices: List[int],
        loss_reduction_type: str = "mean",
        client_gradient_scaling: str = "sum",
        diversity_metric_type: DiversityMetricType = DiversityMetricType.gradient_diversity,
    ) -> DiversityMetrics:
        """Calculates the Gradient Diversity of a cohort -- see arXiv:1706.05699
        For the FL application, define a aggregate gradient for each user, which is the sum of
        the gradient of the loss function for all the user's data samples (denoted grad f_i).
        We compute two quantities: norm_of_sum is the norm of the sum of these user
        gradients, and sum_of_norms is the sum of the norms of these gradients. These quantities
        are passed to a DiversityMetrics object which computes the gradient diversity metrics.

        norm_of_sum = || sum grad f_i ||_2^2
        sum_of_norms = sum || grad f_i ||_2^2

        loss_reduction_type should match the loss reduction type of the model (pytorch defaults to mean)
        client_gradient_scaling determines how the gradient for each client is computed. Either
        the gradient is summed over all the user's data, or it is the average client gradient.
        """

        if loss_reduction_type not in {"sum", "mean"}:
            raise ValueError('loss_reduction_type must be "sum" or "mean"')
        if client_gradient_scaling not in {"sum", "mean"}:
            raise ValueError('client_gradient_scaling must be "sum" or "mean"')

        global_model.fl_get_module().train()
        params = list(global_model.fl_get_module().parameters())

        def zero_like_grad(param_list: List[torch.Tensor]) -> List[torch.Tensor]:
            zero_gradient = []
            for group in param_list:
                if group.requires_grad:
                    zero_gradient.append(torch.zeros_like(group, requires_grad=False))
            return zero_gradient

        gradient_sum = zero_like_grad(params)
        norm_of_sum, sum_of_norms = 0.0, 0.0

        # Accumulate the gradient over a user's batches, normalized by batch size
        for user_idx in user_indices:
            user_data = data_provider.get_user_data(user_idx)
            accumulated_gradient = zero_like_grad(params)

            for batch in user_data:
                global_model.fl_get_module().zero_grad()
                batch_metrics = global_model.fl_forward(batch)
                batch_metrics.loss.backward()

                for group_in, group_out in zip(params, accumulated_gradient):
                    if loss_reduction_type == "mean":
                        group_out += group_in.grad * batch_metrics.num_examples
                    else:
                        group_out += group_in.grad

            for group_in, group_out in zip(accumulated_gradient, gradient_sum):
                if client_gradient_scaling == "mean":
                    group_in = group_in / user_data.num_examples()

                group_out += group_in
                sum_of_norms += torch.sum(group_in * group_in).item()

        for group in gradient_sum:
            norm_of_sum += torch.sum(group * group).item()

        return DiversityMetrics(
            norm_of_sum=norm_of_sum,
            sum_of_norms=sum_of_norms,
            diversity_metric_type=diversity_metric_type,
        )

    @staticmethod
    def select_diverse_cohort(
        data_provider: IFLDataProvider,
        global_model: IFLModel,
        users_per_round: int,
        available_users: List[int],
        rng: torch.Generator,
        num_search_samples: int,
        maximize_metric: bool,
        loss_reduction_type: str = "mean",
        client_gradient_scaling: str = "sum",
        diversity_metric_type: DiversityMetricType = DiversityMetricType.gradient_diversity,
    ) -> Tuple[List[int], DiversityStatistics]:
        """Choose a cohort which optimizes the diversity metric type out of some number
        of sample cohorts. Calculate the max, min, and average metric of the sample cohorts.

        Inputs:
        maximize_metric: whether the diversity metric is maximized (True) or minimized (False)

        Outputs:
        Cohort (out of the sample cohorts) which maximizes the metric
        Statistics of the metric of interest of the cohort.
        """

        if users_per_round >= len(available_users):
            candidate_selected_indices = copy.copy(available_users)
            candidate_metrics = DiverseUserSelectorUtils.calculate_diversity_metrics(
                data_provider=data_provider,
                global_model=global_model,
                user_indices=candidate_selected_indices,
                loss_reduction_type=loss_reduction_type,
                client_gradient_scaling=client_gradient_scaling,
                diversity_metric_type=diversity_metric_type,
            )
            cohort_stats = DiversityStatistics([candidate_metrics])

            return candidate_selected_indices, cohort_stats

        # Choose cohorts from the available users and calculate their diversity statistics.
        selected_indices = torch.multinomial(
            torch.ones(len(available_users), dtype=torch.float),
            users_per_round,
            replacement=False,
            generator=rng,
        ).tolist()

        candidate_user_indices = [available_users[idx] for idx in selected_indices]

        candidate_diversity_metrics = (
            DiverseUserSelectorUtils.calculate_diversity_metrics(
                data_provider=data_provider,
                global_model=global_model,
                user_indices=candidate_user_indices,
                loss_reduction_type=loss_reduction_type,
                client_gradient_scaling=client_gradient_scaling,
                diversity_metric_type=diversity_metric_type,
            )
        )

        sample_cohort_metrics = [candidate_diversity_metrics]

        for _ in range(num_search_samples - 1):
            selected_indices = torch.multinomial(
                torch.ones(len(available_users), dtype=torch.float),
                users_per_round,
                replacement=False,
                generator=rng,
            ).tolist()

            user_indices = [available_users[idx] for idx in selected_indices]
            cohort_diversity_metrics = (
                DiverseUserSelectorUtils.calculate_diversity_metrics(
                    data_provider=data_provider,
                    global_model=global_model,
                    user_indices=user_indices,
                    loss_reduction_type=loss_reduction_type,
                    client_gradient_scaling=client_gradient_scaling,
                    diversity_metric_type=diversity_metric_type,
                )
            )

            sample_cohort_metrics.append(cohort_diversity_metrics)

            if maximize_metric and (
                cohort_diversity_metrics > candidate_diversity_metrics
            ):
                candidate_user_indices = user_indices
                candidate_diversity_metrics = cohort_diversity_metrics
            elif (not maximize_metric) and (
                cohort_diversity_metrics < candidate_diversity_metrics
            ):
                candidate_user_indices = user_indices
                candidate_diversity_metrics = cohort_diversity_metrics

        cohort_stats = DiversityStatistics(sample_cohort_metrics)

        return candidate_user_indices, cohort_stats


class DiversityReportingUserSelector(ActiveUserSelector):
    """User Selector which chooses users uniformly randomly and records
    the Gradient Diversity (GD) of the cohort. For GD, see arXiv:1706.05699.

    If constant_cohorts is set to True, choose a set of cohorts at the start of
    training and at each round, report the GD of each of those sample cohorts.
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=DiversityReportingUserSelectorConfig,
            **kwargs,
        )

        super().__init__(**kwargs)

        if not isinstance(self.cfg.diversity_metric_type, DiversityMetricType):
            raise ValueError("diversity_metric_type must be of DiversityMetricType")
        if self.cfg.loss_reduction_type not in {"sum", "mean"}:
            raise ValueError('loss_reduction_type must be "sum" or "mean"')
        if self.cfg.client_gradient_scaling not in {"sum", "mean"}:
            raise ValueError('client_gradient_scaling must be "sum" or "mean"')

        self.sample_cohorts = []

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def get_user_indices(self, **kwargs) -> List[int]:
        required_inputs = [
            "num_total_users",
            "users_per_round",
            "data_provider",
            "global_model",
            "epoch",
        ]
        (
            num_total_users,
            users_per_round,
            data_provider,
            global_model,
            epoch,
        ) = self.unpack_required_inputs(required_inputs, kwargs)

        selected_indices = torch.multinomial(
            torch.ones(num_total_users, dtype=torch.float),
            users_per_round,
            replacement=False,
            generator=self.rng,
        ).tolist()

        currently_active = (
            # pyre-fixme[16]: `DiversityReportingUserSelector` has no attribute `cfg`.
            epoch >= self.cfg.epochs_before_active
            and epoch < self.cfg.epochs_before_active + self.cfg.num_epochs_active
        )

        if not currently_active:
            return selected_indices

        if len(self.sample_cohorts) == 0 or not self.cfg.constant_cohorts:
            self.sample_cohorts = []
            for _ in range(self.cfg.num_candidate_cohorts):
                sample_cohort_indices = torch.multinomial(
                    torch.ones(num_total_users, dtype=torch.float),
                    users_per_round,
                    replacement=False,
                    generator=self.rng,
                ).tolist()

                self.sample_cohorts.append(sample_cohort_indices)

        diversity_metrics_list = []
        for sample_cohort in self.sample_cohorts:
            diversity_metrics_list.append(
                DiverseUserSelectorUtils.calculate_diversity_metrics(
                    data_provider=data_provider,
                    global_model=global_model,
                    user_indices=sample_cohort,
                    loss_reduction_type=self.cfg.loss_reduction_type,
                    client_gradient_scaling=self.cfg.client_gradient_scaling,
                    diversity_metric_type=self.cfg.diversity_metric_type,
                )
            )

        return selected_indices


class DiversityStatisticsReportingUserSelector(ActiveUserSelector):
    """Find the statistics of the diversity metric of interest.
    The cohort is selected uniformly randomly with replacement.
    Record the sample maximum, average, and minimum metric for:
        1) cohorts chosen with replacement
        2) cohorts chosen without replacement (round-robin), where the cohort
        with the maximum metric is removed from the list of available users.
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=DiversityStatisticsReportingUserSelectorConfig,
            **kwargs,
        )
        super().__init__(**kwargs)

        if not isinstance(self.cfg.diversity_metric_type, DiversityMetricType):
            raise ValueError("diversity_metric_type must be of DiversityMetricType")
        if self.cfg.loss_reduction_type not in {"sum", "mean"}:
            raise ValueError('loss_reduction_type must be "sum" or "mean"')
        if self.cfg.client_gradient_scaling not in {"sum", "mean"}:
            raise ValueError('client_gradient_scaling must be "sum" or "mean"')

        self.available_users = []
        self.cohort_stats_with_replacement = []
        self.cohort_stats_without_replacement = []

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def get_user_indices(self, **kwargs) -> List[int]:
        required_inputs = [
            "num_total_users",
            "users_per_round",
            "data_provider",
            "global_model",
            "epoch",
        ]
        (
            num_total_users,
            users_per_round,
            data_provider,
            global_model,
            epoch,
        ) = self.unpack_required_inputs(required_inputs, kwargs)

        if len(self.available_users) == 0:
            self.available_users = list(range(num_total_users))

        # First uniformly randomly select the actual cohort used for training
        baseline_selected_indices = torch.multinomial(
            torch.ones(num_total_users, dtype=torch.float),
            users_per_round,
            replacement=False,
            generator=self.rng,
        ).tolist()

        if (
            # pyre-fixme[16]: `DiversityStatisticsReportingUserSelector` has no
            #  attribute `cfg`.
            epoch < self.cfg.epochs_before_active
            or epoch >= self.cfg.epochs_before_active + self.cfg.num_epochs_active
        ):
            return baseline_selected_indices

        (_, stat_with_replacement) = DiverseUserSelectorUtils.select_diverse_cohort(
            data_provider=data_provider,
            global_model=global_model,
            users_per_round=users_per_round,
            available_users=list(range(num_total_users)),
            rng=self.rng,
            num_search_samples=self.cfg.num_candidate_cohorts,
            maximize_metric=self.cfg.maximize_metric,
            loss_reduction_type=self.cfg.loss_reduction_type,
            client_gradient_scaling=self.cfg.client_gradient_scaling,
            diversity_metric_type=self.cfg.diversity_metric_type,
        )

        (
            candidate_user_indices,
            stat_without_replacement,
        ) = DiverseUserSelectorUtils.select_diverse_cohort(
            data_provider=data_provider,
            global_model=global_model,
            users_per_round=users_per_round,
            available_users=self.available_users,
            rng=self.rng,
            num_search_samples=self.cfg.num_candidate_cohorts,
            maximize_metric=self.cfg.maximize_metric,
            loss_reduction_type=self.cfg.loss_reduction_type,
            client_gradient_scaling=self.cfg.client_gradient_scaling,
            diversity_metric_type=self.cfg.diversity_metric_type,
        )

        self.cohort_stats_with_replacement.append(stat_with_replacement)
        self.cohort_stats_without_replacement.append(stat_without_replacement)

        # Update the list of available users for round-robin selector
        self.available_users = [
            idx for idx in self.available_users if idx not in candidate_user_indices
        ]

        return baseline_selected_indices


class DiversityMaximizingUserSelector(ActiveUserSelector):
    """A user selector class which chooses cohorts to greedily maximize their Diversity.

    Config options:
    num_candidate_cohorts: The number of cohorts considered; the cohort with maximum Diversity
    loss_reduction_type: This should match the loss reduction type in the learning model used
    client_gradient_scaling: How to compute the gradient summary of a client -- whether is should
        be the sum or average of the gradients of its data.
    diversity_metric_type: What is the Diversity Metric which is being optimized
    maximize_metric: Whether to maximize (True) or minimize (False) the specified diversity metric
    with_replacement: If false, choose cohorts in a round-robin fashion. If true, select each cohort
        from all clients at each round.
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=DiversityMaximizingUserSelectorConfig,
            **kwargs,
        )

        super().__init__(**kwargs)

        if not isinstance(self.cfg.diversity_metric_type, DiversityMetricType):
            raise ValueError("diversity_metric_type must be of DiversityMetricType")
        if self.cfg.loss_reduction_type not in {"sum", "mean"}:
            raise ValueError('loss_reduction_type must be "sum" or "mean"')
        if self.cfg.client_gradient_scaling not in {"sum", "mean"}:
            raise ValueError('client_gradient_scaling must be "sum" or "mean"')

        self.available_users = []

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def get_user_indices(self, **kwargs) -> List[int]:
        required_inputs = [
            "num_total_users",
            "users_per_round",
            "data_provider",
            "global_model",
            "epoch",
        ]
        (
            num_total_users,
            users_per_round,
            data_provider,
            global_model,
            epoch,
        ) = self.unpack_required_inputs(required_inputs, kwargs)

        if len(self.available_users) == 0:
            self.available_users = list(range(num_total_users))

        if (
            # pyre-fixme[16]: `DiversityMaximizingUserSelector` has no attribute `cfg`.
            epoch < self.cfg.epochs_before_active
            or epoch >= self.cfg.epochs_before_active + self.cfg.num_epochs_active
        ):
            baseline_selected_indices = torch.multinomial(
                torch.ones(num_total_users, dtype=torch.float),
                users_per_round,
                replacement=False,
                generator=self.rng,
            ).tolist()
            return baseline_selected_indices

        (candidate_user_indices, _,) = DiverseUserSelectorUtils.select_diverse_cohort(
            data_provider=data_provider,
            global_model=global_model,
            users_per_round=users_per_round,
            available_users=self.available_users,
            rng=self.rng,
            num_search_samples=self.cfg.num_candidate_cohorts,
            maximize_metric=self.cfg.maximize_metric,
            loss_reduction_type=self.cfg.loss_reduction_type,
            client_gradient_scaling=self.cfg.client_gradient_scaling,
            diversity_metric_type=self.cfg.diversity_metric_type,
        )

        if not self.cfg.with_replacement:
            self.available_users = [
                idx for idx in self.available_users if idx not in candidate_user_indices
            ]

        return candidate_user_indices


@dataclass
class DiversityReportingUserSelectorConfig(ActiveUserSelectorConfig):
    _target_: str = fullclassname(DiversityReportingUserSelector)
    diversity_metric_type: DiversityMetricType = DiversityMetricType.gradient_diversity
    epochs_before_active: int = 0
    num_epochs_active: int = int(10e8)
    num_candidate_cohorts: int = 10
    loss_reduction_type: str = "mean"
    client_gradient_scaling: str = "sum"
    constant_cohorts: bool = False


@dataclass
class DiversityStatisticsReportingUserSelectorConfig(ActiveUserSelectorConfig):
    _target_: str = fullclassname(DiversityStatisticsReportingUserSelector)
    diversity_metric_type: DiversityMetricType = DiversityMetricType.gradient_diversity
    maximize_metric: bool = True
    epochs_before_active: int = 0
    num_epochs_active: int = int(10e8)
    num_candidate_cohorts: int = 10
    loss_reduction_type: str = "mean"
    client_gradient_scaling: str = "sum"


@dataclass
class DiversityMaximizingUserSelectorConfig(ActiveUserSelectorConfig):
    _target_: str = fullclassname(DiversityMaximizingUserSelector)
    diversity_metric_type: DiversityMetricType = DiversityMetricType.gradient_diversity
    maximize_metric: bool = True
    epochs_before_active: int = 0
    num_epochs_active: int = int(10e8)
    num_candidate_cohorts: int = 10
    loss_reduction_type: str = "mean"
    client_gradient_scaling: str = "sum"
    with_replacement: bool = False
