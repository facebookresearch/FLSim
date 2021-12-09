#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from hydra.core.config_store import ConfigStore  #  @manual

from .training_duration_distribution import (
    PerExampleGaussianDurationDistributionConfig,
    PerUserGaussianDurationDistributionConfig,
    PerUserHalfNormalDurationDistributionConfig,
    PerUserUniformDurationDistributionConfig,
    PerUserExponentialDurationDistributionConfig,
    DurationDistributionFromListConfig,
)


ConfigStore.instance().store(
    name="base_per_example_gaussian_duration_distribution",
    node=PerExampleGaussianDurationDistributionConfig,
    group="duration_distribution_generator",
)


ConfigStore.instance().store(
    name="base_per_user_gaussian_duration_distribution",
    node=PerUserGaussianDurationDistributionConfig,
    group="duration_distribution_generator",
)


ConfigStore.instance().store(
    name="base_per_user_half_normal_duration_distribution",
    node=PerUserHalfNormalDurationDistributionConfig,
    group="duration_distribution_generator",
)


ConfigStore.instance().store(
    name="base_per_user_uniform_duration_distribution",
    node=PerUserUniformDurationDistributionConfig,
    group="duration_distribution_generator",
)


ConfigStore.instance().store(
    name="base_per_user_exponential_duration_distribution",
    node=PerUserExponentialDurationDistributionConfig,
    group="duration_distribution_generator",
)


ConfigStore.instance().store(
    name="base_duration_distribution_from_list",
    node=DurationDistributionFromListConfig,
    group="duration_distribution_generator",
)
