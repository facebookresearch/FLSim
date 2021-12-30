#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
