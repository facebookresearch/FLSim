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

from .diverse_user_selector import (
    DiversityReportingUserSelectorConfig,
    DiversityStatisticsReportingUserSelectorConfig,
    DiversityMaximizingUserSelectorConfig,
)
from .simple_user_selector import (
    UniformlyRandomActiveUserSelectorConfig,
    SequentialActiveUserSelectorConfig,
    RandomRoundRobinActiveUserSelectorConfig,
    NumberOfSamplesActiveUserSelectorConfig,
    HighLossActiveUserSelectorConfig,
)


ConfigStore.instance().store(
    name="base_uniformly_random_active_user_selector",
    node=UniformlyRandomActiveUserSelectorConfig,
    group="active_user_selector",
)


ConfigStore.instance().store(
    name="base_sequential_active_user_selector",
    node=SequentialActiveUserSelectorConfig,
    group="active_user_selector",
)


ConfigStore.instance().store(
    name="base_random_round_robin_active_user_selector",
    node=RandomRoundRobinActiveUserSelectorConfig,
    group="active_user_selector",
)


ConfigStore.instance().store(
    name="base_number_of_samples_active_user_selector",
    node=NumberOfSamplesActiveUserSelectorConfig,
    group="active_user_selector",
)

ConfigStore.instance().store(
    name="base_high_loss_active_user_selector",
    node=HighLossActiveUserSelectorConfig,
    group="active_user_selector",
)

ConfigStore.instance().store(
    name="base_diversity_reporting_user_selector",
    node=DiversityReportingUserSelectorConfig,
    group="active_user_selector",
)

ConfigStore.instance().store(
    name="base_diversity_statistics_reporting_user_selector",
    node=DiversityStatisticsReportingUserSelectorConfig,
    group="active_user_selector",
)

ConfigStore.instance().store(
    name="base_diversity_maximizing_user_selector",
    node=DiversityMaximizingUserSelectorConfig,
    group="active_user_selector",
)
