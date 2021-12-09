#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
