#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from hydra.core.config_store import ConfigStore  #  @manual

from .simple_user_selector import (
    ImportanceSamplingActiveUserSelectorConfig,
    RandomMultiStepActiveUserSelectorConfig,
    RandomRoundRobinActiveUserSelectorConfig,
    SequentialActiveUserSelectorConfig,
    UniformlyRandomActiveUserSelectorConfig,
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
    name="base_importance_sampling_active_user_selector",
    node=ImportanceSamplingActiveUserSelectorConfig,
    group="active_user_selector",
)

ConfigStore.instance().store(
    name="base_random_multi_step_active_user_selector",
    node=RandomMultiStepActiveUserSelectorConfig,
    group="active_user_selector",
)
