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

from .base_round_reducer import RoundReducerConfig
from .dp_round_reducer import DPRoundReducerConfig
from .secure_round_reducer import SecureRoundReducerConfig
from .weighted_dp_round_reducer import WeightedDPRoundReducerConfig


ConfigStore.instance().store(
    name="base_reducer",
    node=RoundReducerConfig,
    group="reducer",
)

ConfigStore.instance().store(
    name="base_dp_reducer",
    node=DPRoundReducerConfig,
    group="reducer",
)

ConfigStore.instance().store(
    name="base_secure_reducer",
    node=SecureRoundReducerConfig,
    group="reducer",
)

ConfigStore.instance().store(
    name="base_weighted_dp_reducer",
    node=WeightedDPRoundReducerConfig,
    group="reducer",
)
