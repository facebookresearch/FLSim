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

from hydra.core.config_store import ConfigStore

from .data_sharder import (
    RandomSharderConfig,
    SequentialSharderConfig,
    RoundRobinSharderConfig,
    BroadcastSharderConfig,
    ColumnSharderConfig,
    PowerLawSharderConfig,
)

ConfigStore.instance().store(
    name="base_random_sharder",
    node=RandomSharderConfig,
    group="sharder",
)

ConfigStore.instance().store(
    name="base_sequential_sharder",
    node=SequentialSharderConfig,
    group="sharder",
)

ConfigStore.instance().store(
    name="base_round_robin_sharder",
    node=RoundRobinSharderConfig,
    group="sharder",
)

ConfigStore.instance().store(
    name="base_column_sharder",
    node=ColumnSharderConfig,
    group="sharder",
)

ConfigStore.instance().store(
    name="base_broadcast_sharder",
    node=BroadcastSharderConfig,
    group="sharder",
)

ConfigStore.instance().store(
    name="base_power_law_sharder",
    node=PowerLawSharderConfig,
    group="sharder",
)
