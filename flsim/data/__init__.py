#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
