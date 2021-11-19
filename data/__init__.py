#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

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
