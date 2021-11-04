#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from hydra.core.config_store import ConfigStore

from .async_trainer import AsyncTrainerConfig
from .private_sync_trainer import PrivateSyncTrainerConfig
from .sync_trainer import SyncTrainerConfig

ConfigStore.instance().store(
    name="base_sync_trainer",
    node=SyncTrainerConfig,
    group="trainer",
)

ConfigStore.instance().store(
    name="base_async_trainer",
    node=AsyncTrainerConfig,
    group="trainer",
)

ConfigStore.instance().store(
    name="base_private_sync_trainer",
    node=PrivateSyncTrainerConfig,
    group="trainer",
)
