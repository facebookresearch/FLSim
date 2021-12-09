#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
