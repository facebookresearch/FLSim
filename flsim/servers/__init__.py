#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from hydra.core.config_store import ConfigStore  #  @manual

from .sync_dp_servers import SyncDPSGDServerConfig
from .sync_secagg_servers import SyncSecAggServerConfig
from .sync_servers import SyncServerConfig

ConfigStore.instance().store(
    name="base_sync_server",
    node=SyncServerConfig,
    group="server",
)

ConfigStore.instance().store(
    name="base_sync_dp_server",
    node=SyncDPSGDServerConfig,
    group="server",
)

ConfigStore.instance().store(
    name="base_sync_secagg_server",
    node=SyncSecAggServerConfig,
    group="server",
)
