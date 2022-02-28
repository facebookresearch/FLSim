#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from hydra.core.config_store import ConfigStore  #  @manual

from .cd_server import CDServerConfig
from .sarah_server import SarahServerConfig
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

ConfigStore.instance().store(
    name="base_cd_server",
    node=CDServerConfig,
    group="server",
)

ConfigStore.instance().store(
    name="base_sarah_server",
    node=SarahServerConfig,
    group="server",
)
