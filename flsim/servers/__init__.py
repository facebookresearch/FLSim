#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from hydra.core.config_store import ConfigStore  #  @manual

from .sync_dp_servers import SyncDPSGDServerConfig
from .sync_fedshuffle_servers import SyncFedShuffleServerConfig
from .sync_ftrl_servers import SyncFTRLServerConfig
from .sync_mime_servers import SyncMimeServerConfig
from .sync_mimelite_servers import SyncMimeLiteServerConfig
from .sync_secagg_servers import SyncSecAggServerConfig, SyncSecAggSQServerConfig
from .sync_servers import (
    SyncPQServerConfig,
    SyncServerConfig,
    SyncSharedSparseServerConfig,
    SyncSQServerConfig,
)

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
    name="base_sync_mime_server",
    node=SyncMimeServerConfig,
    group="server",
)

ConfigStore.instance().store(
    name="base_sync_mimelite_server",
    node=SyncMimeLiteServerConfig,
    group="server",
)

ConfigStore.instance().store(
    name="base_sync_fedshuffle_server",
    node=SyncFedShuffleServerConfig,
    group="server",
)

ConfigStore.instance().store(
    name="base_sync_sq_server",
    node=SyncSQServerConfig,
    group="server",
)

ConfigStore.instance().store(
    name="base_sync_secagg_sq_server",
    node=SyncSecAggSQServerConfig,
    group="server",
)

ConfigStore.instance().store(
    name="base_sync_ftrl_server",
    node=SyncFTRLServerConfig,
    group="server",
)

ConfigStore.instance().store(
    name="base_sync_pq_server",
    node=SyncPQServerConfig,
    group="server",
)

ConfigStore.instance().store(
    name="base_sync_shared_sparse_server",
    node=SyncSharedSparseServerConfig,
    group="server",
)
