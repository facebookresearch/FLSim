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
