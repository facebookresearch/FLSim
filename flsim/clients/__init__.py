#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from hydra.core.config_store import ConfigStore

from .base_client import ClientConfig
from .dp_client import DPClientConfig
from .sync_fedshuffle_client import FedShuffleClientConfig
from .sync_mime_client import MimeClientConfig
from .sync_mimelite_client import MimeLiteClientConfig


ConfigStore.instance().store(
    name="base_client",
    node=ClientConfig,
    group="client",
)

ConfigStore.instance().store(
    name="base_dp_client",
    node=DPClientConfig,
    group="client",
)

ConfigStore.instance().store(
    name="base_mime_client",
    node=MimeClientConfig,
    group="client",
)

ConfigStore.instance().store(
    name="base_mimelite_client",
    node=MimeLiteClientConfig,
    group="client",
)

ConfigStore.instance().store(
    name="base_fedshuffle_client",
    node=FedShuffleClientConfig,
    group="client",
)
