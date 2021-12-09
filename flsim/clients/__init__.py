#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from hydra.core.config_store import ConfigStore

from .base_client import ClientConfig
from .dp_client import DPClientConfig


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
