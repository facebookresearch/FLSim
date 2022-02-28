#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from hydra.core.config_store import ConfigStore

from .base_client import ClientConfig
from .bilevel_client import BiLevelClientConfig
from .cd_client import CDClientConfig
from .dp_client import DPClientConfig
from .sarah_client import SarahClientConfig

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
    name="base_sarah_client",
    node=SarahClientConfig,
    group="client",
)

ConfigStore.instance().store(
    name="base_bilevel_client",
    node=BiLevelClientConfig,
    group="client",
)

ConfigStore.instance().store(
    name="base_cd_client",
    node=CDClientConfig,
    group="client",
)

ConfigStore.instance().store(
    name="base_ditto_client",
    node=ClientConfig,
    group="client",
)
