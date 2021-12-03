#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

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
