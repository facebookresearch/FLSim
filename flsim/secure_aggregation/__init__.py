#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from hydra.core.config_store import ConfigStore  #  @manual

from .secure_aggregator import FixedPointConfig

ConfigStore.instance().store(
    name="base_fixedpoint",
    node=FixedPointConfig,
    group="fixedpoint",
)
