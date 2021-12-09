#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from hydra.core.config_store import ConfigStore  #  @manual

from .secure_aggregator import FixedPointConfig

ConfigStore.instance().store(
    name="base_fixedpoint",
    node=FixedPointConfig,
    group="fixedpoint",
)
