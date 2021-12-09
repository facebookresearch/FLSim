#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from hydra.core.config_store import ConfigStore  #  @manual

from .base_round_reducer import RoundReducerConfig
from .dp_round_reducer import DPRoundReducerConfig
from .secure_round_reducer import SecureRoundReducerConfig
from .weighted_dp_round_reducer import WeightedDPRoundReducerConfig


ConfigStore.instance().store(
    name="base_reducer",
    node=RoundReducerConfig,
    group="reducer",
)

ConfigStore.instance().store(
    name="base_dp_reducer",
    node=DPRoundReducerConfig,
    group="reducer",
)

ConfigStore.instance().store(
    name="base_secure_reducer",
    node=SecureRoundReducerConfig,
    group="reducer",
)

ConfigStore.instance().store(
    name="base_weighted_dp_reducer",
    node=WeightedDPRoundReducerConfig,
    group="reducer",
)
