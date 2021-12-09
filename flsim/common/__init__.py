#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from hydra.core.config_store import ConfigStore  #  @manual

from .timeout_simulator import (
    NeverTimeOutSimulatorConfig,
    GaussianTimeOutSimulatorConfig,
)


ConfigStore.instance().store(
    name="base_never_timeout_simulator",
    node=NeverTimeOutSimulatorConfig,
    group="timeout_simulator",
)


ConfigStore.instance().store(
    name="base_gaussian_timeout_simulator",
    node=GaussianTimeOutSimulatorConfig,
    group="timeout_simulator",
)
