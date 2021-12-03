#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

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
