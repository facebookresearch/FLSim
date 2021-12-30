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
