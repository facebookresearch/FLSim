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

from hydra.core.config_store import ConfigStore

from .base_channel import FLChannelConfig
from .half_precision_channel import HalfPrecisionChannelConfig
from .scalar_quantization_channel import ScalarQuantizationChannelConfig
from .sparse_mask_channel import SparseMaskChannelConfig

ConfigStore.instance().store(
    name="base_identity_channel",
    node=FLChannelConfig,
    group="channel",
)

ConfigStore.instance().store(
    name="base_sparse_mask_channel",
    node=SparseMaskChannelConfig,
    group="channel",
)

ConfigStore.instance().store(
    name="base_half_precision_channel",
    node=HalfPrecisionChannelConfig,
    group="channel",
)

ConfigStore.instance().store(
    name="base_scalar_quantization_channel",
    node=ScalarQuantizationChannelConfig,
    group="channel",
)
