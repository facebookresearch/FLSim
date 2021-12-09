#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
