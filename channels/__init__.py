#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from hydra.core.config_store import ConfigStore

from .base_channel import FLChannelConfig
from .half_precision_channel import HalfPrecisionChannelConfig
from .product_quantization_channel import ProductQuantizationChannelConfig
from .random_mask_channel import RandomMaskChannelConfig
from .scalar_quantization_channel import ScalarQuantizationChannelConfig
from .sketch_channel import SketchChannelConfig

ConfigStore.instance().store(
    name="base_identity_channel",
    node=FLChannelConfig,
    group="channel",
)

ConfigStore.instance().store(
    name="base_random_mask_channel",
    node=RandomMaskChannelConfig,
    group="channel",
)

ConfigStore.instance().store(
    name="base_sketch_channel",
    node=SketchChannelConfig,
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

ConfigStore.instance().store(
    name="base_product_quantization_channel",
    node=ProductQuantizationChannelConfig,
    group="channel",
)
