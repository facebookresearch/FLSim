#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from dataclasses import dataclass

from omegaconf import MISSING
from torch.utils.data import Dataset


class FLDataset(Dataset, abc.ABC):
    """Abstract class that all Dataset for FL will be implementing for
    stronger type-checking and better abstraction. (e.g. FLDatasetDataLoader)
    """

    pass


@dataclass
class DatasetConfig:
    _target_: str = MISSING
