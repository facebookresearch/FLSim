#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

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
