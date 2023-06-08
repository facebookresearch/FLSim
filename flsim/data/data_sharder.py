#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, TypeVar

import numpy as np
import torch
from flsim.utils.config_utils import fullclassname, init_self_cfg
from flsim.data.partition_utils import create_lda_partitions
from omegaconf import MISSING
from torch.utils.data import Dataset
from tqdm import tqdm


Shardable = TypeVar("Shardable", Iterable[Dict[str, Any]], Dataset)


class FLDataSharder(abc.ABC):
    """This class takes in a file, and partitions it into a list of datasets.
    It supports random partitioning, broadcasting,
    round-robining and sharding by a column
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=FLDataSharderConfig,
            **kwargs,
        )

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    @abc.abstractmethod
    def shard_for_row(self, csv_row: Dict[Any, Any]) -> List[Any]:
        """
        Determine which shard a row should belong to.
        """
        pass

    def shard_rows(self, data_rows: Shardable) -> Iterable[Tuple[str, Any]]:
        """Partition a set of rows into multiple sets using a sharding strategy.

        Args:
            data_rows: Iterable[Dict[str, Any]]]: iterable over dictionary mapping column
            name to value.
        """
        shards = defaultdict(list)
        # pyre-fixme[16]: `Dataset` has no attribute `__iter__`.
        for one_row in data_rows:
            for shard_id in self.shard_for_row(one_row):
                shards[str(shard_id)].append(one_row)
        return shards.items()


class RandomSharder(FLDataSharder):
    """Splits training data randomly. num_shards should be specified.
    Assigns first num_shards rows to different shards to avoid empty shards."""

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=RandomSharderConfig,
            **kwargs,
        )
        super().__init__(**kwargs)
        self._assignments: List[int] = list(range(self.cfg.num_shards))
        random.shuffle(self._assignments)

    def shard_for_row(self, csv_row: Dict[Any, Any]) -> List[int]:
        if self._assignments:
            return [self._assignments.pop(0)]
        # pyre-fixme[16]: `RandomSharder` has no attribute `cfg`.
        return [random.randint(0, self.cfg.num_shards - 1)]

    def shard_rows(self, data_rows: Shardable) -> Iterable[Tuple[str, Any]]:
        # sanity check to avoid empty shards
        shards = super().shard_rows(data_rows)
        assert (
            sum(1 for _shard in shards)
            # pyre-fixme[16]: `RandomSharder` has no attribute `cfg`.
            >= self.cfg.num_shards
        ), "number of rows must be at least the number of shards"
        return shards


class SequentialSharder(FLDataSharder):
    """Assign first N rows to shard A, the next N rows to shard B, and so
    on. Mostly used for testing purpose (e.g. sanity-check with
    conventional SGD training based on PyTorch Dataset and its batching
    mechanism).
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=SequentialSharderConfig,
            **kwargs,
        )
        super().__init__(**kwargs)
        self.filled_in_current_bucket = 0
        self.index = 0

    def shard_for_row(self, csv_row: Dict[Any, Any]) -> List[int]:
        # pyre-fixme[16]: `SequentialSharder` has no attribute `cfg`.
        if self.filled_in_current_bucket == self.cfg.examples_per_shard:
            self.index += 1
            self.filled_in_current_bucket = 1
        else:
            self.filled_in_current_bucket += 1
        return [self.index]


class BroadcastSharder(FLDataSharder):
    """Copy each training datum to all shards. num_shards should be specified"""

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=BroadcastSharderConfig,
            **kwargs,
        )
        super().__init__(**kwargs)

    def shard_for_row(self, csv_row: Dict[Any, Any]) -> List[int]:
        # pyre-fixme[16]: `BroadcastSharder` has no attribute `cfg`.
        return list(range(self.cfg.num_shards))


class ColumnSharder(FLDataSharder):
    """Specify a column name used to shard.
    It should be the last column in the file,
    and sharding_column must be specified.
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=ColumnSharderConfig,
            **kwargs,
        )
        super().__init__(**kwargs)

    def shard_for_row(self, csv_row: Dict[Any, Any]) -> List[Any]:
        # pyre-fixme[16]: `ColumnSharder` has no attribute `cfg`.
        unwrapped_colindex = self.cfg.sharding_col
        if unwrapped_colindex.isdigit():
            unwrapped_colindex = int(unwrapped_colindex)
            assert unwrapped_colindex < len(csv_row), "Sharding index out of bounds: "
            f"{unwrapped_colindex}"

        shard_idx = csv_row[unwrapped_colindex]

        # shard_idx can be a 0-dim tensor when a table column is an integer.
        if isinstance(shard_idx, torch.Tensor):
            shard_idx = shard_idx.item()

        return [shard_idx]


class RoundRobinSharder(FLDataSharder):
    """Splits training in a round-robin fashion between all shards"""

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=RoundRobinSharderConfig,
            **kwargs,
        )
        super().__init__(**kwargs)
        self._last_shard: int = -1

    def shard_for_row(self, csv_row: Dict[Any, Any]) -> List[int]:
        # pyre-fixme[16]: `RoundRobinSharder` has no attribute `cfg`.
        self._last_shard = (self._last_shard + 1) % self.cfg.num_shards
        return [self._last_shard]

    def shard_rows(self, data_rows: Shardable) -> Iterable[Tuple[str, Any]]:
        # sanity check to avoid empty shards
        shards = super().shard_rows(data_rows)
        assert (
            sum(1 for _shard in shards)
            # pyre-fixme[16]: `RoundRobinSharder` has no attribute `cfg`.
            >= self.cfg.num_shards
        ), "number of rows must be at least the number of shards"
        return shards


class PowerLawSharder(FLDataSharder):
    """
    Splits training data based on power law distribution with order
    alpha where alpha between [0.0, 1.0] to ensure right skewed

    This strategy will shard in a round robin at first to ensure all shard
    will get at least one example. After that, it will shard following the power
    law distribution
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=PowerLawSharderConfig,
            **kwargs,
        )
        super().__init__(**kwargs)
        assert 0.0 < self.cfg.alpha <= 1.0, "alpha must be in the interval (0, 1]"
        self._last_shard: int = -1
        self._weights = np.random.power(self.cfg.alpha, self.cfg.num_shards)
        # normalize to sum to 1.0
        self._weights = self._weights / sum(self._weights)
        self._choices = np.arange(0, self.cfg.num_shards)

    def shard_for_row(self, csv_row: Dict[Any, Any]) -> List[int]:
        # pyre-fixme[16]: `PowerLawSharder` has no attribute `cfg`.
        if self._last_shard < self.cfg.num_shards - 1:
            self._last_shard += 1
            return [self._last_shard]
        else:
            return [np.random.choice(self._choices, p=self._weights)]

    def shard_rows(self, data_rows: Shardable) -> Iterable[Tuple[str, Any]]:
        # sanity check to avoid empty shards
        shards = super().shard_rows(data_rows)
        assert (
            sum(1 for _shard in shards)
            # pyre-fixme[16]: `PowerLawSharder` has no attribute `cfg`.
            >= self.cfg.num_shards
        ), "number of rows must be at least the number of shards"
        return shards


def uncollate_fn(data_rows):
    features, labels = np.array([]), []
    for row in tqdm(data_rows):
        feat = row['features']
        lab = row['labels']

        if features.size == 0:
            features = np.array(feat[None, :, :, :])
        else:
            features = np.vstack((features, np.array(feat[None, :, :, :])))
        labels += [lab]

    return features, np.array(labels)


def collate_fn(batch: Tuple) -> Dict[str, Any]:
    feature, label = batch
    return {"features": feature, "labels": label}


class LDASharder(FLDataSharder):
    """
    LDA Sharder
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=LDASharderConfig,
            **kwargs,
        )
        super().__init__(**kwargs)
        assert 0.0 < self.cfg.alpha <= 1.0, "alpha must be in the interval (0, 1]"

    def shard_for_row(self, csv_row: Dict[Any, Any]) -> List[int]:
        # pyre-fixme[16]: `FLDataSharder` has no attribute `cfg`.
        pass

    def shard_rows(self, data_rows: Shardable) -> Iterable[Tuple[str, Any]]:
        # sanity check to avoid empty shards
        dirichlet_dist = np.random.dirichlet(
            alpha=[self.cfg.alpha]*self.cfg.num_classes, size=self.cfg.num_shards)

        uncollated_dataset = uncollate_fn(data_rows)
        partitions, dirichlet_dist = create_lda_partitions(
            uncollated_dataset, dirichlet_dist=dirichlet_dist,
            num_partitions=self.cfg.num_shards, concentration=self.cfg.alpha, accept_imbalanced=True
        )

        shards = defaultdict(list)

        for i, partition in enumerate(partitions):
            for j in range(len(partition[1])):
                shards[str(i)].append(collate_fn(
                    (torch.Tensor(partition[0][j]), partition[1][j])))

        assert (
            sum(1 for _shard in shards)
            # pyre-fixme[16]: `FLDataSharder` has no attribute `cfg`.
            >= self.cfg.num_shards
        ), "number of rows must be at least the number of shards"
        return shards.items()


@dataclass
class FLDataSharderConfig:
    _target_: str = MISSING
    _recursive_: bool = False


@dataclass
class RandomSharderConfig(FLDataSharderConfig):
    _target_: str = fullclassname(RandomSharder)
    num_shards: int = MISSING


@dataclass
class SequentialSharderConfig(FLDataSharderConfig):
    _target_: str = fullclassname(SequentialSharder)
    examples_per_shard: int = MISSING


@dataclass
class BroadcastSharderConfig(FLDataSharderConfig):
    _target_: str = fullclassname(BroadcastSharder)
    num_shards: int = MISSING


@dataclass
class ColumnSharderConfig(FLDataSharderConfig):
    _target_: str = fullclassname(ColumnSharder)
    sharding_col: str = MISSING


@dataclass
class RoundRobinSharderConfig(FLDataSharderConfig):
    _target_: str = fullclassname(RoundRobinSharder)
    num_shards: int = MISSING


@dataclass
class PowerLawSharderConfig(FLDataSharderConfig):
    _target_: str = fullclassname(PowerLawSharder)
    num_shards: int = MISSING
    alpha: float = MISSING


@dataclass
class LDASharderConfig(FLDataSharderConfig):
    _target_: str = fullclassname(LDASharder)
    alpha: float = MISSING
    num_shards: int = MISSING
    num_classes: int = MISSING
