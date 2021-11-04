#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import random
from abc import abstractmethod
from collections import defaultdict, namedtuple
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from torch.utils.data import Dataset


Shardable = TypeVar("Shardable", Iterable[Dict[str, Any]], Dataset)


class ShardingStrategyType:
    RANDOM: str = "random"
    BROADCAST: str = "broadcast"
    COLUMN: str = "column"
    ROUND_ROBIN: str = "round_robin"
    SEQUENTIAL: str = "sequential"
    POWER_LAW: str = "power_law"


class FLDataSharder:
    """This class takes in a file, and partitions it into a list of datasets.
    It supports random partitioning, broadcasting,
    round-robining and sharding by a column
    """

    # Similar to the comment in __init__() below. This is also a temporary hack
    # until we consolidate design for FLConfig. This Config inner class is a
    # thin namedtuple wrapper for FLDataSharderConfig so that we don't have to
    # change the code that relies on calls like `self.config.xxx` for this time
    # and the future.
    Config = namedtuple(
        "Config",
        [
            "sharding_strategy",
            "num_shards",
            "sharding_colindex",
            "sharding_col_name",
            "shard_size_for_sequential",
            "alpha",
        ],
    )

    def __init__(
        self,
        sharding_strategy: str,
        num_shards: Optional[int] = None,
        sharding_colindex: Optional[int] = None,
        sharding_col_name: Optional[str] = None,
        shard_size_for_sequential: Optional[int] = None,
        alpha: Optional[float] = None,
    ):
        """This initializer is a temporary workaround until we decide on Config
        design. Note that in order to avoid any PyText dependency, this
        initializer requires each property unpacked from FLDataSharderConfig,
        which currently has dependency on PyText's ConfigBase.
        """
        self.config = FLDataSharder.Config(
            sharding_strategy,
            num_shards,
            sharding_colindex,
            sharding_col_name,
            shard_size_for_sequential,
            alpha,
        )

    class ShardingStrategy:
        @abstractmethod
        def shard_for_row(self, csv_row: Dict[Any, Any]) -> List[Any]:
            pass

    class RandomSharding(ShardingStrategy):
        """Splits training data randomly. num_shards should be specified.
        Assigns first num_shards rows to different shards to avoid empty shards."""

        def __init__(self, num_shards: int):
            assert num_shards is not None, "num_shards must be provided."
            self._num_shards: int = num_shards
            self._assignments: List[int] = list(range(self._num_shards))
            random.shuffle(self._assignments)

        def shard_for_row(self, csv_row: Dict[Any, Any]) -> List[int]:
            if self._assignments:
                return [self._assignments.pop(0)]
            return [random.randint(0, self._num_shards - 1)]

    class SequentialSharding(ShardingStrategy):
        """Assign first N rows to shard A, the next N rows to shard B, and so
        on. Mostly used for testing purpose (e.g. sanity-check with
        conventional SGD training based on PyTorch Dataset and its batching
        mechanism).
        """

        def __init__(self, examples_per_shard: int):
            assert isinstance(
                examples_per_shard, int
            ), f"shard_size_for_sequential should be an integer for \
             sequential sharding. got {examples_per_shard}"
            self.examples_per_shard = examples_per_shard
            self.filled_in_current_bucket = 0
            self.index = 0

        def shard_for_row(self, csv_row: Dict[Any, Any]) -> List[int]:
            if self.filled_in_current_bucket == self.examples_per_shard:
                self.index += 1
                self.filled_in_current_bucket = 1
            else:
                self.filled_in_current_bucket += 1
            return [self.index]

    class BroadcastSharding(ShardingStrategy):
        """Copy each training datum to all shards. num_shards should be specified"""

        def __init__(self, num_shards: int):
            assert num_shards is not None, "num_shards must be provided."
            self._num_shards: int = num_shards

        def shard_for_row(self, csv_row: Dict[Any, Any]) -> List[int]:
            return list(range(self._num_shards))

    class ColumnSharding(ShardingStrategy):
        """Specify a column name used to shard.
        It should be the last column in the file,
        and sharding_column must be specified.
        """

        def __init__(self, sharding_col: Union[int, str]):
            self.sharding_col: Union[int, str] = sharding_col

        def shard_for_row(self, csv_row: Dict[Any, Any]) -> List[Any]:
            # Pyre needs this manual unwrap of "Optional" for class
            # attribute (see https://fburl.com/wmn9bmac).
            unwrapped_colindex = self.sharding_col
            if isinstance(unwrapped_colindex, int):
                assert unwrapped_colindex < len(
                    csv_row
                ), "Sharding index out of bounds: "
                f"{unwrapped_colindex}"

            shard_idx = csv_row[unwrapped_colindex]

            # shard_idx can be a 0-dim tensor when a Hive column is an integer.
            if isinstance(shard_idx, torch.Tensor):
                shard_idx = shard_idx.item()

            return [shard_idx]

    class RoundRobinSharding(ShardingStrategy):
        """Splits training in a round-robin fashion between all shards"""

        def __init__(self, num_shards: int):
            assert num_shards is not None, "num_shards must be provided."
            self._num_shards: int = num_shards
            self._last_shard: int = -1

        def shard_for_row(self, csv_row: Dict[Any, Any]) -> List[int]:
            self._last_shard = (self._last_shard + 1) % self._num_shards
            return [self._last_shard]

    class PowerLawSharding(ShardingStrategy):
        """
        Splits training data based on power law distribution with order
        alpha where alpha between [0.0, 1.0] to ensure right skewed

        This strategy will shard in a round robin at first to ensure all shard
        will get at least one example. After that, it will shard following the power
        law distribution
        """

        def __init__(self, num_shards: int, alpha: float):
            assert num_shards is not None, "num_shards must be provided."
            assert alpha is not None, "alpha must be provided."
            assert 0.0 < alpha <= 1.0, "alpha must be in the interval (0, 1]"
            self._num_shards: int = num_shards
            self._last_shard: int = -1
            self._weights = np.random.power(alpha, self._num_shards)
            # normalize to sum to 1.0
            self._weights = self._weights / sum(self._weights)
            self._choices = np.arange(0, self._num_shards)

        def shard_for_row(self, csv_row: Dict[Any, Any]) -> List[int]:
            if self._last_shard < self._num_shards - 1:
                self._last_shard += 1
                return [self._last_shard]
            else:
                return [np.random.choice(self._choices, p=self._weights)]

    def shard_rows(self, data_rows: Shardable) -> Iterable[Tuple[str, Any]]:
        """Partition a set of rows into mulitple sets using a sharding strategy.

        Args:
            data_rows: Iterable[Dict[str, Any]]]: iterable over dictionary mapping column
            name to value.
        """
        shards = defaultdict(list)
        sharding_strategy = ShardingStrategyFactory.create(self.config)
        for one_row in data_rows:
            for shard_id in sharding_strategy.shard_for_row(one_row):
                shards[str(shard_id)].append(one_row)

        # sanity check to avoid empty shards
        if self.config.sharding_strategy in (
            ShardingStrategyType.RANDOM,
            ShardingStrategyType.ROUND_ROBIN,
            ShardingStrategyType.POWER_LAW,
        ):
            assert (
                len(shards.keys()) >= self.config.num_shards
            ), "number of rows must be at least the number of shards"
        return shards.items()


class ShardingStrategyFactory:
    @staticmethod
    def create(config: FLDataSharder.Config):
        if config.sharding_strategy == ShardingStrategyType.RANDOM:
            return FLDataSharder.RandomSharding(config.num_shards)
        elif config.sharding_strategy == ShardingStrategyType.BROADCAST:
            return FLDataSharder.BroadcastSharding(config.num_shards)
        elif config.sharding_strategy == ShardingStrategyType.COLUMN:
            if config.sharding_colindex is not None:
                param = config.sharding_colindex
            elif config.sharding_col_name is not None:
                param = config.sharding_col_name
            else:
                raise ValueError("Must provide a value to shard by column.")
            return FLDataSharder.ColumnSharding(sharding_col=param)
        elif config.sharding_strategy == ShardingStrategyType.ROUND_ROBIN:
            return FLDataSharder.RoundRobinSharding(config.num_shards)
        elif config.sharding_strategy == ShardingStrategyType.SEQUENTIAL:
            return FLDataSharder.SequentialSharding(config.shard_size_for_sequential)
        elif config.sharding_strategy == ShardingStrategyType.POWER_LAW:
            return FLDataSharder.PowerLawSharding(config.num_shards, alpha=config.alpha)
        else:
            assert f"Invalid sharding strategy: {config.sharding_strategy}."
