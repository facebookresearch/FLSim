#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import deque
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd


class AverageType(Enum):
    SMA = "sma"
    EMA = "ema"

    @staticmethod
    def from_str(name: str):
        name_upper = name.upper()
        names = [e.name for e in AverageType]
        assert name_upper in names, "Unknown average type:" + name
        return AverageType[name_upper]


class QuantilesTracker:
    """
    Tracks the mean, standard deviation, and quantiles of a random variable

    Note:
        We store the samples in memory so be careful with large number of
        samples
    """

    def __init__(self):
        self._samples: List = []

    def update(self, val: float) -> None:
        self._samples.append(val)

    def quantile(self, p) -> float:
        if len(self._samples) == 0:
            return float("Inf")
        return np.quantile(self._samples, p)

    @property
    def median_val(self) -> float:
        return self.quantile(0.5)

    @property
    def lower_quartile_val(self) -> float:
        return self.quantile(0.25)

    @property
    def upper_quartile_val(self) -> float:
        return self.quantile(0.75)


class ModelSequenceNumberTracker:
    r"""
    Class to keep track of the current global sequence number and statistics.

    Keeps track of "model_seqnum," which indicates the checkpoint version of global model.

    Example:
        For example, if global model_seqnum is 10 and the local model_seqnum
        (i.e. model_seqnum of a particular device state) is 7, then we record
        the diff of model_seqnum, which is 3 in this case.
    """

    def __init__(self):
        self._current_model_seqnum = 0
        self.seqnum_diff_stats = RandomVariableStatsTracker()

    def increment(self) -> int:
        r"""
        Increments the global model_seqnum
        """
        self._current_model_seqnum += 1
        return self._current_model_seqnum

    def get_staleness_and_update_stats(self, client_seqnum: int) -> int:
        r"""
        Compares the current global model_seqnum with model_seqnum of
        a particular client
        """
        # Seqnum_diff will be 0 for sequential training.
        seqnum_diff = self._current_model_seqnum - client_seqnum
        self.seqnum_diff_stats.update(seqnum_diff)
        return seqnum_diff

    def print_stats(self) -> None:
        print(f"ModelSeqNum: {self.current_seqnum}")
        print(f"\tSeqnumDiff, {self.seqnum_diff_stats.as_str()}")

    def mean(self) -> float:
        r"""
        Returns the mean difference between the global seq_num and local seq_num
        """
        return self.seqnum_diff_stats.mean()

    def standard_deviation(self) -> float:
        r"""
        Returns the SD difference between the global seq_num and local seq_num
        """
        return self.seqnum_diff_stats.standard_deviation()

    @property
    def current_seqnum(self) -> int:
        r"""
        Current global model seq num
        """
        return self._current_model_seqnum


class RandomVariableStatsTracker:
    """Keeps track of mean, variance, min and max values of a random variable"""

    def __init__(self, tracks_quantiles: bool = False):
        self._sum: float = 0
        self._sum_squares: float = 0
        self._min_val: float = float("Inf")
        self._max_val: float = -float("Inf")
        self._num_samples: int = 0
        self._quant_tracker: Optional[QuantilesTracker] = (
            QuantilesTracker() if tracks_quantiles else None
        )

    def update(self, val: float) -> None:
        self._sum += val
        self._sum_squares += val * val
        self._min_val = min(self._min_val, val)
        self._max_val = max(self._max_val, val)
        self._num_samples += 1
        if self._quant_tracker is not None:
            self._quant_tracker.update(val)

    def mean(self) -> float:
        if not self._num_samples:
            return float("Inf")
        return self._sum / self._num_samples

    def standard_deviation(self) -> float:
        if not self._num_samples:
            return float("Inf")
        mean_sum_squares = self._sum_squares / self._num_samples
        mean_squared = self.mean() * self.mean()
        variance = mean_sum_squares - mean_squared
        return math.sqrt(variance + 1e-6)

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def min_val(self) -> float:
        return self._min_val

    @property
    def max_val(self) -> float:
        return self._max_val

    @property
    def mean_val(self) -> float:
        return self.mean()

    @property
    def standard_deviation_val(self) -> float:
        return self.standard_deviation()

    @property
    def median_val(self) -> float:
        if self._quant_tracker is None:
            return float("inf")
        return self._quant_tracker.median_val

    @property
    def lower_quartile_val(self) -> float:
        if self._quant_tracker is None:
            return float("inf")
        return self._quant_tracker.lower_quartile_val

    @property
    def upper_quartile_val(self) -> float:
        if self._quant_tracker is None:
            return float("inf")
        return self._quant_tracker.upper_quartile_val

    def as_str(self) -> str:
        return (
            f"Mean:{self.mean():.3f}, "
            f"SD:{self.standard_deviation():.3f}, "
            f"Min:{self.min_val:.3f}, Max:{self.max_val:.3f}"
        )


class RandomVariableStatsTrackerMA(RandomVariableStatsTracker):
    """
    Tracks the simple moving or exponential moving mean and
    standard deviation of a random variable

    Note:
        We store the window_size number of samples in memory so please
        keep window_size to be a reasonable number
    """

    def __init__(self, window_size: int, mode=AverageType.SMA, decay_factor=0.5):
        super().__init__()
        self._samples = deque(maxlen=window_size)
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.mode = mode

    def update(self, val: float) -> None:
        super().update(val)
        self._samples.append(val)

    def mean(self) -> float:
        if not self._num_samples:
            raise ValueError("There are no samples in tracker.")
        return (
            np.mean(self._samples)
            if self.mode == AverageType.SMA
            else pd.Series(self._samples).ewm(alpha=self.decay_factor).mean().iloc[-1]
        )

    def standard_deviation(self) -> float:
        if not self._num_samples:
            raise ValueError("There are no samples in tracker.")
        return (
            np.std(self._samples)
            if self.mode == AverageType.SMA
            else pd.Series(self._samples).ewm(alpha=self.decay_factor).std().iloc[-1]
        )
