#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import pytest
from flsim.common.pytest_helper import (
    assertEqual,
    assertAlmostEqual,
)
from flsim.utils.fl.stats import (
    ModelSequenceNumberTracker,
    RandomVariableStatsTracker,
    RandomVariableStatsTrackerMA,
    AverageType,
)


class TestStatsTracker:
    def test_stats_tracker(self) -> None:
        """Test that we can accurately keep track of stats"""
        np.random.seed(100)
        # check mean and standard_deviation using a normal random
        stats_tracker = RandomVariableStatsTracker()
        for _ in range(1000):
            # test with normal random
            rv = np.random.normal(loc=0, scale=10)
            stats_tracker.update(rv)
        assertAlmostEqual(stats_tracker.mean(), 0.0, delta=0.5)
        assertAlmostEqual(stats_tracker.standard_deviation(), 10.0, delta=1.0)

        stats_tracker2 = RandomVariableStatsTracker()
        for i in range(1000):
            stats_tracker2.update(i - 10)
        assertEqual(stats_tracker2.min_val, -10)
        assertEqual(stats_tracker2.max_val, 989)

    def test_sequence_tracker(self) -> None:
        seqnum_tracker = ModelSequenceNumberTracker()
        num_global_step = 5
        for _ in range(num_global_step):
            seqnum_tracker.increment()

        assertEqual(num_global_step, seqnum_tracker.current_seqnum)

        num_clients = 100
        client_seqnums = np.random.randint(10000, size=num_clients)
        staleness_weights = []
        for client_seqnum in client_seqnums:
            staleness = seqnum_tracker.get_staleness_and_update_stats(
                client_seqnum=client_seqnum
            )
            assertEqual(num_global_step - client_seqnum, staleness)
            staleness_weights.append(staleness)

        expected_mean = np.mean(staleness_weights)
        assertAlmostEqual(expected_mean, seqnum_tracker.mean(), delta=1e-6)

        expected_sd = np.std(staleness_weights)
        assertAlmostEqual(expected_sd, seqnum_tracker.standard_deviation(), delta=1e-6)

    @pytest.mark.parametrize(
        "max_val",
        [10, 100, 1000],
    )
    @pytest.mark.parametrize(
        "window_size",
        [5, 50],
    )
    @pytest.mark.parametrize(
        "average_type",
        [AverageType.SMA, AverageType.EMA],
    )
    def test_moving_average(self, max_val, window_size, average_type) -> None:
        decay_factor = 0.5
        stats_tracker = RandomVariableStatsTrackerMA(
            window_size=window_size, mode=average_type
        )

        values = np.arange(1, max_val, 1)
        for i in values:
            stats_tracker.update(i)

        values = np.array(values[-window_size:])

        if average_type == AverageType.SMA:
            expected_mean = values.mean()
            expected_std = values.std()
        else:
            v = pd.Series(values)
            expected_mean = v.ewm(alpha=decay_factor).mean().iloc[-1]
            expected_std = v.ewm(alpha=decay_factor).std().iloc[-1]

        assertEqual(stats_tracker.mean(), expected_mean)
        assertEqual(stats_tracker.standard_deviation(), expected_std)

    def test_quantiles_tracker(self):
        stats_tracker = RandomVariableStatsTracker(tracks_quantiles=True)
        values = []
        for i in range(100):
            stats_tracker.update(i)
            values.append(i)

        assertEqual(stats_tracker.median_val, np.quantile(values, 0.5))
        assertEqual(stats_tracker.lower_quartile_val, np.quantile(values, 0.25))
        assertEqual(stats_tracker.upper_quartile_val, np.quantile(values, 0.75))
