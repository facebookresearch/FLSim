#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from flsim.utils.fl.stats import (
    AverageType,
)
from flsim.utils.fl.target_metric import TargetMetricTracker, TargetMetricDirection
from libfb.py import testutil


class TargetMetricTestTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_target_metric_optimize_for_max(self) -> None:
        """
        Test if target tracker returns true when the sliding window returns
        true when optimizing for max value (e.g accuracy)
        For example, target = 90, window_size = 3
        eval accuracies = [80, 81, 89, 90, 95]
        [80, 81, 89] -> false
        [81, 89, 90] -> false
        the window [89, 90, 95] -> true
        """
        metrics = [80, 81, 89, 90, 95]
        target_value = 90
        for average_type in [AverageType.SMA, AverageType.EMA]:
            target_tracker = TargetMetricTracker(
                target_value=target_value,
                window_size=3,
                average_type=average_type,
                direction=TargetMetricDirection.MAX,
            )
            for metric in metrics[:-1]:
                self.assertFalse(target_tracker.update_and_check_target(metric))
                self.assertLess(target_tracker.mean, target_value)

            self.assertTrue(target_tracker.update_and_check_target(metrics[-1]))
            self.assertGreater(target_tracker.mean, target_value)

    def test_target_metric_optimize_for_min(self) -> None:
        """
        Test if target tracker returns true when the sliding window returns
        true when optimizing for min value (e.g loss)
        For example, target = 0.1, window_size = 3
        eval loss = [0.5, 0.4, 0.15, 0.04, 0.1
        [0.5, 0.4, 0.15] -> false
        [0.4, 0.15, 0.04] -> false
        the window [0.15, 0.04, 0.1] -> true
        """
        metrics = [0.5, 0.4, 0.15, 0.04, 0.1]
        target_value = 0.1
        for average_type in [AverageType.SMA, AverageType.EMA]:
            target_tracker = TargetMetricTracker(
                target_value=target_value,
                window_size=3,
                average_type=average_type,
                direction=TargetMetricDirection.MIN,
            )
            for metric in metrics[:-1]:
                self.assertFalse(target_tracker.update_and_check_target(metric))
                self.assertGreater(target_tracker.mean, target_value)

            self.assertTrue(target_tracker.update_and_check_target(metrics[-1]))
            self.assertLess(target_tracker.mean, target_value)
