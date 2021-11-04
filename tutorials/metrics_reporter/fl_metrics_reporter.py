#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

"""In this file, we show an example of an FLSim metrics reporter.
"""
from typing import Any, Dict, List, Optional

import torch
from flsim.common.timeline import Timeline
from flsim.interfaces.metrics_reporter import Channel, TrainingStage
from flsim.metrics_reporter.tensorboard_metrics_reporter import FLMetricsReporter
from flsim.utils.fl.stats import (
    AverageType,
)
from flsim.utils.fl.target_metric import TargetMetricTracker, TargetMetricDirection


class MetricsReporter(FLMetricsReporter):
    ACCURACY = "Accuracy"
    ROUND_TO_TARGET = "round_to_target"

    def __init__(
        self,
        channels: List[Channel],
        target_eval: float = 0.0,
        window_size: int = 5,
        average_type: str = "sma",
        log_dir: Optional[str] = None,
    ):
        super().__init__(channels, log_dir)
        self.set_summary_writer(log_dir=log_dir)
        self._round_to_target = float(1e10)
        self._stats = TargetMetricTracker(
            target_value=target_eval,
            window_size=window_size,
            average_type=AverageType.from_str(average_type),
            direction=TargetMetricDirection.MAX,
        )

    def compare_metrics(self, eval_metrics, best_metrics):
        print(f"Current eval accuracy: {eval_metrics}%, Best so far: {best_metrics}%")
        if best_metrics is None:
            return True

        current_accuracy = eval_metrics.get(self.ACCURACY, float("-inf"))
        best_accuracy = best_metrics.get(self.ACCURACY, float("-inf"))
        return current_accuracy > best_accuracy

    def compute_scores(self) -> Dict[str, Any]:
        # compute accuracy
        correct = torch.Tensor([0])
        for i in range(len(self.predictions_list)):
            all_preds = self.predictions_list[i]
            pred = all_preds.data.max(1, keepdim=True)[1]

            assert pred.device == self.targets_list[i].device, (
                f"Pred and targets moved to different devices: "
                f"pred >> {pred.device} vs. targets >> {self.targets_list[i].device}"
            )
            if i == 0:
                correct = correct.to(pred.device)

            correct += pred.eq(self.targets_list[i].data.view_as(pred)).sum()

        # total number of data
        total = sum(len(batch_targets) for batch_targets in self.targets_list)

        accuracy = 100.0 * correct.item() / total
        return {self.ACCURACY: accuracy, self.ROUND_TO_TARGET: self._round_to_target}

    def create_eval_metrics(
        self, scores: Dict[str, Any], total_loss: float, **kwargs
    ) -> Any:
        timeline: Timeline = kwargs.get("timeline", Timeline(global_round=1))
        stage: TrainingStage = kwargs.get("stage", None)
        accuracy = scores[self.ACCURACY]
        if (
            stage
            in [
                TrainingStage.EVAL,
                TrainingStage.TEST,
            ]
            and self._stats.update_and_check_target(accuracy)
        ):
            self._round_to_target = min(
                timeline.global_round_num(), self._round_to_target
            )
        return {
            self.ACCURACY: accuracy,
            self.ROUND_TO_TARGET: self._round_to_target,
        }
