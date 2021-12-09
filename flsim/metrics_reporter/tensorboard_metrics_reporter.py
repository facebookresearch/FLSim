#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import copy
from typing import Any, Dict, List, Optional, Tuple

from flsim.common.timeline import Timeline
from flsim.interfaces.batch_metrics import IFLBatchMetrics
from flsim.interfaces.metrics_reporter import (
    Channel,
    IFLMetricsReporter,
    Metric,
    TrainingStage,
)
from torch.utils.tensorboard import SummaryWriter


class FLMetricsReporter(IFLMetricsReporter, abc.ABC):
    """
    This is a MetricsReporter with Tensorboard support.
    """

    def __init__(self, channels: List[Channel], log_dir: Optional[str] = None):
        self.channels = channels
        self.log_dir = log_dir
        if Channel.TENSORBOARD in channels:
            self.set_summary_writer(log_dir)
        if Channel.STDOUT in channels:
            self.print = print
        self.losses = []
        self.num_examples_list = []
        self.predictions_list = []
        self.targets_list = []
        self.model_inputs_list = []
        self.latest_scores: Dict[str, Any] = {}
        self.best_eval_metrics = None

    def set_summary_writer(self, log_dir: Optional[str]):
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_batch_metrics(self, metrics: IFLBatchMetrics) -> None:
        self.losses.append(metrics.loss.item())
        self.num_examples_list.append(metrics.num_examples)
        self.predictions_list.append(metrics.predictions)
        self.targets_list.append(metrics.targets)
        self.model_inputs_list.append(metrics.model_inputs)

    def aggregate(self, one_user_metrics):
        pass

    def report_metrics(
        self,
        reset: bool,
        stage: TrainingStage,
        extra_metrics: Optional[List[Metric]] = None,
        **kwargs,
    ) -> Tuple[Any, bool]:
        metrics = self._report_metrics(
            reset=reset, stage=stage, extra_metrics=extra_metrics, **kwargs
        )
        if stage != TrainingStage.EVAL:
            return (metrics, False)

        if self.best_eval_metrics is None or self.compare_metrics(
            metrics, self.best_eval_metrics
        ):
            self.best_eval_metrics = copy.deepcopy(metrics)
            return (metrics, True)
        else:
            return (metrics, False)

    def _report_metrics(
        self,
        reset: bool,
        stage: TrainingStage,
        extra_metrics: Optional[List[Metric]] = None,
        **kwargs,
    ) -> Any:
        timeline: Timeline = kwargs.get("timeline", Timeline(global_round=1))
        # handle legacy case when epoch was provided
        epoch = kwargs.get("epoch", 0)
        if epoch > 0 and timeline.global_round == 1:
            timeline = Timeline(epoch=epoch, round=1)
        eval_metrics = None

        training_stage_in_str = TrainingStage(stage).name.title()
        if len(self.losses) > 0:
            mean_loss = sum(self.losses) / len(self.losses)

            if Channel.STDOUT in self.channels:
                self.print(f"{timeline}, Loss/{training_stage_in_str}: {mean_loss}")
            if Channel.TENSORBOARD in self.channels:
                self.writer.add_scalar(
                    f"Loss/{training_stage_in_str}",
                    mean_loss,
                    timeline.global_round_num(),
                )

            scores = self.compute_scores()
            self.latest_scores = scores

            for score_name, score in scores.items():
                if Channel.STDOUT in self.channels:
                    self.print(
                        f"{timeline}, {score_name}/{training_stage_in_str}: {score}"
                    )
                if Channel.TENSORBOARD in self.channels:
                    self.writer.add_scalar(
                        f"{score_name}/{training_stage_in_str}",
                        score,
                        timeline.global_round_num(),
                    )
            eval_metrics = self.create_eval_metrics(
                scores, mean_loss, timeline=timeline, stage=stage
            )
        # handle misc reporting values
        metrics = extra_metrics or []
        for metric in metrics:
            value = Metric.to_dict(metric.value) if metric.is_compund else metric.value
            if Channel.STDOUT in self.channels:
                self.print(
                    f"{timeline}, {metric.name}/{training_stage_in_str}: {value}"
                )
            if Channel.TENSORBOARD in self.channels:
                self.writer.add_scalars(
                    f"{metric.name}/{training_stage_in_str}",
                    value,
                    timeline.global_round_num(),
                ) if metric.is_compund else self.writer.add_scalar(
                    f"{metric.name}/{training_stage_in_str}",
                    value,
                    timeline.global_round_num(),
                )

        if reset:
            self.reset()

        return eval_metrics

    def reset(self):
        self.losses = []
        self.num_examples_list = []
        self.predictions_list = []
        self.targets_list = []
        self.model_inputs_list = []

    def get_latest_scores(self) -> Dict[str, Any]:
        return self.latest_scores

    @abc.abstractmethod
    def compare_metrics(self, eval_metrics, best_metrics) -> bool:
        """One should provide concrete implementation of how to compare
        eval_metrics and best_metrics.
        Return True if eval_metrics is better than best_metrics
        """
        pass

    @abc.abstractmethod
    def compute_scores(self) -> Dict[str, Any]:
        """One should override this method to specify how to compute scores
        (e.g. accuracy) of the model based on metrics.
        Return dictionary where key is name of the scores and value is
        score.
        """
        pass

    @abc.abstractmethod
    def create_eval_metrics(
        self, scores: Dict[str, Any], total_loss: float, **kwargs
    ) -> Any:
        """One should provide concrete implementation of how to construct
        object that represents evaluation metrics based on scores and total
        loss. Most of the case, one would just pick one of the scores or
        total loss as the evaluation metrics to pick the better model, but
        this interface also allows s/he to make evaluation metrics more
        complex and use it in conjunction with compare_metrics() function
        to determine which metrics is the better one.
        """
        pass
