#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from flsim.fb.metrics_reporters.fb_tensorboard_metrics_reporter import (
    FBFLMetricsReporter,
)
from flsim.interfaces.metrics_reporter import Channel, TrainingStage
from flsim.utils.simple_batch_metrics import FLBatchMetrics
from scipy.interpolate import interp1d  # @manual
from scipy.optimize import brentq  # @manual
from sklearn.metrics import auc, roc_curve


class SpeakerIdFLEvalBatchMetrics(FLBatchMetrics):
    def __init__(
        self,
        loss: torch.Tensor,
        num_examples: int,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model_inputs: Any,
        loss_only_eval: bool = False,
        eval_time_user_predictions: Optional[torch.Tensor] = None,
        eval_time_user_labels: Optional[torch.Tensor] = None,
    ) -> None:
        self._loss = loss
        self._num_examples = num_examples
        self._predictions = predictions
        self._targets = targets
        self._model_inputs = model_inputs
        self.loss_only_eval = loss_only_eval
        self.eval_time_user_predictions = eval_time_user_predictions
        self.eval_time_user_labels = eval_time_user_labels

    @property
    def loss(self) -> torch.Tensor:
        return self._loss

    @property
    def num_examples(self) -> int:
        return self._num_examples

    @property
    def predictions(self) -> torch.Tensor:
        return self._predictions

    @property
    def targets(self) -> torch.Tensor:
        return self._targets

    @property
    def model_inputs(self) -> Any:
        return self._model_inputs


class SpeakerIdMetricsReporter(FBFLMetricsReporter):
    def __init__(self, channels: List[Channel]):
        super().__init__(channels)
        self.client_eval_losses = []
        self.client_eval_preds = []
        self.client_eval_labels = []
        self.client_eval_userid_preds = []
        self.client_eval_userid_labels = []
        self.client_num_samples = []
        self.global_prediction_scores = []
        self.global_true_labels = []
        self.global_eval_losses = []
        self.EER = "eer"
        self.AUC = "auc"
        self.ACCURACY = "accuracy"

    # pyre-fixme[14]: `add_batch_metrics` overrides method defined in
    #  `FLMetricsReporter` inconsistently.
    def add_batch_metrics(
        self, metrics: Union[FLBatchMetrics, SpeakerIdFLEvalBatchMetrics]
    ) -> None:
        """
        This method is called in multiple contexts:
        1. On a client during its training loop, to accumulate batch metrics produced by fl_model.fl_forward()
        2. On a client during its evaluation routine (which may be on the same data that it trained on), to
           accumulate batch metrics produced by fl_model.get_eval_metrics()
        3. On central server during its evaluation routine, to accumulate batch metrics produced by
           fl_model.get_eval_metrics()

        Train and Eval batch metrics are distinguished by their type FLBatchMetrics vs. SpeakerIdFLEvalBatchMetrics
        Evaluation on train data (happens on clients) vs. eval data (happens on central server) is distinguished by
        loss_only_eval flag in SpeakerIdFLEvalBatchMetrics.
        """
        if not isinstance(metrics, SpeakerIdFLEvalBatchMetrics):
            # metric produced during training
            self.losses.append(metrics.loss.item())
            self.num_examples_list.append(metrics.num_examples)
            self.predictions_list.append(metrics.predictions.detach().clone())
            self.targets_list.append(metrics.targets.detach().clone())
            # self.model_inputs_list.append(metrics.model_inputs)
        else:
            if metrics.loss_only_eval:
                # validation metrics evaluated by clients on their training data
                self.client_eval_losses.append(metrics.loss.item())
                self.client_eval_preds.append(metrics.predictions.detach().clone())
                self.client_eval_labels.append(metrics.targets.detach().clone())
                self.client_num_samples.append(metrics.num_examples)
            else:
                # validation metrics evaluated by server on dev/test set
                self.global_prediction_scores.extend(metrics.predictions)
                self.global_true_labels.extend(metrics.targets)
                self.global_eval_losses.append(metrics.loss.item())
                self.client_eval_userid_preds.append(metrics.eval_time_user_predictions)
                self.client_eval_userid_labels.append(metrics.eval_time_user_labels)

    def aggregate(self, one_user_metrics):
        """
        Combine metrics from one user into a global metrics. This is not used in SyncTrainer, so this is a no-op
        for now.
        """
        pass

    def _report_mean_loss(
        self, epoch: int, mean_loss: torch.Tensor, training_stage_in_str: str
    ) -> None:
        if Channel.STDOUT in self.channels:
            print(f"Epoch: {epoch}, Loss/{training_stage_in_str}: {mean_loss}")
        if Channel.TENSORBOARD in self.channels:
            self.writer.add_scalar(f"Loss/{training_stage_in_str}", mean_loss, epoch)

    def _compute_eval_scores(
        self, y_true: Iterable[int], y_preds: Iterable[float]
    ) -> Dict[str, float]:
        fpr, tpr, thresholds = roc_curve(y_true, y_preds, pos_label=1)
        roc_auc = auc(fpr, tpr)
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        return {self.EER: eer, self.AUC: roc_auc}

    def _compute_train_scores(
        self, targets_list: List[torch.Tensor], predictions_list: List[torch.Tensor]
    ) -> Dict[str, float]:
        """
        targets_list: List of items of dim (batch_size,)
        predictions_list: List of items of dim (batch_size, num_classes)
        """
        # compute accuracy
        correct = torch.Tensor([0])
        for i in range(len(predictions_list)):
            all_preds = predictions_list[i]
            pred = all_preds.data.max(1, keepdim=True)[1]
            assert pred.device == targets_list[i].device, (
                f"Pred and targets moved to different devices: "
                f"pred >> {pred.device} vs. targets >> {targets_list[i].device}"
            )
            if i == 0:
                correct = correct.to(pred.device)
            # pyre-fixme[16]: `Tensor` has no attribute `view_as`.
            correct += pred.eq(targets_list[i].data.view_as(pred)).sum()
        # total number of data
        total = sum(len(batch_targets) for batch_targets in targets_list)
        accuracy = 100.0 * correct.item() / total
        return {self.ACCURACY: accuracy}

    def _report_results(
        self,
        scores: Dict[str, Any],
        training_stage_in_str: str,
        overridden_epoch: int,
        **kwargs,
    ) -> None:
        for score_name, score in scores.items():
            if Channel.STDOUT in self.channels:
                print(
                    f"Epoch: {overridden_epoch}, {score_name}/{training_stage_in_str}: {score}"
                )
            if Channel.TENSORBOARD in self.channels:
                self.writer.add_scalar(
                    f"{score_name}/{training_stage_in_str}", score, overridden_epoch
                )

        for priv_type in ["sample_privacy", "user_privacy"]:
            privacy = kwargs.get(priv_type, {})
            if Channel.STDOUT in self.channels:
                print(
                    f"Epoch: {overridden_epoch}, {priv_type}/{training_stage_in_str}: {privacy}"
                )
            if Channel.TENSORBOARD in self.channels:
                self.writer.add_scalars(
                    f"{priv_type}/{training_stage_in_str}", privacy, overridden_epoch
                )

    # pyre-fixme[14,15]: `report_metrics` overrides method defined in
    #  `FLMetricsReporter` inconsistently.
    def report_metrics(self, reset: bool, stage: TrainingStage, **kwargs) -> float:
        """
        This method is called in multiple contexts by SyncTrainer
        1. During training, after all clients have finished running their local training loops and have populated
           self.losses. Traditional multiclass classification accuracy can be evaluated.
        2. During evaluation on server, after all clients have run their local eval routines and have populated
           self.client_eval_losses. This setting is denoted by stage=TrainingStage.AGGREGATION. Same evaluation as
           training may be run.
        3. During evaluation on server, after server has run evaluation on central eval data set and has populated
           self.global_prediction_scores, self.global_true_labels and self.global_eval_losses. This setting is denoted
           by stage=TrainingState.EVAL. Note that traditional multiclass classification accuracy cannot be computed
           here. Instead, eer and auc are reported
        """
        timeline = kwargs.get("timeline", None)
        if timeline is None:
            epoch = kwargs.get("epoch", -1)
        else:
            epoch = timeline.epoch

        training_stage_in_str = TrainingStage(stage).name.title()
        if stage == TrainingStage.AGGREGATION:
            mean_loss = sum(self.client_eval_losses) / len(self.client_eval_losses)
            # pyre-fixme[6]: Expected `Tensor` for 2nd param but got `float`.
            self._report_mean_loss(epoch, mean_loss, training_stage_in_str)
            scores = self._compute_train_scores(
                self.client_eval_labels, self.client_eval_preds
            )
            self._report_results(scores, training_stage_in_str, epoch, **kwargs)
        elif stage == TrainingStage.EVAL:
            mean_loss = sum(self.global_eval_losses) / len(self.global_eval_losses)
            scores = self._compute_eval_scores(
                self.global_true_labels, self.global_prediction_scores
            )
            acc_scores = self._compute_train_scores(
                self.client_eval_userid_labels, self.client_eval_userid_preds
            )
            all_scores = {**scores, **acc_scores}
            # pyre-fixme[6]: Expected `Tensor` for 2nd param but got `float`.
            self._report_mean_loss(epoch, mean_loss, training_stage_in_str)
            self._report_results(all_scores, training_stage_in_str, epoch, **kwargs)
        else:
            mean_loss = sum(self.losses) / len(self.losses)
            # pyre-fixme[6]: Expected `Tensor` for 2nd param but got `float`.
            self._report_mean_loss(epoch, mean_loss, training_stage_in_str)
            scores = self.compute_scores()
            self.latest_scores = scores
            self._report_results(scores, training_stage_in_str, epoch, **kwargs)

        if reset:
            self.reset()
        # pyre-fixme[7]: Expected `float` but got `Dict[str, typing.Any]`.
        return scores
        # return self.create_eval_metrics(scores, mean_loss)

    def reset(self):
        """
        Clean up all aggregations so far.
        """
        self.losses = []
        self.num_examples_list = []
        self.predictions_list = []
        self.targets_list = []
        self.model_inputs_list = []
        self.client_eval_losses = []
        self.client_eval_preds = []
        self.client_eval_labels = []
        self.client_num_samples = []
        self.global_prediction_scores = []
        self.global_true_labels = []
        self.global_eval_losses = []

    def get_latest_scores(self) -> Dict[str, Any]:
        return self.latest_scores

    def compare_metrics(
        self,
        eval_metrics: Union[FLBatchMetrics, SpeakerIdFLEvalBatchMetrics],
        best_metrics: Union[FLBatchMetrics, SpeakerIdFLEvalBatchMetrics],
    ) -> bool:
        """One should provide concrete implementation of how to compare
        eval_metrics and best_metrics.
        Return True if current evaluation metrics is better than the best
        metrics observed so far.
        """
        if best_metrics is None:
            return True
        print(f"Current eval accuracy: {eval_metrics}%, Best so far: {best_metrics}%")
        # pyre-fixme[16]: `FLBatchMetrics` has no attribute `__getitem__`.
        if eval_metrics[self.EER] == best_metrics[self.EER]:
            return eval_metrics[self.AUC] > best_metrics[self.AUC]
        else:
            return eval_metrics[self.EER] < best_metrics[self.EER]

    def compute_scores(self) -> Dict[str, Any]:
        """One should override this method to specify how to compute scores
        (e.g. accuracy) of the model based on metrics.
        Return dictionary where key is name of the scores and value is
        score.

        For speaker ID, metrics are different during train and test. This
        abstract method must be implemented though, so it defaults to
        computing accuracy on the train set (aggregated across all clients).
        """
        return self._compute_train_scores(self.targets_list, self.predictions_list)

    def create_eval_metrics(
        self, scores: Dict[str, float], total_loss: float, **kwargs
    ) -> float:
        """One should provide concrete implementation of how to construct
        object that represents evaluation metrics based on scores and total
        loss. Most of the case, one would just pick one of the scores or
        total loss as the evaluation metrics to pick the better model, but
        this interface also allows s/he to make evaluation metrics more
        complex and use it in conjunction with compare_metrics() function
        to determine which metrics is the better one.
        """
        if self.EER in scores:
            return scores[self.EER]
        elif self.AUC in scores:
            return scores[self.AUC]
        elif self.ACCURACY in scores:
            return scores[self.ACCURACY]
        return total_loss
