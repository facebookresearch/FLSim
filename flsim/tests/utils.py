#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, Dict, List, NamedTuple, Optional, Union, Tuple
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn
from flsim.common.logger import Logger
from flsim.data.data_provider import IFLUserData
from flsim.interfaces.batch_metrics import IFLBatchMetrics
from flsim.interfaces.metrics_reporter import (
    Channel,
    IFLMetricsReporter,
    Metric,
    TrainingStage,
)
from flsim.interfaces.model import IFLModel
from flsim.metrics_reporter.tensorboard_metrics_reporter import FLMetricsReporter
from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.simple_batch_metrics import FLBatchMetrics
from torch.utils.tensorboard import SummaryWriter


class DatasetFromList:
    """
    Simple dataset from  a list of tuples.
    Each item in the outer list will be a batch and each
    batch itself is a tuple of two lists, the raw_batch
    and the batch.
    """

    def __init__(self, list_dataset):
        self.ds = list_dataset

    def __len__(self):
        _, batch = self.ds[0]
        return len(self.ds) * len(batch)

    def __iter__(self):
        return iter(self.ds)


class DummyUserData(IFLUserData):
    def __init__(self, data, model, from_data_provider=False):
        self.data = data
        self._num_examples: int = 0
        self._num_batches: int = 0
        self.model = model
        self.from_data_provider = from_data_provider
        for _, batch in self.data:
            self._num_examples += (
                batch["label"].shape[0] if self.from_data_provider else batch.shape[0]
            )
            self._num_batches += 1

    def __iter__(self):
        for _, batch in self.data:
            yield self.model.fl_create_training_batch(batch=batch)

    def num_batches(self):
        return self._num_batches

    def num_examples(self):
        return self._num_examples


class Quadratic1D(nn.Module):
    """
        a toy optimization example:
            min f(x) = 100 x^2 - 1

    minima is x=0.0, x is initialized at 1.0.
    """

    def __init__(self):
        super(Quadratic1D, self).__init__()
        self.x = nn.Parameter(torch.ones(1))
        self.y = torch.tensor([1.0])

    def forward(self):
        return 100 * torch.square(self.x) - self.y


class MockQuadratic1DFL(IFLModel):
    """
    a dummy IFL wrapper for Quadratic1D
    """

    def __init__(self, model):
        self.model = model

    def fl_forward(self, data=None):
        loss = self.model()
        return FLBatchMetrics(
            loss=loss, num_examples=1, predictions=None, targets=None, model_inputs=None
        )

    def fl_create_training_batch(self):
        pass

    def fl_cuda(self):
        pass

    def fl_get_module(self):
        return self.model

    def get_eval_metrics(self):
        pass

    def get_num_examples(self):
        pass


class TwoFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def fill_all(self, value):
        def fill(layer):
            if type(layer) == nn.Linear:
                layer.bias.data.fill_(value)
                layer.weight.data.fill_(value)

        self.apply(fill)


class Metrics(IFLBatchMetrics):
    def __init__(self, num_examples, loss):
        self._num_examples = num_examples
        self._loss = loss

    @property
    def loss(self) -> torch.Tensor:
        return self._loss

    @property
    def num_examples(self) -> int:
        return self._num_examples

    @property
    def predictions(self):
        pass

    @property
    def targets(self):
        pass

    @property
    def model_inputs(self):
        pass


class FakeMetricReporter(IFLMetricsReporter):
    def add_batch_metrics(self, metrics: IFLBatchMetrics) -> None:
        pass

    def aggregate(self, one_user_metrics: IFLMetricsReporter):
        pass

    def report_metrics(
        self,
        reset: bool,
        stage: TrainingStage,
        extra_metrics: Optional[List[Metric]] = None,
        **kwargs,
    ) -> Tuple[Any, bool]:
        return (None, False)

    def reset(self):
        pass

    def compare_metrics(self, eval_metrics, best_metrics):
        pass


class SimpleMetricReporter(FakeMetricReporter):
    def __init__(self):
        self.batch_metrics = []

    def add_batch_metrics(self, metrics: IFLBatchMetrics) -> None:
        self.batch_metrics.append(metrics)


class SampleNet(IFLModel):
    def __init__(self, model):
        self.sample_nn = model
        self._num_examples = None
        self._out = None

    def fl_forward(self, batch):
        y = self.sample_nn(batch)
        return Metrics(len(batch), y.mean())

    def fl_create_training_batch(self, batch=None, **kwargs):
        return batch

    def fl_get_module(self):
        return self.sample_nn

    def fl_cuda(self):
        pass

    def get_eval_metrics(self, batch):
        pass

    def get_num_examples(self, batch):
        return len(batch)


class SampleNetHive(SampleNet):
    def __init__(self, value=None):
        self.sample_nn = TwoFC()
        if value is not None:
            self.sample_nn.fill_all(value)
        self._num_examples = None
        self._out = None

    def fl_forward(self, batch):
        x, y = batch["user_n"], batch["label"]
        x = x.flatten().repeat(2).float()
        y = y.flatten().float()
        preds = self.sample_nn(x)
        loss = nn.BCEWithLogitsLoss()(preds, y)
        return Metrics(self.get_num_examples(batch), loss=loss)

    def get_num_examples(self, batch):
        return len(batch["label"])


def verify_models_equivalent_after_training(
    model1: Union[nn.Module, IFLModel],
    model2: Union[nn.Module, IFLModel],
    model_init: Optional[Union[nn.Module, IFLModel]] = None,
    rel_epsilon: Optional[float] = None,
    abs_epsilon: Optional[float] = None,
) -> str:
    """This function accepts either nn.Module or IFLModel and checks that:
    a) Model training did something:
        model1 & model2 are different from model_init
    b) model1 and model2 have same parameters

    Return value: str. "" if both a) and b) are satisfied.
        else, error message with the SAD (sum of absolute difference between
        mismatched model parameters)
    """
    model1 = model1.fl_get_module() if isinstance(model1, IFLModel) else model1
    model2 = model2.fl_get_module() if isinstance(model2, IFLModel) else model2
    if model_init is not None:
        model_init = (
            model_init.fl_get_module()
            if isinstance(model_init, IFLModel)
            else model_init
        )
        # Ensure that training actually did something to model 1.
        if (
            FLModelParamUtils.get_mismatched_param(
                [model1, model_init], rel_epsilon=rel_epsilon, abs_epsilon=abs_epsilon
            )
            == ""
        ):
            return "Model 1 training did nothing"

        # Ensure that training actually did something to model 2.
        if (
            FLModelParamUtils.get_mismatched_param(
                [model2, model_init], rel_epsilon=rel_epsilon, abs_epsilon=abs_epsilon
            )
            == ""
        ):
            return "Model 2 training did nothing"

    # check models identical under both configs
    mismatched_param = FLModelParamUtils.get_mismatched_param(
        [model1, model2], rel_epsilon=rel_epsilon, abs_epsilon=abs_epsilon
    )

    if mismatched_param != "":
        summed_absolute_difference = (
            (
                model1.state_dict()[mismatched_param]
                - model2.state_dict()[mismatched_param]
            )
            .abs()
            .sum()
        )
        return (
            f"Model 1, Model 2 mismatch. Param: {mismatched_param},"
            f" Summed Absolute Difference={summed_absolute_difference}"
        )
    else:
        return ""


def model_parameters_equal_to_value(model, value):
    if isinstance(model, IFLModel):
        model = model.fl_get_module()
    for n, p in model.named_parameters():
        if not torch.allclose(p.float(), torch.tensor(value)):
            summed_absolute_difference = (p - torch.tensor(value)).abs().sum()
            return (
                n
                + f"{p} did not match with {value}: Summed Absolute Difference={summed_absolute_difference}"
            )
    return ""


def check_inherit_logging_level(obj: Any, level: int) -> bool:
    Logger.set_logging_level(level)
    return obj.logger.getEffectiveLevel() == level


class MockRecord(NamedTuple):
    tag: str = ""
    value: Union[float, Dict[str, float]] = 0.0
    global_step: int = 0
    walltime: float = 0


class MetricsReporterWithMockedChannels(FLMetricsReporter):
    """Simulates an FL reporter with STDOUT and Tensorboard channels
    STDOUT and Tensorboard channels are mocked
    """

    def __init__(self):
        super().__init__([Channel.STDOUT, Channel.TENSORBOARD])

        self.tensorboard_results: List[MockRecord] = []
        self.stdout_results: List[MockRecord] = []

        def add_scalar(tag, scalar_value, global_step=None, walltime=None):
            self.tensorboard_results.append(
                MockRecord(tag, scalar_value, global_step, walltime)
            )
            return self

        def add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None):
            for tag, value in tag_scalar_dict.items():
                self.tensorboard_results.append(
                    MockRecord(f"{main_tag}/{tag}", value, global_step, walltime)
                )

        def printer(*args):
            self.stdout_results.append(tuple(arg for arg in args))

        SummaryWriter.add_scalar = MagicMock(side_effect=add_scalar)
        SummaryWriter.add_scalars = MagicMock(side_effect=add_scalars)
        self.print = MagicMock(side_effect=printer)

    def compare_metrics(self, eval_metrics, best_metrics) -> bool:
        return True

    def compute_scores(self) -> Dict[str, Any]:
        return {}

    def create_eval_metrics(
        self, scores: Dict[str, Any], total_loss: float, **kwargs
    ) -> Any:
        return None

    def set_summary_writer(self, log_dir: Optional[str]):
        super().set_summary_writer("/tmp/")


class RandomEvalMetricsReporter(IFLMetricsReporter):
    """This metrics reporter is useful for unit testing. It does four things:
    a) When report_metrics(stage=Eval) is called, it produced a random
       eval result
    b) It keeps track of the best eval result produced, and the model that
       produced this result.
    c) When report_metrics(stage=Test) is called, it returns the best eval result
       Why? This is useful in testing the trainer.train() function, which returns
       test eval results.
    d) It can be queried to return the best_model, and the value of
       the best result

    """

    def __init__(self):
        self._best_eval_result: float = -1
        self._best_eval_model: IFLModel = None

    def add_batch_metrics(self, metrics: IFLBatchMetrics) -> None:
        pass

    def aggregate(self, one_user_metrics: "IFLMetricsReporter"):
        pass

    def report_metrics(
        self,
        reset: bool,
        stage: TrainingStage,
        extra_metrics: Optional[List[Metric]] = None,
        **kwargs,
    ) -> Tuple[Any, bool]:
        if stage != TrainingStage.EVAL:
            return self._best_eval_result, False

        assert "model" in kwargs.keys(), f"Did not find model in kwargs: {kwargs}"
        model: IFLModel = kwargs.get("model", None)
        eval_result = np.random.random_sample()

        if eval_result > self._best_eval_result:
            print(
                f"MetricReporter current_eval:{eval_result}, best_eval: {self._best_eval_result}"
            )
            self._best_eval_model = copy.deepcopy(model)
            self._best_eval_result = eval_result
            return (eval_result, True)
        else:
            return (eval_result, False)

    def reset(self):
        pass

    @property
    def best_eval_result(self) -> float:
        return self._best_eval_result

    @property
    def best_eval_model(self) -> IFLModel:
        return self._best_eval_model


def create_model_with_value(value) -> nn.Module:
    model = TwoFC()
    model.fill_all(value)
    return model
