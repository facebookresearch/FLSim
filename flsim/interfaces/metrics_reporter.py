#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union, Tuple

from flsim.interfaces.batch_metrics import IFLBatchMetrics
from torch import Tensor


class Metric:
    r"""
    Wraps a metric.

    A reportable metric is simply the name of the metric,
    and the value of the metric, in its simplest form.
    The value could also be a dict of other metrics, which
    in this case the metric is a set of other metrics, and
    the `is_compound` attribute is set.
    """

    def __init__(self, name: str, value: Union[float, List["Metric"]]):
        self.name = name
        self.value = value

    @property
    def is_compund(self):
        return isinstance(self.value, list)

    def __str__(self):
        return f"{self.name}: {self.value}"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> List["Metric"]:

        metrics = []

        def process_dict(d, metrics):
            for k, v in d.items():
                assert isinstance(k, str), f"{k} must be a string"
                if isinstance(v, dict):
                    sub_metric = []
                    process_dict(v, sub_metric)
                    metrics.append(Metric(k, sub_metric, True))
                else:
                    assert isinstance(
                        v, (int, float, Tensor)
                    ), f"{v} is not of types int, float, or torch.Tensor"
                    metrics.append(Metric(k, float(v)))

        process_dict(d, metrics)
        return metrics

    @classmethod
    def from_args(cls, **kwargs):
        r"""
        Simple util to generate Metrics from kwargs.

        The usage is simple, metrics need to be passed as named arguments
        to the function. The class will throw if the metrics are not
        any of the valid types: int, float, tensor of size 1, or a
        dictionary of such types with string keys. This latter case is considerd
        a metric with sub metrics.

        Example:
        metric = Metrics.from_args(a=1, b=2.0, c={d=1, e=2})
        will result in:

        [Metric(a, 1.0), Metric(b, 2.0), Metric(c, [Metric(d, 1.0), Metric(e, 2.0)])]
        """
        return cls.from_dict(kwargs)

    @classmethod
    def to_dict(cls, metrics):

        d = {}

        def process_list(metrics, d):
            for metric in metrics:
                assert isinstance(metric, Metric)
                value = metric.value
                if metric.is_compund:
                    value = {}
                    process_list(metric.value, value)
                d[metric.name] = value

        process_list(metrics, d)
        return d


class Channel(Enum):
    """Enum that indicates to which "channel" we'd want to log/print our
    metrics to. For example, if on chooses `Channel.TENSORBOARD`, a
    metrics reporter would display the information on the TensorBoard.
    """

    TENSORBOARD = auto()
    STDOUT = auto()


class TrainingStage(Enum):
    """Enum that indicates at which stage training is, which will be used when
    reporting metrics to certain channels.
    """

    TRAINING = auto()
    AGGREGATION = auto()
    TEST = auto()
    EVAL = auto()
    PER_CLIENT_EVAL = auto()


class IFLMetricsReporter(abc.ABC):
    """Interface that all PyML FLMetricsReporter should implement. Each user
    will have 1 reporter throughout one's entire training. At the beginning of
    user’s training (i.e. at the start of epoch for a user), the user starts by
    clearing up any aggregation left by calling reset(). After the batch, each
    user collects one’s own metrics by calling add_batch_metrics() method. When
    all the batches are completed (i.e. at the end of all local epochs for a user),
    a global MetricsReporter (i.e. the MetricsReporter responsible of the whole
    training aggregates all the data by calling aggregate() method, which gets
    a MetricsReporter of an user who just completed one’s own epoch. Then, after
    all users’ local epochs are completed, the global MetricsReporter completes
    its global aggreation and report its metrics to given channels for that global
    epoch.
    Note: 1 global epoch consists of several rounds. In each round, we train
    a subset of users and each user goes through a number of local epochs, where
    each local epoch consists of multiple batches.
    """

    @abc.abstractmethod
    def add_batch_metrics(self, metrics: IFLBatchMetrics) -> None:
        """Take in output of training for a batch (of each user).
        Aggregates metrics (e.g. accuracy, loss, predictions, etc) from batch
        into state.
        """
        pass

    @abc.abstractmethod
    def aggregate(self, one_user_metrics: "IFLMetricsReporter"):
        """Combine metrics from one user into a global metrics."""
        pass

    @abc.abstractmethod
    def report_metrics(
        self,
        reset: bool,
        stage: TrainingStage,
        extra_metrics: Optional[List[Metric]] = None,
        **kwargs,
    ) -> Tuple[Any, bool]:
        """Report metrics to certain channels such as stdout, file, TensorBoard,
        etc. Also, one may want to reset metrics if needed after reporting.
        Return value: A tuple with two elements:
            1. A metrics object
            2. bool: Were the best eval metrics updated?
               Current eval metrics are compared to the best eval metrics. If
               current eval metrics are better than best eval metrics, true is returned.
               Comparing eval metrics to best eval metrics is common in
               many ML training algorithms, e.g, early stopping.
        """
        pass

    # TODO: is this needed? Do we ever call this externally?
    @abc.abstractmethod
    def reset(self):
        """Clean up all aggregations so far."""
        pass
