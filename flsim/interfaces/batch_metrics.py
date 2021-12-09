#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Any, List

import torch


class IFLBatchMetrics(abc.ABC):
    """Each forward run of FL Model (i.e. concrete implementation of IFLModel
    in PyML) will return an IFLBatchMetrics object. This is an encapsulation
    and abstraction around several useful/common metrics such as loss,
    num_examples, predictions, and targets so that IFLMetricsReporter can
    aggregate metrics from any kind of model as long as the model returns
    IFLBatchMetrics.
    """

    @property
    @abc.abstractmethod
    def loss(self) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def num_examples(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def predictions(self) -> List[Any]:
        pass

    @property
    @abc.abstractmethod
    def targets(self) -> List[Any]:
        pass

    @property
    @abc.abstractmethod
    def model_inputs(self) -> Any:
        pass
