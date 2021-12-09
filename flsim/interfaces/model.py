#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Any

import torch.nn as nn
from flsim.interfaces.batch_metrics import IFLBatchMetrics


class IFLModel(abc.ABC):
    @abc.abstractmethod
    def fl_forward(self, batch: Any) -> IFLBatchMetrics:
        pass

    @abc.abstractmethod
    def fl_create_training_batch(self, **kwargs) -> Any:
        pass

    @abc.abstractmethod
    def fl_get_module(self) -> nn.Module:
        pass

    @abc.abstractmethod
    def fl_cuda(self) -> None:
        pass

    @abc.abstractmethod
    def get_eval_metrics(self, batch: Any) -> IFLBatchMetrics:
        pass

    @abc.abstractmethod
    def get_num_examples(self, batch: Any) -> int:
        pass
