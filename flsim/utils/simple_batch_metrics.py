#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from flsim.interfaces.batch_metrics import IFLBatchMetrics


class FLBatchMetrics(IFLBatchMetrics):
    def __init__(
        self,
        *,
        loss: torch.Tensor,
        num_examples: int,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model_inputs: Any,
    ) -> None:
        self._loss = loss
        self._num_examples = num_examples
        self._predictions = predictions
        self._targets = targets
        self._model_inputs = model_inputs

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
