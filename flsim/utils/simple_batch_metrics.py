#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
