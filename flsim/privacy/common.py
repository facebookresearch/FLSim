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


from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional

import numpy as np


@dataclass
class PrivacySetting:
    """
    Contains setting related to Differential Privacy
    """

    alphas: List[float] = field(
        default_factory=lambda: np.arange(1.1, 100, 0.1).tolist()
    )  # Renyi privacy alpha range
    noise_multiplier: float = 0.0  # Normalized Noise Variance
    clipping_value: float = float("inf")
    target_delta: float = 1e-5  # Maximum delta for (epsilon, delta) privacy
    noise_seed: Optional[int] = None  # [debug] Seed of the noise generation function
    secure_rng: bool = False


class PrivacyBudget(NamedTuple):
    """
    Encapsulates a privacy budget as (epsilon, delta)
    """

    epsilon: float = float("inf")
    alpha: float = float(-1)
    delta: float = float("inf")

    def __str__(self):
        return f"eps = {self.epsilon}, delta = {self.delta}, alpha = {self.alpha}"
