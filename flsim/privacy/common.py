#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional

import numpy as np
import torch


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


class PrivateTrainingMetricsUtils:
    @classmethod
    def signal_to_noise_ratio(cls, aggregated_model, noise):
        """
        Following the definition in https://arxiv.org/pdf/2110.05679.pdf
        signal to noise ratio = norm(g) / norm(noise)
        """
        g = cls.l2_norm(aggregated_model.parameters())
        z = cls.l2_norm(noise)
        return (g / z).item()

    @classmethod
    def l2_norm(cls, module: List[torch.Tensor]):
        return torch.tensor([n.norm(2) for n in module]).norm(2)
