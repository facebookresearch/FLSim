#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
This file contains the functions for clipping clients' updates in
an FL simulation.
"""

from __future__ import annotations

import abc
import math

import os
from typing import Optional, Tuple

import torch
from flsim.privacy.common import (
    calc_clip_factor,
    calc_norm,
    ClippingSetting,
    ClippingType,
    PrivacySetting,
)
from flsim.utils.fl.common import FLModelParamUtils
from torch import nn


class IUserClipper(abc.ABC):
    def __init__(
        self,
        max_norm: float,
        precision: Optional[torch.dtype] = None,
    ):
        self._cached_model_diff = None
        self.precision = precision
        self.unclipped_num = 0
        self.max_norm = max_norm

    @classmethod
    def create_clipper(cls, privacy_setting: PrivacySetting) -> IUserClipper:
        if privacy_setting.clipping.clipping_type == ClippingType.FLAT:
            return UserUpdateClipper(max_norm=privacy_setting.clipping.clipping_value)
        elif privacy_setting.clipping.clipping_type == ClippingType.ADAPTIVE:
            return AdaptiveClipper(
                clip_setting=privacy_setting.clipping,
                seed=privacy_setting.noise_seed,
            )
        else:
            raise ValueError("Invalid clipping type.")

    def calc_model_diff(self, new_model: nn.Module, prev_model: nn.Module) -> nn.Module:
        """
        Calculates the difference between the updated model and the old model
        """
        if self._cached_model_diff is None:  # for memory efficiency purposes
            self._cached_model_diff = FLModelParamUtils.clone(new_model, self.precision)
        FLModelParamUtils.linear_comb_models(
            new_model, 1, prev_model, -1, self._cached_model_diff
        )
        return self._cached_model_diff

    def _calc_norm_and_clip_factor(self, model: nn.Module) -> Tuple[float, float]:
        per_user_update_norm = calc_norm(model.parameters())
        clip_factor = calc_clip_factor(self.max_norm, per_user_update_norm)
        self.unclipped_num += 1 if math.isclose(clip_factor, 1.0) else 0
        return per_user_update_norm, clip_factor

    @abc.abstractmethod
    def reset_clipper_stats(self):
        pass

    @abc.abstractmethod
    def update_clipper_stats(self):
        pass

    @abc.abstractmethod
    def clip(self) -> float:
        pass


class UserUpdateClipper(IUserClipper):
    def __init__(self, max_norm: float, precision: Optional[torch.dtype] = None):
        super().__init__(max_norm=max_norm)

    def reset_clipper_stats(self):
        self.unclipped_num = 0

    def clip(self, model_diff: nn.Module) -> float:
        """
        Clips user update (stored in ``model_diff``) by computing clip factor
        and using it to rescale each user's update (operation is in-place).

        This method clips the parameters of the user update. This operation
        is in-place (modifies ``model_diff`` in this method)
        """
        per_user_update_norm, clip_factor = self._calc_norm_and_clip_factor(model_diff)
        with torch.no_grad():
            for parameter in model_diff.parameters():
                parameter.copy_(parameter * clip_factor)
        return per_user_update_norm

    def update_clipper_stats(self):
        pass


class AdaptiveClipper(IUserClipper):
    """
    Federated User Update Clipper that implements
    adaptive clipping strategy https://arxiv.org/pdf/1905.03871.pdf
    """

    def __init__(
        self,
        clip_setting: ClippingSetting,
        seed: Optional[int] = None,
    ):
        super().__init__(max_norm=clip_setting.clipping_value)
        self.unclipped_quantile = clip_setting.unclipped_quantile
        self.clipbound_learning_rate = clip_setting.clipbound_learning_rate
        self.unclipped_num_std = clip_setting.unclipped_num_std
        self.max_clipbound = clip_setting.max_clipbound
        self.min_clipbound = clip_setting.min_clipbound

        self.generator = torch.Generator()
        noise_seed = (
            int.from_bytes(os.urandom(8), byteorder="big", signed=True)
            if seed is None
            else seed
        )
        self.generator.manual_seed(noise_seed)
        self.users_per_round = 0
        # the noisy number of users that did not get clipped
        self.noisy_unclipped_num: float = 0.0
        # the true value of users that did not get clipped
        self.unclipped_num: torch.Tensor = torch.Tensor([0])

    def reset_clipper_stats(self):
        self.users_per_round = 0
        self.unclipped_num = 0

    def clip(self, model_diff: nn.Module) -> float:
        """
        Calculates the l-2 norm of the user updates
        """
        per_user_update_norm, clip_factor = self._calc_norm_and_clip_factor(model_diff)

        self.users_per_round += 1
        unclipped_num_noise = torch.normal(
            mean=0,
            std=self.unclipped_num_std,
            generator=self.generator,
            size=(1,),
        )

        self.noisy_unclipped_num = float(self.unclipped_num)
        # pyre-fixme[8]: Attribute has type `float`; used as `Tensor`.
        self.noisy_unclipped_num += unclipped_num_noise

        with torch.no_grad():
            for parameter in model_diff.parameters():
                parameter.copy_(parameter * clip_factor)

        return per_user_update_norm

    def update_clipper_stats(self):
        """
        Update clipping bound based on unclipped fraction
        """

        self.update_max_grad_norm()

    def update_max_grad_norm(self):
        """
        Update the max grad norm using the following formula
        new_clip_norm = current_clip_norm * e^(-lr * (unclipped_fraction - target_quantile))

        See section 2.1 in https://arxiv.org/pdf/1905.03871.pdf
        """
        unclipped_frac = self.noisy_unclipped_num / self.users_per_round
        self.max_norm *= torch.exp(
            -self.clipbound_learning_rate * (unclipped_frac - self.unclipped_quantile)
        ).item()

        if self.max_norm > self.max_clipbound:
            self.max_norm = self.max_clipbound
        elif self.max_norm < self.min_clipbound:
            self.max_norm = self.min_clipbound
