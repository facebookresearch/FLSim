#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains the functions for clipping clients' updates in
an FL simulation.
"""

from typing import Optional

import torch
from flsim.utils.fl.common import FLModelParamUtils
from torch import nn


__EPS__ = 1e-10


class UserUpdateClipper:
    def __init__(self, precision: Optional[torch.dtype] = None):
        self._cached_model_diff = None
        self.precision = precision

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

    def clip(self, model_diff: nn.Module, max_norm: float) -> None:
        """
        Clips user update (stored in ``model_diff``) by computing clip factor
        and using it to rescale each user's update (operation is in-place).

        This method clips the parameters of the user update. This operation
        is in-place (modifies ``model_diff`` in this method)
        """
        max_norm = float(max_norm)

        per_user_update_norm = self._calc_norm(model_diff.parameters())
        clip_factor = self._calc_clip_factor(max_norm, per_user_update_norm)

        with torch.no_grad():
            for parameter in model_diff.parameters():
                parameter.copy_(parameter * clip_factor)

    def _calc_clip_factor(self, max_norm: float, per_user_norm: float):
        """
        Calculates the clip factor that will be used to clip the user updatas
        """
        if max_norm < 0 or per_user_norm < 0:
            raise ValueError("Error: max_norm and per_user_norm must be both positive.")
        clip_factor = max_norm / (per_user_norm + __EPS__)
        clip_factor = min(clip_factor, 1.0)
        return clip_factor

    def _calc_norm(self, params):
        """
        Calculates the l-2 norm of the user updates
        """
        norms = [param.norm(2) for param in params]
        norm = torch.tensor(norms).norm(2)
        return norm
