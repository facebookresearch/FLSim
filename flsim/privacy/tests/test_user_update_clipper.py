#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from copy import deepcopy

import numpy as np

import torch
from flsim.common.pytest_helper import (
    assertAlmostEqual,
    assertEmpty,
    assertEqual,
    assertFalse,
    assertGreater,
    assertTrue,
)
from flsim.privacy.common import calc_clip_factor, calc_norm, ClippingSetting
from flsim.privacy.user_update_clip import AdaptiveClipper, UserUpdateClipper
from flsim.utils import test_utils as utils


class TestUserUpdateClipper:
    def _init_user_model(self, param_value):
        user_model = utils.TwoFC()
        user_model.fill_all(param_value)
        return user_model

    def test_calc_clip_factor(self) -> None:
        """
        Tests that the clip factor for user updates is calculated correctly.
        """
        clip_factor = calc_clip_factor(clipping_value=5, norm=10)
        assertAlmostEqual(clip_factor, 0.5, places=3)

        clip_factor = calc_clip_factor(clipping_value=1, norm=1)
        assertAlmostEqual(clip_factor, 1, places=3)

        clip_factor = calc_clip_factor(clipping_value=2, norm=0.8)
        assertAlmostEqual(clip_factor, 1, places=3)

    def test_calc_user_update_norm(self) -> None:
        """
        Tests that the user update l-2 norms are calculated correctly.
        """
        model = utils.TwoFC()
        model.fill_all(2.0)
        model_params = model.parameters()
        # norm = sqrt(21*2^2)=sqrt(84)=9.16515138991168
        norm = calc_norm(model_params)
        assertAlmostEqual(norm, 9.16515138991168, delta=1e-06)

        model.fill_all(1.0)
        model_params = model.parameters()
        # norm = sqrt(21*1^2)=sqrt(21)=4.58257569495584
        norm = calc_norm(model_params)
        assertAlmostEqual(norm, 4.58257569495584, delta=1e-06)

    def test_clipped_updates_are_smaller(self) -> None:
        """
        Tests that user updates are clipped and their value is smaller than
        the original updates
        """

        # assert the parameters of model_diff are all = (7 - 6 = 1)
        max_norm = 0.0003
        previous_model = self._init_user_model(6)
        current_model = self._init_user_model(7)
        clipper = UserUpdateClipper(max_norm=max_norm)

        model_diff = clipper.calc_model_diff(current_model, previous_model)
        orignal = deepcopy(model_diff)
        assertGreater(calc_norm(model_diff.parameters()), max_norm)

        clipper.clip(model_diff)
        assertAlmostEqual(calc_norm(model_diff.parameters()), max_norm, delta=1e-6)
        for o, c in zip(orignal.parameters(), model_diff.parameters()):
            assertTrue(torch.all(o.gt(c)))

    def test_clipped_user_updates_non_zero(self) -> None:
        """
        Tests that user updates are not zero by clipping
        """
        max_norm = 0.0003
        previous_model = self._init_user_model(6)
        current_model = self._init_user_model(7)
        clipper = UserUpdateClipper(max_norm=max_norm)
        model_diff = clipper.calc_model_diff(current_model, previous_model)
        clipper.clip(model_diff)

        clipped_model_diff_params = [
            p for p in model_diff.parameters() if p.requires_grad
        ]
        for clipped in clipped_model_diff_params:
            allzeros = torch.zeros_like(clipped)
            assertFalse(torch.allclose(clipped, allzeros))

    def test_clipping_to_high_value_does_not_clip(self) -> None:
        """
        Tests that when clip value is set too high, user
        updates are not clipped
        """
        max_norm = 1000
        previous_model = self._init_user_model(6)
        current_model = self._init_user_model(7)
        clipper = UserUpdateClipper(max_norm=max_norm)
        model_diff = clipper.calc_model_diff(current_model, previous_model)
        original = deepcopy(model_diff)

        clipper.clip(model_diff)

        mismatched = utils.verify_models_equivalent_after_training(
            original,
            model_diff,
        )
        assertEmpty(mismatched, msg=mismatched)


class TestAdaptiveClipper:
    def test_median_norm_clip(self):
        """
        Figure 2 in https://arxiv.org/pdf/1905.03871.pdf
        The clip norm should accruate track the true quantile after 100 rounds
        """
        torch.manual_seed(0)
        users_per_round = 10
        num_rounds = 200
        quantile = 0.5

        clipper = AdaptiveClipper(
            ClippingSetting(
                unclipped_quantile=quantile,
                clipbound_learning_rate=0.1,
                max_clipbound=100,
                min_clipbound=0.05,
                clipping_value=0.1,
                unclipped_num_std=users_per_round / 20,
            ),
            seed=0,
        )
        models = [utils.create_model_with_value(i + 1) for i in range(users_per_round)]
        norms = [calc_norm(m.parameters()) for m in models]

        max_clip_norm = 0
        for _ in range(num_rounds):
            clipper.reset_clipper_stats()
            for model in models:
                clipper.clip(deepcopy(model))
            clipper.update_clipper_stats()
            max_clip_norm = clipper.max_norm

        assertAlmostEqual(max_clip_norm, np.quantile(norms, quantile), delta=3)

    def test_norm_bounds(self):
        """
        Test max norm doesn't violate max bounds
        """
        torch.manual_seed(0)
        users_per_round = 10
        num_rounds = 10
        quantile = 0.5
        max_clipbound = 0.1

        clipper = AdaptiveClipper(
            ClippingSetting(
                unclipped_quantile=quantile,
                clipbound_learning_rate=0.1,
                max_clipbound=max_clipbound,
                min_clipbound=0.05,
                clipping_value=1,
                unclipped_num_std=users_per_round / 20,
            ),
            seed=0,
        )
        # max bounds
        models = [utils.create_model_with_value(i + 1) for i in range(users_per_round)]
        max_clip_norm = 0
        for _ in range(num_rounds):
            clipper.reset_clipper_stats()
            for model in models:
                clipper.clip(deepcopy(model))
            clipper.update_clipper_stats()
            max_clip_norm = clipper.max_norm

        assertEqual(max_clip_norm, max_clipbound)
