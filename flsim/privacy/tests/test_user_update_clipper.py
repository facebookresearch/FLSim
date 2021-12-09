#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from flsim.common.pytest_helper import (
    assertAlmostEqual,
    assertTrue,
    assertEqual,
    assertFalse,
)
from flsim.privacy.user_update_clip import UserUpdateClipper
from flsim.tests import utils


@pytest.fixture(scope="class")
def prepare_user_update_clipper_test(request):
    request.cls.user_update_clipper = UserUpdateClipper()

    def _init_user_model(param_value):
        user_model = utils.TwoFC()
        user_model.fill_all(param_value)  # fill all parameters with a number
        return user_model

    request.cls.previous_user_model = _init_user_model(param_value=6.0)
    request.cls.updated_user_model = _init_user_model(param_value=7.0)
    request.cls.original_model_diff = request.cls.user_update_clipper.calc_model_diff(
        request.cls.updated_user_model, request.cls.previous_user_model
    )
    request.cls.original_model_diff_params = [
        p for p in request.cls.original_model_diff.parameters() if p.requires_grad
    ]


@pytest.mark.usefixtures("prepare_user_update_clipper_test")
class TestUserUpdateClipper:
    def _init_clipped_model_diff(self, max_norm):
        clipped_model_diff = utils.TwoFC()
        clipped_model_diff.load_state_dict(self.original_model_diff.state_dict())

        self.user_update_clipper.clip(clipped_model_diff, max_norm)
        return clipped_model_diff

    def test_calc_clip_factor(self):
        """
        Tests that the clip factor for user updates is calculated correctly.
        """
        clip_factor = self.user_update_clipper._calc_clip_factor(
            max_norm=5, per_user_norm=10
        )
        assertAlmostEqual(clip_factor, 0.5, places=3)

        clip_factor = self.user_update_clipper._calc_clip_factor(
            max_norm=1, per_user_norm=1
        )
        assertAlmostEqual(clip_factor, 1, places=3)

        clip_factor = self.user_update_clipper._calc_clip_factor(
            max_norm=2, per_user_norm=0.8
        )
        assertAlmostEqual(clip_factor, 1, places=3)

    def test_calc_user_update_norm(self):
        """
        Tests that the user update l-2 norms are calculated correctly.
        """
        model = utils.TwoFC()
        model.fill_all(2.0)
        model_params = model.parameters()
        # norm = sqrt(21*2^2)=sqrt(84)=9.16515138991168
        norm = self.user_update_clipper._calc_norm(model_params)
        assertTrue(torch.allclose(norm, torch.tensor(9.16515138991168), rtol=1e-06))

        model.fill_all(1.0)
        model_params = model.parameters()
        # norm = sqrt(21*1^2)=sqrt(21)=4.58257569495584
        norm = self.user_update_clipper._calc_norm(model_params)
        assertTrue(torch.allclose(norm, torch.tensor(4.58257569495584), rtol=1e-06))

    def test_clipped_updates_are_smaller(self):
        """
        Tests that user updates are clipped and their value is smaller than
        the original updates
        """

        # assert the parameters of model_diff are all = (7 - 6 = 1)
        for p in self.original_model_diff_params:
            assertTrue(torch.allclose(p.float(), torch.tensor(1.0)))

        clipped_model_diff = self._init_clipped_model_diff(0.0003)
        clipped_model_diff_params = [
            p for p in clipped_model_diff.parameters() if p.requires_grad
        ]
        for original, clipped in zip(
            self.original_model_diff_params, clipped_model_diff_params
        ):
            assertTrue(torch.all(original.gt(clipped)))

    def test_clipped_user_updates_non_zero(self):
        """
        Tests that user updates are not zero by clipping
        """
        clipped_model_diff = self._init_clipped_model_diff(0.0003)
        clipped_model_diff_params = [
            p for p in clipped_model_diff.parameters() if p.requires_grad
        ]
        for clipped in clipped_model_diff_params:
            allzeros = torch.zeros_like(clipped)
            assertFalse(torch.allclose(clipped, allzeros))

    def test_clipping_to_high_value_does_not_clip(self):
        """
        Tests that when clip value is set too high, user
        updates are not clipped
        """
        clipped_model_diff = self._init_clipped_model_diff(9999)
        mismatched = utils.verify_models_equivalent_after_training(
            self.original_model_diff, clipped_model_diff
        )
        assertEqual(mismatched, "")
