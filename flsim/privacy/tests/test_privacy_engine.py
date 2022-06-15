#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from flsim.common.pytest_helper import (
    assertEqual,
    assertFalse,
    assertNotEqual,
    assertRaises,
    assertTrue,
)
from flsim.privacy.common import PrivacySetting
from flsim.privacy.privacy_engine import (
    GaussianPrivacyEngine,
    PrivacyEngineNotAttachedException,
)
from flsim.utils import test_utils as utils
from flsim.utils.fl.common import FLModelParamUtils
from opacus.accountants.analysis import rdp as privacy_analysis


class TestGaussianPrivacyEngine:
    def _init_privacy_engine(
        self,
        alphas=[1 + x / 10.0 for x in range(1, 100)],
        noise_multiplier=1.0,
        target_delta=1e-5,
        users_per_round=10,
        num_total_users=10,
        global_model_parameter_val=5.0,
        noise_seed=0,
    ):
        privacy_setting = PrivacySetting(
            alphas=alphas,
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
            noise_seed=noise_seed,
        )
        privacy_engine = GaussianPrivacyEngine(
            privacy_setting=privacy_setting,
            users_per_round=users_per_round,
            num_total_users=num_total_users,
        )
        global_model = utils.TwoFC()  # This model has 15 weights and 6 biases
        global_model.fill_all(global_model_parameter_val)
        privacy_engine.attach(global_model)
        return privacy_engine

    def _calc_eps(self, sample_rate, noise_multiplier, steps, alphas, delta):
        rdp = privacy_analysis.compute_rdp(
            q=sample_rate, noise_multiplier=noise_multiplier, steps=steps, orders=alphas
        )
        eps, _ = privacy_analysis.get_privacy_spent(orders=alphas, rdp=rdp, delta=delta)
        return eps

    def test_privacy_analysis_alpha_in_alphas(self):
        """
        Tests that the optimal alphas of DP analysis is in the range
        of the original alphas.
        """
        privacy_engine = self._init_privacy_engine()
        privacy_budget = privacy_engine.get_privacy_spent()
        assertTrue(privacy_budget.alpha in privacy_engine.alphas)

    def test_privacy_analysis_epsilon_reasonable(self):
        """
        Tests that the epsilon is greater than 1 in normal settings.
        Also, when we do not add any noise, the privacy loss should
        be infinite.
        """

        privacy_engine = self._init_privacy_engine()
        privacy_budget = privacy_engine.get_privacy_spent()
        assertTrue(privacy_budget.epsilon > 0)

        privacy_engine.noise_multiplier = 0
        privacy_budget = privacy_engine.get_privacy_spent()
        assertTrue(privacy_budget.epsilon == float("inf"))

    def test_privacy_analysis_epsilon(self):
        """
        Tests that the epsilon calculations are correct
        """

        alphas = [1 + x / 10.0 for x in range(1, 100)]
        noise_multiplier = 1.5
        target_delta = 1e-5

        num_users = 1000
        num_users_per_round = 50
        steps = num_users // num_users_per_round
        user_sampling_rate = num_users_per_round / num_users

        privacy_engine = self._init_privacy_engine(
            alphas=alphas,
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
            num_total_users=num_users,
            users_per_round=num_users_per_round,
        )
        model_diff = utils.TwoFC()
        for _ in range(steps):  # adding noise will increase the steps
            privacy_engine.add_noise(model_diff, 1.0)

        privacy_budget = privacy_engine.get_privacy_spent()
        eps = self._calc_eps(
            user_sampling_rate, noise_multiplier, steps, alphas, target_delta
        )
        assertEqual(privacy_budget.epsilon, eps)

    def test_noise_added(self):
        """
        Tests that noise is successfully added to a model update, by
        checking that the model update after noise addition is different
        from the original model update.
        """
        model_diff = utils.TwoFC()  # model update
        model_diff.fill_all(1.0)

        model_diff_before_noise = FLModelParamUtils.clone(model_diff)

        privacy_engine = self._init_privacy_engine()
        privacy_engine.add_noise(model_diff, sensitivity=0.5)

        mismatched = utils.verify_models_equivalent_after_training(
            model_diff_before_noise, model_diff
        )
        assertNotEqual(mismatched, "")

    def test_deterministic_noise_addition(self):
        """
        Tests when the noise seed is set to the same value, we get
        the same (i.e. deterministic) noise-added model updates. It
        also tests when the seed is set to different values, we will
        get different noise-added model updates.
        """
        model_diff = utils.TwoFC()  # model update
        model_diff.fill_all(1.0)

        model_diff_another_seed = FLModelParamUtils.clone(model_diff)
        model_diff_same_seed = FLModelParamUtils.clone(model_diff)

        privacy_engine = self._init_privacy_engine(noise_seed=1003)
        privacy_engine.add_noise(model_diff, sensitivity=0.5)

        privacy_engine = self._init_privacy_engine(noise_seed=2000)
        privacy_engine.add_noise(model_diff_another_seed, sensitivity=0.5)

        mismatched = utils.verify_models_equivalent_after_training(
            model_diff, model_diff_another_seed
        )
        assertNotEqual(mismatched, "")

        privacy_engine = self._init_privacy_engine(noise_seed=1003)
        privacy_engine.add_noise(model_diff_same_seed, sensitivity=0.5)

        mismatched = utils.verify_models_equivalent_after_training(
            model_diff, model_diff_same_seed
        )
        assertEqual(mismatched, "")

    def test_not_attached_validator(self):
        """
        Tests that the Privacy Engine throws a not attach
        exception if it is not properly attached.
        """
        model_diff = utils.TwoFC()  # model update
        model_diff.fill_all(1.0)

        privacy_setting = PrivacySetting(
            alphas=[1 + x / 10.0 for x in range(1, 100)],
            noise_multiplier=1.0,
            target_delta=1e-6,
        )
        privacy_engine = GaussianPrivacyEngine(
            privacy_setting=privacy_setting, users_per_round=1, num_total_users=1
        )
        sensitivity = 0.5
        with assertRaises(PrivacyEngineNotAttachedException):
            privacy_engine.add_noise(model_diff, sensitivity)

        raised_exception = False
        global_model = utils.TwoFC()
        global_model.fill_all(5.0)
        privacy_engine.attach(global_model)
        try:
            privacy_engine.add_noise(model_diff, sensitivity)
        except PrivacyEngineNotAttachedException:
            raised_exception = True
        assertFalse(raised_exception)
