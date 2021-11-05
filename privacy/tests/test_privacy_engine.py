#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import copy
import math
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn
from flsim.common.pytest_helper import (
    assertEqual,
    assertNotEqual,
    assertAlmostEqual,
    assertRaises,
    assertFalse,
    assertLessEqual,
    assertTrue,
)
from flsim.privacy.common import PrivacySetting
from flsim.privacy.privacy_engine import (
    GaussianPrivacyEngine,
    TreePrivacyEngine,
    PrivacyEngineNotAttachedException,
)
from flsim.privacy.privacy_engine_factory import PrivacyEngineFactory, NoiseType
from flsim.tests import utils
from libfb.py import testutil
from opacus import privacy_analysis


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
        rdp = privacy_analysis.compute_rdp(sample_rate, noise_multiplier, steps, alphas)
        eps, _ = privacy_analysis.get_privacy_spent(alphas, rdp, delta=delta)
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
        be infinity.
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
        for _ in range(steps):  # adding noise will increse the steps
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

        model_diff_before_noise = copy.deepcopy(model_diff)

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

        model_diff_another_seed = copy.deepcopy(model_diff)
        model_diff_same_seed = copy.deepcopy(model_diff)

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


class TreePrivacyEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def _create_delta(self, dim, value=0.0):
        delta = nn.Linear(dim, 1)
        delta.bias.data.fill_(value)
        delta.weight.data.fill_(value)
        return delta, copy.deepcopy(delta)

    def _count_bits(self, n: int):
        """
        Returns the number of
        1s in the binary representations of i
        """
        count = 0
        while n:
            n &= n - 1
            count += 1
        return count

    @testutil.data_provider(
        lambda: (
            {"num_leaf": 4, "max_height": 2},
            {"num_leaf": 8, "max_height": 3},
            {"num_leaf": 16, "max_height": 4},
        )
    )
    def test_build_tree(self, num_leaf, max_height):
        """
        Test that build tree logic is correct.
        For any binary tree with n leaves, the tree's height
        will be log2(n) tall
        """
        tree = TreePrivacyEngine.build_tree(num_leaf)
        for node in tree:
            assertLessEqual(node.height, math.ceil(math.log2(num_leaf)))

    def test_basic_tree_node_sensitivity(self):
        """
        This is a small test. If we set the noise in each node as 1,
        we should be seeing the returned noise as the number of
        1s in the binary representations of ith step
        """

        def generate_noise(*args):
            return 1

        num_steps = 32
        tree = TreePrivacyEngine(
            PrivacySetting(noise_multiplier=1.0, noise_seed=0),
            users_per_round=1,
            num_total_users=num_steps,
            efficient_tree=False,
        )
        tree._generate_noise = MagicMock(side_effect=generate_noise)
        for i in range(num_steps):
            cumsum = tree.range_sum(0, i, size=torch.Size([1]), sensitivity=1.0)
            bits = float(self._count_bits(i + 1))
            assertEqual(cumsum, bits)

    @testutil.data_provider(
        lambda: (
            {"upr": 4, "n_users": 4, "noise_multiplier": 1, "exp_var": 0.57},
            {"upr": 7, "n_users": 7, "noise_multiplier": 1, "exp_var": 2.23},
            {"upr": 8, "n_users": 8, "noise_multiplier": 1, "exp_var": 0.53},
            {"upr": 8, "n_users": 8, "noise_multiplier": 2, "exp_var": 2.13},
            {"upr": 8, "n_users": 8, "noise_multiplier": 0.5, "exp_var": 0.13},
        )
    )
    def test_tree_noise_sum_expected(self, upr, n_users, noise_multiplier, exp_var):
        """
        See D27796316 test plan for explanation
        """
        delta, _ = self._create_delta(dim=1000, value=0)

        setting = PrivacySetting(
            noise_multiplier=noise_multiplier,
            noise_seed=0,
        )
        privacy_engine = PrivacyEngineFactory.create(
            setting,
            users_per_round=upr,
            num_total_users=n_users,
            noise_type=NoiseType.TREE_NOISE,
        )

        privacy_engine.add_noise(delta, sensitivity=1.0)
        noised_delta = torch.flatten(
            torch.stack([p for name, p in delta.named_parameters() if "weight" in name])
        )
        assertAlmostEqual(torch.var(noised_delta), exp_var, delta=0.15)

    @testutil.data_provider(
        lambda: (
            {"steps": 4, "upr": 4, "n_users": 4, "sigma": 1, "exp_var": 0.57},
            {"steps": 7, "upr": 7, "n_users": 7, "sigma": 1, "exp_var": 2.23},
            {"steps": 8, "upr": 8, "n_users": 8, "sigma": 1, "exp_var": 0.53},
            {"steps": 8, "upr": 8, "n_users": 8, "sigma": 2, "exp_var": 2.13},
            {"steps": 8, "upr": 8, "n_users": 8, "sigma": 0.5, "exp_var": 0.13},
        )
    )
    def test_range_sum_noise_expected(self, steps, upr, n_users, sigma, exp_var):
        """
        See D27796316 test plan for explanation
        """
        setting = PrivacySetting(
            noise_multiplier=sigma,
            noise_seed=1,
        )
        privacy_engine = PrivacyEngineFactory.create(
            setting,
            users_per_round=upr,
            num_total_users=n_users,
            noise_type=NoiseType.TREE_NOISE,
        )
        for i in range(steps):
            sum_ = privacy_engine.range_sum(0, i, torch.Size((1000,)), 1.0)

        assertAlmostEqual(torch.var(sum_), exp_var, delta=0.15)

    @testutil.data_provider(
        lambda: (
            {"n_users": 1600, "upr": 100, "sigma": 4.03, "epsilon": 4.19},
            {"n_users": 1600, "upr": 100, "sigma": 6.21, "epsilon": 2.60},
            {"n_users": 1600, "upr": 100, "sigma": 8.83, "epsilon": 1.77},
        )
    )
    def test_privacy_analysis_epsilon(self, n_users, upr, sigma, epsilon):
        delta, _ = self._create_delta(dim=1000, value=0)
        setting = PrivacySetting(
            noise_multiplier=sigma,
            noise_seed=1,
            alphas=np.arange(1.01, 100, 0.01).tolist(),
            target_delta=1e-6,
        )
        privacy_engine = PrivacyEngineFactory.create(
            setting,
            users_per_round=upr,
            num_total_users=n_users,
            noise_type=NoiseType.TREE_NOISE,
        )
        for _ in range(n_users // upr):
            privacy_engine.add_noise(delta, sensitivity=1.0)
        budget = privacy_engine.get_privacy_spent()
        assertAlmostEqual(budget.epsilon, epsilon, delta=0.5)
