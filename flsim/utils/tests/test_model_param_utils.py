#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import collections
import copy
import math

import torch
import torch.nn as nn
from flsim.common.pytest_helper import (
    assertEqual,
    assertTrue,
    assertAlmostEqual,
    assertFalse,
    assertRaises,
)
from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.fl.personalized_model import FLModelWithPrivateModules
from flsim.utils.tests.helpers.test_models import (
    FCModel,
    LinearRegression,
    PersonalizedLinearRegression,
)
from flsim.utils.tests.helpers.test_utils import FLTestUtils

PRIVATE_SLOPE_MODULE_NAME = FLModelWithPrivateModules.USER_PRIVATE_MODULE_PREFIX + "_a"


class TestFLModelParamUtils:
    def test_get_state_dict(self) -> None:
        model = LinearRegression()
        assertEqual(
            set(FLModelParamUtils.get_state_dict(model, False).keys()), {"a", "b"}
        )
        assertEqual(
            set(FLModelParamUtils.get_state_dict(model, True).keys()), {"a", "b"}
        )

        personalized_model = PersonalizedLinearRegression()
        assertEqual(
            set(FLModelParamUtils.get_state_dict(personalized_model, False).keys()),
            {PRIVATE_SLOPE_MODULE_NAME, "b"},
        )
        assertEqual(
            set(FLModelParamUtils.get_state_dict(personalized_model, True).keys()),
            {"b"},
        )

    def test_load_state_dict(self) -> None:
        personalized_model = PersonalizedLinearRegression()

        state_dict = collections.OrderedDict()
        state_dict[PRIVATE_SLOPE_MODULE_NAME] = torch.tensor([1.0])
        state_dict["b"] = torch.tensor([0.5])

        FLModelParamUtils.load_state_dict(personalized_model, state_dict, False)
        assertEqual(
            dict(FLModelParamUtils.get_state_dict(personalized_model, False)),
            dict(state_dict),
        )
        # load_state_dict should work if non-private modules were given with
        # only_federated_params set as True
        state_dict_without_private_module = collections.OrderedDict()
        state_dict_without_private_module["b"] = torch.tensor([0.3])
        FLModelParamUtils.load_state_dict(
            personalized_model, state_dict_without_private_module, True
        )
        assertEqual(
            dict(FLModelParamUtils.get_state_dict(personalized_model, False)),
            {PRIVATE_SLOPE_MODULE_NAME: torch.tensor([1.0]), "b": torch.tensor([0.3])},
        )
        # throws when unexpected key is provided
        state_dict["c"] = torch.tensor([0.0])
        with assertRaises(AssertionError):
            FLModelParamUtils.load_state_dict(personalized_model, state_dict, True)
        # throws when non-private (i.e. federated module) is missing
        state_dict_with_missing_non_private_module = collections.OrderedDict()
        state_dict_with_missing_non_private_module["a"] = torch.tensor([1.0])
        with assertRaises(AssertionError):
            FLModelParamUtils.load_state_dict(
                personalized_model, state_dict_with_missing_non_private_module, True
            )

    def test_zero_weights(self) -> None:
        personalized_model = PersonalizedLinearRegression()
        FLModelParamUtils.load_state_dict(
            personalized_model,
            collections.OrderedDict(
                [
                    (PRIVATE_SLOPE_MODULE_NAME, torch.tensor([2.0])),
                    ("b", torch.tensor([1.0])),
                ]
            ),
            False,
        )
        FLModelParamUtils.zero_weights(personalized_model, True)
        assertEqual(
            dict(FLModelParamUtils.get_state_dict(personalized_model, False)),
            {PRIVATE_SLOPE_MODULE_NAME: torch.tensor([2.0]), "b": torch.tensor([0.0])},
        )
        FLModelParamUtils.zero_weights(personalized_model)
        assertEqual(
            dict(FLModelParamUtils.get_state_dict(personalized_model, False)),
            {PRIVATE_SLOPE_MODULE_NAME: torch.tensor([0.0]), "b": torch.tensor([0.0])},
        )

    def test_get_trainable_params(self):
        fc_model = FCModel()
        assertEqual(len(list(FLModelParamUtils.get_trainable_params(fc_model))), 6)

    def test_get_num_trainable_params(self):
        fc_model = FCModel()
        assertEqual(
            FLModelParamUtils.get_num_trainable_params(fc_model),
            10 * 5 + 5 * 3 + 3 * 1 + 5 + 3 + 1,
        )

    def test_get_gradient_l2_norm_raw(self):
        fc_model = FCModel()
        # set all gradients to 0, l2 norm should be zero
        for p in FLModelParamUtils.get_trainable_params(fc_model):
            p.grad = torch.zeros_like(p)
        assertEqual(FLModelParamUtils.get_gradient_l2_norm_raw(fc_model), 0.0)

        # set all gradients to 1, non-normalized l2 norm should be = sqrt(#params)
        num_trainable_params = FLModelParamUtils.get_num_trainable_params(fc_model)
        for p in FLModelParamUtils.get_trainable_params(fc_model):
            p.grad = torch.ones_like(p)
        assertAlmostEqual(
            FLModelParamUtils.get_gradient_l2_norm_raw(fc_model),
            math.sqrt(num_trainable_params),
            delta=1e-4,
        )

        # all gradients are std-normal-random, normalized grad norm = 1
        torch.manual_seed(1)
        for p in FLModelParamUtils.get_trainable_params(fc_model):
            p.grad = torch.randn_like(p)
        assertAlmostEqual(
            FLModelParamUtils.get_gradient_l2_norm_normalized(fc_model), 1, delta=1e-1
        )

    def test_model_linear_comb(self):
        """Test that computing linear comibination works for a model"""
        FLTestUtils.compare_model_linear_comb(FCModel(), FCModel())

    def test_gradient_reconstruction(self):
        """Test that gradient reconstruction works with a model.
        Create model, run some operations on it.
        """
        model, copy_model, reconstructed_grad = FCModel(), FCModel(), FCModel()
        FLTestUtils.compare_gradient_reconstruction(
            model, copy_model, reconstructed_grad
        )

    def test_fed_async_aggregation_with_weights(self):
        """Test that weights work for FedAsync aggregation"""
        torch.manual_seed(1)
        num_models = 4
        models = [FCModel() for i in range(num_models)]
        temp_model = copy.deepcopy(models[0])
        # verify that 0 weights work as expected
        FLModelParamUtils.average_models(models, temp_model, [0, 0, 0, 1])
        assertTrue(
            FLModelParamUtils.get_mismatched_param([temp_model, models[3]]) == ""
        )
        # verify that equal weights work as expected
        FLModelParamUtils.average_models(models, temp_model, [1, 1, 1, 1])
        temp_model_no_wts = copy.deepcopy(models[0])
        FLModelParamUtils.average_models(models, temp_model_no_wts)
        assertTrue(
            FLModelParamUtils.get_mismatched_param([temp_model, temp_model_no_wts])
            == ""
        )
        # verify that unequal weights work as expected
        temp_model_1 = copy.deepcopy(models[0])
        FLModelParamUtils.average_models(models, temp_model_1, [1, 1, 2, 2])
        temp_model_2 = copy.deepcopy(models[0])
        FLModelParamUtils.average_models(models, temp_model_2, [2, 2, 1, 1])
        temp_model_3 = copy.deepcopy(models[0])
        FLModelParamUtils.average_models([temp_model_1, temp_model_2], temp_model_3)
        temp_model_4 = copy.deepcopy(models[0])
        FLModelParamUtils.average_models(models, temp_model_4, [1, 1, 1, 1])

        mismatched_param = FLModelParamUtils.get_mismatched_param(
            [temp_model_3, temp_model_4], 1e-6
        )
        assertTrue(
            mismatched_param == "",
            (
                f"Mismatched param name: {mismatched_param}\n"
                f"temp_model_3:{temp_model_3}\n"
                f"temp_model_4:{temp_model_4}\n",
                f"total_difference:{self._compute_difference_in_norm(temp_model_3, temp_model_4)}",
            ),
        )

    def _compute_difference_in_norm(
        self, model1: torch.nn.Module, model2: torch.nn.Module
    ):
        total_difference = 0.0
        for (parameter1, parameter2) in zip(model1.parameters(), model2.parameters()):
            total_difference += torch.norm(parameter1.data - parameter2.data)
        return total_difference

    def test_simple_model_copy(self):
        """Test that FedAsync aggregation works for a simple Model"""
        num_models = 4
        orig_models = [FCModel() for i in range(num_models)]
        FLTestUtils.average_and_verify_models(orig_models)

    def test_debug_model_norm(self):
        fc_model = FCModel()
        for p in fc_model.parameters():
            torch.nn.init.constant_(p, 0.0)
        assertEqual(FLModelParamUtils.debug_model_norm(fc_model), 0)
        for p in fc_model.parameters():
            p.data.fill_(1.0)
        assertEqual(
            FLModelParamUtils.debug_model_norm(fc_model),
            FLModelParamUtils.get_num_trainable_params(fc_model),
        )

    def test_set_gradient(self):
        model = LinearRegression()
        reconstructed_gradient = LinearRegression()
        reconstructed_gradient.a.data = torch.FloatTensor([0.5])
        reconstructed_gradient.b.data = torch.FloatTensor([1.0])
        FLModelParamUtils.set_gradient(
            model=model, reference_gradient=reconstructed_gradient
        )
        assertEqual(model.a.grad, reconstructed_gradient.a)
        assertEqual(model.b.grad, reconstructed_gradient.b)

    def test_get_mismatched_param(self):
        a_val, b_val = 0.5, 1.0

        class MismatchingLinearRegression(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Parameter(torch.FloatTensor([a_val]))
                self.c = nn.Parameter(torch.FloatTensor([b_val]))

            def forward(self, x):
                return self.a + self.c * x

        model_1, model_2 = LinearRegression(), LinearRegression()
        model_1.a.data, model_1.b.data = (
            torch.FloatTensor([a_val]),
            torch.FloatTensor([b_val]),
        )
        model_2.a.data, model_2.b.data = (
            torch.FloatTensor([a_val]),
            torch.FloatTensor([b_val]),
        )

        # 1) models have same params => return an empty string
        assertEqual(FLModelParamUtils.get_mismatched_param([model_1, model_2]), "")

        # 2) only param 'a' is different => return 'a'
        model_2.a.data = torch.FloatTensor([b_val])
        assertEqual(FLModelParamUtils.get_mismatched_param([model_1, model_2]), "a")

        # 3) only param 'b' is different => return 'b'
        model_2.a.data, model_2.b.data = (
            torch.FloatTensor([a_val]),
            torch.FloatTensor([a_val]),
        )
        assertEqual(FLModelParamUtils.get_mismatched_param([model_1, model_2]), "b")

        # 4) both param 'a' and 'b' are different
        # => return the first mismatched param, which is 'a'
        model_2.a.data = torch.FloatTensor([b_val])
        assertEqual(FLModelParamUtils.get_mismatched_param([model_1, model_2]), "a")

        # 5) param 'b' in model_1 is missing in MismatchingLinearRegression
        # => return 'b'
        assertEqual(
            FLModelParamUtils.get_mismatched_param(
                [model_1, MismatchingLinearRegression()]
            ),
            "b",
        )

    def test_copy_models(self):
        torch.manual_seed(1)
        fc_model = FCModel()
        torch.manual_seed(2)
        copied_fc_model = FCModel()
        assertFalse(
            FLTestUtils.do_two_models_have_same_weights(fc_model, copied_fc_model)
        )

        FLModelParamUtils.copy_models(fc_model, [copied_fc_model])
        assertTrue(
            FLTestUtils.do_two_models_have_same_weights(fc_model, copied_fc_model)
        )
