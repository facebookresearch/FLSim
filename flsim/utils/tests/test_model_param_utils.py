#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import collections
import math

import torch
import torch.nn as nn
from flsim.common.pytest_helper import (
    assertAlmostEqual,
    assertEqual,
    assertFalse,
    assertRaises,
    assertTrue,
)
from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.fl.personalized_model import FLModelWithPrivateModules
from flsim.utils.tests.helpers.test_models import (
    FCModel,
    LinearRegression,
    PersonalizedLinearRegression,
)
from flsim.utils.tests.helpers.test_utils import FLTestUtils

PRIVATE_SLOPE_MODULE_NAME: str = (
    FLModelWithPrivateModules.USER_PRIVATE_MODULE_PREFIX + "_a"
)


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

    def test_get_trainable_params(self) -> None:
        fc_model = FCModel()
        assertEqual(len(list(FLModelParamUtils.get_trainable_params(fc_model))), 6)

    def test_get_num_trainable_params(self) -> None:
        fc_model = FCModel()
        assertEqual(
            FLModelParamUtils.get_num_trainable_params(fc_model),
            10 * 5 + 5 * 3 + 3 * 1 + 5 + 3 + 1,
        )

    def test_get_gradient_l2_norm_raw(self) -> None:
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

    def test_model_linear_comb(self) -> None:
        """Test that computing linear comibination works for a model"""
        FLTestUtils.compare_model_linear_comb(FCModel(), FCModel())

    def test_gradient_reconstruction(self) -> None:
        """Test that gradient reconstruction works with a model.
        Create model, run some operations on it.
        """
        model, copy_model, reconstructed_grad = FCModel(), FCModel(), FCModel()
        FLTestUtils.compare_gradient_reconstruction(
            model, copy_model, reconstructed_grad
        )

    def test_fed_async_aggregation_with_weights(self) -> None:
        """Test that weights work for FedAsync aggregation"""
        torch.manual_seed(1)
        num_models = 4
        models = [FCModel() for i in range(num_models)]
        temp_model = FLModelParamUtils.clone(models[0])
        # verify that 0 weights work as expected
        # pyre-fixme[6]: Expected `List[nn.modules.module.Module]` for 1st param but
        #  got `List[FCModel]`.
        FLModelParamUtils.average_models(models, temp_model, [0, 0, 0, 1])
        assertTrue(
            FLModelParamUtils.get_mismatched_param([temp_model, models[3]]) == ""
        )
        # verify that equal weights work as expected
        # pyre-fixme[6]: Expected `List[nn.modules.module.Module]` for 1st param but
        #  got `List[FCModel]`.
        FLModelParamUtils.average_models(models, temp_model, [1, 1, 1, 1])
        temp_model_no_wts = FLModelParamUtils.clone(models[0])
        # pyre-fixme[6]: Expected `List[nn.modules.module.Module]` for 1st param but
        #  got `List[FCModel]`.
        FLModelParamUtils.average_models(models, temp_model_no_wts)
        assertTrue(
            FLModelParamUtils.get_mismatched_param([temp_model, temp_model_no_wts])
            == ""
        )
        # verify that unequal weights work as expected
        temp_model_1 = FLModelParamUtils.clone(models[0])
        # pyre-fixme[6]: Expected `List[nn.modules.module.Module]` for 1st param but
        #  got `List[FCModel]`.
        FLModelParamUtils.average_models(models, temp_model_1, [1, 1, 2, 2])
        temp_model_2 = FLModelParamUtils.clone(models[0])
        # pyre-fixme[6]: Expected `List[nn.modules.module.Module]` for 1st param but
        #  got `List[FCModel]`.
        FLModelParamUtils.average_models(models, temp_model_2, [2, 2, 1, 1])
        temp_model_3 = FLModelParamUtils.clone(models[0])
        FLModelParamUtils.average_models([temp_model_1, temp_model_2], temp_model_3)
        temp_model_4 = FLModelParamUtils.clone(models[0])
        # pyre-fixme[6]: Expected `List[nn.modules.module.Module]` for 1st param but
        #  got `List[FCModel]`.
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
    ) -> float:
        total_difference = 0.0
        for parameter1, parameter2 in zip(model1.parameters(), model2.parameters()):
            total_difference += torch.norm(parameter1.data - parameter2.data)
        return total_difference

    def test_simple_model_copy(self) -> None:
        """Test that FedAsync aggregation works for a simple Model"""
        num_models = 4
        orig_models = [FCModel() for i in range(num_models)]
        # pyre-fixme[6]: Expected `List[nn.modules.module.Module]` for 1st param but
        #  got `List[FCModel]`.
        FLTestUtils.average_and_verify_models(orig_models)

    def test_debug_model_norm(self) -> None:
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

    def test_set_gradient(self) -> None:
        model = LinearRegression()
        reconstructed_gradient = LinearRegression()
        reconstructed_gradient.a.data = torch.FloatTensor([0.5])
        reconstructed_gradient.b.data = torch.FloatTensor([1.0])
        FLModelParamUtils.set_gradient(
            model=model, reference_gradient=reconstructed_gradient
        )
        assertEqual(model.a.grad, reconstructed_gradient.a)
        assertEqual(model.b.grad, reconstructed_gradient.b)

    def test_gradient_linear_combine(self) -> None:
        """Test linear combination for gradients"""
        # Test whether function works if model to be saved is one of the two models
        # and if model to be saved is a completely new model
        for save_idx in range(3):
            for null_idx in range(4):
                models = [LinearRegression(), LinearRegression(), LinearRegression()]
                for m_idx in range(3):
                    if m_idx != null_idx:
                        models[m_idx].a.grad = torch.FloatTensor([0.5])
                        models[m_idx].b.grad = torch.FloatTensor([1.0])
                expected_grad_a = torch.FloatTensor([0])
                expected_grad_b = torch.FloatTensor([0])
                if models[0].a.grad is not None:
                    # pyre-fixme[58]: `*` is not supported for operand types `int` and `typing.Optional[torch._tensor.Tensor]`.
                    expected_grad_a += 3 * models[0].a.grad
                    # pyre-fixme[58]: `*` is not supported for operand types `int` and `typing.Optional[torch._tensor.Tensor]`.
                    expected_grad_b += 3 * models[0].b.grad
                if models[1].a.grad is not None:
                    # pyre-fixme[58]: `*` is not supported for operand types `int` and `typing.Optional[torch._tensor.Tensor]`.
                    expected_grad_a += 5 * models[1].a.grad
                    # pyre-fixme[58]: `*` is not supported for operand types `int` and `typing.Optional[torch._tensor.Tensor]`.
                    expected_grad_b += 5 * models[1].b.grad
                FLModelParamUtils.linear_combine_gradient(
                    models[0], 3, models[1], 5, models[save_idx]
                )
                assertEqual(models[save_idx].a.grad, expected_grad_a)
                assertEqual(models[save_idx].b.grad, expected_grad_b)
        models = [LinearRegression(), LinearRegression(), LinearRegression()]
        FLModelParamUtils.linear_combine_gradient(models[0], 3, models[1], 5, models[2])
        assert models[2].a.grad is None
        assert models[2].b.grad is None

    def test_add_gradients(self):
        """Test adding the gradients of two models"""
        models = [LinearRegression(), LinearRegression(), LinearRegression()]
        models[0].a.grad = torch.FloatTensor([1.0])
        FLModelParamUtils.add_gradients(models[0], models[1], models[2])
        assertEqual(models[2].a.grad, models[0].a.grad)
        assert models[2].b.grad is None
        models[1].a.grad = torch.FloatTensor([0.5])
        FLModelParamUtils.add_gradients(models[0], models[1], models[2])
        assertEqual(models[2].a.grad, torch.FloatTensor([1.5]))

    def test_subtract_gradients(self):
        """Test subtracting the gradients of a model with the gradients of another model"""
        models = [LinearRegression(), LinearRegression(), LinearRegression()]
        models[1].a.grad = torch.FloatTensor([1.0])
        FLModelParamUtils.subtract_gradients(models[0], models[1], models[2])
        assertEqual(models[2].a.grad, torch.FloatTensor([-1.0]))

    def test_copy_gradients(self):
        """Test copying the gradients of a model"""
        model = LinearRegression()
        model_copy = LinearRegression()
        model_copy.a.data.fill_(1.0)
        model.a.grad = torch.FloatTensor([0.5])
        FLModelParamUtils.copy_gradients(model, model_copy)
        assertEqual(model.a.grad, model_copy.a.grad)
        assertEqual(model_copy.a.data, torch.FloatTensor([1.0]))

    def test_multiply_gradients(self):
        """Test multiplying gradients of a model with a given weight"""
        model = LinearRegression()
        model2 = LinearRegression()
        model.a.grad = torch.FloatTensor([0.5])
        FLModelParamUtils.multiply_gradient_by_weight(model, 2, model2)
        assertEqual(model2.a.grad, torch.FloatTensor([1.0]))
        FLModelParamUtils.multiply_gradient_by_weight(model, 2, model)
        assertEqual(model.a.grad, torch.FloatTensor([1.0]))

    def test_get_mismatched_param(self) -> None:
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

    def test_copy_models(self) -> None:
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

    def test_scale_optimizer_lr(self) -> None:
        model = FCModel()

        # Test LR scaling with Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02, betas=(0.9, 0.99))
        FLModelParamUtils.scale_optimizer_lr(optimizer, 1 / 2.0)
        for param_group in optimizer.param_groups:
            assertEqual(
                param_group["lr"],
                0.04,
                "Adam LR does not match expected value after scaling",
            )
            assertEqual(
                param_group["betas"],
                (0.9, 0.99),
                "Adam betas does not match expected value after scaling",
            )

        # Test LR scaling with SGD momentum
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
        FLModelParamUtils.scale_optimizer_lr(optimizer, 1 / 2.0)
        for param_group in optimizer.param_groups:
            assertEqual(
                param_group["lr"],
                0.04,
                "SGD LR does not match expected value after scaling",
            )
            assertEqual(
                param_group["momentum"],
                0.9,
                "SGD momentum does not match expected value after scaling",
            )

        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
        with assertRaises(AssertionError):
            FLModelParamUtils.scale_optimizer_lr(optimizer, -2.0)
