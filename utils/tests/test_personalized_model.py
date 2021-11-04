#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.fl.personalized_model import FLModelWithPrivateModules
from flsim.utils.tests.helpers.test_models import (
    FC_PRIVATE_MODULE_NAMES,
    PersonalizedFCModel,
)
from libfb.py import testutil


class FLModelWithPrivateModulesTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.model_with_private_modules = PersonalizedFCModel()

    def test_clear_dict_module(self) -> None:
        # number of private parameters are number of fc layers in
        # private_module_names times 2 (W and b in each layer)
        self.assertEqual(
            len(list(FLModelWithPrivateModules.get_user_private_parameters())),
            len(FC_PRIVATE_MODULE_NAMES) * 2,
        )
        FLModelWithPrivateModules.clear_user_private_module_dict()
        self.assertEmpty(list(FLModelWithPrivateModules.get_user_private_parameters()))

    def test_get_user_private_attr(self) -> None:
        fc1_layer = self.model_with_private_modules.get_user_private_attr("fc1")
        self.assertEqual(
            FLModelParamUtils.get_num_trainable_params(fc1_layer), 10 * 5 + 5
        )
        with self.assertRaises(AttributeError):
            self.model_with_private_modules.get_user_private_attr("fc2")
