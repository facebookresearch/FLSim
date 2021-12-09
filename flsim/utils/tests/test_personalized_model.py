#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from flsim.common.pytest_helper import (
    assertEqual,
    assertRaises,
    assertEmpty,
)
from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.fl.personalized_model import FLModelWithPrivateModules
from flsim.utils.tests.helpers.test_models import (
    FC_PRIVATE_MODULE_NAMES,
    PersonalizedFCModel,
)


@pytest.fixture(scope="class")
def prepare_fl_model_with_private_modules(request):
    request.cls.model_with_private_modules = PersonalizedFCModel()


@pytest.mark.usefixtures("prepare_fl_model_with_private_modules")
class TestFLModelWithPrivateModules:
    def test_clear_dict_module(self) -> None:
        # number of private parameters are number of fc layers in
        # private_module_names times 2 (W and b in each layer)
        assertEqual(
            len(list(FLModelWithPrivateModules.get_user_private_parameters())),
            len(FC_PRIVATE_MODULE_NAMES) * 2,
        )
        FLModelWithPrivateModules.clear_user_private_module_dict()
        assertEmpty(list(FLModelWithPrivateModules.get_user_private_parameters()))

    def test_get_user_private_attr(self) -> None:
        # pyre-ignore[16]: for pytest fixture
        fc1_layer = self.model_with_private_modules.get_user_private_attr("fc1")
        assertEqual(FLModelParamUtils.get_num_trainable_params(fc1_layer), 10 * 5 + 5)
        with assertRaises(AttributeError):
            self.model_with_private_modules.get_user_private_attr("fc2")
