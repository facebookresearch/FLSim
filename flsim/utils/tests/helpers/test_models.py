#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable

import torch
import torch.nn as nn
from flsim.utils.fl.personalized_model import FLModelWithPrivateModules


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        return self.a + self.b * x


LR_PRIVATE_MODULE_NAMES = {"a"}


class PersonalizedLinearRegression(LinearRegression, FLModelWithPrivateModules):
    @classmethod
    def _get_user_private_module_names(cls) -> Iterable[str]:
        return LR_PRIVATE_MODULE_NAMES

    def __init__(self):
        super().__init__()

        # Set up user-private module attributes whenever we create a new
        # instance.
        self._maybe_set_up_user_private_modules()

        # Set forward hooks to reuse the forward() of the parent class.
        self._set_forward_hooks()


class FCModel(nn.Module):
    def __init__(self):
        super(FCModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


FC_PRIVATE_MODULE_NAMES = {"fc1", "fc3"}


class PersonalizedFCModel(FCModel, FLModelWithPrivateModules):
    @classmethod
    def _get_user_private_module_names(cls) -> Iterable[str]:
        return FC_PRIVATE_MODULE_NAMES

    def __init__(self):
        super().__init__()

        # Set up user-private module attributes whenever we create a new
        # instance.
        self._maybe_set_up_user_private_modules()

        # Set forward hooks to reuse the forward() of the parent class.
        self._set_forward_hooks()
