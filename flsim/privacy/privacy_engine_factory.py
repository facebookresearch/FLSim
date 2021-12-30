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

from enum import Enum

from flsim.privacy.privacy_engine import (
    TreePrivacyEngine,
    GaussianPrivacyEngine,
    PrivacySetting,
    IPrivacyEngine,
)


class NoiseType(Enum):
    TREE_NOISE = "tree"
    GAUSSIAN = "guassian"


class PrivacyEngineFactory:
    @classmethod
    def create(
        cls,
        privacy_setting: PrivacySetting,
        users_per_round: int,
        num_total_users: int,
        noise_type: NoiseType,
    ) -> IPrivacyEngine:
        if noise_type == NoiseType.TREE_NOISE:
            return TreePrivacyEngine(
                privacy_setting=privacy_setting,
                users_per_round=users_per_round,
                num_total_users=num_total_users,
            )
        else:
            return GaussianPrivacyEngine(
                privacy_setting=privacy_setting,
                users_per_round=users_per_round,
                num_total_users=num_total_users,
            )
