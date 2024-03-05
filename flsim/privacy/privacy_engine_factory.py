#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from enum import Enum

from flsim.privacy.privacy_engine import (
    GaussianPrivacyEngine,
    IPrivacyEngine,
    PrivacySetting,
)


class NoiseType(Enum):
    GAUSSIAN = "gaussian"


class PrivacyEngineFactory:
    @classmethod
    def create(
        cls,
        privacy_setting: PrivacySetting,
        users_per_round: int,
        num_total_users: int,
        noise_type: NoiseType,
    ) -> IPrivacyEngine:
        assert noise_type == NoiseType.GAUSSIAN
        return GaussianPrivacyEngine(
            privacy_setting=privacy_setting,
            users_per_round=users_per_round,
            num_total_users=num_total_users,
        )
