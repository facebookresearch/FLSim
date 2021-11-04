#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

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
