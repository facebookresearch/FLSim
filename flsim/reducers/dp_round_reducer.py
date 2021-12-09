#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Optional, Tuple

from flsim.channels.base_channel import IdentityChannel
from flsim.interfaces.model import IFLModel
from flsim.privacy.common import PrivacyBudget, PrivacySetting
from flsim.privacy.privacy_engine import IPrivacyEngine
from flsim.privacy.privacy_engine_factory import PrivacyEngineFactory, NoiseType
from flsim.privacy.user_update_clip import UserUpdateClipper
from flsim.reducers.secure_round_reducer import (
    SecureRoundReducer,
    SecureRoundReducerConfig,
)
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.distributed.fl_distributed import FLDistributedUtils, OperationType
from flsim.utils.fl.common import FLModelParamUtils
from torch import nn


class DPRoundReducer(SecureRoundReducer):
    def __init__(
        self,
        *,
        global_model: IFLModel,
        num_users_per_round: int,
        total_number_of_users: int,
        channel: Optional[IdentityChannel] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=DPRoundReducerConfig,
            **kwargs,
        )

        super().__init__(
            global_model=global_model,
            num_users_per_round=num_users_per_round,
            total_number_of_users=total_number_of_users,
            channel=channel,
            name=name,
            **kwargs,
        )
        self.num_users_per_round = num_users_per_round

        self.privacy_on = (
            # pyre-ignore[16]
            self.cfg.privacy_setting.noise_multiplier >= 0
            and self.cfg.privacy_setting.clipping_value < float("inf")
        )
        self.clipping_value = self.cfg.privacy_setting.clipping_value
        self.user_update_clipper = UserUpdateClipper(self.dtype)
        if self.privacy_on:
            self.privacy_engine: IPrivacyEngine = PrivacyEngineFactory.create(
                self.cfg.privacy_setting,
                num_users_per_round,
                total_number_of_users,
                noise_type=self.cfg.noise_type,
            )
            self.privacy_engine.attach(global_model=self.ref_model.fl_get_module())
        self._privacy_budget = PrivacyBudget(
            delta=self.cfg.privacy_setting.target_delta
        )

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def update_reduced_module(self, delta_module: nn.Module, weight: float) -> None:
        """
        Please refer to ``RoundReducer.update_reduced_module`` for more info.
        Notes
        -----
        """
        if self.privacy_on:
            self.user_update_clipper.clip(delta_module, self.clipping_value)
        super().update_reduced_module(delta_module, weight)

    def reduce(self) -> Tuple[nn.Module, float]:
        if not self.privacy_on:
            return super().reduce()
        # only sum in the rank 0 reducer (no broadcast yet)
        self._reduce_all(OperationType.SUM)  # OperationType.SUM)
        self.logger.debug(f"Sum of weights after aggregation: {self.sum_weights}")
        if FLDistributedUtils.is_master_worker():
            total_weights = float(self.sum_weights.item())
            if abs(total_weights - self.num_users_per_round) > 1e-5:
                self.logger.error(
                    f"total weights {total_weights} is not equal to "
                    f"number of users {self.num_users_per_round}. "
                    "Please make sure reduction_type=AVERGAE."
                )

            """
            The final amount of noise added must be equal to
            (max_norm * noise_multiplier) / users_per_round, similar to
            Google's user-level DP https://arxiv.org/pdf/1710.06963.pdf.
            Note that in the _generate_noise() function, the noise_multiplier
            is already multiplied.
            """
            self.privacy_engine.add_noise(
                self.reduced_module, self.clipping_value / total_weights
            )

        # broadcast the new noisy model to all workers.
        state_dict = FLModelParamUtils.get_state_dict(
            self.reduced_module,
            # pyre-fixme[16]: `DPRoundReducer` has no attribute `cfg`.
            only_federated_params=self.cfg.only_federated_params,
        )
        FLDistributedUtils.distributed_operation(
            chain([self.sum_weights], state_dict.values()), OperationType.BROADCAST
        )
        self.logger.debug(
            f"Sum of client weights after reduction on worker: {self.sum_weights}"
        )
        self._privacy_budget = self.privacy_engine.get_privacy_spent()
        self.logger.info(f"User Privacy Budget: {self._privacy_budget}")
        return self.reduced_module, float(self.sum_weights.item())

    @property
    def privacy_budget(self) -> PrivacyBudget:
        return self._privacy_budget

    def reset(self, ref_model: IFLModel) -> None:
        super().reset(ref_model)
        if self.privacy_on:
            self.privacy_engine.attach(global_model=self.ref_model.fl_get_module())


@dataclass
class DPRoundReducerConfig(SecureRoundReducerConfig):
    """
    Contains configurations for a round reducer based on DP (user-level dp)
    """

    _target_: str = fullclassname(DPRoundReducer)
    privacy_setting: PrivacySetting = PrivacySetting()
    noise_type: NoiseType = NoiseType.GAUSSIAN
