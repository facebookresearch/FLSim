#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import abc
import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from flsim.common.pytest_helper import assertNotEmpty
from flsim.data.data_provider import IFLDataProvider
from flsim.utils.config_utils import fullclassname, init_self_cfg
from omegaconf import MISSING


class ActiveUserSelectorUtils:
    @staticmethod
    def convert_to_probability(
        user_utility: torch.Tensor,
        fraction_with_zero_prob: float,
        softmax_temperature: float,
        weights=None,
    ) -> torch.Tensor:
        if weights is None:
            weights = torch.ones(len(user_utility), dtype=torch.float)
        num_to_zero_out = math.floor(fraction_with_zero_prob * len(user_utility))

        sorted_indices = torch.argsort(user_utility, descending=True).tolist()
        unnormalized_probs = torch.exp(softmax_temperature * user_utility) * weights
        if num_to_zero_out > 0:
            for i in sorted_indices[-num_to_zero_out:]:
                unnormalized_probs[i] = 0

        tmp_sum = sum(unnormalized_probs.tolist())
        assert tmp_sum > 0
        normalized_probs = unnormalized_probs / tmp_sum

        return normalized_probs

    @staticmethod
    def normalize_by_sample_count(
        user_utility: torch.Tensor,
        user_sample_counts: torch.Tensor,
        averaging_exponent: float,
    ) -> torch.Tensor:
        # pyre-fixme[58]: `/` is not supported for operand types `int` and `Tensor`.
        sample_averaging_weights = 1 / torch.pow(user_sample_counts, averaging_exponent)
        user_utility = sample_averaging_weights * user_utility
        return user_utility

    @staticmethod
    def samples_per_user(data_provider: IFLDataProvider) -> torch.Tensor:
        samples_per_user = [
            data_provider.get_train_user(u).num_train_examples()
            for u in data_provider.train_user_ids()
        ]
        samples_per_user = torch.tensor(samples_per_user, dtype=torch.float)
        return samples_per_user

    @staticmethod
    def select_users(
        users_per_round: int,
        probs: torch.Tensor,
        fraction_uniformly_random: float,
        rng: Any,
    ) -> List[int]:
        num_total_users = len(probs)
        num_randomly_selected = math.floor(users_per_round * fraction_uniformly_random)
        num_actively_selected = users_per_round - num_randomly_selected
        assert len(torch.nonzero(probs)) >= num_actively_selected

        if num_actively_selected > 0:

            actively_selected_indices = torch.multinomial(
                probs, num_actively_selected, replacement=False, generator=rng
            ).tolist()
        else:
            actively_selected_indices = []

        if num_randomly_selected > 0:
            tmp_probs = torch.tensor(
                [
                    0 if x in actively_selected_indices else 1
                    for x in range(num_total_users)
                ],
                dtype=torch.float,
            )

            randomly_selected_indices = torch.multinomial(
                tmp_probs, num_randomly_selected, replacement=False, generator=rng
            ).tolist()
        else:
            randomly_selected_indices = []

        selected_indices = actively_selected_indices + randomly_selected_indices
        return selected_indices

    @staticmethod
    def sample_available_users(
        users_per_round: int, available_users: List[int], rng: torch.Generator
    ) -> List[int]:
        if users_per_round >= len(available_users):
            return copy.copy(available_users)

        selected_indices = torch.multinomial(
            torch.ones(len(available_users), dtype=torch.float),
            users_per_round,
            replacement=False,
            generator=rng,
        ).tolist()

        return [available_users[idx] for idx in selected_indices]


class ActiveUserSelector(abc.ABC):
    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=ActiveUserSelectorConfig,
            **kwargs,
        )

        self.rng = torch.Generator()
        if self.cfg.user_selector_seed is not None:
            self.rng = self.rng.manual_seed(self.cfg.user_selector_seed)
        else:
            self.rng.seed()

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    @abc.abstractmethod
    def get_user_indices(self, **kwargs) -> List[int]:
        pass

    def get_users_unif_rand(
        self, num_total_users: int, users_per_round: int
    ) -> List[int]:
        selected_indices = torch.multinomial(
            torch.ones(num_total_users, dtype=torch.float),
            users_per_round,
            replacement=False,
            generator=self.rng,
        ).tolist()

        return selected_indices

    def unpack_required_inputs(
        self, required_inputs: List[str], kwargs: Dict[str, Any]
    ) -> List[Any]:
        inputs = []
        for key in required_inputs:
            input = kwargs.get(key, None)
            assert (
                input is not None
            ), "Input `{}` is required for get_user_indices in active_user_selector {}.".format(
                key, self.__class__.__name__
            )
            inputs.append(input)
        return inputs


class UniformlyRandomActiveUserSelector(ActiveUserSelector):
    """Simple User Selector which does random sampling of users"""

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=UniformlyRandomActiveUserSelectorConfig,
            **kwargs,
        )

        super().__init__(**kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def get_user_indices(self, **kwargs) -> List[int]:
        required_inputs = ["num_total_users", "users_per_round"]
        num_total_users, users_per_round = self.unpack_required_inputs(
            required_inputs, kwargs
        )

        selected_indices = torch.multinomial(
            torch.ones(num_total_users, dtype=torch.float),
            users_per_round,
            # pyre-fixme[16]: `UniformlyRandomActiveUserSelector` has no attribute
            #  `cfg`.
            replacement=self.cfg.random_with_replacement,
            generator=self.rng,
        ).tolist()

        return selected_indices


class SequentialActiveUserSelector(ActiveUserSelector):
    """Simple User Selector which chooses users in sequential manner.
    e.g. if 2 users (user0 and user1) were trained in the previous round,
    the next 2 users (user2 and user3) will be picked in the current round.
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=SequentialActiveUserSelectorConfig,
            **kwargs,
        )

        super().__init__(**kwargs)
        self.cur_round_user_index = 0

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def get_user_indices(self, **kwargs) -> List[int]:
        required_inputs = ["num_total_users", "users_per_round"]
        num_total_users, users_per_round = self.unpack_required_inputs(
            required_inputs, kwargs
        )

        # when having covered all the users, return the cursor to 0
        if num_total_users <= self.cur_round_user_index:
            self.cur_round_user_index = 0

        next_round_user_index = self.cur_round_user_index + users_per_round
        user_indices = list(
            range(
                self.cur_round_user_index, min(next_round_user_index, num_total_users)
            )
        )
        self.cur_round_user_index = next_round_user_index
        return user_indices


class RandomRoundRobinActiveUserSelector(ActiveUserSelector):
    """User Selector which chooses users randomly in a round-robin fashion.
    Each round users are selected uniformly randomly from the users not
    yet selected in that epoch.
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=RandomRoundRobinActiveUserSelectorConfig,
            **kwargs,
        )

        super().__init__(**kwargs)
        self.available_users = []

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def get_user_indices(self, **kwargs) -> List[int]:
        required_inputs = ["num_total_users", "users_per_round"]
        num_total_users, users_per_round = self.unpack_required_inputs(
            required_inputs, kwargs
        )
        # when having covered all the users, reset the list of available users
        if len(self.available_users) == 0:
            self.available_users = list(range(num_total_users))

        user_indices = ActiveUserSelectorUtils.sample_available_users(
            users_per_round, self.available_users, self.rng
        )

        # Update the list of available users
        # TODO(dlazar): ensure this is the fastest method. If not, write a util
        self.available_users = [
            idx for idx in self.available_users if idx not in user_indices
        ]

        return user_indices


class ImportanceSamplingActiveUserSelector(ActiveUserSelector):
    """User selector which performs Important Sampling.
    Each user is randomly selected with probability =
        `number of samples in user * clients per round / total samples in dataset`
    Ref: https://arxiv.org/pdf/1809.04146.pdf
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=ImportanceSamplingActiveUserSelectorConfig,
            **kwargs,
        )

        super().__init__(**kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def get_user_indices(self, **kwargs) -> List[int]:
        required_inputs = ["num_total_users", "users_per_round", "num_samples_per_user"]
        (
            num_total_users,
            users_per_round,
            num_samples_per_user,
        ) = self.unpack_required_inputs(required_inputs, kwargs)

        assert (
            len(num_samples_per_user) == num_total_users
        ), "Mismatch between num_total_users and num_samples_per_user length"
        assert users_per_round > 0, "users_per_round must be greater than 0"

        prob = torch.tensor(num_samples_per_user).float()
        total_samples = torch.sum(prob)
        assert total_samples > 0, "All clients have empty data"
        prob = prob * users_per_round / total_samples

        # Iterate num_tries times to ensure that selected indices is non-empty
        selected_indices = []
        # pyre-fixme[16]: `ImportanceSamplingActiveUserSelector` has no attribute `cfg`.
        for _ in range(self.cfg.num_tries):
            selected_indices = (
                torch.nonzero(torch.rand(num_total_users, generator=self.rng) < prob)
                .flatten()
                .tolist()
            )
            if len(selected_indices) > 0:
                break

        assertNotEmpty(
            selected_indices,
            "Importance Sampling did not return any clients for the current round",
        )

        return selected_indices


class RandomMultiStepActiveUserSelector(ActiveUserSelector):
    """Simple User Selector which does random sampling of users"""

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=RandomMultiStepActiveUserSelectorConfig,
            **kwargs,
        )
        self.gamma = self.cfg.gamma
        self.milestones = self.cfg.milestones
        self.users_per_round = 0
        super().__init__(**kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def get_user_indices(self, **kwargs) -> List[int]:
        required_inputs = ["num_total_users", "users_per_round", "global_round_num"]
        (
            num_total_users,
            users_per_round,
            global_round_num,
        ) = self.unpack_required_inputs(required_inputs, kwargs)

        if global_round_num in self.milestones:
            self.users_per_round *= self.gamma
            print(f"Increase Users Per Round to {self.users_per_round}")
        elif self.users_per_round == 0:
            self.users_per_round = users_per_round

        selected_indices = torch.multinomial(
            torch.ones(num_total_users, dtype=torch.float),
            self.users_per_round,
            # pyre-ignore[16]
            replacement=self.cfg.random_with_replacement,
            generator=self.rng,
        ).tolist()

        return selected_indices


@dataclass
class ActiveUserSelectorConfig:
    _target_: str = MISSING
    _recursive_: bool = False
    user_selector_seed: Optional[int] = None


@dataclass
class UniformlyRandomActiveUserSelectorConfig(ActiveUserSelectorConfig):
    _target_: str = fullclassname(UniformlyRandomActiveUserSelector)
    random_with_replacement: bool = False


@dataclass
class SequentialActiveUserSelectorConfig(ActiveUserSelectorConfig):
    _target_: str = fullclassname(SequentialActiveUserSelector)


@dataclass
class RandomRoundRobinActiveUserSelectorConfig(ActiveUserSelectorConfig):
    _target_: str = fullclassname(RandomRoundRobinActiveUserSelector)


@dataclass
class ImportanceSamplingActiveUserSelectorConfig(ActiveUserSelectorConfig):
    _target_: str = fullclassname(ImportanceSamplingActiveUserSelector)
    num_tries: int = 10


@dataclass
class RandomMultiStepActiveUserSelectorConfig(ActiveUserSelectorConfig):
    _target_: str = fullclassname(RandomMultiStepActiveUserSelector)
    random_with_replacement: bool = False
    gamma: int = 10
    milestones: List[int] = field(default_factory=list)
