#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from flsim.data.data_provider import IFLDataProvider, IFLUserData
from flsim.interfaces.model import IFLModel
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
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
        sample_averaging_weights = 1 / torch.pow(user_sample_counts, averaging_exponent)
        user_utility = sample_averaging_weights * user_utility
        return user_utility

    @staticmethod
    def samples_per_user(data_provider: IFLDataProvider) -> torch.Tensor:
        samples_per_user = [
            data_provider.get_user_data(u).num_examples()
            for u in data_provider.user_ids()
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
    next 2 users (user2 and user3) will be picked in the current round.
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


class NumberOfSamplesActiveUserSelector(ActiveUserSelector):
    """Active User Selector which chooses users with probability weights determined
    by the number of training points on the client. Clients with more samples are
    given higher probability of being selected. Assumes that number of samples
    on each client is known by the server and updated constantly.
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=NumberOfSamplesActiveUserSelectorConfig,
            **kwargs,
        )

        super().__init__(**kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def get_user_indices(self, **kwargs) -> List[int]:
        required_inputs = ["users_per_round", "data_provider"]
        users_per_round, data_provider = self.unpack_required_inputs(
            required_inputs, kwargs
        )

        user_utility = ActiveUserSelectorUtils.samples_per_user(data_provider)
        # pyre-fixme[16]: `NumberOfSamplesActiveUserSelector` has no attribute `cfg`.
        user_utility = torch.pow(user_utility, self.cfg.exponent)
        probs = ActiveUserSelectorUtils.convert_to_probability(
            user_utility=torch.log(user_utility),
            fraction_with_zero_prob=self.cfg.fraction_with_zero_prob,
            softmax_temperature=1,
        )
        selected_indices = ActiveUserSelectorUtils.select_users(
            users_per_round=users_per_round,
            probs=probs,
            fraction_uniformly_random=self.cfg.fraction_uniformly_random,
            rng=self.rng,
        )

        return selected_indices


class HighLossActiveUserSelector(ActiveUserSelector):
    """Active User Selector which chooses users with probability weights determined
    by the loss the model suffers on the user's data. Since this is a function of
    both the data on the client and the model, user loss values are not updated
    for every user before a selection round. Instead the class will keep a record
    of the loss and only update the value for a user when they are used for training
    since in this case the model is being transmitted to the user anyway.

    If this staleness become a significant problem, try using ideas from
    non-stationary UCB: https://arxiv.org/pdf/0805.3415.pdf.
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=HighLossActiveUserSelectorConfig,
            **kwargs,
        )

        super().__init__(**kwargs)

        self.user_losses: torch.Tensor = torch.tensor([], dtype=torch.float)
        self.user_sample_counts: torch.Tensor = torch.tensor([], dtype=torch.float)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def _get_initial_losses_and_counts(
        self,
        num_total_users: int,
        data_provider: IFLDataProvider,
        global_model: IFLModel,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_losses = torch.zeros(num_total_users, dtype=torch.float)
        user_sample_counts = torch.zeros(num_total_users, dtype=torch.float)
        for i in range(num_total_users):
            (
                user_losses[i],
                user_sample_counts[i],
            ) = self._get_user_loss_and_sample_count(
                data_provider.get_user_data(i), global_model
            )
        return user_losses, user_sample_counts

    def _non_active_sampling(
        self, num_total_users: int, users_per_round: int
    ) -> List[int]:
        selected_indices = torch.multinomial(
            torch.ones(num_total_users, dtype=torch.float),
            users_per_round,
            replacement=False,
            generator=self.rng,
        ).tolist()
        return selected_indices

    @staticmethod
    def _get_user_loss_and_sample_count(
        user_data: IFLUserData, model: IFLModel
    ) -> Tuple[float, int]:
        loss = 0
        num_samples = 0
        for batch in user_data:
            metrics = model.get_eval_metrics(batch)
            loss += metrics.loss.item() * metrics.num_examples
            num_samples += metrics.num_examples
        return loss, num_samples

    def get_user_indices(self, **kwargs) -> List[int]:
        required_inputs = [
            "num_total_users",
            "users_per_round",
            "data_provider",
            "global_model",
            "epoch",
        ]
        (
            num_total_users,
            users_per_round,
            data_provider,
            global_model,
            epoch,
        ) = self.unpack_required_inputs(required_inputs, kwargs)

        # pyre-fixme[16]: `HighLossActiveUserSelector` has no attribute `cfg`.
        if epoch < self.cfg.epochs_before_active:
            selected_indices = self._non_active_sampling(
                num_total_users, users_per_round
            )
            return selected_indices

        if self.user_losses.nelement() == 0:
            (
                self.user_losses,
                self.user_sample_counts,
            ) = self._get_initial_losses_and_counts(
                num_total_users, data_provider, global_model
            )

        user_utility = ActiveUserSelectorUtils.normalize_by_sample_count(
            user_utility=self.user_losses,
            user_sample_counts=self.user_sample_counts,
            averaging_exponent=self.cfg.count_normalization_exponent,
        )
        probs = ActiveUserSelectorUtils.convert_to_probability(
            user_utility=user_utility,
            fraction_with_zero_prob=self.cfg.fraction_with_zero_prob,
            softmax_temperature=self.cfg.softmax_temperature,
        )
        selected_indices = ActiveUserSelectorUtils.select_users(
            users_per_round=users_per_round,
            probs=probs,
            fraction_uniformly_random=self.cfg.fraction_uniformly_random,
            rng=self.rng,
        )

        for i in selected_indices:
            (
                self.user_losses[i],
                self.user_sample_counts[i],
            ) = self._get_user_loss_and_sample_count(
                data_provider.get_user_data(i), global_model
            )

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
class NumberOfSamplesActiveUserSelectorConfig(ActiveUserSelectorConfig):
    _target_: str = fullclassname(NumberOfSamplesActiveUserSelector)
    exponent: float = 1.0
    fraction_uniformly_random: float = 0.0
    fraction_with_zero_prob: float = 0.0


@dataclass
class HighLossActiveUserSelectorConfig(ActiveUserSelectorConfig):
    _target_: str = fullclassname(HighLossActiveUserSelector)
    count_normalization_exponent: float = 0.0
    epochs_before_active: int = 0
    fraction_uniformly_random: float = 0.0
    fraction_with_zero_prob: float = 0.0
    softmax_temperature: float = 1.0
