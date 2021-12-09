#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains the noise generation function and the required
DP parameters that an entity such as an FL server uses for user-level DP.
"""

import logging
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import torch
from flsim.common.logger import Logger
from flsim.privacy.common import (
    PrivacyBudget,
    PrivacySetting,
)
from opacus.accountants.analysis import rdp as privacy_analysis
from torch import nn


class PrivacyEngineNotAttachedException(Exception):
    """
    Exception class to be thrown from User Privacy Engine in case
    the User Privacy Engine is not attached.
    """

    pass


class IPrivacyEngine(ABC):
    def __init__(
        self,
        privacy_setting: PrivacySetting,
        users_per_round: int,
        num_total_users: int,
    ):
        self.setting = privacy_setting
        self.users_per_round = users_per_round
        self.num_total_users = num_total_users
        self.steps = 0

    @abstractmethod
    def attach(self, global_model: nn.Module) -> None:
        """
        Attach the privacy engine to the global model by setting
        a reference model
        """
        pass

    @abstractmethod
    def add_noise(self, model_diff: nn.Module, sensitivity: float) -> None:
        pass

    @abstractmethod
    def get_privacy_spent(self, target_delta: Optional[float] = None) -> PrivacyBudget:
        pass


class GaussianPrivacyEngine(IPrivacyEngine):
    """
    DP-SGD privacy engine where noise is independent
    and comes from a gaussian distribution
    """

    logger: logging.Logger = Logger.get_logger(__name__)

    def __init__(
        self,
        privacy_setting: PrivacySetting,
        users_per_round: int,
        num_total_users: int,
    ) -> None:

        super().__init__(privacy_setting, users_per_round, num_total_users)
        self.noise_multiplier = privacy_setting.noise_multiplier
        self.target_delta = privacy_setting.target_delta
        self.alphas = privacy_setting.alphas
        self.user_sampling_rate = float(users_per_round) / num_total_users
        self.device = None
        self.random_number_generator = None
        self.noise_seed = privacy_setting.noise_seed

    def attach(self, global_model: nn.Module):
        self.device = next(global_model.parameters()).device

        noise_seed = (
            int.from_bytes(os.urandom(8), byteorder="big", signed=True)
            if self.noise_seed is None
            else self.noise_seed
        )
        torch.cuda.manual_seed_all(noise_seed)  # safe to call even if no gpu.
        self.random_number_generator = torch.Generator(  # pyre-ignore
            device=self.device
        )
        # pyre-fixme[16]: `Generator` has no attribute `manual_seed`.
        self.random_number_generator.manual_seed(noise_seed)
        self.logger.debug("User privacy engine attached.")

    def add_noise(self, model_diff: nn.Module, sensitivity: float) -> None:
        """
        Adds noise to the model_diff (operation is in-place).

        This method adds noise to the parameters of the input model.
        This operation is in-place (modifies model_diff in this method)
        Noise is sampled from a normal distribution with 0 mean and
        standard deviation equal to sensitivity * noise_multiplier.

        Parameters
        ----------
        model_diff : nn.Module
            Noise will be added to the parameters of this model.
        sensitivity : float
            The sensitivity of the noise that will be added.
        """
        with torch.no_grad():
            for _, parameter in model_diff.named_parameters():
                noise = self._generate_noise(parameter.shape, sensitivity)
                parameter.copy_(parameter + noise)
            self.steps += 1

    def _generate_noise(self, size, sensitivity: float) -> torch.Tensor:
        if self.device is None or self.random_number_generator is None:
            random_gen = "no" if self.random_number_generator is None else "yes"
            raise PrivacyEngineNotAttachedException(
                "User Privacy Engine is not attached to the global model. "
                f"(device={self.device}, random number generator exists: {random_gen})."
                "Call attach() function first before calling."
            )
        if self.noise_multiplier > 0 and sensitivity > 0:
            return torch.normal(
                0,
                self.noise_multiplier * sensitivity,
                size,
                device=self.device,
                generator=self.random_number_generator,
            )
        return torch.zeros(size, device=self.device)

    def get_privacy_spent(self, target_delta: Optional[float] = None):
        if target_delta is None:
            target_delta = self.target_delta

        rdp = privacy_analysis.compute_rdp(
            q=self.user_sampling_rate,
            noise_multiplier=self.noise_multiplier,
            steps=self.steps,
            orders=self.alphas,
        )
        eps, opt_alpha = privacy_analysis.get_privacy_spent(
            orders=self.alphas, rdp=rdp, delta=target_delta
        )

        self.logger.info(
            f"User-level DP Privacy Parameters:"
            f"\tuser sampling rate = {100 * self.user_sampling_rate:.3g}%"
            f"\tnoise_multiplier = {self.noise_multiplier}"
            f"\tsteps = {self.steps}\n  satisfies "
            f"DP with Ɛ = {eps:.3g} "
            f"and δ = {target_delta}."
            f"  The optimal α is {opt_alpha}."
        )
        if opt_alpha == max(self.alphas) or opt_alpha == min(self.alphas):
            self.logger.info(
                "The privacy estimate is likely to be improved by expanding "
                "the set of alpha orders."
            )
        return PrivacyBudget(eps, opt_alpha, target_delta)


@dataclass
class TreeNode:
    start: int
    end: int
    height: int
    efficient: bool

    @property
    def weight(self):
        return (1 / (2 - math.pow(2, -self.height))) ** 0.5 if self.efficient else 1.0


class TreePrivacyEngine(IPrivacyEngine):
    """
    DP-FTRL privacy engine where noise is the cummulated noise from
    a private binary tree
    """

    logger: logging.Logger = Logger.get_logger(__name__)

    def __init__(
        self,
        privacy_setting: PrivacySetting,
        users_per_round: int,
        num_total_users: int,
        restart_rounds: int = 100,
        efficient_tree: bool = True,
    ) -> None:
        super().__init__(privacy_setting, users_per_round, num_total_users)
        self.num_leaf = min(num_total_users, restart_rounds * users_per_round)
        self.restart_rounds = restart_rounds
        self.num_tree: int = 1
        self.ref_model = None
        self.device = None

        self.tree = TreePrivacyEngine.build_tree(self.num_leaf, efficient_tree)

    @classmethod
    def build_tree(cls, num_leaf: int, efficient_tree: bool = True) -> List[TreeNode]:
        tree = [TreeNode(-1, -1, -1, True)] * (2 * num_leaf)
        # store leaf nodes at back of array
        for i, j in enumerate(range(num_leaf, num_leaf * 2)):
            tree[j] = TreeNode(start=i, end=i, height=0, efficient=efficient_tree)

        # fill in prefix sum internal nodes
        for i in range(num_leaf - 1, 0, -1):
            left = tree[i * 2]
            right = tree[i * 2 + 1]
            height = int(math.log2(abs(right.end - left.start) + 1))
            tree[i] = TreeNode(
                start=left.start, end=right.end, height=height, efficient=efficient_tree
            )
        return tree

    @classmethod
    def compute_rdp(cls, alphas, epoch, steps, sigma):
        alphas = np.array(alphas)
        return alphas * epoch * np.ceil(np.log2(steps + 1e-6)) / (2 * sigma ** 2)

    def get_privacy_spent(self, target_delta: Optional[float] = None) -> PrivacyBudget:
        target_delta = (
            self.setting.target_delta if target_delta is None else target_delta
        )

        rdp = TreePrivacyEngine.compute_rdp(
            alphas=self.setting.alphas,
            epoch=self.num_tree,
            steps=self.num_leaf,
            sigma=self.setting.noise_multiplier,
        )
        eps, opt_alpha = privacy_analysis.get_privacy_spent(
            orders=self.setting.alphas, rdp=rdp, delta=target_delta
        )

        if opt_alpha == max(self.setting.alphas) or opt_alpha == min(
            self.setting.alphas
        ):
            self.logger.info(
                "The privacy estimate is likely to be improved by expanding "
                "the set of alpha orders."
            )
        return PrivacyBudget(eps, opt_alpha, self.setting.target_delta)

    def attach(self, global_model: nn.Module, **kwargs) -> None:
        """
        Reset the tree by incrementing num_tree and reset steps to 0
        these will be used to do privacy calculations
        """
        self.device = next(global_model.parameters()).device
        self.num_tree += 1
        self.steps = 0

    def add_noise(self, model_diff: nn.Module, sensitivity: float) -> None:
        """
        Adds noise to cummulated noise to model diff
        """
        with torch.no_grad():
            for parameter in model_diff.parameters():
                noise = self.range_sum(
                    left=0,
                    right=self.users_per_round - 1,
                    size=parameter.shape,
                    sensitivity=sensitivity,
                )
                parameter.copy_(parameter + noise)

            self.steps += 1

    def range_sum(
        self, left: int, right: int, size: torch.Size, sensitivity: float
    ) -> torch.Tensor:
        left += self.num_leaf
        right += self.num_leaf
        sum_ = torch.zeros(size)
        while left <= right:
            noise_std = self.setting.noise_multiplier * sensitivity

            if left % 2 == 1:
                sum_ += self._generate_noise(size, noise_std) * self.tree[left].weight

                left += 1

            if right % 2 == 0:
                sum_ += self._generate_noise(size, noise_std) * self.tree[right].weight
                right -= 1

            left = left // 2
            right = right // 2
        return sum_

    def _generate_noise(self, size: torch.Size, noise_std: float):
        return torch.normal(
            mean=0,
            std=noise_std,
            size=size,
            device=self.device,
        )
