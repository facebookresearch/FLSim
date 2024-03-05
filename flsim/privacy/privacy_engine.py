#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
This file contains the noise generation function and the required
DP parameters that an entity such as an FL server uses for user-level DP.
"""
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
from flsim.common.logger import Logger
from flsim.privacy.common import ClippingType, PrivacyBudget, PrivacySetting
from opacus.accountants.analysis import rdp as privacy_analysis
from torch import nn


@dataclass
class TreeNode:
    height: int
    value: Any


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
        if privacy_setting.clipping.clipping_type == ClippingType.ADAPTIVE:
            assert (
                privacy_setting.clipping.unclipped_num_std > 0
            ), "Adaptive clipping noise term must be greater than 0"
            # theorem 1 in https://arxiv.org/abs/1905.03871
            self.noise_multiplier = pow(
                pow(self.noise_multiplier + 1e-10, -2)
                - pow(2.0 * privacy_setting.clipping.unclipped_num_std, -2),
                -0.5,
            )

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
        self.random_number_generator = torch.Generator(device=self.device)
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

        if self.noise_multiplier > 0.0 and sensitivity > 0.0:
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


class CummuNoiseTorch:
    @torch.no_grad()
    def __init__(self, std, shapes, device, test_mode=False, seed=None):
        """
        :param std: standard deviation of the noise
        :param shapes: shapes of the noise, which is basically shape of the gradients
        :param device: device for pytorch tensor
        :param test_mode: if in test mode, noise will be 1 in each node of the tree
        """
        seed = (
            seed
            if seed is not None
            else int.from_bytes(os.urandom(8), byteorder="big", signed=True)
        )
        self.std = std
        self.shapes = shapes
        self.device = device
        self.step = 0
        self.binary = [0]
        self.noise_sum = [torch.zeros(shape).to(self.device) for shape in shapes]
        self.recorded = [[torch.zeros(shape).to(self.device) for shape in shapes]]
        torch.cuda.manual_seed_all(seed)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.test_mode = test_mode

    @torch.no_grad()
    def __call__(self):
        """
        :return: the noise to be added by DP-FTRL
        """
        self.step += 1
        if self.std <= 0 and not self.test_mode:
            return self.noise_sum

        idx = 0
        while idx < len(self.binary) and self.binary[idx] == 1:
            self.binary[idx] = 0
            for ns, re in zip(self.noise_sum, self.recorded[idx]):
                ns -= re
            idx += 1
        if idx >= len(self.binary):
            self.binary.append(0)
            self.recorded.append(
                [torch.zeros(shape).to(self.device) for shape in self.shapes]
            )

        for shape, ns, re in zip(self.shapes, self.noise_sum, self.recorded[idx]):
            if not self.test_mode:
                n = torch.normal(
                    0, self.std, shape, generator=self.generator, device=self.device
                )
            else:
                n = torch.ones(shape).to(self.device)
            ns += n
            re.copy_(n)

        self.binary[idx] = 1
        return self.noise_sum

    @torch.no_grad()
    def proceed_until(self, step_target):
        """
        Proceed until the step_target-th step. This is for the binary tree completion trick.
        :return: the noise to be added by DP-FTRL
        """
        if self.step >= step_target:
            raise ValueError(f"Already reached {step_target}.")
        while self.step < step_target:
            noise_sum = self.__call__()
        return noise_sum


class CummuNoiseEffTorch:
    """
    The tree aggregation protocol with the trick in Honaker, "Efficient Use of Differentially Private Binary Trees", 2015
    """

    @torch.no_grad()
    def __init__(self, std, shapes, device, seed, test_mode=False):
        """
        :param std: standard deviation of the noise
        :param shapes: shapes of the noise, which is basically shape of the gradients
        :param device: device for pytorch tensor
        """
        seed = (
            seed
            if seed is not None
            else int.from_bytes(os.urandom(8), byteorder="big", signed=True)
        )
        self.test_mode = test_mode

        self.std = std
        self.shapes = shapes
        self.device = device
        torch.cuda.manual_seed_all(seed)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.step = 0
        self.noise_sum = [torch.zeros(shape).to(self.device) for shape in shapes]
        self.stack = []

    @torch.no_grad()
    def get_noise(self):
        return [
            torch.normal(
                0, self.std, shape, generator=self.generator, device=self.device
            )
            for shape in self.shapes
        ]

    @torch.no_grad()
    def push(self, elem):
        for i in range(len(self.shapes)):
            self.noise_sum[i] += elem.value[i] / (2.0 - 1 / 2**elem.height)
        self.stack.append(elem)

    @torch.no_grad()
    def pop(self):
        elem = self.stack.pop()
        for i in range(len(self.shapes)):
            self.noise_sum[i] -= elem.value[i] / (2.0 - 1 / 2**elem.height)

    @torch.no_grad()
    def __call__(self):
        """
        :return: the noise to be added by DP-FTRL
        """
        self.step += 1

        # add new element to the stack
        self.push(TreeNode(0, self.get_noise()))

        # pop the stack
        while len(self.stack) >= 2 and self.stack[-1].height == self.stack[-2].height:
            # create new element
            left_value, right_value = self.stack[-2].value, self.stack[-1].value
            new_noise = self.get_noise()
            new_elem = TreeNode(
                self.stack[-1].height + 1,
                [
                    x + (y + z) / 2
                    for x, y, z in zip(new_noise, left_value, right_value)
                ],
            )

            # pop the stack, update sum
            self.pop()
            self.pop()

            # append to the stack, update sum
            self.push(new_elem)
        return self.noise_sum

    @torch.no_grad()
    def proceed_until(self, step_target):
        """
        Proceed until the step_target-th step. This is for the binary tree completion trick.
        :return: the noise to be added by DP-FTRL
        """
        if self.step >= step_target:
            raise ValueError(f"Already reached {step_target}.")
        while self.step < step_target:
            noise_sum = self.__call__()
        return noise_sum
