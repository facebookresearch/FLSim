#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import torch
from flsim.common.logger import Logger
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from hydra.utils import instantiate
from omegaconf import MISSING
from torch import nn


class FixedPointConverter:
    r"""
    The main class that is responsible for conversion between
    fixed point and floating point.
    """

    MAX_WIDTH = 8
    logger: logging.Logger = Logger.get_logger(__name__)

    def __init__(self, **kwargs):
        r"""
        Args:
            cfg: The config for FixedPointConverter

        Raises:
            ValueError: if the ``num_bytes`` is not between 1 and 8, or if
            ``config.scaling_factor`` is not greater than 0.
        """
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=FixedPointConfig,
            **kwargs,
        )

        if self.cfg.num_bytes < 1 or self.cfg.num_bytes > self.MAX_WIDTH:
            error_msg = (
                f"Width {self.cfg.num_bytes} is not supported. "
                f"Please enter a width between 1 and {self.MAX_WIDTH}."
            )
            raise ValueError(error_msg)
        if self.cfg.scaling_factor <= 0:
            raise ValueError("scaling factor must be greater than 0.")
        num_bits = self.cfg.num_bytes * 8
        self.max_value = 2 ** (num_bits - 1) - 1
        self.min_value = -(2 ** (num_bits - 1))
        self.scaling_factor = self.cfg.scaling_factor
        self._overflows = 0
        self._underflows = 0

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def to_fixedpoint(self, numbers: torch.Tensor) -> torch.Tensor:
        """
        Converts numbers in a tensor from floating point to fixed point.

        During conversion, the floats are multiplied by ``scaling_factor``.
        Now if some of these numbers are outside the range that can be represented by
        ``num_bytes`` bytes, they will be clamped to fit in the range.

        Args:
            numbers: the tensor containing the floating point numbers to convert

        Returns:
            A tensor containing the converted numbers to fixed point.
        """

        numbers = numbers.mul(self.scaling_factor)
        if (numbers < self.min_value).any():  # pyre-ignore[16]
            self._underflows = self._underflows + 1
        if (numbers > self.max_value).any():
            self._overflows = self._overflows + 1
        numbers = numbers.clamp(self.min_value, self.max_value)
        return torch.round(numbers)

    def to_float(self, numbers: torch.Tensor) -> torch.Tensor:
        """
        Converts numbers in a tensor from fixed point to floating point.

        Note that this method does not check if the fixed point numbers
        is withing the range of numbers that can be represented by
        ``num_bytes`` bytes.

        Args:
            numbers: the tensor containing the fixed point numbers to convert

        Returns:
            A tensor containing the converted number to floating point.
        """
        return numbers.div(self.scaling_factor)

    def _show_debug_vars(self):
        """
        Shows the underflow and overflow counts if debugging is enabled
        """
        if self.logger.isEnabledFor(logging.DEBUG):
            msg = (
                f"{self._underflows} tensor(s) have underflow and "
                f"{self._overflows} tensor(s) have overflow during fixed point conversion"
            )
            self.logger.warning(msg)


def utility_config_flatter(
    model: nn.Module, flat_config: FixedPointConfig
) -> Dict[str, FixedPointConfig]:
    """
    A utility function to use a "flat" (same config for all layers)
    FixedPointConfig for all layers of a model.

    Args:
        model: the reference model to obtain the named parameters
        flat_config: The flat config to use for all layers

    Returns:
        returns the flat fixedpoint_config_dict
    """
    config: Dict[str, FixedPointConfig] = {}
    for name, _ in model.named_parameters():
        config[name] = flat_config
    return config


class SecureAggregator:
    r"""
    The main class that is responsible for secure aggregation.

    Notes:
        Since this is a simulation of secure aggregation, it is simplified and
        not all details of secure aggregation are implemented. For instance, the
        noise generation, sharing random seed, denoising for secure aggregation
        are not implemented. Also, entities such as secure enclaves are not
        implemented.
    """

    def __init__(
        self,
        config: Dict[str, FixedPointConfig],
    ):
        r"""
        Args:
            config: a dictionary of fixed-point configs for different layers of
                neural network. If the utility ``utility_config_flatter`` is used,
                same config will be used for all layers of the neural network.

        """
        self.converters = {}
        for key in config.keys():
            self.converters[key] = instantiate(config[key])

    def _check_converter_dict_items(self, model: nn.Module) -> None:
        """
        Checks if all layers of a model have their corresponding configs

        Args:
            model: the model

        Raises:
            ValueError: If some layers of the model do not have their
                corresponding configs
        """
        unset_configs = set(model.state_dict()) - set(self.converters)
        if unset_configs:
            error_msg = (
                "Not all "
                "layers have their corresponding fixed point config. "
                f"The layers {unset_configs} do not have configs."
            )
            raise ValueError(error_msg)

    def params_to_fixedpoint(self, model: nn.Module) -> None:
        """
        Converts parameters of a model from floating point to fixed point.

        Args:
            model: the model whose parameters will be converted

        Raises:
            ValueError: If some layers of the model do not have their
                corresponding configs
        """
        self._check_converter_dict_items(model)
        state_dict = model.state_dict()
        for name in state_dict.keys():
            state_dict[name] = self.converters[name].to_fixedpoint(state_dict[name])
            self.converters[name]._show_debug_vars()
        model.load_state_dict(state_dict)

    def params_to_float(self, model: nn.Module) -> None:
        """
        Converts parameters of a model from fixed point to floating point.

        Args:
            model: the model whose parameters will be converted

        Raises:
            ValueError: If some layers of the model do not have their
                corresponding configs
        """
        self._check_converter_dict_items(model)
        state_dict = model.state_dict()
        for name in state_dict.keys():
            state_dict[name] = self.converters[name].to_float(state_dict[name])
        model.load_state_dict(state_dict)

    def _generate_noise_mask(
        self, update_params: Iterator[Tuple[str, nn.Parameter]]
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        """
        Generates noise mask, same shape as the update params

        Args:
            update_params: the parameters of the update sent from
                clients. Used to infer the shape of the noise mask

        Returns:
            noise mask
        """
        pass

    def apply_noise_mask(
        self, update_params: Iterator[Tuple[str, nn.Parameter]]
    ) -> None:
        """
        Applies noise mask to the parameters of the update sent from
            clients.

        Args:
            update_params: the parameters of the update sent from
                clients.

        Note:
            To properly implement this method, call ``_generate_noise_mask()``
            as ``noise_mask = self._generate_noise_mask(update_params)``. Then
            add the ``noise_mask`` to ``update_params`` and return the new
            ``update_params``.
        """
        pass

    def _get_denoise_mask(self) -> Iterator[Tuple[str, nn.Parameter]]:
        """
        Gets the aggregated denoised mask for all participating clients
        from secure enclave.

        Returns:
            aggregated denoised mask for all participating clients
        """
        pass

    def apply_denoise_mask(
        self, model_aggregate_params: Iterator[Tuple[str, nn.Parameter]]
    ) -> None:
        """
        Applies denoise mask to the noised aggregated updates from clients

        Args:
            model_aggregate_params: the parameters of the noised aggragated
                client updates. Used to infer the shape of the denoise mask

        Note:
            To properly implement this method, call ``_get_denoise_mask()``
            as ``denoise_mask = self._get_denoise_mask()``. Then add the
            ``denoise_mask`` to ``model_aggregate_params`` and return the
            new ``model_aggregate_params``.
        """
        pass


@dataclass
class FixedPointConfig:
    _target_: str = fullclassname(FixedPointConverter)
    _recursive_: bool = False
    # size in bytes of single fixed point number. 1 to 8 inclusive.
    num_bytes: int = MISSING
    # multiplier to convert from floating to fixed point
    scaling_factor: int = MISSING
