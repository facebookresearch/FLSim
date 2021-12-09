#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from flsim.channels.base_channel import (
    IdentityChannel,
    FLChannelConfig,
    Message,
)
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from torch.quantization.observer import (
    MinMaxObserver,
    PerChannelMinMaxObserver,
)


class ScalarQuantizationChannel(IdentityChannel):
    """
    Implements a channel that emulates scalar quantization from 1 to 8
    bits per weight (8 bits per weight corresponds to int8 quantization).
    We emulate this by successively quantizing and dequantizing. This way,
    the rest of the training is transparent for aggreagators, reducers,
    trainers and so on.

    Notes:
        - We can perform either per_tensor quantization (same scale and
          zero_point for every parameters in a weight matrix) or per_channel
          quantization (each channel has its own scale and zero_point). Set
          quantize_per_tensor = False to perform per_channel quantization.
        - We rely on the very simple MinMax observers for both per_tensor
          and per_channel quantization. This can be refined by leveraging the
          HistogramObserver for instance.
        - We do not quantize the biases for the moment since their compression
          overhead is very small.
        - We arbitrarily choose to set the int_repr() of a quantized tensor
          to [0, 2 ** n_bits - 1]. This is called unsigned quantization.
        - All the quantized tensors share the same type, ie `torch.quint8`.
          However, when quantizing to less than 8 bits, this is not memory
          efficient since each element is stored over 1 byte anyway. Since
          we are interested only in emulation for the moment, that's good.
          We could also have relied on the fake_quantize primitives but we
          prefer to rely on the true quantization operators.
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=ScalarQuantizationChannelConfig,
            **kwargs,
        )
        super().__init__(**kwargs)
        self.n_bits = self.cfg.n_bits
        self.quantize_per_tensor = self.cfg.quantize_per_tensor
        assert (
            0 < self.n_bits <= 8
        ), "ScalarQuantizationChannel expects n_bits between 1 and 8 (included)."
        self.quant_min = 0
        self.quant_max = 2 ** self.n_bits - 1
        self.observer, self.quantizer = self._get_observers_and_quantizers()

    def _calc_message_size_client_to_server(self, message: Message):
        """
        We compute the size of the compressed message as follows:
            - for the weights (compressed): n_bits / 8 bytes per element
            - for the biases (not compressed): 4 bytes per element
            - for the scales (one for each layer or one for each layer channel
              depending on quantize_per_tensor): 8 bytes / element (fp64)
            - for the zerp_points (one for each layer or one for each layer channel
              depending on quantize_per_tensor): 4 bytes / element (int32)
        """
        message_size_bytes = 0
        for param in message.model_state_dict.values():
            if param.ndim > 1:
                # integer representation
                message_size_bytes += param.numel() * self.n_bits / 8
                # size of scale(s) (fp64) and zero_point(s) (int32)
                if self.quantize_per_tensor:
                    message_size_bytes += ScalarQuantizationChannel.BYTES_PER_FP64
                    message_size_bytes += ScalarQuantizationChannel.BYTES_PER_FP32
                else:
                    # pyre-ignore[16]: `torch.Tensor` has no attribute `q_per_channel_scales`
                    n_scales = param.q_per_channel_scales().numel()
                    # pyre-ignore[16]: `torch.Tensor` has no attribute `q_per_channel_zero_points`
                    n_zero_points = param.q_per_channel_zero_points().numel()
                    message_size_bytes += (
                        ScalarQuantizationChannel.BYTES_PER_FP64 * n_scales
                    )
                    message_size_bytes += (
                        ScalarQuantizationChannel.BYTES_PER_FP32 * n_zero_points
                    )
            else:
                message_size_bytes += 4 * param.numel()
        return message_size_bytes

    def _get_observers_and_quantizers(self):
        if self.quantize_per_tensor:
            observer = MinMaxObserver(
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                quant_min=self.quant_min,
                quant_max=self.quant_max,
                reduce_range=False,
            )
            quantizer = torch.quantize_per_tensor
        else:
            observer = PerChannelMinMaxObserver(
                dtype=torch.quint8,
                qscheme=torch.per_channel_affine,
                quant_min=self.quant_min,
                quant_max=self.quant_max,
                reduce_range=False,
                ch_axis=0,
            )
            quantizer = torch.quantize_per_channel
        return observer, quantizer

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:

        # important to reset values, otherwise takes running min and max
        self.observer.reset_min_max_vals()

        # forward through the observer to get scale(s) and zero_point(s)
        _ = self.observer(x)
        scale, zero_point = self.observer.calculate_qparams()

        # Emulate quantization. Not a no-op since we lose precision when quantizing.
        if self.quantize_per_tensor:
            xq = self.quantizer(x, float(scale), int(zero_point), dtype=torch.quint8)
        else:
            scale = scale.to(x.device)
            zero_point = zero_point.to(x.device)
            xq = self.quantizer(x, scale, zero_point, axis=0, dtype=torch.quint8)
        return xq

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def _on_client_before_transmission(self, message: Message) -> Message:
        """
        We quantize the weights but do not quantize the biases since
        the overhead is very small. We copy the state dict since the
        tensor format changes.
        """
        message.populate_state_dict()
        new_state_dict = OrderedDict()
        for name, param in message.model_state_dict.items():
            if param.ndim > 1:
                new_state_dict[name] = self._quantize(param.data)
            else:
                new_state_dict[name] = param.data

        message.model_state_dict = new_state_dict
        return message

    def _on_server_before_transmission(self, message: Message) -> Message:
        message.populate_state_dict()
        return message

    def _on_server_after_reception(self, message: Message) -> Message:
        """
        We dequantize the weights and do not dequantize the biases
        since they have not beed quantized in the first place. We
        copy the state dict since the tensor format changes.
        """
        new_state_dict = OrderedDict()
        for name, param in message.model_state_dict.items():
            if param.ndim > 1:
                # pyre-ignore[16]: `torch.Tensor` has no attribute `dequantize`
                new_state_dict[name] = param.data.dequantize()
            else:
                new_state_dict[name] = param.data
        message.model_state_dict = new_state_dict
        message.update_model_()
        return message


@dataclass
class ScalarQuantizationChannelConfig(FLChannelConfig):
    _target_: str = fullclassname(ScalarQuantizationChannel)
    n_bits: int = 8
    quantize_per_tensor: bool = True
