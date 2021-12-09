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


class SparseMaskChannel(IdentityChannel):
    """
    Implements a channel where the message sent from client to server is
    masked by the top-k absolute values in the model parameter weights.
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=SparseMaskChannelConfig,
            **kwargs,
        )
        super().__init__()
        self.proportion_of_zero_weights = self.cfg.proportion_of_zero_weights
        self.sparsity_method = self.cfg.sparsity_method
        self.compressed_size_measurement = self.cfg.compressed_size_measurement
        assert self.cfg.sparsity_method in {
            "random",
            "topk",
        }, "Compression method must be one of 'random' or 'topk'"
        assert (
            0 <= self.cfg.proportion_of_zero_weights < 1
        ), "Compression rate must be in [0, 1)"
        assert self.cfg.compressed_size_measurement in {
            "coo",
            "bitmask",
        }, "Compressed size measurement must be one of 'coo' or 'bitmask'"

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def _calc_message_size_client_to_server(self, message: Message):
        """
        For COO format : We compute the size of the compressed message as follows: for any
        parameter, we count the number of non-zero entries. Then, we assume
        that the sparse tensor is stored in the COO format: each non-zero
        entry is stored with a value (in fp32) and an index (int64 * ndim).

        For bitmask format : We compute the size of the compressed message as follows when
        serial format is specified: the non-zero entries are transmitted as is in fp32 format,
        and the sparsity mask is transmitted as a stream of bits to represent sparse locations.
        """
        message_size_bytes = 0
        for param in message.model_state_dict.values():

            # get number of non-sparse entries (nse)
            nse = param.numel() - int(self.proportion_of_zero_weights * param.numel())
            # size of the index
            if self.compressed_size_measurement == "coo":
                message_size_bytes += (
                    param.ndim * SparseMaskChannel.BYTES_PER_INT64 * nse
                )
            elif self.compressed_size_measurement == "bitmask":
                message_size_bytes += param.numel() * SparseMaskChannel.BYTES_PER_BIT

            # size of the values
            message_size_bytes += nse * SparseMaskChannel.BYTES_PER_FP32
        return message_size_bytes

    def _on_server_before_transmission(self, message: Message) -> Message:
        message.populate_state_dict()
        return message

    def _on_client_before_transmission(self, message: Message) -> Message:
        """
        Here we apply a sparse mask to the parameter updates before sending the message.

        Notes:
            - There are two options for sparsity: random and topk
            - In random sparsity, the mask is randomly selecting parameter weight updates
            - In TopK sparsity, sparsity is applied on each parameter's weight update
              separately depending on the magnitude of the values; the smallest values
              get pruned.
            - The message is pruned so that the number of non-sparse entries is
              deterministc and constant across runs for a given weight matrix.
        """
        message.populate_state_dict()
        new_state_dict = OrderedDict()

        for name, param in message.model_state_dict.items():
            # exact number of elements to prune
            num_params_to_prune = int(self.proportion_of_zero_weights * param.numel())

            # select flat indices to prune
            top_k = torch.topk(
                (
                    torch.rand_like(param.data)
                    if self.sparsity_method == "random"
                    else torch.abs(param.data)
                ).view(-1),
                k=num_params_to_prune,
                largest=False,
            )

            # prune top-K
            param.data.view(-1)[top_k.indices] = 0
            new_state_dict[name] = param.data

        message.model_state_dict = new_state_dict
        return message


@dataclass
class SparseMaskChannelConfig(FLChannelConfig):
    _target_: str = fullclassname(SparseMaskChannel)
    proportion_of_zero_weights: float = 0.5
    sparsity_method: str = "random"
    compressed_size_measurement: str = "bitmask"
