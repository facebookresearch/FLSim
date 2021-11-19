#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from flsim.channels.base_channel import (
    IdentityChannel,
    FLChannelConfig,
    Message,
)
from flsim.interfaces.model import IFLModel
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg


class RandomMaskChannel(IdentityChannel):
    """
    Implements a channel where the message sent from client to server is
    randomly masked by some proportion. We prune so that the number of
    non-sparse entries is deterministc and constant across runs for a given
    weight matrix.

    For message measurement size, we assume the tensors will be stored in
    the COO format. Hence, the message size may be higher than if we were to
    store all the matrices in the dense, `fp32` format for small values of
    `proportion_of_zero_weights`.
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=RandomMaskChannelConfig,
            **kwargs,
        )
        super().__init__()
        self.proportion_of_zero_weights = self.cfg.proportion_of_zero_weights
        assert (
            0 <= self.cfg.proportion_of_zero_weights < 1
        ), "Compression rate must be in [0, 1)"

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def _calc_message_size_client_to_server(self, message: Message):
        """
        We compute the size of the compressed message as follows: for any
        parameter, we count the number of non-zero entries. Then, we assume
        that the sparse tensor is stored in the COO format: each non-zero
        entry is stored with a value (in fp32) and an index (int64 * ndim).
        """
        message_size_bytes = 0
        for param in message.model_state_dict.values():

            # get number of non-sparse entries (nse)
            nse = param.numel() - int(self.proportion_of_zero_weights * param.numel())

            # size of the index
            message_size_bytes += param.ndim * RandomMaskChannel.BYTES_PER_INT64 * nse

            # size of the values
            message_size_bytes += nse * RandomMaskChannel.BYTES_PER_FP32
        return message_size_bytes

    def _on_client_before_transmission(self, message: Message) -> Message:
        """
        Here we apply a random mask to the parameters before sending the message.

        Notes:
            - The message is pruned so that the number of non-sparse entries is
               deterministc and constant across runs for a given weight matrix.
        """
        message.populate_state_dict()
        new_state_dict = OrderedDict()

        for name, param in message.model_state_dict.items():
            # exact number of elements to prune
            num_params_to_prune = int(self.proportion_of_zero_weights * param.numel())

            # select flat indices to prune
            prob = torch.rand_like(param.data)
            top_k = torch.topk(prob.view(-1), k=num_params_to_prune)

            # prune and adjust scale
            param.data.view(-1)[top_k.indices] = 0
            new_param = param.data / (1 - self.proportion_of_zero_weights)
            new_state_dict[name] = new_param

        message.model_state_dict = new_state_dict
        return message

    def _on_server_after_reception(self, message: Message) -> Message:
        message.update_model_()
        return message


@dataclass
class RandomMaskChannelConfig(FLChannelConfig):
    _target_: str = fullclassname(RandomMaskChannel)
    proportion_of_zero_weights: float = 0.5
