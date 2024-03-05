#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

from flsim.channels.base_channel import FLChannelConfig, IdentityChannel
from flsim.channels.message import Message
from flsim.channels.pq_utils.pq import PQ
from flsim.utils.config_utils import fullclassname, init_self_cfg


class ProductQuantizationChannel(IdentityChannel):
    """
    Implements a channel that emulates Product Quantization.
    The idea is to split the weight matrices (linear or
    convolutional) to a set of subvectors and to learn a
    codebook on these subvectors using k-means. More details
    on the procedure in the files em.py and pq.py. See paper for
    more details: https://arxiv.org/abs/1907.05686.

    Notes:
        - We do not quantize the biases since their compression
          overhead is very small.
        - We do not quantize small layers having less than
          `min_numel_to_quantize` elements.
        - There is the possibility to learn multiple codebooks
          per matrix by setting num_codebooks.
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=ProductQuantizationChannelConfig,
            **kwargs,
        )
        super().__init__(**kwargs)
        self.num_updates = 0

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def _on_server_before_transmission(self, message: Message) -> Message:
        message.populate_state_dict()
        return message

    def _calc_message_size_client_to_server(self, message: Message):
        """
        We compute the size of the compressed message as follows:
            - for the weights (compressed):
                * log(n_centroids) / 8 bytes per element (for the assignment)
                * num_codebooks * block_size * n_centroids fp32 elements for the centroids
            - for the biases (not compressed): 4 bytes per element

        Notes:
            - n_centroids is not necessarily equal to max_num_centroids, hence we
              recover it from the shape of `param["centroids"]`.
        """
        message_size_bytes = 0
        for param in message.model_state_dict.values():
            # param was compressed with PQ
            if type(param) is dict:
                block_size = param["centroids"].size(1)
                n_subvectors = param["assignments"].size(0)
                n_centroids = param["centroids"].size(0) // self.cfg.num_codebooks

                assignments_bytes = math.log2(n_centroids) / 8.0 * n_subvectors
                centroids_bytes = (
                    self.cfg.num_codebooks
                    * n_centroids
                    * block_size
                    * ProductQuantizationChannel.BYTES_PER_FP32
                )
                message_size_bytes += assignments_bytes + centroids_bytes
            # param is a non-compressed torch.Tensor
            else:
                message_size_bytes += (
                    ProductQuantizationChannel.BYTES_PER_FP32 * param.numel()
                )
        return message_size_bytes

    def _on_client_before_transmission(self, message: Message) -> Message:
        """
        We quantize the weights under the form of centroids
        and assignments and do not quantize the biases.
        """
        message.populate_state_dict()
        self.num_updates += 1

        # pyre-fixme[16]: `ProductQuantizationChannel` has no attribute `cfg`.
        if self.cfg.use_seed_centroids:
            seed_centroids = message.seed_centroids
            assert seed_centroids is not None, "Please provide seed centroids"
        else:
            seed_centroids = {}

        if self.num_updates > self.cfg.num_warmup_updates:
            new_state_dict = OrderedDict()
            for name, param in message.model_state_dict.items():
                # compress only large weight matrices
                if param.ndim > 1 and param.numel() >= self.cfg.min_numel_to_quantize:
                    pq = PQ(
                        param.data.size(),
                        self.cfg.max_block_size,
                        self.cfg.num_codebooks,
                        self.cfg.max_num_centroids,
                        self.cfg.num_k_means_iter,
                        self.cfg.verbose,
                    )
                    layer_seed_centroids = seed_centroids.get(name)
                    centroids, assignments = pq.encode(
                        param.data.cpu(), seed_centroids=layer_seed_centroids
                    )
                    compressed_param = {
                        "sizes": pq.sizes,
                        "centroids": centroids.data,
                        "assignments": assignments.data,
                    }
                    new_state_dict[name] = compressed_param
                # do not compress biases and small layers
                else:
                    new_state_dict[name] = param.data

            message.model_state_dict = new_state_dict
        return message

    def _on_server_after_reception(self, message: Message) -> Message:
        """
        We reconstruct the weights from the centroids
        and the assignments.
        """

        new_state_dict = OrderedDict()
        for name, param in message.model_state_dict.items():
            # param was compressed with PQ. TODO: more robust check than `type(param)`
            if type(param) is dict:
                pq = PQ(
                    param["sizes"],
                    self.cfg.max_block_size,
                    self.cfg.num_codebooks,
                    self.cfg.max_num_centroids,
                    self.cfg.num_k_means_iter,
                    self.cfg.verbose,
                )
                decompressed_param = pq.decode(
                    param["centroids"].data, param["assignments"].data
                )
                new_state_dict[name] = decompressed_param
            # param is a non-compressed torch.Tensor
            else:
                new_state_dict[name] = param.data
        message.model_state_dict = new_state_dict
        message.update_model_()
        return message


@dataclass
class ProductQuantizationChannelConfig(FLChannelConfig):
    _target_: str = fullclassname(ProductQuantizationChannel)
    max_num_centroids: int = 256
    min_numel_to_quantize: int = 10
    num_codebooks: int = 1
    max_block_size: int = 9
    num_k_means_iter: int = 20
    verbose: bool = False
    num_warmup_updates: int = 0
    use_seed_centroids: bool = False
    seed_centroids_refresh_freq: int = 1
