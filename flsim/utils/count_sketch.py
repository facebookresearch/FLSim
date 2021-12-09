#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from copy import deepcopy
from typing import Optional

import torch
from flsim.interfaces.model import IFLModel

# there are some pyre errors here related to torch.Tensor operations, please ignore
# them as they work fine despite the error saying otherwise.
class CountSketch:
    """
    Implementation of the CountSketch data structure described here:
    http://dimacs.rutgers.edu/~graham/ssbd/ssbd3.pdf

    CountSketch is a data structure that can used to be compress a series of
    numbers and then decompress them using a fix-sized matrix and a set
    of pairwise independent hash functions.

    This version is designed to compressed IFLModels, where each weight
    simply gets an unique id, which is simply the number of weights added before
    this weight, and the model's parameter names and weight
    tensor shapes are stored for decompression.
    """

    def __init__(
        self,
        width: int = 10000,
        depth: int = 11,
        prime: int = 2 ** 31 - 1,
        independence: int = 2,
        h: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
        device="cpu",
    ):
        self.width: int = width
        self.depth: int = depth
        self.prime = prime
        self.buckets = torch.zeros((self.depth, self.width), device=device)
        self.independence = 4 if independence == 4 else 2
        self.device = device

        if h is None:
            self.h = torch.randint(
                low=1,
                high=self.prime,
                size=(self.depth, self.independence),
                device=self.device,
            )
        else:
            if list(h.size()) != [self.depth, self.independence]:
                raise AssertionError(
                    f"Hash function h should be of size {[self.depth, self.independence]}, but got {list(h.size())}"
                )
            self.h = h.to(device=self.device)

        if g is None:
            self.g = torch.randint(
                low=1,
                high=self.prime,
                size=(self.depth, self.independence),
                device=self.device,
            )
        else:
            if list(g.size()) != [self.depth, self.independence]:
                raise AssertionError(
                    f"Hash function g should be of size {[self.depth, self.independence]}, but got {list(g.size())}"
                )
            self.g = g.to(device=self.device)

        self.n = 0
        self.param_sizes = OrderedDict()

    def compute_hash_vector(self, x: torch.Tensor, hash: torch.Tensor) -> torch.Tensor:
        """
        Computes the hash of a vector x that represents the ids using the hash functions in the parameter hash.

        Args:
            x: the vector of ids, the expected shape is [num_ids]
            hash: the set of the hash functions to use. Expected shape is [self.depth, self.independence]

        Returns:
            Hash values as a torch.Tensor in the size [num_ids, num_hash]
        """

        def pairwise(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
            """
            Perform a pairwise hash function that takes advantage of the broadcasting
            rules of tensor operations.

            Args:
                x: the vector of ids, expected shape is [num_ids]
                a: the coefficents used in the hash function,
                    expected shape is [num_ids, self.depth] or [self.depth]
                b: the offset used in the hash function,
                    expected shape is [num_ids, self.depth] or [self.depth]

            Returns:
                A torch.Tensor of size [num_ids, self.depth] that represents the hashes.
            """
            # pyre-ignore
            return (a * x.unsqueeze(-1) + b) % self.prime

        # should actually work for any independence, but need more testing
        # alternative way to do with list comprehension:
        # torch.cat([x ** i for i in range(self.independence)].view(1, -1), dim = 0)
        if self.independence == 4:
            a = hash[:, 0]
            for i in range(1, hash.size(1)):
                b = hash[:, i]
                a = pairwise(x, a, b)
            return a

        return pairwise(x, hash[:, 0], hash[:, 1])

    def h_hash(self, x: torch.Tensor) -> torch.Tensor:
        # pyre-ignore
        return self.compute_hash_vector(x, self.h) % self.width

    def g_hash(self, x: torch.Tensor) -> torch.Tensor:
        # pyre-ignore
        return 2 * (self.compute_hash_vector(x, self.g) % 2) - 1

    def update(self, x: torch.Tensor, weights: torch.Tensor) -> None:
        self.n += x.numel()
        idx = self.h_hash(x)  # [num_id, self.depth]
        sign = self.g_hash(x)  # [num_id, self.depth]
        signed_weights = (sign * weights.view(-1, 1)).t()  # [self.depth, num_id]

        # offset is used because put_ treats self as a 1D tensor.
        offset = (
            torch.arange(0, self.depth, device=self.device).view(-1, 1) * self.width
        )

        # use put_ instead of index_put_ due to better performance on GPU (100x)
        # see N795398 for empirical results.
        self.buckets.put_(idx.t() + offset, signed_weights, accumulate=True)

    def query(self, x: torch.Tensor) -> torch.Tensor:
        idx = self.h_hash(x)
        sign = self.g_hash(x)
        sketched_weights = self.buckets[range(0, self.depth), idx]
        return torch.median(sketched_weights * sign, dim=1)[0]

    def sketch_state_dict(self, state_dict: OrderedDict):
        """
        Sketch a state_dict and all its weights while also resetting params and n.

        Args:
            state_dict: the dictionary containing parameter names and weights, usually
            obtained by calling state_dict() on a nn.Module
        """
        self.reset_buckets()
        self.param_sizes = OrderedDict()
        self.n = 0

        for name, param in state_dict.items():
            self.param_sizes[name] = param.size()
            self.update(
                torch.arange(self.n, self.n + param.numel(), device=self.device),
                param.view(-1),
            )

    def reset_buckets(self):
        self.buckets.fill_(0)

    def set_params(self, state_dict: OrderedDict):
        """
        Sketch a state_dict and all its weights while also resetting params and n.

        Args:
            state_dict: the dictionary containing parameter names and weights, usually
            obtained by calling state_dict() on a nn.Module
        """
        self.reset_buckets()
        self.param_sizes = OrderedDict()
        self.n = 0

        for name, param in state_dict.items():
            self.param_sizes[name] = param.size()
            self.n += param.numel()

    def sketch_model(self, model: IFLModel) -> None:
        self.sketch_state_dict(model.fl_get_module().state_dict())

    def unsketch_model(self, k: int = -1) -> OrderedDict:
        """
        Unsketchs the model by reconstructing the OrderDict of the
        parameters and their weights from self.buckets and self.param_sizes.
        Supports taking the top_k parameters with the largest weights
        and zero out all the other weights.
        """
        if k == -1:
            k = self.n
        elif k > self.n:
            raise AssertionError(
                "Cannot unsketch with a top_k greater than the number of parameters"
            )

        weights = self.query(torch.arange(0, self.n, device=self.device))

        top, indices = torch.topk(torch.abs(weights), k, sorted=True, largest=True)
        mask = torch.zeros_like(weights, device=weights.device)
        mask[indices] = 1
        weights[mask != 1] = 0

        count = 0
        state_dict = OrderedDict()
        for param_name, param_size in self.param_sizes.items():
            state_dict[param_name] = weights[count : count + param_size.numel()].view(
                param_size
            )
            count += param_size.numel()

        return state_dict

    def linear_comb(self, wt1: float, cs, wt2: float):
        self.buckets *= wt1
        self.buckets += cs.buckets * wt2

    # from N778597
    def approx_L1(self):
        estimates = torch.sum(torch.abs(self.buckets), dim=1)
        return torch.median(estimates)

    def approx_L2(self):
        estimates = torch.sum(self.buckets ** 2)
        return torch.sqrt(torch.median(estimates))

    def get_size_in_bytes(self):
        """
        Calculate CountSketch size in bytes.
        """
        return self.buckets.numel() * self.buckets.element_size()

    def to(self, device):
        """
        Moves the CountSketch to device. Up to the user to make sure the device is valid.
        """
        self.buckets = self.buckets.to(device)
        self.h = self.h.to(device)
        self.g = self.g.to(device)
        self.device = device


def clone_count_sketch(copy: CountSketch) -> CountSketch:
    cs = CountSketch(
        copy.width,
        copy.depth,
        copy.prime,
        copy.independence,
        copy.h,
        copy.g,
        copy.device,
    )
    cs.param_sizes = deepcopy(copy.param_sizes)
    cs.buckets = copy.buckets.detach().clone()
    return cs


def linear_comb_count_sketch(
    cs1: CountSketch, wt1: float, cs2: CountSketch, wt2: float
) -> CountSketch:
    cs = clone_count_sketch(cs1)
    cs.buckets = wt1 * cs.buckets + wt2 * cs2.buckets
    return cs
