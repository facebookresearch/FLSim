#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

# Utilities related to model measurement, for example model size in terms of total
# parameters, number of non-zero parameters and byte size.

from typing import OrderedDict

import torch


def calc_model_size(state_dict: OrderedDict):
    """
    Calculates model size in bytes given a state dict.
    """
    model_size_bytes = sum(
        p.numel() * p.element_size() for (_, p) in state_dict.items()
    )
    return model_size_bytes


def calc_model_sparsity(state_dict: OrderedDict):
    """
    Calculates model sparsity (fraction of zeroed weights in state_dict).
    """
    non_zero = 0
    tot = 1e-6
    for _, param in state_dict.items():
        non_zero += torch.count_nonzero(param).item()
        tot += float(param.numel())
    return 1.0 - non_zero / (tot + 1e-6)
