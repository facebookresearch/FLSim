#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from typing import Any

import torch
from torch import Tensor


def to_onehot(x: Any, num_classes: int):
    """
    converts a vector if indices to a one hot vector
    """
    if isinstance(x, list):
        x = torch.as_tensor(x, dtype=torch.int64)
    assert isinstance(x, Tensor)
    assert len(x.shape) == 1 or x.shape[1] == 1, "input should be a vector"
    x = x.view(-1).to(torch.int64)
    return torch.nn.functional.one_hot(x, num_classes).float()


def multiple_batches_to_one(x: Any, merge: bool = True):
    """
    Given multiple similar batches, combines them to one batch.

    if merge is true features from different batches are concatenated
    into one vector (or tensor joined on first dimension)
    o.w. they will be a vector of predictions.
    """
    if isinstance(x, list):
        batch_size = len(x[0])
        assert merge, "For list inputs there is no option with merge = false"

        def concat(a):
            assert len(a) == batch_size, "cannot concat features of differnet sizes"
            return a

        return torch.cat([concat(a) for a in x], dim=1)
    # at this point we have a stack of bacth of features
    assert isinstance(x, Tensor)
    x = x.transpose(0, 1).contiguous()
    batch_size, num_batches = x.shape[0:2]
    num_dims = len(x.shape)
    return (
        x
        if merge is False
        else (
            x.view(batch_size, -1)
            if num_dims == 3
            else x.view(batch_size, -1, *x.shape[3:])
        )
    )


def append_batches(x: Tensor, y: Tensor):
    assert len(x) == len(
        y
    ), "batch sizes of x and y should match, x: {x.shape}, y: {y.shape}"
    assert (
        len(x.shape) == 2 and len(y.shape) == 2
    ), "Only one dimensional features are supported"
    return torch.cat([x, y], dim=-1)
