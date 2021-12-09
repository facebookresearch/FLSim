#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import time
from collections import defaultdict
from itertools import zip_longest
from typing import Any, Dict, Generator, Iterable, List, Optional

import torch


def batchify(
    iterable: Iterable[Any], batch_size: int, drop_last: Optional[bool] = False
) -> Generator:
    """
    Groups list into batches
    Example:
    >>> batchify([1, 2, 3, 4, 5], 2)
    >>> [[1, 2], [3, 4], [5]]
    """
    iterators = [iter(iterable)] * batch_size
    for batch in zip_longest(*iterators, fillvalue=None):
        batch = [ex for ex in batch if ex is not None]
        if drop_last and len(batch) != batch_size:
            break
        yield batch


def merge_dicts(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Merge a list of dictionaries into one dictionary
    Example:
    >>> merge_dicts([{"a": torch.Tensor([1])}, {"a": torch.Tensor([2])}])
    >>> {"a": torch.Tensor([1.0, 2.0])},
    """
    res = defaultdict(list)
    for ex in batch:
        for key, value in ex.items():
            res[key].append(value)
    return {k: torch.cat(v) for k, v in res.items()}


def stable_hash(base: int = 100000) -> int:
    md5 = hashlib.md5(str(time.time()).encode("utf-8"))
    return int.from_bytes(md5.digest(), byteorder="little") % base
