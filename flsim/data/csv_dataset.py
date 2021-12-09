#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Any, Dict

# see https://fb.workplace.com/groups/fbcode/permalink/2440457449324413/
# @manual=third-party//pandas:pandas-py
import pandas as pd
from flsim.interfaces.dataset import FLDataset


class FLCSVDataset(FLDataset):
    """Abstract class that can be easily extended by implementing one light
    method (_get_processed_row_from_single_raw_row()) that is based on
    PyTorch's Dataset class. For example,
    ```
    class AdsCVRDataset(FLCSVDataset):
        def _get_processed_row_from_single_raw_row(self, raw_row: Any):
            return {
                "float_features": torch.Tensor(featurize(raw_row["feat"])),
                "label": torch.Tensor(raw_row["label"]),
            }
    ```
    then, one can simply enumerate over this Dataset as s/he usually doee
    with PyTorch Dataset to iterate over training/eval/test batches.
    However, note that you will most likely be encapsulating this within
    a concrete implementation of IFLDataLoader to use this within FL
    simulation with minimal effort (e.g. see FLAdsDataLoader).
    """

    def __init__(self, path: str):
        self.data_frame = pd.read_csv(path)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        raw_row = self.data_frame.iloc[idx]
        return self._get_processed_row_from_single_raw_row(raw_row)

    @abc.abstractmethod
    def _get_processed_row_from_single_raw_row(self, raw_row: Any) -> Dict[str, Any]:
        """This method should be overridden by the child class. One should
        provide logic to convert a raw row read from data into a processed
        row (i.e. dictionary where key is column name and value is whatever
        model expects such as tensor for features and integer for labels).
        """
        pass
