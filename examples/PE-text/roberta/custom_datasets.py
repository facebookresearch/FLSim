# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.data import Dataset


class ListDataset(Dataset):
    def __init__(self, text_list):
        self.text_list = text_list

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        return self.text_list[idx]


class MatrixDataset(Dataset):
    def __init__(self, inputs):
        self.input_ids = inputs["input_ids"]
        self.attention_mask = inputs["attention_mask"]

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx, :][None, :],
            "attention_mask": self.attention_mask[idx, :][None, :],
        }
