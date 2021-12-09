#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import string
from typing import Tuple

import torch
from flsim.data.data_provider import FLDataProviderFromList
from flsim.data.data_sharder import SequentialSharder, PowerLawSharder
from flsim.data.dataset_data_loader import FLDatasetDataLoaderWithBatch
from flsim.interfaces.model import IFLModel
from flsim.utils.sample_model import TestDataSetting
from torch.utils.data import Dataset


class Utils:
    @staticmethod
    def get_label(character: str) -> torch.Tensor:
        return torch.tensor(ord(character) % 2 == 0, dtype=torch.bool)

    @staticmethod
    def get_text(character: str) -> torch.Tensor:
        return torch.tensor(ord(character) - ord("a"), dtype=torch.long)

    @staticmethod
    def get_characters(num: int):
        """Return [a,b,c,d,.....z,a,b,c,d....] till we get num total chars"""
        characters = list(string.ascii_lowercase) * (1 + num // 26)
        return characters[:num]


class DummyAlphabetDataset(Dataset):
    """
    create a dummy PyTorch Dataset of k characters
    """

    def __init__(self, num_rows: int = 26):
        self.num_rows = num_rows
        self._data_rows = DummyAlphabetDataset.provide_data(self.num_rows)

    def __getitem__(self, index):
        return self._data_rows[index]

    def __len__(self):
        return self.num_rows

    @staticmethod
    def provide_data(num_rows: int = 26):
        """Generate num_row rows of data. Each row:
        "label":0 or 1, "text":int between 0 and 25
        """
        characters = Utils.get_characters(num_rows)
        return [
            {
                TestDataSetting.LABEL_COL_NAME: Utils.get_label(character),
                TestDataSetting.TEXT_COL_NAME: Utils.get_text(character),
            }
            for character in characters
        ]

    @staticmethod
    def create_data_provider_and_loader(
        dataset: Dataset, examples_per_user: int, batch_size: int, model
    ) -> Tuple[FLDataProviderFromList, FLDatasetDataLoaderWithBatch]:
        """
        Creates a data provider and data loader of type IFLDataProvider for a dataset
        """
        fl_data_sharder = SequentialSharder(examples_per_shard=examples_per_user)
        fl_data_loader = FLDatasetDataLoaderWithBatch(
            dataset,
            dataset,
            dataset,
            fl_data_sharder,
            batch_size,
            batch_size,
            batch_size,
        )
        fl_data_provider = FLDataProviderFromList(
            fl_data_loader.fl_train_set(),
            fl_data_loader.fl_eval_set(),
            fl_data_loader.fl_test_set(),
            model,
        )
        return fl_data_provider, fl_data_loader

    @staticmethod
    def create_data_provider_and_loader_uneven_split(
        num_examples: int,
        num_fl_users: int,
        batch_size: int,
        model: IFLModel,
        alpha: float = 0.2,
    ) -> Tuple[FLDataProviderFromList, FLDatasetDataLoaderWithBatch]:
        """
        Creates a data proivder and data loader with uneven number of
        examples per user following the power law distribution with order alpha
        """
        dataset = DummyAlphabetDataset(num_examples)
        fl_data_sharder = PowerLawSharder(num_shards=num_fl_users, alpha=alpha)
        fl_data_loader = FLDatasetDataLoaderWithBatch(
            dataset,
            dataset,
            dataset,
            fl_data_sharder,
            batch_size,
            batch_size,
            batch_size,
        )
        fl_data_provider = FLDataProviderFromList(
            fl_data_loader.fl_train_set(),
            fl_data_loader.fl_eval_set(),
            fl_data_loader.fl_test_set(),
            model,
        )
        return fl_data_provider, fl_data_loader


class NonOverlappingDataset(Dataset):
    """
    Create a dataset with non-overlapping non-zero entries for users. This will be useful
    for testing that clients will have orthogonal updates in linear regression problems.
    """

    def __init__(
        self,
        num_users: int = 10,
        num_nonzeros_per_user: int = 4,
        num_data_per_user: int = 6,
    ):
        self.num_users = num_users
        self.num_nonzeros_per_user = num_nonzeros_per_user
        self.num_data_per_user = num_data_per_user
        self._data_rows = NonOverlappingDataset.provide_data(
            num_users=self.num_users,
            num_nonzeros_per_user=self.num_nonzeros_per_user,
            num_data_per_user=self.num_data_per_user,
        )

    def __getitem__(self, index):
        return self._data_rows[index]

    def __len__(self):
        return self.num_data_per_user * self.num_users

    @staticmethod
    def provide_data(
        num_users: int = 10, num_nonzeros_per_user: int = 4, num_data_per_user: int = 6
    ):
        """
        Generate data. Each successive group of num_user_data rows has
        num_user_nonzeros non-overlapping non-zero entries
        """

        num_rows = num_data_per_user * num_users
        num_cols = num_nonzeros_per_user * num_users
        non_overlap_data = torch.zeros(num_rows, num_cols)

        for row in range(num_rows):
            col_start = math.floor(row / num_data_per_user) * num_nonzeros_per_user
            non_overlap_data[
                row, col_start : col_start + num_nonzeros_per_user
            ] = torch.rand(1, num_nonzeros_per_user)

        labels = torch.rand(num_rows, 1)

        return [
            {
                TestDataSetting.LABEL_COL_NAME: labels[row, :],
                TestDataSetting.TEXT_COL_NAME: non_overlap_data[row, :],
            }
            for row in range(num_rows)
        ]

    @staticmethod
    def create_data_provider_and_loader(
        dataset: Dataset, examples_per_user: int, batch_size: int, model: IFLModel
    ) -> Tuple[FLDataProviderFromList, FLDatasetDataLoaderWithBatch]:
        """
        Creates a data provider and data loader of type IFLDataProvider for a dataset
        """
        fl_data_sharder = SequentialSharder(examples_per_shard=examples_per_user)
        fl_data_loader = FLDatasetDataLoaderWithBatch(
            dataset,
            dataset,
            dataset,
            fl_data_sharder,
            batch_size,
            batch_size,
            batch_size,
        )
        fl_data_provider = FLDataProviderFromList(
            fl_data_loader.fl_train_set(),
            fl_data_loader.fl_eval_set(),
            fl_data_loader.fl_test_set(),
            model,
        )
        return fl_data_provider, fl_data_loader


class RandomDataset(Dataset):
    """
    Create a dataset with random entries and labels.
    """

    def __init__(
        self, num_users: int = 10, num_data_per_user: int = 6, dim_data: int = 40
    ):
        self.num_users = num_users
        self.num_data_per_user = num_data_per_user
        self.dim_data = dim_data
        self._data_rows = RandomDataset.provide_data(
            num_users=self.num_users,
            num_data_per_user=self.num_data_per_user,
            dim_data=self.dim_data,
        )

    def __getitem__(self, index):
        return self._data_rows[index]

    def __len__(self):
        return self.num_data_per_user * self.num_users

    @staticmethod
    def provide_data(
        num_users: int = 10, num_data_per_user: int = 6, dim_data: int = 40
    ):
        """
        Generate data which is a random matrix.
        """

        num_rows = num_data_per_user * num_users
        num_cols = dim_data
        random_data = torch.randn(num_rows, num_cols)
        labels = torch.rand(num_rows, 1)

        return [
            {
                TestDataSetting.LABEL_COL_NAME: labels[row, :],
                TestDataSetting.TEXT_COL_NAME: random_data[row, :],
            }
            for row in range(num_rows)
        ]

    @staticmethod
    def create_data_provider_and_loader(
        dataset: Dataset, examples_per_user: int, batch_size: int, model: IFLModel
    ) -> Tuple[FLDataProviderFromList, FLDatasetDataLoaderWithBatch]:
        """
        Creates a data provider and data loader of type IFLDataProvider for a dataset
        """
        fl_data_sharder = SequentialSharder(examples_per_shard=examples_per_user)
        fl_data_loader = FLDatasetDataLoaderWithBatch(
            dataset,
            dataset,
            dataset,
            fl_data_sharder,
            batch_size,
            batch_size,
            batch_size,
        )
        fl_data_provider = FLDataProviderFromList(
            fl_data_loader.fl_train_set(),
            fl_data_loader.fl_eval_set(),
            fl_data_loader.fl_test_set(),
            model,
        )
        return fl_data_provider, fl_data_loader
