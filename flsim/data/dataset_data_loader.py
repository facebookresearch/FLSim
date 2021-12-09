#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Iterable, Optional, Type

import torch
from flsim.data.data_sharder import FLDataSharder
from flsim.interfaces.data_loader import IFLDataLoader
from flsim.interfaces.dataset import FLDataset
from torch.utils.data import DataLoader, Dataset


class FLDatasetDataLoader(IFLDataLoader):
    def __init__(
        self,
        dataset_class: Type[FLDataset],
        train_path: str,
        test_path: str,
        eval_path: str,
        sharder: FLDataSharder,
    ):
        # pyre-fixme[19]: Expected 0 positional arguments.
        self.train_dataset = dataset_class(train_path)
        # pyre-fixme[19]: Expected 0 positional arguments.
        self.test_dataset = dataset_class(test_path)
        # pyre-fixme[19]: Expected 0 positional arguments.
        self.eval_dataset = dataset_class(eval_path)
        self.sharder = sharder

    def fl_train_set(self, **kwargs) -> Iterable[Iterable[Any]]:
        train_batches = []
        for sharded_rows in self.sharder.shard_rows(self.train_dataset):
            train_batches.append(sharded_rows)
        return train_batches

    def fl_eval_set(self, **kwargs) -> Iterable[Any]:
        return self.eval_dataset

    def fl_test_set(self, **kwargs) -> Iterable[Any]:
        return self.test_dataset


class FLDatasetDataLoaderWithBatch(IFLDataLoader):
    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        eval_dataset: Dataset,
        sharder: FLDataSharder,
        train_batch_size: int,
        eval_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
    ):
        assert train_batch_size > 0, "Batch size should be a positive integer."
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.eval_dataset = eval_dataset
        self.sharder = sharder
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self._num_total_users: int = -1

    @property
    def num_total_users(self):
        assert (
            self._num_total_users != -1
        ), "num_total_users is valid only after fl_train_set() has been called"
        return self._num_total_users

    def fl_train_set(self, **kwargs) -> Iterable[Iterable[Any]]:
        self._num_total_users = 0
        rank = kwargs.get("rank", 0)
        world_size = kwargs.get("world_size", 1)

        train_batches = [
            user_data for _, user_data in self.sharder.shard_rows(self.train_dataset)
        ]
        # batch train_batches collected above
        final_train_batches = []
        # fetch attributes for each row
        keys = list(train_batches[0][0].keys())
        for one_user_data in train_batches:
            batched_user_data = []
            for i, single_data in enumerate(one_user_data):
                if i % self.train_batch_size == 0:
                    batched_user_data.append([])
                batched_user_data[-1].append(single_data)

            new_batched_user_data = []
            for a_batched_user_data in batched_user_data:
                batched_data_rows = {}
                for key in keys:
                    batched_data_rows[key] = []
                for single_user_data in a_batched_user_data:
                    for key in keys:
                        batched_data_rows[key].append(single_user_data[key])

                for key in keys:
                    batched_data_rows[key] = torch.stack(batched_data_rows[key])

                new_batched_user_data.append(batched_data_rows)
            # divide the total number of users evenly into world_size # of workers
            if self.num_total_users % world_size == rank:
                final_train_batches.append(new_batched_user_data)
            # count the total number of users
            self._num_total_users += 1

        return final_train_batches

    def fl_eval_set(self, **kwargs) -> Iterable[Any]:
        collate_fn = kwargs.get("collate_fn", None)  # identity function
        return DataLoader(
            self.eval_dataset, batch_size=self.eval_batch_size, collate_fn=collate_fn
        )

    def fl_test_set(self, **kwargs) -> Iterable[Any]:
        collate_fn = kwargs.get("collate_fn", None)  # identity function
        return DataLoader(
            self.test_dataset, batch_size=self.test_batch_size, collate_fn=collate_fn
        )
