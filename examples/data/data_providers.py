#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import random
from typing import Any, Dict, Generator, Iterable, Iterator, List, Tuple

import torch
from flsim.data.data_provider import IFLDataProvider, IFLUserData
from flsim.data.data_sharder import FLDataSharder
from flsim.interfaces.data_loader import IFLDataLoader
from flsim.utils.data.data_utils import batchify
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from tqdm import tqdm


def collate_fn(batch: Tuple) -> Dict[str, Any]:
    feature, label = batch
    return {"features": feature, "labels": label}


class FLVisionDataLoader(IFLDataLoader):
    SEED = 2137
    random.seed(SEED)

    def __init__(
        self,
        train_dataset: VisionDataset,
        eval_dataset: VisionDataset,
        test_dataset: VisionDataset,
        sharder: FLDataSharder,
        batch_size: int,
        drop_last: bool = False,
        collate_fn=collate_fn,
    ):
        assert batch_size > 0, "Batch size should be a positive integer."
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sharder = sharder
        self.collate_fn = collate_fn

    def fl_train_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        rank = kwargs.get("rank", 0)
        world_size = kwargs.get("world_size", 1)
        yield from self._batchify(self.train_dataset, self.drop_last, world_size, rank)

    def fl_eval_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.eval_dataset, drop_last=False)

    def fl_test_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.test_dataset, drop_last=False)

    def _batchify(
        self,
        dataset: VisionDataset,
        drop_last: bool = False,
        world_size: int = 1,
        rank: int = 0,
    ) -> Generator[Dict[str, Generator], None, None]:
        data_rows: List[Dict[str, Any]] = [self.collate_fn(batch) for batch in dataset]
        for index, (_, user_data) in enumerate(self.sharder.shard_rows(data_rows)):
            if index % world_size == rank and len(user_data) > 0:
                batch = {}
                keys = user_data[0].keys()
                for key in keys:
                    attribute = {
                        key: batchify(
                            [row[key] for row in user_data],
                            self.batch_size,
                            drop_last,
                        )
                    }
                    batch = {**batch, **attribute}
                yield batch


class LEAFDataLoader(IFLDataLoader):
    SEED = 2137
    random.seed(SEED)

    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int,
        drop_last: bool = False,
    ):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def fl_train_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.train_dataset, self.drop_last)

    def fl_eval_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.eval_dataset, drop_last=False)

    def fl_test_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.test_dataset, drop_last=False)

    def _batchify(
        self, dataset: Dataset, drop_last=False
    ) -> Generator[Dict[str, Generator], None, None]:
        for one_user_inputs, one_user_labels in dataset:
            data = list(zip(one_user_inputs, one_user_labels))
            random.shuffle(data)
            one_user_inputs, one_user_labels = zip(*data)
            batch = {
                "features": batchify(one_user_inputs, self.batch_size, drop_last),
                "labels": batchify(one_user_labels, self.batch_size, drop_last),
            }
            yield batch


class LEAFUserData(IFLUserData):
    def __init__(self, user_data: Dict[str, Generator]):
        self._user_batches = []
        self._num_batches = 0
        self._num_examples = 0
        for features, labels in zip(user_data["features"], user_data["labels"]):
            self._num_batches += 1
            self._num_examples += LEAFUserData.get_num_examples(labels)
            self._user_batches.append(LEAFUserData.fl_training_batch(features, labels))

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterator to return a user batch data
        """
        for batch in self._user_batches:
            yield batch

    def num_examples(self) -> int:
        """
        Returns the number of examples
        """
        return self._num_examples

    def num_batches(self) -> int:
        """
        Returns the number of batches
        """
        return self._num_batches

    @staticmethod
    def get_num_examples(batch: List) -> int:
        return len(batch)

    @staticmethod
    def fl_training_batch(
        features: List[torch.Tensor], labels: List[float]
    ) -> Dict[str, torch.Tensor]:
        return {"features": torch.stack(features), "labels": torch.Tensor(labels)}


class LEAFDataProvider(IFLDataProvider):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.train_users = self._create_fl_users(data_loader.fl_train_set())
        self.eval_users = self._create_fl_users(data_loader.fl_eval_set())
        self.test_users = self._create_fl_users(data_loader.fl_test_set())

    def user_ids(self) -> List[int]:
        return list(self.train_users.keys())

    def num_users(self) -> int:
        return len(self.train_users)

    def get_user_data(self, user_index: int) -> IFLUserData:
        if user_index in self.train_users:
            return self.train_users[user_index]
        else:
            raise IndexError(
                f"Index {user_index} is out of bound for list with len {self.num_users()}"
            )

    def train_data(self) -> Iterable[IFLUserData]:
        for user_data in self.train_users.values():
            yield user_data

    def eval_data(self) -> Iterable[Dict[str, torch.Tensor]]:
        for user_data in self.eval_users.values():
            for batch in user_data:
                yield batch

    def test_data(self) -> Iterable[Dict[str, torch.Tensor]]:
        for user_data in self.test_users.values():
            for batch in user_data:
                yield batch

    def _create_fl_users(self, iterator: Iterator) -> Dict[int, IFLUserData]:
        return {
            user_index: LEAFUserData(user_data)
            for user_index, user_data in tqdm(
                enumerate(iterator), desc="Creating FL User", unit="user"
            )
        }
