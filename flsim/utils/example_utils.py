#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Cifar-10 dataset specific utils for use in the tutorials

import random
from typing import Any, Tuple, Iterator, List, Generator, Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from flsim.data.data_provider import IFLDataProvider, IFLUserData
from flsim.data.data_sharder import FLDataSharder
from flsim.data.data_sharder import SequentialSharder
from flsim.interfaces.data_loader import IFLDataLoader
from flsim.interfaces.metrics_reporter import Channel
from flsim.interfaces.model import IFLModel
from flsim.metrics_reporter.tensorboard_metrics_reporter import FLMetricsReporter
from flsim.utils.data.data_utils import batchify
from flsim.utils.simple_batch_metrics import FLBatchMetrics
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm


def collate_fn(batch: Tuple) -> Dict[str, Any]:
    feature, label = batch
    return {"features": feature, "labels": label}


class DataLoader(IFLDataLoader):
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
        for _, (_, user_data) in enumerate(self.sharder.shard_rows(data_rows)):
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


class UserData(IFLUserData):
    def __init__(self, user_data: Dict[str, Generator]):
        self._user_batches = []
        self._num_batches = 0
        self._num_examples = 0
        for features, labels in zip(user_data["features"], user_data["labels"]):
            self._num_batches += 1
            self._num_examples += UserData.get_num_examples(labels)
            self._user_batches.append(UserData.fl_training_batch(features, labels))

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


class DataProvider(IFLDataProvider):
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
            user_index: UserData(user_data)
            for user_index, user_data in tqdm(
                enumerate(iterator), desc="Creating FL User", unit="user"
            )
        }


def build_data_provider(local_batch_size, examples_per_user, image_size):

    # 1. Create training, eval, and test datasets like in non-federated learning.
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = CIFAR10(
        root="./cifar10", train=True, download=True, transform=transform
    )
    test_dataset = CIFAR10(
        root="./cifar10", train=False, download=True, transform=transform
    )

    # 2. Create a sharder, which maps samples in the training data to clients.
    sharder = SequentialSharder(examples_per_shard=examples_per_user)

    # 3. Shard and batchify training, eval, and test data.
    fl_data_loader = DataLoader(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        test_dataset=test_dataset,
        sharder=sharder,
        batch_size=local_batch_size,
        drop_last=False,
    )

    # 4. Wrap the data loader with a data provider.
    data_provider = DataProvider(fl_data_loader)
    print(f"Clients in total: {data_provider.num_users()}")
    return data_provider


class SimpleConvNet(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0):
        super(SimpleConvNet, self).__init__()
        self.out_channels = 32
        self.stride = 1
        self.padding = 2
        self.layers = []
        in_dim = in_channels
        for _ in range(4):
            self.layers.append(
                nn.Conv2d(in_dim, self.out_channels, 3, self.stride, self.padding)
            )
            in_dim = self.out_channels
        self.layers = nn.ModuleList(self.layers)

        self.gn_relu = nn.Sequential(
            nn.GroupNorm(self.out_channels, self.out_channels, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        num_features = (
            self.out_channels
            * (self.stride + self.padding)
            * (self.stride + self.padding)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        for conv in self.layers:
            x = self.gn_relu(conv(x))

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(self.dropout(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class FLModel(IFLModel):
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device

    def fl_forward(self, batch) -> FLBatchMetrics:
        features = batch["features"]  # [B, C, 28, 28]
        batch_label = batch["labels"]
        stacked_label = batch_label.view(-1).long().clone().detach()
        if self.device is not None:
            features = features.to(self.device)

        output = self.model(features)

        if self.device is not None:
            output, batch_label, stacked_label = (
                output.to(self.device),
                batch_label.to(self.device),
                stacked_label.to(self.device),
            )

        loss = F.cross_entropy(output, stacked_label)
        num_examples = self.get_num_examples(batch)
        output = output.detach().cpu()
        stacked_label = stacked_label.detach().cpu()
        del features
        return FLBatchMetrics(
            loss=loss,
            num_examples=num_examples,
            predictions=output,
            targets=stacked_label,
            model_inputs=[],
        )

    def fl_create_training_batch(self, **kwargs):
        features = kwargs.get("features", None)
        labels = kwargs.get("labels", None)
        return UserData.fl_training_batch(features, labels)

    def fl_get_module(self) -> nn.Module:
        return self.model

    def fl_cuda(self) -> None:
        self.model = self.model.to(self.device)

    def get_eval_metrics(self, batch) -> FLBatchMetrics:
        with torch.no_grad():
            return self.fl_forward(batch)

    def get_num_examples(self, batch) -> int:
        return UserData.get_num_examples(batch["labels"])


class MetricsReporter(FLMetricsReporter):
    ACCURACY = "Accuracy"

    def __init__(
        self,
        channels: List[Channel],
        target_eval: float = 0.0,
        window_size: int = 5,
        average_type: str = "sma",
        log_dir: Optional[str] = None,
    ):
        super().__init__(channels, log_dir)
        self.set_summary_writer(log_dir=log_dir)
        self._round_to_target = float(1e10)

    def compare_metrics(self, eval_metrics, best_metrics):
        print(f"Current eval accuracy: {eval_metrics}%, Best so far: {best_metrics}%")
        if best_metrics is None:
            return True

        current_accuracy = eval_metrics.get(self.ACCURACY, float("-inf"))
        best_accuracy = best_metrics.get(self.ACCURACY, float("-inf"))
        return current_accuracy > best_accuracy

    def compute_scores(self) -> Dict[str, Any]:
        # compute accuracy
        correct = torch.Tensor([0])
        for i in range(len(self.predictions_list)):
            all_preds = self.predictions_list[i]
            pred = all_preds.data.max(1, keepdim=True)[1]

            assert pred.device == self.targets_list[i].device, (
                f"Pred and targets moved to different devices: "
                f"pred >> {pred.device} vs. targets >> {self.targets_list[i].device}"
            )
            if i == 0:
                correct = correct.to(pred.device)

            correct += pred.eq(self.targets_list[i].data.view_as(pred)).sum()

        # total number of data
        total = sum(len(batch_targets) for batch_targets in self.targets_list)

        accuracy = 100.0 * correct.item() / total
        return {self.ACCURACY: accuracy}

    def create_eval_metrics(
        self, scores: Dict[str, Any], total_loss: float, **kwargs
    ) -> Any:
        accuracy = scores[self.ACCURACY]
        return {self.ACCURACY: accuracy}


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
            user_index: UserData(user_data)
            for user_index, user_data in tqdm(
                enumerate(iterator), desc="Creating FL User", unit="user"
            )
        }
