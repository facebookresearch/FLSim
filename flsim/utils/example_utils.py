#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# utils for use in the examples and tutorials

import random
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.vision import VisionDataset
from flsim.data.data_provider import IFLDataProvider, IFLUserData
from flsim.data.data_sharder import FLDataSharder, SequentialSharder
from flsim.interfaces.data_loader import IFLDataLoader
from flsim.interfaces.metrics_reporter import Channel
from flsim.interfaces.model import IFLModel
from flsim.metrics_reporter.tensorboard_metrics_reporter import FLMetricsReporter
from flsim.utils.data.data_utils import batchify
from flsim.utils.simple_batch_metrics import FLBatchMetrics







def collate_fn(batch: Any) -> Dict[str, Any]:
    if isinstance(batch, tuple):
        feature, label = batch
    elif isinstance(batch, dict):
        feature = batch["image"]
        label = batch["label"]
    else:
        raise TypeError("The batch must be a tuple or dict")
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
        # pyre-fixme[16]: `VisionDataset` has no attribute `__iter__`.
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
    def __init__(self, user_data: Dict[str, Generator], eval_split: float = 0.0):
        self._train_batches = []
        self._num_train_batches = 0
        self._num_train_examples = 0

        self._eval_batches = []
        self._num_eval_batches = 0
        self._num_eval_examples = 0

        self._eval_split = eval_split

        user_features = list(user_data["features"])
        user_labels = list(user_data["labels"])
        total = sum(len(batch) for batch in user_labels)

        for features, labels in zip(user_features, user_labels):
            if self._num_eval_examples < int(total * self._eval_split):
                self._num_eval_batches += 1
                self._num_eval_examples += UserData.get_num_examples(labels)
                self._eval_batches.append(UserData.fl_training_batch(features, labels))
            else:
                self._num_train_batches += 1
                self._num_train_examples += UserData.get_num_examples(labels)
                self._train_batches.append(UserData.fl_training_batch(features, labels))

    def num_train_examples(self) -> int:
        """
        Returns the number of train examples
        """
        return self._num_train_examples

    def num_eval_examples(self):
        """
        Returns the number of eval examples
        """
        return self._num_eval_examples

    def num_train_batches(self):
        """
        Returns the number of train batches
        """
        return self._num_train_batches

    def num_eval_batches(self):
        """
        Returns the number of eval batches
        """
        return self._num_eval_batches

    def train_data(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterator to return a user batch data for training
        """
        for batch in self._train_batches:
            yield batch

    def eval_data(self):
        """
        Iterator to return a user batch data for evaluation
        """
        for batch in self._eval_batches:
            yield batch

    @staticmethod
    def get_num_examples(batch: List) -> int:
        return len(batch)

    @staticmethod
    def fl_training_batch(
        features: List[torch.Tensor], labels: List[float]
    ) -> Dict[str, torch.Tensor]:
        # Check the type of the first element in labels list to determine if conversion is needed
        if not isinstance(labels[0], torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)
        else:
            labels = torch.stack(labels)

        return {"features": torch.stack(features), "labels": labels}


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
        # pyre-fixme[16]: `Dataset` has no attribute `__iter__`.
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
        self._train_users = self._create_fl_users(
            data_loader.fl_train_set(), eval_split=0.0
        )
        self._eval_users = self._create_fl_users(
            data_loader.fl_eval_set(), eval_split=1.0
        )
        self._test_users = self._create_fl_users(
            data_loader.fl_test_set(), eval_split=1.0
        )

    def train_user_ids(self) -> List[int]:
        return list(self._train_users.keys())

    def num_train_users(self) -> int:
        return len(self._train_users)

    def get_train_user(self, user_index: int) -> IFLUserData:
        if user_index in self._train_users:
            return self._train_users[user_index]
        raise IndexError(
            f"Index {user_index} is out of bound for list with len {self.num_train_users()}"
        )

    def train_users(self) -> Iterable[IFLUserData]:
        for user_data in self._train_users.values():
            yield user_data

    def eval_users(self) -> Iterable[IFLUserData]:
        for user_data in self._eval_users.values():
            yield user_data

    def test_users(self) -> Iterable[IFLUserData]:
        for user_data in self._test_users.values():
            yield user_data

    def _create_fl_users(
        self, iterator: Iterator, eval_split: float = 0.0
    ) -> Dict[int, IFLUserData]:
        return {
            user_index: UserData(user_data, eval_split=eval_split)
            for user_index, user_data in tqdm(
                enumerate(iterator), desc="Creating FL User", unit="user"
            )
        }


def build_data_provider(
    local_batch_size, examples_per_user, image_size
) -> DataProvider:
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
    print(f"Clients in total: {data_provider.num_train_users()}")
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
