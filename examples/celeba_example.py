#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In this tutorial, we will train a binary classifier on LEAF's CelebA dataset with FLSim.

Before running this file, you need to download the dataset and partition the data by users.
1. Clone the leaf dataset by running `git clone https://github.com/TalwalkarLab/leaf.git`
2. Change direectory to celeba: `cd leaf/data/celeba || exit`
3. Download the data from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    - Download or request the metadata files `identity_CelebA.txt` and `list_attr_celeba.txt`,
      and place them inside the data/raw folder.
    - Download the celebrity faces dataset from the same site. Place the images in a folder
       named `img_align_celeba` in the same folder as above.
4. Run the pre-processing script:
    - `./preprocess.sh --sf 1.0 -k 0 -iu 1 -s niid -t 'user' --tf 0.90 --spltseed 1`

Typical usage example:
    python3 celeba_example.py --config-file configs/celeba_config.json
"""
import json
import os
import random
from typing import Any, Iterator, List, Tuple

import flsim.configs  # noqa
import hydra  # @manual
import torch
import torch.nn as nn
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from flsim.utils.example_utils import (
    DataProvider,
    FLModel,
    LEAFDataLoader,
    MetricsReporter,
)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from opacus.validators.module_validator import ModuleValidator
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


class CelebaDataset(Dataset):
    def __init__(
        self,
        data_root,
        image_root,
        num_users=None,
        transform=None,
        target_transform=None,
    ):
        with open(data_root, "r+") as f:
            self.dataset = json.load(f)

        user_ids = self.dataset["users"]
        num_users = num_users if num_users is not None else len(user_ids)
        user_ids = random.sample(user_ids, min(len(user_ids), num_users))

        self.transform = transform
        self.target_transform = target_transform

        self.image_root = image_root
        self.image_folder = ImageFolder(image_root, transform)
        self.data = {}
        self.targets = {}
        # Populate self.data and self.targets
        for user_id, user_data in self.dataset["user_data"].items():
            if user_id in user_ids:
                self.data[user_id] = [
                    int(os.path.splitext(img_path)[0]) for img_path in user_data["x"]
                ]
                self.targets[user_id] = list(user_data["y"])

    def __iter__(self) -> Iterator[Tuple[List[torch.Tensor], List[Any]]]:
        for user_id in self.data.keys():
            yield self.__getitem__(user_id)

    def __getitem__(self, user_id: str) -> Tuple[List[torch.Tensor], List[Any]]:

        if user_id not in self.data or user_id not in self.targets:
            raise IndexError(f"User {user_id} is not in dataset")

        user_imgs = []
        for image_index in self.data[user_id]:
            user_imgs.append(self.image_folder[image_index - 1][0])
        user_targets = self.targets[user_id]

        if self.target_transform is not None:
            user_targets = [self.target_transform(target) for target in user_targets]

        return user_imgs, user_targets

    def __len__(self) -> int:
        return len(self.data)


def build_data_provider(data_config):
    IMAGE_SIZE: int = 32
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = CelebaDataset(
        data_root="leaf/data/celeba/data/train/all_data_0_0_keep_0_train_9.json",
        image_root="leaf/data/celeba/data/raw/",
        transform=transform,
    )
    test_dataset = CelebaDataset(
        data_root="leaf/data/celeba/data/test/all_data_0_0_keep_0_test_9.json",
        transform=transform,
        image_root=train_dataset.image_root,
    )

    print(
        f"Created datasets with {len(train_dataset)} train users and {len(test_dataset)} test users"
    )

    dataloader = LEAFDataLoader(
        train_dataset,
        test_dataset,
        test_dataset,
        batch_size=data_config.local_batch_size,
        drop_last=data_config.drop_last,
    )
    data_provider = DataProvider(dataloader)
    print(f"Training clients in total: {data_provider.num_train_users()}")
    return data_provider


class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet18()
        # Replace batch norm with group norm
        self.backbone = ModuleValidator.fix(self.backbone)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class SimpleConvNet(nn.Module):
    r"""
    Simple CNN model following architecture from
    https://github.com/TalwalkarLab/leaf/blob/master/models/celeba/cnn.py#L19
    and https://arxiv.org/pdf/1903.03934.pdf
    """

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


def main_worker(
    trainer_config,
    data_config,
    model_config,
    use_cuda_if_available: bool = True,
    distributed_world_size: int = 1,
) -> None:
    data_provider = build_data_provider(data_config)
    model = (
        Resnet18(num_classes=2)
        if model_config.use_resnet
        else SimpleConvNet(
            in_channels=3, num_classes=2, dropout_rate=model_config.dropout
        )
    )

    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")
    # pyre-fixme[6]: Expected `Optional[str]` for 2nd param but got `device`.
    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()

    trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)

    final_model, eval_score = trainer.train(
        data_provider=data_provider,
        metrics_reporter=MetricsReporter(
            [Channel.TENSORBOARD, Channel.STDOUT],
            target_eval=model_config.target_eval,
            window_size=model_config.window_size,
            average_type=model_config.average_type,
        ),
        num_total_users=data_provider.num_train_users(),
        distributed_world_size=distributed_world_size,
    )
    trainer.test(
        data_provider=data_provider,
        metrics_reporter=MetricsReporter([Channel.STDOUT]),
    )


@hydra.main(config_path=None, config_name="celeba_config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    main_worker(
        trainer_config=cfg.trainer,
        data_config=cfg.data,
        model_config=cfg.model,
        use_cuda_if_available=cfg.use_cuda_if_available,
        distributed_world_size=cfg.distributed_world_size,
    )


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)
