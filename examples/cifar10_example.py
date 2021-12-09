#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In this tutorial, we will train an image classifier with FLSim to simulate federated learning training environment.

With this tutorial, you will learn the following key components of FLSim:
1. Data loading
2. Model construction
3. Trainer construction

  Typical usage example:
    python3 cifar10_example.py --config-file configs/cifar10_config.json
"""
import flsim.configs  # noqa
import hydra
import torch
from flsim.data.data_sharder import SequentialSharder
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from flsim.utils.example_utils import (
    DataLoader,
    DataProvider,
    SimpleConvNet,
    FLModel,
    MetricsReporter,
)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10


IMAGE_SIZE = 32


def build_data_provider(local_batch_size, examples_per_user, drop_last=False):

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = CIFAR10(
        root="../cifar10", train=True, download=True, transform=transform
    )
    test_dataset = CIFAR10(
        root="../cifar10", train=False, download=True, transform=transform
    )
    sharder = SequentialSharder(examples_per_shard=examples_per_user)
    fl_data_loader = DataLoader(
        train_dataset, test_dataset, test_dataset, sharder, local_batch_size, drop_last
    )
    data_provider = DataProvider(fl_data_loader)
    print(f"Clients in total: {data_provider.num_users()}")
    return data_provider


def main(
    trainer_config,
    data_config,
    use_cuda_if_available=True,
):
    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")
    model = SimpleConvNet(in_channels=3, num_classes=10)
    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()
    trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)
    print(f"Created {trainer_config._target_}")
    data_provider = build_data_provider(
        local_batch_size=data_config.local_batch_size,
        examples_per_user=data_config.examples_per_user,
        drop_last=False,
    )

    metrics_reporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])

    final_model, eval_score = trainer.train(
        data_provider=data_provider,
        metric_reporter=metrics_reporter,
        num_total_users=data_provider.num_users(),
        distributed_world_size=1,
    )

    trainer.test(
        data_iter=data_provider.test_data(),
        metric_reporter=MetricsReporter([Channel.STDOUT]),
    )


@hydra.main(config_path=None, config_name="cifar10_tutorial")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    trainer_config = cfg.trainer
    data_config = cfg.data

    main(
        trainer_config,
        data_config,
    )


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)
