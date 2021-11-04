#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

"""
Example usages:
FL Training with JSON config
buck run @mode/dev-nosan papaya/toolkit/simulation:run_cifar10_fl -- --config-file \
fblearner/flow/projects/papaya/examples/hydra_configs/cifar10_single_process.json

FL Training with YAML config (only with Hydra 1.1)
buck run @mode/dev-nosan papaya/toolkit/simulation:run_cifar10_fl -- --config-dir \
fblearner/flow/projects/papaya/examples/hydra_configs --config-name cifar10_single_process
"""
from tempfile import NamedTemporaryFile
from typing import Dict, List

import flsim.configs  # noqa
import hydra
import torch
from flsim.baselines.data_providers import FLVisionDataLoader, LEAFDataProvider
from flsim.baselines.models.cnn import Resnet18, SimpleConvNet
from flsim.baselines.models.cv_model import FLModel
from flsim.baselines.utils import train_non_fl
from flsim.data.data_sharder import FLDataSharder, ShardingStrategyType
from flsim.fb.metrics_reporters.cv_reporter import CVMetricsReporter
from flsim.fb.process_state import FBProcessState
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from flsim.utils.distributed.fl_distributed import FLDistributedUtils
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from primal.datasets.cifar import (
    CIFAR10,
    CIFAR100,
    gen_partial_cifar,
    PartitionerType,
    CIFARType,
)
from torch.multiprocessing import spawn
from torchvision import transforms

VIS_PATH = "log_dir"
SCORES_PATH = "scores"

CIFAR_MEAN_STD = {
    CIFARType.CIFAR10: ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    CIFARType.CIFAR20: None,  # TODO krp to add CIFAR20
    CIFARType.CIFAR100: ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}
IMAGE_SIZE = 24


def build_data_provider(local_batch_size, examples_per_user, drop_last=False):

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(*CIFAR_MEAN_STD[CIFARType.CIFAR10]),
        ]
    )
    train_dataset = CIFAR10(train=True, download=True, transform=transform)
    test_dataset = CIFAR10(train=False, download=True, transform=transform)

    print(
        f"Created datasets with {len(train_dataset)} train users and {len(test_dataset)} test users"
    )
    sharder = FLDataSharder(
        ShardingStrategyType.SEQUENTIAL, shard_size_for_sequential=examples_per_user
    )
    fl_data_loader = FLVisionDataLoader(
        train_dataset, test_dataset, test_dataset, sharder, local_batch_size, drop_last
    )
    data_provider = LEAFDataProvider(fl_data_loader)
    print(f"Clients in total: {data_provider.num_users()}")
    return data_provider


def dev_build_data_provider(
    cifar_type: CIFARType,
    random_augmentation: bool,
    partitioner_type: PartitionerType,
    num_partitions: int,
    label_to_partitions: Dict[int, List[float]],
    seed: int,
    partitions: List[int],
    local_batch_size: int,
    examples_per_user: int,
    drop_last=False,
):
    train_transforms = (
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if random_augmentation
        else []
    )
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(*CIFAR_MEAN_STD[cifar_type]),
    ]
    train_transforms.extend(normalize)  # pyre-ignore

    cifar = CIFAR10 if cifar_type == CIFARType.CIFAR10 else CIFAR100

    def partial_dataset(tf):
        main = cifar(train=tf is train_transforms, transform=transforms.Compose(tf))
        partials = gen_partial_cifar(
            main, partitioner_type, num_partitions, label_to_partitions, seed
        )
        return (
            partials.partition(partitions[0])
            if len(partitions) == 1
            else partials.joined_parition(partitions)
        )

    train_data, test_data = [
        partial_dataset(tf) for tf in [train_transforms, normalize]
    ]

    print(
        f"Created datasets with {len(train_data)} train users and {len(test_data)} test users"
    )
    sharder = FLDataSharder(
        ShardingStrategyType.SEQUENTIAL, shard_size_for_sequential=examples_per_user
    )
    fl_data_loader = FLVisionDataLoader(
        train_data, test_data, test_data, sharder, local_batch_size, drop_last
    )
    data_provider = LEAFDataProvider(fl_data_loader)
    print(f"Clients in total: {data_provider.num_users()}")
    return data_provider


def train(
    rank,
    result,
    dist_init_method,
    trainer_config,
    data_config,
    model_config,
    use_cuda_if_available=True,
    fb_info=None,
    world_size=1,
    dev_mode=False,
):
    FBProcessState.getInstance(rank=rank, fb_info=fb_info)
    metrics_reporter = CVMetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])
    print("Created metrics reporter")

    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{rank}" if cuda_enabled else "cpu")

    print(f"Training launched on device: {device}")
    FLDistributedUtils.dist_init(
        rank=rank,
        world_size=world_size,
        init_method=dist_init_method,
        use_cuda=cuda_enabled,
    )

    model = (
        Resnet18(num_classes=10)
        if model_config.use_resnet
        else SimpleConvNet(in_channels=3, num_classes=10)
    )
    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()

    trainer = instantiate(trainer_config, model=global_model, cuda_enabled=cuda_enabled)
    print(f"Created {trainer_config._target_}")

    if dev_mode:
        data_provider = dev_build_data_provider(
            cifar_type=CIFARType[data_config.get("cifar_type", "CIFAR10")],
            random_augmentation=data_config.get("random_augmentation", False),
            num_partitions=data_config.get("num_partitions", 2),
            partitioner_type=PartitionerType[
                data_config.get("partition_type", "Random")
            ],
            label_to_partitions=data_config.get("label_to_partitions", {}),
            seed=data_config.get("seed"),
            partitions=data_config.get("partitions", [0]),
            local_batch_size=data_config.local_batch_size,
            examples_per_user=data_config.examples_per_user,
            drop_last=False,
        )
    else:
        data_provider = build_data_provider(
            local_batch_size=data_config.local_batch_size,
            examples_per_user=data_config.examples_per_user,
            drop_last=False,
        )

    final_model, eval_score = trainer.train(
        data_provider=data_provider,
        metric_reporter=metrics_reporter,
        num_total_users=data_provider.num_users(),
        distributed_world_size=world_size,
        rank=rank,
    )
    if FLDistributedUtils.is_master_worker():
        result[VIS_PATH] = metrics_reporter.writer.log_dir
        result[SCORES_PATH] = metrics_reporter.get_best_score()
    return final_model, metrics_reporter


@hydra.main(config_path=None, config_name="cifar10_single_process")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    trainer_config = cfg.trainer
    model_config = cfg.model
    data_config = cfg.data

    if cfg.non_fl:
        cuda_enabled = torch.cuda.is_available() and cfg.use_cuda_if_available
        device = torch.device("cuda:0" if cuda_enabled else "cpu")
        model = (
            Resnet18(num_classes=10)
            if model_config.use_resnet
            else SimpleConvNet(in_channels=3, num_classes=10)
        )
        data_provider = build_data_provider(
            local_batch_size=data_config.local_batch_size,
            examples_per_user=data_config.examples_per_user,
            drop_last=False,
        )
        train_non_fl(
            data_provider=data_provider,
            model=model,
            device=device,
            cuda_enabled=cuda_enabled,
            lr=trainer_config.client.optimizer.lr,
            momentum=trainer_config.client.optimizer.momentum,
            epochs=trainer_config.epochs,
        )
    else:
        manager = torch.multiprocessing.get_context("spawn").Manager()
        result = manager.dict({})
        with NamedTemporaryFile(delete=False, suffix=".dist_sync") as sync_file:
            dist_init_method = "file://" + sync_file.name
            args = (
                result,
                dist_init_method,
                trainer_config,
                data_config,
                model_config,
                cfg.use_cuda_if_available,
                None,
                cfg.distributed_world_size,
            )
            spawn(train, args=args, nprocs=cfg.distributed_world_size)
            print(result)


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)
