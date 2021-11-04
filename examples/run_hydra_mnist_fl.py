#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import argparse
import tempfile
from typing import Any

import hydra
import numpy as np
import torch
from flsim.configs.config import ConfigBase, Configurable
from flsim.configs.configs.hydra_sync_trainer_config import HydraSyncTrainerConfig
from flsim.data.data_provider import FLDataProviderFromList
from flsim.data.data_sharder import FLDataSharder
from flsim.data.dataset_data_loader import FLDatasetDataLoaderWithBatch
from flsim.examples.mnist_fl_dataset import MNISTDataset
from flsim.examples.mnist_fl_metrics_reporter import MNISTMetricsReporter
from flsim.examples.mnist_fl_model import create_lighter_fl_model_for_mnist
from flsim.fb.process_state import FBProcessState
from flsim.interfaces.metrics_reporter import Channel
from flsim.trainers.sync_trainer import SyncTrainerConfig
from flsim.trainers.trainer_base import FLTrainerBase
from hydra.core.config_store import ConfigStore
from manifold.clients.python import ManifoldClient
from torchvision import transforms


"""
Example python script that kicks off FL simulator run
for MNIST training using hydra config as input.

Test locally:
buck run papaya/toolkit/simulation:run_hydra_mnist_fl
"""


MANIFOLD_BUCKET: str = "papaya"
TRAIN_DATASET_PATH: str = "tree/data/flow/training.pt"
TEST_DATASET_PATH: str = "tree/data/flow/test.pt"


def _create_config_from_hydra(config: HydraSyncTrainerConfig) -> SyncTrainerConfig:
    """Instantiate and returns a SyncTrainerConfig object from `config`,
    a HydraSyncTrainerConfig object. Hydra dynamically instantiates `config`
    from a yaml file and schema validates it by name-matching the yaml to
    the `sync_trainer_config` node in ConfigStore.
    """
    fl_config = SyncTrainerConfig(
        optimizer={
            "lr": config.optimizer.lr,
            "momentum": config.optimizer.momentum,
            "type": config.optimizer.type,
            "weight_decay": config.optimizer.weight_decay,
        },
        aggregator={
            "type": config.aggregator.type,
            "reducer_config": {
                "only_federated_params": config.aggregator.reducer_config.only_federated_params,
                "reduction_type": config.aggregator.reducer_config.reduction_type,
            },
        },
        timeout_simulator_config={"type": config.timeout_simulator_config.type},
        users_per_round=config.users_per_round,
        epochs=config.epochs,
        user_epochs_per_round=config.user_epochs_per_round,
        always_keep_trained_model=config.always_keep_trained_model,
        train_metrics_reported_per_epoch=config.train_metrics_reported_per_epoch,
        report_train_metrics=config.report_train_metrics,
        max_clip_norm_normalized=config.max_clip_norm_normalized,
        eval_epoch_frequency=config.eval_epoch_frequency,
        do_eval=config.do_eval,
        active_user_selector=config.active_user_selector,
        local_lr_scheduler={
            "type": config.local_lr_scheduler.type,
            "base_lr": config.local_lr_scheduler.base_lr,
        },
        report_train_metrics_after_aggregation=config.report_train_metrics_after_aggregation,
    )
    _compare_config_components(config, fl_config)
    return fl_config


def _compare_config_components(config: Any, fl_config: ConfigBase) -> None:
    """Print config components before and after translation"""
    print("HYDRA CONFIG")
    print("=" * 80)
    print(config.pretty())

    print("FL CONFIG")
    print("=" * 80)
    _print_fl_config(fl_config, 0)


def _print_fl_config(config: ConfigBase, level: int) -> None:
    for key, val in config.__dict__.items():
        if isinstance(val, ConfigBase):
            print("  " * level + f"{key}:")
            _print_fl_config(val, level + 1)
        else:
            print("  " * level + f"{key}: {val}")


@hydra.main(config_path="../configs/", config_name="sync_trainer_config")
def train(config: HydraSyncTrainerConfig) -> None:
    """The main entrypoint for yaml configuration data from hydra."""

    distributed_world_size = 1
    rank = 0
    FBProcessState.getInstance(rank=rank, fb_info=None)

    # Load MNIST data into custom Pytorch dataset class
    with tempfile.NamedTemporaryFile() as train_dataset_file, tempfile.NamedTemporaryFile() as test_dataset_file:

        with ManifoldClient.get_client(MANIFOLD_BUCKET) as client:
            client.sync_get(f"{TEST_DATASET_PATH}", test_dataset_file.name)
            client.sync_get(f"{TRAIN_DATASET_PATH}", train_dataset_file.name)

        train_dataset = MNISTDataset(
            train_dataset_file.name,
            train=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

        test_dataset = MNISTDataset(
            test_dataset_file.name,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

    # Instantiate Data Sharder to split dataset
    shard_size = 1000
    fl_data_sharder = FLDataSharder("sequential", None, None, None, shard_size)

    # Instantiate data loader
    local_batch_size = 100
    data_loader = FLDatasetDataLoaderWithBatch(
        train_dataset,
        test_dataset,
        test_dataset,
        fl_data_sharder,
        local_batch_size,
        local_batch_size,
        local_batch_size,
    )

    # override use_cuda_if_available as False if cuda is not available
    use_cuda_if_available = torch.cuda.is_available()

    # Create a predefined IFL nn model
    global_model = create_lighter_fl_model_for_mnist()
    if use_cuda_if_available:
        global_model.fl_cuda()

    # Custom metrics report specifies which channels to report metrics to
    # TODO [kychow][T71370034] Add Tensorboard logging metric visualization
    metrics_reporter = MNISTMetricsReporter([Channel.STDOUT])

    # Create SyncTrainer with config object instantiated from hydra input
    trainer = Configurable.create(
        cls_type=FLTrainerBase,
        config=_create_config_from_hydra(config),
        model=global_model,
        cuda_enabled=use_cuda_if_available,
    )

    # TODO len(data_loader.fl_train_set()) only counts users on a single worker
    # for distributed training. hence need to parse in the total number of users
    # from data_loader
    trainer.train(
        data_provider=FLDataProviderFromList(
            data_loader.fl_train_set(),
            data_loader.fl_eval_set(),
            data_loader.fl_test_set(),
            global_model,
        ),
        metric_reporter=metrics_reporter,
        num_total_users=data_loader.num_total_users,
        distributed_world_size=distributed_world_size,
        rank=rank,
    )


if __name__ == "__main__":
    # register HydraSyncTrainerConfig dataclass in ConfigStore for schema validation
    cs = ConfigStore.instance()
    cs.store(name="sync_trainer_config", node=HydraSyncTrainerConfig)

    # Set up pytorch training and environment parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=777)
    args = parser.parse_args()

    random_seed = args.random_seed

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train()
