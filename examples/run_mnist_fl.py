#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import argparse
from argparse import Namespace
from typing import List, Optional

import numpy as np
import torch
from flsim.active_user_selectors.simple_user_selector import (
    SequentialActiveUserSelectorConfig,
)
from flsim.baselines.utils import GlobalOptimizerType, create_global_aggregator
from flsim.clients.base_client import ClientConfig
from flsim.clients.dp_client import DPClientConfig
from flsim.data.data_provider import FLDataProviderFromList
from flsim.data.data_sharder import FLDataSharder
from flsim.data.dataset_data_loader import FLDatasetDataLoaderWithBatch
from flsim.examples.mnist_fl_dataset import MNISTDataset
from flsim.examples.mnist_fl_metrics_reporter import MNISTMetricsReporter
from flsim.examples.mnist_fl_model import create_lighter_fl_model_for_mnist
from flsim.interfaces.metrics_reporter import Channel
from flsim.optimizers.local_optimizers import LocalOptimizerSGDConfig
from flsim.privacy.common import PrivacySetting
from flsim.reducers.base_round_reducer import ReductionType
from flsim.reducers.dp_round_reducer import DPRoundReducerConfig
from flsim.trainers.private_sync_trainer import (
    PrivateSyncTrainer,
    PrivateSyncTrainerConfig,
)
from flsim.trainers.sync_trainer import SyncTrainer, SyncTrainerConfig
from omegaconf import OmegaConf
from torchvision import transforms

# Refer to N180779 to download and process the MNIST dataset
# You need to have a proxy connection setup along with PyTorch installed
# If that's not the case, follow steps 1 & 2 at https://fburl.com/wiki/9ct3chzy
TRAINING_DATASET_PATH = "/tmp/mnist/processed/training.pt"
TEST_DATASET_PATH = "/tmp/mnist/processed/test.pt"


def train(
    epochs: int,
    user_epochs_per_round: int,
    users_per_round: int,
    local_batch_size: int,
    global_optimizer_type: GlobalOptimizerType,
    global_optimizer_lr: float,
    global_optimizer_momentum: float,
    global_optimizer_beta1: float,
    global_optimizer_beta2: float,
    local_optimizer_lr: float,
    local_optimizer_momentum: float,
    report_channel: Channel,
    use_cuda_if_available: bool,
    use_example_dp: bool,
    noise_multiplier: float = 0.0,
    clipping_value: Optional[float] = None,
    delta: float = 1e-6,
    alphas: Optional[List[float]] = None,
) -> None:
    train_dataset = MNISTDataset(
        TRAINING_DATASET_PATH,
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    test_dataset = MNISTDataset(
        TEST_DATASET_PATH,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    shard_size = 1000
    fl_data_sharder = FLDataSharder("sequential", None, None, None, shard_size)

    data_loader = FLDatasetDataLoaderWithBatch(
        train_dataset,
        test_dataset,
        test_dataset,
        fl_data_sharder,
        # will revisit the proper batch size later when we revisit this flow
        # for flow testing, baselines, and etc
        # In the meantime, we will keep the previous behavior as-is.
        local_batch_size,
        local_batch_size,
        local_batch_size,
    )

    always_keep_trained_model = False
    train_metrics_reported_per_epoch = 1
    report_train_metrics = True
    eval_epoch_frequency = 1
    do_eval = True
    active_user_selector_config = SequentialActiveUserSelectorConfig()

    global_model = create_lighter_fl_model_for_mnist()
    if use_cuda_if_available:
        global_model.fl_cuda()

    aggregator_config = create_global_aggregator(
        global_optimizer_type,
        global_optimizer_lr,
        global_optimizer_momentum,
        global_optimizer_beta1,
        global_optimizer_beta2,
    )

    metrics_reporter = MNISTMetricsReporter([report_channel])

    sync_trainer = (
        SyncTrainer(
            model=global_model,
            cuda_enabled=use_cuda_if_available,
            **OmegaConf.structured(
                SyncTrainerConfig(
                    client=ClientConfig(
                        epochs=user_epochs_per_round,
                        optimizer=LocalOptimizerSGDConfig(
                            lr=local_optimizer_lr, momentum=local_optimizer_momentum
                        ),
                    ),
                    epochs=epochs,
                    users_per_round=users_per_round,
                    aggregator=aggregator_config,
                    always_keep_trained_model=always_keep_trained_model,
                    train_metrics_reported_per_epoch=train_metrics_reported_per_epoch,
                    report_train_metrics=report_train_metrics,
                    eval_epoch_frequency=eval_epoch_frequency,
                    do_eval=do_eval,
                    active_user_selector=active_user_selector_config,
                    report_train_metrics_after_aggregation=True,
                )
            ),
        )
        if not use_example_dp
        else PrivateSyncTrainer(
            model=global_model,
            cuda_enabled=use_cuda_if_available,
            **OmegaConf.structured(
                PrivateSyncTrainerConfig(
                    aggregator=aggregator_config,
                    users_per_round=users_per_round,
                    epochs=epochs,
                    always_keep_trained_model=always_keep_trained_model,
                    train_metrics_reported_per_epoch=train_metrics_reported_per_epoch,
                    report_train_metrics=report_train_metrics,
                    eval_epoch_frequency=eval_epoch_frequency,
                    do_eval=do_eval,
                    active_user_selector=active_user_selector_config,
                    report_train_metrics_after_aggregation=True,
                    client=DPClientConfig(
                        epochs=1,
                        privacy_setting=PrivacySetting(
                            noise_multiplier=0.0, clipping_value=float("inf")
                        ),
                        optimizer=LocalOptimizerSGDConfig(
                            lr=local_optimizer_lr, momentum=local_optimizer_momentum
                        ),
                    ),
                    reducer=DPRoundReducerConfig(
                        reduction_type=ReductionType.AVERAGE,
                        privacy_setting=PrivacySetting(
                            noise_multiplier=noise_multiplier,
                            clipping_value=clipping_value
                            if clipping_value is not None
                            else float("inf"),
                            alphas=alphas
                            if alphas is not None
                            else [1 + x / 10.0 for x in range(1, 100)]
                            + [float(i) for i in range(12, 64)],
                            target_delta=delta,
                        ),
                    ),
                )
            ),
        )
    )

    # TODO len(data_loader.fl_train_set()) only counts users on a single worker
    # for distributed training. hence need to parse in the total number of users
    # from data_loader
    train_set = data_loader.fl_train_set()
    # pyre-fixme[6]
    num_train_samples = len(train_set)
    sync_trainer.train(
        data_provider=FLDataProviderFromList(
            train_set,
            data_loader.fl_eval_set(),
            data_loader.fl_test_set(),
            global_model,
        ),
        metric_reporter=metrics_reporter,
        num_total_users=num_train_samples,
        distributed_world_size=1,
    )


def _pretty_print_hyperparameters(args: Namespace):
    hyperparameters_dict = {}
    for key, val in args.__dict__.items():
        hyperparameters_dict[key] = val
    if args.global_optimizer_type == GlobalOptimizerType.SGD:
        del hyperparameters_dict["global_optimizer_beta1"]
        del hyperparameters_dict["global_optimizer_beta2"]
    elif args.global_optimizer_type == GlobalOptimizerType.ADAM:
        del hyperparameters_dict["global_optimizer_momentum"]
    elif args.global_optimizer_type == GlobalOptimizerType.LARS:
        del hyperparameters_dict["global_optimizer_momentum"]
        del hyperparameters_dict["global_optimizer_beta1"]
    elif args.global_optimizer_type == GlobalOptimizerType.LAMB:
        del hyperparameters_dict["global_optimizer_momentum"]
    elif args.global_optimizer_type == GlobalOptimizerType.FEDAVG:
        del hyperparameters_dict["global_optimizer_momentum"]
        del hyperparameters_dict["global_optimizer_beta1"]
        del hyperparameters_dict["global_optimizer_beta2"]
    else:
        raise TypeError("Shouldn't reach here.")

    print("=" * 80)
    print("Hyperparameters provided...")
    print(hyperparameters_dict)
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--random_seed", type=int, default=777)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--user_epochs_per_round", type=int, default=5)
    parser.add_argument("--users_per_round", type=int, default=10)
    parser.add_argument("--local_batch_size", type=int, default=100)

    parser.add_argument(
        "--global_optimizer_type",
        type=GlobalOptimizerType,
        choices=list(GlobalOptimizerType),
        help="Need to provide one of FedAVG, SGD, and Adam. Default is FedAVG.",
        default=GlobalOptimizerType.FEDAVG,
    )
    parser.add_argument("--global_optimizer_lr", type=float)

    parser.add_argument("--global_optimizer_momentum", type=float, default=0.9)
    parser.add_argument("--global_optimizer_beta1", type=float, default=0.9)
    parser.add_argument("--global_optimizer_beta2", type=float, default=0.99)

    parser.add_argument("--local_optimizer_lr", type=float, default=0.1)
    parser.add_argument("--local_optimizer_momentum", type=float, default=0.0)

    parser.add_argument("--report_channel", type=str, default=Channel.STDOUT)
    parser.add_argument(
        "--use_example_dp",
        action="store_true",
        default=False,
        help="Enable private training with example level DP instead of just training with vanilla SGD",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-6,
        metavar="D",
        help="Target delta (default: 1e-6)",
    )
    parser.add_argument(
        "--clipping_value",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "--use_cuda_if_available",
        action="store_true",
        default=False,
        help="Whether to use cuda if GPUs are available",
    )
    parser.add_argument(
        "--alphas",
        type=List[float],
        nargs="*",
        default=[1 + x / 10.0 for x in range(1, 100)]
        + [float(i) for i in range(12, 64)],
        help="Space separate list of alpha values used in Renyi differential privacy",
    )
    args = parser.parse_args()
    if args.global_optimizer_type != GlobalOptimizerType.FEDAVG:
        assert (
            args.global_optimizer_lr is not None
        ), "--global_optimizer_lr should be provided."

    _pretty_print_hyperparameters(args)

    random_seed = args.random_seed

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train(
        args.epochs,
        args.user_epochs_per_round,
        args.users_per_round,
        args.local_batch_size,
        args.global_optimizer_type,
        args.global_optimizer_lr,
        args.global_optimizer_momentum,
        args.global_optimizer_beta1,
        args.global_optimizer_beta2,
        args.local_optimizer_lr,
        args.local_optimizer_momentum,
        args.report_channel,
        args.use_cuda_if_available,
        args.use_example_dp,
        args.noise_multiplier,
        args.clipping_value,
        args.delta,
        args.alphas,
    )
