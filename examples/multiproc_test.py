#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import argparse
import os
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.fb.test.test_util as test_util
import torch.multiprocessing as mp
from flsim.baselines.utils import (
    GlobalOptimizerType,
    create_global_aggregator,
    fake_data,
)
from flsim.clients.base_client import ClientConfig
from flsim.clients.dp_client import DPClientConfig
from flsim.common.active_user_selectors.simple_user_selector import (
    SequentialActiveUserSelectorConfig,
)
from flsim.data.data_provider import FLDataProviderFromList
from flsim.optimizers.local_optimizers import LocalOptimizerSGDConfig
from flsim.privacy.common import PrivacySetting
from flsim.reducers.base_round_reducer import ReductionType
from flsim.reducers.dp_round_reducer import DPRoundReducerConfig
from flsim.tests import utils
from flsim.trainers.private_sync_trainer import (
    PrivateSyncTrainer,
    PrivateSyncTrainerConfig,
)
from flsim.trainers.sync_trainer import SyncTrainer, SyncTrainerConfig
from flsim.utils.distributed.fl_distributed import FLDistributedUtils
from omegaconf import OmegaConf


@dataclass
class MultiprocConfig:
    private: bool = True
    random_seed: int = 777
    epochs: int = 1
    world_size: int = 4
    global_optimizer_type: GlobalOptimizerType = GlobalOptimizerType.FEDAVG
    users_per_round = 16


def train(model, rank, settings) -> None:
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    world_size = settings.world_size
    epochs = settings.epochs
    users_per_round = settings.users_per_round
    global_optimizer_type = settings.global_optimizer_type
    global_optimizer_lr = 0.1
    global_optimizer_momentum = 0.9
    global_optimizer_beta1 = 0.9
    global_optimizer_beta2 = 0.99
    local_optimizer_lr = 0.1
    local_optimizer_momentum = 0.9
    private = settings.private

    train_set = [
        fake_data(i + 1, i + 2, device)
        for i in range(users_per_round)
        if i % world_size == rank
    ]

    always_keep_trained_model = False
    train_metrics_reported_per_epoch = 1
    report_train_metrics = True
    eval_epoch_frequency = 1
    use_cuda_if_available = False
    active_user_selector_config = SequentialActiveUserSelectorConfig()
    global_model = deepcopy(model)

    aggregator_config = create_global_aggregator(
        global_optimizer_type,
        global_optimizer_lr,
        global_optimizer_momentum,
        global_optimizer_beta1,
        global_optimizer_beta2,
    )

    sync_trainer = (
        SyncTrainer(
            model=global_model,
            cuda_enabled=use_cuda_if_available,
            **OmegaConf.structured(
                SyncTrainerConfig(
                    client=ClientConfig(
                        epochs=1,
                        optimizer=LocalOptimizerSGDConfig(
                            lr=local_optimizer_lr, momentum=local_optimizer_momentum
                        ),
                    ),
                    epochs=epochs,
                    aggregator=aggregator_config,
                    users_per_round=users_per_round,
                    always_keep_trained_model=always_keep_trained_model,
                    train_metrics_reported_per_epoch=train_metrics_reported_per_epoch,
                    report_train_metrics=report_train_metrics,
                    eval_epoch_frequency=eval_epoch_frequency,
                    do_eval=False,
                    active_user_selector=active_user_selector_config,
                    report_train_metrics_after_aggregation=True,
                )
            ),
        )
        if not private
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
                    do_eval=False,
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
                            noise_multiplier=0.0, clipping_value=0.1
                        ),
                    ),
                )
            ),
        )
    )

    FLDistributedUtils.distributed_training_on_cpu()
    sync_trainer.train(
        data_provider=FLDataProviderFromList(train_set, [], [], global_model),
        metric_reporter=utils.FakeMetricReporter(),
        num_total_users=users_per_round,
        distributed_world_size=world_size,
    )
    print(f"number of users for this process: {len(train_set)}")
    params = list(global_model.fl_get_module().parameters())
    print(
        f"sum mean params: {sum(p.mean() for p in params)} "
        f"sum abs params: {sum(p.abs().sum() for p in params)} "
    )


def init_process(model, rank, settings, train, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(test_util.find_free_port())
    dist.init_process_group(backend, rank=rank, world_size=settings.world_size)
    train(model, rank, settings)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--private", action="store_true", default=False)
    parser.add_argument("--random_seed", type=int, default=777)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=1)

    parser.add_argument("--users_per_round", type=int, default=8)

    parser.add_argument(
        "--global_optimizer_type",
        type=GlobalOptimizerType,
        choices=list(GlobalOptimizerType),
        help="Need to provide one of FedAVG, SGD, and Adam. Default is FedAVG.",
        default=GlobalOptimizerType.FEDAVG,
    )

    settings = parser.parse_args()

    torch.manual_seed(settings.random_seed)
    np.random.seed(settings.random_seed)
    torch.cuda.manual_seed_all(settings.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = utils.SampleNet(utils.TwoFC())
    FLDistributedUtils.WORLD_SIZE = settings.world_size
    processes = []
    for rank in range(settings.world_size):
        p = mp.Process(target=init_process, args=(model, rank, settings, train))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
