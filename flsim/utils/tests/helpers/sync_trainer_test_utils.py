#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from flsim.active_user_selectors.simple_user_selector import (
    SequentialActiveUserSelectorConfig,
)
from flsim.channels.base_channel import FLChannelConfig
from flsim.clients.base_client import ClientConfig
from flsim.common.timeout_simulator import (
    NeverTimeOutSimulatorConfig,
    TimeOutSimulatorConfig,
)
from flsim.interfaces.model import IFLModel
from flsim.optimizers.local_optimizers import LocalOptimizerSGDConfig
from flsim.optimizers.optimizer_scheduler import (
    OptimizerSchedulerConfig,
    ConstantLRSchedulerConfig,
)
from flsim.servers.sync_servers import SyncServerConfig
from flsim.trainers.sync_trainer import SyncTrainer, SyncTrainerConfig
from omegaconf import OmegaConf

# have to create a variable because python linter doesn't like performing function calls
# in argument defaults (B008, https://github.com/PyCQA/flake8-bugbear#list-of-warnings)
NEVER_TIMEOUT_CONFIG = NeverTimeOutSimulatorConfig()
CONSTANT_LR_SCHEDULER_CONFIG = ConstantLRSchedulerConfig()
FED_AVG_SYNC_SERVER_CONFIG = SyncServerConfig(
    active_user_selector=SequentialActiveUserSelectorConfig()
)


def create_sync_trainer(
    model: IFLModel,
    local_lr: float,
    users_per_round: int,
    epochs: int,
    user_epochs_per_round: int = 1,
    do_eval: bool = True,
    server_config: SyncServerConfig = FED_AVG_SYNC_SERVER_CONFIG,
    timeout_simulator_config: TimeOutSimulatorConfig = NEVER_TIMEOUT_CONFIG,
    local_lr_scheduler: OptimizerSchedulerConfig = CONSTANT_LR_SCHEDULER_CONFIG,
    report_train_metrics: bool = False,
    dropout_rate: float = 1.0,
):
    # first disable report_train_metrics_after_aggregation. we will call
    # it outside of train() afterwise the post aggregation train metrics is
    # not returned
    sync_trainer = SyncTrainer(
        model=model,
        cuda_enabled=False,
        **OmegaConf.structured(
            SyncTrainerConfig(
                epochs=epochs,
                do_eval=do_eval,
                always_keep_trained_model=False,
                timeout_simulator=timeout_simulator_config,
                train_metrics_reported_per_epoch=1,
                eval_epoch_frequency=1,
                report_train_metrics=report_train_metrics,
                report_train_metrics_after_aggregation=False,
                client=ClientConfig(
                    epochs=user_epochs_per_round,
                    optimizer=LocalOptimizerSGDConfig(
                        lr=local_lr,
                    ),
                    lr_scheduler=local_lr_scheduler,
                    shuffle_batch_order=False,
                ),
                channel=FLChannelConfig(),
                server=server_config,
                users_per_round=users_per_round,
                dropout_rate=dropout_rate,
            )
        ),
    )
    return sync_trainer
