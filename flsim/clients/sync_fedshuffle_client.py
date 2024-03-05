# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""This class defines a synchronous Client for the FedShuffle framework.
Should be used in conjunction with the synchronous FedShuffle server
Works under assumption of equal number of epochs per client
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from flsim.interfaces.metrics_reporter import IFLMetricsReporter
from flsim.interfaces.model import IFLModel
from flsim.optimizers.optimizer_scheduler import OptimizerScheduler
from flsim.utils.config_utils import fullclassname
from flsim.utils.fl.common import FLModelParamUtils

from .base_client import Client, ClientConfig


class FedShuffleClient(Client):
    def copy_and_train_model(
        self,
        model: IFLModel,
        epochs: Optional[int] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizer_scheduler: Optional[OptimizerScheduler] = None,
        metrics_reporter: Optional[IFLMetricsReporter] = None,
    ) -> Tuple[IFLModel, float, torch.optim.Optimizer]:
        """Copy the source model to client-side model and use it to train on the
        client's train split.

        FedShuffle:
        Scale client learning rate as follows:
        client_lr = config_lr / num_train_examples
        Scale back lr after training

        Returns:
            (trained model, client's weight, optimizer used)
        """
        # 1. Pass source model through channel and use it to set client-side model state
        updated_model = self.receive_through_channel(model)
        # 2. Set up model and default optimizer in the client
        updated_model, default_optim, default_scheduler = self.prepare_for_training(
            updated_model
        )
        optim = default_optim if optimizer is None else optimizer
        optim_scheduler = (
            default_scheduler if optimizer_scheduler is None else optimizer_scheduler
        )

        # 3. Scale LR = LR / num_train_examples
        scaling_factor = self.dataset.num_train_examples()
        FLModelParamUtils.scale_optimizer_lr(optim, scaling_factor)

        # 4. Kick off training on client
        updated_model, weight = self.train(
            updated_model,
            optim,
            optim_scheduler,
            metrics_reporter=metrics_reporter,
            epochs=epochs,
        )

        # 5. Reverse the scaling
        # Used to prevent repeated scaling in case copy_and_train_model is called again on the same client
        FLModelParamUtils.scale_optimizer_lr(optim, 1.0 / scaling_factor)

        return updated_model, weight, optim


@dataclass
class FedShuffleClientConfig(ClientConfig):
    _target_: str = fullclassname(FedShuffleClient)
    shuffle_batch_order: bool = True  # FedShuffle specific
