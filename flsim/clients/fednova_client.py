#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file defines the concept of a base client for a
federated learning setting. Also defines basic config,
for an FL client.

Note:
    This is just a base class and needs to be overridden
    for different use cases.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from flsim.channels.base_channel import IdentityChannel
from flsim.channels.message import FedNovaMessage
from flsim.clients.base_client import Client, ClientConfig
from flsim.common.logger import Logger
from flsim.common.timeout_simulator import (
    TimeOutSimulator,
)
from flsim.data.data_provider import IFLUserData
from flsim.interfaces.metrics_reporter import IFLMetricsReporter
from flsim.interfaces.model import IFLModel
from flsim.optimizers.local_optimizers import (
    LocalOptimizerSGDConfig,
)
from flsim.optimizers.optimizer_scheduler import (
    OptimizerScheduler,
    ConstantLRSchedulerConfig,
)
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.cuda import ICudaStateManager, DEFAULT_CUDA_MANAGER
from omegaconf import OmegaConf


class FedNovaClient(Client):
    logger: logging.Logger = Logger.get_logger(__name__)

    def __init__(
        self,
        *,
        dataset: IFLUserData,
        channel: Optional[IdentityChannel] = None,
        timeout_simulator: Optional[TimeOutSimulator] = None,
        store_last_updated_model: Optional[bool] = False,
        cuda_manager: ICudaStateManager = DEFAULT_CUDA_MANAGER,
        name: Optional[str] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=ClientConfig,
            **kwargs,
        )
        super().__init__(
            dataset=dataset,
            channel=channel,
            timeout_simulator=timeout_simulator,
            store_last_updated_model=store_last_updated_model,
            name=name,
            cuda_manager=cuda_manager,
            **kwargs,
        )
        self.num_steps = 0

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        if OmegaConf.is_missing(cfg.optimizer, "_target_"):
            cfg.optimizer = LocalOptimizerSGDConfig()
        if OmegaConf.is_missing(cfg.lr_scheduler, "_target_"):
            cfg.lr_scheduler = ConstantLRSchedulerConfig()

    def generate_local_update(
        self, model: IFLModel, metric_reporter: Optional[IFLMetricsReporter] = None
    ) -> FedNovaMessage:
        updated_model, num_samples, optimizer = self.copy_and_train_model(
            model=model, metric_reporter=metric_reporter
        )
        # 5. compute delta
        delta = self.compute_delta(
            before=model, after=updated_model, model_to_save=updated_model
        )
        # 6. track state of the client
        self.track(delta=delta, weight=num_samples, optimizer=optimizer)
        return FedNovaMessage(
            delta, num_local_steps=self.num_steps, num_examples=num_samples
        )

    def train(
        self,
        model: IFLModel,
        optimizer: Any,
        optimizer_scheduler: OptimizerScheduler,
        metric_reporter: Optional[IFLMetricsReporter] = None,
        epochs: Optional[int] = None,
    ) -> Tuple[IFLModel, float]:
        samples = 0
        epochs = epochs if epochs is not None else self.cfg.epochs
        if self.seed is not None:
            torch.manual_seed(self.seed)

        for epoch in range(epochs):
            dataset = list(self.dataset.train_data())
            if self.cfg.shuffle_batch_order:
                random.shuffle(dataset)
            for batch in dataset:
                samples += self._batch_train(
                    model=model,
                    optimizer=optimizer,
                    training_batch=batch,
                    epoch=epoch,
                    metric_reporter=metric_reporter,
                    optimizer_scheduler=optimizer_scheduler,
                )
                self.num_steps += 1
        # tell cuda manager we're done with training
        # cuda manager may move model out of GPU memory if needed
        self.cuda_state_manager.after_train_or_eval(model)
        return model, float(samples)


@dataclass
class FedNovaClientConfig(ClientConfig):
    _target_: str = fullclassname(FedNovaClient)
