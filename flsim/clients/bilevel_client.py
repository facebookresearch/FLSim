#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file defines the concept of a BiLevel client
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Optional, Tuple

import torch
from flsim.channels.base_channel import IdentityChannel
from flsim.clients.base_client import Client, ClientConfig
from flsim.common.timeout_simulator import (
    TimeOutSimulator,
)
from flsim.data.data_provider import IFLUserData
from flsim.interfaces.metrics_reporter import IFLMetricsReporter
from flsim.interfaces.model import IFLModel
from flsim.optimizers.local_optimizers import (
    LocalOptimizerProximalConfig,
    LocalOptimizerConfig,
)
from flsim.optimizers.optimizer_scheduler import OptimizerScheduler
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.cuda import ICudaStateManager, DEFAULT_CUDA_MANAGER
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate


class BiLevelClient(Client):
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
            config_class=BiLevelClientConfig,
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

    def generate_local_update(
        self, model: IFLModel, metric_reporter: Optional[IFLMetricsReporter] = None
    ) -> Tuple[IFLModel, float]:
        local_model = self.receive_through_channel(model) # global model; performs deepcopy inside

        # put model in train mode
        local_model.fl_get_module().train()
        # create optimizer
        # pyre-fixme[16]: `Client` has no attribute `cfg`.
        optimizer = instantiate(self.cfg.optimizer, model=local_model.fl_get_module())
        optimizer_scheduler = instantiate(self.cfg.lr_scheduler, optimizer=optimizer)

        local_model, weight = self.train(
            local_model, optimizer, optimizer_scheduler, metric_reporter
        )

        if self.store_last_updated_model:
            self.last_updated_model = deepcopy(local_model)
        # ( w^t - v_i^k )
        FLModelParamUtils.subtract_model(
            model.fl_get_module(),
            local_model.fl_get_module(),
            local_model.fl_get_module(),
        )
        # delta = \lambda * ( w^t - v_i^k )
        FLModelParamUtils.multiply_model_by_weight(
            local_model.fl_get_module(),
            weight=self.cfg.optimizer.lambda_,
            model_to_save=local_model.fl_get_module(),
        )
        # # send to server
        # for param in local_model.fl_get_module().parameters():
        #     print(self.name, "local params:", param[0][0][0])
        #     break
        return local_model, weight

    def train(
        self,
        model: IFLModel,
        optimizer: Any,
        optimizer_scheduler: OptimizerScheduler,
        metric_reporter: Optional[IFLMetricsReporter] = None,
        epoch: int = 1,
    ) -> Tuple[IFLModel, float]:

        total_samples = 0
        num_examples_processed = 0
        num_local_steps = 0
        if self.seed is not None:
            torch.manual_seed(self.seed)

        dataset = cycle(list(iter(self.dataset)))
        grad_norm_sq = float("inf")
        while num_local_steps < self.cfg.max_local_steps:
            if grad_norm_sq < self.cfg.target_local_acc:
                break
            batch = next(dataset)
            sample_count, grad_norm_sq = self.train_one_batch(
                model=model,
                optimizer=optimizer,
                training_batch=batch,
                metric_reporter=metric_reporter,
                optimizer_scheduler=optimizer_scheduler,
            )
            total_samples += sample_count
            num_examples_processed += sample_count
            num_local_steps += 1

        example_weight = min([num_examples_processed, total_samples])
        return model, float(example_weight)

    def train_one_batch(
        self,
        model,
        optimizer,
        training_batch,
        metric_reporter,
        optimizer_scheduler,
    ) -> Tuple[int, Any]:

        optimizer.zero_grad()
        batch_metrics = model.fl_forward(training_batch)
        loss = batch_metrics.loss

        loss.backward()

        num_examples = batch_metrics.num_examples
        # adjust lr and take a step
        optimizer_scheduler.step(batch_metrics, model, training_batch, 1)
        loss, grad_norm_sq = optimizer.step()

        if metric_reporter is not None:
            metric_reporter.add_batch_metrics(batch_metrics)
        return num_examples, grad_norm_sq


@dataclass
class BiLevelClientConfig(ClientConfig):
    _target_: str = fullclassname(BiLevelClient)
    epochs: int = 1  # No. of epochs for local training
    optimizer: LocalOptimizerConfig = LocalOptimizerProximalConfig()
    max_local_steps: int = 1
    target_local_acc: float = 1e-1
