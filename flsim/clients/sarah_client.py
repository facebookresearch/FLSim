#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file defines the concept of a SARAH client
"""
from __future__ import annotations

import random
from dataclasses import dataclass
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
from flsim.optimizers.optimizer_scheduler import (
    OptimizerScheduler,
)
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg, is_target
from flsim.utils.cuda import ICudaStateManager, DEFAULT_CUDA_MANAGER
from flsim.utils.fl.common import FLModelParamUtils


class SarahClient(Client):
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
            config_class=SarahClientConfig,
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
        self,
        model: IFLModel,
        prev_model: IFLModel,
        round_number: int,
        large_cohort_period: int,
        metric_reporter: Optional[IFLMetricsReporter] = None,
    ) -> Tuple[IFLModel, float]:

        if (round_number) % large_cohort_period == 0: # large co-hort
            local_model = self.receive_through_channel(model)
            local_model, optim, optim_scheduler = self.prepare_for_training(local_model)
            # w^{t}
            local_model, weight = self.train(
                local_model, optim, optim_scheduler, metric_reporter
            )
            # ( w^t - v_i^k )
            FLModelParamUtils.subtract_model(
                model.fl_get_module(),
                local_model.fl_get_module(),
                local_model.fl_get_module(),
            )
            # Multiply with lambda: \nabla f_i^{\lamda, v} (w^{t}) = \lambda * ( w^t - v_i^k )
            FLModelParamUtils.multiply_model_by_weight(
                local_model.fl_get_module(),
                weight=self.cfg.optimizer.lambda_,
                model_to_save=local_model.fl_get_module(),
            )
            return local_model, weight
        else:
            local_model = self.receive_through_channel(model)
            prev_local_model = self.receive_through_channel(prev_model)

            local_model, optim, optim_scheduler = self.prepare_for_training(local_model)
            prev_local_model, prev_optim, prev_scheduler = self.prepare_for_training(
                prev_local_model
            )
            # compute v_i^k based on w^t
            local_model, weight = self.train(
                local_model, optim, optim_scheduler, metric_reporter
            )
            # ( w^t - v_i^k )
            FLModelParamUtils.subtract_model(
                model.fl_get_module(),
                local_model.fl_get_module(),
                local_model.fl_get_module(),
            )
            # Multiply with lambda: \nabla f_i^{\lamda, v} (w^{t}) = \lambda * ( w^t - v_i^k )
            FLModelParamUtils.multiply_model_by_weight(
                local_model.fl_get_module(),
                weight=self.cfg.optimizer.lambda_,
                model_to_save=local_model.fl_get_module(),
            )

            # compute v_i^k based on w^{t - 1}
            prev_local_model, weight = self.train(
                prev_local_model, prev_optim, prev_scheduler, metric_reporter
            )
            # ( w^{t - 1} - v_i^k )
            FLModelParamUtils.subtract_model(
                prev_model.fl_get_module(),
                prev_local_model.fl_get_module(),
                prev_local_model.fl_get_module(),
            )
            # Multiply with lambda: \nabla f_i^{\lamda, v} (w^{t-1}) = \lambda * ( w^{t - 1} - v_i^k )
            FLModelParamUtils.multiply_model_by_weight(
                prev_local_model.fl_get_module(),
                weight=self.cfg.optimizer.lambda_,
                model_to_save=prev_local_model.fl_get_module(),
            )
            
            # local_model = self.compute_delta(
            #     before=local_model, after=prev_model, model_to_save=local_model
            # )
            # return difference \nabla f_i^{\lamda, v} (w^{t}) - \nabla f_i^{\lamda, v} (w^{t-1})
            FLModelParamUtils.subtract_model(
                local_model.fl_get_module(),
                prev_local_model.fl_get_module(),
                local_model.fl_get_module(),
            )
            return local_model, weight

    def train(
        self,
        model: IFLModel,
        optimizer: Any,
        optimizer_scheduler: OptimizerScheduler,
        metric_reporter: Optional[IFLMetricsReporter] = None,
        epoch: Optional[int] = None,
    ) -> Tuple[IFLModel, float]:
        total_samples = 0
        num_examples_processed = 0
        epoch = epoch if epoch is not None else self.cfg.epochs
        for epoch in range(self.cfg.epochs):

            dataset = list(iter(self.dataset))
            if self.cfg.shuffle_batch_order:
                random.shuffle(dataset)
            for batch in dataset:

                if self.seed is not None:
                    torch.manual_seed(self.seed)

                sample_count = self._batch_train(
                    model=model,
                    optimizer=optimizer,
                    training_batch=batch,
                    epoch=epoch,
                    metric_reporter=metric_reporter,
                    optimizer_scheduler=optimizer_scheduler,
                )
                total_samples += 0 if epoch else sample_count
                num_examples_processed += sample_count
        example_weight = min([num_examples_processed, total_samples])
        return model, float(example_weight)

    def _batch_train(
        self,
        model,
        optimizer,
        training_batch,
        epoch,
        metric_reporter,
        optimizer_scheduler,
    ) -> int:
        optimizer.zero_grad()
        batch_metrics = model.fl_forward(training_batch)
        loss = batch_metrics.loss

        loss.backward()

        num_examples = batch_metrics.num_examples
        # adjust lr and take a step
        optimizer_scheduler.step(batch_metrics, model, training_batch, epoch)
        loss, _ = optimizer.step()

        if metric_reporter is not None:
            metric_reporter.add_batch_metrics(batch_metrics)
        return num_examples


@dataclass
class SarahClientConfig(ClientConfig):
    _target_: str = fullclassname(SarahClient)
    epochs: int = 1  # No. of epochs for local training
    optimizer: LocalOptimizerConfig = LocalOptimizerProximalConfig()
