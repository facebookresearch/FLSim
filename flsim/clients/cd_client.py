#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Algorithm 6:
    Client
    Randomized coordinate descent for personalized FL (without variance reduction
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Any

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


class CDClient(Client):
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
            config_class=CDClientConfig,
            **kwargs,
        )
        super().__init__(
            dataset=dataset,
            channel=channel,
            timeout_simulator=timeout_simulator,
            store_last_updated_model=True,
            name=name,
            **kwargs,
        )
        self.global_model = None
        self.optimizer = None
        self.optimizer_scheduler = None
        self.local_model = None

    def generate_local_update(
        self, model: IFLModel, metric_reporter: Optional[IFLMetricsReporter] = None
    ) -> Tuple[IFLModel, float]:
        self.global_model = self.receive_through_channel(model)
        first_time_sampled = True if self.local_model is None else False

        # commented out for debug
        if self.local_model is None:
            self.local_model = deepcopy(self.global_model)
        # this line below is for debug
        # self.local_model = deepcopy(self.global_model)

        # save local model before training:
        model_before_training = deepcopy(self.local_model)

        # put model in train mode
        self.local_model.fl_get_module().train()
        # use the same optimizer every time, to preserve state
        # if self.optimizer is None:
        self.optimizer = instantiate(
            self.cfg.optimizer, model=self.local_model.fl_get_module()
        )
        self.optimizer_scheduler = instantiate(
            self.cfg.lr_scheduler, optimizer=self.optimizer
        )
        # else:
            # update global model
        self.optimizer.set_new_global_model(self.global_model.fl_get_module())

        self.local_model, weight = self.train(
            self.local_model, self.optimizer, self.optimizer_scheduler, metric_reporter
        )
        # for param in local_model.fl_get_module().parameters():
        #     print(self.name, "local params:", param[0][0][0])
        #     break
        # TODO check: self.local_model.fl_get_module() == local_model.fl_get_module()
        delta = deepcopy(self.local_model)
        if not first_time_sampled:
            FLModelParamUtils.subtract_model(
                self.local_model.fl_get_module(),
                model_before_training.fl_get_module(),
                delta.fl_get_module(),
            )
        # TODO check: self.local_model.fl_get_module() != local_model.fl_get_module()
        # send v_i^{t + 1} - v_i^t (v_i^t == 0 if sampled the first time)
        return delta, weight

    def eval(
        self,
        model: IFLModel,
        dataset: Optional[IFLUserData] = None,
        metric_reporter: Optional[IFLMetricsReporter] = None,
        fine_tune: bool = False,
        personalized_epoch: int = 1,
    ):
        """
        Evaluate the given `model` with the given `dataset`. `model` defaults
        to the current global_model if nothing is provided, and `dataset`
        defaults to client's dataset.
        """
        if fine_tune:
            model = self.local_model if self.local_model is not None else model
            model = self.receive_through_channel(model)
            model, optim, optim_scheduler = self.prepare_for_training(model)
            personalized_epoch = self.cfg.eval_personalized_epoch if self.local_model is None else 1

            model, _ = self.train(
                model=model,
                optimizer=optim,
                optimizer_scheduler=optim_scheduler,
                metric_reporter=metric_reporter,
                epoch=personalized_epoch,
            )
        else:
            model = self.local_model

        data = self.dataset
        self.cuda_state_manager.before_train_or_eval(model)
        with torch.no_grad():
            if self.seed is not None:
                torch.manual_seed(self.seed)

            model.fl_get_module().eval()
            for batch in data.eval_data():
                batch_metrics = model.get_eval_metrics(batch)
                if metric_reporter is not None:
                    metric_reporter.add_batch_metrics(batch_metrics)
        model.fl_get_module().train()
        self.cuda_state_manager.after_train_or_eval(model)

    def train(
        self,
        model: IFLModel,
        optimizer: Any,
        optimizer_scheduler: OptimizerScheduler,
        metric_reporter: Optional[IFLMetricsReporter] = None,
        epoch: int = 1,
    ) -> Tuple[IFLModel, float]:
        # on finetuning, global model may be none, so save a copy
        if self.global_model is None:
            self.global_model = deepcopy(model)
        # calling base client train
        return super().train(
            model, optimizer, optimizer_scheduler, metric_reporter, epoch
        )

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
        # pyre-fixme[16]: `Client` has no attribute `cfg`.
        if self.cfg.max_clip_norm_normalized is not None:
            max_norm = self.cfg.max_clip_norm_normalized
            FLModelParamUtils.clip_gradients(
                max_normalized_l2_norm=max_norm, model=model.fl_get_module()
            )

        num_examples = batch_metrics.num_examples
        # adjust lr and take a step
        optimizer_scheduler.step(batch_metrics, model, training_batch, epoch)

        optimizer.step()

        if metric_reporter is not None:
            metric_reporter.add_batch_metrics(batch_metrics)

        return num_examples


@dataclass
class CDClientConfig(ClientConfig):
    _target_: str = fullclassname(CDClient)
    epochs: int = 1  # No. of epochs for local training
    optimizer: LocalOptimizerConfig = LocalOptimizerProximalConfig()
    eval_personalized_epoch: int = 1
