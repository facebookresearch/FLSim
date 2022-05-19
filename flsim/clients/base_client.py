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
from typing import Any, List, Optional, Tuple

import torch
from flsim.channels.base_channel import IdentityChannel
from flsim.channels.message import Message
from flsim.common.logger import Logger
from flsim.common.timeout_simulator import (
    NeverTimeOutSimulator,
    NeverTimeOutSimulatorConfig,
    TimeOutSimulator,
)
from flsim.data.data_provider import IFLUserData
from flsim.interfaces.metrics_reporter import IFLMetricsReporter
from flsim.interfaces.model import IFLModel
from flsim.optimizers.local_optimizers import (
    LocalOptimizerConfig,
    LocalOptimizerSGDConfig,
)
from flsim.optimizers.optimizer_scheduler import (
    ConstantLRSchedulerConfig,
    OptimizerScheduler,
    OptimizerSchedulerConfig,
)
from flsim.utils.config_utils import fullclassname, init_self_cfg
from flsim.utils.cuda import DEFAULT_CUDA_MANAGER, ICudaStateManager
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate
from omegaconf import OmegaConf


class Client:
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

        self.dataset = dataset
        self.cuda_state_manager = cuda_manager
        self.channel = channel or IdentityChannel()
        self.timeout_simulator = timeout_simulator or NeverTimeOutSimulator(
            **OmegaConf.structured(NeverTimeOutSimulatorConfig())
        )
        self.store_last_updated_model = store_last_updated_model
        self.name = name or "unnamed_client"

        # base lr needs to match LR in optimizer config, overwrite it
        # pyre-ignore [16]
        self.cfg.lr_scheduler.base_lr = self.cfg.optimizer.lr
        self.per_example_training_time = (
            self.timeout_simulator.simulate_per_example_training_time()
        )
        self.ref_model = None
        self.num_samples = 0
        self.times_selected = 0
        self._tracked = {}
        self.last_updated_model = None
        self.logger.setLevel(logging.INFO)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        if OmegaConf.is_missing(cfg.optimizer, "_target_"):
            cfg.optimizer = LocalOptimizerSGDConfig()
        if OmegaConf.is_missing(cfg.lr_scheduler, "_target_"):
            cfg.lr_scheduler = ConstantLRSchedulerConfig()

    @property
    def seed(self) -> Optional[int]:
        """if should set random_seed or not."""
        # pyre-fixme[16]: `Client` has no attribute `cfg`.
        return self.cfg.random_seed

    @property
    def model_deltas(self) -> List[IFLModel]:
        """
        return the stored deltas for all rounds that
        this user was selected.
        """
        return [self._tracked[s]["delta"] for s in range(self.times_selected)]

    @property
    def optimizers(self) -> List[Any]:
        """Look at {self.model}"""
        return [self._tracked[s]["optimizer"] for s in range(self.times_selected)]

    @property
    def weights(self) -> List[float]:
        """Look at {self.model}"""
        return [self._tracked[s]["weight"] for s in range(self.times_selected)]

    def generate_local_update(
        self, model: IFLModel, metric_reporter: Optional[IFLMetricsReporter] = None
    ) -> Tuple[IFLModel, float]:
        r"""
        wrapper around all functions called on a client for generating an
        updated local model.

        Note:
        -----
        Only pass a ``metric_reporter`` if reporting is needed, i.e.
        report_metrics will be called on the reporter, o.w. reports will be
        accumulated in memory.
        """
        updated_model, weight, optimizer = self.copy_and_train_model(
            model, metric_reporter=metric_reporter
        )
        # 4. Store updated model if being tracked
        if self.store_last_updated_model:
            self.last_updated_model = FLModelParamUtils.clone(updated_model)
        # 5. compute delta
        delta = self.compute_delta(
            before=model, after=updated_model, model_to_save=updated_model
        )
        # 6. track state of the client
        self.track(delta=delta, weight=weight, optimizer=optimizer)
        return delta, weight

    def copy_and_train_model(
        self,
        model: IFLModel,
        epochs: Optional[int] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizer_scheduler: Optional[OptimizerScheduler] = None,
        metric_reporter: Optional[IFLMetricsReporter] = None,
    ) -> Tuple[IFLModel, float, torch.optim.Optimizer]:
        """Copy the model then use that model to train on the client's train split

        Note: Optional optimizer and optimizer_scheduler are there for easier testing

        Returns:
            Tuple[IFLModel, float, torch.optim.Optimizer]: The trained model, the client's weight, the optimizer used
        """
        # 1. pass through channel, set initial state
        updated_model = self.receive_through_channel(model)
        # 2. set up model and optimizer in the client
        updated_model, default_optim, default_scheduler = self.prepare_for_training(
            updated_model
        )
        optim = default_optim if optimizer is None else optimizer
        optim_scheduler = (
            default_scheduler if optimizer_scheduler is None else optimizer_scheduler
        )

        # 3. kick off training on client
        updated_model, weight = self.train(
            updated_model,
            optim,
            optim_scheduler,
            metric_reporter=metric_reporter,
            epochs=epochs,
        )
        return updated_model, weight, optim

    def compute_delta(
        self, before: IFLModel, after: IFLModel, model_to_save: IFLModel
    ) -> IFLModel:
        """
        Computes the delta between the before training and after training model
        """
        FLModelParamUtils.subtract_model(
            minuend=before.fl_get_module(),
            subtrahend=after.fl_get_module(),
            difference=model_to_save.fl_get_module(),
        )
        return model_to_save

    def receive_through_channel(self, model: IFLModel) -> IFLModel:
        """
        Receives a reference to a state (referred to as model state_dict)
        over the channel. Any channel effect is applied as part of this
        receive function.
        """
        # keep a reference to global model
        self.ref_model = model

        # need to clone the model because it's a reference to the global model
        # modifying model will modify the global model
        message = self.channel.server_to_client(
            Message(model=FLModelParamUtils.clone(model))
        )

        return message.model

    def prepare_for_training(
        self, model: IFLModel
    ) -> Tuple[IFLModel, torch.optim.Optimizer, OptimizerScheduler]:
        """
        1- instantiate a model with the given initial state
        2- create an optimizer
        """
        # inform cuda_state_manager that we're about to train a model
        # it may move model to GPU
        self.cuda_state_manager.before_train_or_eval(model)
        # put model in train mode
        model.fl_get_module().train()
        # create optimizer
        # pyre-fixme[16]: `Client` has no attribute `cfg`.
        optimizer = instantiate(self.cfg.optimizer, model=model.fl_get_module())
        optimizer_scheduler = instantiate(self.cfg.lr_scheduler, optimizer=optimizer)
        return model, optimizer, optimizer_scheduler

    def get_total_training_time(self) -> float:
        return self.timeout_simulator.simulate_training_time(
            self.per_example_training_time, self.dataset.num_train_examples()
        )

    def stop_training(self, num_examples_processed) -> bool:
        training_time = self.timeout_simulator.simulate_training_time(
            self.per_example_training_time, num_examples_processed
        )
        return self.timeout_simulator.user_timeout(training_time)

    def train(
        self,
        model: IFLModel,
        optimizer: Any,
        optimizer_scheduler: OptimizerScheduler,
        metric_reporter: Optional[IFLMetricsReporter] = None,
        epochs: Optional[int] = None,
    ) -> Tuple[IFLModel, float]:
        total_samples = 0
        # NOTE currently weight = total_sampls, this might be a bad strategy
        # plus there are privacy implications that must be taken into account.
        num_examples_processed = 0  # number of examples processed during training
        # pyre-ignore[16]:
        epochs = epochs if epochs is not None else self.cfg.epochs
        if self.seed is not None:
            torch.manual_seed(self.seed)

        for epoch in range(epochs):
            if self.stop_training(num_examples_processed):
                break

            # if user has too many examples and times-out, we want to process
            # different portion of the dataset each time
            dataset = list(self.dataset.train_data())
            if self.cfg.shuffle_batch_order:
                random.shuffle(dataset)
            for batch in dataset:
                sample_count = self._batch_train(
                    model=model,
                    optimizer=optimizer,
                    training_batch=batch,
                    epoch=epoch,
                    metric_reporter=metric_reporter,
                    optimizer_scheduler=optimizer_scheduler,
                )
                self.post_batch_train(epoch, model, sample_count, optimizer)
                total_samples += 0 if epoch else sample_count
                num_examples_processed += sample_count
                # stop training depending on time-out condition
                if self.stop_training(num_examples_processed):
                    break
        # tell cuda manager we're done with training
        # cuda manager may move model out of GPU memory if needed
        self.cuda_state_manager.after_train_or_eval(model)
        self.logger.debug(
            f"Processed {num_examples_processed} of {self.dataset.num_train_examples()}"
        )
        self.post_train(model, total_samples, optimizer)
        # if training stops early, used partial training weight
        example_weight = min([num_examples_processed, total_samples])
        return model, float(example_weight)

    def post_train(self, model: IFLModel, total_samples: int, optimizer: Any):
        pass

    def post_batch_train(
        self, epoch: int, model: IFLModel, sample_count: int, optimizer: Any
    ):
        pass

    def track(self, delta: IFLModel, weight: float, optimizer: Any):
        """Tracks metric when the client is selected multiple times"""
        # pyre-fixme[16]: `Client` has no attribute `cfg`.
        if self.cfg.store_models_and_optimizers:
            self._tracked[self.times_selected] = {
                "delta": FLModelParamUtils.clone(delta),
                "weight": weight,
                "optimizer": optimizer,
            }
        self.times_selected += 1

    def eval(
        self,
        model: IFLModel,
        dataset: Optional[IFLUserData] = None,
        metric_reporter: Optional[IFLMetricsReporter] = None,
    ):
        """
        Client evaluates the model based on its evaluation data split
        """
        data = dataset or self.dataset
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

    def _batch_train(
        self,
        model,
        optimizer,
        training_batch,
        epoch,
        metric_reporter,
        optimizer_scheduler,
    ) -> int:
        """Trainer for NewDocModel based FL Tasks
        Run a single iteration of minibatch-gradient descent on a single user.
        Compatible with the new tasks in which the model is responsible for
        arranging its inputs, targets and context.
        Return number of examples in the batch.
        """
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
class ClientConfig:
    _target_: str = fullclassname(Client)
    _recursive_: bool = False
    epochs: int = 1  # No. of epochs for local training
    optimizer: LocalOptimizerConfig = LocalOptimizerConfig()
    lr_scheduler: OptimizerSchedulerConfig = OptimizerSchedulerConfig()
    max_clip_norm_normalized: Optional[float] = None  # gradient clip value
    only_federated_params: bool = True  # flag to only use certain params
    random_seed: Optional[int] = None  # random seed for deterministic response
    shuffle_batch_order: bool = False  # shuffle the ordering of batches
    store_models_and_optimizers: bool = False  # name clear
    track_multiple_selection: bool = False  # track if the client appears in 2+ rounds.
