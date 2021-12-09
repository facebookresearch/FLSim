#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
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
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Iterable

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
    OptimizerScheduler,
    OptimizerSchedulerConfig,
    ConstantLRSchedulerConfig,
)
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.cuda import ICudaStateManager, DEFAULT_CUDA_MANAGER
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
        # pyre-fixme[16]: `Client` has no attribute `cfg`.
        self.cfg.lr_scheduler.base_lr = self.cfg.optimizer.lr
        self.per_example_training_time = (
            self.timeout_simulator.simulate_per_example_training_time()
        )
        self.ref_model = None
        self.num_samples = 0
        self.times_selected = 0
        self._tracked = {}
        self._last_updated_model = None
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
    def last_updated_model(self):
        """
        return the most recent model on the client
        """
        return self._last_updated_model

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
        report_metrics will be called ont the reporter, o.w. reports will be
        accumulated in memory.
        """
        # 1. pass through channel, set initial state
        updated_model = self.receive_through_channel(model)
        # 2. set up model and optimizer in the client
        updated_model, optim, optim_scheduler = self.prepare_for_training(updated_model)
        # 3. kick off training on client
        updated_model, weight = self.train(
            updated_model, optim, optim_scheduler, metric_reporter
        )
        # 4. Store updated model if being tracked
        if self.store_last_updated_model:
            self._last_updated_model = deepcopy(updated_model)
        # 5. compute delta
        delta = self.compute_delta(
            before=model, after=updated_model, model_to_save=updated_model
        )
        # 6. track state of the client
        self.track(delta=delta, weight=weight, optimizer=optim)
        return delta, weight

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
        Receives a reference to a state (refered to a model state_dict)
        over the channel. Any channel effect is applied as part of this
        receive function.
        """
        # keep a reference to global model
        self.ref_model = model

        # need to deepcopy the model because it's a reference to the global model
        # modifying model will moidify the global model
        message = self.channel.server_to_client(Message(model=deepcopy(model)))

        return message.model

    def prepare_for_training(
        self, model: IFLModel
    ) -> Tuple[IFLModel, torch.optim.Optimizer, OptimizerScheduler]:
        """
        1- instansiate a model with the given initial state
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
            self.per_example_training_time, self.dataset.num_examples()
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
    ) -> Tuple[IFLModel, float]:
        total_samples = 0
        # NOTE currently weight = total_sampls, this might be a bad strategy
        # plus there are privcay implications that must be taken into account.
        num_examples_processed = 0  # number of examples processed during training
        # pyre-fixme[16]: `Client` has no attribute `cfg`.
        for epoch in range(self.cfg.epochs):
            if self.stop_training(num_examples_processed):
                break

            # if user has too many examples and times-out, we want to process
            # different portion of the dataset each time
            dataset = list(iter(self.dataset))
            if self.cfg.shuffle_batch_order:
                random.shuffle(dataset)
            for batch in dataset:
                # TODO use an independent random generator
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
            f"Processed {num_examples_processed} of {self.dataset.num_examples()}"
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
                "delta": deepcopy(delta),
                "weight": weight,
                "optimizer": optimizer,
            }
        self.times_selected += 1

    def eval(
        self,
        model: Optional[IFLModel] = None,
        dataset: Optional[Iterable] = None,
        metric_reporter: Optional[IFLMetricsReporter] = None,
    ):
        """
        Evaluate the given `model` with the given `dataset`. `model` defaults
        to the current global_model if nothing is provided, and `dataset`
        defaults to client's dataset.
        """
        # Note here we play a trick, model and dataset provided are not
        # passed through a channel, in ptactice this needs to be done if
        # either the model or dataset is not what is currently being held
        # locally at the client.
        model = model or self.ref_model
        data = dataset or self.dataset
        self.cuda_state_manager.before_train_or_eval(model)
        with torch.no_grad():
            if self.seed is not None:
                torch.manual_seed(self.seed)

            model.fl_get_module().eval()
            for batch in data:
                batch_metrics = model.get_eval_metrics(batch)
                if metric_reporter is not None:
                    # TODO MM make sure metric reporter is multi-process safe.
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
        Compatible with the new tasks in which model is reponsible to
        arrange its inputs, targets and context.
        Return number of examples in the batch.
        """
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "L1-norm of Parameters of initial state:",
                sum(p.abs().sum() for p in self.ref_model.fl_get_module().parameters()),
            )
            self.logger.debug(
                "L1-norm of Parameters before step:",
                sum(p.abs().sum() for p in model.fl_get_module().parameters()),
            )

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

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "L1-norm of Parameters after step:",
                sum(p.abs().sum() for p in model.fl_get_module().parameters()),
            )
            l2_norm_raw = FLModelParamUtils.get_gradient_l2_norm_raw(
                model.fl_get_module()
            )
            l2_norm_normalized = FLModelParamUtils.get_gradient_l2_norm_normalized(
                model.fl_get_module()
            )
            self.logger.debug(
                f"Train Loss:{loss},"
                f"GradNorm:{l2_norm_raw},"
                f"NormalizedGradNorm:{l2_norm_normalized},",
                f"NumExamples:{num_examples}",
            )
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
    track_multiple_selection: bool = False  # track if client appears in 2+ rounds.
