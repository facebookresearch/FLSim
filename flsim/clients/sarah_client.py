#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file defines the concept of a SARAH clients
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from flsim.channels.base_channel import IdentityChannel
from flsim.clients.base_client import Client, ClientConfig
from flsim.common.timeout_simulator import TimeOutSimulator
from flsim.data.data_provider import IFLUserData
from flsim.interfaces.model import IFLModel
from flsim.privacy.common import PrivacyBudget, PrivacySetting
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.cuda import ICudaStateManager, DEFAULT_CUDA_MANAGER



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

    def generate_local_update(
        self, model: IFLModel, prev_model: IFLModel, round_number: int, metric_reporter: Optional[IFLMetricsReporter] = None
    ) -> Tuple[IFLModel, float]:
        
        if (round_number + 1) % self.cfg.large_cohort_period:
            local_model = self.receive_through_channel(model)
            # compute \nabla f_i^{\lamda, v} (w^{t})
            local_model, weight = self.train(
                local_model, optim, optim_scheduler, metric_reporter
            )
            # return w^t - \nabla f_i^{\lamda, v} (w^{t}) ?
            delta = self.compute_delta(
                before=model, after=local_model, model_to_save=local_model
            )
        else:
            local_model = self.receive_through_channel(model)
            prev_model = self.receive_through_channel(prev_model)

            local_model, optim, optim_scheduler = self.prepare_for_training(local_model)
            prev_model, prev_optim, prev_scheduler = self.prepare_for_training(prev_model)

            # compute \nabla f_i^{\lamda, v} (w^{t})
            local_model, weight = self.train(
                local_model, optim, optim_scheduler, metric_reporter
            )
            # compute \nabla f_i^{\lamda, v} (w^{t-1})
            prev_model, weight = self.train(
                prev_model, prev_optim, prev_scheduler, metric_reporter
            )
            # return difference \nabla f_i^{\lamda, v} (w^{t}) - \nabla f_i^{\lamda, v} (w^{t-1})
            delta = self.compute_delta(
                before=local_model, after=prev_model, model_to_save=local_model
            )
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

    def train(
        self,
        model: IFLModel,
        optimizer: Any,
        optimizer_scheduler: OptimizerScheduler,
        metric_reporter: Optional[IFLMetricsReporter] = None,
    ) -> Tuple[IFLModel, float]:
        total_samples = 0
        num_examples_processed = 0 
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
        optimizer.zero_grad()
        batch_metrics = model.fl_forward(training_batch)
        loss = batch_metrics.loss

        loss.backward()

        num_examples = batch_metrics.num_examples
        # adjust lr and take a step
        optimizer_scheduler.step(batch_metrics, model, training_batch, epoch)
        optimizer.step()

        if metric_reporter is not None:
            metric_reporter.add_batch_metrics(batch_metrics)
        return num_examples






@dataclass
class SarahClientConfig(ClientConfig):
    _target_: str = fullclassname(DPClient)
    epochs: int = 1  # No. of epochs for local training
    optimizer: LocalOptimizerConfig = LocalOptimizerConfig()
    large_cohort_period: int = 100