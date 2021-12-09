#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Any, Optional, Tuple

from flsim.channels.base_channel import IdentityChannel
from flsim.clients.base_client import Client, ClientConfig
from flsim.common.timeout_simulator import TimeOutSimulator
from flsim.interfaces.model import IFLModel
from flsim.utils.async_trainer.async_user_selector import (
    AsyncUserSelector,
    AsyncUserSelectorInfo,
)
from flsim.utils.async_trainer.device_state import (
    DeviceState,
    TrainingSchedule,
    TrainingScheduleFactory,
    TrainingState,
)
from flsim.utils.async_trainer.training_event_generator import (
    IEventGenerator,
)
from flsim.utils.cuda import (
    ICudaStateManager,
    DEFAULT_CUDA_MANAGER,
)
from omegaconf import OmegaConf


class AsyncClientFactory:
    @classmethod
    def create(
        cls,
        current_time: float,
        event_generator: IEventGenerator,
        user_selector: AsyncUserSelector,
        client_config: ClientConfig,
        cuda_manager: ICudaStateManager = DEFAULT_CUDA_MANAGER,
        timeout_simulator: Optional[TimeOutSimulator] = None,
        channel: Optional[IdentityChannel] = None,
    ):
        user_info = user_selector.get_random_user()
        training_schedule = TrainingScheduleFactory.create(
            current_time, event_generator, user_info.user_data.num_examples()
        )
        client = Client(
            **OmegaConf.structured(client_config),
            dataset=user_info.user_data,
            name=f"client_{user_info.user_index}",
            timeout_simulator=timeout_simulator,
            channel=channel,
            cuda_manager=cuda_manager,
        )
        return AsyncClientDevice(training_schedule, client, user_info)


class AsyncClientDevice(DeviceState):
    r"""
    Class to represent a single async device. This class is responsible for
    maintaining the training state and training the local model
    """

    def __init__(
        self,
        training_schedule: TrainingSchedule,
        client: Client,
        user_info: AsyncUserSelectorInfo,
    ):
        self.client: Client = client
        self.local_model: IFLModel = None  # pyre-ignore[8]
        self.model_seqnum: int = -1
        self.user_info = user_info
        self.training_schedule = training_schedule
        super().__init__(training_schedule)

    def training_started(
        self, model_seqnum: int, init_model: Optional[IFLModel] = None
    ) -> None:
        r"""
        Starts the client training event by saving a copy of the current global model and seqnum
        """
        if init_model is not None:
            self.local_model = self.client.receive_through_channel(init_model)

        super().training_started()
        self.model_seqnum = model_seqnum

    def train_local_model(
        self, metric_reporter: Any = None
    ) -> Tuple[IFLModel, IFLModel, float]:
        r"""
        Performs local training loop
        """
        assert (
            self.local_model is not None
        ), "Client has not started training, local_model is None"
        # 1. Save the init model to compute delta
        before_train_local = deepcopy(self.local_model)
        # 2. Get ready for training
        self.local_model, optim, optim_scheduler = self.client.prepare_for_training(
            self.local_model
        )
        # 3. Train model on local data
        after_train_local, weight = self.client.train(
            self.local_model, optim, optim_scheduler, metric_reporter
        )
        # 3. Compute delta
        delta = self.client.compute_delta(
            before_train_local, after_train_local, model_to_save=before_train_local
        )
        # 4. Track client models if specified by config
        self.client.track(delta=delta, weight=weight, optimizer=optim)
        return delta, after_train_local, weight

    def is_waiting_to_start(self):
        return self.training_state == TrainingState.WAITING_FOR_START
