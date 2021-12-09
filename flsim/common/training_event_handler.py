#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import List, Tuple

import numpy as np
from flsim.clients.async_client import AsyncClientDevice
from flsim.utils.async_trainer.device_state import DeviceState, TrainingState


class IAsyncTrainingEventHandler(abc.ABC):
    r"""
    Interface class that performs asynchronous FL model training.

    Handles start training for a particular device state and end training for a particular device state.
    The AsyncTrainer should implement this class along with FLTrainerBase
    """

    @abc.abstractmethod
    def on_training_start(self, client: AsyncClientDevice) -> None:
        r"""
        Callback to notify that a client is starting training

        Marks the client state as training.

        Note:
        -----
        The actual training process doesn't start until on_training_end

        There's a choice of when should training ACTUALLY happen: when the
        training_start event fires, or when training_end event fires
        Answer: when training_end event fires. That's because training often
        produces side effects - like metric collection. So we should only
        train when we're sure that we want those side-effects
        """
        pass

    @abc.abstractmethod
    def on_training_end(self, client: AsyncClientDevice) -> None:
        r"""
        Callback to notify the trainer of a training end event.

        Trains local model for the client and incoporates it to the global model
        """
        pass

    @abc.abstractmethod
    def train_and_update_global_model(self, client: AsyncClientDevice) -> None:
        pass


class AsyncTrainingEventHandler(IAsyncTrainingEventHandler):
    def on_training_start(self, client: AsyncClientDevice) -> None:
        pass

    def on_training_end(self, client: AsyncClientDevice) -> None:
        r"""
        Trains the client's local model and update seqnum

        This function does the following:
            1. Changes training state to `TRAINING_FINISHED`
            2. Trains client's local model by calling `train_and_update_global_model`
        """
        client.training_ended()
        self.train_and_update_global_model(client=client)

    def train_and_update_global_model(self, client: AsyncClientDevice) -> None:
        pass


class TestAsyncTrainingEventHandler(AsyncTrainingEventHandler):
    r"""
    AsyncTrainingEventHandler that stores all model_states in a list.
    Useful for testing training simulation.
    """

    def __init__(self):
        super().__init__()
        self.training_events: List[Tuple(DeviceState, TrainingState)] = []
        self.num_unseen_global_model_updates: List[int] = []
        self.current_seqnum = 0

    def on_training_start(self, client: AsyncClientDevice) -> None:
        client.training_started(self.current_seqnum)
        self.training_events.append((client, TrainingState.TRAINING))

    def on_training_end(self, client: AsyncClientDevice) -> None:
        super().on_training_end(client)
        self.training_events.append((client, TrainingState.TRAINING_FINISHED))
        self.current_seqnum += 1

    def train_and_update_global_model(self, client: AsyncClientDevice) -> None:
        self.num_unseen_global_model_updates.append(
            self.current_seqnum - client.model_seqnum
        )

    def seqnum_diff_mean(self):
        return np.mean(self.num_unseen_global_model_updates)

    def seqnum_std(self):
        return np.std(self.num_unseen_global_model_updates)
