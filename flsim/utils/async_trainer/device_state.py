#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum, auto

from flsim.utils.async_trainer.training_event_generator import IEventGenerator


class TrainingState(Enum):
    # Orderinig is important
    # For devices that have the same next_event_time(), we want devices that
    # "further along" in training to be chosen first
    # hence, TRAINING_FINISHED < TRAINING < WAITING_FOR_START
    TRAINING_FINISHED = auto()
    TRAINING = auto()
    WAITING_FOR_START = auto()

    # https://docs.python.org/3/library/enum.html#orderedenum
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


@dataclass
class TrainingSchedule:
    r"""
    Class to represent a client training time duration
    """
    creation_time: float
    start_time: float
    end_time: float


class TrainingScheduleFactory:
    @classmethod
    def create(
        cls,
        current_time: float,
        event_generator: IEventGenerator,
        num_examples: int,
    ):
        creation_time = current_time
        start_time = creation_time + event_generator.time_to_next_event_start()
        duration = event_generator.training_duration(num_examples)
        end_time = start_time + duration
        return TrainingSchedule(creation_time, start_time, end_time)


class DeviceState:
    r"""
    Represents the state of a device that's either waiting to start training,
    or in the middle of training
    """

    def __init__(self, training_schedule: TrainingSchedule):
        self.training_schedule: TrainingSchedule = training_schedule
        self.training_state: TrainingState = TrainingState.WAITING_FOR_START

    def get_training_state(self):
        return self.training_state

    # when we start training, we get initial model as input
    def training_started(self) -> None:
        self.training_state = TrainingState.TRAINING

    def training_ended(self) -> None:
        self.training_state = TrainingState.TRAINING_FINISHED

    def next_event_time(self):
        if self.training_state == TrainingState.WAITING_FOR_START:
            return self.training_schedule.start_time
        else:
            return self.training_schedule.end_time

    def __lt__(self, other):
        # if two device states have the same 'next_event_time', chose the one
        # that has 'smaller' training_state
        # smaller training state => further along in training
        if self.next_event_time() == other.next_event_time():
            return self.training_state < other.training_state
        return self.next_event_time() < other.next_event_time()
