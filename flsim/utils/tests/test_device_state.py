#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from flsim.common.pytest_helper import (
    assertEqual,
    assertLess,
    assertGreater,
)
from flsim.utils.async_trainer.device_state import DeviceState, TrainingSchedule


class TestDeviceStateUtil:
    def test_next_event_time_update(self):
        """Check that next_event_time in DeviceState is updated correctly depending
        on its state
        """
        training_schedule = TrainingSchedule(
            creation_time=0, start_time=12, end_time=16
        )
        device_state = DeviceState(training_schedule)
        assertEqual(device_state.next_event_time(), training_schedule.start_time)
        device_state.training_started()
        assertEqual(device_state.next_event_time(), training_schedule.end_time)
        device_state.training_ended()
        assertEqual(device_state.next_event_time(), training_schedule.end_time)

    def test_device_next_event_time_comparison(self):
        """Check whether comparison operator for DeviceState acts as expected"""
        # create two devices, 1 & 2
        # device1 has current, training_start and training_end times
        # that are slightly before device2's respective times
        # 1. verify initial ordering
        # 2. advance training for device1 followed by device2,
        # verify ordering is as expected in every step
        training_schedule1 = TrainingSchedule(
            creation_time=0, start_time=12, end_time=16
        )
        device_state_1 = DeviceState(training_schedule1)
        training_schedule2 = TrainingSchedule(
            creation_time=0, start_time=12.1, end_time=16.1
        )
        device_state_2 = DeviceState(training_schedule2)
        assertLess(device_state_1, device_state_2)

        device_state_1.training_started()
        assertGreater(device_state_1, device_state_2)

        device_state_2.training_started()
        assertLess(device_state_1, device_state_2)

        device_state_1.training_ended()
        assertLess(device_state_1, device_state_2)

        device_state_2.training_ended()
        assertLess(device_state_1, device_state_2)

    def test_device_next_event_time_comparison_equality(self):
        """Check whether comparison operator for DeviceState acts as expected
        when next_event_time is equal
        """
        # create two devices, 1 & 2
        # device1.training_start = device2.current_time
        # device1.training_end = device2.training_schedule
        # 1. verify initial ordering
        # 2. advance device training
        # verify ordering is as expected in every step: when next_event_time is
        # equal, device that is 'further along' in training is 'lesser'
        training_schedule1 = TrainingSchedule(
            creation_time=0, start_time=12, end_time=16
        )
        device_state_1 = DeviceState(training_schedule1)
        training_schedule2 = TrainingSchedule(
            creation_time=0, start_time=16, end_time=20
        )
        device_state_2 = DeviceState(training_schedule2)
        assertLess(device_state_1, device_state_2)

        device_state_1.training_started()  # relevant time for both is 12
        # device that is further along in training is 'Less'
        assertLess(device_state_1, device_state_2)
        # verify that ordering is independent of which device_state is the
        # first parameter
        assertGreater(device_state_2, device_state_1)

        device_state_2.training_started()
        assertLess(device_state_1, device_state_2)

        device_state_1.training_ended()
        assertLess(device_state_1, device_state_2)

        device_state_2.training_ended()
        assertLess(device_state_1, device_state_2)
