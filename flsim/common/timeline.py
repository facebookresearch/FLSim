#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import NamedTuple


EPS = 1e-10


class Timeline(NamedTuple):
    """
    A point in time of the simulation.

    This contains epoch, round, and rounds_per_epoch or
    global_round and rounds_per_epoch. All of these values
    start from 1 by flsim convention. If any of these
    need not be set the default value of 0 is assinged, which
    helps us handle collision cases / initialization errors.

    Timeline usually refers to how many epochs/rounds have *finished*
    """

    epoch: int = 0
    round: int = 0
    global_round: int = 0
    rounds_per_epoch: int = 1

    def global_round_num(self) -> int:
        r"""
        Returns the global round number starting from 1.

        For example, if round is 2 and epoch is 10 and num_rounds_per_epoch is 10
        it will return 102.
        """
        assert (self.epoch and self.round) or self.global_round
        return (
            self.global_round or (self.epoch - 1) * self.rounds_per_epoch + self.round
        )

    def as_float(self, offset: int = 0) -> float:
        r"""
        prints the time-line as a floating number.

        E.g. if round is 2 and epoch is 10 and num_rounds_per_epoch is 10
        it will return 10.2. By default uses the round value of the object
        but can also show a value for a run that is offset away from the
        current round.
        """
        return (self.global_round_num() + offset) / self.rounds_per_epoch

    def tick(self, tick_interval: float) -> bool:
        r"""
        Returns true 'tick_interval' times every epoch
        E.g: if tick_interval is 10, it returns true 10 times every epoch
        This function is useful for deciding when to report train/eval results
        """
        return (self.as_float() + EPS) // tick_interval > (
            self.as_float(-1) + EPS
        ) // tick_interval

    def __str__(self):
        assert (self.epoch > 0 and self.round > 0) or self.global_round > 0
        e = self.epoch or ((self.global_round - 1) // self.rounds_per_epoch + 1)
        r = self.round or (((self.global_round - 1) % self.rounds_per_epoch) + 1)
        gr = self.global_round or ((e - 1) * self.rounds_per_epoch + r)
        return f"(epoch = {e}, round = {r}, global round = {gr})"
