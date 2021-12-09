#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from flsim.common.pytest_helper import (
    assertEqual,
    assertAlmostEqual,
    assertFalse,
    assertTrue,
)
from flsim.common.timeline import Timeline


class TestTimeline:
    def test_global_round_num(self):
        # test first round
        tl = Timeline(epoch=1, round=1)
        assertEqual(tl.global_round_num(), 1)

        # test first round does not depned on rounds_per_epoch
        tl = Timeline(epoch=1, round=1, rounds_per_epoch=100)
        assertEqual(tl.global_round_num(), 1)

        # test global_round_num properly calculated if global_round not given
        tl = Timeline(epoch=2, round=3, rounds_per_epoch=4)
        assertEqual(tl.global_round_num(), 7)

        # test global_round_num equals global_round regardless of epoch/round
        tl = Timeline(epoch=1, round=1, rounds_per_epoch=1, global_round=123)
        assertEqual(tl.global_round_num(), tl.global_round)

    def test_as_float(self):
        # test first round
        tl = Timeline(epoch=1, round=1)
        assertAlmostEqual(tl.as_float(), 1.0)  # float starts from 1

        # test first round
        tl = Timeline(epoch=1, round=1, rounds_per_epoch=100)
        assertAlmostEqual(tl.as_float(), 1 / 100)

        # first round of 10th epoch
        tl = Timeline(epoch=10, round=1)
        assertAlmostEqual(tl.as_float(), 10.0)

        # test first round does not depned on rounds_per_epoch
        tl = Timeline(epoch=10, round=1, rounds_per_epoch=100)
        assertAlmostEqual(tl.as_float(), 9 + (1 / 100))

        # test offset
        tl = Timeline(epoch=2, round=1, rounds_per_epoch=100)
        assertAlmostEqual(tl.as_float(-1), 1.0)

        # test from global_round
        tl = Timeline(global_round=12, rounds_per_epoch=10)
        assertAlmostEqual(tl.as_float(), 1.2)
        assertAlmostEqual(tl.as_float(1), 1.3)

        # test global_round dominates
        tl = Timeline(global_round=12, rounds_per_epoch=10, epoch=3, round=4)
        assertAlmostEqual(tl.as_float(), 1.2)

        # test random
        tl = Timeline(epoch=5, round=2, rounds_per_epoch=3)
        assertAlmostEqual(tl.as_float(), 4.66, delta=0.01)

    def test_string(self):
        tl = Timeline(epoch=2, round=2, rounds_per_epoch=10)
        assertEqual(f"{tl}", "(epoch = 2, round = 2, global round = 12)")

        tl = Timeline(global_round=12, rounds_per_epoch=10)
        assertEqual(f"{tl}", "(epoch = 2, round = 2, global round = 12)")

        tl = Timeline(global_round=10, rounds_per_epoch=10)
        assertEqual(f"{tl}", "(epoch = 1, round = 10, global round = 10)")

    def test_tick_simple(self):
        # test every epoch, i.e.  tick_interval = 1
        tl = Timeline(epoch=1, round=1, rounds_per_epoch=10)
        assertFalse(tl.tick(tick_interval=1))

        tl = Timeline(epoch=1, round=10, rounds_per_epoch=10)
        assertTrue(tl.tick(tick_interval=1))

        tl = Timeline(epoch=2, round=1, rounds_per_epoch=10)
        assertFalse(tl.tick(tick_interval=1))

        tl = Timeline(epoch=2, round=10, rounds_per_epoch=10)
        assertTrue(tl.tick(tick_interval=1))

        # test every 2 epoch
        tl = Timeline(epoch=1, round=1, rounds_per_epoch=10)
        assertFalse(tl.tick(tick_interval=2))

        tl = Timeline(epoch=1, round=10, rounds_per_epoch=10)
        assertFalse(tl.tick(tick_interval=2))

        tl = Timeline(epoch=2, round=1, rounds_per_epoch=10)
        assertFalse(tl.tick(tick_interval=2))

        tl = Timeline(epoch=2, round=10, rounds_per_epoch=10)
        assertTrue(tl.tick(tick_interval=2))

        # test every 0.2 epoch
        tl = Timeline(epoch=1, round=1, rounds_per_epoch=10)
        assertFalse(tl.tick(tick_interval=0.2))

        tl = Timeline(epoch=1, round=2, rounds_per_epoch=10)
        assertTrue(tl.tick(tick_interval=0.2))

        tl = Timeline(epoch=2, round=1, rounds_per_epoch=10)
        assertFalse(tl.tick(tick_interval=0.2))

        tl = Timeline(epoch=1, round=2, rounds_per_epoch=10)
        assertTrue(tl.tick(tick_interval=0.2))

        tl = Timeline(epoch=1, round=3, rounds_per_epoch=10)
        assertFalse(tl.tick(tick_interval=0.2))

        tl = Timeline(epoch=1, round=4, rounds_per_epoch=10)
        assertTrue(tl.tick(tick_interval=0.2))

        # test with global round
        tl = Timeline(global_round=4, rounds_per_epoch=2)
        assertFalse(tl.tick(tick_interval=3))
        assertTrue(tl.tick(tick_interval=2))

    def test_tick_complex(self):
        # doing 100 epochs, 10 rounds per epoch, ticking interval = 10 epochs
        sum = 0
        for e in range(1, 101):
            for r in range(1, 11):
                tl = Timeline(epoch=e, round=r, rounds_per_epoch=10)
                sum += tl.tick(10)
        assertEqual(sum, 10)

        # doing 100 epochs, 10 rounds per epoch, ticking interval = 0.9 epochs
        sum = 0
        for e in range(1, 101):
            for r in range(1, 11):
                tl = Timeline(epoch=e, round=r, rounds_per_epoch=10)
                sum += tl.tick(0.9)
        assertEqual(sum, 111)
