#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

"""Contains helper for Python PyTest tests.

There are a list of helper utilities missing in PyTest.
We define some of the helpful utilities in this one file
so that all FL-sim PyTests can reuse them easily.
"""
import numpy as np


def assertIsInstance(o: object, t: type) -> None:
    assert isinstance(o, t)


def assertAlmostEqual(f1: float, f2: float, decimal=7) -> None:
    np.testing.assert_almost_equal(f1, f2, decimal=decimal)


def assertEqual(object1: object, object2: object, e: object = None) -> None:
    assert object1 == object2, e


def assertNotEqual(object1: object, object2: object, e: object = None) -> None:
    assert object1 != object2, e


def assertIsNotNone(o: object) -> None:
    assert o is not None


def assertTrue(o: object) -> None:
    assert o


def assertFalse(o: object, e: object = None) -> None:
    assert not o, e
