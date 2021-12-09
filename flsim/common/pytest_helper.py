#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Contains helper for Python PyTest tests.

There are a list of helper utilities missing in PyTest.
We define some of the helpful utilities in this one file
so that all FL-sim PyTests can reuse them easily.
"""
from typing import Any, Sized, List


def assertIsInstance(o: object, t: type) -> None:
    assert isinstance(o, t)


def assertAlmostEqual(first, second, places=None, msg=None, delta=None):
    """Assert that ``first`` and ``second`` is almost equal to each other.

    The equality of ``first`` and ``second`` is determined in a similar way to
    the ``assertAlmostEqual`` function of the standard library.
    """
    if delta is not None and places is not None:
        raise TypeError("specify delta or places - not both")

    msg: str = ""
    diff = abs(second - first)
    if delta is not None:
        if diff <= delta:
            return

        msg = (
            f"The difference between {first} and {second} is not within {delta} delta."
        )
    else:
        if places is None:
            places = 7

        if round(diff, places) == 0:
            return

        msg = f"The difference between {first} and {second} is not within {places} decimal places."

    raise AssertionError(msg)


def assertEqual(object1: object, object2: object, e: object = None) -> None:
    assert object1 == object2, e


def assertNotEqual(object1: object, object2: object, e: object = None) -> None:
    assert object1 != object2, e


def assertLess(object1, object2) -> None:
    assert object1 < object2


def assertGreater(object1, object2) -> None:
    assert object1 > object2


def assertLessEqual(object1, object2) -> None:
    assert object1 <= object2


def assertGreaterEqual(object1, object2) -> None:
    assert object1 >= object2


def assertIsNotNone(o: object) -> None:
    assert o is not None


def assertTrue(o: object, e: object = None) -> None:
    assert o, e


def assertFalse(o: object, e: object = None) -> None:
    assert not o, e


def assertEmpty(o: Sized, msg: object = None) -> None:
    assert len(o) == 0, msg


def assertNotEmpty(o: Sized, msg: object = None) -> None:
    assert len(o) > 0, msg


def assertListEqual(l1: List, l2: List) -> None:
    assert l1 == l2


class assertRaises(object):
    def __init__(self, expected_exc: type) -> None:
        self.expected_exc = expected_exc

    def __enter__(self) -> "assertRaises":
        return self

    def __exit__(self, exc_type: type, exc_value: Exception, traceback: Any) -> bool:
        if exc_type is None:
            raise AssertionError(f"{self.expected_exc} was not raised")
        return isinstance(exc_value, self.expected_exc)
