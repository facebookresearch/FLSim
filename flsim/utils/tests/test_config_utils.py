#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from flsim.common.pytest_helper import assertEqual
from flsim.utils.config_utils import _flatten_dict, fl_json_to_dotlist


class TestConfigUtils:
    def test_flatten_dict(self) -> None:
        assertEqual(_flatten_dict({}), {})
        assertEqual(_flatten_dict({"a": 1}), {"a": 1})
        assertEqual(_flatten_dict({"a": None}), {"a": None})

        # checks neesting, exp notation
        assertEqual(
            _flatten_dict(
                {
                    "a": {
                        "b": {"c": {"val": 3}, "val": 2, "_base_": "b"},
                        "_base_": "a",
                        "val": 1,
                    },
                    "d": "1e-4",
                }
            ),
            {
                "a.b.c.val": 3,
                "a.b.val": 2,
                "a.b._base_": "b",
                "a._base_": "a",
                "a.val": 1,
                "d": "1e-4",
            },
        )

        # checks string floats
        assertEqual(_flatten_dict({"e": "5.5"}), {"e": '"5.5"'})

        # make sure json in list remains untouched
        assertEqual(
            _flatten_dict(
                {
                    "a": {"b": 1},
                    "l": [1, 2, 3],
                    "ld": [{"a": 1, "b": {"bb": 2}, "c": [11, 22]}, {"z": "xyz"}],
                    "ldc": [
                        {
                            "_base_": "parent_base",
                            "x": 1,
                            "child": {
                                "_base_": "child_base",
                                "yl": [2, 3],
                                "yd": {"z": "4"},
                                "yld": [{"z": "4"}],
                                "yc": [{"_base_": "gc_base", "dummy": "one"}],
                            },
                        },
                        {
                            "_base_": "parent_base",
                            "x": 11,
                            "child": {
                                "_base_": "child_base",
                                "yl": [12, 13],
                                "yd": {"z": "14"},
                                "yld": [{"z": "14"}],
                                "yc": [{"_base_": "gc_base", "dummy": "two"}],
                            },
                        },
                    ],
                }
            ),
            {
                "a.b": 1,
                "l": [1, 2, 3],
                "ld": [{"a": 1, "b": {"bb": 2}, "c": [11, 22]}, {"z": "xyz"}],
                "ldc.0._base_": "parent_base",
                "ldc.0.x": 1,
                "ldc.0.child._base_": "child_base",
                "ldc.0.child.yl": [2, 3],
                "ldc.0.child.yd.z": '"4"',
                "ldc.0.child.yld": [{"z": "4"}],
                "ldc.0.child.yc.0._base_": "gc_base",
                "ldc.0.child.yc.0.dummy": "one",
                "ldc.1._base_": "parent_base",
                "ldc.1.x": 11,
                "ldc.1.child._base_": "child_base",
                "ldc.1.child.yl": [12, 13],
                "ldc.1.child.yd.z": '"14"',
                "ldc.1.child.yld": [{"z": "14"}],
                "ldc.1.child.yc.0._base_": "gc_base",
                "ldc.1.child.yc.0.dummy": "two",
            },
        )

        # make sure json in key with suffix _dict remains untouched
        assertEqual(
            _flatten_dict(
                {
                    "a": {"b": 1},
                    "c": {"d_dict": {"A": 1, "B": "2.2", "C": {"key": "three"}}},
                }
            ),
            {
                "a.b": 1,
                "c.d_dict": {"A": 1, "B": "2.2", "C": {"key": "three"}},
            },
        )

        # check with _base_
        assertEqual(
            _flatten_dict(
                {
                    "_base_": {"_base_": "base1", "_base": "base2", "base_": "base3"},
                    "_base": {"_base_": "base1", "_base": "base2", "base_": "base3"},
                    "base_": {"_base_": "base1", "_base": "base2", "base_": "base3"},
                }
            ),
            {
                "_base_._base_": "base1",
                "_base_._base": "base2",
                "_base_.base_": "base3",
                "_base._base_": "base1",
                "_base._base": "base2",
                "_base.base_": "base3",
                "base_._base_": "base1",
                "base_._base": "base2",
                "base_.base_": "base3",
            },
        )

    def test_json_to_dotlist(self) -> None:
        assertEqual(fl_json_to_dotlist({}, append_or_override=False), [])
        assertEqual(fl_json_to_dotlist({"a": 1}, append_or_override=False), ["a=1"])
        assertEqual(
            fl_json_to_dotlist({"a": None}, append_or_override=False), ["a=null"]
        )

        # checks neesting, exp notation
        assertEqual(
            fl_json_to_dotlist(
                {
                    "a": {
                        "b": {"c": {"val": 3}, "val": 2, "_base_": "b"},
                        "_base_": "a",
                        "val": 1,
                    },
                    "d": "1e-4",
                },
                append_or_override=False,
            ),
            ["+a@a=a", "+b@a.b=b", "d=1e-4", "a.val=1", "a.b.val=2", "a.b.c.val=3"],
        )

        # checks string floats
        assertEqual(
            fl_json_to_dotlist({"e": "5.5"}, append_or_override=False), ['e="5.5"']
        )

        # make sure json in list remains untouched
        assertEqual(
            fl_json_to_dotlist(
                {
                    "a": {"b": 1},
                    "l": [1, 2, 3],
                    "ld": [{"a": 1, "b": {"bb": 2}, "c": [11, 22]}, {"z": "xyz"}],
                },
                append_or_override=False,
            ),
            [
                "l=[1, 2, 3]",
                "ld=[{'a': 1, 'b': {'bb': 2}, 'c': [11, 22]}, {'z': 'xyz'}]",
                "a.b=1",
            ],
        )

        # make sure json in key with suffix _dict is handled correctly
        assertEqual(
            fl_json_to_dotlist(
                {
                    "a": {"b": 1},
                    "c": {"d_dict": {"A": 1, "B": "2.2", "C": {"3": "three"}}},
                },
                append_or_override=False,
            ),
            [
                "a.b=1",
                'c.d_dict="{\\"A\\": 1, \\"B\\": \\"2.2\\", \\"C\\": {\\"3\\": \\"three\\"}}"',
            ],
        )

        # check with _base_
        assertEqual(
            fl_json_to_dotlist(
                {
                    "_base_": {
                        "_base_": "base1",
                        "_base": "base2",
                        "base_": "base3",
                        "base": "just_base",
                    },
                    "_base": {
                        "_base_": "base1",
                        "_base": "base2",
                        "base_": "base3",
                        "base": "just_base",
                    },
                    "base_": {
                        "_base_": "base1",
                        "_base": "base2",
                        "base_": "base3",
                        "base": "just_base",
                    },
                    "base": "just_base",
                },
                append_or_override=False,
            ),
            [
                "+_base@_base=base1",
                "+_base_@_base_=base1",
                "+base_@base_=base1",
                "base=just_base",
                "_base._base=base2",
                "_base.base=just_base",
                "_base.base_=base3",
                "_base_._base=base2",
                "_base_.base=just_base",
                "_base_.base_=base3",
                "base_._base=base2",
                "base_.base=just_base",
                "base_.base_=base3",
            ],
        )

    def test_json_to_dotlist_append_or_override(self) -> None:

        assertEqual(fl_json_to_dotlist({}), [])
        assertEqual(fl_json_to_dotlist({"a": 1}), ["++a=1"])
        assertEqual(fl_json_to_dotlist({"a": None}), ["++a=null"])

        # checks neesting, exp notation
        assertEqual(
            fl_json_to_dotlist(
                {
                    "a": {
                        "b": {"c": {"val": 3}, "val": 2, "_base_": "b"},
                        "_base_": "a",
                        "val": 1,
                    },
                    "d": "1e-4",
                }
            ),
            [
                "+a@a=a",
                "+b@a.b=b",
                "++d=1e-4",
                "++a.val=1",
                "++a.b.val=2",
                "++a.b.c.val=3",
            ],
        )

        # checks string floats
        assertEqual(fl_json_to_dotlist({"e": "5.5"}), ['++e="5.5"'])

        # check handling of list values
        assertEqual(
            fl_json_to_dotlist(
                {
                    "a": {"b": 1},
                    "l": [1, 2, 3],
                    "ld": [{"a": 1, "b": {"bb": 2}, "c": [11, 22]}, {"z": "xyz"}],
                    "ldc": [
                        {
                            "_base_": "parent_base",
                            "x": 1,
                            "child": {
                                "_base_": "child_base",
                                "yl": [2, 3],
                                "yd": {"z": "4"},
                                "yld": [{"z": "4"}],
                                "yc": [{"_base_": "gc_base", "dummy": "one"}],
                            },
                        },
                        {
                            "_base_": "parent_base",
                            "x": 11,
                            "child": {
                                "_base_": "child_base",
                                "yl": [12, 13],
                                "yd": {"z": "14"},
                                "yld": [{"z": "14"}],
                                "yc": [{"_base_": "gc_base", "dummy": "two"}],
                            },
                        },
                    ],
                }
            ),
            [
                "+ldc@ldc.0=parent_base",
                "+ldc@ldc.1=parent_base",
                "+child@ldc.0.child=child_base",
                "+child@ldc.1.child=child_base",
                "+yc@ldc.0.child.yc.0=gc_base",
                "+yc@ldc.1.child.yc.0=gc_base",
                "++l=[1, 2, 3]",
                "++ld=[{'a': 1, 'b': {'bb': 2}, 'c': [11, 22]}, {'z': 'xyz'}]",
                "++a.b=1",
                "++ldc.0.x=1",
                "++ldc.1.x=11",
                "++ldc.0.child.yl=[2, 3]",
                "++ldc.0.child.yld=[{'z': '4'}]",
                "++ldc.1.child.yl=[12, 13]",
                "++ldc.1.child.yld=[{'z': '14'}]",
                '++ldc.0.child.yd.z="4"',
                '++ldc.1.child.yd.z="14"',
                "++ldc.0.child.yc.0.dummy=one",
                "++ldc.1.child.yc.0.dummy=two",
            ],
        )

        # make sure json in key with suffix _dict is handled correctly
        assertEqual(
            fl_json_to_dotlist(
                {
                    "a": {"b": 1},
                    "c": {"d_dict": {"A": 1, "B": "2.2", "C": {"3": "three"}}},
                }
            ),
            [
                "++a.b=1",
                '++c.d_dict="{\\"A\\": 1, \\"B\\": \\"2.2\\", \\"C\\": {\\"3\\": \\"three\\"}}"',
            ],
        )

        # check with _base_
        assertEqual(
            fl_json_to_dotlist(
                {
                    "_base_": {
                        "_base_": "base1",
                        "_base": "base2",
                        "base_": "base3",
                        "base": "just_base",
                    },
                    "_base": {
                        "_base_": "base1",
                        "_base": "base2",
                        "base_": "base3",
                        "base": "just_base",
                    },
                    "base_": {
                        "_base_": "base1",
                        "_base": "base2",
                        "base_": "base3",
                        "base": "just_base",
                    },
                    "base": "just_base",
                }
            ),
            [
                "+_base@_base=base1",
                "+_base_@_base_=base1",
                "+base_@base_=base1",
                "++base=just_base",
                "++_base._base=base2",
                "++_base.base=just_base",
                "++_base.base_=base3",
                "++_base_._base=base2",
                "++_base_.base=just_base",
                "++_base_.base_=base3",
                "++base_._base=base2",
                "++base_.base=just_base",
                "++base_.base_=base3",
            ],
        )
