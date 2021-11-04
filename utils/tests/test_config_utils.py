#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from flsim.utils.config_utils import _flatten_dict, fl_json_to_dotlist
from libfb.py import testutil


class ConfigUtilsTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_flatten_dict(self):
        self.assertEqual(_flatten_dict({}), {})
        self.assertEqual(_flatten_dict({"a": 1}), {"a": 1})
        self.assertEqual(_flatten_dict({"a": None}), {"a": None})

        # checks neesting, exp notation
        self.assertEqual(
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
        self.assertEqual(_flatten_dict({"e": "5.5"}), {"e": '"5.5"'})

        # make sure json in list remains untouched
        self.assertEqual(
            _flatten_dict(
                {
                    "a": {"b": 1},
                    "l": [1, 2, 3],
                    "ld": [{"a": 1, "b": {"bb": 2}, "c": [11, 22]}, {"z": "xyz"}],
                }
            ),
            {
                "a.b": 1,
                "l": [1, 2, 3],
                "ld": [{"a": 1, "b": {"bb": 2}, "c": [11, 22]}, {"z": "xyz"}],
            },
        )

        # make sure json in key with suffix _dict remains untouched
        self.assertEqual(
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
        self.assertEqual(
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

    def test_json_to_dotlist(self):
        self.assertEqual(fl_json_to_dotlist({}, append_or_override=False), [])
        self.assertEqual(
            fl_json_to_dotlist({"a": 1}, append_or_override=False), ["a=1"]
        )
        self.assertEqual(
            fl_json_to_dotlist({"a": None}, append_or_override=False), ["a=null"]
        )

        # checks neesting, exp notation
        self.assertEqual(
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
        self.assertEqual(
            fl_json_to_dotlist({"e": "5.5"}, append_or_override=False), ['e="5.5"']
        )

        # make sure json in list remains untouched
        self.assertEqual(
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
        self.assertEqual(
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
        self.assertEqual(
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

    def test_json_to_dotlist_append_or_override(self):

        self.assertEqual(fl_json_to_dotlist({}), [])
        self.assertEqual(fl_json_to_dotlist({"a": 1}), ["++a=1"])
        self.assertEqual(fl_json_to_dotlist({"a": None}), ["++a=null"])

        # checks neesting, exp notation
        self.assertEqual(
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
        self.assertEqual(fl_json_to_dotlist({"e": "5.5"}), ['++e="5.5"'])

        # make sure json in list remains untouched
        self.assertEqual(
            fl_json_to_dotlist(
                {
                    "a": {"b": 1},
                    "l": [1, 2, 3],
                    "ld": [{"a": 1, "b": {"bb": 2}, "c": [11, 22]}, {"z": "xyz"}],
                }
            ),
            [
                "++l=[1, 2, 3]",
                "++ld=[{'a': 1, 'b': {'bb': 2}, 'c': [11, 22]}, {'z': 'xyz'}]",
                "++a.b=1",
            ],
        )

        # make sure json in key with suffix _dict is handled correctly
        self.assertEqual(
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
        self.assertEqual(
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
