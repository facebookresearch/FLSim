#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import json

import pkg_resources
from flsim.scripts.old_config_converter import get_new_fl_config
from libfb.py import testutil

OLD_CONFIGS = [
    "configs/hybrid_fedadam_old.json",
    "configs/async_fedsgd_old.json",
    "configs/sync_fedavg_old.json",
    "configs/privatesync_fedsgd_old.json",
]

NEW_CONFIGS = [
    "configs/hybrid_fedadam_new.json",
    "configs/async_fedsgd_new.json",
    "configs/sync_fedavg_new.json",
    "configs/privatesync_fedsgd_new.json",
]


class OldConfigConveterTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

    @testutil.data_provider(
        lambda: [
            {"old_file_path": ofp, "new_file_path": nfp}
            for (ofp, nfp) in zip(OLD_CONFIGS, NEW_CONFIGS)
        ]
    )
    def test_conversion(self, old_file_path, new_file_path) -> None:
        # for old_file_path, new_file_path in zip(OLD_CONFIGS, NEW_CONFIGS):
        old_file_path = pkg_resources.resource_filename(__name__, old_file_path)
        new_file_path = pkg_resources.resource_filename(__name__, new_file_path)
        with open(old_file_path) as old_file:
            old = json.load(old_file)
        with open(new_file_path) as new_file:
            new = json.load(new_file)

        converted_old = get_new_fl_config(old, flsim_example=True)
        self.assertTrue(dict(converted_old) == dict(new))
