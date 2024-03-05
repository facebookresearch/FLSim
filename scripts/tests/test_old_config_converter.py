#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json

import pkg_resources
from flsim.common.pytest_helper import assertTrue
from scripts.old_config_converter import get_new_fl_config

OLD_CONFIGS = [
    "configs/fedbuff_fedadam_old.json",
    "configs/async_fedsgd_old.json",
    "configs/sync_fedavg_old.json",
    "configs/privatesync_fedsgd_old.json",
]

NEW_CONFIGS = [
    "configs/fedbuff_fedadam_new.json",
    "configs/async_fedsgd_new.json",
    "configs/sync_fedavg_new.json",
    "configs/privatesync_fedsgd_new.json",
]


class TestOldConfigConveter:
    def test_conversion(self) -> None:
        for old_file_path, new_file_path in zip(OLD_CONFIGS, NEW_CONFIGS):
            old_file_path = pkg_resources.resource_filename(__name__, old_file_path)
            new_file_path = pkg_resources.resource_filename(__name__, new_file_path)
            with open(old_file_path) as old_file:
                old = json.load(old_file)
            with open(new_file_path) as new_file:
                new = json.load(new_file)

            converted_old = get_new_fl_config(old, flsim_example=True)
            assertTrue(dict(converted_old) == dict(new))
