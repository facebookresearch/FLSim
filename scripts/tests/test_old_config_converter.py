#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pkg_resources
from flsim.common.pytest_helper import assertTrue
from flsim.scripts.old_config_converter import get_new_fl_config

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
