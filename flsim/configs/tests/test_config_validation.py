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

import flsim.configs  # noqa
import omegaconf
from flsim.common.pytest_helper import assertRaises
from flsim.utils.sample_model import DummyAlphabetFLModel
from hydra.experimental import compose, initialize
from hydra.utils import instantiate


class TestConfigValidation:
    def test_throw_exception_on_missing_field(self) -> None:
        with initialize(config_path=None):
            cfg = compose(
                config_name=None,
                overrides=[
                    "+trainer=base_async_trainer",
                    # ThresholdStalenessWeightConfig has MISSING fields.
                    # Hydra should throw an exception at instantiation time
                    # since we won't be setting those fields here.
                    "+staleness_weight@trainer.async_weight.staleness_weight=base_threshold_staleness_weight",
                ],
            )
        with assertRaises(
            omegaconf.errors.MissingMandatoryValue,  # with Hydra 1.1
        ):
            instantiate(cfg.trainer, model=DummyAlphabetFLModel(), cuda_enabled=False)
