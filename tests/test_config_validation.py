#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import flsim.configs  # noqa
import hydra
import omegaconf
from flsim.utils.sample_model import DummyAlphabetFLModel
from hydra.experimental import compose, initialize
from hydra.utils import instantiate
from libfb.py import testutil


class ConfigValidation(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

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
        with self.assertRaises(
            (
                omegaconf.errors.MissingMandatoryValue,  # with Hydra 1.1
                hydra.errors.HydraException,  # with Hydra 1.0
            ),
        ):
            instantiate(cfg.trainer, model=DummyAlphabetFLModel(), cuda_enabled=False)
