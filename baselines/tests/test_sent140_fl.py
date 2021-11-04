#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import json

import flsim.configs  # noqa
import hydra
import pkg_resources
from flsim.baselines.run_sent140_fl import main_worker
from flsim.utils.config_utils import fl_config_from_json
from hydra.experimental import compose, initialize
from libfb.py import testutil

CONFIG_PATH = "configs"
CONFIG_NAME = "test_sent140_fl"


class TestSent140FL(testutil.BaseFacebookTestCase):
    def test_sent140_sync_fl_training_with_json_config(self) -> None:
        json_file_name = f"{CONFIG_PATH}/{CONFIG_NAME}.json"
        file_path = pkg_resources.resource_filename(__name__, json_file_name)
        with open(file_path, "r") as parameters_file:
            json_config = json.load(parameters_file)
        config = fl_config_from_json(json_config)

        main_worker(
            trainer_config=config.trainer,
            model_config=config.model,
            data_config=config.data,
            use_cuda_if_available=config.use_cuda_if_available,
            distributed_world_size=config.distributed_world_size,
            fb_info=None,
        )

    def test_sent140_sync_fl_training_with_yaml_config(self) -> None:
        """
        This test only works with Hydra versions 1.1 and above.
        """
        is_hydra11 = tuple(int(x) for x in hydra.__version__.split(".")[:2]) >= (1, 1)
        if not is_hydra11:
            return
        with initialize(config_path=CONFIG_PATH):
            config = compose(config_name=CONFIG_NAME)
            main_worker(
                trainer_config=config.trainer,
                model_config=config.model,
                data_config=config.data,
                use_cuda_if_available=config.use_cuda_if_available,
                distributed_world_size=config.distributed_world_size,
                fb_info=None,
            )
