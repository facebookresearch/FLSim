#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import json

import flsim.configs  # noqa
import pkg_resources
from flsim.baselines.run_celeba_fl import main_worker
from flsim.utils.config_utils import fl_config_from_json
from hydra.experimental import compose, initialize
from libfb.py import testutil

CONFIG_PATH = "configs"
CONFIG_NAME = "test_celeba_fl"


class TestCelebAFL(testutil.BaseFacebookTestCase):
    def test_celeba_fl_training_with_json_config(self) -> None:
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
            is_unit_test=True,
        )

    def test_celeba_fl_training_with_yaml_config(self) -> None:
        with initialize(config_path="configs"):
            config = compose(config_name="test_celeba_fl")
            main_worker(
                trainer_config=config.trainer,
                model_config=config.model,
                data_config=config.data,
                use_cuda_if_available=config.use_cuda_if_available,
                distributed_world_size=config.distributed_world_size,
                fb_info=None,
                is_unit_test=True,
            )
