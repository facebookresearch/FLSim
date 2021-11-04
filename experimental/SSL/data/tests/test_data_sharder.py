#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import flsim.experimental.ssl.data.data_sharder as sharder
from flsim.utils.data.dummy_image_dataset import DummyImageDataset
from libfb.py import testutil

NUM_SAMPLES_PER_CLASS = 50
NUM_CLASSES = 10


class TestSSLSharder(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.setup_dataset()

    def setup_dataset(self):
        self.dataset = DummyImageDataset(
            num_classes=NUM_CLASSES, num_images_per_class=NUM_SAMPLES_PER_CLASS
        )

    def _get_uniform_num_samples_skew(self, client_idx):
        return 18.0

    @testutil.data_provider(
        lambda: (
            {
                "dict_config": {
                    "frac_server": 0.3,
                    "frac_client_labeled": 0.5,
                    "iidness_client_labeled": 0.5,
                    "iidness_client_unlabeled": 0.5,
                    "iidness_server": 0.5,
                    "num_clients": 18,
                }
            },
        )
    )
    def test_frac_server(self, dict_config):
        ssl_sharder = sharder.CustomSSLSharding(
            dataset=self.dataset,
            **dict_config,
            num_samples_skewness_client=self._get_uniform_num_samples_skew,
        )
        ssl_sharder.create_sharding()
        self.assertEqual(
            len(ssl_sharder.get_shard_dataset(0)), int(0.3 * len(self.dataset))
        )
        self.assertEqual(len(ssl_sharder.get_shard_dataset(1)), 0)
