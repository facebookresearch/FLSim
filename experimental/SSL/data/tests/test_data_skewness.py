#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import flsim.experimental.ssl.data.data_sharder as sharder
import flsim.experimental.ssl.data.data_skewness as skew
from flsim.utils.data.dummy_image_dataset import DummyImageDataset
from libfb.py import testutil

NUM_SAMPLES_PER_CLASS = 50
NUM_CLASSES = 10


class TestSkewnessFactory(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.setup_dataset()

    def setup_dataset(self):
        self.dataset = DummyImageDataset(
            num_classes=NUM_CLASSES, num_images_per_class=NUM_SAMPLES_PER_CLASS
        )

    @testutil.data_provider(
        lambda: (
            {
                "skewness_config": {
                    "skewness_type": "uniform",
                }
            },
        )
    )
    def test_uniform_skewness(self, skewness_config):
        skewness = skew.SkewnessFactory().create(**skewness_config)
        self.assertEqual(
            skewness.num_samples(0), skewness.num_samples(1), skewness.num_samples(100)
        )

    @testutil.data_provider(
        lambda: (
            {
                "skewness_config": {
                    "num_clients": 100,
                    "skewness_type": "powerlaw",
                    "alpha": 0.0005,
                }
            },
        )
    )
    def test_powerlaw_skewness(self, skewness_config):
        top_portion = []
        for _i in range(100):
            skewness = skew.SkewnessFactory.create(**skewness_config)
            num_examples_per_user = [
                skewness.num_samples(client_idx)
                for client_idx in range(skewness_config["num_clients"])
            ]
            top_portion.append(max(num_examples_per_user) / sum(num_examples_per_user))

        # assert that top user (on average) will have more than 90% of the examples
        self.assertTrue(sum(top_portion) / len(top_portion) > 0.9)

    @testutil.data_provider(
        lambda: (
            {
                "skewness_config": {
                    "skewness_type": "uniform",
                },
                "dict_config": {
                    "frac_server": 0.3,
                    "frac_client_labeled": 0.5,
                    "iidness_client_labeled": 0.5,
                    "iidness_client_unlabeled": 0.5,
                    "iidness_server": 0.5,
                    "num_clients": 18,
                },
            },
        )
    )
    def test_skewness_integration(self, skewness_config, dict_config):
        skewness = skew.SkewnessFactory.create(**skewness_config)
        ssl_sharder = sharder.CustomSSLSharding(
            dataset=self.dataset,
            **dict_config,
            num_samples_skewness_client=skewness.num_samples,
        )
        ssl_sharder.create_sharding()
        # check labeled data
        self.assertEqual(
            len(ssl_sharder.get_shard_dataset(2)), len(ssl_sharder.get_shard_dataset(4))
        )
        # check unlabeled data
        self.assertEqual(
            len(ssl_sharder.get_shard_dataset(3)), len(ssl_sharder.get_shard_dataset(5))
        )
