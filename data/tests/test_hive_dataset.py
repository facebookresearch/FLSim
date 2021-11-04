#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from flsim.fb.data.hive_dataset import (
    KoskiHiveDataset,
    KoskiHiveDatasetConfig,
    InlineDatasetConfig,
    simple_hive_column_reader,
)
from hydra.utils import instantiate
from libfb.py import testutil


class genericTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_hive_creation(self) -> None:
        config = KoskiHiveDatasetConfig(
            namespace="ad_delivery",
            data_table="fl_sim_hive_test",
            schema=["label", "user_n"],
            batch_size=1,
            oncall="oncall_primal",
            partitions=["ds=2000-01-01"],
        )
        self.assertTrue(
            isinstance(
                instantiate(config, metadata_map={}),
                KoskiHiveDataset,
            )
        )

    def test_simple_hive_column_reader(self):
        rows = simple_hive_column_reader(
            name_space="ad_delivery",
            data_table="fl_sim_hive_test",
            filters=["ds=2020-01-03"],
            columns_to_read=["user_n"],
            oncall="oncall_primal",
        )
        self.assertEqual(len(rows), 100)


class KoskiHiveDatasetTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

    def _build_inline_dataset(self, num_users, batch_size=1, num_samples_per_user=1):
        config = KoskiHiveDatasetConfig(
            batch_size=batch_size,
            oncall="oncall_primal",
            schema=["id"],
            inline_config=InlineDatasetConfig(
                num_users=num_users,
                data_cols=[],
                id_col="id",
                num_samples_per_user=num_samples_per_user,
            ),
        )
        return instantiate(config, metadata_map={})

    def test_koski_hive_dataset(self):
        num_users = 35
        df = self._build_inline_dataset(num_users=num_users)
        num_batches = 0
        for _ in df:
            num_batches += 1
        self.assertEqual(num_batches, num_users)

    def test_apply_filter(self):
        batch_size = 1
        df = self._build_inline_dataset(num_users=35, batch_size=batch_size)

        df.apply_filter("id < 10")
        user_ids = [row["id"][0] for row in df]
        for user in user_ids:
            self.assertEqual(user.shape[0], batch_size)
            self.assertLess(user.item(), 10)
