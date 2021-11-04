#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import math
import random
import string

import torch
from flsim.data.data_sharder import (
    FLDataSharder,
    ShardingStrategyFactory,
    ShardingStrategyType,
)
from flsim.data.dataset_data_loader import FLDatasetDataLoaderWithBatch
from flsim.utils.sample_model import TestDataSetting
from flsim.utils.tests.helpers.test_data_utils import (
    DummyAlphabetDataset,
    Utils,
)
from libfb.py import testutil


class MockData:
    @staticmethod
    def provide_data():
        """Generate 26 rows of data. Each row:
        "label":0 or 1, "text":int between 0 and 25
        "user_id": between 'a' and 'z'
        """
        characters = Utils.get_characters(num=26)
        return [
            {
                TestDataSetting.LABEL_COL_NAME: Utils.get_label(character),
                TestDataSetting.TEXT_COL_NAME: Utils.get_text(character),
                TestDataSetting.USER_ID_COL_NAME: torch.tensor(
                    ord(character), dtype=torch.int8
                ),
            }
            for character in characters
        ]


def provide_configs():
    return {
        FLDataSharder.RandomSharding: FLDataSharder.Config(
            sharding_strategy=ShardingStrategyType.RANDOM,
            num_shards=TestDataSetting.NUM_SHARDS,
            sharding_colindex=None,
            sharding_col_name=None,
            shard_size_for_sequential=None,
            alpha=None,
        ),
        FLDataSharder.BroadcastSharding: FLDataSharder.Config(
            sharding_strategy=ShardingStrategyType.BROADCAST,
            num_shards=TestDataSetting.NUM_SHARDS,
            sharding_colindex=None,
            sharding_col_name=None,
            shard_size_for_sequential=None,
            alpha=None,
        ),
        FLDataSharder.ColumnSharding: FLDataSharder.Config(
            sharding_strategy=ShardingStrategyType.COLUMN,
            num_shards=None,
            sharding_colindex=None,
            sharding_col_name=TestDataSetting.USER_ID_COL_NAME,
            shard_size_for_sequential=None,
            alpha=None,
        ),
        FLDataSharder.RoundRobinSharding: FLDataSharder.Config(
            sharding_strategy=ShardingStrategyType.ROUND_ROBIN,
            num_shards=TestDataSetting.NUM_SHARDS,
            sharding_colindex=None,
            sharding_col_name=None,
            shard_size_for_sequential=None,
            alpha=None,
        ),
        FLDataSharder.SequentialSharding: FLDataSharder.Config(
            sharding_strategy=ShardingStrategyType.SEQUENTIAL,
            num_shards=None,
            sharding_colindex=None,
            sharding_col_name=None,
            shard_size_for_sequential=2,
            alpha=None,
        ),
        FLDataSharder.PowerLawSharding: FLDataSharder.Config(
            sharding_strategy=ShardingStrategyType.POWER_LAW,
            num_shards=5,
            sharding_colindex=None,
            sharding_col_name=None,
            shard_size_for_sequential=None,
            alpha=0.2,
        ),
    }


class DataSharderTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_sharder_strategy_factory(self) -> None:
        for sharder_type, config in provide_configs().items():
            sharder = ShardingStrategyFactory.create(config)
            self.assertIsInstance(sharder, sharder_type)

    def test_random_sharder(self) -> None:
        """Only tests random sharding strategy in this test case. All the other
        strategies are tested in test_shard_rows().
        """
        random_sharder_config = provide_configs()[FLDataSharder.RandomSharding]
        random.seed(1)
        random_sharder = ShardingStrategyFactory.create(random_sharder_config)
        for i in range(random_sharder_config.num_shards + 1):
            shard = random_sharder.shard_for_row(
                MockData.provide_data()[i % sum(1 for row in MockData.provide_data())]
            )
            if i == 0:
                self.assertEqual(shard, [6])
            elif i == random_sharder_config.num_shards:
                self.assertEqual(shard, [3])

    def test_shard_rows_random(self) -> None:
        random_sharder_config = provide_configs()[FLDataSharder.RandomSharding]
        random_fl_sharder = FLDataSharder(
            sharding_strategy=random_sharder_config.sharding_strategy,
            num_shards=random_sharder_config.num_shards,
        )
        random_sharded_rows = list(
            random_fl_sharder.shard_rows(
                MockData.provide_data()[
                    : min(
                        int(random_sharder_config.num_shards * 1.2 + 1),
                        sum(1 for row in MockData.provide_data()),
                    )
                ]
            )
        )
        self.assertEqual(len(random_sharded_rows), random_sharder_config.num_shards)

    def test_shard_rows_power_law(self) -> None:
        power_sharder_config = provide_configs()[FLDataSharder.PowerLawSharding]
        power_fl_sharder = FLDataSharder(
            sharding_strategy=power_sharder_config.sharding_strategy,
            num_shards=power_sharder_config.num_shards,
            alpha=power_sharder_config.alpha,
        )
        power_law_sharded_rows = list(
            power_fl_sharder.shard_rows(MockData.provide_data())
        )
        num_examples_per_user = [len(p) for _, p in power_law_sharded_rows]
        self.assertEqual(len(num_examples_per_user), power_sharder_config.num_shards)
        # assert that top user will have more than 20% of the examples
        self.assertTrue(max(num_examples_per_user) / sum(num_examples_per_user) > 0.20)

    def test_shard_rows_broadcast(self) -> None:
        broadcast_sharder_config = provide_configs()[FLDataSharder.BroadcastSharding]
        broadcast_fl_sharder = FLDataSharder(
            broadcast_sharder_config.sharding_strategy,
            broadcast_sharder_config.num_shards,
            broadcast_sharder_config.sharding_colindex,
            broadcast_sharder_config.sharding_col_name,
            broadcast_sharder_config.alpha,
        )
        # all rows from dataset should be replicated to all shards
        self.assertEqual(
            [
                user_data
                for _, user_data in broadcast_fl_sharder.shard_rows(
                    MockData.provide_data()
                )
            ],
            [
                MockData.provide_data()
                for shard_idx in range(TestDataSetting.NUM_SHARDS)
            ],
        )

    def test_shard_rows_column(self) -> None:
        column_sharder_config = provide_configs()[FLDataSharder.ColumnSharding]
        column_fl_sharder = FLDataSharder(
            column_sharder_config.sharding_strategy,
            column_sharder_config.num_shards,
            column_sharder_config.sharding_colindex,
            column_sharder_config.sharding_col_name,
            column_sharder_config.alpha,
        )
        # each data row should be assigned to a unique shard, since column
        # sharding is based on user id here and each mocked user id is unique
        self.assertEqual(
            [
                user_data
                for _, user_data in column_fl_sharder.shard_rows(
                    MockData.provide_data()
                )
            ],
            [[one_data_row] for one_data_row in MockData.provide_data()],
        )

    def test_shard_rows_round_robin(self) -> None:
        roundrobin_sharder_config = provide_configs()[FLDataSharder.RoundRobinSharding]
        roundrobin_fl_sharder = FLDataSharder(
            roundrobin_sharder_config.sharding_strategy,
            roundrobin_sharder_config.num_shards,
            roundrobin_sharder_config.sharding_colindex,
            roundrobin_sharder_config.sharding_col_name,
            roundrobin_sharder_config.alpha,
        )
        sharded_rows_from_roundrobin = [
            user_data
            for _, user_data in roundrobin_fl_sharder.shard_rows(
                MockData.provide_data()
            )
        ]
        # there are 26 data rows here and it should be sharded with round-
        # robin fashion with 10 shards. e.g. 1th, 11th, 21th data row
        # should be in the same shard.
        for shard_index in range(TestDataSetting.NUM_SHARDS):
            self.assertEqual(
                sharded_rows_from_roundrobin[shard_index],
                [
                    one_data_row
                    for row_index, one_data_row in enumerate(MockData.provide_data())
                    if row_index % TestDataSetting.NUM_SHARDS == shard_index
                ],
            )

    def test_shard_rows_sequential(self) -> None:
        sequential_sharder_config = provide_configs()[FLDataSharder.SequentialSharding]
        sequential_fl_sharder = FLDataSharder(
            sequential_sharder_config.sharding_strategy,
            sequential_sharder_config.num_shards,
            sequential_sharder_config.sharding_colindex,
            sequential_sharder_config.sharding_col_name,
            sequential_sharder_config.shard_size_for_sequential,
            sequential_sharder_config.alpha,
        )
        sharded_rows_from_sequential = [
            user_data
            for _, user_data in sequential_fl_sharder.shard_rows(
                MockData.provide_data()
            )
        ]
        self.assertEqual(
            len(sharded_rows_from_sequential),
            len(string.ascii_lowercase)
            / sequential_sharder_config.shard_size_for_sequential,
        )
        dataset = MockData.provide_data()
        for shard_index, data_for_a_shard in enumerate(sharded_rows_from_sequential):
            start_index = (
                shard_index * sequential_sharder_config.shard_size_for_sequential
            )
            self.assertEqual(
                data_for_a_shard,
                dataset[
                    start_index : start_index
                    + sequential_sharder_config.shard_size_for_sequential
                ],
            )

    def test_distributed_user_sharding(self) -> None:
        # mock 26 rows
        shard_size = 4
        local_batch_size = 2
        world_size = 4

        # mock world_size parallel creation of fl_train_set
        for rank in range(world_size):
            fl_data_sharder = FLDataSharder("sequential", None, None, None, shard_size)
            data_loader = FLDatasetDataLoaderWithBatch(
                MockData.provide_data(),
                MockData.provide_data(),
                MockData.provide_data(),
                fl_data_sharder,
                local_batch_size,  # train_batch_size
                local_batch_size,  # eval_batch_size
                local_batch_size,  # test_batch_size
            )
            train_set = data_loader.fl_train_set(rank=rank, world_size=world_size)
            self.assertEqual(data_loader.num_total_users, math.ceil(26 / shard_size))
            number_of_users_on_worker = 2 if rank in (0, 1, 2) else 1
            # pyre-fixme[6]
            train_set_size = len(train_set)
            self.assertEqual(train_set_size, number_of_users_on_worker)

    def test_pytorch_dataset_wrapper(self) -> None:
        # mock 26 rows
        shard_size = 4
        local_batch_size = 2

        fl_data_sharder = FLDataSharder("sequential", None, None, None, shard_size)
        dummy_dataset = DummyAlphabetDataset()
        data_loader = FLDatasetDataLoaderWithBatch(
            dummy_dataset,
            dummy_dataset,
            dummy_dataset,
            fl_data_sharder,
            local_batch_size,  # train_batch_size
            local_batch_size,  # eval_batch_size
            local_batch_size,  # test_batch_size
        )

        data_loader.fl_train_set()
        self.assertEqual(data_loader.num_total_users, math.ceil(26 / shard_size))
        eval_set = data_loader.fl_eval_set()
        # pyre-fixme[6]
        self.assertEqual(len(eval_set), (26 / local_batch_size))
