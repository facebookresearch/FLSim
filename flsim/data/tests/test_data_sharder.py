#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import string

import torch
from flsim.common.pytest_helper import assertEqual, assertTrue
from flsim.data.data_sharder import (
    RandomSharderConfig,
    SequentialSharderConfig,
    RoundRobinSharderConfig,
    ColumnSharderConfig,
    BroadcastSharderConfig,
    PowerLawSharderConfig,
)
from flsim.data.dataset_data_loader import FLDatasetDataLoaderWithBatch
from flsim.utils.sample_model import TestDataSetting
from flsim.utils.tests.helpers.test_data_utils import (
    DummyAlphabetDataset,
    Utils,
)
from hydra.utils import instantiate


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


class TestDataSharder:
    def test_random_sharder(self) -> None:
        """Only tests random sharding strategy in this test case. All the other
        strategies are tested in test_shard_rows().
        """
        random.seed(1)
        random_sharder = instantiate(
            RandomSharderConfig(num_shards=TestDataSetting.NUM_SHARDS)
        )
        for i in range(random_sharder.cfg.num_shards + 1):
            shard = random_sharder.shard_for_row(
                MockData.provide_data()[i % sum(1 for row in MockData.provide_data())]
            )
            if i == 0:
                assertEqual(shard, [6])
            elif i == random_sharder.cfg.num_shards:
                assertEqual(shard, [3])

    def test_shard_rows_random(self) -> None:
        random_sharder = instantiate(
            RandomSharderConfig(num_shards=TestDataSetting.NUM_SHARDS)
        )
        random_sharded_rows = list(
            random_sharder.shard_rows(
                MockData.provide_data()[
                    : min(
                        int(random_sharder.cfg.num_shards * 1.2 + 1),
                        sum(1 for row in MockData.provide_data()),
                    )
                ]
            )
        )
        assertEqual(len(random_sharded_rows), random_sharder.cfg.num_shards)

    def test_shard_rows_power_law(self) -> None:
        power_law_sharder = instantiate(PowerLawSharderConfig(num_shards=2, alpha=0.2))
        power_law_sharded_rows = list(
            power_law_sharder.shard_rows(MockData.provide_data())
        )
        num_examples_per_user = [len(p) for _, p in power_law_sharded_rows]
        assertEqual(len(num_examples_per_user), power_law_sharder.cfg.num_shards)
        # assert that top user will have more than 20% of the examples
        assertTrue(max(num_examples_per_user) / sum(num_examples_per_user) > 0.20)

    def test_shard_rows_broadcast(self) -> None:
        broadcast_sharder = instantiate(
            BroadcastSharderConfig(num_shards=TestDataSetting.NUM_SHARDS)
        )
        # all rows from dataset should be replicated to all shards
        assertEqual(
            [
                user_data
                for _, user_data in broadcast_sharder.shard_rows(
                    MockData.provide_data()
                )
            ],
            [
                MockData.provide_data()
                for shard_idx in range(TestDataSetting.NUM_SHARDS)
            ],
        )

    def test_shard_rows_column(self) -> None:
        column_sharder = instantiate(
            ColumnSharderConfig(sharding_col=TestDataSetting.USER_ID_COL_NAME)
        )
        # each data row should be assigned to a unique shard, since column
        # sharding is based on user id here and each mocked user id is unique
        assertEqual(
            [
                user_data
                for _, user_data in column_sharder.shard_rows(MockData.provide_data())
            ],
            [[one_data_row] for one_data_row in MockData.provide_data()],
        )

    def test_shard_rows_round_robin(self) -> None:
        round_robin_sharder = instantiate(
            RoundRobinSharderConfig(num_shards=TestDataSetting.NUM_SHARDS)
        )
        sharded_rows_from_round_robin = [
            user_data
            for _, user_data in round_robin_sharder.shard_rows(MockData.provide_data())
        ]
        # there are 26 data rows here and it should be sharded with round-
        # robin fashion with 10 shards. e.g. 1th, 11th, 21th data row
        # should be in the same shard.
        for shard_index in range(round_robin_sharder.cfg.num_shards):
            assertEqual(
                sharded_rows_from_round_robin[shard_index],
                [
                    one_data_row
                    for row_index, one_data_row in enumerate(MockData.provide_data())
                    if row_index % round_robin_sharder.cfg.num_shards == shard_index
                ],
            )

    def test_shard_rows_sequential(self) -> None:
        sequential_sharder = instantiate(SequentialSharderConfig(examples_per_shard=2))
        sharded_rows_from_sequential = [
            user_data
            for _, user_data in sequential_sharder.shard_rows(MockData.provide_data())
        ]
        assertEqual(
            len(sharded_rows_from_sequential),
            len(string.ascii_lowercase) / sequential_sharder.cfg.examples_per_shard,
        )
        dataset = MockData.provide_data()
        for shard_index, data_for_a_shard in enumerate(sharded_rows_from_sequential):
            start_index = shard_index * sequential_sharder.cfg.examples_per_shard
            assertEqual(
                data_for_a_shard,
                dataset[
                    start_index : start_index
                    + sequential_sharder.cfg.examples_per_shard
                ],
            )

    def test_distributed_user_sharding(self) -> None:
        # mock 26 rows
        shard_size = 4
        local_batch_size = 2
        world_size = 4

        # mock world_size parallel creation of fl_train_set
        for rank in range(world_size):
            fl_data_sharder = instantiate(
                SequentialSharderConfig(examples_per_shard=shard_size)
            )
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
            assertEqual(data_loader.num_total_users, math.ceil(26 / shard_size))
            number_of_users_on_worker = 2 if rank in (0, 1, 2) else 1
            # pyre-fixme[6]: Expected `Sized` for 1st param but got
            #  `Iterable[typing.Iterable[typing.Any]]`.
            train_set_size = len(train_set)
            assertEqual(train_set_size, number_of_users_on_worker)

    def test_pytorch_dataset_wrapper(self) -> None:
        # mock 26 rows
        shard_size = 4
        local_batch_size = 2

        fl_data_sharder = instantiate(
            SequentialSharderConfig(examples_per_shard=shard_size)
        )
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
        assertEqual(data_loader.num_total_users, math.ceil(26 / shard_size))
        eval_set = data_loader.fl_eval_set()
        # pyre-fixme[6]
        assertEqual(len(eval_set), (26 / local_batch_size))
