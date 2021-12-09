#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from flsim.common.pytest_helper import assertEqual, assertTrue, assertLessEqual
from flsim.utils.data.data_utils import batchify, merge_dicts
from flsim.utils.data.fake_data_utils import FakeDataProvider, FakeUserData


class TestDataUtils:
    def test_batchify(self) -> None:
        assertEqual(list(batchify([1, 2, 3, 4, 5], 2)), [[1, 2], [3, 4], [5]])
        assertEqual(list(batchify([1, 2, 3, 4, 5], 3)), [[1, 2, 3], [4, 5]])
        assertEqual(list(batchify([1, 2, 3, 4], 2)), [[1, 2], [3, 4]])
        assertEqual(list(batchify([1, 2, 3, 4], 1)), [[1], [2], [3], [4]])

    def test_merge_dicts(self) -> None:
        expected = {"a": torch.Tensor([1.0, 2.0])}
        for key, actual in merge_dicts(
            [{"a": torch.Tensor([1])}, {"a": torch.Tensor([2])}]
        ).items():
            assertTrue(key in expected)
            assertTrue(torch.all(actual.eq(expected[key])))

        expected = {"a": torch.Tensor([1.0]), "b": torch.Tensor([2.0])}
        for key, actual in merge_dicts(
            [{"a": torch.Tensor([1])}, {"b": torch.Tensor([2])}]
        ).items():
            assertTrue(key in expected)
            assertTrue(torch.all(actual.eq(expected[key])))

    def user_data_test_util(
        self,
        user_dataset,
        expected_num_examples,
        expected_batch_size,
        expected_num_batches,
    ):
        assertEqual(user_dataset.num_examples(), expected_num_examples)

        for i, batch in enumerate(user_dataset):
            assertLessEqual(len(batch["data"]), expected_batch_size)
            last_batch = i
        assertEqual(last_batch + 1, expected_num_batches)

    def test_fake_user_data(self):
        def gen_batch(n, value=None):
            return {"data": [torch.ones(n, 10)], "label": [1] * n}

        num_examples = 100
        batch_size = 10
        num_batches = num_examples // batch_size
        user_dataset = FakeUserData(gen_batch, num_batches, batch_size)
        self.user_data_test_util(user_dataset, num_examples, batch_size, num_batches)

    def test_fake_data_provider(self):
        def gen_batch(n, value=None):
            return {"data": [torch.ones(n, 10)], "label": [1] * n}

        num_batches = 2
        batch_size = 10
        num_users = 100
        fl_data_provider = FakeDataProvider(
            gen_batch, num_batches, batch_size, num_users
        )

        assertEqual(fl_data_provider.num_users(), num_users)
        assertEqual(fl_data_provider.user_ids(), list(range(num_users)))
        ad_hoc_users = [0, 3, 10, 50, 99]
        num_examples = num_batches * batch_size
        for user in ad_hoc_users:
            user_dataset = fl_data_provider.get_user_data(user)
            self.user_data_test_util(
                user_dataset, num_examples, batch_size, num_batches
            )

        self.user_data_test_util(
            fl_data_provider.test_data(), num_examples, batch_size, num_batches
        )
