#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, Callable, Iterable, Iterator, List, Optional

from flsim.data.data_provider import (
    FLDataProviderFromList,
    IFLDataProvider,
    IFLUserData,
)
from flsim.utils.sample_model import MockFLModel


def create_mock_data_provider(
    num_users: int, examples_per_user: int
) -> IFLDataProvider:
    # one_user_data has 1 batch of len = examples_per_user
    one_user_data = [list(range(examples_per_user))]
    data = [one_user_data] * num_users

    return FLDataProviderFromList(
        train_user_list=data,
        eval_user_list=data,
        test_user_list=data,
        model=MockFLModel(num_examples_per_user=examples_per_user),
    )


class FakeUserData(IFLUserData):
    """
    fake data for a single user.
    """

    def __init__(
        self,
        gen_batch: Callable[[int, Any], Any],
        num_batches: int = 1,
        batch_size: int = 2,
        val: Optional[float] = None,
    ):
        """

        gen_batch is a callable that gets a batch_size
        and generates a simulated batch with the same size
        """
        self.gen_batch = gen_batch
        self._num_batches = num_batches
        self.batch_size = batch_size
        self.val = val

    def train_data(self) -> Iterator[Any]:
        # TODO add flag for a final batch being incomplete
        for _ in range(self._num_batches):
            yield self.gen_batch(self.batch_size, self.val)

    def num_train_examples(self) -> int:
        return self._num_batches * self.batch_size

    def num_batches(self) -> int:
        return self._num_batches

    def eval_data(self):
        # TODO add flag for a final batch being incomplete
        for _ in range(self._num_batches):
            yield self.gen_batch(self.batch_size, self.val)

    def num_eval_batches(self):
        return self._num_batches

    def num_train_batches(self):
        return self._num_batches

    def num_eval_examples(self):
        return self._num_batches * self.batch_size


class FakeDataProvider(IFLDataProvider):
    def __init__(
        self,
        gen_batch: Callable[[int, Any], Any],
        num_batches: int = 1,
        batch_size: int = 2,
        num_users: int = 10,
        random: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.user_data = [
            FakeUserData(
                gen_batch, num_batches, batch_size, None if random else i / num_users
            )
            for i in range(rank, num_users, world_size)
        ]
        self.eval = FakeUserData(
            gen_batch, num_batches, batch_size, None if random else 1
        )
        self._num_users = num_users // world_size
        self._num_total_users = num_users

    def __iter__(self) -> Iterable[IFLUserData]:
        yield from self.user_data

    def __getitem__(self, index) -> IFLUserData:
        return self.user_data[index]

    def train_user_ids(self) -> List[int]:
        return list(range(self._num_users))

    def num_train_users(self) -> int:
        return self._num_users

    def num_users(self):
        return self._num_users

    def num_total_users(self) -> int:
        return self._num_total_users

    def get_train_user(self, user_index: int) -> IFLUserData:
        return self.user_data[user_index]

    def train_users(self) -> Iterable[IFLUserData]:
        return self.user_data

    def eval_users(self) -> Iterable[IFLUserData]:
        return [self.eval]

    def test_users(self) -> Iterable[IFLUserData]:
        return [self.eval]
