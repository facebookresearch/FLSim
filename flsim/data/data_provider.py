#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator, List, Optional

from flsim.interfaces.model import IFLModel


class IFLUserData(ABC):
    """
    Wraps data for a single user

    IFLUserData is responsible for
        1. Keeping track of the number of examples for a particular user
        2. Keeping track of the number of batches for a particular user
        3. Providing an iterator over all the user batches
    """

    def num_total_examples(self) -> int:
        """
        Returns the number of examples
        """
        return self.num_train_examples() + self.num_eval_examples()

    def num_total_batches(self) -> int:
        """
        Returns the number of batches
        """
        return self.num_train_batches() + self.num_eval_batches()

    @abstractmethod
    def num_train_examples(self) -> int:
        """
        Returns the number of train examples
        """

    @abstractmethod
    def num_eval_examples(self) -> int:
        """
        Returns the number of eval examples
        """

    @abstractmethod
    def num_train_batches(self) -> int:
        """
        Returns the number of training batches
        """

    @abstractmethod
    def num_eval_batches(self) -> int:
        """
        Returns the number of eval batches
        """

    @abstractmethod
    def train_data(self) -> Iterator[Any]:
        """
        Returns the training batches
        """

    @abstractmethod
    def eval_data(self) -> Iterator[Any]:
        """
        Returns the eval batches
        """


class IFLDataProvider(ABC):
    """
    Provides data to the trainer

    IFLDataProvider is responsible for
        1. Enforcing a uniform interface that trainer expects
        2. Transforming data into what IFLModel.fl_forward() is going to consume
        3. Keeping track of the sharded client data
    """

    @abstractmethod
    def train_user_ids(self) -> List[int]:
        """
        Returns a list of user ids in the data set
        """

    @abstractmethod
    def num_train_users(self) -> int:
        """
        Returns the number of users in train set
        """

    @abstractmethod
    def get_train_user(self, user_index: int) -> IFLUserData:
        """
        Returns train user from user_index
        """

    @abstractmethod
    def train_users(self) -> Iterable[IFLUserData]:
        """
        Returns training users iterable
        """

    @abstractmethod
    def eval_users(self) -> Iterable[IFLUserData]:
        """
        Returns evaluation users iterable
        """

    @abstractmethod
    def test_users(self) -> Iterable[IFLUserData]:
        """
        Returns test users iterable
        """


class FLUserDataFromList(IFLUserData):
    """
    Util class to create an IFLUserData from a list of user batches
    """

    def __init__(
        self, data: Iterable, model: IFLModel, eval_batches: Optional[Iterable] = None
    ):
        self.data = data
        self._num_examples: int = 0
        self._num_batches: int = 0
        self.model = model
        self.training_batches = []
        self.eval_batches = eval_batches if eval_batches is not None else []
        self._num_eval_batches: int = 0
        self._num_eval_examples: int = 0

        for batch in self.data:
            training_batch = self.model.fl_create_training_batch(batch=batch)
            self.training_batches.append(training_batch)
            self._num_examples += model.get_num_examples(training_batch)
            self._num_batches += 1

        for batch in self.eval_batches:
            eval_batch = self.model.fl_create_training_batch(batch=batch)
            self._num_eval_examples += model.get_num_examples(eval_batch)
            self._num_eval_batches += 1

    def train_data(self):
        for batch in self.training_batches:
            yield batch

    def eval_data(self):
        for batch in self.eval_batches:
            yield self.model.fl_create_training_batch(batch=batch)

    def num_batches(self):
        return self._num_batches

    def num_train_examples(self):
        return self._num_examples

    def num_eval_batches(self):
        return self._num_eval_batches

    def num_train_batches(self):
        return self._num_batches

    def num_eval_examples(self):
        return self._num_eval_examples


class FLDataProviderFromList(IFLDataProvider):
    """Utility class to help ease the transition to IFLDataProvider

    Args:
        train_user_list: (Iterable[Iterable[Any]]): train data
        eval_user_list: (Iterable[Iterable[Any]]): eval data
        test_user_list (Iterable[Iterable[Any]]): test data
        model: (IFLModel): the IFLModel to create training batch for
    """

    def __init__(
        self,
        train_user_list: Iterable[Iterable[Any]],
        eval_user_list: Iterable[Iterable[Any]],
        test_user_list: Iterable[Iterable[Any]],
        model: IFLModel,
    ):
        self.train_user_list = train_user_list
        self.eval_user_list = eval_user_list
        self.test_user_list = test_user_list
        self.model = model
        self._train_users = {
            user_id: FLUserDataFromList(
                data=user_data, eval_batches=user_data, model=model
            )
            for user_id, user_data in enumerate(train_user_list)
        }
        self._eval_users = {
            user_id: FLUserDataFromList(data=[], eval_batches=user_data, model=model)
            for user_id, user_data in enumerate(eval_user_list)
        }
        self._test_users = {
            user_id: FLUserDataFromList(data=[], eval_batches=user_data, model=model)
            for user_id, user_data in enumerate(test_user_list)
        }

    def train_user_ids(self):
        """List of all train user IDs."""
        return list(self._train_users.keys())

    def num_train_users(self):
        """Number of train users."""
        return len(self.train_user_ids())

    def get_train_user(self, user_index: int):
        """Returns a train user given user index."""
        if user_index in self._train_users:
            return self._train_users[user_index]
        else:
            raise IndexError(f"Index {user_index} not in {self.train_user_ids()}")

    def train_users(self):
        """Returns a list of all train users."""
        return list(self._train_users.values())

    def eval_users(self):
        """Returns a list of all eval users."""
        return list(self._eval_users.values())

    def test_users(self):
        """Returns a list of test users."""
        return list(self._test_users.values())
