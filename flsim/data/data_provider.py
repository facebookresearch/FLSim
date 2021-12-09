#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator, List

from flsim.interfaces.model import IFLModel


class IFLData(ABC):
    """
    Wraps data for a single entity.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """
        Iterator to return a user batch data
        """

    @abstractmethod
    def num_examples(self) -> int:
        """
        Returns the number of examples
        """

    @abstractmethod
    def num_batches(self) -> int:
        """
        Returns the number of batches
        """


class IFLUserData(IFLData):
    """
    Wraps data for a single user

    IFLUserData is responsible for
        1. Keeping track of the number of examples for a particular user
        2. Keeping track of the number of batches for a particular user
        3. Providing an iterator over all the user batches
    """


class IFLDataProvider(ABC):
    """
    Provides data to the trainer

    IFLDataProvider is resposible for
        1. Enforcing a uniform interface that trainer expects
        2. Transforming data into what IFLModel.fl_forward() is going to consume
        3. Keeping track of the sharded client data
    """

    def __iter__(self) -> Iterable[IFLUserData]:
        """
        Returns training data iterable
        """
        yield from self.train_data()

    def __getitem__(self, index) -> IFLUserData:
        """
        Returns a user from index
        """
        return self.get_user_data(index)

    @abstractmethod
    def user_ids(self) -> List[int]:
        """
        Returns a list of user ids in the data set
        """

    @abstractmethod
    def num_users(self) -> int:
        """
        Returns the number of users in train set
        """

    @abstractmethod
    def get_user_data(self, user_index: int) -> IFLUserData:
        """
        Returns user data from user_index
        """

    @abstractmethod
    def train_data(self) -> Iterable[IFLUserData]:
        """
        Returns training data iterable
        """

    @abstractmethod
    def eval_data(self) -> Iterable[Any]:
        """
        Returns evaluation data iterable
        """

    @abstractmethod
    def test_data(self) -> Iterable[Any]:
        """
        Returns test data iterable
        """


class FLUserDataFromList(IFLUserData):
    """
    Util class to create an IFLUserData from a list of user batches
    """

    def __init__(self, data: Iterable, model: IFLModel):
        self.data = data
        self._num_examples: int = 0
        self._num_batches: int = 0
        self.model = model
        for batch in self.data:
            training_batch = self.model.fl_create_training_batch(batch=batch)
            self._num_examples += model.get_num_examples(training_batch)
            self._num_batches += 1

    def __iter__(self):
        for batch in self.data:
            yield self.model.fl_create_training_batch(batch=batch)

    def num_batches(self):
        return self._num_batches

    def num_examples(self):
        return self._num_examples


class FLDataProviderFromList(IFLDataProvider):
    r"""
    Util class to help ease the transition to IFLDataProvider

    =======
    Args:
        train_user_list: (Iterable[Iterable[Any]]): train data
        eval_user_list: (Iterable[Any]): eval data
        test_user_list (Iterable[Any]): test data
        model: (IFLModel): the IFLModel to create training batch for
    """

    def __init__(
        self,
        train_user_list: Iterable[Iterable[Any]],
        eval_user_list: Iterable[Any],
        test_user_list: Iterable[Any],
        model: IFLModel,
    ):
        self.train_user_list = train_user_list
        self.eval_user_list = eval_user_list
        self.test_user_list = test_user_list
        self.model = model
        self._users = {
            user_id: FLUserDataFromList(user_data, model)
            for user_id, user_data in enumerate(train_user_list)
        }

    def user_ids(self):
        return list(self._users.keys())

    def num_users(self):
        return len(self.user_ids())

    def get_user_data(self, user_index: int):
        if user_index in self._users:
            return self._users[user_index]
        else:
            raise IndexError(f"Index {user_index} not in {self.user_ids()}")

    def train_data(self):
        return list(self._users.values())

    def eval_data(self):
        return [
            self.model.fl_create_training_batch(batch=batch)
            for batch in self.eval_user_list
        ]

    def test_data(self):
        return [
            self.model.fl_create_training_batch(batch=batch)
            for batch in self.test_user_list
        ]
