#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from flsim.data.data_provider import IFLDataProvider, IFLUserData


@dataclass
class AsyncUserSelectorInfo:
    r"""
    Dataclass to encapsulate a selected user for async training

    user_data (IFLUserData): seleected user data in the dataset
    user_index (int): the index for user_data assuming IFLDataProvider.train_data is a List
    """
    user_data: IFLUserData
    user_index: int


class AsyncUserSelector(abc.ABC):
    def __init__(self, data_provider: IFLDataProvider):
        self.data_provider: IFLDataProvider = data_provider

    @abc.abstractmethod
    def get_random_user(self) -> AsyncUserSelectorInfo:
        r"""
        Returns a random IFLUserData from the dataset and the user index (for testing)
        """
        pass


class RandomAsyncUserSelector(AsyncUserSelector):
    def __init__(self, data_provider: IFLDataProvider):
        super().__init__(data_provider)

    def get_random_user(self) -> AsyncUserSelectorInfo:
        user_index = np.random.randint(0, self.data_provider.num_users())
        return AsyncUserSelectorInfo(self.data_provider[user_index], user_index)


class RoundRobinAsyncUserSelector(AsyncUserSelector):
    r"""
    Chooses users in round-robin order, starting from user=0.
    Particularly useful for testing.
    """

    def __init__(self, data_provider: IFLDataProvider):
        super().__init__(data_provider)
        self.current_user_index: int = 0

    def get_random_user(self) -> AsyncUserSelectorInfo:
        user_index = self.current_user_index
        self.current_user_index = (
            self.current_user_index + 1
        ) % self.data_provider.num_users()
        return AsyncUserSelectorInfo(self.data_provider[user_index], user_index)


class AsyncUserSelectorType(Enum):
    RANDOM = auto()
    ROUND_ROBIN = auto()


class AsyncUserSelectorFactory:
    @classmethod
    def create_users_selector(
        cls, type: AsyncUserSelectorType, data_provider: IFLDataProvider
    ):
        if type == AsyncUserSelectorType.RANDOM:
            return RandomAsyncUserSelector(data_provider)
        elif type == AsyncUserSelectorType.ROUND_ROBIN:
            return RoundRobinAsyncUserSelector(data_provider)
        else:
            raise AssertionError(f"Unknown user selector type: {type}")
