#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import abc
from typing import Any, Iterable


class IFLDataLoader(abc.ABC):
    @abc.abstractmethod
    def fl_train_set(self, **kwargs) -> Iterable[Any]:
        pass

    @abc.abstractmethod
    def fl_eval_set(self, **kwargs) -> Iterable[Any]:
        pass

    @abc.abstractmethod
    def fl_test_set(self, **kwargs) -> Iterable[Any]:
        pass
