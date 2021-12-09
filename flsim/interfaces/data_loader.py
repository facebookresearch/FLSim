#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
