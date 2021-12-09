#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional


class ProcessState:
    _instance: Optional["ProcessState"] = None

    @staticmethod
    def getInstance(**kwargs):
        """kwargs should specify:
        rank: int, workflow_name: Optional[str], chronos_id: Optional[int]
        """
        if ProcessState._instance is None:
            ProcessState(**kwargs)
        return ProcessState._instance

    def __init__(self, rank: int):
        """
        Virtually private constructor.
        Handles logic for Singleton pattern.
        """
        self._rank = rank
        if ProcessState._instance is not None:
            raise RuntimeError(
                "ProcessState is a singleton. Cannot instantiate multiple times!"
            )
        else:
            ProcessState._instance = self

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, value):
        if self._rank is not None:
            raise RuntimeError("Shouldn't change 'rank' after initialized")
        self._rank = value
