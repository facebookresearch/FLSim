#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum, auto

from flsim.utils.fl.stats import RandomVariableStatsTracker


class ChannelDirection(Enum):
    CLIENT_TO_SERVER = auto()
    SERVER_TO_CLIENT = auto()


class ChannelStatsCollector:
    def __init__(self):
        self.reset_channel_stats()

    def reset_channel_stats(self):
        self.communication_stats = {
            ChannelDirection.CLIENT_TO_SERVER: RandomVariableStatsTracker(),
            ChannelDirection.SERVER_TO_CLIENT: RandomVariableStatsTracker(),
        }

    def get_channel_stats(self):
        return self.communication_stats

    def collect_channel_stats(
        self, message_size_bytes: float, client_to_server: bool = True
    ):
        """
        Collect statistics about the updates/model transmitted both
        for client to server and server to client directions.
        """
        direction = (
            ChannelDirection.CLIENT_TO_SERVER
            if client_to_server
            else ChannelDirection.SERVER_TO_CLIENT
        )
        self.communication_stats[direction].update(message_size_bytes)
