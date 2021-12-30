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
