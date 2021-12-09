#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import OrderedDict

from flsim.channels.communication_stats import ChannelStatsCollector
from flsim.channels.message import Message
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg


class IFLChannel(abc.ABC):
    """
    Base interface for `IFLChannel` that takes care of transmitting messages
    between clients and the server and collecting some metrics on the way.

    This is by nature a *bi-directional* channel (server to client and
    client to server).

    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=FLChannelConfig,
            **kwargs,
        )
        self.stats_collector = (
            ChannelStatsCollector() if self.cfg.report_communication_metrics else None
        )

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    @abc.abstractmethod
    def server_to_client(self, message: Message) -> Message:
        """
        Simulates the manipulation and transmission of a `Message` from
        the server to a client. Also handles relevant stats accounting.
        """
        pass

    @abc.abstractmethod
    def client_to_server(self, message: Message) -> Message:
        """
        Simulates the manipulation and transmission of a `Message` from
        a client to the server. Also handles relevant stats accounting.
        """

        pass


class IdentityChannel(IFLChannel):
    """
    Implements a *bi-directional* channel which is pass-through: the message sent
    is identical to the message received when sending from the server to a
    client and when sending from a client to the server.

    It simulates what happens in the real world. For instance, with the
    direction client to server, there are three methods:
        - ``_on_client_before_transmission``: this is what happens in the client
          right before sending the message, such as compression.
        - ``_during_transmission_client_to_server``: this is what happens during the
          transmission of the message, such as communication measurements,
          communication noise.
        - ``_on_server_after_reception``: this is what happens first when the server
          receives the message, such as decompression.

    Notes:
        - We recommend that all channels inherit `IdentityChannel` and override
          only the necessary methods.
        - Each new channel is responsible for the measurement of its message size,
          i.e. it is the user's responsibility to override the right methods.
    """

    # defining useful general constants for channel size measurements
    BYTES_PER_FP32 = 4
    BYTES_PER_FP64 = 8
    BYTES_PER_INT64 = 8
    BYTES_PER_BIT = 0.125

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    @classmethod
    def calc_model_size_float_point(cls, state_dict: OrderedDict):
        """
        Calculates model size in bytes given a state dict.
        """
        model_size_bytes = sum(
            p.numel() * p.element_size() for (_, p) in state_dict.items()
        )
        return model_size_bytes

    def _calc_message_size_client_to_server(self, message: Message):
        return self.calc_model_size_float_point(message.model_state_dict)

    def _calc_message_size_server_to_client(self, message: Message):
        return self.calc_model_size_float_point(message.model_state_dict)

    def _on_client_before_transmission(self, message: Message) -> Message:
        """
        Implements message manipulation that would be done on a client in the real world
        just before transmitting a message to the server.
        """
        return message

    def _during_transmission_client_to_server(self, message: Message) -> Message:
        """
        Manipulation to the message in transit from client to server.
        """

        if self.stats_collector:
            message_size_bytes = self._calc_message_size_client_to_server(message)
            self.stats_collector.collect_channel_stats(
                message_size_bytes, client_to_server=True
            )
        return message

    def _on_server_after_reception(self, message: Message) -> Message:
        """
        Implements message manipulation that would be done on the server in the real world
        just after receiving a message from a client.
        """
        return message

    def _on_server_before_transmission(self, message: Message) -> Message:
        """
        Implements message manipulation that would be done on the server in the real world
        just before transmitting a message to a client.
        """
        return message

    def _during_transmission_server_to_client(self, message: Message) -> Message:
        """
        Manipulation to the message in transit from server to client.
        """
        if self.stats_collector:
            message_size_bytes = self._calc_message_size_server_to_client(message)
            self.stats_collector.collect_channel_stats(
                message_size_bytes, client_to_server=False
            )
        return message

    def _on_client_after_reception(self, message: Message) -> Message:
        """
        Implements message manipulation that would be done on a client in the real world
        just after receiving a message from the server.
        """
        return message

    def client_to_server(self, message: Message) -> Message:
        """
        Performs three successive steps to send a message from a client to the server.
        """
        # process through channel
        message = self._on_client_before_transmission(message)
        message = self._during_transmission_client_to_server(message)
        message = self._on_server_after_reception(message)

        return message

    def server_to_client(self, message: Message) -> Message:
        """
        Performs three successive steps to send a message from the server to a client.
        """

        # process through channel
        message = self._on_server_before_transmission(message)
        message = self._during_transmission_server_to_client(message)
        message = self._on_client_after_reception(message)

        return message


@dataclass
class FLChannelConfig:
    """
    Base Config for a channel defining channel properties
    such as channel drop rate, quantization effects,
    channel bandwidth settings, etc.
    """

    # Add attributes as needed.
    _target_: str = fullclassname(IdentityChannel)
    _recursive_: bool = False
    # Whether communication metrics (between server and clients) should be
    # reported
    report_communication_metrics: bool = False
