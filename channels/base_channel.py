#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

import abc
from dataclasses import dataclass

from flsim.channels.message import ChannelMessage
from flsim.interfaces.model import IFLModel
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.model_size_utils import calc_model_size


class IFLChannel(abc.ABC):
    """
    Base interface for `IFLChannel` that takes care of transmitting messages
    between clients and the server and collecting some metrics on the way.

    This is by nature a *bi-directional* channel (server to client and
    client to server).

    Notes:
        - It is the responsibility of the trainer (see trainer_base) to call
          ``attach_stats_collector`` to attach its ``ChannelStatsCollector``
          to the channel.
    """

    def __init__(self):
        self.stats_collector = None

    @abc.abstractmethod
    def create_channel_message(self, model: IFLModel):
        """
        Creates a channel message. This message is used to communicate with
        ServerChannelEndPoint
        """
        pass

    @abc.abstractmethod
    def server_to_client(self, message: ChannelMessage) -> ChannelMessage:
        """
        Simulates the manipulation and transmission of a `ChannelMessage` from
        the server to a client. Also handles relevant stats accounting.
        """

        pass

    @abc.abstractmethod
    def client_to_server(self, message: ChannelMessage) -> ChannelMessage:
        """
        Simulates the manipulation and transmission of a `ChannelMessage` from
        a client to the server. Also handles relevant stats accounting.
        """

        pass

    def attach_stats_collector(self, stats_collector):
        """
        Allows for measurements of the average message size from client to
        server and from server to client.
        """

        self.stats_collector = stats_collector


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

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=FLChannelConfig,
            **kwargs,
        )
        self.stats_collector = None

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def _on_client_before_transmission(self, message: ChannelMessage) -> ChannelMessage:
        """
        Implements message manipulation that would be done on a client in the real world
        just before transmitting a message to the server.
        """
        return message

    def _during_transmission_client_to_server(
        self, message: ChannelMessage
    ) -> ChannelMessage:
        """
        Manipulation to the message in transit from client to server.
        """

        if self.stats_collector:
            message_size_bytes = calc_model_size(message.model_state_dict)
            self.stats_collector.collect_channel_stats(
                message_size_bytes, client_to_server=True
            )
        return message

    def _on_server_after_reception(self, message: ChannelMessage) -> ChannelMessage:
        """
        Implements message manipulation that would be done on the server in the real world
        just after receiving a message from a client.
        """
        return message

    def _on_server_before_transmission(self, message: ChannelMessage) -> ChannelMessage:
        """
        Implements message manipulation that would be done on the server in the real world
        just before transmitting a message to a client.
        """
        return message

    def _during_transmission_server_to_client(
        self, message: ChannelMessage
    ) -> ChannelMessage:
        """
        Manipulation to the message in transit from server to client.
        """
        if self.stats_collector:
            message_size_bytes = calc_model_size(message.model_state_dict)
            self.stats_collector.collect_channel_stats(
                message_size_bytes, client_to_server=False
            )
        return message

    def _on_client_after_reception(self, message: ChannelMessage) -> ChannelMessage:
        """
        Implements message manipulation that would be done on a client in the real world
        just after receiving a message from the server.
        """
        return message

    def create_channel_message(self, model: IFLModel) -> ChannelMessage:
        """
        In an IndentiyChannel, we will not copy the model state_dict as
        loading and saving state_dict is a heavy operation and introduces
        unnecessary latency.
        """
        message = ChannelMessage()
        return message

    def client_to_server(self, message: ChannelMessage) -> ChannelMessage:
        """
        Performs three successive steps to send a message from a client to the server.
        """

        # process through channel
        message = self._on_client_before_transmission(message)
        message = self._during_transmission_client_to_server(message)
        message = self._on_server_after_reception(message)

        return message

    def server_to_client(self, message: ChannelMessage) -> ChannelMessage:
        """
        Performs three successive steps to send a message from the server to a client.
        """

        # process through channel
        message = self._on_server_before_transmission(message)
        message = self._during_transmission_server_to_client(message)
        message = self._on_client_after_reception(message)

        return message

    def get_client_channel_endpoint(self) -> ClientChannelEndPoint:
        """
        From the perspective of a client, this simulates a *single direction*
        channel with only one public method regarding message transmission:
        receive (from the server). Under the hood, receive will call the method
        ``client_to_server``.
        """

        return ClientChannelEndPoint(self)

    def get_server_channel_endpoint(self) -> ServerChannelEndPoint:
        """
        From the perspective of the server, this simulates a *single direction*
        channel with only one public method regarding message transmission:
        receive (from a client). Under the hood, receive will call the method
        ``client_to_server``.
        """

        return ServerChannelEndPoint(self)


class ServerChannelEndPoint:
    """
    From the perspective of the server, this simulates a *single direction*
    channel with only one public method: receive (from a client). Under
    the hood, receive will call the method ``client_to_server``.

    The goal is to avoid confusion between the ``server_to_client``
    and ``client_to_server`` methods by having two distinct end points,
    ClientChannelEndPoint and ServerChannelEndPoint with a single
    method regarding message transmission: receive.
    """

    def __init__(self, channel: IFLChannel):
        self._channel = channel

    def receive(self, message: ChannelMessage) -> ChannelMessage:
        return self._channel.client_to_server(message)

    def attach_stats_collector(self, stats_collector):
        self._channel.attach_stats_collector(stats_collector)


class ClientChannelEndPoint:
    """
    From the perspective of a client, this simulates a *single direction*
    channel with only one public method: receive (from the server). Under
    the hood, receive will call the method ``client_to_server``.

    The goal is to avoid confusion between the ``server_to_client``
    and ``client_to_server`` methods by having two distinct end points,
    ClientChannelEndPoint and ServerChannelEndPoint with a single
    method regarding message transmission: receive.
    """

    def __init__(self, channel: IFLChannel):
        self._channel = channel

    def receive(self, message: ChannelMessage) -> ChannelMessage:
        return self._channel.server_to_client(message)

    def attach_stats_collector(self, stats_collector):
        self._channel.attach_stats_collector(stats_collector)


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
