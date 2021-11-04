#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from flsim.channels.base_channel import (
    IdentityChannel,
    FLChannelConfig,
    ChannelMessage,
)
from flsim.interfaces.model import IFLModel
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg


class HalfPrecisionChannel(IdentityChannel):
    """
    Implements a channel that emulates upload of model from client
    to server using half precision (fp16). We emulate this by
    casting successively to half() and back to float(). This way,
    the rest of the training is transparent for aggreagators, reducers,
    trainers and so on.

    Note that model.half().float() is *not* in general a no-operation,
    since we lose a bit of information when casting to half (and that
    is the desired behavior). That's why the corresponding channel_test
    is done with a larger `atol` in the direction client->server.

    We do not override the method ``_during_transmission_client_to_server``
    from IdentityChannel since it can be applied here without modification to
    measure the message size. Inded, it relies on ``element_size()``, which is,
    as expected, 2 bytes for half precision and 4 bytes for fp32.
    """

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=HalfPrecisionChannelConfig,
            **kwargs,
        )
        self.stats_collector = None

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def create_channel_message(self, model: IFLModel) -> ChannelMessage:
        message = ChannelMessage()
        message.populate(model)
        return message

    def _on_client_before_transmission(self, message: ChannelMessage) -> ChannelMessage:
        """
        Here we cast all parameters to half precision. We copy the
        state dict since the tensor format changes.
        """
        new_state_dict = OrderedDict()
        for name, param in message.model_state_dict.items():
            new_state_dict[name] = param.data.half()

        message.model_state_dict = new_state_dict
        return message

    def _on_server_after_reception(self, message: ChannelMessage) -> ChannelMessage:
        """
        We decompress the message by casting back all parameters
        to full precision (fp32). Note that this is not a no-op
        since there is some loss in precision when casting to half().
        We copy the state dict since the tensor format changes.
        """
        new_state_dict = OrderedDict()
        for name, param in message.model_state_dict.items():
            new_state_dict[name] = param.data.float()

        message.model_state_dict = new_state_dict
        return message


@dataclass
class HalfPrecisionChannelConfig(FLChannelConfig):
    _target_: str = fullclassname(HalfPrecisionChannel)
