#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from copy import deepcopy
from dataclasses import dataclass, field
from typing import OrderedDict

import torch.nn as nn
from flsim.interfaces.model import IFLModel
from flsim.utils.count_sketch import CountSketch
from torch import Tensor


@dataclass
class ChannelMessage:
    """
    Message that is sent from a client to a server and vice-versa through
    a channel.

    We discourage calling `ChannelMessage` in the code directly and instead
    we advise to construct a `ChannelMessage` using

    ```
    channel_message = channel.create_channel_message(model)
    ```

    Notes:
      - The packet may change after transmission through a channel. For
        instance, the SketchChannel takes as input a model state dict
        and outputs a CountSketch (but no model state dict).
    """

    # here we store state dict for conveninence
    model_state_dict: OrderedDict[str, Tensor] = field(
        default_factory=OrderedDict[str, Tensor]
    )
    # count sketch
    count_sketch: CountSketch = field(default_factory=CountSketch)

    def populate(self, model: IFLModel, **kwargs):
        """
        We copy the model's state dict and add is as attribute to the message.

        Notes:
          - We deepcopy the state dict to avoid side effects in case we manipulate
            the state dict inplace.
          - We rely on a model's state dict as it will be easier to change the
            type of the underlying tensors (say int8) versus replacing every
            nn.Module with its corresponding counterpart.
        """

        self.model_state_dict = deepcopy(model.fl_get_module().state_dict())

    def update_model_(self, model: IFLModel):
        """
        Updates model with the state dict stored in the message. May be useful
        when receiving a `ChannelMessage` and wanting to update the local model.
        """

        model.fl_get_module().load_state_dict(self.model_state_dict)


@dataclass
class Message:
    """
    Generic message dataclass, composed of:
        - model: a model containing information that will be sent
        - meta: any meta information about a client or a server
          for instance.

    This dataclass can be extended to your custom needs see the
    ``ChannelMessage` example.
    """

    # model
    model: IFLModel = field(default_factory=nn.Module)

    # add any meta information here
    weight: float = field(default_factory=float)


@dataclass(frozen=True)
class SyncServerMessage:
    """
    TODO@john, adapt this with the new Message framework
    """

    delta: nn.Module
    weight: float
