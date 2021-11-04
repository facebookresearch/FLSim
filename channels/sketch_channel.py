#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from flsim.channels.base_channel import IdentityChannel, FLChannelConfig
from flsim.channels.message import ChannelMessage
from flsim.common.logger import Logger
from flsim.interfaces.model import IFLModel
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.count_sketch import CountSketch


class SketchChannel(IdentityChannel):
    """
    Implements a channel which sketches/compresses the model sent
    from client to server.
    """

    logger: logging.Logger = Logger.get_logger(__name__)
    random_seed = None

    def __init__(self, **kwargs):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=SketchChannelConfig,
            **kwargs,
        )
        self.num_col = self.cfg.num_col
        self.num_hash = self.cfg.num_hash
        self.prime = self.cfg.prime
        self.independence = self.cfg.independence

        self.stats_collector = None
        self.h = torch.randint(
            low=1, high=self.cfg.prime, size=(self.cfg.num_hash, self.cfg.independence)
        )
        self.g = torch.randint(
            low=1, high=self.cfg.prime, size=(self.cfg.num_hash, self.cfg.independence)
        )

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        pass

    def create_channel_message(self, model: IFLModel) -> ChannelMessage:
        message = ChannelMessage()
        message.populate(model)
        return message

    def _on_client_before_transmission(self, message: ChannelMessage) -> ChannelMessage:
        """
        Curerntly sends whole CountSketch class. We could only share relevant attributes
        to re-instantiate the class on the server side, TODO in a next refactor.
        """

        device = next(iter(message.model_state_dict.values())).device
        cs = CountSketch(
            width=self.num_col,
            depth=self.num_hash,
            prime=self.prime,
            independence=self.independence,
            h=self.h,
            g=self.g,
            device=device,
        )
        cs.sketch_state_dict(message.model_state_dict)
        message.count_sketch = cs

        return message

    def _during_transmission_client_to_server(
        self, message: ChannelMessage
    ) -> ChannelMessage:
        """
        Raise an error here is we want to measure message size, TODO: fix
        this in a next refactor.
        """
        if self.stats_collector:
            raise NotImplementedError(
                "Channel size measurement not implemented for CountSketch"
            )

        return message


@dataclass
class SketchChannelConfig(FLChannelConfig):
    _target_: str = fullclassname(SketchChannel)
    # number of columns in the table
    num_col: int = 10000
    # number of hash functions to use, also the number of rows in the table
    num_hash: int = 11
    # prime number used in hash functions
    prime: int = 2 ** 31 - 1
    independence: int = 2
