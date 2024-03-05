#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, OrderedDict, Tuple

import torch.nn as nn
from flsim.interfaces.model import IFLModel
from torch import Tensor


@dataclass
class Message:
    """
    Generic message dataclass, composed of:
        - model: a model containing information that will be sent
        - meta: any meta information about a client or a server
          for instance.

    This dataclass can be extended to your custom needs see the
    ``Message` example.

    Notes:
      - The packet may change after transmission through a channel. For
        instance, the SketchChannel takes as input a model state dict
        and outputs a CountSketch (but no model state dict).
    """

    # model
    model: IFLModel = field(default_factory=nn.Module)

    # add any meta information here
    weight: float = field(default_factory=float)

    # here we store state dict for convenience
    model_state_dict: OrderedDict[str, Tensor] = field(
        default_factory=OrderedDict[str, Tensor], init=True
    )
    # count sketch
    count_sketch: Optional[Any] = field(default=None)

    # MIME Specific Information below:
    # Store servers global optimizer state
    server_opt_state: Optional[Dict] = field(default=None)
    # Stores average gradient of the global_model over all client training datasets
    mime_control_variate: Optional[nn.Module] = field(default=None)

    # meta data for scalar quantization
    qparams: Optional[Dict[str, Tuple[Tensor, Tensor]]] = field(default=None)

    # meta data for product quantization
    seed_centroids: Optional[Dict[str, Tensor]] = field(default=None)

    # meta data for sparsity masks
    sparsity_mask_params: Optional[Dict[str, Tensor]] = field(default=None)

    global_round_num: int = field(default_factory=int)

    def populate_state_dict(self, **kwargs):
        """
        We copy the model's state dict and add it as an attribute to the message.

        Notes:
          - We deepcopy the state dict to avoid side effects in case we manipulate
            the state dict in place.
          - We rely on a model's state dict as it will be easier to change the
            type of the underlying tensors (say int8) versus replacing every
            nn.Module with its corresponding counterpart.
        """

        self.model_state_dict = deepcopy(self.model.fl_get_module().state_dict())

    def update_model_(self):
        """
        Updates model with the state dict stored in the message. May be useful
        when receiving a `Message` and wanting to update the local model.
        """
        assert (
            self.model_state_dict
        ), "Message state dict is empty. Please check if message.state_dict is populated."
        self.model.fl_get_module().load_state_dict(self.model_state_dict)
