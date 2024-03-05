#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from flsim.active_user_selectors.simple_user_selector import (
    ActiveUserSelectorConfig,
    UniformlyRandomActiveUserSelectorConfig,
)
from flsim.channels.base_channel import IdentityChannel, IFLChannel
from flsim.channels.message import Message
from flsim.channels.scalar_quantization_channel import ScalarQuantizationChannel
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.model import IFLModel
from flsim.optimizers.server_optimizers import FedAvgOptimizerConfig
from flsim.secure_aggregation.secure_aggregator import (
    FixedPointConfig,
    SecureAggregator,
    utility_config_flatter,
)
from flsim.servers.aggregator import AggregationType, Aggregator
from flsim.servers.sync_servers import ISyncServer, SyncServerConfig
from flsim.utils.config_utils import fullclassname, init_self_cfg
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import Tensor


class SyncSecAggServer(ISyncServer):
    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[IFLChannel] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=SyncSecAggServerConfig,
            **kwargs,
        )
        self._optimizer = instantiate(
            # pyre-fixme[16]: `SyncServer` has no attribute `cfg`.
            config=self.cfg.server_optimizer,
            model=global_model.fl_get_module(),
        )
        self._global_model: IFLModel = global_model
        self._aggregator: Aggregator = Aggregator(
            module=global_model.fl_get_module(),
            aggregation_type=self.cfg.aggregation_type,
            only_federated_params=self.cfg.only_federated_params,
        )
        self._secure_aggregator = SecureAggregator(
            utility_config_flatter(
                global_model.fl_get_module(),
                self.cfg.fixedpoint,
            )
        )
        self._active_user_selector = instantiate(self.cfg.active_user_selector)
        self._channel: IFLChannel = channel or IdentityChannel()

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        if OmegaConf.is_missing(cfg.active_user_selector, "_target_"):
            cfg.active_user_selector = UniformlyRandomActiveUserSelectorConfig()
        if OmegaConf.is_missing(cfg.server_optimizer, "_target_"):
            cfg.server_optimizer = FedAvgOptimizerConfig()

    @property
    def global_model(self):
        return self._global_model

    def select_clients_for_training(
        self,
        num_total_users,
        users_per_round,
        data_provider: Optional[IFLDataProvider] = None,
        global_round_num: Optional[int] = None,
    ):
        return self._active_user_selector.get_user_indices(
            num_total_users=num_total_users,
            users_per_round=users_per_round,
            data_provider=data_provider,
            global_round_num=global_round_num,
        )

    def init_round(self):
        self._aggregator.zero_weights()
        self._optimizer.zero_grad()

    def receive_update_from_client(self, message: Message):
        message = self._channel.client_to_server(message)

        self._aggregator.apply_weight_to_update(
            delta=message.model.fl_get_module(), weight=message.weight
        )

        self._secure_aggregator.params_to_fixedpoint(message.model.fl_get_module())
        self._secure_aggregator.apply_noise_mask(
            message.model.fl_get_module().named_parameters()
        )
        self._aggregator.add_update(
            delta=message.model.fl_get_module(), weight=message.weight
        )
        self._secure_aggregator.update_aggr_overflow_and_model(
            model=self._aggregator._buffer_module
        )

    def step(self):
        aggregated_model = self._aggregator.aggregate()
        self._secure_aggregator.apply_denoise_mask(aggregated_model.named_parameters())
        self._secure_aggregator.params_to_float(aggregated_model)

        FLModelParamUtils.set_gradient(
            model=self._global_model.fl_get_module(),
            reference_gradient=aggregated_model,
        )
        self._optimizer.step()

    def calc_avg_overflow_percentage(
        self,
        users_per_round: int,
        model: IFLModel,
        report_rounds: int,
    ) -> Tuple[float, float]:
        return self._secure_aggregator.calc_avg_overflow_percentage(
            users_per_round, model.fl_get_module(), report_rounds
        )


class SyncSecAggSQServer(SyncSecAggServer):
    def __init__(
        self,
        *,
        global_model: IFLModel,
        channel: Optional[ScalarQuantizationChannel] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,
            config_class=SyncSecAggSQServerConfig,
            **kwargs,
        )
        # perform all the parental duties
        super().__init__(global_model=global_model, channel=channel, **kwargs)
        # ensure correct channel is used for SQ
        if not isinstance(self._channel, ScalarQuantizationChannel):
            raise TypeError(
                "SyncSecAggSQServer expects channel of type ScalarQuantizationChannel,",
                f" {type(self._channel)} given.",
            )
        # ensure correct qparam sharing is used for secagg
        if not self._channel.use_shared_qparams:
            raise ValueError(
                "SyncSecAggSQServer expects qparams to be shared across all clients."
                " Have you set sec_agg_mode to True in channel config?"
            )
        # set scaling factor for quantized params
        for n, p in self.global_model.fl_get_module().named_parameters():
            # non-bias parameters are assumed to be quantized when using SQ channel
            if p.ndim > 1:
                self._secure_aggregator.converters[n].scaling_factor = (
                    # pyre-ignore [16]
                    self.cfg.secagg_scaling_factor_for_quantized
                )
        # set global qparams (need to be empty at the beginning of every round)
        self._global_qparams: Dict[str, Tuple[Tensor, Tensor]] = {}

    @property
    def global_qparams(self):
        return self._global_qparams

    def receive_update_from_client(self, message: Message):
        message.qparams = self.global_qparams
        message = self._channel.client_to_server(message)

        self._aggregator.apply_weight_to_update(
            delta=message.model.fl_get_module(), weight=message.weight
        )
        # params that are in int form are being converted to fixedpoint
        self._secure_aggregator.params_to_fixedpoint(message.model.fl_get_module())
        self._secure_aggregator.apply_noise_mask(
            message.model.fl_get_module().named_parameters()
        )
        self._aggregator.add_update(
            delta=message.model.fl_get_module(), weight=message.weight
        )
        self._secure_aggregator.update_aggr_overflow_and_model(
            model=self._aggregator._buffer_module
        )

    def step(self):
        aggregated_model = self._aggregator.aggregate()
        self._secure_aggregator.apply_denoise_mask(aggregated_model.named_parameters())
        self._secure_aggregator.params_to_float(aggregated_model)
        # non bias parameters have to be dequantized.
        self._dequantize(aggregated_model)

        FLModelParamUtils.set_gradient(
            model=self._global_model.fl_get_module(),
            reference_gradient=aggregated_model,
        )
        self._optimizer.step()

    def _dequantize(self, aggregated_model: torch.nn.Module):
        model_state_dict = aggregated_model.state_dict()
        new_state_dict = OrderedDict()
        for name, param in model_state_dict.items():
            if param.ndim > 1:
                scale, zero_point = self._global_qparams[name]
                int_param = param.data.to(dtype=torch.int8)
                q_param = torch._make_per_tensor_quantized_tensor(
                    int_param, scale.item(), int(zero_point.item())
                )
                deq_param = q_param.dequantize()
                new_state_dict[name] = deq_param
            else:
                new_state_dict[name] = param.data
        aggregated_model.load_state_dict(new_state_dict)

    def update_qparams(self, aggregated_model: torch.nn.Module):
        observer, _ = self._channel.get_observers_and_quantizers()  # pyre-ignore [16]
        for name, param in aggregated_model.state_dict().items():
            observer.reset_min_max_vals()
            _ = observer(param.data)
            self._global_qparams[name] = observer.calculate_qparams()


@dataclass
class SyncSecAggServerConfig(SyncServerConfig):
    """
    Contains configurations for a server with Secure Aggregation
    """

    _target_: str = fullclassname(SyncSecAggServer)
    aggregation_type: AggregationType = AggregationType.WEIGHTED_AVERAGE
    fixedpoint: Optional[FixedPointConfig] = None
    active_user_selector: ActiveUserSelectorConfig = ActiveUserSelectorConfig()


@dataclass
class SyncSecAggSQServerConfig(SyncSecAggServerConfig):
    _target_: str = fullclassname(SyncSecAggSQServer)
    secagg_scaling_factor_for_quantized: float = 1.0
