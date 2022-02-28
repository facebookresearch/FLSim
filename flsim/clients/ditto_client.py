#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Optional, Tuple

import torch
from flsim.channels.base_channel import IdentityChannel
from flsim.clients.base_client import Client, ClientConfig
from flsim.common.timeout_simulator import (
    TimeOutSimulator,
)
from flsim.data.data_provider import IFLUserData
from flsim.interfaces.metrics_reporter import IFLMetricsReporter
from flsim.interfaces.model import IFLModel
from flsim.optimizers.local_optimizers import (
    LocalOptimizerProximalConfig,
    LocalOptimizerConfig,
    LocalOptimizerSGDConfig,
)
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.cuda import ICudaStateManager, DEFAULT_CUDA_MANAGER
from hydra.utils import instantiate


class DittoClient(Client):
    def __init__(
        self,
        *,
        dataset: IFLUserData,
        channel: Optional[IdentityChannel] = None,
        timeout_simulator: Optional[TimeOutSimulator] = None,
        store_last_updated_model: Optional[bool] = False,
        cuda_manager: ICudaStateManager = DEFAULT_CUDA_MANAGER,
        name: Optional[str] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=DittoClientConfig,
            **kwargs,
        )
        super().__init__(
            dataset=dataset,
            channel=channel,
            timeout_simulator=timeout_simulator,
            store_last_updated_model=store_last_updated_model,
            name=name,
            cuda_manager=cuda_manager,
            **kwargs,
        )

    def eval(
        self,
        model: IFLModel,
        dataset: Optional[IFLUserData] = None,
        metric_reporter: Optional[IFLMetricsReporter] = None,
        fine_tune: bool = False,
        personalized_epoch: int = 1,
    ):
        if fine_tune:
            model = self.receive_through_channel(model)
            # inform cuda_state_manager that we're about to train a model
            # it may move model to GPU
            self.cuda_state_manager.before_train_or_eval(model)
            # put model in train mode
            model.fl_get_module().train()
            # create optimizer
            optimizer = instantiate(
                # pyre-ignore[16]
                self.cfg.prox_optimizer,
                model=model.fl_get_module(),
            )
            optimizer_scheduler = instantiate(
                self.cfg.lr_scheduler, optimizer=optimizer
            )
            model, _ = self.train(
                model,
                optimizer,
                optimizer_scheduler,
                metric_reporter,
                personalized_epoch,
            )
        else:
            model = model or self.ref_model

        data = self.dataset
        self.cuda_state_manager.before_train_or_eval(model)
        with torch.no_grad():
            if self.seed is not None:
                torch.manual_seed(self.seed)

            model.fl_get_module().eval()
            for batch in data.eval_data():
                batch_metrics = model.get_eval_metrics(batch)
                if metric_reporter is not None:
                    metric_reporter.add_batch_metrics(batch_metrics)
        model.fl_get_module().train()
        self.cuda_state_manager.after_train_or_eval(model)


@dataclass
class DittoClientConfig(ClientConfig):
    _target_: str = fullclassname(DittoClient)
    epochs: int = 1  # No. of epochs for local training
    optimizer: LocalOptimizerConfig = LocalOptimizerSGDConfig()
    prox_optimizer: LocalOptimizerConfig = LocalOptimizerProximalConfig()
