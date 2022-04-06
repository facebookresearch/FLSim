from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
from flsim.channels.base_channel import IdentityChannel
from flsim.channels.message import Message
from flsim.common.logger import Logger
from flsim.common.timeout_simulator import (
    NeverTimeOutSimulator,
    NeverTimeOutSimulatorConfig,
    TimeOutSimulator,
)
from flsim.data.data_provider import IFLUserData
from flsim.interfaces.metrics_reporter import IFLMetricsReporter
from flsim.interfaces.model import IFLModel
from flsim.optimizers.local_optimizers import (
    LocalOptimizerConfig,
    LocalOptimizerSGDConfig,
)
from flsim.optimizers.optimizer_scheduler import (
    OptimizerScheduler,
    OptimizerSchedulerConfig,
    ConstantLRSchedulerConfig,
)
from flsim.utils.config_utils import fullclassname
from flsim.utils.config_utils import init_self_cfg
from flsim.utils.cuda import ICudaStateManager, DEFAULT_CUDA_MANAGER
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate
from omegaconf import OmegaConf
from flsim.clients.base_client import Client, ClientConfig


class FedSGDClient(Client):
    """
    FEDSGD client runs full SGD on the client local dataset and returns the gradient
    """

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
            component_class=__class__,  # pyre-ignore[16]
            config_class=FedSGDClientConfig,
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

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        if OmegaConf.is_missing(cfg.optimizer, "_target_"):
            cfg.optimizer = LocalOptimizerSGDConfig(lr=1.0)
        if OmegaConf.is_missing(cfg.lr_scheduler, "_target_"):
            cfg.lr_scheduler = ConstantLRSchedulerConfig()

    def post_train(self, model: IFLModel, total_samples: int, optimizer: Any):
        optimizer.step()

    def _batch_train(
        self,
        model,
        optimizer,
        training_batch,
        epoch,
        metric_reporter,
        optimizer_scheduler,
    ) -> int:
        batch_metrics = model.fl_forward(training_batch)
        loss = batch_metrics.loss

        loss.backward()
        # pyre-ignore[16]: `Client` has no attribute `cfg`.
        if self.cfg.max_clip_norm_normalized is not None:
            max_norm = self.cfg.max_clip_norm_normalized
            FLModelParamUtils.clip_gradients(
                max_normalized_l2_norm=max_norm, model=model.fl_get_module()
            )

        num_examples = batch_metrics.num_examples
        if metric_reporter is not None:
            metric_reporter.add_batch_metrics(batch_metrics)

        return num_examples


@dataclass
class FedSGDClientConfig(ClientConfig):
    _target_: str = fullclassname(FedSGDClient)
    optimizer: LocalOptimizerConfig = LocalOptimizerSGDConfig(lr=1.0)
