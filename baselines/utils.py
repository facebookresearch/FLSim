#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import json
from enum import Enum

import torch
import torch.optim as optim
from flsim.baselines.models.cv_model import FLModel
from flsim.baselines.models.seq2seq_model import FLModel as Seq2SeqFLModel
from flsim.fb.metrics_reporters.cv_reporter import (
    CVMetricsReporter,
    Seq2SeqMetricsReporter,
)
from flsim.fb.process_state import FBProcessState
from flsim.interfaces.metrics_reporter import Channel, TrainingStage
from flsim.optimizers.sync_aggregators import (
    FedAdamSyncAggregatorConfig,
    FedAvgSyncAggregatorConfig,
    FedLAMBSyncAggregatorConfig,
    FedLARSSyncAggregatorConfig,
    FedAvgWithLRSyncAggregatorConfig,
)
from flsim.tests import utils
from tqdm import tqdm


class GlobalOptimizerType(Enum):
    FEDAVG = "FedAVG"
    SGD = "SGD"
    ADAM = "Adam"
    LARS = "LARS"
    LAMB = "LAMB"

    def __str__(self):
        return self.value


def create_global_aggregator(
    global_optimizer_type: GlobalOptimizerType,
    global_optimizer_lr: float,
    global_optimizer_momentum: float,
    global_optimizer_beta1: float,
    global_optimizer_beta2: float,
):
    if global_optimizer_type == GlobalOptimizerType.SGD:
        return FedAvgWithLRSyncAggregatorConfig(
            lr=global_optimizer_lr, momentum=global_optimizer_momentum
        )
    elif global_optimizer_type == GlobalOptimizerType.ADAM:
        return FedAdamSyncAggregatorConfig(lr=global_optimizer_lr)
    elif global_optimizer_type == GlobalOptimizerType.LARS:
        return FedLARSSyncAggregatorConfig(
            lr=global_optimizer_lr, beta=global_optimizer_beta1
        )
    elif global_optimizer_type == GlobalOptimizerType.LAMB:
        return FedLAMBSyncAggregatorConfig(
            lr=global_optimizer_lr,
            beta1=global_optimizer_beta1,
            beta2=global_optimizer_beta2,
        )
    else:
        return FedAvgSyncAggregatorConfig()


def fake_data(num_batches, batch_size, device):
    torch.manual_seed(0)
    dataset = [torch.rand(batch_size, 2, device=device) for _ in range(num_batches)]
    return utils.DatasetFromList(dataset)


def pretty_print(config):
    print(json.dumps(config, indent=4, sort_keys=True))


def train_non_fl(data_provider, model, device, cuda_enabled, lr, momentum, epochs=1):
    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()
    FBProcessState.getInstance(rank=0, fb_info=None)
    metrics_reporter = CVMetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])
    optimizer = optim.SGD(
        global_model.fl_get_module().parameters(), lr=lr, momentum=momentum
    )
    for epoch in range(epochs):
        for one_user_data in tqdm(data_provider.train_data()):
            for batch in one_user_data:
                optimizer.zero_grad()
                batch_metrics = global_model.fl_forward(batch)
                metrics_reporter.add_batch_metrics(batch_metrics)

                batch_metrics.loss.backward()
                optimizer.step()
        metrics_reporter.report_metrics(
            reset=False, stage=TrainingStage.TRAINING, epoch=epoch
        )


def train_non_fl_seq2seq(
    data_provider,
    model,
    device,
    cuda_enabled,
    lr,
    momentum,
    vocab_info,
    epochs=1,
):
    global_model = Seq2SeqFLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()

    FBProcessState.getInstance(rank=0, fb_info=None)
    metrics_reporter = Seq2SeqMetricsReporter(
        vocab_info, [Channel.TENSORBOARD, Channel.STDOUT]
    )

    optimizer = optim.SGD(
        global_model.fl_get_module().parameters(), lr=lr, momentum=momentum
    )
    for epoch in range(epochs):
        for one_user_data in tqdm(data_provider.train_data()):
            for batch in one_user_data:
                optimizer.zero_grad()
                batch_metrics = global_model.fl_forward(batch)
                metrics_reporter.add_batch_metrics(batch_metrics)

                batch_metrics.loss.backward()
                optimizer.step()
        metrics_reporter.report_metrics(
            reset=False, stage=TrainingStage.TRAINING, epoch=epoch
        )
