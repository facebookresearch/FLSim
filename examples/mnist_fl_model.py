#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from flsim.interfaces.model import IFLModel
from flsim.utils.simple_batch_metrics import FLBatchMetrics


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class DemoNet(nn.Module):
    def __init__(self):
        super(DemoNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, 1)


class FLModel(IFLModel):
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device

    def fl_forward(self, batch) -> FLBatchMetrics:
        float_features = batch["float_features"]
        batch_label = batch["label"]
        stacked_label = batch_label.view(-1).long().clone().detach()

        if self.device is not None:
            float_features = float_features.to(self.device)

        out = self.model(float_features)
        if self.device is not None:
            out, batch_label, stacked_label = (
                out.to(self.device),
                batch["label"].to(self.device),
                stacked_label.to(self.device),
            )

        loss = F.nll_loss(out, stacked_label)
        num_examples = self.get_num_examples(batch)
        return FLBatchMetrics(
            loss, num_examples, out, batch_label, [batch["float_features"]]
        )

    def fl_create_training_batch(self, **kwargs):
        return kwargs.get("batch", None)

    def fl_get_module(self) -> nn.Module:
        return self.model

    def fl_cuda(self) -> None:
        self.model = self.model.to(self.device)

    def get_eval_metrics(self, batch) -> FLBatchMetrics:
        with torch.no_grad():
            return self.fl_forward(batch)

    def get_num_examples(self, batch) -> int:
        return len(batch["label"])


def create_fl_model_for_mnist() -> IFLModel:
    return FLModel(Net())


def create_lighter_fl_model_for_mnist(device: Optional[str] = None) -> IFLModel:
    return FLModel(DemoNet(), device)
