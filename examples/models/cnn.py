#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import torch.nn as nn


class SimpleConvNet(nn.Module):
    r"""
    Simple CNN model following architecture from
    https://github.com/TalwalkarLab/leaf/blob/master/models/celeba/cnn.py#L19
    and https://arxiv.org/pdf/1903.03934.pdf
    """

    def __init__(self, in_channels, num_classes, dropout_rate=0):
        super(SimpleConvNet, self).__init__()
        self.out_channels = 32
        self.stride = 1
        self.padding = 2

        self.conv1 = nn.Conv2d(
            in_channels, self.out_channels, 3, self.stride, self.padding
        )
        self.conv2 = nn.Conv2d(
            self.out_channels, self.out_channels, 3, self.stride, self.padding
        )
        self.conv3 = nn.Conv2d(
            self.out_channels, self.out_channels, 3, self.stride, self.padding
        )
        self.conv4 = nn.Conv2d(
            self.out_channels, self.out_channels, 3, self.stride, self.padding
        )
        self.bn_relu = nn.Sequential(
            nn.GroupNorm(self.out_channels, self.out_channels, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        num_features = (
            self.out_channels
            * (self.stride + self.padding)
            * (self.stride + self.padding)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.bn_relu(self.conv1(x))
        x = self.bn_relu(self.conv2(x))
        x = self.bn_relu(self.conv3(x))
        x = self.bn_relu(self.conv4(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(self.dropout(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
