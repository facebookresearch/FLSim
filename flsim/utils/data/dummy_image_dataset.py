#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from io import BytesIO

import numpy as np
import torch
import torch.utils.data as data


class DummyImageDataset(data.Dataset):
    def __init__(
        self,
        num_classes=10,
        num_images_per_class=10,
        num_channels=1,
        image_dim=(28, 28),
    ):
        self.num_classes = num_classes
        self.num_images_per_class = num_images_per_class
        if num_channels == 1:
            data_dim = (
                self.num_classes * self.num_images_per_class,
                image_dim[0],
                image_dim[1],
            )
        else:
            data_dim = (
                self.num_classes * self.num_images_per_class,
                num_channels,
                image_dim[0],
                image_dim[1],
            )

        self.data = torch.from_numpy(
            np.random.uniform(low=0.0, high=1.0, size=data_dim)
        )
        self.targets = list(range(self.num_classes))
        self.labels = torch.LongTensor(self.targets * self.num_images_per_class)

    def get_dataset(self):
        stream = BytesIO()
        torch.save((self.data, self.labels), stream)
        return stream

    def __len__(self):
        return self.num_classes * self.num_images_per_class

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx].item()
