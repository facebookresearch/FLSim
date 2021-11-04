#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import torch
import torch.utils.data as data
from PIL import Image


class MNISTDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None, target_transform=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if self.train:
            self.train_data, self.train_labels = torch.load(self.path)
        else:
            self.test_data, self.test_labels = torch.load(self.path)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return {"float_features": img, "label": torch.Tensor([int(target) * 1.0])}

    def __len__(self):
        if self.train:
            return 60000
        else:
            return 10000
