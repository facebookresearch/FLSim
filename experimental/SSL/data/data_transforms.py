#!/usr/bin/env python3
# Portions (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

# ------------------------------------------------------------------------------------

# FlipAndShift, Weak, Strong, and FixMatch transforms are adapted from kekmodel's
# FixMatch implementation, see license below

# Source files:
#   - https://github.com/kekmodel/FixMatch-pytorch/blob/master/dataset/randaugment.py
#   - https://github.com/kekmodel/FixMatch-pytorch/blob/master/dataset/cifar.py

# Date & time: October 20, 2021, 10:54 PM UTC


# MIT License

# Copyright (c) 2019 Jungdae Kim, Qing Yu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ------------------------------------------------------------------------------------


from abc import abstractmethod
from typing import Tuple, Union

import torch
from flsim.experimental.ssl.data.randaugment import RandAugment
from PIL.Image import Image
from torchvision import transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


class TransformType:
    NORMALIZE: str = "normalize"
    WEAK: str = "weak"
    STRONG: str = "strong"
    FIXMATCH: str = "fixmatch"
    FEDMATCH: str = "fedmatch"


class CustomSSLTransforms:
    """This class implements transforms relevant to SSL and FixMatch in
    particular. Note that RandAugment is used for strong augmentation,
    CTAugment is NOT supported.
    """

    class TransformBase:
        @abstractmethod
        def __call__(
            self, x
        ) -> Union[Image, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            """Override this function to apply a transform to x. Return type is
            Image for partial transforms, tensor for simple transforms, and
            Tuple[tensor, tensor] for unlabeled data augmentation (consistency
            regularization).
            """
            pass

    class Normalize(TransformBase):
        def __call__(self, x) -> torch.Tensor:
            # pyre-ignore[7]
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
                ]
            )(x)

    class FlipAndShift(TransformBase):
        def __call__(self, x) -> Union[torch.Tensor, Image]:
            return transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(
                        size=32, padding=int(32 * 0.125), padding_mode="reflect"
                    ),
                ]
            )(x)

    class Weak(TransformBase):
        """Flip-and-shift + normalization. Used for labeled data and to
        generate pseudolabels.
        """

        def __call__(self, x) -> torch.Tensor:
            # pyre-ignore[7]
            return transforms.Compose(
                [CustomSSLTransforms.FlipAndShift(), CustomSSLTransforms.Normalize()]
            )(x)

    class Strong(TransformBase):
        """Flip-and-shift + RandAugment + normalization. Used for unlabeled
        data. Does not support CTAugment.
        """

        def __call__(self, x) -> torch.Tensor:
            # pyre-ignore[7]
            return transforms.Compose(
                [
                    CustomSSLTransforms.FlipAndShift(),
                    RandAugment(n=2, m=10),
                    CustomSSLTransforms.Normalize(),
                ]
            )(x)

    class FixMatch(TransformBase):
        """FixMatch uses weakly augmented images to generate pseudolabels
        and trains on strongly augmented images (consistency regularization).
        """

        def __call__(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
            return CustomSSLTransforms.Weak()(x), CustomSSLTransforms.Strong()(x)

    class FedMatch(TransformBase):
        """FedMatch uses original images instead of weakly augmented images."""

        def __call__(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
            return transforms.ToTensor()(x), CustomSSLTransforms.Strong()(x)

    @staticmethod
    def create(transform_type: str, **kwargs):
        if transform_type == TransformType.NORMALIZE:
            return CustomSSLTransforms.Normalize()
        if transform_type == TransformType.WEAK:
            return CustomSSLTransforms.Weak()
        if transform_type == TransformType.STRONG:
            return CustomSSLTransforms.Strong()
        elif transform_type == TransformType.FIXMATCH:
            return CustomSSLTransforms.FixMatch()
        elif transform_type == TransformType.FEDMATCH:
            return CustomSSLTransforms.FedMatch()
        else:
            assert f"Invalid transform type: {transform_type}."
