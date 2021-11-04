#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import flsim.experimental.ssl.data.data_transforms as tfs
import numpy as np
import torch
from libfb.py import testutil
from torchvision import transforms


class TestDataTransforms(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

        # create PIL Image with same dimensions as CIFAR-10
        self.x = transforms.ToPILImage()(
            torch.from_numpy(np.random.uniform(low=0.0, high=1.0, size=(3, 32, 32)))
        )
        self.x_tensor = transforms.ToTensor()(self.x)
        self.random_seed = np.random.randint(100)

    def test_normalize_transform(self):
        normalized = tfs.CustomSSLTransforms.create("normalize")(self.x)
        self.assertFalse(torch.equal(self.x_tensor, normalized))

    def test_weak_strong_transforms(self):
        torch.manual_seed(self.random_seed)
        weak = tfs.CustomSSLTransforms.create("weak")(self.x)
        torch.manual_seed(self.random_seed)
        strong = tfs.CustomSSLTransforms.create("strong")(self.x)
        self.assertFalse(torch.equal(self.x_tensor, weak))
        self.assertFalse(torch.equal(self.x_tensor, strong))
        self.assertFalse(torch.equal(weak, strong))

    def test_fixmatch_transform(self):
        """FixMatch(x) --> weak(x), strong(x)"""
        torch.manual_seed(self.random_seed)
        weak = tfs.CustomSSLTransforms.create("weak")(self.x)
        torch.manual_seed(self.random_seed)
        fixmatch = tfs.CustomSSLTransforms.create("fixmatch")(self.x)
        self.assertTrue(torch.equal(weak, fixmatch[0]))
        self.assertFalse(torch.equal(fixmatch[0], fixmatch[1]))

    def test_fedmatch_transform(self):
        """FedMatch(x) --> x, strong(x)"""
        fedmatch = tfs.CustomSSLTransforms.create("fedmatch")(self.x)
        self.assertTrue(torch.equal(self.x_tensor, fedmatch[0]))
        self.assertFalse(torch.equal(fedmatch[0], fedmatch[1]))
