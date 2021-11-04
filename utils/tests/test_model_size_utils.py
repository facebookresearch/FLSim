#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import unittest

from flsim.utils.model_size_utils import calc_model_size, calc_model_sparsity
from flsim.utils.tests.helpers.test_models import FCModel
from torch.nn.utils import prune


class FLModelSizeUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_fp16_model_size(self) -> None:
        model = FCModel()
        original_model_size = calc_model_size(model.state_dict())
        model.half()
        fp16_model_size = calc_model_size(model.state_dict())
        self.assertEqual(
            original_model_size / 2.0,
            fp16_model_size,
        )

    def test_sparse_model_size(self) -> None:
        model = FCModel()
        # Prune model to a quarter of its size
        params_to_prune = [
            (model.fc1, "weight"),
            (model.fc1, "bias"),
            (model.fc2, "weight"),
            (model.fc2, "bias"),
            (model.fc3, "weight"),
            (model.fc3, "bias"),
        ]
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.75,
        )
        for module, name in params_to_prune:
            prune.remove(module, name)
        sparsity = calc_model_sparsity(model.state_dict())
        self.assertAlmostEqual(
            0.75,
            sparsity,
            delta=0.02,  # Accounts for 2 percentage points difference
        )
