#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import pkg_resources
import pytest
import torch
from flsim.common.pytest_helper import assertEqual
from flsim.data.csv_dataset import FLCSVDataset
from flsim.data.data_sharder import ColumnSharderConfig
from flsim.data.dataset_data_loader import FLDatasetDataLoaderWithBatch
from hydra.utils import instantiate


@pytest.fixture(scope="class")
def prepare_dataset_data_loader_with_batch(request):
    request.cls.test_csv_path = "test_resources/data.csv"
    request.cls.total_data_count = 15
    request.cls.train_batch_size = 1
    request.cls.eval_batch_size = 3
    request.cls.test_batch_size = 5


class TestDataset(FLCSVDataset):
    def _get_processed_row_from_single_raw_row(self, raw_row: Any) -> Dict[str, Any]:
        return {
            "userid": torch.Tensor([raw_row["userid"]]),
            "label": torch.Tensor([raw_row["label"]]),
        }


@pytest.mark.usefixtures("prepare_dataset_data_loader_with_batch")
class TestDatasetDataLoaderWithBatch:
    def test_batch_size(self) -> None:
        # pyre-ignore[16]: for pytest fixture
        file_path = pkg_resources.resource_filename(__name__, self.test_csv_path)
        dataset = TestDataset(file_path)

        fl_data_sharder = instantiate(ColumnSharderConfig(sharding_col="userid"))
        data_loader = FLDatasetDataLoaderWithBatch(
            dataset,
            dataset,
            dataset,
            fl_data_sharder,
            # pyre-ignore[16]: for pytest fixture
            self.train_batch_size,
            # pyre-ignore[16]: for pytest fixture
            self.eval_batch_size,
            # pyre-ignore[6]
            self.test_batch_size,
        )
        assertEqual(
            len(list(data_loader.fl_train_set())),
            # pyre-ignore[16]: for pytest fixture
            self.total_data_count / self.train_batch_size,
        )
        assertEqual(
            len(list(data_loader.fl_eval_set())),
            self.total_data_count / self.eval_batch_size,
        )
        assertEqual(
            len(list(data_loader.fl_test_set())),
            self.total_data_count / self.test_batch_size,
        )
