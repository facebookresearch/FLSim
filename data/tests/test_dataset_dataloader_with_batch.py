#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from typing import Any, Dict

import pkg_resources
import torch
from flsim.data.csv_dataset import FLCSVDataset
from flsim.data.data_sharder import FLDataSharder
from flsim.data.dataset_data_loader import FLDatasetDataLoaderWithBatch
from libfb.py import testutil


class TestDataset(FLCSVDataset):
    def _get_processed_row_from_single_raw_row(self, raw_row: Any) -> Dict[str, Any]:
        return {
            "userid": torch.Tensor([raw_row["userid"]]),
            "label": torch.Tensor([raw_row["label"]]),
        }


class DatasetDataLoaderWithBatchTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.test_csv_path = "test_resources/data.csv"
        self.total_data_count = 15
        self.train_batch_size = 1
        self.eval_batch_size = 3
        # pyre-fixme[8]: Attribute has type
        #  `BoundMethod[typing.Callable(DatasetDataLoaderWithBatchTest.test_batch_size)[[Named(self,
        #  DatasetDataLoaderWithBatchTest)], None], DatasetDataLoaderWithBatchTest]`;
        #  used as `int`.
        self.test_batch_size = 5

    def test_batch_size(self) -> None:
        file_path = pkg_resources.resource_filename(__name__, self.test_csv_path)
        dataset = TestDataset(file_path)

        fl_data_sharder = FLDataSharder(
            "column", None, None, "userid", None  # sharding_strategy  # userid column
        )
        data_loader = FLDatasetDataLoaderWithBatch(
            dataset,
            dataset,
            dataset,
            fl_data_sharder,
            self.train_batch_size,
            self.eval_batch_size,
            # pyre-fixme[6]: Expected `Optional[int]` for 7th param but got
            #  `BoundMethod[typing.Callable(DatasetDataLoaderWithBatchTest.test_batch_size)[[Named(self,
            #  DatasetDataLoaderWithBatchTest)], None],
            #  DatasetDataLoaderWithBatchTest]`.
            self.test_batch_size,
        )
        self.assertEqual(
            len(list(data_loader.fl_train_set())),
            self.total_data_count / self.train_batch_size,
        )
        self.assertEqual(
            len(list(data_loader.fl_eval_set())),
            self.total_data_count / self.eval_batch_size,
        )
        self.assertEqual(
            len(list(data_loader.fl_test_set())),
            # pyre-fixme[58]: `/` is not supported for operand types `int` and `Bound...
            self.total_data_count / self.test_batch_size,
        )
