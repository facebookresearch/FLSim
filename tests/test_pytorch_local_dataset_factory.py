#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from typing import List

import torch
from caffe2_fb.io_metadata.types import DenseFeatureMetadata, FeatureDescription
from flsim.fb.data.hive_dataset import PyTorchLocalDatasetFactory
from libfb.py import testutil

# @manual=//caffe2/torch/fb/model_metadata:module_metadata
from torch.fb.model_metadata.module_metadata import Metadata


# TODO: (jesikmin) T61017955 Revisit test for PyTorchLocalDataset to make it more
# stable
class PyTorchLocalDatasetFactoryTest(testutil.BaseFacebookTestCase):
    NUM_FEAT: int = 1199
    NAMESPACE: str = "ad_delivery"
    TABLE: str = "daiquery_319222928981453"
    FEATURES: List[str] = ["label", "float_features", "prod_prediction"]
    DATA_SPLIT_COL = "data_split"
    TRAIN_CLASS = "train"
    TEST_CLASS = "test"

    NUM_TRAIN: int = 3
    NUM_TEST: int = 97

    def setUp(self) -> None:
        super().setUp()
        feat_ids = list(range(self.NUM_FEAT))
        self.metadata_map = {
            # pyre-ignore[16]: Pyre complains torch.fb.model_metadata.module_metadata
            # has no attribute Metadata
            "float_features": Metadata(
                dense_features=DenseFeatureMetadata(
                    feature_desc=[FeatureDescription(feature_id=i) for i in feat_ids]
                )
            )
        }

    def DISABLED_test_create_dataset(self) -> None:
        train_dataset = PyTorchLocalDatasetFactory.get_dataset(
            namespace=self.NAMESPACE,
            data_table=self.TABLE,
            features=self.FEATURES,
            batch_size=1,
            metadata_map=self.metadata_map,
            data_split_colname=self.DATA_SPLIT_COL,
            data_split=self.TRAIN_CLASS,
        )
        test_dataset = PyTorchLocalDatasetFactory.get_dataset(
            namespace=self.NAMESPACE,
            data_table=self.TABLE,
            features=self.FEATURES,
            batch_size=1,
            metadata_map=self.metadata_map,
            data_split_colname=self.DATA_SPLIT_COL,
            data_split=self.TEST_CLASS,
        )

        train_data = list(torch.utils.data.DataLoader(train_dataset, num_workers=0))
        test_data = list(torch.utils.data.DataLoader(test_dataset, num_workers=0))

        self.assertEqual(len(train_data), self.NUM_TRAIN)
        self.assertEqual(len(test_data), self.NUM_TEST)

    def DISABLED_test_create_dataset_with_limit(self) -> None:
        max_rows = 1
        train_dataset = PyTorchLocalDatasetFactory.get_dataset(
            namespace=self.NAMESPACE,
            data_table=self.TABLE,
            features=self.FEATURES,
            batch_size=1,
            metadata_map=self.metadata_map,
            data_split_colname=self.DATA_SPLIT_COL,
            data_split=self.TRAIN_CLASS,
            max_rows=max_rows,
        )

        train_data = list(torch.utils.data.DataLoader(train_dataset, num_workers=0))

        self.assertEqual(len(train_data), max_rows)

    def DISABLED_test_dataset_transform(self) -> None:
        def transform(row):
            return (
                {"float_features": row["float_features"][0]},
                {"label": row["label"][0]},
            )

        train_dataset = PyTorchLocalDatasetFactory.get_dataset(
            namespace=self.NAMESPACE,
            data_table=self.TABLE,
            features=self.FEATURES,
            batch_size=1,
            metadata_map=self.metadata_map,
            data_split_colname=self.DATA_SPLIT_COL,
            data_split=self.TRAIN_CLASS,
        )

        train_dataset_with_transform = PyTorchLocalDatasetFactory.get_dataset(
            namespace=self.NAMESPACE,
            data_table=self.TABLE,
            features=self.FEATURES,
            transforms=transform,
            batch_size=1,
            metadata_map=self.metadata_map,
            data_split_colname=self.DATA_SPLIT_COL,
            data_split=self.TRAIN_CLASS,
        )

        # Koski API to read data from Hive table does not guarantee any ordering
        # while reading data. So, in this tests, we do the following:
        #   - read data with batch size of 1
        #   - store each data row to the sets
        #   - assert (i.e. compare on the sets with some proper flattening)
        #     that labels and features are identical between original dataset
        #     and the dataset with transformation
        float_features = set()
        float_features_from_transformed = set()
        labels = set()
        labels_from_transformed = set()

        for data_row, data_row_with_transform in zip(
            train_dataset, train_dataset_with_transform
        ):
            float_features.add(data_row["float_features"][0])
            labels.add(data_row["label"][0])

            float_features_from_transformed.add(
                data_row_with_transform[0]["float_features"]
            )
            labels_from_transformed.add(data_row_with_transform[1]["label"])

        # for the quirky in comparing set of tensors to each other, compute
        # L2 norm, flatten them, and compare them
        self.assertSetEqual(
            {
                norm[0]
                for norm in [
                    float_feature.norm(p=2, dim=1).tolist()
                    for float_feature in float_features
                ]
            },
            {
                norm[0]
                for norm in [
                    float_feature.norm(p=2, dim=1).tolist()
                    for float_feature in float_features_from_transformed
                ]
            },
        )
        self.assertSetEqual(
            {_label[0] for _label in [label.tolist() for label in labels]},
            {
                _label[0]
                for _label in [label.tolist() for label in labels_from_transformed]
            },
        )

    def DISABLED_test_data_split_colname_not_provided(self) -> None:
        # if column name is not provided, entire dataset should be fetched
        entire_dataset = PyTorchLocalDatasetFactory.get_dataset(
            namespace=self.NAMESPACE,
            data_table=self.TABLE,
            features=self.FEATURES,
            transforms=None,
            batch_size=1,
            metadata_map=self.metadata_map,
            data_split_colname=None,
            data_split=None,
        )

        entire_data = list(torch.utils.data.DataLoader(entire_dataset, num_workers=0))
        self.assertEqual(len(entire_data), self.NUM_TRAIN + self.NUM_TEST)
