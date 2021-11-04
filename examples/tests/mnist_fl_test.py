#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from unittest import mock

from flsim.data.data_provider import FLDataProviderFromList
from flsim.data.data_sharder import FLDataSharder
from flsim.data.dataset_data_loader import FLDatasetDataLoaderWithBatch
from flsim.examples.mnist_fl_dataset import MNISTDataset
from flsim.examples.mnist_fl_metrics_reporter import MNISTMetricsReporter
from flsim.examples.mnist_fl_model import create_lighter_fl_model_for_mnist
from flsim.interfaces.metrics_reporter import Channel
from flsim.trainers.sync_trainer import SyncTrainerConfig  # noqa
from flsim.utils.config_utils import fullclassname
from flsim.utils.data.dummy_image_dataset import DummyImageDataset
from hydra.experimental import compose, initialize
from hydra.utils import instantiate
from libfb.py import testutil
from torch import cuda
from torchvision import transforms


def get_dataset():
    mnist_stream = DummyImageDataset(
        num_images_per_class=10,
    ).get_dataset()
    mnist_stream.seek(0)
    mnist_dataset = MNISTDataset(
        path=mnist_stream,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    return mnist_dataset


class TestMNISTFL(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_mnist_training(self) -> None:
        initialize(config_path="confs")
        hydra_config = compose(config_name="config")
        dataset = get_dataset()
        metrics_reporter = MNISTMetricsReporter([Channel.STDOUT])
        global_model = create_lighter_fl_model_for_mnist()
        shard_size = 10
        num_total_users = 10
        fl_data_sharder = FLDataSharder("sequential", None, None, None, shard_size)
        data_loader = FLDatasetDataLoaderWithBatch(
            dataset,
            dataset,
            dataset,
            fl_data_sharder,
            hydra_config.local_batch_size,
            hydra_config.local_batch_size,
            hydra_config.local_batch_size,
        )
        sync_trainer = instantiate(
            hydra_config.trainer,
            model=global_model,
            cuda_enabled=hydra_config.use_cuda_if_available and cuda.is_available(),
        )
        train_set = data_loader.fl_train_set()
        with mock.patch(f"{fullclassname(MNISTDataset)}.__len__") as mock_mnist_len:
            mock_mnist_len.return_value = 100
            sync_trainer.train(
                data_provider=FLDataProviderFromList(
                    train_set,
                    data_loader.fl_eval_set(),
                    data_loader.fl_test_set(),
                    global_model,
                ),
                metric_reporter=metrics_reporter,
                num_total_users=num_total_users,
                distributed_world_size=1,
            )
