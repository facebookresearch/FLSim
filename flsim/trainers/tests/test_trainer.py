#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import math
from typing import List

import flsim.configs  # noqa
import pkg_resources
import pytest
import torch
from flsim.active_user_selectors.simple_user_selector import (
    SequentialActiveUserSelectorConfig,
)
from flsim.clients.base_client import ClientConfig
from flsim.common.pytest_helper import (
    assertEqual,
    assertTrue,
    assertIsInstance,
    assertEmpty,
)
from flsim.common.timeout_simulator import GaussianTimeOutSimulatorConfig
from flsim.data.data_provider import FLDataProviderFromList
from flsim.data.data_sharder import SequentialSharder
from flsim.data.dataset_data_loader import FLDatasetDataLoaderWithBatch
from flsim.fb.data.hive_data_utils import create_dataloader
from flsim.fb.data.hive_dataset import InlineDatasetConfig
from flsim.fb.data.paged_dataloader import PagedDataProvider
from flsim.interfaces.metrics_reporter import TrainingStage
from flsim.optimizers.async_aggregators import FedAdamAsyncAggregatorConfig
from flsim.optimizers.local_optimizers import (
    LocalOptimizerSGDConfig,
    LocalOptimizerFedProxConfig,
)
from flsim.optimizers.optimizer_scheduler import ArmijoLineSearchSchedulerConfig
from flsim.optimizers.server_optimizers import (
    FedAdamOptimizerConfig,
    FedAvgWithLROptimizerConfig,
)
from flsim.servers.sync_servers import (
    SyncServerConfig,
)
from flsim.tests.utils import (
    FakeMetricReporter,
    MockRecord,
    MetricsReporterWithMockedChannels,
    SimpleMetricReporter,
    verify_models_equivalent_after_training,
)
from flsim.tests.utils import SampleNetHive
from flsim.trainers.async_trainer import AsyncTrainer, AsyncTrainerConfig
from flsim.trainers.sync_trainer import SyncTrainer, SyncTrainerConfig
from flsim.utils.config_utils import fl_config_from_json
from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.sample_model import DummyAlphabetFLModel
from flsim.utils.tests.helpers.sync_trainer_test_utils import create_sync_trainer
from flsim.utils.tests.helpers.test_data_utils import DummyAlphabetDataset
from flsim.utils.tests.helpers.test_utils import FLTestUtils
from flsim.utils.timing.training_duration_distribution import (
    PerExampleGaussianDurationDistributionConfig,
)
from hydra.experimental import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

CONFIG_PATH = "test_resources"

SYNC_TRAINER_JSON = f"{CONFIG_PATH}/sync_trainer.json"
ASYNC_TRAINER_JSON = f"{CONFIG_PATH}/async_trainer.json"
SYNC_TRAINER_WITH_DP_JSON = f"{CONFIG_PATH}/sync_trainer_with_dp.json"
SYNC_TRAINER_WRONG_JSON = f"{CONFIG_PATH}/sync_trainer_wrong_dp_config.json"
SYNC_TRAINER_WITH_SECAGG_JSON = f"{CONFIG_PATH}/sync_trainer_with_secagg.json"

SYNC_TRAINER_YAML = "sync_trainer"
ASYNC_TRAINER_YAML = "async_trainer"
SYNC_TRAINER_WITH_DP_YAML = "sync_trainer_with_dp"
SYNC_TRAINER_WRONG_YAML = "sync_trainer_wrong_dp_config"
SYNC_TRAINER_WITH_SECAGG_YAML = "sync_trainer_with_secagg"


class TestTrainer:
    @pytest.mark.parametrize(
        "json_file_name,trainer_class ",
        [
            (SYNC_TRAINER_JSON, SyncTrainer),
            (ASYNC_TRAINER_JSON, AsyncTrainer),
            (SYNC_TRAINER_WITH_DP_JSON, SyncTrainer),
            (SYNC_TRAINER_WITH_SECAGG_JSON, SyncTrainer),
        ],
    )
    def test_trainer_creation_from_json_config(
        self, json_file_name: str, trainer_class: type
    ):
        trainer = None
        file_path = pkg_resources.resource_filename(__name__, json_file_name)
        with open(file_path, "r") as parameters_file:
            json_cfg = json.load(parameters_file)
        cfg = fl_config_from_json(json_cfg)
        trainer = instantiate(
            cfg.trainer,
            model=DummyAlphabetFLModel(),
            cuda_enabled=False,
        )
        assertIsInstance(trainer, trainer_class)

    def test_trainer_sync_server_creation_from_json_config(self):
        file_path = pkg_resources.resource_filename(__name__, SYNC_TRAINER_JSON)
        with open(file_path, "r") as parameters_file:
            json_cfg = json.load(parameters_file)
        cfg = fl_config_from_json(json_cfg)
        trainer = instantiate(
            cfg.trainer,
            model=DummyAlphabetFLModel(),
            cuda_enabled=False,
        )
        assertIsInstance(trainer.server._optimizer, torch.optim.Adam)

    @pytest.mark.parametrize(
        "yaml_file_name,trainer_class ",
        [
            (SYNC_TRAINER_YAML, SyncTrainer),
            (ASYNC_TRAINER_YAML, AsyncTrainer),
            (SYNC_TRAINER_WITH_DP_YAML, SyncTrainer),
            (SYNC_TRAINER_WITH_SECAGG_YAML, SyncTrainer),
        ],
    )
    def test_trainer_creation_from_yaml_config(
        self, yaml_file_name: str, trainer_class: type
    ):
        trainer = None
        with initialize(config_path=CONFIG_PATH):
            cfg = compose(config_name=yaml_file_name)
            trainer = instantiate(
                cfg.trainer,
                model=DummyAlphabetFLModel(),
                cuda_enabled=False,
            )
        assertIsInstance(trainer, trainer_class)

    def test_async_trainer_with_dp_creation_from_json_config(self):
        trainer = None
        file_path = pkg_resources.resource_filename(__name__, ASYNC_TRAINER_JSON)
        with open(file_path, "r") as parameters_file:
            json_cfg = json.load(parameters_file)
        cfg = fl_config_from_json(json_cfg)
        trainer = instantiate(
            cfg.trainer,
            model=DummyAlphabetFLModel(),
            cuda_enabled=False,
        )
        assertIsInstance(trainer, AsyncTrainer)
        assertTrue(trainer.aggregator.is_private)

    def test_async_trainer_with_dp_creation_from_yaml_config(self):
        trainer = None
        with initialize(config_path=CONFIG_PATH):
            cfg = compose(config_name=ASYNC_TRAINER_YAML)
            trainer = instantiate(
                cfg.trainer,
                model=DummyAlphabetFLModel(),
                cuda_enabled=False,
            )
        assertIsInstance(trainer, AsyncTrainer)
        assertTrue(trainer.aggregator.is_private)

    def test_global_model_unchanged_after_metrics_reporting(self):
        """
        reporting metrics after aggregation should NOT update the global model
        """
        # mock 26 rows
        shard_size = 4
        local_batch_size = 4
        world_size = 1
        dummy_dataset = DummyAlphabetDataset()
        fl_data_sharder = SequentialSharder(examples_per_shard=shard_size)
        data_loader = FLDatasetDataLoaderWithBatch(
            dummy_dataset,
            dummy_dataset,
            dummy_dataset,
            fl_data_sharder,
            local_batch_size,
            local_batch_size,
            local_batch_size,
        )

        global_model = DummyAlphabetFLModel()

        local_optimizer_lr = 0.1
        metrics_reporter = FakeMetricReporter()
        users_per_round = 2
        sync_trainer_no_report = create_sync_trainer(
            model=global_model,
            local_lr=local_optimizer_lr,
            users_per_round=users_per_round,
            epochs=3,
            user_epochs_per_round=2,
        )

        sync_trainer_report = create_sync_trainer(
            model=copy.deepcopy(global_model),
            local_lr=local_optimizer_lr,
            users_per_round=users_per_round,
            epochs=3,
            user_epochs_per_round=2,
            report_train_metrics=True,
        )
        sync_trainer_report.cfg.report_train_metrics_after_aggregation = True

        data_provider = FLDataProviderFromList(
            data_loader.fl_train_set(),
            data_loader.fl_eval_set(),
            data_loader.fl_test_set(),
            global_model,
        )

        # training with reporting the train metrics after aggregation
        modules = []
        for trainer in [sync_trainer_report, sync_trainer_no_report]:
            model, _ = trainer.train(
                data_provider,
                metrics_reporter,
                num_total_users=data_provider.num_users(),
                distributed_world_size=world_size,
            )
            modules.append(model.fl_get_module())

        # make sure metrics reporting after aggregation does not change global model
        assertEqual(FLModelParamUtils.get_mismatched_param(modules), "")

    def test_client_optimizer_creation_from_config(self):
        """
        Test if trainer can instanciate the correct client optimizer from config
        """
        for optimizer_config, trainer_type in zip(
            [LocalOptimizerSGDConfig(lr=1.0), LocalOptimizerFedProxConfig(mu=1.0)],
            [SyncTrainer, AsyncTrainer],
        ):
            config = (
                SyncTrainerConfig(client=ClientConfig(optimizer=optimizer_config))
                if trainer_type == SyncTrainer
                else AsyncTrainerConfig(client=ClientConfig(optimizer=optimizer_config))
            )
            trainer = instantiate(
                config, model=DummyAlphabetFLModel(), cuda_enabled=False
            )
            assertTrue(isinstance(trainer, trainer_type))

    @pytest.mark.parametrize(
        "config,trainer_type ",
        [
            (
                SyncServerConfig(
                    server_optimizer=FedAvgWithLROptimizerConfig(lr=1.0, momentum=0.0)
                ),
                SyncTrainer,
            ),
            (FedAdamAsyncAggregatorConfig(beta1=0.1), AsyncTrainer),
        ],
    )
    def test_server_optimizer_creation_from_config(self, config, trainer_type):
        """
        Test if trainer can instanciate correct aggregator config
        """
        config = (
            SyncTrainerConfig(server=config)
            if trainer_type == SyncTrainer
            else AsyncTrainerConfig(aggregator=config)
        )
        trainer = instantiate(config, model=DummyAlphabetFLModel(), cuda_enabled=False)
        assertTrue(isinstance(trainer, trainer_type))

    def test_same_training_results_with_post_aggregation_reporting(self):
        """
        creat two training instances,
        one with report_train_metrics_after_aggregation=True, another with False,
        check training results match
        """
        torch.manual_seed(1)
        # mock 26 rows
        shard_size = 4
        local_batch_size = 4
        world_size = 1
        dummy_dataset = DummyAlphabetDataset()
        fl_data_sharder = SequentialSharder(examples_per_shard=shard_size)
        data_loader = FLDatasetDataLoaderWithBatch(
            dummy_dataset,
            dummy_dataset,
            dummy_dataset,
            fl_data_sharder,
            local_batch_size,
            local_batch_size,
            local_batch_size,
        )

        users_per_round = 2
        local_optimizer_lr = 0.1
        torch.manual_seed(1)
        # first training instance
        global_model_1 = DummyAlphabetFLModel()
        metrics_reporter_1 = FakeMetricReporter()
        sync_trainer_1 = create_sync_trainer(
            model=global_model_1,
            local_lr=local_optimizer_lr,
            users_per_round=users_per_round,
            epochs=3,
            user_epochs_per_round=2,
        )
        data_provider = FLDataProviderFromList(
            data_loader.fl_train_set(),
            data_loader.fl_eval_set(),
            data_loader.fl_test_set(),
            global_model_1,
        )
        assertEqual(data_provider.num_users(), data_loader.num_total_users)
        # training with reporting the train metrics after aggregation
        sync_trainer_1.cfg.report_train_metrics = True
        sync_trainer_1.cfg.report_train_metrics_after_aggregation = True
        global_model_1, best_metric_1 = sync_trainer_1.train(
            data_provider,
            metrics_reporter_1,
            num_total_users=data_provider.num_users(),
            distributed_world_size=world_size,
        )

        torch.manual_seed(1)
        # second training instance
        global_model_2 = DummyAlphabetFLModel()
        metrics_reporter_2 = FakeMetricReporter()
        sync_trainer_2 = create_sync_trainer(
            model=global_model_2,
            users_per_round=users_per_round,
            local_lr=local_optimizer_lr,
            epochs=3,
            user_epochs_per_round=2,
        )
        # training without reporting the train metrics after aggregation
        sync_trainer_2.cfg.report_train_metrics = True
        sync_trainer_2.cfg.report_train_metrics_after_aggregation = False
        global_model_2, best_metric_2 = sync_trainer_2.train(
            data_provider,
            metrics_reporter_2,
            num_total_users=data_provider.num_users(),
            distributed_world_size=world_size,
        )

        # check two training instance produce same results
        assertEqual(
            FLModelParamUtils.get_mismatched_param(
                [global_model_1.fl_get_module(), global_model_2.fl_get_module()]
            ),
            "",
        )

    def test_different_metrics_with_aggregation_client_reporting(self):
        """
        creat two training instances,
        one with use_train_clients_for_aggregation_metrics=True, another with False,
        check that the aggregation eval metrics are different
        """
        torch.manual_seed(1)
        # mock 26 rows
        shard_size = 4
        local_batch_size = 4
        world_size = 1
        dummy_dataset = DummyAlphabetDataset()
        fl_data_sharder = SequentialSharder(examples_per_shard=shard_size)
        data_loader = FLDatasetDataLoaderWithBatch(
            dummy_dataset,
            dummy_dataset,
            dummy_dataset,
            fl_data_sharder,
            local_batch_size,
            local_batch_size,
            local_batch_size,
        )

        users_per_round = 2
        local_optimizer_lr = 0.1
        torch.manual_seed(1)
        # first training instance
        global_model_1 = DummyAlphabetFLModel()
        metrics_reporter_1 = SimpleMetricReporter()
        sync_trainer_1 = create_sync_trainer(
            model=global_model_1,
            local_lr=local_optimizer_lr,
            users_per_round=users_per_round,
            epochs=3,
            user_epochs_per_round=2,
        )
        data_provider = FLDataProviderFromList(
            data_loader.fl_train_set(),
            data_loader.fl_eval_set(),
            data_loader.fl_test_set(),
            global_model_1,
        )
        assertEqual(data_provider.num_users(), data_loader.num_total_users)
        # training with using training clients for aggregation training metrics
        sync_trainer_1.cfg.report_train_metrics_after_aggregation = True
        sync_trainer_1.cfg.use_train_clients_for_aggregation_metrics = True
        sync_trainer_1.cfg.report_train_metrics = True
        global_model_1, best_metric_1 = sync_trainer_1.train(
            data_provider,
            metrics_reporter_1,
            num_total_users=data_provider.num_users(),
            distributed_world_size=world_size,
        )

        torch.manual_seed(1)
        # second training instance
        global_model_2 = DummyAlphabetFLModel()
        metrics_reporter_2 = SimpleMetricReporter()
        sync_trainer_2 = create_sync_trainer(
            model=global_model_2,
            users_per_round=users_per_round,
            local_lr=local_optimizer_lr,
            epochs=3,
            user_epochs_per_round=2,
        )
        # training with using training clients for aggregation training metrics
        sync_trainer_2.cfg.report_train_metrics_after_aggregation = True
        sync_trainer_2.cfg.use_train_clients_for_aggregation_metrics = False
        sync_trainer_2.cfg.report_train_metrics = True
        global_model_2, best_metric_2 = sync_trainer_2.train(
            data_provider,
            metrics_reporter_2,
            num_total_users=data_provider.num_users(),
            distributed_world_size=world_size,
        )

        # Check that the reported metrics are different
        for batch_metrics_1, batch_metrics_2 in zip(
            metrics_reporter_1.batch_metrics, metrics_reporter_2.batch_metrics
        ):
            if (
                batch_metrics_1.loss != batch_metrics_2.loss
                or batch_metrics_1.num_examples != batch_metrics_2.num_examples
            ):
                return
        assert True, "Batch metrics same whether using training or random clients"

    def test_one_user_sequential_user_equivalent(self):
        """
        test equivalence of the following scenario,

        1. one user who got all the example, and BS = all examples,
        local optimizer LR = 1.0
        2. number of users = number of examples, and one exampler per user,
        users per round = all users, use FedAvg
        """

        torch.manual_seed(1)
        # create dummy FL model on alphabet
        global_model = DummyAlphabetFLModel()
        global_model_init = copy.deepcopy(global_model)
        # will be used later to verify training indeed took place
        global_model_init_copy = copy.deepcopy(global_model)
        metrics_reporter = FakeMetricReporter()

        num_training_examples = 32
        # one user, who got 32 examples
        shard_size = num_training_examples
        local_batch_size = num_training_examples
        world_size = 1
        dummy_dataset = DummyAlphabetDataset(num_rows=num_training_examples)
        (
            data_provider,
            data_loader,
        ) = DummyAlphabetDataset.create_data_provider_and_loader(
            dummy_dataset, shard_size, local_batch_size, global_model
        )
        assertEqual(
            data_loader.num_total_users, math.ceil(num_training_examples / shard_size)
        )

        users_per_round = 1
        local_optimizer_lr = 1.0
        epochs = 5
        torch.manual_seed(1)
        sync_trainer = create_sync_trainer(
            model=global_model,
            local_lr=local_optimizer_lr,
            users_per_round=users_per_round,
            epochs=epochs,
        )
        one_user_global_model, _eval_metric_one_user = sync_trainer.train(
            data_provider,
            metrics_reporter,
            num_total_users=data_provider.num_users(),
            distributed_world_size=world_size,
        )
        metrics_reporter.reset()

        # 32 users, one example each
        shard_size = 1
        local_batch_size = 1
        world_size = 1
        dummy_dataset = DummyAlphabetDataset(num_rows=num_training_examples)
        (
            data_provider,
            data_loader,
        ) = DummyAlphabetDataset.create_data_provider_and_loader(
            dummy_dataset, shard_size, local_batch_size, global_model
        )
        assertEqual(
            data_loader.num_total_users, math.ceil(num_training_examples / shard_size)
        )
        assertEqual(data_provider.num_users(), data_loader.num_total_users)
        # select all users to train in each round
        users_per_round = data_provider.num_users()
        torch.manual_seed(1)
        sync_trainer = create_sync_trainer(
            model=global_model_init,
            local_lr=local_optimizer_lr,
            users_per_round=users_per_round,
            epochs=epochs,
        )
        all_users_global_model, _eval_metric_all_user = sync_trainer.train(
            data_provider,
            metrics_reporter,
            num_total_users=data_provider.num_users(),
            distributed_world_size=world_size,
        )
        assertEqual(
            verify_models_equivalent_after_training(
                one_user_global_model,
                all_users_global_model,
                global_model_init_copy,
                rel_epsilon=1e-4,
                abs_epsilon=1e-6,
            ),
            "",
        )

    def test_training_with_armijo_line_search(self):
        """
        test Armijo line-search for local LR scheduling

        using shrinking factor = 1.0  == constant LR
        """

        torch.manual_seed(1)
        # create dummy FL model on alphabet
        global_model = DummyAlphabetFLModel()
        global_model_init = copy.deepcopy(global_model)
        # will be used later to verify training indeed took place
        global_model_init_copy = copy.deepcopy(global_model)
        metrics_reporter = FakeMetricReporter()

        # one user, who got 26 examples
        shard_size = 26
        local_batch_size = 26
        world_size = 1
        dummy_dataset = DummyAlphabetDataset()
        (
            data_provider,
            data_loader,
        ) = DummyAlphabetDataset.create_data_provider_and_loader(
            dummy_dataset, shard_size, local_batch_size, global_model
        )
        assertEqual(data_loader.num_total_users, math.ceil(26 / shard_size))
        assertEqual(data_provider.num_users(), data_loader.num_total_users)
        users_per_round = 1
        local_optimizer_lr = 1.0
        epochs = 5
        torch.manual_seed(1)
        sync_trainer = create_sync_trainer(
            model=global_model,
            local_lr=local_optimizer_lr,
            users_per_round=users_per_round,
            epochs=epochs,
        )

        constant_lr_model, _ = sync_trainer.train(
            data_provider,
            metrics_reporter,
            num_total_users=data_provider.num_users(),
            distributed_world_size=world_size,
        )
        metrics_reporter.reset()

        torch.manual_seed(1)
        # training with Armijo line-search LR scheduler with
        # shrinking factor = 1.0 (no shrinking)
        local_lr_scheduler_config = ArmijoLineSearchSchedulerConfig(
            shrinking_factor=1.0
        )
        sync_trainer_with_scheduler = create_sync_trainer(
            model=global_model_init,
            local_lr=local_optimizer_lr,
            users_per_round=users_per_round,
            epochs=epochs,
            local_lr_scheduler=local_lr_scheduler_config,
        )

        armijo_ls_model, _ = sync_trainer_with_scheduler.train(
            data_provider,
            metrics_reporter,
            num_total_users=data_provider.num_users(),
            distributed_world_size=world_size,
        )
        metrics_reporter.reset()
        assertEqual(
            verify_models_equivalent_after_training(
                constant_lr_model,
                armijo_ls_model,
                global_model_init_copy,
                rel_epsilon=1e-6,
                abs_epsilon=1e-6,
            ),
            "",
        )

    def _test_fl_nonfl_equivalent(
        self,
        num_examples: int,
        num_fl_users: int,
        epochs: int,
        local_lr: float,
        server_config: SyncServerConfig,
    ):
        """
        Given:
            data_for_fl={user1:batch1, user2:batch2}
            data_for_non_fl={batch1, batch2}

        Check that the following produce the same trained model:
        1. FL training, 1 user per round. global_opt=SGD, local_lr=x, global_lr=x
        2. Non-FL training, opt=SGD, lr=x
        """
        torch.manual_seed(1)
        # create dummy FL model on alphabet
        global_model = DummyAlphabetFLModel()
        # will be used later to verify training indeed took place
        global_model_init_copy = copy.deepcopy(global_model)
        # num_fl_users users, each with num_examples/num_fl_users training examples
        assertTrue(
            num_examples % num_fl_users == 0,
            f"Expect num_examples({num_examples}) to be multiple of num_fl_users({num_fl_users})",
        )
        shard_size = num_examples // num_fl_users
        batch_size = num_examples // num_fl_users
        world_size = 1
        dummy_dataset = DummyAlphabetDataset(num_examples)
        (
            data_provider,
            data_loader,
        ) = DummyAlphabetDataset.create_data_provider_and_loader(
            dummy_dataset, shard_size, batch_size, global_model
        )
        assertEqual(data_loader.num_total_users, math.ceil(num_examples / shard_size))
        assertEqual(data_provider.num_users(), data_loader.num_total_users)

        torch.manual_seed(1)

        sync_trainer = create_sync_trainer(
            model=global_model,
            local_lr=local_lr,
            users_per_round=1,
            epochs=epochs,
            do_eval=False,
            server_config=server_config,
        )
        fl_model, _ = sync_trainer.train(
            data_provider,
            metric_reporter=FakeMetricReporter(),
            num_total_users=data_provider.num_users(),
            distributed_world_size=world_size,
        )

        # non-FL training
        dummy_dataset = DummyAlphabetDataset(num_examples)
        data_loader = torch.utils.data.DataLoader(
            dummy_dataset, batch_size=batch_size, shuffle=False
        )
        nonfl_model = copy.deepcopy(global_model_init_copy)

        optimizer = instantiate(
            config=server_config.server_optimizer,
            model=nonfl_model.fl_get_module(),
        )

        FLTestUtils.run_nonfl_training(
            model=nonfl_model,
            optimizer=optimizer,
            data_loader=data_loader,
            epochs=epochs,
        )

        error_msg = verify_models_equivalent_after_training(
            fl_model,
            nonfl_model,
            global_model_init_copy,
            rel_epsilon=1e-4,
            abs_epsilon=1e-6,
        )
        assertEmpty(error_msg, msg=error_msg)

    def test_fl_nonfl_equivalent_global_optimizer_sgd(self):
        """
        Given:
            batch_size=13
            data_for_fl={user1:batch1, user2:batch2}
            data_for_non_fl={batch1, batch2}

        Check that the following produce the same trained model:
        1. FL training, 1 user per round. global_opt=SGD, local_lr=x, global_lr=x
        2. Non-FL training, opt=SGD, lr=x
        """
        self._test_fl_nonfl_equivalent(
            num_examples=26,
            num_fl_users=2,
            epochs=5,
            local_lr=1.0,
            server_config=SyncServerConfig(
                server_optimizer=FedAvgWithLROptimizerConfig(lr=1.0, momentum=0.0),
                active_user_selector=SequentialActiveUserSelectorConfig(),
            ),
        )

    def test_fl_nonfl_equivalent_global_optimizer_adam(self):
        """
        Given:
            batch_size=16 (bs=num_examples/num_fl_users)
            data_for_fl={user1:batch1, user2:batch2}
            data_for_non_fl={batch1, batch2}

        Check that the following produce the same trained model:
        1. FL training, 1 user per round. GlobalOpt=Adam, local_lr=1.0, global_lr=x
        2. Non-FL training, opt=Adam, lr=x
        """
        self._test_fl_nonfl_equivalent(
            num_examples=32,
            num_fl_users=2,
            epochs=5,
            local_lr=1.0,
            server_config=SyncServerConfig(
                server_optimizer=FedAdamOptimizerConfig(lr=0.001, eps=1e-2),
                active_user_selector=SequentialActiveUserSelectorConfig(),
            ),
        )

    def test_client_overselection(self):
        """
        test client overselection by equivalence of the two setups:

        1. two users in the population, user 0 has two more examples than user 1.
        UPR = 1, dropout_rate = 0.5. Hence (1/0.5=2) users will be over-selected.
        simulate training time with mean = 1.0, std = 0.0 per example.
        Hence user 0 will always be dropped due. Trains with 1 FL epochs. rounds
        per epoch = (epoch / UPR = 2), hence user 1 will be trained twice.

        2. single user in the population, which is user 1 the the setting above.
        UPR = 1, dropout = 1.0. two FL epochs.
        """
        torch.manual_seed(1)
        # create dummy FL model on alphabet
        global_model = DummyAlphabetFLModel()
        # keep a copy of initial model to trigger another training instance
        global_model_init = copy.deepcopy(global_model)
        # dummy alphabet dataset
        dummy_dataset = DummyAlphabetDataset()
        # two users, one gets 1 more example than the other
        shard_size = len(dummy_dataset) // 2 + 1
        local_batch_size = 2
        (
            data_provider,
            data_loader,
        ) = DummyAlphabetDataset.create_data_provider_and_loader(
            dummy_dataset, shard_size, local_batch_size, global_model
        )
        assertEqual(data_provider.num_users(), data_loader.num_total_users)
        # assert first user gets (dummy_dataset.num_rows / 2) + 1 data point,
        # the second user gets (dummy_dataset.num_rows / 2) - 1 data point
        assertEqual(data_provider[0].num_examples(), dummy_dataset.num_rows / 2 + 1)
        assertEqual(data_provider[1].num_examples(), dummy_dataset.num_rows / 2 - 1)
        # shared trainer config between two training instance
        users_per_round = 1
        local_optimizer_lr = 1.0
        epochs = 1
        torch.manual_seed(1)
        # training with overselection
        dropout_rate = 0.5
        sync_trainer_overselection = create_sync_trainer(
            model=global_model,
            local_lr=local_optimizer_lr,
            users_per_round=users_per_round,
            epochs=epochs,
            dropout_rate=dropout_rate,
            timeout_simulator_config=GaussianTimeOutSimulatorConfig(
                duration_distribution_generator=PerExampleGaussianDurationDistributionConfig(
                    training_duration_mean=1.0, training_duration_sd=0.0
                ),
                timeout_wall_per_round=999999,
                fl_stopping_time=9999999,
            ),
        )
        model_with_overselection, _ = sync_trainer_overselection.train(
            data_provider,
            metric_reporter=FakeMetricReporter(),
            num_total_users=data_provider.num_users(),
            distributed_world_size=1,
        )
        # another training instance: only user[1] in the user population
        # removing user 0 from dataset, assign user 1 to be user 0
        data_provider._users[0] = copy.deepcopy(data_provider._users[1])
        data_provider._users.pop(1)
        # only a single user after remove user 0
        assertEqual(data_provider.num_users(), 1)
        global_model = copy.deepcopy(global_model_init)
        torch.manual_seed(1)
        dropout_rate = 1.0
        epochs = 2
        sync_trainer_single_user = create_sync_trainer(
            model=global_model,
            local_lr=local_optimizer_lr,
            users_per_round=users_per_round,
            epochs=epochs,
            dropout_rate=dropout_rate,
        )
        model_single_user, _ = sync_trainer_single_user.train(
            data_provider,
            metric_reporter=FakeMetricReporter(),
            num_total_users=data_provider.num_users(),
            distributed_world_size=1,
        )
        assertEqual(
            verify_models_equivalent_after_training(
                model_with_overselection,
                model_single_user,
                global_model_init,
                rel_epsilon=1e-4,
                abs_epsilon=1e-6,
            ),
            "",
        )

    def test_partial_update_from_clients(self):
        """
        test the equivalence of these two training instance
        1. UPR=1. User dataset has 52 characters [a,b,c,d .... x, y ,z, a, b, c... x,y,z],
        where each character appears twice in order. Timeout limit is set to
        just enough for training half of the dataset

        2. UPR=1. User daraset has 26 characters [a,b,c,d ...,x,y,z]. no timeout
        """
        torch.manual_seed(1)
        # create dummy FL model on alphabet
        global_model = DummyAlphabetFLModel()
        # keep a copy of initial model to trigger another training instance
        global_model_init = copy.deepcopy(global_model)
        # dummy alphabet dataset, getting each character twrice
        num_rows = 52
        dummy_dataset = DummyAlphabetDataset(num_rows)
        # a single user getting all examples
        shard_size = len(dummy_dataset)
        local_batch_size = 2
        (
            data_provider,
            data_loader,
        ) = DummyAlphabetDataset.create_data_provider_and_loader(
            dummy_dataset, shard_size, local_batch_size, global_model
        )
        # shared trainer config between two training instance
        users_per_round = 1
        local_optimizer_lr = 1.0
        epochs = 1
        torch.manual_seed(1)
        # training time just enought for 26 examples, although user has
        # 52 examples
        sync_trainer_timeout = create_sync_trainer(
            model=copy.deepcopy(global_model),
            local_lr=local_optimizer_lr,
            users_per_round=users_per_round,
            epochs=epochs,
            timeout_simulator_config=GaussianTimeOutSimulatorConfig(
                duration_distribution_generator=PerExampleGaussianDurationDistributionConfig(
                    training_duration_mean=1.0, training_duration_sd=0.0
                ),
                timeout_wall_per_round=26,
                fl_stopping_time=9999999,
            ),
        )
        model_with_timeout, _ = sync_trainer_timeout.train(
            data_provider,
            metric_reporter=FakeMetricReporter(),
            num_total_users=data_provider.num_users(),
            distributed_world_size=1,
        )
        # dummy alphabet dataset, getting each character once
        num_rows = 26
        torch.manual_seed(1)
        dummy_dataset = DummyAlphabetDataset(num_rows)
        # a single user getting all examples
        shard_size = len(dummy_dataset)
        local_batch_size = 2
        (
            data_provider,
            data_loader,
        ) = DummyAlphabetDataset.create_data_provider_and_loader(
            dummy_dataset, shard_size, local_batch_size, global_model
        )
        torch.manual_seed(1)
        # training time just enought for 26 examples
        sync_trainer_timeout = create_sync_trainer(
            model=copy.deepcopy(global_model),
            local_lr=local_optimizer_lr,
            users_per_round=users_per_round,
            epochs=epochs,
        )
        model_no_timeout, _ = sync_trainer_timeout.train(
            data_provider,
            metric_reporter=FakeMetricReporter(),
            num_total_users=data_provider.num_users(),
            distributed_world_size=1,
        )
        assertEqual(
            verify_models_equivalent_after_training(
                model_with_timeout,
                model_no_timeout,
                global_model_init,
                rel_epsilon=1e-4,
                abs_epsilon=1e-6,
            ),
            "",
        )

    def test_client_metric_reporting(self):
        """
        Test that per-client reporting reports exactly every
        ``client_metrics_reported_per_epoch`` as defined in the config.
        """
        torch.manual_seed(1)
        # create dummy FL model on alphabet
        global_model = DummyAlphabetFLModel()
        # dummy alphabet dataset, getting each character twrice
        num_rows = 52
        num_users = 4
        dummy_dataset = DummyAlphabetDataset(num_rows)
        # a single user getting all examples
        shard_size = int(len(dummy_dataset) / num_users)
        local_batch_size = 2
        (
            data_provider,
            data_loader,
        ) = DummyAlphabetDataset.create_data_provider_and_loader(
            dummy_dataset, shard_size, local_batch_size, global_model
        )
        users_per_round = 1
        local_optimizer_lr = 1.0
        momentum = 0.9
        epochs = 6

        sync_trainer_with_client_reports = SyncTrainer(
            model=global_model,
            cuda_enabled=False,
            **OmegaConf.structured(
                SyncTrainerConfig(
                    epochs=epochs,
                    do_eval=True,
                    users_per_round=users_per_round,
                    always_keep_trained_model=False,
                    train_metrics_reported_per_epoch=1,
                    eval_epoch_frequency=1,
                    report_train_metrics=False,
                    report_train_metrics_after_aggregation=True,
                    client=ClientConfig(
                        epochs=1,
                        optimizer=LocalOptimizerSGDConfig(
                            lr=local_optimizer_lr, momentum=momentum
                        ),
                        max_clip_norm_normalized=None,
                    ),
                    server=SyncServerConfig(
                        server_optimizer=FedAvgWithLROptimizerConfig(
                            lr=1.0, momentum=0.0
                        ),
                        active_user_selector=SequentialActiveUserSelectorConfig(),
                    ),
                    report_client_metrics=True,
                    report_client_metrics_after_epoch=True,
                    client_metrics_reported_per_epoch=3,
                )
            ),
        )
        metrics_reporter = MetricsReporterWithMockedChannels()

        model, _ = sync_trainer_with_client_reports.train(
            data_provider,
            metric_reporter=metrics_reporter,
            num_total_users=data_provider.num_users(),
            distributed_world_size=1,
        )

        def count_word(result, word):
            return str(result).count(word)

        # If client_metrics_reported_per_epoch = 3 and number of epochs = 6
        # per client eval metrics should be reported twice.
        assertEqual(
            count_word(metrics_reporter.stdout_results, "Per_Client_Eval"),
            2,
            metrics_reporter.stdout_results,
        )

    def _get_tensorboard_results_from_training(
        self, num_total_users, num_epochs, users_per_round
    ) -> List[MockRecord]:
        # dataset has 26 rows
        assertTrue(num_total_users <= 26, "Can't have more than 26 users")
        torch.manual_seed(1)
        # 26 rows in data
        shard_size = int(math.ceil(26 / num_total_users))
        local_batch_size = 4
        dummy_dataset = DummyAlphabetDataset()
        fl_data_sharder = SequentialSharder(examples_per_shard=shard_size)
        data_loader = FLDatasetDataLoaderWithBatch(
            dummy_dataset,
            dummy_dataset,
            dummy_dataset,
            fl_data_sharder,
            local_batch_size,
            local_batch_size,
            local_batch_size,
        )

        torch.manual_seed(1)
        # first training instance
        global_model = DummyAlphabetFLModel()
        metrics_reporter = MetricsReporterWithMockedChannels()
        sync_trainer = create_sync_trainer(
            model=global_model,
            local_lr=0.1,
            users_per_round=users_per_round,
            epochs=num_epochs,
            user_epochs_per_round=1,
        )
        data_provider = FLDataProviderFromList(
            data_loader.fl_train_set(),
            data_loader.fl_eval_set(),
            data_loader.fl_test_set(),
            global_model,
        )
        assertEqual(data_provider.num_users(), data_loader.num_total_users)
        sync_trainer.cfg.report_train_metrics = True
        sync_trainer.cfg.report_train_metrics_after_aggregation = True
        global_model, best_metric = sync_trainer.train(
            data_provider,
            metrics_reporter,
            num_total_users=num_total_users,
            distributed_world_size=1,
        )
        return metrics_reporter.tensorboard_results

    def test_tensorboard_metrics_reporting_simple(self):
        """Train with tensorboard metrics reporter for one epoch, 5 rounds per epoch.
        Test for 2 things:
        a) Train, Aggregation and Eval metrics are reported once
        b) Metric reporting happens at the end of the epoch (when global_round=5)
        """
        num_total_users = 5
        num_epochs = 1
        users_per_round = 1
        tensorboard_results: List[
            MockRecord
        ] = self._get_tensorboard_results_from_training(
            num_total_users=num_total_users,
            num_epochs=num_epochs,
            users_per_round=users_per_round,
        )

        # ensure that train, aggregation and eval metrics are all reported once
        for stage in [
            TrainingStage.TRAINING,
            TrainingStage.AGGREGATION,
            TrainingStage.EVAL,
        ]:
            tag = f"Loss/{TrainingStage(stage).name.title()}"
            num_entries = sum(record.tag == tag for record in tensorboard_results)
            # we report once per epoch
            assertEqual(num_entries, 1)

        # training runs for 1 epoch, or 5 rounds
        # when train/eval results are reported, global_round_num should be 5
        global_steps_reported_expected = [5, 5, 5]  # noqa: F841
        global_steps_reported_actual = [
            record.global_step for record in tensorboard_results
        ]
        assertEqual(
            global_steps_reported_actual,
            global_steps_reported_expected,
            f"Actual global steps: {global_steps_reported_actual}, Expected global steps:{global_steps_reported_expected}",
        )

    def test_tensorboard_metrics_reporting_complex(self):
        """Train with tensorboard metrics reporter. Ensure eval and train metrics are
        correctly reported to tensorboard
        """
        # TODO: run with different values of num_epochs, num_total_users, users_per_round,
        # train_metrics_reported_per_epoch, eval_epoch_frequency
        num_total_users = 10
        num_epochs = 3
        users_per_round = 2
        tensorboard_results: List[
            MockRecord
        ] = self._get_tensorboard_results_from_training(
            num_total_users=num_total_users,
            num_epochs=num_epochs,
            users_per_round=users_per_round,
        )

        # ensure that train, aggregation and eval metrics are all reported once per epoch
        for stage in [
            TrainingStage.TRAINING,
            TrainingStage.AGGREGATION,
            TrainingStage.EVAL,
        ]:
            tag = f"Loss/{TrainingStage(stage).name.title()}"
            num_entries = sum(record.tag == tag for record in tensorboard_results)
            # we report once per epoch
            assertEqual(num_entries, num_epochs)

        rounds_per_epoch = int(math.ceil(num_total_users / users_per_round))
        # example. if num_epochs=3, and rounds_per_epoch=5, global_steps_at_epoch_end will be [5, 10, 15]
        epochs = list(range(1, num_epochs + 1))
        global_steps_at_epoch_end = [e * rounds_per_epoch for e in epochs]

        # we report 3 metrics in each epoch: Training, Aggregation and Eval
        num_records_in_report = 3
        # example. if global_steps_at_epoch_end = [5, 10, 15], and num_records_in_report = 2,
        # then global_steps_reported_expected = [5, 5, 10,10, 15,15]
        # noqa: F841  variable not used, fixed in next diff
        global_steps_reported_expected = [
            step
            for step in global_steps_at_epoch_end
            for k in range(num_records_in_report)
        ]
        # noqa: F841  variable not used, fixed in next diff
        global_steps_reported_actual = [
            record.global_step for record in tensorboard_results
        ]
        assertEqual(
            global_steps_reported_actual,
            global_steps_reported_expected,
            f"Actual global steps: {global_steps_reported_actual}, Expected global steps:{global_steps_reported_expected}",
        )

    @pytest.mark.parametrize(
        "page_turn_freq,users_per_round, pages_used",
        [
            (0.99, 5, 6),
            (0.99, 10, 6),
            (0.50, 10, 9),
            (0.50, 5, 9),
        ],
    )
    def test_sync_trainer_with_page_data_provider(
        self, page_turn_freq, users_per_round, pages_used
    ):
        """
        Test if sync trainer works properly with paged data loader
        Assumptions:
            users_per_page = 10 total number of users = 50
            batch_size = 1
            train for 2 global epochs

        Cases:
        1. Base case: user_per_round is half of page_size, turn page after all users are used
        2. Boundary case: users_per_round = page_size, turn page after all users are used
        3. Boundary case: users_per_round = page_size, turn page after half of users in the page are used
        """
        id_col = "label"
        num_total_users = 50
        inline_config = InlineDatasetConfig(
            num_users=num_total_users,
            data_cols=["user_n"],
            id_col=id_col,
        )
        dataloader = create_dataloader(
            num_users_per_page=10,
            num_total_users=num_total_users,
            batch_size=1,
            inline_config=inline_config,
            sharding_col_name=id_col,
            use_nid=True,
        )
        global_model = SampleNetHive()
        data_provider = PagedDataProvider(
            dataloader, model=global_model, page_turn_freq=page_turn_freq
        )
        sync_trainer = create_sync_trainer(
            model=global_model,
            local_lr=0.1,
            users_per_round=users_per_round,
            epochs=1,
        )

        sync_trainer.train(
            data_provider,
            metric_reporter=FakeMetricReporter(),
            # Note: We're using num_total_users and
            # not data_provider.num_users()
            num_total_users=num_total_users,
            distributed_world_size=1,
        )
        assertEqual(data_provider.pages_used, pages_used)
