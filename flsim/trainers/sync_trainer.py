#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from time import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from flsim.channels.message import Message
from flsim.clients.base_client import Client
from flsim.clients.dp_client import DPClient, DPClientConfig
from flsim.common.timeline import Timeline
from flsim.data.data_provider import IFLDataProvider
from flsim.interfaces.metrics_reporter import IFLMetricsReporter, Metric, TrainingStage
from flsim.interfaces.model import IFLModel
from flsim.servers.sync_dp_servers import SyncDPSGDServerConfig
from flsim.servers.sync_secagg_servers import SyncSecAggServerConfig
from flsim.servers.sync_servers import (
    FedAvgOptimizerConfig,
    ISyncServer,
    SyncServerConfig,
)
from flsim.trainers.trainer_base import FLTrainer, FLTrainerConfig
from flsim.utils.config_utils import fullclassname, init_self_cfg, is_target
from flsim.utils.distributed.fl_distributed import FLDistributedUtils
from flsim.utils.fl.common import FLModelParamUtils
from flsim.utils.fl.stats import RandomVariableStatsTracker
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm


class SyncTrainer(FLTrainer):
    """Implements synchronous Federated Learning Training.

    Defaults to Federated Averaging (FedAvg): https://arxiv.org/abs/1602.05629
    """

    def __init__(
        self,
        *,
        model: IFLModel,
        cuda_enabled: bool = False,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=SyncTrainerConfig,
            **kwargs,
        )

        super().__init__(model=model, cuda_enabled=cuda_enabled, **kwargs)
        self.server: ISyncServer = instantiate(
            # pyre-ignore[16]
            self.cfg.server,
            global_model=model,
            channel=self.channel,
        )
        self.clients = {}

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        if OmegaConf.is_missing(cfg.server, "_target_"):
            cfg.server = SyncServerConfig(optimizer=FedAvgOptimizerConfig())

    def global_model(self) -> IFLModel:
        """self.global_model() is owned by the server, not by SyncTrainer"""
        return self.server.global_model

    @property
    def is_user_level_dp(self):
        return is_target(self.cfg.server, SyncDPSGDServerConfig)

    @property
    def is_sample_level_dp(self):
        return is_target(self.cfg.client, DPClientConfig)

    @property
    def is_secure_aggregation_enabled(self):
        return is_target(self.cfg.server, SyncSecAggServerConfig)

    def create_or_get_client_for_data(self, dataset_id: int, datasets: Any):
        """Creates clients for a round. This function is called UPR * num_rounds
        times per training run. Here, we use <code>OmegaConf.structured</code>
        instead of <code>hydra.instantiate</code> to minimize the overhead of
        hydra object creation.
        """
        if self.is_sample_level_dp:
            client = DPClient(
                # pyre-ignore[16]
                **OmegaConf.structured(self.cfg.client),
                dataset=datasets.get_train_user(dataset_id),
                name=f"client_{dataset_id}",
                timeout_simulator=self._timeout_simulator,
                store_last_updated_model=self.cfg.report_client_metrics,
                channel=self.channel,
                cuda_manager=self._cuda_state_manager,
            )
        else:
            client = Client(
                **OmegaConf.structured(self.cfg.client),
                dataset=datasets.get_train_user(dataset_id),
                name=f"client_{dataset_id}",
                timeout_simulator=self._timeout_simulator,
                store_last_updated_model=self.cfg.report_client_metrics,
                channel=self.channel,
                cuda_manager=self._cuda_state_manager,
            )
        self.clients[dataset_id] = client
        return self.clients[dataset_id]

    def train(
        self,
        data_provider: IFLDataProvider,
        metric_reporter: IFLMetricsReporter,
        num_total_users: int,
        distributed_world_size: int,
        rank: int = 0,
    ) -> Tuple[IFLModel, Any]:
        """Trains and evaluates the model, modifying the model state. Iterates over the
        number of epochs specified in the config, and for each epoch iterates over the
        number of rounds per epoch, i.e. the number of total users divided by the number
        of users per round. For each round:

            1. Trains the model in a federated way: different local models are trained
                with local data from different clients, and are averaged into a new
                global model at the end of each round.
            2. Evaluates the new global model using evaluation data, if desired.
            3. Calculates metrics based on evaluation results and selects the best model.

        Args:
            data_provider: provides training, evaluation, and test data iterables and
                gets a user's data based on user ID
            metric_reporter: computes and reports metrics of interest such as accuracy
                or perplexity
            num_total_users: number of total users for training

        Returns:
            model, best_metric: the trained model together with the best metric

        Note:
            Depending on the chosen active user selector, we may not iterate over
            all users in a given epoch.
        """
        # set up synchronization utilities for distributed training
        FLDistributedUtils.setup_distributed_training(
            distributed_world_size, use_cuda=self.cuda_enabled
        )  # TODO do not call distributed utils here, this is upstream responsibility
        self.logger.info(f" dist world size = {distributed_world_size}")

        if rank != 0:
            FLDistributedUtils.suppress_output()

        # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
        assert self.cfg.users_per_round % distributed_world_size == 0

        best_metric = None
        best_model_state = self.global_model().fl_get_module().state_dict()
        # last_best_epoch = 0
        users_per_round = min(self.cfg.users_per_round, num_total_users)

        self.data_provider = data_provider
        num_rounds_in_epoch = self.rounds_in_one_epoch(num_total_users, users_per_round)
        num_users_on_worker = data_provider.num_train_users()
        self.logger.debug(
            f"num_users_on_worker: {num_users_on_worker}, "
            f"users_per_round: {users_per_round}, "
            f"num_total_users: {num_total_users}"
        )
        # torch.multinomial requires int instead of float, cast it as int
        users_per_round_on_worker = int(users_per_round / distributed_world_size)
        self._validate_users_per_round(users_per_round_on_worker, num_users_on_worker)

        self.logger.info("Start training")
        if self.logger.isEnabledFor(logging.DEBUG):
            norm = FLModelParamUtils.debug_model_norm(
                self.global_model().fl_get_module()
            )
            self.logger.debug(
                self.cuda_enabled and distributed_world_size > 1,
                f"from worker {rank}: model norm is {norm} after round {iter}",
            )

        # main training loop
        num_int_epochs = math.ceil(self.cfg.epochs)
        for epoch in tqdm(
            range(1, num_int_epochs + 1), desc="Epoch", unit="epoch", position=0
        ):
            for round in tqdm(
                range(1, num_rounds_in_epoch + 1),
                desc="Round",
                unit="round",
                position=0,
            ):
                timeline = Timeline(
                    epoch=epoch, round=round, rounds_per_epoch=num_rounds_in_epoch
                )

                t = time()
                clients = self._client_selection(
                    num_users_on_worker,
                    users_per_round_on_worker,
                    data_provider,
                    self.global_model(),
                    epoch,
                )
                self.logger.info(f"Client Selection took: {time() - t} s.")

                agg_metric_clients = self._choose_clients_for_post_aggregation_metrics(
                    train_clients=clients,
                    num_total_users=num_users_on_worker,
                    users_per_round=users_per_round_on_worker,
                )

                # training on selected clients for the round
                self.logger.info(f"# clients/round on worker {rank}: {len(clients)}.")
                self._train_one_round(
                    timeline=timeline,
                    clients=clients,
                    agg_metric_clients=agg_metric_clients,
                    users_per_round=users_per_round,
                    metric_reporter=metric_reporter
                    if self.cfg.report_train_metrics
                    else None,
                )

                if self.logger.isEnabledFor(logging.DEBUG):
                    norm = FLModelParamUtils.debug_model_norm(
                        self.global_model().fl_get_module()
                    )
                    self.logger.debug(
                        self.cuda_enabled and distributed_world_size > 1,
                        f"from worker {rank}: model norm: {norm} @ "
                        f"epoch:{epoch}, round:{round}",
                    )

                # report training success rate and training time variance
                if rank == 0:
                    if (
                        self._timeout_simulator.sample_mean_per_user != 0
                        or self._timeout_simulator.sample_var_per_user != 0
                    ):
                        self.logger.info(
                            f"mean training time/user: "
                            f"{self._timeout_simulator.sample_mean_per_user}",
                            f"variance of training time/user: "
                            f"{self._timeout_simulator.sample_var_per_user}",
                        )

                    t = time()
                    (best_metric, best_model_state,) = self._maybe_run_evaluation(
                        timeline=timeline,
                        data_provider=data_provider,
                        metric_reporter=metric_reporter,
                        best_metric=best_metric,
                        best_model_state=best_model_state,
                    )
                    self.logger.info(f"Evaluation took {time() - t} s.")

                if self.stop_fl_training(
                    epoch=epoch, round=round, num_rounds_in_epoch=num_rounds_in_epoch
                ):
                    break

            # pyre-fixme[61]: `timeline` may not be initialized here.
            self._report_post_epoch_client_metrics(timeline, metric_reporter)
            if self.stop_fl_training(
                epoch=epoch,
                round=round,  # pyre-fixme[61]: `round` may not be initialized here.
                num_rounds_in_epoch=num_rounds_in_epoch,
            ):
                break

        if rank == 0 and best_metric is not None:
            self._save_model_and_metrics(self.global_model(), best_model_state)

        return self.global_model(), best_metric

    def stop_fl_training(self, *, epoch, round, num_rounds_in_epoch) -> bool:
        """Stops FL training when the necessary number of steps/epochs have been
        completed in case of fractional epochs or if clients time out.
        """
        global_round_num = (epoch - 1) * num_rounds_in_epoch + round
        return (
            (global_round_num / num_rounds_in_epoch)
            >= self.cfg.epochs  # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
            or self._timeout_simulator.stop_fl()
        )

    def _drop_overselected_users(
        self, clents_triggered: List[Client], num_users_keep: int
    ) -> List[Client]:
        """Sorts users by their training time, and only keeps num_users_keep users."""
        all_training_times = [c.get_total_training_time() for c in clents_triggered]
        all_training_times.sort()
        # only select first num_users_keep userids sort by their finish time
        num_users_keep = min([num_users_keep, len(all_training_times)])
        last_user_time = all_training_times[num_users_keep - 1]
        num_users_added = 0
        clients_used = []
        for c in clents_triggered:
            # if two clients finished at the same time, order for entering
            # the cohort is arbitrary
            if (c.get_total_training_time() <= last_user_time) and (
                num_users_added < num_users_keep
            ):
                num_users_added += 1
                clients_used.append(c)

        return clients_used

    def _client_selection(
        self,
        num_users: int,
        users_per_round: int,
        data_provider: IFLDataProvider,
        global_model: IFLModel,
        epoch: int,
    ) -> List[Client]:
        # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
        num_users_overselected = math.ceil(users_per_round / self.cfg.dropout_rate)
        user_indices_overselected = self.server.select_clients_for_training(
            num_total_users=num_users,
            users_per_round=num_users_overselected,
            data_provider=data_provider,
            epoch=epoch,
        )
        clients_triggered = [
            self.create_or_get_client_for_data(i, self.data_provider)
            for i in user_indices_overselected
        ]
        clients_to_train = self._drop_overselected_users(
            clients_triggered, users_per_round
        )
        return clients_to_train

    def _save_model_and_metrics(self, model: IFLModel, best_model_state):
        model.fl_get_module().load_state_dict(best_model_state)

    def _train_one_round(
        self,
        timeline: Timeline,
        clients: Iterable[Client],
        agg_metric_clients: Iterable[Client],
        users_per_round: int,
        metric_reporter: Optional[IFLMetricsReporter],
    ) -> None:
        """Trains the global model for one training round.

        Args:
            timeline: information about the round, epoch, round number, ...
            clients: clients for the round
            agg_metric_clients: clients for evaluating the post-aggregation
                training metrics
            users_per_round: the number of participating users
            metric_reporter: the metric reporter to pass to other methods
        """
        t = time()
        self.server.init_round()
        self.logger.info(f"Round initialization took {time() - t} s.")

        def update(client):
            client_delta, weight = client.generate_local_update(
                self.global_model(), metric_reporter
            )
            self.server.receive_update_from_client(Message(client_delta, weight))

        t = time()
        for client in clients:
            update(client)
        self.logger.info(f"Collecting round's clients took {time() - t} s.")

        t = time()
        self.server.step()
        self.logger.info(f"Finalizing round took {time() - t} s.")

        t = time()
        self._report_train_metrics(
            model=self.global_model(),
            timeline=timeline,
            metric_reporter=metric_reporter,
        )
        self._evaluate_global_model_after_aggregation_on_train_clients(
            clients=agg_metric_clients,
            model=self.global_model(),
            timeline=timeline,
            users_per_round=users_per_round,
            metric_reporter=metric_reporter,
        )
        self._calc_post_epoch_communication_metrics(
            timeline,
            metric_reporter,
        )
        self.logger.info(f"Aggregate round reporting took {time() - t} s.")

    def _choose_clients_for_post_aggregation_metrics(
        self,
        train_clients: Iterable[Client],
        num_total_users: int,
        users_per_round: int,
    ) -> Iterable[Client]:
        """Chooses clients for the post-aggregation training metrics.
        Depending on the config parameters, either returns the round's
        training clients or new randomly drawn clients.
        """
        # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
        if self.cfg.use_train_clients_for_aggregation_metrics:
            return train_clients

        # For the post-aggregation metrics, evaluate on new users
        agg_metric_client_idcs = torch.multinomial(
            torch.ones(num_total_users, dtype=torch.float),
            users_per_round,
            replacement=False,
        ).tolist()

        agg_metric_clients = [
            self.create_or_get_client_for_data(i, self.data_provider)
            for i in agg_metric_client_idcs
        ]
        return agg_metric_clients

    def _calc_privacy_metrics(
        self,
        clients: Iterable[Client],
        model: IFLModel,
        metric_reporter: Optional[IFLMetricsReporter],
    ) -> List[Metric]:
        """Calculates privacy metrics."""
        metrics = []
        if self.is_user_level_dp:
            user_eps = self.server.privacy_budget.epsilon  # pyre-fixme
            metrics.append(Metric("user level dp (eps)", user_eps))
        if self.is_sample_level_dp:
            # calculate sample level dp privacy loss statistics.
            all_client_eps = torch.Tensor(
                [c.privacy_budget.epsilon for c in clients]  # pyre-fixme
            )
            mean_client_eps = all_client_eps.mean()
            max_client_eps = all_client_eps.max()
            min_client_eps = all_client_eps.min()
            p50_client_eps = torch.median(all_client_eps)
            sample_dp_metrics: List[Metric] = Metric.from_args(
                mean=mean_client_eps,
                min=min_client_eps,
                max=max_client_eps,
                median=p50_client_eps,
            )
            metrics.append(Metric("sample level dp (eps)", sample_dp_metrics))

        return metrics

    def _calc_overflow_metrics(
        self,
        clients: Iterable[Client],
        model: IFLModel,
        users_per_round: int,
        metric_reporter: Optional[IFLMetricsReporter],
    ) -> List[Metric]:
        """Calculates overflow metrics."""
        metrics = []
        if self.is_secure_aggregation_enabled:
            for client in clients:
                client.eval(model=model, metric_reporter=metric_reporter)
            (
                convert_overflow_perc,
                aggregate_overflow_perc,
            ) = self.server.calc_avg_overflow_percentage(  # pyre-fixme
                users_per_round, model
            )
            overflow_metrics: List[Metric] = Metric.from_args(
                convert_overflow_percentage=convert_overflow_perc,
                aggregate_overflow_percentage=aggregate_overflow_perc,
            )
            metrics.append(Metric("overflow per round", overflow_metrics))

        return metrics

    def _calc_post_epoch_client_metrics(
        self,
        client_models: Dict[Client, IFLModel],
        round_timeline: Timeline,
        metric_reporter: IFLMetricsReporter,
    ) -> List[List[Metric]]:
        """Calculates client-side metrics on the overall evaluation set."""
        client_metrics = []
        if metric_reporter is not None:
            for client, model in tqdm(client_models.items()):
                metric_reporter.reset()
                client.eval(
                    model=model,
                    metric_reporter=metric_reporter,
                )
                # pyre-fixme[16]: `IFLMetricsReporter` has no attribute
                #  `compute_scores`.
                score = metric_reporter.compute_scores()
                client_metrics.append(Metric.from_dict(score))

        return client_metrics

    def _evaluate_global_model_after_aggregation_on_train_clients(
        self,
        clients: Iterable[Client],
        model: IFLModel,
        timeline: Timeline,
        users_per_round: int,
        metric_reporter: Optional[IFLMetricsReporter] = None,
    ):
        if (
            metric_reporter is not None
            # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
            and self.cfg.report_train_metrics
            and self.cfg.report_train_metrics_after_aggregation
            and timeline.tick(1.0 / self.cfg.train_metrics_reported_per_epoch)
        ):
            with torch.no_grad():
                model.fl_get_module().eval()
                for client in clients:
                    for batch in client.dataset.train_data():
                        batch_metrics = model.get_eval_metrics(batch)
                        if metric_reporter is not None:
                            metric_reporter.add_batch_metrics(batch_metrics)
                model.fl_get_module().train()

            privacy_metrics = self._calc_privacy_metrics(
                clients, model, metric_reporter
            )
            overflow_metrics = self._calc_overflow_metrics(
                clients, model, users_per_round, metric_reporter
            )

            metric_reporter.report_metrics(
                model=model,
                reset=True,
                stage=TrainingStage.AGGREGATION,
                timeline=timeline,
                epoch=timeline.global_round_num(),  # for legacy
                print_to_channels=True,
                extra_metrics=privacy_metrics + overflow_metrics,
            )

    def _validate_users_per_round(
        self, users_per_round_on_worker: int, num_users_on_worker: int
    ):
        assert users_per_round_on_worker <= num_users_on_worker, (
            "Users per round is greater than the number of users in the data provider for the worker."
            "If you are using paged dataloader, increase your num_users_per_page >> users_per_round"
        )

    def _report_post_epoch_client_metrics(
        self,
        timeline: Timeline,
        metric_reporter: Optional[IFLMetricsReporter],
    ):
        if (
            metric_reporter is not None
            # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
            and self.cfg.report_client_metrics
            and self.cfg.report_client_metrics_after_epoch
            and (timeline.epoch % self.cfg.client_metrics_reported_per_epoch == 0)
        ):
            client_models = {
                client: client.last_updated_model for client in self.clients.values()
            }
            client_scores = self._calc_post_epoch_client_metrics(
                client_models, timeline, metric_reporter
            )

            # Find stats over the client_metrics (mean, min, max, median, std)
            client_stats_trackers = {}
            score_names = [metric.name for metric in next(iter(client_scores))]
            for score_name in score_names:
                client_stats_trackers[score_name] = RandomVariableStatsTracker(
                    tracks_quantiles=True
                )
            for client_metric_list in client_scores:
                for client_metric in client_metric_list:
                    client_stats_trackers[client_metric.name].update(
                        client_metric.value
                    )

            reportable_client_metrics = []
            for score_name in score_names:
                for stat_name, stat_key in [
                    ("Mean", "mean_val"),
                    ("Median", "median_val"),
                    ("Upper Quartile", "upper_quartile_val"),
                    ("Lower Quartile", "lower_quartile_val"),
                    ("Min", "min_val"),
                    ("Max", "max_val"),
                    ("Standard Deviation", "standard_deviation_val"),
                    ("Num Samples", "num_samples"),
                ]:
                    score = client_stats_trackers[score_name].__getattribute__(stat_key)
                    reportable_client_metrics.append(Metric(stat_name, score))

            metric_reporter.report_metrics(
                model=None,
                reset=True,
                stage=TrainingStage.PER_CLIENT_EVAL,
                timeline=timeline,
                epoch=timeline.global_round_num(),  # for legacy
                print_to_channels=True,
                extra_metrics=reportable_client_metrics,
            )

    @staticmethod
    def rounds_in_one_epoch(num_total_users: int, users_per_round: int) -> int:
        return math.ceil(num_total_users / users_per_round)


def force_print(is_distributed: bool, *args, **kwargs) -> None:
    if is_distributed:
        try:
            device_info = f" [device:{torch.cuda.current_device()}]"
            # pyre-fixme[28]: Unexpected keyword argument `force`.
            print(*args, device_info, **kwargs, force=True)
        except TypeError:
            pass
    else:
        print(*args, **kwargs)


@dataclass
class SyncTrainerConfig(FLTrainerConfig):
    _target_: str = fullclassname(SyncTrainer)
    server: SyncServerConfig = SyncServerConfig()
    users_per_round: int = 10
    # overselect users_per_round / dropout_rate users, only use first
    # users_per_round updates
    dropout_rate: float = 1.0
    report_train_metrics_after_aggregation: bool = False
    report_client_metrics_after_epoch: bool = False
    # Whether client metrics on eval data should be computed and reported.
    report_client_metrics: bool = False
    # how many times per epoch should we report client metrics
    # numbers greater than 1 help with plotting more precise training curves
    client_metrics_reported_per_epoch: int = 1
