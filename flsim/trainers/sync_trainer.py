#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

import logging
import math
import random
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
from flsim.servers.sync_dp_servers import SyncDPSGDServer
from flsim.servers.sync_secagg_servers import SyncSecAggServer
from flsim.servers.sync_servers import FedAvgOptimizerConfig, SyncServerConfig
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
            # pyre-fixme[10]: Name `__class__` is used but not defined.
            component_class=__class__,
            config_class=SyncTrainerConfig,
            **kwargs,
        )

        super().__init__(model=model, cuda_enabled=cuda_enabled, **kwargs)
        self.server = instantiate(
            # pyre-ignore[16]
            self.cfg.server,
            global_model=model,
            channel=self.channel,
        )
        # Dictionary that maps a dataset ID to the associated client object:
        # Key: dataset_id
        # Value: client object
        self.clients = {}
        self._last_report_round_after_aggregation = 0

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        """Set default configs if missing.
        In addition to default configs set by base class, set default config for server.
        """
        if OmegaConf.is_missing(cfg.server, "_target_"):
            cfg.server = SyncServerConfig(optimizer=FedAvgOptimizerConfig())

    def global_model(self) -> IFLModel:
        """Returns global model.
        NOTE: self.global_model() is owned by the server, not by SyncTrainer.
        """
        return self.server.global_model

    def client_models(self) -> Dict[Client, IFLModel]:
        """Returns the list of latest client-side models."""
        client_models = {
            client: client.last_updated_model for client in self.clients.values()
        }
        return client_models

    @property
    def is_user_level_dp(self):
        """Whether the server is differentially private wrt each user."""
        return isinstance(self.server, SyncDPSGDServer)

    @property
    def is_sample_level_dp(self):
        """Whether the client is differentially private wrt each sample."""
        return is_target(self.cfg.client, DPClientConfig)

    @property
    def is_secure_aggregation_enabled(self):
        """Whether secure aggregation is used."""
        return isinstance(self.server, SyncSecAggServer)

    def create_or_get_client_for_data(self, dataset_id: int, datasets: Any):
        """Creates one training client for given dataset ID.
        This function is called <code>users_per_round * num_rounds</code> times per
        training epoch. Here, we use <code>OmegaConf.structured</code> instead of
        <code>hydra.instantiate</code> to minimize the overhead of hydra object creation.

        Args:
            dataset_id: Dataset ID that will be the client's dataset. For each client,
                we assign it a unique dataset ID. In practice, dataset_id is the same as
                client index.
            datasets: Data provider object to output training clients.
        Returns:
            Client object associated with `dataset_id`. In addition, also modify
                `self.clients` dictionary by adding a key-value pair of
                (dataset ID, client object).
        """
        if self.is_sample_level_dp:
            # Differentially private client (sample-level)
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
            client = instantiate(
                self.cfg.client,
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
        metrics_reporter: IFLMetricsReporter,
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
            metrics_reporter: computes and reports metrics of interest such as accuracy
                or perplexity
            num_total_users: number of total users for training
            distributed_world_size: world size for distributed training
            rank: worker index for distributed training

        Returns:
            model, best_metric: the trained model together with the best metric

        Note:
            Depending on the chosen active user selector, we may not iterate over
            all users in a given epoch.
        """
        # Set up synchronization utilities for distributed training
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
        users_per_round = min(self.cfg.users_per_round, num_total_users)

        self.data_provider = data_provider
        num_rounds_in_epoch = self.rounds_in_one_epoch(num_total_users, users_per_round)
        num_users_on_worker = data_provider.num_train_users()
        self.logger.debug(
            f"num_users_on_worker: {num_users_on_worker}, "
            f"users_per_round: {users_per_round}, "
            f"num_total_users: {num_total_users}"
        )
        # torch.multinomial requires int instead of float; cast it as int
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

        # Main training loop
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
                #### Initial setup ####
                # Initialize point of time for logging
                timeline = Timeline(
                    epoch=epoch,
                    round=round,
                    rounds_per_epoch=num_rounds_in_epoch,
                    total_epochs=self.cfg.epochs,
                )

                # Select clients for training this round
                t = time()
                clients = self._client_selection(
                    num_users=num_users_on_worker,
                    users_per_round=users_per_round_on_worker,
                    data_provider=data_provider,
                    timeline=timeline,
                )
                self.logger.info(f"Client Selection took: {time() - t} s.")

                # Select clients for calculating post-aggregation *training* metrics
                agg_metric_clients = self._choose_clients_for_post_aggregation_metrics(
                    train_clients=clients,
                    num_total_users=num_users_on_worker,
                    users_per_round=users_per_round_on_worker,
                )

                #### Training phase ####
                # Training on selected clients for this round; also calculate training
                # metrics on `agg_metric_clients`
                self.logger.info(f"# clients/round on worker {rank}: {len(clients)}.")
                self._train_one_round(
                    timeline=timeline,
                    clients=clients,
                    agg_metric_clients=agg_metric_clients,
                    users_per_round=users_per_round,
                    metrics_reporter=(
                        metrics_reporter if self.cfg.report_train_metrics else None
                    ),
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

                #### Evaluation phase ####
                if rank == 0:
                    # Report training time
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

                    # Report evaluation metric on evaluation clients
                    t = time()
                    (
                        best_metric,
                        best_model_state,
                    ) = self._maybe_run_evaluation(
                        timeline=timeline,
                        data_provider=data_provider,
                        metrics_reporter=metrics_reporter,
                        best_metric=best_metric,
                        best_model_state=best_model_state,
                    )
                    self.logger.info(f"Evaluation took {time() - t} s.")

                if self.stop_fl_training(
                    epoch=epoch, round=round, num_rounds_in_epoch=num_rounds_in_epoch
                ):
                    break

            # pyre-fixme[61]: `timeline` may not be initialized here.
            # Report evaluation metrics for client-side models
            self._report_post_epoch_client_metrics(timeline, metrics_reporter)

            if self.stop_fl_training(
                epoch=epoch,
                # pyre-fixme[61]: `round` may not be initialized here.
                round=round,
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
            # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
            >= self.cfg.epochs
            or self._timeout_simulator.stop_fl()
        )

    def _drop_overselected_users(
        self, clents_triggered: List[Client], num_users_keep: int
    ) -> List[Client]:
        """Keeps top `num_users_keep` users with least training times."""
        all_training_times = [c.get_total_training_time() for c in clents_triggered]
        all_training_times.sort()
        # only select first num_users_keep userids sorted by their finish time
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
        timeline: Timeline,
    ) -> List[Client]:
        """Select client for training each round."""
        # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
        num_users_overselected = math.ceil(users_per_round / self.cfg.dropout_rate)
        # pyre-fixme[16]: `SyncTrainer` has no attribute `_user_indices_overselected`.
        self._user_indices_overselected = self.server.select_clients_for_training(
            num_total_users=num_users,
            users_per_round=num_users_overselected,
            data_provider=data_provider,
            global_round_num=timeline.global_round_num(),
        )
        clients_to_train = [
            self.create_or_get_client_for_data(i, self.data_provider)
            for i in self._user_indices_overselected
        ]
        if not math.isclose(self.cfg.dropout_rate, 1.0):
            clients_to_train = self._drop_overselected_users(
                clients_to_train, users_per_round
            )
        return clients_to_train

    def _save_model_and_metrics(self, model: IFLModel, best_model_state):
        model.fl_get_module().load_state_dict(best_model_state)

    def _update_clients(
        self,
        clients: Iterable[Client],
        server_state_message: Message,
        metrics_reporter: Optional[IFLMetricsReporter] = None,
    ) -> None:
        """Update each client-side model from server message."""
        for client in clients:
            client_delta, weight = client.generate_local_update(
                message=server_state_message,
                metrics_reporter=metrics_reporter,
            )
            self.server.receive_update_from_client(Message(client_delta, weight))

    def _train_one_round(
        self,
        timeline: Timeline,
        clients: Iterable[Client],
        agg_metric_clients: Iterable[Client],
        users_per_round: int,
        metrics_reporter: Optional[IFLMetricsReporter] = None,
    ) -> None:
        """Trains the global model for one training round.

        Args:
            timeline: Information about the round, epoch, round number, etc.
            clients: Clients for this round.
            agg_metric_clients: Clients for calculating the post-aggregation
                training metrics.
            users_per_round: Number of participating users.
            metrics_reporter: Metric reporter to pass to other methods.
        """
        server_return_metrics = self._train_one_round_apply_updates(
            timeline=timeline,
            clients=clients,
            agg_metric_clients=agg_metric_clients,
            users_per_round=users_per_round,
            metrics_reporter=metrics_reporter,
        )

        self._train_one_round_report_metrics(
            timeline=timeline,
            clients=clients,
            agg_metric_clients=agg_metric_clients,
            users_per_round=users_per_round,
            metrics_reporter=metrics_reporter,
            server_return_metrics=server_return_metrics,
        )

        self._post_train_one_round(timeline)

    def _train_one_round_apply_updates(
        self,
        timeline: Timeline,
        clients: Iterable[Client],
        agg_metric_clients: Iterable[Client],
        users_per_round: int,
        metrics_reporter: Optional[IFLMetricsReporter] = None,
    ) -> Optional[List[Metric]]:
        """Apply updates to client and server models during train one round.
        See `_train_one_round` for argument descriptions.
        Returns: Optional list of `Metric`, same as the return value of `step`
            method in `ISyncServer`.
        """
        t = time()
        self.server.init_round()
        self.logger.info(f"Round initialization took {time() - t} s.")

        # Receive message from server to clients, i.e. global model state
        server_state_message = self.server.broadcast_message_to_clients(
            clients=clients, global_round_num=timeline.global_round_num()
        )

        # Hook before client updates
        self.on_before_client_updates(global_round_num=timeline.global_round_num())

        # Update client-side models from server-side model (in `server_state_message`)
        t = time()
        self._update_clients(
            clients=clients,
            server_state_message=server_state_message,
            metrics_reporter=metrics_reporter,
        )
        self.logger.info(f"Collecting round's clients took {time() - t} s.")

        # After all clients finish their updates, update the global model
        t = time()
        server_return_metrics = self.server.step()
        self.logger.info(f"Finalizing round took {time() - t} s.")
        return server_return_metrics

    def _train_one_round_report_metrics(
        self,
        timeline: Timeline,
        clients: Iterable[Client],
        agg_metric_clients: Iterable[Client],
        users_per_round: int,
        metrics_reporter: Optional[IFLMetricsReporter] = None,
        server_return_metrics: Optional[List[Any]] = None,
    ) -> None:
        """Report metrics during train one round.
        See `_train_one_round` for argument descriptions.
        """
        # Calculate and report metrics for this round
        t = time()
        # Train metrics of global model (e.g. loss and accuracy)
        self._report_train_metrics(
            model=self.global_model(),
            timeline=timeline,
            metrics_reporter=metrics_reporter,
            extra_metrics=server_return_metrics,
        )
        # Evaluation metrics of global model on training data of `agg_metric_clients`
        self._evaluate_global_model_after_aggregation_on_train_clients(
            clients=agg_metric_clients,
            model=self.global_model(),
            timeline=timeline,
            users_per_round=users_per_round,
            metrics_reporter=metrics_reporter,
        )
        # Communication metrics (e.g. amount of data sent between client and server)
        self._calc_post_epoch_communication_metrics(
            timeline,
            metrics_reporter,
        )

        self._post_train_one_round(timeline)

        self.logger.info(f"Aggregate round reporting took {time() - t} s.")

    def _post_train_one_round(self, timeline: Timeline):
        """Optional processing after training for one round is finished."""
        pass

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

    def on_before_client_updates(self, **kwargs):
        global_round_num = kwargs.get("global_round_num", 1)
        # SyncSQServer: SQ channel with `use_shared_qparams` enabled
        if getattr(self.server, "_global_qparams", None) is not None:
            self._init_global_qparams(global_round_num=global_round_num)

        elif getattr(self.server, "_global_mask_params", None) is not None:
            self._init_global_mask_params(global_round_num=global_round_num)

        # SyncPQServer: PQ channel with `use_seed_centroids` enabled
        elif getattr(self.server, "_seed_centroids", None) is not None:
            self._init_global_pq_centroids(global_round_num=global_round_num)

    def _create_mock_client(self):

        # exclude triggered clients for this round
        all_clients_idx = set(range(self.data_provider.num_train_users()))

        # select at random among clients not triggered
        clients_idx_to_exclude = set(self._user_indices_overselected)
        clients_idx_to_select = list(all_clients_idx - clients_idx_to_exclude)
        rand_client_idx = random.choice(clients_idx_to_select)

        # create mock client
        mock_client = self.create_or_get_client_for_data(
            rand_client_idx, self.data_provider
        )
        return mock_client

    def _init_global_qparams(self, global_round_num: int) -> None:
        # TODO make it work for distributed setup
        if not getattr(self.channel, "use_shared_qparams", False):
            return
        if (global_round_num - 1) % self.channel.cfg.qparams_refresh_freq != 0:
            return

        # generate mock client delta
        mock_client = self._create_mock_client()
        mock_message = Message(self.global_model())
        mock_client_delta, mock_client_weight = mock_client.generate_local_update(
            mock_message
        )

        # update server qparams using mock delta
        self.server.update_qparams(mock_client_delta.fl_get_module())

    def _init_global_pq_centroids(self, global_round_num: int) -> None:
        # TODO make it work for distributed setup
        if not self.channel.cfg.use_seed_centroids:
            return
        if (global_round_num - 1) % self.channel.cfg.seed_centroids_refresh_freq != 0:
            return

        # generate mock client delta
        mock_client = self._create_mock_client()
        mock_message = Message(self.global_model())
        mock_client_delta, mock_client_weight = mock_client.generate_local_update(
            mock_message
        )

        # update server qparams using mock delta
        self.server.update_seed_centroids(mock_client_delta.fl_get_module())

    def _init_global_mask_params(self, global_round_num: int) -> None:
        # TODO make it work for distributed setup
        if not getattr(self.channel, "use_shared_masks", False):
            return
        if (global_round_num - 1) % self.channel.cfg.mask_params_refresh_freq != 0:
            return
        # create mock model to generate random mask
        mock_model = FLModelParamUtils.clone(self.global_model()).fl_get_module()
        self.server.update_mask_params(mock_model, "random")

    def _calc_privacy_metrics(
        self,
        clients: Iterable[Client],
        model: IFLModel,
        metrics_reporter: Optional[IFLMetricsReporter],
    ) -> List[Metric]:
        """Calculates privacy metrics if algorithm is differentially private."""
        metrics = []
        if self.is_user_level_dp:
            user_eps = self.server.privacy_budget.epsilon
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
        report_rounds: int,
        metrics_reporter: Optional[IFLMetricsReporter],
    ) -> List[Metric]:
        """Calculates overflow metrics when using secure aggregation."""
        metrics = []
        if self.is_secure_aggregation_enabled:
            for client in clients:
                client.eval(model=model, metrics_reporter=metrics_reporter)
            (
                convert_overflow_perc,
                aggregate_overflow_perc,
            ) = self.server.calc_avg_overflow_percentage(
                users_per_round, model, report_rounds
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
        metrics_reporter: IFLMetricsReporter,
    ) -> List[List[Metric]]:
        """Calculates client-side metrics on each client's evaluation data.
        Returns:
            List of client-side metrics for each client. Each client's metrics are a
                list of `Metric`s.
        """
        client_metrics = []
        if metrics_reporter is not None:
            for client, model in tqdm(client_models.items()):
                metrics_reporter.reset()
                client.eval(
                    model=model,
                    metrics_reporter=metrics_reporter,
                )
                # pyre-fixme[16]: `IFLMetricsReporter` has no attribute
                #  `compute_scores`.
                score = metrics_reporter.compute_scores()
                client_metrics.append(Metric.from_dict(score))

        return client_metrics

    def _evaluate_global_model_after_aggregation_on_train_clients(
        self,
        clients: Iterable[Client],
        model: IFLModel,
        timeline: Timeline,
        users_per_round: int,
        metrics_reporter: Optional[IFLMetricsReporter] = None,
    ):
        """Evaluate global model.
        Args:
            clients: List of clients. We evaluate on the training data of these clients.
            model: Model to evaluate on.
            timeline: Timeline object to keep track of current point of time.
            users_per_round: Number of users. Used for calculating overflow metrics when
                using secure aggregation.
            metrics_reporter: Metric reporter object. If None, do not evaluate.
        """
        if (
            metrics_reporter is not None
            # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
            and self.cfg.report_train_metrics
            and self.cfg.report_train_metrics_after_aggregation
            and timeline.tick(1.0 / self.cfg.train_metrics_reported_per_epoch)
        ):
            current_round = timeline.global_round_num()
            report_rounds = current_round - self._last_report_round_after_aggregation
            self._last_report_round_after_aggregation = current_round

            model.fl_get_module().eval()
            self._calc_eval_metrics_on_clients(
                model=model,
                clients_data=[client.dataset for client in clients],
                data_split="train",
                metrics_reporter=metrics_reporter,
            )
            model.fl_get_module().train()

            privacy_metrics = self._calc_privacy_metrics(
                clients, model, metrics_reporter
            )
            overflow_metrics = self._calc_overflow_metrics(
                clients, model, users_per_round, report_rounds, metrics_reporter
            )

            metrics_reporter.report_metrics(
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
        metrics_reporter: Optional[IFLMetricsReporter],
    ):
        """Report evaluation metrics of client-side models.
        This function is called after each *trainer* epoch.
        """
        if (
            metrics_reporter is not None
            # pyre-fixme[16]: `SyncTrainer` has no attribute `cfg`.
            and self.cfg.report_client_metrics
            and self.cfg.report_client_metrics_after_epoch
            and (timeline.epoch % self.cfg.client_metrics_reported_per_epoch == 0)
        ):
            # Calculate scores for each client-side model on that client's eval data
            client_scores = self._calc_post_epoch_client_metrics(
                self.client_models(), timeline, metrics_reporter
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

            metrics_reporter.report_metrics(
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
