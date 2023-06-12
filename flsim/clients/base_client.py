#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file defines the concept of a base client for a
federated learning setting. Also defines basic config,
for an FL client.

Note:
    This is just a base class and needs to be overridden
    for different use cases.
"""
from __future__ import annotations

import numpy as np
import logging
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
from flsim.channels.base_channel import IdentityChannel
from flsim.channels.message import Message
from flsim.common.logger import Logger
from flsim.common.timeout_simulator import (
    NeverTimeOutSimulator,
    NeverTimeOutSimulatorConfig,
    TimeOutSimulator,
)
from flsim.data.data_provider import IFLUserData
from flsim.interfaces.metrics_reporter import IFLMetricsReporter
from flsim.interfaces.model import IFLModel
from flsim.optimizers.local_optimizers import (
    LocalOptimizerConfig,
    LocalOptimizerSGDConfig,
)
from flsim.optimizers.optimizer_scheduler import (
    ConstantLRSchedulerConfig,
    OptimizerScheduler,
    OptimizerSchedulerConfig,
)
from flsim.utils.config_utils import fullclassname, init_self_cfg
from flsim.utils.cuda import DEFAULT_CUDA_MANAGER, ICudaStateManager
from flsim.utils.fl.common import FLModelParamUtils
from hydra.utils import instantiate
from omegaconf import OmegaConf
from functorch import make_functional_with_buffers
from captum.attr import ShapleyValueSampling
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

class Client:
    logger = Logger.get_logger(__name__)

    def __init__(
        self,
        *,
        dataset: IFLUserData,
        channel: Optional[IdentityChannel] = None,
        timeout_simulator: Optional[TimeOutSimulator] = None,
        store_last_updated_model: Optional[bool] = False,
        cuda_manager: ICudaStateManager = DEFAULT_CUDA_MANAGER,
        name: Optional[str] = None,
        **kwargs,
    ):
        init_self_cfg(
            self,
            component_class=__class__,  # pyre-fixme[10]: Name `__class__` is used but not defined.
            config_class=ClientConfig,
            **kwargs,
        )

        self.dataset = dataset
        self.cuda_state_manager = cuda_manager
        self.channel = channel or IdentityChannel()
        self.timeout_simulator = timeout_simulator or NeverTimeOutSimulator(
            **OmegaConf.structured(NeverTimeOutSimulatorConfig())
        )
        self.store_last_updated_model = store_last_updated_model
        self._name = name or "unnamed_client"

        # base lr needs to match LR in optimizer config, overwrite it
        # pyre-ignore [16]
        self.cfg.lr_scheduler.base_lr = self.cfg.optimizer.lr
        self.per_example_training_time = (
            self.timeout_simulator.simulate_per_example_training_time()
        )
        self.ref_model = None
        self.num_samples = 0
        self.times_selected = 0
        # Tracks client state history
        # Key: i-th time the client is selected for training
        # Value: client state (i.e. model delta, client weight, optimizer used)
        self._tracked = {}
        self.last_updated_model = None
        self.logger.setLevel(logging.INFO)

    @classmethod
    def _set_defaults_in_cfg(cls, cfg):
        if OmegaConf.is_missing(cfg.optimizer, "_target_"):
            cfg.optimizer = LocalOptimizerSGDConfig()
        if OmegaConf.is_missing(cfg.lr_scheduler, "_target_"):
            cfg.lr_scheduler = ConstantLRSchedulerConfig()

    @property
    def seed(self) -> Optional[int]:
        """if should set random_seed or not."""
        # pyre-fixme[16]: `Client` has no attribute `cfg`.
        return self.cfg.random_seed

    @property
    def model_deltas(self) -> List[IFLModel]:
        """Return the stored deltas for all rounds that this user was selected."""
        return [self._tracked[s]["delta"] for s in range(self.times_selected)]

    @property
    def optimizers(self) -> List[Any]:
        """Look at {self.model}"""
        return [self._tracked[s]["optimizer"] for s in range(self.times_selected)]

    @property
    def weights(self) -> List[float]:
        """Look at {self.model}"""
        return [self._tracked[s]["weight"] for s in range(self.times_selected)]

    @property
    def name(self) -> str:
        return self._name

    def generate_local_update(
        self, message: Message, metrics_reporter: Optional[IFLMetricsReporter] = None
    ) -> Tuple[IFLModel, float]:
        """Wrapper around all functions called on a client for generating an updated
        local model.

        Args:
            message: Message object. Must include the global model with optional metadata.

        NOTE:
            Only pass a `metrics_reporter` if reporting is needed, i.e. report_metrics
            will be called on the reporter; else reports will be accumulated in memory.
        """
        model = message.model

        updated_model, weight, optimizer = self.copy_and_train_model(
            model, metrics_reporter=metrics_reporter
        )
        # 4. Store updated model if being tracked
        if self.store_last_updated_model:
            self.last_updated_model = FLModelParamUtils.clone(updated_model)

        # 5. Compute model delta
        delta = self.compute_delta(
            before=model, after=updated_model, model_to_save=updated_model
        )

        # 6. Track the state of the client
        self.track(delta=delta, weight=weight, optimizer=optimizer)

        return delta, weight

    def copy_and_train_model(
        self,
        model: IFLModel,
        epochs: Optional[int] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizer_scheduler: Optional[OptimizerScheduler] = None,
        metrics_reporter: Optional[IFLMetricsReporter] = None,
    ) -> Tuple[IFLModel, float, torch.optim.Optimizer]:
        """Copy the source model to client-side model and use it to train on the
        client's train split.

        NOTE: Optional optimizer and optimizer_scheduler are there for easier testing

        Returns:
            (trained model, client's weight, optimizer used)
        """
        # 1. Pass source model through channel and use it to set client-side model state
        updated_model = self.receive_through_channel(model)
        # 2. Set up model and default optimizer in the client
        updated_model, default_optim, default_scheduler = self.prepare_for_training(
            updated_model
        )
        optim = default_optim if optimizer is None else optimizer
        optim_scheduler = (
            default_scheduler if optimizer_scheduler is None else optimizer_scheduler
        )

        # 3. Kick off training on client
        updated_model, weight = self.train(
            updated_model,
            optim,
            optim_scheduler,
            metrics_reporter=metrics_reporter,
            epochs=epochs,
        )
        return updated_model, weight, optim

    def compute_delta(
        self, before: IFLModel, after: IFLModel, model_to_save: IFLModel
    ) -> IFLModel:
        """Computes the delta of the model before and after training."""
        FLModelParamUtils.subtract_model(
            minuend=before.fl_get_module(),
            subtrahend=after.fl_get_module(),
            difference=model_to_save.fl_get_module(),
        )
        return model_to_save

    def receive_through_channel(self, model: IFLModel) -> IFLModel:
        """Receives a reference to a state (referred to as model state_dict) over the
        channel. Any channel effect is applied as part of this function.
        """
        # Keep a reference to global model
        self.ref_model = model

        # Need to clone the model because it's a reference to the global model; else
        # modifying model will modify the global model.
        message = self.channel.server_to_client(
            Message(model=FLModelParamUtils.clone(model))
        )

        return message.model

    def prepare_for_training(
        self, model: IFLModel
    ) -> Tuple[IFLModel, torch.optim.Optimizer, OptimizerScheduler]:
        """Prepare for training by:
        1. Instantiating a model on a correct compute device
        2. Creating an optimizer
        """
        # Inform cuda_state_manager that we're about to train a model; it may involve
        # moving the model to GPU.
        self.cuda_state_manager.before_train_or_eval(model)
        # Put model in train mode
        model.fl_get_module().train()
        # Create optimizer
        # pyre-fixme[16]: `Client` has no attribute `cfg`.
        optimizer = instantiate(self.cfg.optimizer, model=model.fl_get_module())
        optimizer_scheduler = instantiate(self.cfg.lr_scheduler, optimizer=optimizer)
        return model, optimizer, optimizer_scheduler

    def get_total_training_time(self) -> float:
        return self.timeout_simulator.simulate_training_time(
            self.per_example_training_time, self.dataset.num_train_examples()
        )

    def stop_training(self, num_examples_processed) -> bool:
        training_time = self.timeout_simulator.simulate_training_time(
            self.per_example_training_time, num_examples_processed
        )
        return self.timeout_simulator.user_timeout(training_time)

    def train(
        self,
        model: IFLModel,
        optimizer: Any,
        optimizer_scheduler: OptimizerScheduler,
        metrics_reporter: Optional[IFLMetricsReporter] = None,
        epochs: Optional[int] = None,
    ) -> Tuple[IFLModel, float]:
        """Trains the client-side model for a number of epochs.
        Returns:
            (trained model, weight for this client)
            Currently, weight for this client is the number of samples it trained on.
            This might be a bad strategy because it favors users with more data, and
            there might be privacy implications.
        """
        # Total number of training samples
        total_samples = 0

        # Total number of sampled processed during training
        # In general, this equals total_samples * epochs
        num_examples_processed = 0

        # pyre-ignore[16]: `Client` has no attribute `cfg`.
        epochs = epochs if epochs is not None else self.cfg.epochs
        if self.seed is not None:
            torch.manual_seed(self.seed)

        assert self.dataset.num_train_batches() > 0, "Client has no training data"

        for epoch in range(epochs):
            if self.stop_training(num_examples_processed):
                break

            dataset = list(self.dataset.train_data())
            # If the user has too many examples and times-out, we want to process a
            # different portion of the dataset each time, hence shuffle batch order.
            if self.cfg.shuffle_batch_order:
                random.shuffle(dataset)
            for batch in dataset:
                sample_count = self._batch_train(
                    model=model,
                    optimizer=optimizer,
                    training_batch=batch,
                    epoch=epoch,
                    metrics_reporter=metrics_reporter,
                    optimizer_scheduler=optimizer_scheduler,
                )
                # Optional post-processing after training on a batch
                self.post_batch_train(epoch, model, sample_count, optimizer)
                # Only add in epoch 0 because we want to get the training set size
                total_samples += 0 if epoch else sample_count
                num_examples_processed += sample_count
                # Stop training depending on time-out condition
                if self.stop_training(num_examples_processed):
                    break
                # dimasquest: The vog stuff could potentially go in here?
        if model.use_bw_metrics:
            plis_scores = {key: model.calculate_plis(pl).item() for key, pl in metrics_reporter.plis_dict.items()}
            vogs = {key: model.calculate_vog(grads).item() for key, grads in metrics_reporter.grads_dict.items()}
            desc_vog_idxs, asc_vog_idxs, vogs_per_class = model.process(vogs)
            desc_plis_idxs, asc_plis_idxs, plis_per_class = model.process(plis_scores)
            desc_norm_idxs, asc_norm_idxs, norms_per_class = model.process(metrics_reporter.norm_dict, normalise=False)
            metrics_reporter.add_idxs(desc_vog_idxs, 'vogs_desc')
            metrics_reporter.add_idxs(desc_plis_idxs, 'plis_desc')
            metrics_reporter.add_idxs(desc_norm_idxs, 'norms_desc')
            metrics_reporter.add_idxs(asc_vog_idxs, 'vogs_asc')
            metrics_reporter.add_idxs(asc_plis_idxs, 'plis_asc')
            metrics_reporter.add_idxs(asc_norm_idxs, 'norms_asc')
        
        # generate the cross-epoch metrics
        desc_loss_idxs, asc_loss_idxs, losses_per_class = model.process(metrics_reporter.loss_dict, normalise=False)
        desc_entropy_idxs, asc_entropy_idxs, entropies_per_class = model.process(metrics_reporter.entropy_dict, normalise=False)
        
        # add these to the resulting metrics_reporter
        metrics_reporter.add_idxs(desc_loss_idxs, 'losses_desc')
        metrics_reporter.add_idxs(desc_entropy_idxs, 'entropies_desc')
        
        metrics_reporter.add_idxs(asc_loss_idxs, 'losses_asc')
        metrics_reporter.add_idxs(asc_entropy_idxs, 'entropies_asc')
        if model.use_bw_metrics:
            metrics_reporter.add_full_scores(vogs_per_class, plis_per_class, norms_per_class, losses_per_class, entropies_per_class)
        else: 
            metrics_reporter.add_forward_scores(losses_per_class, entropies_per_class)
            
        # this Shapley implementation uses the entire dataset, need to change
        if model.shap:
            svs = ShapleyValueSampling(model.model)
            current_train_dataset = list(self.dataset.train_data())
            def split_dataset(dataset):
                idxs, data, labels = [], [], []
                for batch in dataset:
                    ids, data_points, targets = batch['idx'], batch['data'], batch['target']
                    for id, datapoint, label in zip(ids, data_points, targets):
                        idxs.append(id)
                        data.append(datapoint)
                        labels.append(label)
                return idxs, data, labels
            shap_ids, shap_images, shap_labels = split_dataset(current_train_dataset)
            flattened_images = np.array([np.array(image.flatten()) for image in shap_images])
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(flattened_images)
            dbscan = DBSCAN(eps=3, min_samples=5)
            labels = dbscan.fit_predict(X_pca)
            
            # these are sorted
            unique_labels = np.unique(labels[labels != -1])
            centroids = []
            for label in unique_labels:
                cluster_points = flattened_images[labels == label]
                centroid = np.mean(cluster_points, axis=0)
                centroids.append(centroid)
            # once the centroids are estanblished:
            # 1. run shapley on these, get the rest as a background
            # 2. discard the poorest performers (rn the shape is static)
            centroid_tensors = torch.stack([torch.from_numpy(item).reshape((3,32,32)) for item in centroids])
            attr = svs.attribute(centroid_tensors, show_progress=True)
            l2_norms_shap = [shap.norm(2) for shap in attr]
            metrics_reporter.shap_norms = dict(zip(shap_ids, l2_norms_shap))
            desc_shap, asc_shap, shap_per_class = process_metric(scores.shap_norms, idxs_to_labels, cfg.dataset.num_classes, normalise=False)
            metrics_reporter.add_idxs(desc_shap, "shap_desc")
            metrics_reporter.add_idxs(asc_shap, "shap_asc")
            metrics_reporter.add_score_dict(shap_per_class, "shap")

        # Tell cuda manager we're done with training
        # cuda manager may move model out of GPU memory if needed
        self.cuda_state_manager.after_train_or_eval(model)
        self.logger.debug(
            f"Processed {num_examples_processed} of {self.dataset.num_train_examples()}"
        )
        # dimasquest: Maybe need to throw the processing into here to de-clutter?
        # can even modify the dataset here, because this method is not used elsewhere
        # total samples changes after this, but for the purposes of training this round
        # it should be ok?
        # Optional post-processing after the entire training is done
        self.post_train(model, total_samples, optimizer, metrics_reporter)

        # If training stops early (i.e. `num_examples_processed < total_samples`), use
        # `num_examples_processed` instead as weight.
        example_weight = min([num_examples_processed, total_samples])

        return model, float(example_weight)

    def post_train(self, model: IFLModel, total_samples: int, optimizer: Any, metrics_reporter: Optional[IFLMetricsReporter]):
        """Post-processing after the entire training on multiple epochs is done."""
        current_train_dataset = list(self.dataset.train_data())
        batch_size = len(current_train_dataset[0]['idx'])
        # select the points to remove
        num_points_to_remove = int(total_samples * model.proportion_to_remove)
        idxs = metrics_reporter.idx_dict[model.removal_metric]
        
        top_values_to_be_removed_per_class = {}
        for key, value in idxs.items():
            top_values_to_be_removed_per_class[key] = value[:num_points_to_remove]
        idxs_to_be_removed = []
        for value in top_values_to_be_removed_per_class.values():
            idxs_to_be_removed.extend(value)
            
        # remove the data points
        def remove_datapoints(dataset, idxs_to_remove, batch_size):
            updated_dataset = []
            for batch in dataset:
                idxs, data_points, labels = batch['idx'], batch['data'], batch['target']
                for idx, data, label in zip(idxs, data_points, labels):
                    if idx.item() not in idxs_to_remove:
                        updated_dataset.append({'idx': idx, 'data': data, 'target': label})
            batched_list = [updated_dataset[i:i+batch_size] for i in range(0, len(updated_dataset), batch_size)]
            final_batches = []
            for row in batched_list:
                new_row = {'idx': [], 'data': [], 'target': []}
                for record in row:
                    new_row['idx'].append(record['idx'])
                    new_row['data'].append(record['data'])
                    new_row['target'].append(record['target'])
                final_batches.append(new_row)
            return final_batches
        train_dataset = remove_datapoints(current_train_dataset, idxs_to_be_removed, batch_size)
        # overwrite the dataset (as currently it is a generator)
        # TODO: This is where we need to take care of all dataset size changes
        self.dataset.modify_dataset_after_training(train_dataset)

    def post_batch_train(
        self, epoch: int, model: IFLModel, sample_count: int, optimizer: Any
    ):
        """Post-processing after training on one batch is done."""
        pass

    def track(self, delta: IFLModel, weight: float, optimizer: Any):
        """Tracks the client state each time the client is selected.
        Modifies `self_tracked`, where keys are the i-th time this client is selected
        and values are the state of the client.
        """
        # pyre-fixme[16]: `Client` has no attribute `cfg`.
        if self.cfg.store_models_and_optimizers:
            self._tracked[self.times_selected] = {
                "delta": FLModelParamUtils.clone(delta),
                "weight": weight,
                "optimizer": optimizer,
            }
        self.times_selected += 1

    def eval(
        self,
        model: IFLModel,
        dataset: Optional[IFLUserData] = None,
        metrics_reporter: Optional[IFLMetricsReporter] = None,
    ):
        """Evaluates client-side model with its evaluation data split."""
        data = dataset or self.dataset
        self.cuda_state_manager.before_train_or_eval(model)
        with torch.no_grad():
            if self.seed is not None:
                torch.manual_seed(self.seed)

            model.fl_get_module().eval()
            for batch in data.eval_data():
                batch_metrics = model.get_eval_metrics(batch)
                if metrics_reporter is not None:
                    metrics_reporter.add_batch_metrics(batch_metrics)
        model.fl_get_module().train()
        self.cuda_state_manager.after_train_or_eval(model)

# dimasquest: I am modifying this part of training to try to accomodate for vogs etc.
    def _batch_train(
        self,
        model,
        optimizer,
        training_batch,
        epoch,
        metrics_reporter,
        optimizer_scheduler,
    ) -> int:
        """Trains the client-side model for one batch.
        Compatible with the new tasks in which the model is responsible for
        arranging its inputs, targets and context.
        Returns:
            Number of samples in the batch.
        """
        optimizer.zero_grad()
        batch_metrics = model.fl_forward(training_batch)
        loss = batch_metrics.loss

        loss.backward()
        if model.use_bw_metrics:
            images, target = torch.stack(training_batch['data']), torch.tensor(training_batch['target'])
            with torch.no_grad():
                plis_scores, grad_norms = model.plis_func(images, target, model.fmodel, model.params, model.buffers, model.criterion)
                plis_scores, grad_norms = plis_scores.cpu(), grad_norms.cpu()
            # pyre-fixme[16]: `Client` has no attribute `cfg`.
            # Optional gradient clipping
        if self.cfg.max_clip_norm_normalized is not None:
            max_norm = self.cfg.max_clip_norm_normalized
            FLModelParamUtils.clip_gradients(
                max_normalized_l2_norm=max_norm, model=model.fl_get_module()
            )

        num_examples = batch_metrics.num_examples
        # Adjust lr and take a gradient step
        optimizer_scheduler.step(batch_metrics, model, training_batch, epoch)
        optimizer.step()
        
        if model.use_bw_metrics:
            _, model.params, model.buffers = make_functional_with_buffers(model.fl_get_module())
            with torch.no_grad():
                input_grads = model.input_grad_func(images, target, model.fmodel, model.params, model.buffers, model.criterion).cpu()
            for id, gr, pls, n, loss, e in zip(training_batch['idx'], input_grads, plis_scores,
                                                            grad_norms, batch_metrics.losses, batch_metrics.entropies):
                batch_metrics.add_full_training_metrics(id, gr, pls, n, loss.detach().cpu(), e)
        else:
            for id, loss, e in zip(training_batch['idx'], batch_metrics.losses, batch_metrics.entropies):
                batch_metrics.add_forward_training_metrics(id, loss.detach().cpu(), e)
            

        # dimasquest: We need to A) collate the metrics from batches in metrics reporter
        # and B) make them persistent across epochs (for VoGs for instance)
        if metrics_reporter is not None:
            metrics_reporter.add_batch_metrics(batch_metrics)

        return num_examples

    def full_dataset_gradient(self, module: IFLModel):
        """Performs a pass over the entire training set and returns the average gradient.
        Currently only called from SyncMimeServer.

        Args:
            module (IFLModel): Model over which to return gradients

        Returns:
            grads (nn.Module): Dummy module with the desired gradients contained in its
                parameters (access via .grad).
            num_examples (int): Number of training examples in the dataset.
        """
        # `grads` is a dummy module for storing the average gradient.
        # Its network weights are never used.
        module.fl_get_module().train()
        grads = FLModelParamUtils.clone(module.fl_get_module())
        grads.zero_grad()
        num_examples = 0

        for batch in self.dataset.train_data():
            module.fl_get_module().zero_grad()
            batch_metrics = module.fl_forward(batch)
            batch_metrics.loss.backward()
            FLModelParamUtils.multiply_gradient_by_weight(
                module.fl_get_module(),
                batch_metrics.num_examples,
                module.fl_get_module(),
            )
            FLModelParamUtils.add_gradients(grads, module.fl_get_module(), grads)
            num_examples += batch_metrics.num_examples
        assert num_examples > 0, "Client has no training data"
        # Average out the gradient over the client
        FLModelParamUtils.multiply_gradient_by_weight(
            model=grads, weight=1.0 / num_examples, model_to_save=grads
        )
        return grads, num_examples


@dataclass
class ClientConfig:
    _target_: str = fullclassname(Client)
    _recursive_: bool = False
    epochs: int = 1  # Number of epochs for local training
    optimizer: LocalOptimizerConfig = LocalOptimizerConfig()
    lr_scheduler: OptimizerSchedulerConfig = OptimizerSchedulerConfig()
    max_clip_norm_normalized: Optional[float] = None  # gradient clip value
    only_federated_params: bool = True  # flag to only use certain params
    random_seed: Optional[int] = None  # random seed for deterministic response
    shuffle_batch_order: bool = False  # shuffle the ordering of batches
    store_models_and_optimizers: bool = False  # name clear
    track_multiple_selection: bool = False  # track if the client appears in 2+ rounds.
