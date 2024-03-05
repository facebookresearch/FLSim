#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""Utils class to fine tune a global model on client's train set then set the personalized model
on the client's eval set
"""
from typing import Any, Iterable, Tuple

from flsim.clients.base_client import ClientConfig
from flsim.common.timeline import Timeline
from flsim.data.data_provider import IFLUserData
from flsim.interfaces.metrics_reporter import IFLMetricsReporter, TrainingStage
from flsim.interfaces.model import IFLModel
from flsim.utils.cuda import CudaTransferMinimizer
from hydra.utils import instantiate
from tqdm import tqdm


class FineTuner:
    @classmethod
    def fine_tune_and_evaluate(
        cls,
        data: Iterable[IFLUserData],
        global_model: IFLModel,
        client_config: ClientConfig,
        metrics_reporter: IFLMetricsReporter,
        cuda_state_manager: CudaTransferMinimizer,
        training_stage: TrainingStage,
        timeline: Timeline,
        epochs: int,
    ) -> Tuple[Any, bool]:
        for user_data in tqdm(
            data,
            desc="Fine-tune clients",
            unit="client",
        ):
            FineTuner.fine_tune_model(
                global_model=global_model,
                data=user_data,
                client_config=client_config,
                metrics_reporter=metrics_reporter,
                cuda_state_manager=cuda_state_manager,
                epochs=epochs,
            )

        return metrics_reporter.report_metrics(
            model=global_model,
            reset=True,
            stage=training_stage,
            timeline=timeline,
            epoch=timeline.global_round_num(),  # for legacy
            print_to_channels=True,
        )

    @classmethod
    def fine_tune_model(
        cls,
        global_model: IFLModel,
        data: IFLUserData,
        client_config: ClientConfig,
        metrics_reporter: IFLMetricsReporter,
        cuda_state_manager: CudaTransferMinimizer,
        epochs: int,
    ) -> IFLModel:
        eval_client = instantiate(
            client_config,
            _partial_=False,
            dataset=data,
            cuda_manager=cuda_state_manager,
        )
        fine_tuned_model, _, _ = eval_client.copy_and_train_model(
            model=global_model, epochs=epochs
        )
        eval_client.eval(model=fine_tuned_model, metrics_reporter=metrics_reporter)
        return fine_tuned_model
