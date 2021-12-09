#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from queue import PriorityQueue
from typing import Optional

from flsim.channels.base_channel import IdentityChannel
from flsim.clients.async_client import AsyncClientDevice, AsyncClientFactory
from flsim.clients.base_client import ClientConfig
from flsim.common.logger import Logger
from flsim.common.timeout_simulator import TimeOutSimulator
from flsim.common.training_event_handler import AsyncTrainingEventHandler
from flsim.utils.async_trainer.async_user_selector import AsyncUserSelector
from flsim.utils.async_trainer.training_event_generator import (
    IEventGenerator,
)
from flsim.utils.cuda import ICudaStateManager, DEFAULT_CUDA_MANAGER
from tqdm import tqdm


class JobQueueStats:
    """Keeps track of #pending jobs in a job queue.
    Computes event based average of jobs
    """

    def __init__(self):
        self.num_finished_jobs: float = 0
        self.pending_jobs_sum: float = 0
        self.num_started_jobs: float = 0

    def on_job_start(self) -> None:
        self.num_started_jobs += 1

    def on_job_end(self) -> None:
        self.num_finished_jobs += 1
        self.pending_jobs_sum += self.num_started_jobs
        self.num_started_jobs -= 1

    def avg_pending_jobs(self) -> float:
        if not self.num_finished_jobs:
            return float("Inf")
        return self.pending_jobs_sum / self.num_finished_jobs

    def as_str(self) -> str:
        return f"Mean:{self.avg_pending_jobs():.3f}"


class AsyncTrainingSimulator:
    """This class simulates asynchronous training of devices where training
    start times are assumed to be drawn from some probability Generator
    (e.g. IEventGenerator). It has a priority queue to keep track
    of event starts and event ends. The queue will have exactly one start
    event at any point (start events are typically poisson distributed).
    Exactly how many end events are there depends on the Generator of
    training time compared to the rate of the poisson process for event
    starts. It is initialized with:
    1) event_generator: IEventGenerator,
    2) model_trainer: AsyncTrainingEventHandler, and
    3) num_train_end_events_per_epoch: int.
    Async trainer's top-level run() function will be called for each epoch.
    Then, the simulator will first fetch a DeviceState from its min_heap and
    updates the simulator’s timestamp to the event’s relevant_time. If the
    fetched DeviceState is waiting for training, we start training with
    AsyncJobScheduler’s on_training_start_function, put the DeviceState back
    to the min_heap, and log to the JobQueueStats.
    Otherwise, it simply ends the training. Note that this works, because
    relevant_tmie is set to training end time after training is triggered.
    """

    logger: logging.Logger = Logger.get_logger(__name__)

    def __init__(
        self,
        event_generator: IEventGenerator,
        job_scheduler: AsyncTrainingEventHandler,
        user_selector: AsyncUserSelector,
        shared_client_config: ClientConfig,
        num_train_end_events_per_epoch: int,
        cuda_manager: ICudaStateManager = DEFAULT_CUDA_MANAGER,
        timeout_simulator: Optional[TimeOutSimulator] = None,
        channel: Optional[IdentityChannel] = None,
    ):
        self.job_scheduler: AsyncTrainingEventHandler = job_scheduler
        self.user_selector: AsyncUserSelector = user_selector
        self.event_generator: IEventGenerator = event_generator
        self.shared_client_config = shared_client_config
        self.num_train_end_events_per_epoch: int = num_train_end_events_per_epoch
        self.num_train_end_events: int = 0
        self.current_time: float = 0
        self.min_heap: PriorityQueue[AsyncClientDevice] = PriorityQueue()
        self.queue_stats: JobQueueStats = JobQueueStats()
        self.timeout_simulator = timeout_simulator
        self.channel = channel
        self.cuda_manager = cuda_manager
        # init the first event
        self.create_future_training_start_event()

    def create_future_training_start_event(self) -> None:
        # create training client and insert into heap
        new_client = AsyncClientFactory.create(
            current_time=self.current_time,
            event_generator=self.event_generator,
            user_selector=self.user_selector,
            client_config=self.shared_client_config,
            timeout_simulator=self.timeout_simulator,
            channel=self.channel,
            cuda_manager=self.cuda_manager,
        )
        # put will query __lt__ on top
        self.min_heap.put(new_client)

    def start_training(self, top: AsyncClientDevice) -> None:
        self.job_scheduler.on_training_start(top)
        # put will query __lt__ on top
        self.min_heap.put(top)
        self.queue_stats.on_job_start()

    def end_training(self, top: AsyncClientDevice) -> None:
        self.queue_stats.on_job_end()
        self.job_scheduler.on_training_end(top)

    def run_one_epoch(self) -> None:
        with tqdm(
            total=self.num_train_end_events_per_epoch,
            desc="Client Training Per Epoch",
            unit="client",
            position=0,
        ) as pbar:
            while not self.min_heap.empty():
                # notify async trainer that num_train_end_events_per_epoch have ended training
                if self.num_train_end_events == self.num_train_end_events_per_epoch:
                    self.num_train_end_events = 0
                    break

                top = self.min_heap.get()
                # forward time
                self.current_time = top.next_event_time()

                if top.is_waiting_to_start():
                    self.start_training(top)
                    self.create_future_training_start_event()
                else:
                    self.num_train_end_events += 1
                    self.end_training(top)
                    pbar.update(1)

    def avg_pending_jobs(self) -> float:
        return self.queue_stats.avg_pending_jobs()

    def print_stats(self, prefix: str) -> None:
        print(f"{prefix}PendingJobs, {self.queue_stats.as_str()}")
        print(f"Current Time: {self.current_time:.4f}")
