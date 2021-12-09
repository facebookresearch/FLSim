#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from tempfile import mkstemp

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from flsim.common.pytest_helper import (
    assertEqual,
    assertNotEqual,
    assertAlmostEqual,
    assertEmpty,
)
from flsim.servers.aggregator import Aggregator, AggregationType
from flsim.tests.utils import (
    create_model_with_value,
    model_parameters_equal_to_value,
)
from flsim.utils.distributed.fl_distributed import FLDistributedUtils, OperationType


def init_process(
    rank,
    world_size,
    aggregator,
    models,
    file_loc,
    pipe,
    distributed_op,
):
    FLDistributedUtils.dist_init(
        rank=rank,
        world_size=world_size,
        init_method=f"file://{file_loc}",
        use_cuda=False,
    )
    aggregator.zero_weights()
    for i, m in enumerate(models):
        if i % world_size == rank:
            weight = i + 1
            aggregator.apply_weight_to_update(delta=m, weight=weight)
            aggregator.add_update(m, weight=weight)
    module = aggregator.aggregate(distributed_op)
    sums, weights = 0.0, 0.0
    all_sum = [(p.sum(), p.numel()) for p in module.parameters()]
    for s, w in all_sum:
        sums += float(s)
        weights += float(w)
    pipe.send(sums / weights)
    dist.destroy_process_group()


def run_multiprocess_aggregation_test(
    aggregator,
    num_processes=1,
    num_models=4,
    distributed_op=OperationType.SUM_AND_BROADCAST,
):
    _, tmpfile = mkstemp(dir="/tmp")
    pipe_out, pipe_in = mp.Pipe(False)
    models = [create_model_with_value(1.0) for i in range(num_models)]

    processes = []
    results = []
    FLDistributedUtils.WORLD_SIZE = num_processes
    for pid in range(num_processes):
        p = mp.Process(
            target=init_process,
            args=(
                pid,
                num_processes,
                aggregator,
                models,
                tmpfile,
                pipe_in,
                distributed_op,
            ),
        )
        p.start()
        processes.append(p)
        results.append(pipe_out)
    for p in processes:
        p.join()
    res = [r.recv() for r in results]
    return res


AGGREGATION_TYPES = [
    AggregationType.AVERAGE,
    AggregationType.WEIGHTED_AVERAGE,
    AggregationType.SUM,
    AggregationType.WEIGHTED_SUM,
]


class TestAggregator:
    def test_zero_weights(self) -> None:
        model = create_model_with_value(0)
        ag = Aggregator(module=model, aggregation_type=AggregationType.AVERAGE)
        weight = 1.0
        steps = 5

        for _ in range(steps):
            delta = create_model_with_value(1.0)
            ag.apply_weight_to_update(delta=delta, weight=weight)
            ag.add_update(delta=delta, weight=weight)
        assertEqual(ag.sum_weights.item(), weight * steps)

        ag.zero_weights()
        assertEqual(ag.sum_weights.item(), 0)

    @pytest.mark.parametrize(
        "agg_type,num_process,num_models,expected_value",
        [
            (AggregationType.AVERAGE, 4, 10, 1.0),
            (AggregationType.WEIGHTED_AVERAGE, 4, 10, 1.0),
            (AggregationType.WEIGHTED_SUM, 4, 10, 55.0),
            (AggregationType.SUM, 4, 10, 10.0),
        ],
    )
    def test_multiprocess_aggregation(
        self, agg_type, num_process, num_models, expected_value
    ):
        model = create_model_with_value(0)
        ag = Aggregator(module=model, aggregation_type=agg_type)

        results = run_multiprocess_aggregation_test(
            ag, num_processes=num_process, num_models=num_models
        )
        for result in results:
            assertAlmostEqual(result, expected_value, places=5)

    @pytest.mark.parametrize(
        "agg_type,expected_value",
        [
            (AggregationType.AVERAGE, 1.0),
            (AggregationType.WEIGHTED_AVERAGE, 1.0),
            (AggregationType.WEIGHTED_SUM, 55.0),
            (AggregationType.SUM, 10.0),
        ],
    )
    def test_aggregate(self, agg_type, expected_value):

        model = create_model_with_value(0)
        ag = Aggregator(module=model, aggregation_type=agg_type)

        ag.zero_weights()
        for i in range(10):
            delta = create_model_with_value(1.0)
            weight = i + 1
            ag.apply_weight_to_update(delta=delta, weight=weight)
            ag.add_update(delta=delta, weight=weight)

        model = ag.aggregate()
        error_msg = model_parameters_equal_to_value(model, expected_value)
        assertEmpty(error_msg, msg=error_msg)

    @pytest.mark.parametrize(
        "agg_type,expected_value",
        [
            (AggregationType.AVERAGE, 10.0),
            (AggregationType.WEIGHTED_AVERAGE, 55.0),
            (AggregationType.WEIGHTED_SUM, 55.0),
            (AggregationType.SUM, 10.0),
        ],
    )
    def test_add_update(self, agg_type, expected_value):
        model = create_model_with_value(0)
        ag = Aggregator(module=model, aggregation_type=agg_type)
        ag.zero_weights()
        for i in range(10):
            delta = create_model_with_value(1.0)
            weight = i + 1
            ag.apply_weight_to_update(delta=delta, weight=weight)
            ag.add_update(delta=delta, weight=weight)

        assertEqual(ag.sum_weights.item(), expected_value)

    @pytest.mark.parametrize(
        "agg_type,dist_op",
        [
            (AggregationType.AVERAGE, OperationType.SUM),
            (AggregationType.WEIGHTED_AVERAGE, OperationType.SUM),
            (AggregationType.WEIGHTED_SUM, OperationType.SUM),
            (AggregationType.SUM, OperationType.SUM),
        ],
    )
    def test_distributed_op_aggregation(self, agg_type, dist_op):
        """
        Test aggregation with only SUM and no BROADTCAST then each worker should have
        different parameters.
        """
        model = create_model_with_value(0)
        ag = Aggregator(module=model, aggregation_type=agg_type)

        results = run_multiprocess_aggregation_test(
            ag,
            num_processes=4,
            num_models=10,
            distributed_op=dist_op,
        )
        for r, v in zip(results, results[1:]):
            assertNotEqual(r, v)
