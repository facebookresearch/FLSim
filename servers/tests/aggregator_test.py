#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.


from tempfile import mkstemp

import torch.distributed as dist
import torch.multiprocessing as mp
from flsim.servers.aggregator import Aggregator, AggregationType
from flsim.tests.utils import (
    create_model_with_value,
    model_parameters_equal_to_value,
)
from flsim.utils.distributed.fl_distributed import FLDistributedUtils, OperationType
from libfb.py import testutil


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


class AggregatorTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_zero_weights(self) -> None:
        model = create_model_with_value(0)
        ag = Aggregator(module=model, aggregation_type=AggregationType.AVERAGE)
        weight = 1.0
        steps = 5

        for _ in range(steps):
            delta = create_model_with_value(1.0)
            ag.apply_weight_to_update(delta=delta, weight=weight)
            ag.add_update(delta=delta, weight=weight)
        self.assertEqual(ag.weights, weight * steps)

        ag.zero_weights()
        self.assertEqual(ag.weights, 0)

    @testutil.data_provider(
        lambda: (
            {
                "agg_type": AggregationType.AVERAGE,
                "num_process": 4,
                "num_models": 10,
                "expected_value": 1.0,
            },
            {
                "agg_type": AggregationType.WEIGHTED_AVERAGE,
                "num_process": 4,
                "num_models": 10,
                "expected_value": 1.0,
            },
            {
                "agg_type": AggregationType.WEIGHTED_SUM,
                "num_process": 4,
                "num_models": 10,
                "expected_value": 55.0,
            },
            {
                "agg_type": AggregationType.SUM,
                "num_process": 4,
                "num_models": 10,
                "expected_value": 10.0,
            },
        )
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
            self.assertAlmostEqual(result, expected_value, places=5)

    @testutil.data_provider(
        lambda: (
            {"agg_type": AggregationType.AVERAGE, "expected_value": 1.0},
            {"agg_type": AggregationType.WEIGHTED_AVERAGE, "expected_value": 1.0},
            {"agg_type": AggregationType.WEIGHTED_SUM, "expected_value": 55.0},
            {"agg_type": AggregationType.SUM, "expected_value": 10.0},
        )
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
        self.assertEmpty(error_msg, msg=error_msg)

    @testutil.data_provider(
        lambda: (
            {"agg_type": AggregationType.AVERAGE, "expected_value": 10.0},
            {"agg_type": AggregationType.WEIGHTED_AVERAGE, "expected_value": 55.0},
            {"agg_type": AggregationType.WEIGHTED_SUM, "expected_value": 55.0},
            {"agg_type": AggregationType.SUM, "expected_value": 10.0},
        )
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

        self.assertEqual(ag.weights, expected_value)

    @testutil.data_provider(
        lambda: (
            {"dist_op": OperationType.SUM, "agg_type": AggregationType.SUM},
            {"dist_op": OperationType.SUM, "agg_type": AggregationType.AVERAGE},
            {
                "dist_op": OperationType.SUM,
                "agg_type": AggregationType.WEIGHTED_AVERAGE,
            },
            {"dist_op": OperationType.SUM, "agg_type": AggregationType.WEIGHTED_SUM},
        )
    )
    def test_distributed_op_aggregation(self, dist_op, agg_type):
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
            self.assertNotEqual(r, v)
