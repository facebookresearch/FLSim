#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import torch
from flsim.channels.sketch_channel import SketchChannelConfig
from flsim.common.pytest_helper import assertIsInstance, assertEqual
from flsim.optimizers.sketch_aggregator import SketchAggregator, SketchAggregatorConfig
from flsim.tests import utils
from hydra.utils import instantiate


def test_sketch_aggregator() -> None:
    """
    Tests Sketch Aggregator and make sure its init, init_round, collect_update,
    and step are all correct.
    """

    # init
    channel = instantiate(SketchChannelConfig(independence=4))
    model = utils.SampleNet(utils.TwoFC())
    global_param = 1.0
    model.fl_get_module().fill_all(global_param)

    aggregator = instantiate(
        SketchAggregatorConfig(top_k_ratio=0.5),
        global_model=model,
        channel=channel,
    )
    assertIsInstance(aggregator, SketchAggregator)
    assertEqual(0.5, aggregator.cfg.top_k_ratio)
    assertEqual(4, aggregator.grad_cs.independence)
    assert torch.allclose(aggregator.grad_cs.buckets, torch.tensor([0.0]))
    assert torch.allclose(aggregator.error_cs.buckets, torch.tensor([0.0]))

    # init_round
    aggregator.init_round()
    assert torch.allclose(aggregator.reducer.round_cs.buckets, torch.tensor([0.0]))

    # collect_client_update
    model = utils.SampleNet(utils.TwoFC())
    model.fl_get_module().fill_all(0.6)
    aggregator.collect_client_update(model, 2.0)

    model = utils.SampleNet(utils.TwoFC())
    model.fl_get_module().fill_all(0.3)
    aggregator.collect_client_update(model, 1.0)

    expected_value1 = 0.6 * 2.0 + 0.3 * 1.0
    total_weight = 2.0 + 1.0

    params = aggregator.reducer.round_cs.unsketch_model()
    for _, param in params.items():
        assert torch.allclose(param, torch.tensor([expected_value1]))

    # step
    aggregator.step()
    ones = 0
    correct = 0
    for _, param in aggregator.global_model.fl_get_module().named_parameters():
        ones += torch.sum(param == 1)
        correct += torch.sum(
            param == global_param - expected_value1 / total_weight * aggregator.cfg.lr
        )

    assertEqual(correct, aggregator.top_k)
    assertEqual(ones, aggregator.grad_cs.n - aggregator.top_k)

    # init round
    aggregator.init_round()

    # collect_client_update again
    model = utils.SampleNet(utils.TwoFC())
    model.fl_get_module().fill_all(1.2)
    aggregator.collect_client_update(model, 1.0)

    model = utils.SampleNet(utils.TwoFC())
    model.fl_get_module().fill_all(0.9)
    aggregator.collect_client_update(model, 2.0)

    expected_value2 = 0.9 * 2.0 + 1.2 * 1.0
    total_weight = 2.0 + 1.0

    params = aggregator.reducer.round_cs.unsketch_model()
    for _, param in params.items():
        assert torch.allclose(param, torch.tensor([expected_value2]))

    # step again
    expected_grad = (
        aggregator.cfg.momentum * expected_value1 / total_weight
        + expected_value2 / total_weight
    )

    aggregator.step()
    assert torch.allclose(
        aggregator.grad_cs.query(torch.arange(0, 21)),
        torch.tensor([expected_grad]),
    )
