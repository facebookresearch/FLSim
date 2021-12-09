#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from flsim.common.pytest_helper import assertEqual, assertTrue, assertRaises
from flsim.secure_aggregation.secure_aggregator import (
    FixedPointConverter,
    SecureAggregator,
    FixedPointConfig,
    utility_config_flatter,
)
from flsim.tests import utils
from omegaconf import OmegaConf


class TestSecureAggregator:
    def _create_model(self, model_param_value):
        """
        Creates a two-layer model
        """
        fl_model = utils.SampleNet(utils.TwoFC())
        fl_model.fl_get_module().fill_all(model_param_value)
        return fl_model.fl_get_module()

    def test_fixedpoint_init(self):
        """
        Tests that FixedPointConverter init works correctly
        """

        converter = FixedPointConverter(
            **OmegaConf.structured(FixedPointConfig(num_bytes=2, scaling_factor=1000))
        )
        assertEqual(converter.max_value, 32767)
        assertEqual(converter.min_value, -32768)

        with assertRaises(ValueError):
            converter = FixedPointConverter(
                **OmegaConf.structured(
                    FixedPointConfig(num_bytes=9, scaling_factor=1000)
                )
            )

        with assertRaises(ValueError):
            converter = FixedPointConverter(
                **OmegaConf.structured(
                    FixedPointConfig(num_bytes=3, scaling_factor=-100)
                )
            )

    def test_floating_to_fixedpoint(self):
        """
        Tests whether conversion from floating point to fixed point works
        """

        #  hence minValue = -32768, maxValue = 32767
        converter = FixedPointConverter(
            **OmegaConf.structured(FixedPointConfig(num_bytes=2, scaling_factor=100))
        )

        x = torch.tensor(17.42)
        y = converter.to_fixedpoint(x)
        # y = x * scaling_factor = 1742.0 ==> round to 1742
        assertEqual(y, torch.tensor(1742))

        x = torch.tensor(17.4298)
        y = converter.to_fixedpoint(x)
        # y = x * scaling_factor = 1742.98 ==> round to 1743
        assertEqual(y, torch.tensor(1743))

        x = torch.tensor(-2.34)
        y = converter.to_fixedpoint(x)
        # y = x * scaling_factor = -234.0 ==> round to -234
        assertEqual(y, torch.tensor(-234))

        x = torch.tensor(-2.3456)
        y = converter.to_fixedpoint(x)
        # y = x * scaling_factor = -234.56 ==> round to -235
        assertEqual(y, torch.tensor(-235))

        x = torch.tensor(-2.3416)
        y = converter.to_fixedpoint(x)
        # y = x * scaling_factor = -234.16 ==> round to -234
        assertEqual(y, torch.tensor(-234))

        x = torch.tensor(12345.0167)
        y = converter.to_fixedpoint(x)
        # y = x * scaling_factor = 1234501.67 ==> adjust to maxValue 32767
        assertEqual(y, torch.tensor(32767))

        x = torch.tensor(-327.69)
        y = converter.to_fixedpoint(x)
        # y = x * scaling_factor = -32769 ==> adjust to minValue -32768
        assertEqual(y, torch.tensor(-32768))

    def test_fixed_to_floating_point(self):
        """
        Tests whether conversion from fixed point to floating point works
        """

        converter = FixedPointConverter(
            **OmegaConf.structured(FixedPointConfig(num_bytes=1, scaling_factor=85))
        )

        x = torch.tensor(85)
        y = converter.to_float(x)
        # y = x / scaling_factor = 1.0
        assertTrue(torch.allclose(y, torch.tensor(1.0), rtol=1e-10))

        x = torch.tensor(157)
        y = converter.to_float(x)
        # y = x / scaling_factor = 1.847058823529412
        assertTrue(torch.allclose(y, torch.tensor(1.847058823529412), rtol=1e-10))

    def test_params_floating_to_fixedpoint(self):
        """
        Tests whether the parameters of a model are converted correctly
        from floating point to fixed point
        """

        #  hence minValue = -32768, maxValue = 32767
        config = FixedPointConfig(num_bytes=2, scaling_factor=100)

        model = self._create_model(6.328)
        secure_aggregator = SecureAggregator(utility_config_flatter(model, config))
        secure_aggregator.params_to_fixedpoint(model)
        mismatched = utils.model_parameters_equal_to_value(model, 633.0)
        assertEqual(mismatched, "", mismatched)

        model = self._create_model(-3.8345)
        secure_aggregator = SecureAggregator(utility_config_flatter(model, config))
        secure_aggregator.params_to_fixedpoint(model)
        mismatched = utils.model_parameters_equal_to_value(model, -383.0)
        assertEqual(mismatched, "", mismatched)

    def test_params_floating_to_fixedpoint_different_config_for_layers(self):
        """
        Tests whether the parameters of a model are converted correctly
        from floating point to fixed point, when we have different
        FixedPointConverter configs for different layers

        """

        config_layer1 = FixedPointConfig(num_bytes=2, scaling_factor=100)
        #  hence minValue = -32768, maxValue = 32767
        config_layer2 = FixedPointConfig(num_bytes=1, scaling_factor=10)
        #  hence minValue = -128, maxValue = 127

        config = {}
        config["fc1.weight"] = config_layer1
        config["fc1.bias"] = config_layer1
        config["fc2.weight"] = config_layer2
        config["fc2.bias"] = config_layer2

        model = self._create_model(5.4728)
        secure_aggregator = SecureAggregator(config)
        secure_aggregator.params_to_fixedpoint(model)
        for name, p in model.named_parameters():
            if name == "fc1.weight" or name == "fc1.bias":
                # round 547.28 to 547
                assertTrue(torch.allclose(p, torch.tensor(547.0), rtol=1e-10))
            if name == "fc2.weight" or name == "fc2.bias":
                # round 54.728 to 55
                assertTrue(torch.allclose(p, torch.tensor(55.0), rtol=1e-10))

    def test_error_raised_per_layer_config_not_set(self):
        """
        Tests whether all layers have their corresponding configs, when
        per layer fixed point converter is used.
        """

        config_layer1 = FixedPointConfig(num_bytes=8, scaling_factor=10000)

        config = {}
        config["fc1.weight"] = config_layer1
        config["fc1.bias"] = config_layer1

        model = self._create_model(600)
        secure_aggregator = SecureAggregator(config)

        with assertRaises(ValueError):
            secure_aggregator.params_to_float(model)

        with assertRaises(ValueError):
            secure_aggregator.params_to_fixedpoint(model)

    def test_params_fixed_to_floating_point(self):
        """
        Tests whether the parameters of a model are converted correctly
        from fixed point to floating point
        """
        config = FixedPointConfig(num_bytes=3, scaling_factor=40)
        model = self._create_model(880.0)
        secure_aggregator = SecureAggregator(utility_config_flatter(model, config))
        secure_aggregator.params_to_float(model)
        mismatched = utils.model_parameters_equal_to_value(model, 22.0)
        assertEqual(mismatched, "", mismatched)

    def test_params_fixed_to_floating_point_different_config_for_layers(self):
        """
        Tests whether the parameters of a model are converted correctly
        from fixed point to floating point, when we have different
        FixedPointConverter configs for different layers
        """
        config_layer1 = FixedPointConfig(num_bytes=2, scaling_factor=30)
        config_layer2 = FixedPointConfig(num_bytes=1, scaling_factor=80)

        config = {}
        config["fc1.weight"] = config_layer1
        config["fc1.bias"] = config_layer1
        config["fc2.weight"] = config_layer2
        config["fc2.bias"] = config_layer2

        model = self._create_model(832.8)
        secure_aggregator = SecureAggregator(config)
        secure_aggregator.params_to_float(model)
        for name, p in model.named_parameters():
            if name == "fc1.weight" or name == "fc1.bias":
                # 832.8 / 30 = 27.76
                assertTrue(torch.allclose(p, torch.tensor(27.76), rtol=1e-10))
            if name == "fc2.weight" or name == "fc2.bias":
                # 832.8 / 80 = 10.41
                assertTrue(torch.allclose(p, torch.tensor(10.41), rtol=1e-10))
