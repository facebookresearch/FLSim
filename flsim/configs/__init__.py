#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The base hydra configs for different components of FLSim are defined
# in their respective packages' __init__.py files.
# Importing the trainer module will execute flsim.trainers' __init__
# and cause chain reaction leading to execution and consequent import
# of all the config store definitions. Similarly, importing data module
# will import all necessary dataloader configs.

import flsim.data  # noqa
import flsim.trainers  # noqa
