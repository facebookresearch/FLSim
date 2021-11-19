#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

# The base hydra configs for different components of FLSim are defined
# in their respective packages' __init__.py files.
# Importing the trainer module will execute flsim.trainers' __init__
# and cause chain reaction leading to execution and consequent import
# of all the config store definitions. Similarly, importing data module
# will import all necessary dataloader configs.

import flsim.data  # noqa
import flsim.trainers  # noqa
