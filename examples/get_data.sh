#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

git clone https://github.com/TalwalkarLab/leaf.git
cd leaf/data/femnist || exit
./preprocess.sh --sf 1.0 -s niid -t 'user' --tf 0.90 -k 2 --spltseed 1
