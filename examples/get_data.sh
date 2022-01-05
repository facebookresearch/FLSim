#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

git clone https://github.com/TalwalkarLab/leaf.git
cd leaf/data/sent140 || exit
./preprocess.sh --sf 0.01 -s niid -t 'user' --tf 0.90 -k 1 --spltseed 1
