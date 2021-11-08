#!/bin/bash
git clone https://github.com/TalwalkarLab/leaf.git
cd leaf/data/sent140 || exit
./preprocess.sh --sf 0.01 -s niid -t 'user' --tf 0.90 -k 1 --spltseed 1
