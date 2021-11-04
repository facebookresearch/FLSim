#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import numpy as np
import torch
import torch.multiprocessing as mp
from flsim.examples.multiproc_test import train, init_process, MultiprocConfig
from flsim.tests import utils
from flsim.utils.distributed.fl_distributed import FLDistributedUtils
from libfb.py import testutil


def multiprocess_training(world_size) -> None:
    settings = MultiprocConfig(world_size=world_size)
    torch.manual_seed(settings.random_seed)
    np.random.seed(settings.random_seed)
    torch.cuda.manual_seed_all(settings.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = utils.SampleNet(utils.TwoFC())
    FLDistributedUtils.WORLD_SIZE = settings.world_size
    processes = []
    for rank in range(settings.world_size):
        p = mp.Process(target=init_process, args=(model, rank, settings, train))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


class FLSimMultiprocTest(testutil.BaseFacebookTestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_mp_world_size_1(self):
        multiprocess_training(world_size=1)

    def test_mp_world_size_2(self):
        multiprocess_training(world_size=2)

    def test_mp_world_size_4(self):
        multiprocess_training(world_size=4)
