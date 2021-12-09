#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc

import torch
from flsim.interfaces.model import IFLModel


def FloatTensor(cuda_enabled: bool, *args):
    if cuda_enabled:
        return torch.cuda.FloatTensor(*args)
    else:
        return torch.FloatTensor(*args)


def tensor(data, dtype, cuda_enabled: bool):
    return torch.tensor(data, dtype=dtype, device=device(cuda_enabled))


def device(cuda_enabled: bool):
    return "cuda:{}".format(torch.cuda.current_device()) if cuda_enabled else "cpu"


class ICudaStateManager(abc.ABC):
    """Sends model from CPU-->GPU, and from GPU-->CPU,
    if required, at 3 different times:
    a) When trainer is initialized
    b) Before training or eval is done
    c) After training or eval is done
    Centralizes all CPU-->GPU/GPU-->CPU moves
    """

    @abc.abstractmethod
    def on_trainer_init(self, model: IFLModel):
        pass

    @abc.abstractmethod
    def before_train_or_eval(self, model: IFLModel):
        pass

    @abc.abstractmethod
    def after_train_or_eval(self, model: IFLModel):
        pass


class NoopCudaStateManager(ICudaStateManager):
    def __init__(self):
        pass

    def on_trainer_init(self, model: IFLModel):
        pass

    def before_train_or_eval(self, model: IFLModel):
        pass

    def after_train_or_eval(self, model: IFLModel):
        pass


class CudaTransferMinimizer(ICudaStateManager):
    """Minimize CPU<-->GPU memory bandwidth,
    at the cost of increasing GPU memory consumption
    Model is moved to GPU right when trainer is initialized.
    All model copies stay on the GPU.
    E.g: when Sync trainer creates clients, all of them
    get models that are already in GPU
    GPU<-->GPUMemory bandwidth is really high, 2 orders of magnitude
    larger than CPUMemory<-->GPUMemory bandwidth.
    THIS SHOULD BE THE DEFAULT UNLESS RUNNING OUT OF GPU MEMORY
    """

    def __init__(self, cuda_enabled: bool):
        self._cuda_enabled = cuda_enabled

    def on_trainer_init(self, model: IFLModel):
        """When trainer is initialized, we move the model to GPU
        Any furter copies of the model will stay on GPU
        """
        if self._cuda_enabled:
            model.fl_cuda()

    def before_train_or_eval(self, model: IFLModel):
        pass

    def after_train_or_eval(self, model: IFLModel):
        pass


class GPUMemoryMinimizer(ICudaStateManager):
    """Minimize GPU memory at the cost of increasing
    CPU-->GPU and GPU-->CPU memory bandwidth consumption
    Model is moved to GPU right before train/eval is called,
    and moved out of GPU right after.
    All operations other than training/eval happen on CPU.
    E.g: global model aggregation happens on CPU

    DONT USE THIS UNLESS YOU ARE RUNNING OUT OF GPU MEMORY
    """

    def __init__(self, cuda_enabled: bool):
        self._cuda_enabled = cuda_enabled

    def on_trainer_init(self, model: IFLModel):
        if self._cuda_enabled:
            model.fl_get_module().to("cpu")

    def before_train_or_eval(self, model: IFLModel):
        if self._cuda_enabled:
            model.fl_cuda()

    def after_train_or_eval(self, model: IFLModel):
        if self._cuda_enabled:
            model.fl_get_module().to("cpu")


# default manager, does nothing
DEFAULT_CUDA_MANAGER: ICudaStateManager = NoopCudaStateManager()
