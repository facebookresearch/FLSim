#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import IntEnum
from itertools import chain
from typing import List, Iterable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from flsim.common.logger import Logger
from flsim.utils.fl.common import FLModelParamUtils


class OperationType(IntEnum):
    BROADCAST = 0
    SUM_AND_BROADCAST = 1
    SUM = 2


class FLDistributedUtils:
    """
    More detailed note: https://fburl.com/5n8mnaf3. We cannot use PyTorch
    DDP here, because DDP is tied to backward() and only provides
    high-level APIs for reducing gradients, but in FL, after each round, we
    need to perform all-reduce on the models (optionally with some pre and
    post-processing) instead of just model gradients (e.g. FedAvg).
    We reduce the number of all-reduce operations by flatten an entire model
    into a 1D tensor, if the model size is less than a buffer limit of 256MB.
    For large models, we group the model parameters into flatten buckets of
    256MB each and call all-reduce() in each bucket using async operations.
    """

    logger: logging.Logger = Logger.get_logger(__name__)

    # equivalent to 256 MB of floats, same buffer size as in PyTorch DDP
    MAX_BUFFER_SIZE = 2 ** 28
    WORLD_SIZE = 1
    # allow distributed training on CPU, default False
    DISTRIBUTED_TRAINING_ON_CPU = False
    DISTRIBUTED_BACKED = dist.Backend.NCCL

    @classmethod
    def distributed_training_on_cpu(cls):
        cls.DISTRIBUTED_TRAINING_ON_CPU = True
        cls.DISTRIBUTED_BACKED = dist.Backend.GLOO

    @classmethod
    def distributed_training_on_cuda(cls):
        """
        this is on by default, use if you have called
        distributed_training_on_cpu and want to enable
        cpu distributed again.
        """
        cls.DISTRIBUTED_TRAINING_ON_CPU = False
        cls.DISTRIBUTED_BACKED = dist.Backend.NCCL

    @classmethod
    def check_distributed_training_configs(cls, distributed_world_size: int):
        if cls.DISTRIBUTED_TRAINING_ON_CPU:
            assert distributed_world_size <= mp.cpu_count(), (
                f"Only {mp.cpu_count()} GPUs are available, "
                f"but {distributed_world_size} CPUs were requested."
            )
        elif distributed_world_size > 1:
            assert torch.cuda.is_available(), (
                "distributed_world_size is greater than 1 "
                "use only if cuda is supported or distributed_training_on_cpu"
                "has been called!"
            )
            assert distributed_world_size <= torch.cuda.device_count(), (
                f"Only {torch.cuda.device_count()} GPUs are available, "
                f"but {distributed_world_size} GPUs were requested."
            )

    @classmethod
    def setup_distributed_training(cls, distributed_world_size: int, use_cuda=True):
        cls.WORLD_SIZE = distributed_world_size
        if use_cuda:
            cls.distributed_training_on_cuda()
        else:
            cls.distributed_training_on_cpu()
        cls.check_distributed_training_configs(distributed_world_size)

    @classmethod
    def distributed_operation(
        cls,
        params: Iterable[torch.Tensor],
        op: OperationType = OperationType.SUM_AND_BROADCAST,
        src: int = -1,
        dst: int = -1,
    ):
        """
        Group params into a list of flatten buffers and call the distributed
        operation on each buffer asynchronously.

        The actual async operation for each buffer is done in the helper function
        `_distributed_operation`

        Starting with an unprocessed buffer, loops over params and does one of the following:

        * appends the param to the current unprocessed buffer if buffer has space
        * if buffer cannot fit the param, if the param can fit into a new buffer
          sends the current buffer `_distributed_operation` and creates a new buffer
          or else sends param to `_distributed_operation` and keeps the buffer for the
          next param in the list.

        At the end the function joins all async ops and puts processed values from each flattened
        buffer in to their respective param.

        Note:
            In all operations it is assumed that the master worker is the worker with rank 0.

        """

        if cls.WORLD_SIZE == 1:
            return
        # temp variable of list of model params sent organized into one buffer
        operation_results = []  # operation results a list of (handle, buffer)
        param_references = []  # list of param-lists in each buffer
        buffered_params = []  # buffer to hord tensors until enough for dist operation

        offset = 0
        with torch.no_grad():
            for param in params:
                sz = param.numel()
                if sz + offset <= cls.MAX_BUFFER_SIZE:
                    # append the params and postpone the operation
                    buffered_params.append(param)
                    offset += sz
                    continue
                # do the operation, the buffer cannot be appended anymore
                process_independently = sz > cls.MAX_BUFFER_SIZE
                tensor_list = [param] if process_independently else buffered_params
                operation_result = cls._distributed_operation(
                    tensor_list, sz, op, src, dst
                )  # operation result is a tuple of (handle, buffer)
                operation_results.append(operation_result)
                param_references.append(tensor_list)
                offset = offset if process_independently else sz
                buffered_params = buffered_params if process_independently else [param]

            if len(buffered_params) > 0:
                operation_result = cls._distributed_operation(
                    buffered_params, offset, op, src, dst
                )  # operation result is a tuple of (handle, buffer)
                operation_results.append(operation_result)
                param_references.append(buffered_params)

        # wait on the async handle
        for handle, _ in operation_results:
            handle.wait()

        # copy data from flattened buffers to the actual tensors.
        for params, (_, buffer) in zip(param_references, operation_results):
            cls._get_params_from_buffer(params, buffer)

    @classmethod
    def _distributed_operation(
        cls,
        params: List[torch.Tensor],
        numels: int,
        op: OperationType,
        src: int = -1,
        dst: int = -1,
    ):
        """
        Returns a tuple of handle and buffer. Caller is RESPONSIBLE for awaiting
        on handle and then use whatever that's filled in the buffer.
        Creates a buffer of the size of 'numels'. Then, we loop over the
        'params', which is a list of tensors, and copy each tensor (which is
        avset of parameters from model) to buffer one by one. After that, we
        callvall_reduce() function in PyTorch distributed as an async
        operation tovall processes in the group (and get async handle to
        return after this).

        Args:
            params: List[torch.Tensor], a buffer group of parameters to perform
            async operation at one time
            numels: total number of scalar elements in params

        Returns:
            handle: an async handle
            buffer: within distributed operation, params: List[torch.Tensor] is flattened
            as a buffer (1D Tensor) and sent to all_reduce. buffer will store the
            result of distributed option once it is finished.

        Note:
            Size of each param in params are not required to be the same. params is first flatten
            to a 1D tensor. E.g:

                params = Tensor(
                    [1,2,3,4], [ [5,6], [7,8] ], [9,10]
                )

                then buffer is

                [1,2,3,4,5,6,7,8,9,10]

        Example:
            if worker 1 has
                params = [
                    Tensor([1,2,3,4]),
                    Tensor([ [5,6], [7,8] ]),
                    Tensor([9,10])
                ]

            and worker 2 has
                params = [
                    Tensor([10,20,30,40]),
                    Tensor([ [50,60], [70,80] ]),
                    Tensor([90,100])
                ]

            and if the operation type is sum, the returned buffer will be:

            Tensor([11, 22, 33, 44, 55, 66, 77, 88, 99, 110])

        """
        # TODO: enable all_reduce on mixed dtypes with dtype-based bucketing
        # currently the assumption is that there is at least one float tensor
        # so all layers could be casted to float
        # NOTE: seems to work for mixed int and float types
        generic_type = torch.float
        for p in params:
            if p.dtype != generic_type:
                cls.logger.warning("non float tensor types sent to all reduce")

        buffer = params[0].new_empty(numels, dtype=generic_type)
        offset = 0
        for p in params:
            sz = p.numel()
            buffer[offset : offset + sz].copy_(p.data.view(-1))
            offset += sz

        if op == OperationType.SUM_AND_BROADCAST:
            handle = dist.all_reduce(
                buffer,
                op=dist.ReduceOp.SUM,
                group=cls._get_default_group(),
                async_op=True,
            )
        elif op == OperationType.SUM:
            if dst < 0:
                cls.logger.debug("dst is not defined setting 0 as the default value")
                dst = 0
            cls.logger.warning("Operation reduce is not supported on CPU on ")

            handle = dist.reduce(
                buffer,
                dst,
                op=dist.ReduceOp.SUM,
                group=cls._get_default_group(),
                async_op=True,
            )
        elif op == OperationType.BROADCAST:
            if src < 0:
                cls.logger.debug(
                    "Distributed copy operation (broadcast) needs a source."
                    "Assigning 0 as the default source"
                )
                src = 0
            handle = dist.broadcast(
                buffer,
                src,
                group=cls._get_default_group(),
                async_op=True,
            )
        else:
            raise ValueError(f"Operation {op} not found. Please check the parameters.")
        return (handle, buffer)

    @classmethod
    def _get_params_from_buffer(cls, params: List[torch.Tensor], buffer: torch.Tensor):
        """
        Inverse the buffering operation in all_reduce and copies the data
        in buffer into each param in params.
        i.e. Copies all-reduced grads back into their original place. However,
        more generally speaking, what this function actually does is treating the
        'buffer' (i.e. the 2nd param) as a well-flattened 1D tensor of the list
        of params and copy all the params back to buffer.
        """
        # the total number of elements of params
        # copy all-reduced grads back into their original place
        offset = 0
        for p in params:
            sz = p.numel()
            p.data.copy_(buffer[offset : offset + sz].view_as(p))
            offset += sz

    @classmethod
    def _get_default_group(cls):
        return dist.group.WORLD

    @classmethod
    def is_master_worker(cls):
        """
        we assume that worker 0 is the master worker."
        """
        return (not dist.is_initialized()) or dist.get_rank() == 0

    @classmethod
    def suppress_output(cls):
        import builtins as __builtin__

        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            # force print the result when kwargs contains force and value is True
            if kwargs.pop("force", False):
                builtin_print(*args, **kwargs)

        __builtin__.print = print

    @classmethod
    def dist_init(
        cls,
        rank: int,
        world_size: int,
        init_method: str,
        use_cuda: bool = True,
    ):
        cls.setup_distributed_training(world_size, use_cuda)
        if not cls.DISTRIBUTED_TRAINING_ON_CPU:
            device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(device)
        if world_size > 1:
            dist.init_process_group(
                backend=cls.DISTRIBUTED_BACKED,
                init_method=init_method,
                world_size=world_size,
                rank=rank,
            )

    @classmethod
    def synchronize_across_ranks(
        cls,
        model: nn.Module,
        weights: torch.Tensor,
        operation: OperationType,
        only_federated_params: bool = False,
    ):
        state_dict = FLModelParamUtils.get_state_dict(
            model, only_federated_params=only_federated_params
        )
        cls.distributed_operation(
            params=chain([weights], state_dict.values()), op=operation
        )
