#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.optim.optimizer import Optimizer


class LARS(Optimizer):
    r"""Implements LARS algorithm.

    It has been proposed in `Large Batch Training of Convolutional Networks`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta (float, optional): coefficient used for computing
            running averages of gradient. (default: 0.9)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-3, beta=0.9, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = {"lr": lr, "beta": beta, "weight_decay": weight_decay}
        super(LARS, self).__init__(params, defaults)

    @torch.no_grad()
    def get_update(self, p, grad, state, group):

        if group["weight_decay"] != 0:
            grad.add_(p.data, alpha=group["weight_decay"])
        # State initialization
        if len(state) == 0:
            state["step"] = 0
            # Moving averages will be updated _in place_
            # Exponential moving average of gradient values
            state["exp_avg"] = torch.clone(grad).detach()

        # m_{t-1}
        exp_avg = state["exp_avg"]
        beta = group["beta"]

        state["step"] += 1

        # Decay the first moment running average coefficient
        exp_avg.mul_(beta).add_(grad, alpha=1 - beta)

        return exp_avg

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LARS does not support sparse gradients")

                state = self.state[p]

                update = self.get_update(p, grad, state, group)
                update_norm = update.pow(2).sum().sqrt()
                weight_norm = p.data.pow(2).sum().sqrt()
                # The LAMB paper suggests bounding the weight norm by some
                # hyperparameters but we choose to eliminate unnecessary
                # hyperparameters
                scaling_function = weight_norm
                assert update_norm != 0

                update.mul_(scaling_function / update_norm)
                p.data.add_(update, alpha=-group["lr"])

        return loss


class LAMB(LARS):
    r"""Implements LAMB algorithm.

    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta1 (float, optional): coefficient used for computing
            running averages of gradient (default 0.9)
        beta2 (float, optional): coefficient used for computing
            running average of gradient squared (default 0.999)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)


    """

    def __init__(
        self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta1: {}".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta2: {}".format(beta2))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        Optimizer.__init__(self, params, defaults)

    @torch.no_grad()
    def get_update(self, p, grad, state, group):
        # State initialization
        if len(state) == 0:
            state["step"] = 0

            # Moving averages will be updated _in place_
            # Exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state["exp_avg_sq"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )

        # m_{t-1} and v_{t-1}
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1 = group["beta1"]
        beta2 = group["beta2"]

        state["step"] += 1
        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]

        # m_t = (beta1 * m_{t-1} + (1-beta1)*g_t)
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        # v_t = (beta2 * v_{t-1} + (1-beta2)*g_t^2)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        m_update = torch.div(exp_avg, bias_correction1)
        v_update = torch.div(exp_avg_sq, bias_correction2)

        # denom = sqrt(v_t) + eps
        denom = torch.add(v_update.sqrt(), group["eps"])

        # update = l2-penalty + m_{t} / denom
        m_update.div_(denom)
        if group["weight_decay"] != 0:
            m_update.add_(p.data, alpha=group["weight_decay"])

        return m_update
