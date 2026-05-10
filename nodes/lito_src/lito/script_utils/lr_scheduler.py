#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements learning rate schedulers.


import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmup(_LRScheduler):
    """
    Args:
        optimizer (Optimizer):
            Wrapped optimizer.
        first_cycle_steps (int):
            First cycle step size.
        cycle_mult(float):
            Cycle steps magnification. Default: -1.
        max_lr(float):
            First cycle's max learning rate. Default: 0.1.
        min_lr(float):
            Min learning rate. Default: 0.001.
        warmup_steps(int):
            Linear warmup step size. Default: 0.
        gamma(float):
            Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int):
            The index of last epoch. Default: -1.

    Ref:
        https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ):
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size

        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(LinearWarmup, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs
            ]
        else:
            return [self.max_lr for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.step_in_cycle = epoch
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Args:
        optimizer (Optimizer):
            Wrapped optimizer.
        first_cycle_steps (int):
            First cycle step size.
        cycle_mult(float):
            Cycle steps magnification. Default: -1.
        max_lr(float):
            First cycle's max learning rate. Default: 0.1.
        min_lr(float):
            Min learning rate. Default: 0.001.
        warmup_steps(int):
            Linear warmup step size. Default: 0.
        gamma(float):
            Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int):
            The index of last epoch. Default: -1.

    Ref:
        https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi * (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult**n - 1) / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class TFScheduler(_LRScheduler):
    """
    The learning rate schedule used in Attention is all you need.
    Ref: https://arxiv.org/pdf/1706.03762
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_size: int,
        lr_factor: float = 1.0,
        warmup_steps: int = 4000,
        last_epoch: int = -1,
    ):
        r"""
        Vary the learning rate every training step according to:

            .. math::
                lr = \text{factor} * d_{model}^{-0.5} * \min(\text{iter}^{-0.5}, \text{iter} * \text{warmup}^{-1.5})

        This corresponds to linearly increasing the learning rate for the first `warmup` training steps,
        and after that decreasing it proportionally to the inverse square root of the iteration (step).
        This optimizer was used by Vaswani, Ashish, et al. ("Attention is all you need." 2017).


        Args:
            optimizer (torch.optim.Optimizer):
                pytorch optimizer to warp around
            model_size (int):
                primary dimension of the model
            lr_factor (float):
                scalar to multiply. It is to scale the learning rate.
            warmup_steps (int):
                number of warm up steps to linearly increase the learning rate with
            last_epoch:
                the last iteration count (for recovery)
        """
        num_groups = len(optimizer.param_groups)

        if isinstance(warmup_steps, (int, float)):
            warmup_steps = [warmup_steps] * num_groups
        self.warmup_steps = warmup_steps

        if isinstance(lr_factor, (int, float)):
            lr_factor = [lr_factor] * num_groups
        self.lr_factor = lr_factor

        if isinstance(model_size, (int, float)):
            model_size = [model_size] * num_groups
        self.model_size = model_size

        super().__init__(
            optimizer=optimizer,
            last_epoch=last_epoch,
        )

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        # self.last_epoch is the step count (ie, number of iterations)
        return [
            self.rate(
                lr_factor=self.lr_factor[i],
                model_size=self.model_size[i],
                warmup_steps=self.warmup_steps[i],
                step=None,
            )
            for i in range(len(self.optimizer.param_groups))
        ]

    def rate(
        self,
        lr_factor: float,
        model_size: float,
        warmup_steps: int,
        step: int = None,
    ) -> float:
        """
        Compute the learning rate at a given step.

        Args:
            step (int or None):
                the step to compute the learning rate.
                If None, use the current step of the optimizer.

        Returns:
            learning rate (float)
        """
        if step is None:
            step = self.last_epoch

        step = max(1, step)
        return lr_factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5)))


class TFSchedulerWithBaseLR(TFScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_size: int,
        lr_factor: float = 1.0,
        warmup_steps: int = 4000,
        last_epoch: int = -1,
        base_lr: float = None,
    ):
        # Must be placed before super().__init__() since LRScheduler will call get_lr() during initialization.
        self.base_lr = base_lr

        super().__init__(
            optimizer=optimizer,
            model_size=model_size,
            lr_factor=lr_factor,
            warmup_steps=warmup_steps,
            last_epoch=last_epoch,
        )

    def rate(
        self,
        lr_factor: float,
        model_size: float,
        warmup_steps: int,
        step: int = None,
    ) -> float:
        rate = super().rate(lr_factor=lr_factor, model_size=model_size, warmup_steps=warmup_steps, step=step)
        if self.base_lr is not None:
            rate = min(self.base_lr, rate)
        return rate


class ConstantLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr: float,
        last_epoch: int = -1,
    ):
        self.lr = lr
        super().__init__(
            optimizer=optimizer,
            last_epoch=last_epoch,
        )

    def get_lr(self):
        return [self.lr for i in range(len(self.optimizer.param_groups))]
