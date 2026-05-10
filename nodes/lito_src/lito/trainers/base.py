#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements the base trainer.


import gc
import os
import random
import typing as T

import lightning as L
import numpy as np

import torch


class BaseTrainer(L.LightningModule):
    def __init__(self):
        super().__init__()

    def on_train_epoch_start(self):
        # the function is called before creating dataloading workers
        # we can change dataloader.dataset and it will change the
        # dataset state each worker gets
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        os.environ["EPOCH"] = str(self.current_epoch)

    def on_save_checkpoint(self, checkpoint):
        # save the config if available
        try:
            checkpoint["config"] = self.config
        except Exception:
            pass

    def on_load_checkpoint(self, checkpoint):
        # save the config if available
        try:
            self.config = checkpoint["config"]
        except Exception:
            pass

    def on_fit_start(self, *args, **kwargs):
        # create a torch generator that has the same
        # seed on all global rank
        self.generator_shared = torch.Generator(device=self.device)

        initial_seed = torch.initial_seed()
        self.generator_shared.manual_seed(initial_seed)

        # seed differently for different global rank
        seed = initial_seed + self.global_rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # just to be safe
        np.random.seed(seed)
        random.seed(seed)

        # print(f"({self.global_rank}) on_fit_start:  self.device: {self.device}")
        # for name, param in self.named_parameters():
        #     if param is not None and isinstance(param, torch.Tensor):
        #         if param.device != self.device:
        #             print(f"({self.global_rank}) name: {name}, device: {param.device}")


class SkipGradNaNTrainer(BaseTrainer):
    """This class checks the gradient of every backward and skip that step for all ranks if NaN or overflow (infinity)
    observed in the gradient.
    """

    def on_after_backward(self):
        # Guard: check grads before optimizer.step

        debug_param_sum = 0.0
        debug_grad_sum = 0.0

        bad = False
        for p in self.parameters():
            if p.grad is not None:
                debug_param_sum += torch.sum(p.data.detach().abs())
                debug_grad_sum += torch.sum(p.grad.detach().abs())

                p_infinite = not torch.isfinite(p.grad).all()
                p_nan = torch.isnan(p.grad).any()
                if p_infinite or p_nan:
                    bad = True
                    break

        # mark to skip step via a flag
        self._skip_optimizer_step = bad

        # print(f"\n\n{debug_grad_sum=}, {bad=}\n\n")

        if bad:
            # NOTE: this is better than setting gradient to be all zeros.
            # As for optimizer like Adam, gradient of zeros will update the optimizer state.
            opt = self.optimizers()
            if isinstance(opt, (list, tuple)):  # handle multiple opts
                for o in opt:
                    o.zero_grad(set_to_none=True)
            else:
                opt.zero_grad(set_to_none=True)

        self.log(
            "train/has_bad_grad",
            float(bad),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
            # reduce_fx="max",  # PL >= 2.0
        )

        for tmp_name, tmp_v in zip(("debug_param_sum", "debug_grad_sum"), (debug_param_sum, debug_grad_sum)):
            self.log(
                tmp_name,
                tmp_v,
                prog_bar=True,
                logger=True,
                rank_zero_only=True,
                on_epoch=True,
                on_step=True,
                batch_size=1,
            )

    def optimizer_step(self, *args, **kwargs):
        optimizer_closure = kwargs.get("optimizer_closure", None)
        if optimizer_closure is None:
            args = list(args)
            n_callable_args = 0
            for tmp_i, tmp in enumerate(args):
                if callable(tmp):
                    n_callable_args += 1
                    optimizer_closure = tmp
                    # NOTE: optimizer_closure should only be run exactly once!
                    # Thus, if we manually run it, we should make the position argument to be None.
                    args[tmp_i] = None
            assert n_callable_args == 1, f"{n_callable_args=}"
        else:
            # NOTE: optimizer_closure should only be run exactly once!
            # Thus, if we manually run it, we should remove it from the kwargs.
            del kwargs["optimizer_closure"]

        assert optimizer_closure is not None
        closure_result = optimizer_closure()

        # Skip the step if grads were bad
        if getattr(self, "_skip_optimizer_step", False):
            return closure_result

        return super().optimizer_step(*args, **kwargs)
