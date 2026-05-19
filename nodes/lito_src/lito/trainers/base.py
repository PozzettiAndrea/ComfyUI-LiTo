#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# Originally a pytorch-lightning LightningModule. Rebased on torch.nn.Module
# for the ComfyUI inference wrapper — we don't train, and Lightning's
# read-only `device` @property collides with comfy.model_patcher.ModelPatcher
# (which writes `self.model.device = ...`). Training hooks
# (on_train_epoch_start, on_save_checkpoint, on_load_checkpoint, on_fit_start)
# and SkipGradNaNTrainer are dropped — none are reachable from inference.

import torch
import torch.nn as nn


class BaseTrainer(nn.Module):
    def __init__(self):
        super().__init__()

    def freeze(self) -> None:
        """nn.Module equivalent of Lightning's freeze(): turn off autograd
        on every parameter and switch to eval mode."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def save_hyperparameters(self, *args, **kwargs) -> None:
        """No-op stand-in for Lightning's save_hyperparameters(). The two
        trainer __init__s call this; nothing in the inference codebase
        reads `self.hparams` back, so we don't store anything."""
        pass

    @property
    def device(self) -> torch.device:
        """Read-write `device` replacing Lightning's read-only @property.

        Reader: prefer the override Comfy's ModelPatcher stashed via the
        setter; fall back to the device of the first parameter (Lightning's
        original behavior). The fallback matters for callsites like
        lito_trainer.py: `self.voxel_ss_pipeline.device != self.device`.

        Writer: comfy.model_patcher.ModelPatcher writes self.model.device
        in load() / unpatch_model() (model_patcher.py:936, :990). Stash
        the value so Comfy and downstream code see the same thing.
        """
        v = self.__dict__.get("_lito_device_override")
        if v is not None:
            return v if isinstance(v, torch.device) else torch.device(v)
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @device.setter
    def device(self, value) -> None:
        self.__dict__["_lito_device_override"] = value
