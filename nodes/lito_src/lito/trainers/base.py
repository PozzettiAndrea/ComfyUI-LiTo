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
