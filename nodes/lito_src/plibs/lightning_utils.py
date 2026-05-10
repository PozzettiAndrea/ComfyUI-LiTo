#
# Copyright (C) 2026 Apple Inc. All rights reserved.
#
# The file implements util functions for pytorch lightning.

import contextlib
import typing as T

import lightning as L


@contextlib.contextmanager
def local_rank_first(module: L.LightningModule, name: str = "wait_for_downloads"):
    """
    Context manager that forces Local Rank 0 to execute the block first.
    All other ranks will wait until Rank 0 finishes the block.

    This is useful if we want local_rank to download something first to a
    local dir, then other ranks execute after local rank had finished.

    Args:
        module:
            The LightningModule instance (usually `self`).
        name:
            A unique string name for the barrier sync.

    """

    # Safety check: If there's no trainer, we aren't in a distributed run yet.
    # Just yield and move on!
    try:
        # Just accessing it will raise the error if it's not attached
        _ = module.trainer
        has_trainer = True
    except RuntimeError:
        has_trainer = False

    if not has_trainer:
        yield
        return

    # 1. Ranks 1+ stop here and wait. Rank 0 skips this.
    if module.local_rank != 0:
        module.trainer.strategy.barrier(name)

    # 2. Execute the code block inside the 'with' statement
    yield

    # 3. Rank 0 hits this after finishing the block, syncing with Ranks 1+
    if module.local_rank == 0:
        module.trainer.strategy.barrier(name)
