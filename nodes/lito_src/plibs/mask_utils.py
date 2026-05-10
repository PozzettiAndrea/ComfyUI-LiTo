#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#

import typing as T

import numpy as np

import torch


def get_valid_mask(
    valid_lens: T.Union[T.Sequence[int], torch.LongTensor, np.ndarray],
    max_len: int = None,
    device=torch.device("cpu"),
    invalid=False,
) -> torch.BoolTensor:
    """
    Returns a BoolTensor B, where B[i,j] = True if j < valid_lens[i].

    Args:
        valid_lens: (batch,)
        max_len: int, max length of the mask. If None, use max(valid_len)
        device: the device of the output mask.
        invalid: invert the result

    Returns:
         boolTensor (batch, max_len)
    """
    if max_len is None:
        max_len = torch.max(valid_lens)

    if isinstance(valid_lens, torch.Tensor):
        pass
    elif isinstance(valid_lens, (list, tuple)):
        valid_lens = torch.tensor(valid_lens, dtype=torch.long, device=device)
    elif isinstance(valid_lens, np.ndarray):
        valid_lens = torch.from_numpy(valid_lens).to(dtype=torch.long, device=device)
    else:
        raise NotImplementedError
    idxs = torch.arange(0, max_len, device=valid_lens.device)  # (max_len,)
    if not invalid:
        mask = idxs < valid_lens.unsqueeze(1)
    else:
        mask = idxs >= valid_lens.unsqueeze(1)
    return mask.to(device=device)
