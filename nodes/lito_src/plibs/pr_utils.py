#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import typing as T

import torch


def pack(
    dense_matrix: torch.Tensor,  # (m, n, *)  any type
    valid_mask: T.Optional[torch.Tensor] = None,  # (m, n)  bool
):
    """
    Convert a dense tensor into pack format by folding the first dimension.

    For example, say we have a tensor of shape (m, n, *),
    after packing, we will get a tensor of shape (n1 + n2 + ... + nm, *),
    where the ni portion corresponds to the i-th row (ie, first dimension)
    in the original dense tensor.

    Args:
        dense_matrix:
            (m, n, *) any type.
        valid_mask:
            (m, n) bool.  This is to remove some entries during the process.
            If None, all true.

    Returns:
        packed_arr:  (p = n1 + n2 + ... + nm, *) float,  the value bank
        start_idxs:  (m,)  long


        idx_arr:  (p = n1 + n2 + ... + nm,) long,  the row index each value in `val_arr` belongs to.
        counts: (m,)  long
    """

    m, n, *d_shape = dense_matrix.shape
    mn = m * n

    if valid_mask is None:
        packed_arr = dense_matrix.reshape(mn, *d_shape)  # (mn, *d)
        start_idxs = torch.arange(m, device=dense_matrix.device) * n  # (m,)
        counts = torch.ones(m, dtype=torch.long, device=dense_matrix.device) * n  # (m,)
        return dict(
            packed_arr=packed_arr,  # (mn, *d)
            start_idxs=start_idxs,  # (m,)
            counts=counts,  # (m,)
        )
    else:
        assert valid_mask.shape == (m, n)
        counts = valid_mask.sum(dim=-1)  # (m,)
        csum = counts.cumsum(dim=0)  # (m,)
        start_idxs = torch.cat([torch.zeros(1, dtype=counts.dtype, device=counts.device), csum[:-1]])  # (m,)

        valid_mask = valid_mask.reshape(mn)  # (mn,)
        packed_arr = dense_matrix.reshape(mn, *d_shape)  # (mn, *)
        packed_arr = packed_arr[valid_mask]  # (n', *)

        return dict(
            packed_arr=packed_arr,  # (n', *d)
            start_idxs=start_idxs,  # (m,)
            counts=counts,  # (m,)
        )
