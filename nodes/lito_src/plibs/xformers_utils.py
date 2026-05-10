#
# Copyright (C) 2025 Apple Inc. All rights reserved.
#
# The file implements utility functions for using xformers.

import typing as T

try:
    import xformers.ops.fmha.attn_bias
except ImportError:
    print("xformers.ops.fmha.attn_bias not imported.")

import torch


def get_seqstart(
    seqlens: T.Union[T.List[int], torch.Tensor],
) -> T.Tuple[int, int, T.List[int], torch.Tensor]:
    """
    Given sequence lengths, returns the min/max value and the sequence start
    positions (offsets), with first element being 0 (returned in list and Tensor).

    Args:
        seqlens:
            (b,) sequence lengths

    Returns:
        min_seqlen: int
        max_seqlen: int
        seqstart_py:
            (b+1,) list of int
        seqstart:
            (b+1,) int32
    """

    if isinstance(seqlens, (tuple, list)):
        return xformers.ops.fmha.attn_bias._SeqLenInfo._get_seqstart(
            seqlens,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        )

    assert isinstance(seqlens, torch.Tensor)

    min_seqlen = seqlens.min().item()
    max_seqlen = seqlens.max().item()
    seqstart = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=seqlens.device),
            torch.cumsum(seqlens, dim=0, dtype=torch.int32),
        ],
        dim=0,
    )  # (b+1,)
    seqstart_py = seqstart.tolist()

    return (min_seqlen, max_seqlen, seqstart_py, seqstart)


def create_block_diagonal_attn_bias_from_seq_lens(
    q_seqlen: T.Union[T.List[int], torch.Tensor],
    kv_seqlen: T.Union[T.List[int], torch.Tensor] = None,
) -> "xformers.ops.fmha.attn_bias.BlockDiagonalMask":
    """
    Get the BlockDiagonalMask attention bias from
    sequence lengths of q and k.

    Args:
        q_seqlen:
            (b,) list or tensor
        kv_seqlen:
            (b,)  list or tensor

    Returns:
        BlockDiagonalMask
    """

    assert kv_seqlen is None or len(q_seqlen) == len(kv_seqlen)
    min_seqlen, max_seqlen, seqstart_py, seqstart = get_seqstart(
        seqlens=q_seqlen,
    )
    q_seqinfo = xformers.ops.fmha.attn_bias._SeqLenInfo(
        max_seqlen=max_seqlen,
        min_seqlen=min_seqlen,
        seqstart=seqstart,
        seqstart_py=seqstart_py,
    )
    if (kv_seqlen is None) or (q_seqlen is kv_seqlen):
        k_seqinfo = q_seqinfo
    else:
        min_seqlen, max_seqlen, seqstart_py, seqstart = get_seqstart(
            seqlens=kv_seqlen,
        )
        k_seqinfo = xformers.ops.fmha.attn_bias._SeqLenInfo(
            max_seqlen=max_seqlen,
            min_seqlen=min_seqlen,
            seqstart=seqstart,
            seqstart_py=seqstart_py,
        )
    return xformers.ops.fmha.attn_bias.BlockDiagonalMask(q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo)
