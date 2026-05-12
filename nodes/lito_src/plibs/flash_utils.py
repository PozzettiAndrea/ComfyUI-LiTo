#
# Copyright (C) 2025 Apple Inc. All rights reserved.
# Modified to drop xformers dependency in favor of flash_attn varlen.
#
# Originally implemented helpers for constructing xformers BlockDiagonalMask
# attention biases. Rewritten to expose the cu_seqlens / max_seqlen the
# flash_attn varlen kernels consume directly, while keeping the same
# .q_seqinfo / .k_seqinfo attribute surface that downstream callers already
# read (e.g. plibs/ppoint.py reads attn_bias.q_seqinfo.seqstart_py[-1]).
#

import typing as T
from dataclasses import dataclass

import torch


@dataclass
class SeqLenInfo:
    """Mirror of xformers.ops.fmha.attn_bias._SeqLenInfo's read surface."""

    max_seqlen: int
    min_seqlen: int
    seqstart_py: T.List[int]
    seqstart: torch.Tensor  # int32 cumulative seqlens, length num_blocks+1, starts at 0


@dataclass
class BlockDiagonalSeqLens:
    """Drop-in replacement for xformers.ops.fmha.attn_bias.BlockDiagonalMask.

    Carries enough information to dispatch to flash_attn.flash_attn_varlen_func
    via the cu_seqlens_q / cu_seqlens_k / max_seqlen_q / max_seqlen_k properties.
    """

    q_seqinfo: SeqLenInfo
    k_seqinfo: SeqLenInfo

    @property
    def cu_seqlens_q(self) -> torch.Tensor:
        return self.q_seqinfo.seqstart

    @property
    def cu_seqlens_k(self) -> torch.Tensor:
        return self.k_seqinfo.seqstart

    @property
    def max_seqlen_q(self) -> int:
        return self.q_seqinfo.max_seqlen

    @property
    def max_seqlen_k(self) -> int:
        return self.k_seqinfo.max_seqlen


def get_seqstart(
    seqlens: T.Union[T.List[int], torch.Tensor],
) -> T.Tuple[int, int, T.List[int], torch.Tensor]:
    """Compute (min_seqlen, max_seqlen, seqstart_py, seqstart) from seqlens.

    seqstart is a length-(N+1) int32 tensor of cumulative offsets starting at 0,
    matching the cu_seqlens contract of flash_attn varlen kernels.
    """

    if isinstance(seqlens, (tuple, list)):
        if len(seqlens) == 0:
            import comfy.model_management as _comfy_mm
            device = _comfy_mm.get_torch_device()
            zero = torch.zeros(1, dtype=torch.int32, device=device)
            return 0, 0, [0], zero
        import comfy.model_management as _comfy_mm
        device = _comfy_mm.get_torch_device()
        seqlens_tensor = torch.tensor(seqlens, dtype=torch.int32, device=device)
        max_seqlen = int(max(seqlens))
        min_seqlen = int(min(seqlens))
        seqstart_py = [0]
        running = 0
        for s in seqlens:
            running += int(s)
            seqstart_py.append(running)
        seqstart = torch.tensor(seqstart_py, dtype=torch.int32, device=device)
        return min_seqlen, max_seqlen, seqstart_py, seqstart

    assert isinstance(seqlens, torch.Tensor)

    min_seqlen = int(seqlens.min().item())
    max_seqlen = int(seqlens.max().item())
    seqstart = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=seqlens.device),
            torch.cumsum(seqlens, dim=0, dtype=torch.int32),
        ],
        dim=0,
    )
    seqstart_py = seqstart.tolist()
    return min_seqlen, max_seqlen, seqstart_py, seqstart


def create_block_diagonal_attn_bias_from_seq_lens(
    q_seqlen: T.Union[T.List[int], torch.Tensor],
    kv_seqlen: T.Union[T.List[int], torch.Tensor] = None,
) -> BlockDiagonalSeqLens:
    """Build a BlockDiagonalSeqLens carrying cu_seqlens for flash_attn varlen.

    Same call surface as the previous xformers-based helper. Returns a
    BlockDiagonalSeqLens, which exposes q_seqinfo / k_seqinfo attributes
    plus convenience cu_seqlens_q / max_seqlen_q properties consumed by
    the flash_attn varlen call sites in lito.models.struct_attn.
    """

    assert kv_seqlen is None or len(q_seqlen) == len(kv_seqlen)
    min_seqlen, max_seqlen, seqstart_py, seqstart = get_seqstart(seqlens=q_seqlen)
    q_seqinfo = SeqLenInfo(
        max_seqlen=max_seqlen,
        min_seqlen=min_seqlen,
        seqstart_py=seqstart_py,
        seqstart=seqstart,
    )
    if (kv_seqlen is None) or (q_seqlen is kv_seqlen):
        k_seqinfo = q_seqinfo
    else:
        k_min, k_max, k_seqstart_py, k_seqstart = get_seqstart(seqlens=kv_seqlen)
        k_seqinfo = SeqLenInfo(
            max_seqlen=k_max,
            min_seqlen=k_min,
            seqstart_py=k_seqstart_py,
            seqstart=k_seqstart,
        )
    return BlockDiagonalSeqLens(q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo)
