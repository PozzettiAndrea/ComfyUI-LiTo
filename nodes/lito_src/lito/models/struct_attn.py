#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements structural attention.
#
# Originally used xformers.ops.memory_efficient_attention with
# xformers BlockDiagonalMask. Rewritten to call flash_attn directly:
#   - plain attention -> flash_attn.flash_attn_func
#   - block-diagonal -> flash_attn.flash_attn_varlen_func with cu_seqlens
# pulled from a BlockDiagonalSeqLens shim (see plibs/flash_utils.py).

import typing as T

import flash_attn
import torch

from plibs.flash_utils import BlockDiagonalSeqLens, create_block_diagonal_attn_bias_from_seq_lens
from contextlib import nullcontext as _nullcontext

# flash attention currently supports only float16 and bfloat16
FLASH_ATTN_DTYPE = torch.bfloat16
# make sure gpu support bf16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    current_device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(current_device)
    # Generally, bfloat16 is supported on GPUs with compute capability 8.0 and higher (Ampere and newer architectures).
    supports_bfloat16 = capability[0] >= 8
    if not supports_bfloat16:
        print(f"bfloat16 not supported: use float16 for flash attention")
        FLASH_ATTN_DTYPE = torch.float16
else:
    print(f"cuda not supported: use float16 for flash attention")
    FLASH_ATTN_DTYPE = torch.float16


def _flash_varlen(
    query: torch.Tensor,  # (1, total_q, h, d) or (total_q, h, d)
    key: torch.Tensor,  # (1, total_kv, h, d) or (total_kv, h, d)
    value: torch.Tensor,  # (1, total_kv, h, dv) or (total_kv, h, dv)
    attn_bias: BlockDiagonalSeqLens,
    p: float,
    scale: T.Optional[float],
    leading_batch_dim: bool,
) -> torch.Tensor:
    """Run flash_attn_varlen_func against a BlockDiagonalSeqLens shim.

    flash_attn varlen wants packed (total, h, d) inputs. xformers expected
    (1, total, h, d) at the call site. ``leading_batch_dim`` controls whether
    we squeeze the leading dim before calling and re-add it on the way out,
    so callers that previously fed (1, ...) tensors do not need to change shape.
    """
    if leading_batch_dim:
        assert query.size(0) == 1 and key.size(0) == 1 and value.size(0) == 1
        q = query.squeeze(0)
        k = key.squeeze(0)
        v = value.squeeze(0)
    else:
        q, k, v = query, key, value

    ori_dtype = v.dtype
    with _nullcontext():
        out = flash_attn.flash_attn_varlen_func(
            q=q.to(dtype=FLASH_ATTN_DTYPE).contiguous(),
            k=k.to(dtype=FLASH_ATTN_DTYPE).contiguous(),
            v=v.to(dtype=FLASH_ATTN_DTYPE).contiguous(),
            cu_seqlens_q=attn_bias.cu_seqlens_q.to(device=q.device, dtype=torch.int32),
            cu_seqlens_k=attn_bias.cu_seqlens_k.to(device=k.device, dtype=torch.int32),
            max_seqlen_q=attn_bias.max_seqlen_q,
            max_seqlen_k=attn_bias.max_seqlen_k,
            dropout_p=p,
            softmax_scale=scale,
        ).to(dtype=ori_dtype)  # (total_q, h, dv)

    if leading_batch_dim:
        out = out.unsqueeze(0)  # (1, total_q, h, dv)
    return out


def construct_structural_attn_given_delta_t(
    input_timestamps: torch.Tensor,
    latent_t0: T.Union[torch.Tensor, float],
    latent_dt: T.Union[torch.Tensor, float],
    num_latents: int,
):
    r"""
    Construct an attention bias that improves the speed of attention and
    causality, etc. The returned attn_bias is a BlockDiagonalSeqLens shim
    that downstream code can pass to flash_attn varlen.

    Given `latent_t_start`, t0, `latent_delta_t`, dt,
    and `num_latents`, n, the latent timestamp
    is calculated as [to + dt, ..., to + n * dt].
    Given the latent timestamps [t1, t2, t3,..., tn] and the
    input timestamps [i1, i2, ...., im], where tj, ij in [0, \inf],
    the latent at tj will attend to all input tokens k,
    whose i_k \in (t_{j-1}, tj].

    Args:
        input_timestamps:
            (b, m)
        latent_t0:
            (b, ) or float
        latent_dt:
            (b, ) or float
        num_latents:
            int,

    Notes:
        1. all out-of-bound inputs will be treated as padding.

        2. ideally, we want to set latent timestamp to be
        [to + 0.5 *dt, to + 1.5 * dt, ...], but let's keep it
        simple for now as the function does not actually need
        latent timestamp

    Examples:
        If input_timestamps is [0 sec, 20 sec] and we want 1 fps,
        set latent_t_start = 0, and latent_delta_t = 1,
        num_latents = 20

    Returns:
        latent_timestamps:
            (b, num_latents)
        structural_attn_dict:
            attn_bias:
                BlockDiagonalSeqLens shim
            sort_idx:
                (b, m) used to sort input tokens
            nonzero_latent_idxs:
                (num_nonzero_latent,) used to remove latents that do not
                attend to any input tokens.
            subseq_count:
                (b, num_latents) number of input tokens will be attended by each latent
            backward_dict:
                latent_idx:
                    (b, m) can be used in write back
            need_to_sort_kv:
                whether we need to sort the input tokens. If we presort it,
                we can set it to False.
    """

    b, m = input_timestamps.shape
    if isinstance(latent_t0, (float, int)):
        latent_t0 = torch.ones(b, dtype=input_timestamps.dtype, device=input_timestamps.device) * latent_t0
    if isinstance(latent_dt, (float, int)):
        latent_dt = torch.ones(b, dtype=input_timestamps.dtype, device=input_timestamps.device) * latent_dt

    assert latent_t0.shape == (b,)
    assert latent_dt.shape == (b,)

    # construct latent time_stamps
    latent_timestamps = (
        1
        + torch.arange(
            num_latents,
            dtype=latent_t0.dtype,
            device=latent_t0.device,
        ).expand(b, -1)
    ) * latent_dt.reshape(b, 1) + latent_t0.reshape(b, 1)  # (b, n)

    # determine latent_idx each input should be assigned to
    latent_idxs = (input_timestamps - latent_t0.reshape(b, 1)) / latent_dt.reshape(b, 1)  # (b, m)
    latent_idxs = torch.floor(latent_idxs).long()  # (b, m)

    # set out of bound to pad_idx
    pad_latent_idx = num_latents
    latent_idxs[latent_idxs < 0] = pad_latent_idx
    latent_idxs[latent_idxs >= num_latents] = pad_latent_idx

    # when we use the attention, we expect latent to be in the shape of (b, n, h, d)
    # and the input tokens of shape (b, m, h, d), where h is number of heads.
    # We will first sort input tokens along dim=1 based on latent_idxs,
    # then we will reshape both input tokens and latents to (1, bn, h, d) and (1, bm, h, d)
    # and we will use block diagonal attention bias to make sure
    # each latent attend to the correct input tokens.
    sorted_latent_idx, sort_idx = torch.sort(latent_idxs, dim=1)  # (b, m)

    # count the subsequence length
    subseq_count = torch.zeros(
        b,
        num_latents,
        dtype=sorted_latent_idx.dtype,
        device=sorted_latent_idx.device,
    )  # (b, num_latent)
    subseq_count.scatter_add_(
        dim=1,
        index=sorted_latent_idx,
        src=torch.ones_like(sorted_latent_idx),
    )  # (b, num_latent)

    # we will remove the latent with no input tokens, so all latent attend to something
    # (a latent attends to nothing causes nan during back-propagation)
    subseq_count = subseq_count.reshape(-1)  # (b * num_latent,)
    nonzero_latent_idxs = (subseq_count > 0).nonzero(as_tuple=True)[0]  # (num_zero_latent,)
    nonzero_subseq_count = subseq_count[nonzero_latent_idxs]  # (num_zero_latent,)

    # The flash_attn varlen kernel has a per-call upper bound on the number of
    # blocks (~65535). In our setting b * num_latent is unlikely to exceed that,
    # so we don't chunk here; if it ever does, raise so the caller can handle it.
    MAX_NUM_BLOCKS = 65535
    if len(nonzero_latent_idxs) > MAX_NUM_BLOCKS:
        raise RuntimeError(
            f"max number of blocks reached, {len(nonzero_latent_idxs)}. Implement chunking."
        )

    # we are going to treat the latent (b, num_latent, h, d) as (1, b * num_latents, h, d)
    attn_bias = create_block_diagonal_attn_bias_from_seq_lens(
        q_seqlen=[1] * (len(nonzero_latent_idxs)),
        kv_seqlen=nonzero_subseq_count.tolist(),
    )

    # backward structure dict (query: input tokens, kv: latent)
    backward_dict = dict(
        latent_idx=latent_idxs,  # (b, m)
        pad_latent_idx=pad_latent_idx,  # int
        q_in_order=True,
    )

    structural_attn_dict = dict(
        attn_bias=attn_bias,
        sort_idx=sort_idx,  # (b, m)  convert input to sorted_x
        nonzero_latent_idxs=nonzero_latent_idxs,  # (num_nonzero_latent,)
        subseq_count=subseq_count.reshape(b, num_latents),  # (b, num_latent)
        backward_dict=backward_dict,  # (b, m)  no need to sort to use it
        kv_in_order=False,
    )

    return dict(
        structural_attn_dict=structural_attn_dict,
        latent_timestamps=latent_timestamps,
    )


def structural_memory_efficient_attention(
    query: torch.Tensor,  # (b, n, h, d)
    key: torch.Tensor,  # (b, m, h, d)
    value: torch.Tensor,  # (b, m, h, dv)
    p: float = 0.0,
    scale: T.Optional[float] = None,
    structural_attn_dict: T.Dict[str, T.Any] = None,
) -> torch.Tensor:
    """
    Structural attention that supports various types of structures.

    Args:
        query:
            (b, n, h, d)
        key:
            (b, m, h, d)
        value:
            (b, m, h, dv)  to use flash attention, dv should be equal to d
        p:
            dropout probability
        scale:
            Scaling factor for Q @ K.transpose(). If set to None,
            the default scale (q.shape[-1]**-0.5) will be used.
        structural_attn_dict:
            dict containing the structural mode and information.
            If None, typical attention is used (flash_attn.flash_attn_func on CUDA,
            torch SDPA elsewhere).

            mode = 'flash_varlen'  (was 'xops' before the xformers->flash rewrite)
                attn_bias: BlockDiagonalSeqLens shim carrying cu_seqlens / max_seqlen
                The (b=1, total, h, d) shape from before is preserved at the call
                site; we squeeze and pass packed (total, h, d) to flash internally.

            mode = 'sort_kv_per_b'
                sort the key and value along the sequence dimension based on the given sort_idx

            mode = 'sorted_kv_per_b'
                the input key and value are already sorted along the sequence dimension based on the given sort_idx

            mode = 'sort_qkv_with_b'
                sort the query, key and value by first reshape to (1, b*m, h, d) then sort based on the given sort_idx

            mode = 'sorted_qkv_with_b'
                first reshape query, key and value to (1, b*m, h, d) then,
                they are already sorted based on the given sort_idx

            mode = 'pointwise'
                each query will only attend to one key, so no need to run attention, simple do an index_select


    Returns:
        attention output:
            (b, n, h, dv=d)
    """

    if structural_attn_dict is None:
        if query.is_cuda:
            with _nullcontext():
                ori_dtype = value.dtype
                out = flash_attn.flash_attn_func(
                    query.to(dtype=FLASH_ATTN_DTYPE).contiguous(),
                    key.to(dtype=FLASH_ATTN_DTYPE).contiguous(),
                    value.to(dtype=FLASH_ATTN_DTYPE).contiguous(),
                    dropout_p=p,
                    softmax_scale=scale,
                ).to(dtype=ori_dtype)  # (b, n, h, dim_head)
            return out
        # Fallback: PyTorch native SDPA (CPU / MPS).
        # (b, n, h, d) -> SDPA layout (b, h, n, d)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=scale,
            dropout_p=p,
        )
        return out.transpose(1, 2)

    # structural attention
    mode = structural_attn_dict["mode"]

    if mode == "flash_varlen":
        assert "attn_bias" in structural_attn_dict
        return _flash_varlen(
            query=query,  # (1, total_q, h, d)
            key=key,
            value=value,
            attn_bias=structural_attn_dict["attn_bias"],
            p=p,
            scale=scale,
            leading_batch_dim=True,
        )
    elif mode in {
        "sort_kv_per_b",
        "sorted_kv_per_b",
        "sort_qkv_with_b",
        "sorted_qkv_with_b",
    }:
        return _structural_attn(
            query=query,  # (b, n, h, d)
            key=key,  # (b, m, h, d)
            value=value,  # (b, m, h, dv=d)
            p=p,
            scale=scale,
            structural_attn_dict=structural_attn_dict,
        )
    elif mode == "pointwise":
        return _structural_attn_pointwise(
            query=query,  # (b, n, h, d)
            key=key,  # (b, m, h, d)
            value=value,  # (b, m, h, dv=d)
            p=p,
            scale=scale,
            structural_attn_dict=structural_attn_dict,
        )
    else:
        raise NotImplementedError(mode)


def _structural_attn(
    query: torch.Tensor,  # (b, n, h, d)
    key: torch.Tensor,  # (b, m, h, d)
    value: torch.Tensor,  # (b, m, h, dv=d)
    p: float = 0.0,
    scale: T.Optional[float] = None,
    structural_attn_dict: T.Dict[str, T.Any] = None,
) -> torch.Tensor:
    r"""
    Structural attention where each query attends to a non-overlapping
    region of the keys.

    Args:
        query:
            (b, n, h, d)
        key:
            (b, m, h, d)
        value:
            (b, m, h, dv)  to use flash attention, dv should be equal to d
        p:
            dropout probability
        scale:
            Scaling factor for Q @ K.transpose(). If set to None,
            the default scale (q.shape[-1]**-0.5) will be used.

        structural_attn_dict:
            mode:
                'sort_kv_per_b': we need to sort the key and value along
                    the sequence dimension based on the given sort_idx
                'sorted_kv_per_b': the input key and value are already sorted
                    along the sequence dimension based on the given sort_idx
                'sort_qkv_with_b': we need to sort the query, key and value
                    after first reshape to (1, b*m, h, d) then sort based on the given sort_idx
                'sorted_qkv_with_b': the input query, key and value are already sorted.
                    all we need is to reshape

            attn_biases:
                list of (num_chunks, ) BlockDiagonalSeqLens shims
                (was xformers BlockDiagonalMask before the rewrite).

            q_start_idxs:
                list of int, (num_chunks + 1,).
                start index of chunk after sorted, reshaped, and removed redundant queries

            kv_start_idxs:
                list of int, (num_chunks + 1,).
                start index of chunk after sorted, reshaped, and removed redundant key and value

            kv_seq_sort_idxs (optinal):
                (b, m) or None. If mode='sort_kv_per_b', used to sort key and value along m (seq dimension)

            kv_bseq_sort_idxs (optinal):
                (bm,) or None. If mode='sort_kv_with_b', used to sort key and value along (b*m,)

            q_bseq_sort_idx (optional):
                (bn,) or None. If mode='sort_kv_with_b', used to sort query along (b*n,)

            nonzero_query_idxs:
                (num_nonzero_query,) after reshape everything to (1, b*seq, h, d),
                it is used to remove queries that will not attend to any keys.
                These queries cause nan during backward.

            num_kv_padded (optional):
                int, after kv reshaped to (1, b*m, h, d), number of redundant kv at the end

    Returns:
        attention output:
            (b, n, h, dv)
    """

    assert structural_attn_dict.get("attn_biases", None) is not None  # (num_chunks, )
    assert structural_attn_dict.get("q_start_idxs", None) is not None  # (num_chunks + 1, )
    assert structural_attn_dict.get("kv_start_idxs", None) is not None  # (num_chunks + 1, )
    mode = structural_attn_dict["mode"]

    b, n, h, d = query.shape
    _b, m, _h, d = key.shape
    _b, _m, _h, dv = value.shape

    # reshape and sort, or sort then reshape, depending on mode
    # regardless, the resulted query, key, value should have shape (b*seq, h, d)
    if mode == "sort_kv_per_b":
        # sort kv along the sequence dimension
        assert structural_attn_dict.get("kv_seq_sort_idxs", None) is not None
        kv_sort_idx = structural_attn_dict["kv_seq_sort_idxs"]  # (b, m)
        key = torch.gather(
            key,  # (b, m, h, d)
            dim=1,
            index=kv_sort_idx.reshape(b, m, 1, 1).expand(-1, -1, h, d),
        )  # (b, m, h, d)
        value = torch.gather(
            value,  # (b, m, h, dv)
            dim=1,
            index=kv_sort_idx.reshape(b, m, 1, 1).expand(-1, -1, h, dv),
        )  # (b, m, h, dv)

        # reshape
        query = query.reshape(b * n, h, d)  # (bn, h, d)
        key = key.reshape(b * m, h, d)  # (b * m, h, d)
        value = value.reshape(b * m, h, dv)  # (b * m, h, dv)

    elif mode == "sorted_kv_per_b":
        # reshape
        query = query.reshape(b * n, h, d)  # (bn, h, d)
        key = key.reshape(b * m, h, d)  # (b * m, h, d)
        value = value.reshape(b * m, h, dv)  # (b * m, h, dv)

    elif mode == "sort_qkv_with_b":
        # we first reshape to (1, b*seq, h, d), then we sort
        assert structural_attn_dict.get("q_bseq_sort_idx", None) is not None
        assert structural_attn_dict.get("kv_bseq_sort_idx", None) is not None
        q_sort_idx = structural_attn_dict["q_bseq_sort_idx"]  # (bn,)
        kv_sort_idx = structural_attn_dict["kv_bseq_sort_idx"]  # (bm,)

        # sort query (b, n, h, d) -> (b * n, h, d)
        query = query.reshape(b * n, h, d)  # (bn, h, d)
        query = query[q_sort_idx]  # (bn, h, d)

        # sort key (b, m, h, d) -> (1, b * m, h, d)
        key = key.reshape(b * m, h, d)  # (bm, h, d)
        key = key[kv_sort_idx]  # (bm, h, d)

        # sort value (b, m, h, dv) -> (1, b * m, h, dv)
        value = value.reshape(b * m, h, dv)  # (bm, h, dv)
        value = value[kv_sort_idx]  # (bm, h, dv)

    elif mode == "sorted_qkv_with_b":
        # we only need to reshape to (1, b*seq, h, d)
        query = query.reshape(b * n, h, d)  # (bn, h, d)
        key = key.reshape(b * m, h, d)  # (bm, h, d)
        value = value.reshape(b * m, h, dv)  # (bm, h, dv)

    else:
        raise NotImplementedError(f"{mode} not implemented")

    # remove redundant query and keys
    assert query.shape == (b * n, h, d)
    assert key.shape == (b * m, h, d)
    assert value.shape == (b * m, h, dv)

    nonzero_query_idxs = structural_attn_dict.get("nonzero_query_idxs", None)  # (num_nonzero_query,)
    if nonzero_query_idxs is not None:
        query = query[nonzero_query_idxs]  # (q', h, d)

    num_kv_padded: int = structural_attn_dict.get("num_kv_padded", None)  # (,)
    if num_kv_padded is not None:
        num_valid = b * m - num_kv_padded
        key = key[:num_valid]  # (k', h, d)
        value = value[:num_valid]  # (k', h, dv)

    # attention via flash varlen
    attn_biases: T.List[BlockDiagonalSeqLens] = structural_attn_dict["attn_biases"]  # (num_chunks,)
    q_start_idxs: T.List[int] = structural_attn_dict["q_start_idxs"]  # (num_chunks + 1, )
    kv_start_idxs: T.List[int] = structural_attn_dict["kv_start_idxs"]  # (num_chunks + 1, )
    num_chunks = len(attn_biases)

    # go through each chunk
    outs = []
    for chunk_idx in range(num_chunks):
        attn_bias = attn_biases[chunk_idx]
        if len(attn_bias.q_seqinfo.seqstart) > 2**16:
            raise RuntimeError(f"attn_bias.q: nblock={len(attn_bias.q_seqinfo.seqstart)}")
        if len(attn_bias.k_seqinfo.seqstart) > 2**16:
            raise RuntimeError(f"attn_bias.k: nblock={len(attn_bias.k_seqinfo.seqstart)}")

        out = _flash_varlen(
            query=query[q_start_idxs[chunk_idx] : q_start_idxs[chunk_idx + 1]],  # (chunk_q, h, d)
            key=key[kv_start_idxs[chunk_idx] : kv_start_idxs[chunk_idx + 1]],  # (chunk_kv, h, d)
            value=value[kv_start_idxs[chunk_idx] : kv_start_idxs[chunk_idx + 1]],  # (chunk_kv, h, dv)
            attn_bias=attn_bias,
            p=p,
            scale=scale,
            leading_batch_dim=False,
        )  # (chunk_q, h, dv)
        outs.append(out)
    out = torch.cat(outs, dim=0)  # (q', h, dv)

    # we need to inflate the result to the original shape with zero
    if nonzero_query_idxs is not None:
        _attn_out = torch.zeros(
            b * n,
            h,
            dv,
            dtype=out.dtype,
            device=out.device,
        )  # (bn, h, dv)
        _attn_out[nonzero_query_idxs] = out
        out = _attn_out.reshape(b, n, h, dv)  # (b, n, h, dv)
    else:
        out = out.reshape(b, n, h, dv)  # (b, n, h, dv)

    return out


def _structural_attn_pointwise(
    query: torch.Tensor,  # (b, n, h, d)
    key: torch.Tensor,  # (b, m, h, d)
    value: torch.Tensor,  # (b, m, h, dv=d)
    p: float = 0.0,
    scale: T.Optional[float] = None,
    structural_attn_dict: T.Dict[str, T.Any] = None,
) -> torch.Tensor:
    r"""
    Structural attention where each query attends to a non-overlapping
    region of the keys.

    Args:
        query:
            (b, n, h, d)
        key:
            (b, m, h, d)
        value:
            (b, m, h, dv)  to use flash attention, dv should be equal to d
        p:
            dropout probability
        scale:
            Scaling factor for Q @ K.transpose(). If set to None,
            the default scale (q.shape[-1]**-0.5) will be used.

        structural_attn_dict:
            mode = 'pointwise'
                each query will only attend to one key, so no need to run attention, simple do an index_select

            valid_query_mask (optional):
                (b, n) bool, whether the query actually attend to at least a key

            kv_idxs:
                (b, n) long, the key index along each b (ie, midx) attended by each query

    Returns:
        attention output:
           (b, n, h, dv)
    """

    mode = structural_attn_dict["mode"]
    assert mode == "pointwise"

    b, n, h, d = query.shape
    _b, m, _h, d = key.shape
    _b, _m, _h, dv = value.shape

    kv_idxs = structural_attn_dict.get("kv_idxs", None)  # (b, n)
    assert kv_idxs is not None

    # gather
    value = value.reshape(b * m, h, dv)  # (b*m, h, dv)
    out = value[kv_idxs.reshape(-1)].reshape(b, n, h, dv)  # (b, n, h, dv)

    # set the out to be zero if valid_query_mask is False
    valid_query_mask = structural_attn_dict.get("valid_query_mask", None)  # (b, n)
    if valid_query_mask is not None:
        out[~valid_query_mask] = 0

    return out
