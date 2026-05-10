# Copyright (C) 2026 Apple Inc. All rights reserved.
#
# MLX inference-only port of the PyTorch GaussianDecoderXv and its
# supporting attention / MLP layers.  Mirrors:
#   - src/lito/models/spoint_encoder.py   (SPointCross/SelfAttentionLayer, PerceiverEncoder)
#   - src/lito/models/point_decoder.py    (GaussianDecoderXv, get_output_mlp)

import typing as T

from mlx import nn
import mlx.core as mx

from lito.mlx.models.layers import FinalLayer, FourierEmbed, Mlp, RMSNorm

# ---------------------------------------------------------------------------
# Utility helpers for packed <-> batched conversion
# ---------------------------------------------------------------------------


def packed_to_batched(
    packed: mx.array,
    seq_lens: T.List[int],
) -> T.Tuple[mx.array, mx.array]:
    """Pad a packed tensor to a batched tensor with a boolean mask.

    Args:
        packed: Packed input. (sum(seq_lens), ...)
        seq_lens: Per-sample sequence lengths.

    Returns:
        batched: Padded tensor. (b, max_len, ...)
        mask: Boolean validity mask. (b, max_len)
    """
    b = len(seq_lens)
    max_len = max(seq_lens)
    trailing = packed.shape[1:]  # e.g. (h, dh)
    batched = mx.zeros((b, max_len, *trailing), dtype=packed.dtype)  # (b, max_len, ...)
    mask = mx.zeros((b, max_len), dtype=mx.bool_)  # (b, max_len)

    offset = 0
    for i, sl in enumerate(seq_lens):
        batched[i, :sl] = packed[offset : offset + sl]  # (sl, ...)
        mask[i, :sl] = True
        offset += sl

    return batched, mask  # (b, max_len, ...), (b, max_len)


def batched_to_packed(
    batched: mx.array,
    seq_lens: T.List[int],
) -> mx.array:
    """Remove padding from a batched tensor and concatenate valid entries.

    Args:
        batched: Padded tensor. (b, max_len, ...)
        seq_lens: Per-sample sequence lengths.

    Returns:
        Packed tensor. (sum(seq_lens), ...)
    """
    parts = [batched[i, :sl] for i, sl in enumerate(seq_lens)]  # list of (sl, ...)
    return mx.concatenate(parts, axis=0)  # (sum(seq_lens), ...)


# ---------------------------------------------------------------------------
# MLXSwiGLU  -- xformers-style SwiGLU
# ---------------------------------------------------------------------------


class MLXSwiGLU(nn.Module):
    """xformers-style SwiGLU feed-forward block.

    ``forward: w3(silu(w1(x)) * w2(x))``

    Where w1 is the value branch, w2 is the gate branch, and w3 is the
    output projection.

    Args:
        in_features: Input dimension.
        hidden_features: Hidden dimension.
        out_features: Output dimension. If ``None``, defaults to ``in_features``.
        bias: Whether to include bias in all three linear layers.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: T.Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor. (*, in_features)

        Returns:
            Output tensor. (*, out_features)
        """
        return self.w3(nn.silu(self.w1(x)) * self.w2(x))  # (*, out_features)


# ---------------------------------------------------------------------------
# MLXSPointCrossAttentionLayer
# ---------------------------------------------------------------------------


class MLXSPointCrossAttentionLayer(nn.Module):
    """Global cross-attention for packed point-cloud data.

    Pre-layernorm on packed query and key-value, linear projections,
    optional RMSNorm on q/k, then SDPA in batched (padded) form.

    Args:
        dim_q: Input dimension of query tokens.
        dim_kv: Input dimension of key-value tokens.
        dim_qkv: Internal qkv dimension (must be divisible by ``num_heads``).
        num_heads: Number of attention heads.
        use_rmsnorm: Apply RMSNorm to projected q and k.
        add_bias: Include bias in linear projections.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        dim_qkv: int,
        num_heads: int = 8,
        use_rmsnorm: bool = True,
        add_bias: bool = True,
    ):
        super().__init__()
        assert dim_qkv % num_heads == 0, f"{dim_qkv}, {num_heads}"
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.dim_qkv = dim_qkv
        self.num_heads = num_heads
        self.use_rmsnorm = use_rmsnorm
        self.dim_head = dim_qkv // num_heads

        self.layernorm_q = nn.LayerNorm(dim_q)
        self.layernorm_kv = nn.LayerNorm(dim_kv)

        self.linear_q = nn.Linear(dim_q, dim_qkv, bias=add_bias)
        self.linear_kv = nn.Linear(dim_kv, 2 * dim_qkv, bias=add_bias)
        self.linear_out = nn.Linear(dim_qkv, dim_q, bias=add_bias)

        if self.use_rmsnorm:
            self.rmsnorm_q = RMSNorm(dim_qkv)
            self.rmsnorm_k = RMSNorm(dim_qkv)

    def __call__(
        self,
        query: mx.array,
        key_value: mx.array,
        q_seq_lens: T.List[int],
        kv_seq_lens: T.List[int],
    ) -> mx.array:
        """Forward pass.

        Args:
            query: Packed query tokens. (bn, dim_q)
            key_value: Packed key-value tokens. (bm, dim_kv)
            q_seq_lens: Per-sample query sequence lengths.
            kv_seq_lens: Per-sample key-value sequence lengths.

        Returns:
            Output tokens. (bn, dim_q)
        """
        bn = query.shape[0]

        # Pre-layernorm
        query = self.layernorm_q(query)  # (bn, dim_q)
        key_value = self.layernorm_kv(key_value)  # (bm, dim_kv)

        # Linear projections
        query = self.linear_q(query)  # (bn, dim_qkv)
        kv = self.linear_kv(key_value)  # (bm, 2 * dim_qkv)
        key, value = mx.split(kv, 2, axis=-1)  # each (bm, dim_qkv)

        # Optional RMSNorm on q and k
        if self.use_rmsnorm:
            query = self.rmsnorm_q(query)  # (bn, dim_qkv)
            key = self.rmsnorm_k(key)  # (bm, dim_qkv)

        # Reshape to multihead
        query = query.reshape(bn, self.num_heads, self.dim_head)  # (bn, h, dh)
        bm = key.shape[0]
        key = key.reshape(bm, self.num_heads, self.dim_head)  # (bm, h, dh)
        value = value.reshape(bm, self.num_heads, self.dim_head)  # (bm, h, dh)

        # Pad to batched tensors
        q_batched, _q_mask = packed_to_batched(query, q_seq_lens)  # (b, max_q, h, dh), (b, max_q)
        k_batched, kv_mask = packed_to_batched(key, kv_seq_lens)  # (b, max_kv, h, dh), (b, max_kv)
        v_batched, _ = packed_to_batched(value, kv_seq_lens)  # (b, max_kv, h, dh)

        # Transpose to (b, h, seq, dh) for SDPA
        q_batched = mx.transpose(q_batched, axes=(0, 2, 1, 3))  # (b, h, max_q, dh)
        k_batched = mx.transpose(k_batched, axes=(0, 2, 1, 3))  # (b, h, max_kv, dh)
        v_batched = mx.transpose(v_batched, axes=(0, 2, 1, 3))  # (b, h, max_kv, dh)

        # KV boolean mask: True for valid positions -> (b, 1, 1, max_kv)
        kv_mask_sdpa = kv_mask[:, None, None, :]  # (b, 1, 1, max_kv)

        out = mx.fast.scaled_dot_product_attention(
            q_batched, k_batched, v_batched, scale=self.dim_head**-0.5, mask=kv_mask_sdpa
        )  # (b, h, max_q, dh)

        # Transpose back and unpad
        out = mx.transpose(out, axes=(0, 2, 1, 3))  # (b, max_q, h, dh)
        out = out.reshape(out.shape[0], out.shape[1], self.dim_qkv)  # (b, max_q, dim_qkv)
        out = batched_to_packed(out, q_seq_lens)  # (bn, dim_qkv)

        out = self.linear_out(out)  # (bn, dim_q)
        return out


# ---------------------------------------------------------------------------
# MLXSPointSelfAttentionLayer
# ---------------------------------------------------------------------------


class MLXSPointSelfAttentionLayer(nn.Module):
    """Self-attention supporting global and localized_voxel modes for packed data.

    Args:
        dim_in: Input/output feature dimension.
        dim_qkv: Internal qkv dimension (must be divisible by ``num_heads``).
        self_attn_type: ``"global"`` or ``"localized_voxel"``.
        num_heads: Number of attention heads.
        use_rmsnorm: Apply RMSNorm to projected q and k.
        add_bias: Include bias in linear projections.
    """

    def __init__(
        self,
        dim_in: int,
        dim_qkv: int,
        self_attn_type: str,
        num_heads: int = 8,
        use_rmsnorm: bool = True,
        add_bias: bool = True,
    ):
        super().__init__()
        assert dim_qkv % num_heads == 0, f"{dim_qkv}, {num_heads}"
        self.dim_in = dim_in
        self.dim_qkv = dim_qkv
        self.self_attn_type = self_attn_type
        self.num_heads = num_heads
        self.use_rmsnorm = use_rmsnorm
        self.dim_head = dim_qkv // num_heads

        self.linear_qkv = nn.Linear(dim_in, 3 * dim_qkv, bias=add_bias)
        self.linear_out = nn.Linear(dim_qkv, dim_in, bias=add_bias)

        if self.use_rmsnorm:
            self.rmsnorm_q = RMSNorm(dim_qkv)
            self.rmsnorm_k = RMSNorm(dim_qkv)

    def _global_attention(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        seq_lens: T.List[int],
    ) -> mx.array:
        """Global self-attention via padded batching.

        Args:
            q: Packed queries. (bn, h, dh)
            k: Packed keys. (bn, h, dh)
            v: Packed values. (bn, h, dh)
            seq_lens: Per-sample lengths.

        Returns:
            Attention output. (bn, h, dh)
        """
        q_batched, mask = packed_to_batched(q, seq_lens)  # (b, max_len, h, dh), (b, max_len)
        k_batched, _ = packed_to_batched(k, seq_lens)  # (b, max_len, h, dh)
        v_batched, _ = packed_to_batched(v, seq_lens)  # (b, max_len, h, dh)

        # Transpose to (b, h, max_len, dh)
        q_batched = mx.transpose(q_batched, axes=(0, 2, 1, 3))  # (b, h, max_len, dh)
        k_batched = mx.transpose(k_batched, axes=(0, 2, 1, 3))  # (b, h, max_len, dh)
        v_batched = mx.transpose(v_batched, axes=(0, 2, 1, 3))  # (b, h, max_len, dh)

        mask_sdpa = mask[:, None, None, :]  # (b, 1, 1, max_len)

        out = mx.fast.scaled_dot_product_attention(
            q_batched, k_batched, v_batched, scale=self.dim_head**-0.5, mask=mask_sdpa
        )  # (b, h, max_len, dh)

        out = mx.transpose(out, axes=(0, 2, 1, 3))  # (b, max_len, h, dh)
        out = batched_to_packed(out, seq_lens)  # (bn, h, dh)
        return out

    def _voxel_attention(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        voxel_info: T.Dict[str, T.Any],
    ) -> mx.array:
        """Localized voxel-windowed self-attention.

        Args:
            q: Packed queries. (bn, h, dh)
            k: Packed keys. (bn, h, dh)
            v: Packed values. (bn, h, dh)
            voxel_info: Dict with keys ``forward_idxs``, ``backward_idxs``,
                ``cu_seq_lens``, ``max_seq_lens``, ``chunk_start_idxs``.

        Returns:
            Attention output. (bn, h, dh)
        """
        forward_idxs: mx.array = voxel_info["forward_idxs"]  # (bn,)
        backward_idxs: mx.array = voxel_info["backward_idxs"]  # (bn,)
        cu_seq_lens_list: T.List[mx.array] = voxel_info["cu_seq_lens"]
        max_seq_lens_list: T.List[int] = voxel_info["max_seq_lens"]
        chunk_start_idxs: T.List[int] = voxel_info["chunk_start_idxs"]

        # Sort by voxel cell
        sorted_q = q[forward_idxs]  # (bn, h, dh)
        sorted_k = k[forward_idxs]  # (bn, h, dh)
        sorted_v = v[forward_idxs]  # (bn, h, dh)

        h = q.shape[1]
        dh = q.shape[2]

        out_chunks: T.List[mx.array] = []
        num_chunks = len(cu_seq_lens_list)
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_start_idxs[chunk_idx]
            chunk_end = chunk_start_idxs[chunk_idx + 1]

            chunk_q = sorted_q[chunk_start:chunk_end]  # (chunk_n, h, dh)
            chunk_k = sorted_k[chunk_start:chunk_end]  # (chunk_n, h, dh)
            chunk_v = sorted_v[chunk_start:chunk_end]  # (chunk_n, h, dh)

            cu = cu_seq_lens_list[chunk_idx]  # (num_cells + 1,)
            max_len = max_seq_lens_list[chunk_idx]
            cell_lens = cu[1:] - cu[:-1]  # (num_cells,)
            num_cells = cell_lens.shape[0]

            # Pad into (num_cells, max_len, h, dh) via loop (num_cells is manageable)
            padded_q = mx.zeros((num_cells, max_len, h, dh), dtype=q.dtype)  # (num_cells, max_len, h, dh)
            padded_k = mx.zeros((num_cells, max_len, h, dh), dtype=k.dtype)  # (num_cells, max_len, h, dh)
            padded_v = mx.zeros((num_cells, max_len, h, dh), dtype=v.dtype)  # (num_cells, max_len, h, dh)

            cell_lens_list = cell_lens.tolist()
            cu_list = cu.tolist()
            for cell_idx in range(num_cells):
                cl = int(cell_lens_list[cell_idx])
                start = int(cu_list[cell_idx])
                padded_q[cell_idx, :cl] = chunk_q[start : start + cl]
                padded_k[cell_idx, :cl] = chunk_k[start : start + cl]
                padded_v[cell_idx, :cl] = chunk_v[start : start + cl]

            # Transpose to (num_cells, h, max_len, dh)
            padded_q = mx.transpose(padded_q, axes=(0, 2, 1, 3))  # (num_cells, h, max_len, dh)
            padded_k = mx.transpose(padded_k, axes=(0, 2, 1, 3))  # (num_cells, h, max_len, dh)
            padded_v = mx.transpose(padded_v, axes=(0, 2, 1, 3))  # (num_cells, h, max_len, dh)

            # Boolean mask: (num_cells, 1, 1, max_len)
            cell_mask = mx.arange(max_len)[None, :] < cell_lens[:, None]  # (num_cells, max_len)
            cell_mask = cell_mask[:, None, None, :]  # (num_cells, 1, 1, max_len)

            chunk_out = mx.fast.scaled_dot_product_attention(
                padded_q, padded_k, padded_v, scale=self.dim_head**-0.5, mask=cell_mask
            )  # (num_cells, h, max_len, dh)

            # Transpose back and unpad
            chunk_out = mx.transpose(chunk_out, axes=(0, 2, 1, 3))  # (num_cells, max_len, h, dh)

            # Unpad: collect valid entries per cell
            cell_parts: T.List[mx.array] = []
            for cell_idx in range(num_cells):
                cl = int(cell_lens_list[cell_idx])
                cell_parts.append(chunk_out[cell_idx, :cl])  # (cl, h, dh)
            out_chunks.append(mx.concatenate(cell_parts, axis=0))  # (chunk_n, h, dh)

        out = mx.concatenate(out_chunks, axis=0)  # (bn, h, dh)

        # Unsort back to original order
        out = out[backward_idxs]  # (bn, h, dh)
        return out

    def __call__(
        self,
        x: mx.array,
        seq_lens: T.List[int],
        voxel_info: T.Optional[T.Dict[str, T.Any]] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Packed input features. (bn, dim_in)
            seq_lens: Per-sample sequence lengths.
            voxel_info: Voxel assignment dict for ``localized_voxel`` mode.
                Keys: ``forward_idxs`` (mx.array), ``backward_idxs`` (mx.array),
                ``cu_seq_lens`` (list of mx.array), ``max_seq_lens`` (list of int),
                ``chunk_start_idxs`` (list of int).

        Returns:
            Output features. (bn, dim_in)
        """
        bn = x.shape[0]

        # Project to q, k, v
        qkv = self.linear_qkv(x)  # (bn, 3 * dim_qkv)
        q, k, v = mx.split(qkv, 3, axis=-1)  # each (bn, dim_qkv)

        if self.use_rmsnorm:
            q = self.rmsnorm_q(q)  # (bn, dim_qkv)
            k = self.rmsnorm_k(k)  # (bn, dim_qkv)

        q = q.reshape(bn, self.num_heads, self.dim_head)  # (bn, h, dh)
        k = k.reshape(bn, self.num_heads, self.dim_head)  # (bn, h, dh)
        v = v.reshape(bn, self.num_heads, self.dim_head)  # (bn, h, dh)

        if self.self_attn_type == "global":
            out = self._global_attention(q, k, v, seq_lens)  # (bn, h, dh)
        elif self.self_attn_type == "localized_voxel":
            assert voxel_info is not None, "voxel_info required for localized_voxel self-attention"
            out = self._voxel_attention(q, k, v, voxel_info)  # (bn, h, dh)
        else:
            raise NotImplementedError(f"self_attn_type={self.self_attn_type}")

        out = out.reshape(bn, self.dim_qkv)  # (bn, dim_qkv)
        out = self.linear_out(out)  # (bn, dim_in)
        return out


# ---------------------------------------------------------------------------
# MLXSPointPerceiverEncoderBlock
# ---------------------------------------------------------------------------


class MLXSPointPerceiverEncoderBlock(nn.Module):
    """One perceiver encoder block: cross-attn, MLP, then N x (self-attn + MLP).

    Args:
        dim_latent: Dimension of latent (query) tokens.
        dim_token: Dimension of input (key-value) tokens.
        dim_qkv: Internal qkv dimension for attention layers.
        cross_attn_type: Cross-attention type (only ``"global"`` for MLX).
        self_attn_type: Self-attention type (``"global"`` or ``"localized_voxel"``).
        num_self_attn: Number of self-attention sub-layers per block.
        num_self_heads: Number of heads in self-attention.
        num_cross_heads: Number of heads in cross-attention.
        use_rmsnorm: Apply RMSNorm in attention layers.
        mlp_ratio: MLP hidden-dim expansion ratio.
        mlp_type: ``"swiglu"`` or ``"timm"``.
        linear_in_attn_add_bias: Bias in attention linear projections.
        mlp_add_bias: Bias in MLP linear layers.
        add_kv_linear: Prepend a linear layer on key-value tokens.
    """

    def __init__(
        self,
        dim_latent: int,
        dim_token: int,
        dim_qkv: int,
        cross_attn_type: str,
        self_attn_type: str,
        num_self_attn: int = 2,
        num_self_heads: int = 8,
        num_cross_heads: int = 8,
        use_rmsnorm: bool = True,
        mlp_ratio: float = 4,
        mlp_type: str = "swiglu",
        linear_in_attn_add_bias: bool = False,
        mlp_add_bias: bool = False,
        add_kv_linear: bool = False,
    ):
        super().__init__()
        self.add_kv_linear = add_kv_linear

        if self.add_kv_linear:
            self.kv_linear = nn.Linear(dim_token, dim_token, bias=False)

        self.ca_layer = MLXSPointCrossAttentionLayer(
            dim_q=dim_latent,
            dim_kv=dim_token,
            dim_qkv=dim_qkv,
            num_heads=num_cross_heads,
            use_rmsnorm=use_rmsnorm,
            add_bias=linear_in_attn_add_bias,
        )

        self.ca_ln = nn.LayerNorm(dim_latent, eps=1e-6)

        mlp_hidden_dim = int(dim_latent * mlp_ratio)
        if mlp_type == "swiglu":
            self.ca_mlp = MLXSwiGLU(
                in_features=dim_latent,
                hidden_features=mlp_hidden_dim,
                out_features=None,
                bias=mlp_add_bias,
            )
        elif mlp_type == "timm":
            self.ca_mlp = Mlp(
                in_features=dim_latent,
                hidden_features=mlp_hidden_dim,
                out_features=dim_latent,
            )
        else:
            raise NotImplementedError(f"mlp_type={mlp_type}")

        # Self-attention blocks
        self.ln1_layers: T.List[nn.LayerNorm] = []
        self.sa_layers: T.List[MLXSPointSelfAttentionLayer] = []
        self.ln2_layers: T.List[nn.LayerNorm] = []
        self.mlp_layers: T.List[nn.Module] = []

        for _ in range(num_self_attn):
            self.ln1_layers.append(nn.LayerNorm(dim_latent, eps=1e-6))
            self.ln2_layers.append(nn.LayerNorm(dim_latent, eps=1e-6))

            self.sa_layers.append(
                MLXSPointSelfAttentionLayer(
                    dim_in=dim_latent,
                    dim_qkv=dim_qkv,
                    self_attn_type=self_attn_type,
                    num_heads=num_self_heads,
                    use_rmsnorm=use_rmsnorm,
                    add_bias=linear_in_attn_add_bias,
                )
            )

            sa_mlp_hidden_dim = int(dim_latent * mlp_ratio)
            if mlp_type == "swiglu":
                self.mlp_layers.append(
                    MLXSwiGLU(
                        in_features=dim_latent,
                        hidden_features=sa_mlp_hidden_dim,
                        out_features=None,
                        bias=mlp_add_bias,
                    )
                )
            elif mlp_type == "timm":
                self.mlp_layers.append(
                    Mlp(
                        in_features=dim_latent,
                        hidden_features=sa_mlp_hidden_dim,
                        out_features=dim_latent,
                    )
                )
            else:
                raise NotImplementedError(f"mlp_type={mlp_type}")

    def __call__(
        self,
        query: mx.array,
        key_value: mx.array,
        q_seq_lens: T.List[int],
        kv_seq_lens: T.List[int],
        voxel_infos: T.Optional[T.List[T.Optional[T.Dict[str, T.Any]]]] = None,
    ) -> mx.array:
        """Forward pass for one perceiver block.

        Args:
            query: Packed latent tokens. (bn, dim_latent)
            key_value: Packed input tokens. (bm, dim_token)
            q_seq_lens: Per-sample query lengths.
            kv_seq_lens: Per-sample key-value lengths.
            voxel_infos: Per-self-attn-layer voxel info dicts (or ``None``).

        Returns:
            Updated latent tokens. (bn, dim_latent)
        """
        if self.add_kv_linear:
            key_value = self.kv_linear(key_value)  # (bm, dim_token)

        # Cross attention
        query = query + self.ca_layer(query, key_value, q_seq_lens, kv_seq_lens)  # (bn, dim_latent)
        query = query + self.ca_mlp(self.ca_ln(query))  # (bn, dim_latent)

        # Self attention blocks
        for i, (ln1, sa, ln2, mlp) in enumerate(zip(self.ln1_layers, self.sa_layers, self.ln2_layers, self.mlp_layers)):
            voxel_info = voxel_infos[i] if voxel_infos is not None else None
            query = query + sa(ln1(query), q_seq_lens, voxel_info=voxel_info)  # (bn, dim_latent)
            query = query + mlp(ln2(query))  # (bn, dim_latent)

        return query  # (bn, dim_latent)


# ---------------------------------------------------------------------------
# MLXSPointPerceiverEncoder
# ---------------------------------------------------------------------------


class MLXSPointPerceiverEncoder(nn.Module):
    """Stack of perceiver encoder blocks.

    Args:
        blocks: List of ``MLXSPointPerceiverEncoderBlock``.
        cross_cell_widths: Per-block cross-attention cell widths (unused in global mode).
        self_cell_widths: Per-block self-attention cell widths (unused in global mode).
        num_clusters: Per-block cluster counts (unused in MLX port).
        num_in_cluster: Per-block in-cluster counts (unused in MLX port).
    """

    def __init__(
        self,
        blocks: T.List[MLXSPointPerceiverEncoderBlock],
        cross_cell_widths: T.Optional[T.List[float]] = None,
        self_cell_widths: T.Optional[T.List[float]] = None,
        num_clusters: T.Optional[T.List[int]] = None,
        num_in_cluster: T.Optional[T.List[int]] = None,
    ):
        super().__init__()
        self.blocks = blocks
        self.cross_cell_widths = cross_cell_widths
        self.self_cell_widths = self_cell_widths
        self.num_clusters = num_clusters
        self.num_in_cluster = num_in_cluster

    def __call__(
        self,
        input_tokens: mx.array,
        latent_tokens: mx.array,
        q_seq_lens: T.List[int],
        kv_seq_lens: T.List[int],
        voxel_infos_per_block: T.Optional[T.List[T.Optional[T.List[T.Optional[T.Dict[str, T.Any]]]]]] = None,
    ) -> mx.array:
        """Forward pass through all perceiver blocks.

        Args:
            input_tokens: Packed key-value tokens. (bm, dim_token)
            latent_tokens: Packed latent (query) tokens. (bn, dim_latent)
            q_seq_lens: Per-sample query lengths.
            kv_seq_lens: Per-sample key-value lengths.
            voxel_infos_per_block: Per-block, per-self-attn-layer voxel info.

        Returns:
            Final latent tokens. (bn, dim_latent)
        """
        for i, block in enumerate(self.blocks):
            voxel_infos = voxel_infos_per_block[i] if voxel_infos_per_block is not None else None
            latent_tokens = block(
                query=latent_tokens,
                key_value=input_tokens,
                q_seq_lens=q_seq_lens,
                kv_seq_lens=kv_seq_lens,
                voxel_infos=voxel_infos,
            )  # (bn, dim_latent)

        return latent_tokens  # (bn, dim_latent)


# ---------------------------------------------------------------------------
# MLXOutputMLP
# ---------------------------------------------------------------------------


class MLXOutputMLP(nn.Module):
    """Output MLP mirroring PyTorch ``get_output_mlp``.

    Structure: ``(LayerNorm -> MLP) x num_layers -> FinalLayer``.
    The ``FinalLayer`` has ``dim_cond_feature=0`` so it reduces to
    ``norm_final -> linear``, both of which work on packed 2D tensors.

    Args:
        num_layers: Number of (norm, mlp) pairs before the final layer.
        dim_in: Input dimension.
        dim_hidden: Hidden dimension for each MLP.
        dim_out: Final output dimension.
        mlp_type: ``"swiglu"`` or ``"timm"``.
        mlp_add_bias: Bias in MLP layers.
        contract: If True, each MLP outputs ``dim_in`` instead of ``dim_hidden``.
    """

    def __init__(
        self,
        num_layers: int,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        mlp_type: str = "swiglu",
        mlp_add_bias: bool = True,
        contract: bool = False,
    ):
        super().__init__()
        self.norms: T.List[nn.LayerNorm] = []
        self.mlps: T.List[nn.Module] = []

        current_dim = dim_in
        for _ in range(num_layers):
            self.norms.append(nn.LayerNorm(current_dim, eps=1e-6))
            out_features = dim_in if contract else dim_hidden
            if mlp_type == "swiglu":
                self.mlps.append(
                    MLXSwiGLU(
                        in_features=current_dim,
                        hidden_features=dim_hidden,
                        out_features=out_features,
                        bias=mlp_add_bias,
                    )
                )
            elif mlp_type == "timm":
                self.mlps.append(
                    Mlp(
                        in_features=current_dim,
                        hidden_features=dim_hidden,
                        out_features=out_features,
                    )
                )
            else:
                raise NotImplementedError(f"mlp_type={mlp_type}")
            current_dim = out_features

        # FinalLayer with dim_cond_feature=0 -> norm_final + linear only
        self.final_layer = FinalLayer(
            dim_input=current_dim,
            dim_output=dim_out,
            dim_cond_feature=0,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Packed input. (bn, dim_in)

        Returns:
            Output. (bn, dim_out)
        """
        for norm, mlp in zip(self.norms, self.mlps):
            x = mlp(norm(x))  # (bn, current_dim)

        # FinalLayer expects (b, n, d).  For packed 2D data we call
        # norm_final and linear directly -- both support arbitrary leading dims.
        x = self.final_layer.norm_final(x)  # (bn, current_dim)
        x = self.final_layer.linear(x)  # (bn, dim_out)
        return x


# ---------------------------------------------------------------------------
# MLXGaussianDecoderXv
# ---------------------------------------------------------------------------


class MLXGaussianDecoderXv(nn.Module):
    """MLX inference-only Gaussian decoder.

    Encodes query coordinates, runs perceiver cross/self attention against
    latent shape tokens, and produces raw shape and color MLP outputs.

    Note: ``decode_gs`` (activation / normalization of raw outputs into
    Gaussian parameters) happens externally in PyTorch.

    Args:
        xyz_encoding: Fourier positional encoding for coordinates.
        point_linear: Linear projection from encoded query features to perceiver dim.
        point_mlp: Optional further MLP on projected query features.
        perceiver: ``MLXSPointPerceiverEncoder`` stack.
        gs_output_shape_mlp: MLP producing raw shape outputs.
        gs_output_color_mlp: MLP producing raw color outputs.
        gs_expansion_ratio: Number of Gaussians generated per query point.
        given_point_inputs: List of input types (``"xyz"``, ``"xyz_encoded"``).
    """

    def __init__(
        self,
        xyz_encoding: FourierEmbed,
        point_linear: nn.Linear,
        point_mlp: T.Optional[MLXOutputMLP],
        perceiver: MLXSPointPerceiverEncoder,
        gs_output_shape_mlp: MLXOutputMLP,
        gs_output_color_mlp: MLXOutputMLP,
        gs_expansion_ratio: int,
        given_point_inputs: T.List[str],
    ):
        super().__init__()
        self.xyz_encoding = xyz_encoding
        self.point_linear = point_linear
        self.point_mlp = point_mlp
        self.perceiver = perceiver
        self.gs_output_shape_mlp = gs_output_shape_mlp
        self.gs_output_color_mlp = gs_output_color_mlp
        self.gs_expansion_ratio = gs_expansion_ratio
        self.given_point_inputs = given_point_inputs

    def __call__(
        self,
        latent: mx.array,
        init_query_coord: mx.array,
        q_seq_lens: T.List[int],
        kv_seq_lens: T.List[int],
        voxel_infos_per_block: T.Optional[T.List[T.Optional[T.List[T.Optional[T.Dict[str, T.Any]]]]]] = None,
    ) -> T.Tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            latent: Packed latent shape tokens. (bn, dim_latent)
            init_query_coord: Packed query 3D coordinates. (bm, 3)
            q_seq_lens: Per-sample query (decoder point) lengths.
            kv_seq_lens: Per-sample key-value (latent token) lengths.
            voxel_infos_per_block: Per-block voxel info for localized attention.

        Returns:
            shape_out: Raw shape MLP output. (bm, dim_shape * k)
            color_out: Raw color MLP output. (bm, dim_color * k)
        """
        # Build init_query from coordinates
        parts: T.List[mx.array] = []
        for name in self.given_point_inputs:
            if name == "xyz":
                parts.append(init_query_coord)  # (bm, 3)
            elif name == "xyz_encoded":
                parts.append(self.xyz_encoding(init_query_coord))  # (bm, dim_enc)
            else:
                raise NotImplementedError(f"given_point_input={name}")

        if len(parts) > 1:
            init_query = mx.concatenate(parts, axis=-1)  # (bm, d_concat)
        else:
            init_query = parts[0]  # (bm, d)

        init_query = self.point_linear(init_query)  # (bm, perceiver_dim)
        if self.point_mlp is not None:
            init_query = self.point_mlp(init_query)  # (bm, perceiver_dim)

        # Perceiver: latent (kv) + init_query (query)
        query_latent = self.perceiver(
            input_tokens=latent,
            latent_tokens=init_query,
            q_seq_lens=q_seq_lens,
            kv_seq_lens=kv_seq_lens,
            voxel_infos_per_block=voxel_infos_per_block,
        )  # (bm, perceiver_dim)

        # Output MLPs
        shape_out = self.gs_output_shape_mlp(query_latent)  # (bm, dim_shape * k)
        color_out = self.gs_output_color_mlp(query_latent)  # (bm, dim_color * k)

        return shape_out, color_out
