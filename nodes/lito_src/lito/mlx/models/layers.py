#
# Copyright (C) 2026 Apple Inc. All rights reserved.
#
# MLX implementations of layers used by the DiffusionTransformer.
# Inference only — mirrors src/lito/models/layers.py

import math
import typing as T

from mlx import nn
import mlx.core as mx


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    """Apply adaptive layer norm modulation.

    Args:
        x: Normalized input. (b, n, d)
        shift: Shift parameter. (b, d)
        scale: Scale parameter. (b, d)

    Returns:
        Modulated output. (b, n, d)
    """
    return x * (1 + mx.expand_dims(scale, axis=1)) + mx.expand_dims(shift, axis=1)  # (b, n, d)


def gelu_tanh(x: mx.array) -> mx.array:
    """GELU activation with tanh approximation.

    Args:
        x: Input tensor. (*)

    Returns:
        Activated tensor. (*)
    """
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Given the input (*, d), normalize the feature dimension to unit norm
    then scale by a learnable parameter.

    Args:
        d: Feature dimension.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.d = d
        self.scale = mx.ones((d,))  # (d,)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMS normalization.

        Internal compute always runs in fp32: the ``x * x`` in the variance
        easily overflows fp16 (max ~65504, reached when any element exceeds 256)
        and the ``rsqrt`` of a tiny variance can underflow. Promotion is cheap
        — one squared-mean and rsqrt per token — and prevents the silent inf/nan
        propagation that otherwise washes out long transformer stacks at fp16.

        Args:
            x: Input tensor. (*, d)

        Returns:
            Normalized tensor. (*, d)
        """
        in_dtype = x.dtype
        if in_dtype != mx.float32:
            x_f = x.astype(mx.float32)
        else:
            x_f = x
        x_normed = x_f * mx.rsqrt(mx.mean(x_f * x_f, axis=-1, keepdims=True) + self.eps)  # (*, d)
        out = self.scale.astype(mx.float32) * x_normed if in_dtype != mx.float32 else self.scale * x_normed
        return out.astype(in_dtype) if in_dtype != mx.float32 else out


class FourierEmbed(nn.Module):
    """Sinusoidal / cosinusoidal positional encoding.

    Args:
        dim_pos: Dimensionality of the input positions.
        include_input: Whether to concatenate the raw input to the encoding.
        min_freq_log2: Log2 of the minimum frequency.
        max_freq_log2: Log2 of the maximum frequency.
        num_freqs: Number of frequency bands.
        log_sampling: Whether to sample frequencies in log space.
    """

    def __init__(
        self,
        dim_pos: int,
        include_input: bool,
        min_freq_log2: float,
        max_freq_log2: float,
        num_freqs: int,
        log_sampling: bool,
    ):
        super().__init__()
        self.dim_pos = dim_pos
        self.include_input = include_input
        self.num_freqs = num_freqs

        if log_sampling:
            freq_bands = 2.0 ** mx.linspace(min_freq_log2, max_freq_log2, num_freqs)  # (nf,)
        else:
            freq_bands = mx.linspace(2.0**min_freq_log2, 2.0**max_freq_log2, num_freqs)  # (nf,)

        self.freq_bands = freq_bands  # (nf,)
        self.dim_out = (dim_pos if include_input else 0) + dim_pos * num_freqs * 2

    def __call__(self, pos: mx.array) -> mx.array:
        """Compute Fourier positional encoding.

        Args:
            pos: Input positions. (*, dim_pos)

        Returns:
            Positional encoding. (*, dim_out)
        """
        out: T.List[mx.array] = []
        if self.include_input:
            out.append(pos)  # (*, dim_pos)

        # Expand pos: (*, dim_pos) -> (*, dim_pos, 1) then broadcast with freq_bands (nf,)
        pos_expanded = mx.expand_dims(pos, axis=-1) * self.freq_bands  # (*, dim_pos, nf)

        # Flatten the last two dims: (*, dim_pos * nf)
        batch_shape = pos_expanded.shape[:-2]
        flat_dim = self.dim_pos * self.num_freqs
        out.append(mx.sin(pos_expanded).reshape(*batch_shape, flat_dim))  # (*, dim_pos * nf)
        out.append(mx.cos(pos_expanded).reshape(*batch_shape, flat_dim))  # (*, dim_pos * nf)

        return mx.concatenate(out, axis=-1)  # (*, dim_out)


class SelfAttentionLayer(nn.Module):
    """Multihead self attention.

    Convert input tokens to q, k, v. Perform scaled dot product attention.
    Convert concatenated head output to output tokens.

    Args:
        dim_in: Feature dimension of the input tokens.
        dim_qkv: Feature dimension of q, k, v (must be divisible by num_heads).
        num_heads: Number of attention heads.
        use_rmsnorm: Whether to apply RMSNorm to q and k.
        add_bias: Whether to include bias in linear projections.
    """

    def __init__(
        self,
        dim_in: int,
        dim_qkv: int,
        num_heads: int = 4,
        use_rmsnorm: bool = True,
        add_bias: bool = True,
    ):
        super().__init__()
        assert dim_qkv % num_heads == 0, f"{dim_qkv}, {num_heads}"
        self.dim_in = dim_in
        self.dim_qkv = dim_qkv
        self.num_heads = num_heads
        self.use_rmsnorm = use_rmsnorm
        self.dim_head = dim_qkv // num_heads

        self.linear_qkv = nn.Linear(dim_in, 3 * dim_qkv, bias=add_bias)
        self.linear_out = nn.Linear(dim_qkv, dim_in, bias=add_bias)

        if self.use_rmsnorm:
            self.rmsnorm_q = RMSNorm(dim_qkv)
            self.rmsnorm_k = RMSNorm(dim_qkv)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass for self attention.

        Args:
            x: Input tokens. (b, n, dim_in)

        Returns:
            Output tokens. (b, n, dim_in)
        """
        b, n, _d = x.shape

        qkv = self.linear_qkv(x)  # (b, n, 3 * dim_qkv)
        q, k, v = mx.split(qkv, 3, axis=-1)  # each (b, n, dim_qkv)

        if self.use_rmsnorm:
            q = self.rmsnorm_q(q)  # (b, n, dim_qkv)
            k = self.rmsnorm_k(k)  # (b, n, dim_qkv)

        q = q.reshape(b, n, self.num_heads, self.dim_head)  # (b, n, h, dh)
        k = k.reshape(b, n, self.num_heads, self.dim_head)  # (b, n, h, dh)
        v = v.reshape(b, n, self.num_heads, self.dim_head)  # (b, n, h, dh)

        # mx.fast.scaled_dot_product_attention expects (b, h, n, dh)
        q = mx.transpose(q, axes=(0, 2, 1, 3))  # (b, h, n, dh)
        k = mx.transpose(k, axes=(0, 2, 1, 3))  # (b, h, n, dh)
        v = mx.transpose(v, axes=(0, 2, 1, 3))  # (b, h, n, dh)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.dim_head**-0.5)  # (b, h, n, dh)

        out = mx.transpose(out, axes=(0, 2, 1, 3))  # (b, n, h, dh)
        out = out.reshape(b, n, self.dim_qkv)  # (b, n, dim_qkv)
        out = self.linear_out(out)  # (b, n, dim_in)

        return out


class CrossAttentionLayer(nn.Module):
    """Multihead cross attention with pre-layer-norm.

    Convert query tokens to q and key-value tokens to k, v.
    Perform scaled dot product attention.

    Args:
        dim_q: Dimension of query input tokens.
        dim_kv: Dimension of key-value input tokens.
        dim_qkv: Internal dimension for q, k, v (must be divisible by num_heads).
        num_heads: Number of attention heads.
        use_rmsnorm: Whether to apply RMSNorm to q and k.
        add_bias: Whether to include bias in linear projections.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        dim_qkv: int,
        num_heads: int = 4,
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

        self.linear_q = nn.Linear(dim_q, dim_qkv, bias=add_bias)
        self.linear_kv = nn.Linear(dim_kv, 2 * dim_qkv, bias=add_bias)
        self.linear_out = nn.Linear(dim_qkv, dim_q, bias=add_bias)

        if self.use_rmsnorm:
            self.rmsnorm_q = RMSNorm(dim_qkv)
            self.rmsnorm_k = RMSNorm(dim_qkv)

        self.layernorm_q = nn.LayerNorm(dim_q)
        self.layernorm_kv = nn.LayerNorm(dim_kv)

    def __call__(self, query: mx.array, key_value: mx.array) -> mx.array:
        """Forward pass for cross attention.

        Args:
            query: Query tokens. (b, n, dim_q)
            key_value: Key-value tokens. (b, m, dim_kv)

        Returns:
            Output tokens. (b, n, dim_q)
        """
        b, n, _dim_q = query.shape
        _b, m, _dim_kv = key_value.shape

        # Pre-layer normalization
        query = self.layernorm_q(query)  # (b, n, dim_q)
        key_value = self.layernorm_kv(key_value)  # (b, m, dim_kv)

        # Linear projections
        query = self.linear_q(query)  # (b, n, dim_qkv)
        kv = self.linear_kv(key_value)  # (b, m, 2 * dim_qkv)
        key, value = mx.split(kv, 2, axis=-1)  # each (b, m, dim_qkv)

        if self.use_rmsnorm:
            query = self.rmsnorm_q(query)  # (b, n, dim_qkv)
            key = self.rmsnorm_k(key)  # (b, m, dim_qkv)

        query = query.reshape(b, n, self.num_heads, self.dim_head)  # (b, n, h, dh)
        key = key.reshape(b, m, self.num_heads, self.dim_head)  # (b, m, h, dh)
        value = value.reshape(b, m, self.num_heads, self.dim_head)  # (b, m, h, dh)

        # mx.fast.scaled_dot_product_attention expects (b, h, seq, dh)
        query = mx.transpose(query, axes=(0, 2, 1, 3))  # (b, h, n, dh)
        key = mx.transpose(key, axes=(0, 2, 1, 3))  # (b, h, m, dh)
        value = mx.transpose(value, axes=(0, 2, 1, 3))  # (b, h, m, dh)

        out = mx.fast.scaled_dot_product_attention(query, key, value, scale=self.dim_head**-0.5)  # (b, h, n, dh)

        out = mx.transpose(out, axes=(0, 2, 1, 3))  # (b, n, h, dh)
        out = out.reshape(b, n, self.dim_qkv)  # (b, n, dim_qkv)
        out = self.linear_out(out)  # (b, n, dim_q)

        return out


class FinalLayer(nn.Module):
    """Final projection layer with optional adaptive layer norm modulation.

    Args:
        dim_input: Input feature dimension.
        dim_output: Output feature dimension.
        dim_cond_feature: Conditioning feature dimension. If 0, no adaLN modulation.
    """

    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        dim_cond_feature: int = 0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_cond_feature = dim_cond_feature

        self.norm_final = nn.LayerNorm(dim_input, affine=(dim_cond_feature == 0), eps=eps)
        self.linear = nn.Linear(dim_input, dim_output, bias=True)

        if dim_cond_feature > 0:
            self.adaLN_linear1 = nn.Linear(dim_cond_feature, dim_cond_feature, bias=True)
            self.adaLN_linear2 = nn.Linear(dim_cond_feature, 2 * dim_input, bias=True)

    def __call__(self, x: mx.array, cond_feature: T.Optional[mx.array] = None) -> mx.array:
        """Forward pass.

        Args:
            x: Input tokens. (b, n, dim_input)
            cond_feature: Conditioning feature for adaLN. (b, dim_cond_feature)

        Returns:
            Output tokens. (b, n, dim_output)
        """
        if cond_feature is not None:
            ada_out = nn.silu(self.adaLN_linear1(cond_feature))  # (b, dim_cond_feature)
            ada_out = self.adaLN_linear2(ada_out)  # (b, 2 * dim_input)
            shift, scale = mx.split(ada_out, 2, axis=1)  # each (b, dim_input)
            x = modulate(self.norm_final(x), shift, scale)  # (b, n, dim_input)
        else:
            x = self.norm_final(x)  # (b, n, dim_input)

        return self.linear(x)  # (b, n, dim_output)


class Mlp(nn.Module):
    """Two-layer MLP with GELU-tanh activation.

    Args:
        in_features: Input feature dimension.
        hidden_features: Hidden layer dimension.
        out_features: Output feature dimension.
        compute_dtype: Optional dtype to promote inputs to before fc1, casting
            the final output back to the original dtype. When set to ``mx.float32``,
            this gives the matmul fp32 accumulation (analogous to PyTorch's autocast
            behaviour, where weights stay in low precision but reductions run in fp32).
            Default ``None`` = no promotion.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        compute_dtype: T.Optional[mx.Dtype] = None,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.compute_dtype = compute_dtype

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor. (*, in_features)

        Returns:
            Output tensor. (*, out_features)
        """
        in_dtype = x.dtype
        if self.compute_dtype is not None and in_dtype != self.compute_dtype:
            x = x.astype(self.compute_dtype)  # promote (e.g. fp16/bf16 → fp32)
        x = self.fc1(x)  # (*, hidden_features)
        x = gelu_tanh(x)  # (*, hidden_features)
        x = self.fc2(x)  # (*, out_features)
        if self.compute_dtype is not None and in_dtype != self.compute_dtype:
            x = x.astype(in_dtype)  # cast back
        return x


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feed-forward network.

    Uses the SwiGLU gating mechanism with a hidden dimension rounded up to
    the nearest multiple of ``multiple_of``.

    Args:
        dim: Input and output dimension.
        hidden_dim: Base hidden dimension (before 2/3 scaling and rounding).
        multiple_of: Round hidden dimension up to nearest multiple.
        compute_dtype: Optional dtype to promote inputs to (see ``Mlp``). Default ``None``.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        compute_dtype: T.Optional[mx.Dtype] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.compute_dtype = compute_dtype

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor. (*, dim)

        Returns:
            Output tensor. (*, dim)
        """
        in_dtype = x.dtype
        if self.compute_dtype is not None and in_dtype != self.compute_dtype:
            x = x.astype(self.compute_dtype)  # promote
        out = self.w2(nn.silu(self.w1(x)) * self.w3(x))  # (*, dim)
        if self.compute_dtype is not None and in_dtype != self.compute_dtype:
            out = out.astype(in_dtype)  # cast back
        return out


class SinusoidalEmbedder(nn.Module):
    """Embeds scalar timesteps into vectors using sinusoidal encoding + MLP.

    Mirrors lito.models.layers.SinusoidalEmbedder. Uses max_period=10000 so
    frequency derivatives are bounded by 1 (JVP-stable for IMF).

    The PyTorch module wraps the MLP in nn.Sequential([Linear, SiLU, Linear]);
    we use explicit attributes mlp_linear1 / mlp_linear2 so weight loading
    can remap mlp.0 / mlp.2 keys.

    Args:
        hidden_size: Output dimension.
        frequency_embedding_size: Sinusoidal encoding dimension before the MLP.
        max_period: Controls minimum frequency.
    """

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: float = 10000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.dim_out = hidden_size

        half = frequency_embedding_size // 2
        # freqs[i] = exp(-log(max_period) * i / half), range [1/max_period, 1]
        freqs = mx.exp(-math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half)  # (half,)
        self.freqs = freqs  # (half,) — placeholder, overwritten by load_weights

        self.mlp_linear1 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.mlp_linear2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def sinusoidal_embedding(self, t: mx.array) -> mx.array:
        """Create sinusoidal timestep embeddings.

        Args:
            t: Scalar timesteps. (b,)

        Returns:
            Sinusoidal embeddings. (b, frequency_embedding_size)
        """
        # Compute in float32 to match torch (t and freqs are cast to float there).
        args = mx.expand_dims(t.astype(mx.float32), axis=-1) * self.freqs.astype(mx.float32)[None]  # (b, half)
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)  # (b, 2*half)
        if self.frequency_embedding_size % 2:
            embedding = mx.concatenate([embedding, mx.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def __call__(self, t: mx.array) -> mx.array:
        """Forward pass.

        Args:
            t: Scalar timesteps in [0, 1]. (b,) or (b, 1)

        Returns:
            Timestep embeddings. (b, hidden_size)
        """
        if t.ndim == 2:
            t = mx.squeeze(t, axis=-1)  # (b,)
        t_freq = self.sinusoidal_embedding(t)  # (b, frequency_embedding_size)
        # Cast back to the MLP's parameter dtype so quantized / bf16 paths work.
        t_freq = t_freq.astype(self.mlp_linear1.weight.dtype)
        t_freq = self.mlp_linear1(t_freq)  # (b, hidden_size)
        t_freq = nn.silu(t_freq)
        return self.mlp_linear2(t_freq)  # (b, hidden_size)


class FinalLayerWithRMSNorm(nn.Module):
    """Final projection: RMSNorm + Linear, no conditioning input.

    Mirrors lito.models.layers.FinalLayerWithRMSNorm.

    Args:
        dim_input: Input feature dimension.
        dim_output: Output feature dimension.
        eps: RMSNorm epsilon.
    """

    def __init__(self, dim_input: int, dim_output: int, eps: float = 1e-6):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.norm = RMSNorm(dim_input, eps=eps)
        self.linear = nn.Linear(dim_input, dim_output, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tokens. (b, n, dim_input)

        Returns:
            Output tokens. (b, n, dim_output)
        """
        return self.linear(self.norm(x))  # (b, n, dim_output)
