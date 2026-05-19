#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#


import math
import typing as T

import numpy as np
from lito.models._comfy_mlp import Mlp

# SwiGLU is bound to the pure-PyTorch implementation in this file at module load
# (late-bound after the class definition below). xformers' SwiGLU is no longer used.
_SwiGLU = None  # set after the SwiGLU class is defined below

import torch
from torch import nn
import torch.utils.checkpoint

from lito.models.struct_attn import structural_memory_efficient_attention
from plibs import ppoint
from contextlib import nullcontext as _nullcontext
_OPS = None
def _ops():
    global _OPS
    if _OPS is None:
        import comfy.ops
        _OPS = comfy.ops.manual_cast
    return _OPS



class OverfitLatent(torch.nn.Module):
    def __init__(
        self,
        batch_size: int,
        num_latent: int,
        dim_latent: int,
    ):
        """
        Args:
            batch_size:
                batch size
            num_latent:
                number of latent tokens used to encode a point cloud
            dim_latent:
                dimension of each latent token
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_latent = num_latent
        self.dim_latent = dim_latent

        _tmp = torch.nn.functional.normalize(
            torch.randn(self.batch_size, self.num_latent, self.dim_latent),
            dim=-1,
        ) * np.sqrt(self.dim_latent)

        self.latents = nn.Parameter(_tmp)  # (b, n, d)

    def forward(self, idxs: T.List[int]):
        """
        Args:
            idxs:
                (b,) list of indices of latent tokens
        Returns:
            (b, num_latent, dim_latent)
        """
        latents = self.latents[idxs]
        return latents  # (b, n, d)


def modulate(x, shift, scale, debug: bool = False):
    """
    Args:
        x:
            (b, n, d)
        shift:
            (b, d)
        scale:
            (b, d)
        debug:

    Returns:
        (b, n, d)
    """

    if debug:
        assert x.isfinite().all(), f"{x.shape}, nan {x.isnan().any()}, inf {x.isinf().any()}"
        assert shift.isfinite().all(), f"{shift.shape}, nan {shift.isnan().any()}, inf {shift.isinf().any()}"
        assert scale.isfinite().all(), f"{scale.shape}, nan {scale.isnan().any()}, inf {scale.isinf().any()}"

    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SelfAttentionLayer(nn.Module):
    """
    Multihead self attention.
    Convert input tokens to q, k, v. Perform dot product attention.
    Convert concat head output to output tokens.
    """

    def __init__(
        self,
        dim_in: int,
        dim_qkv: int,
        num_heads: int = 4,
        dropout_prob: float = 0.0,
        use_rmsnorm: bool = True,
        add_bias: bool = True,
    ):
        """
        Args:
            dim_in:
                feature dimension of the input tokens
            dim_qkv:
                feature dimension of qkv.
                Since we want to use flash attention, force them to be the same.
            num_heads:
                number of heads in the multihead attention.
            dropout_prob:
                dropout probability.
            use_rmsnorm:
                whether to use rms norm.
            add_bias:
                whether to add bias in linear qkv and output
        """
        super().__init__()
        assert dim_qkv % num_heads == 0, f"{dim_qkv}, {num_heads}"
        self.dim_in = dim_in
        self.dim_qkv = dim_qkv
        self.num_heads = num_heads
        self.use_rmsnorm = use_rmsnorm
        self.dim_head = self.dim_qkv // self.num_heads
        self.dropout_prob = dropout_prob
        self.add_bias = add_bias

        # linear projection
        self.linear_qkv = _ops().Linear(
            in_features=self.dim_in,
            out_features=3 * self.dim_qkv,
            bias=self.add_bias,
        )
        self.linear_out = _ops().Linear(
            in_features=self.dim_qkv,
            out_features=self.dim_in,
            bias=self.add_bias,
        )
        if self.use_rmsnorm:
            self.rmsnorm_q = RMSNorm(self.dim_qkv)
            self.rmsnorm_k = RMSNorm(self.dim_qkv)

    def forward(
        self,
        x: torch.Tensor,
        structural_attn_dict: T.Dict[str, T.Any] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                (b, seq_len, dim_in)
            structural_attn_dict:
                structural attention constructed.
                None: all queries attend to all outputs

        Returns:
            (b, seq_len, dim_in)
        """

        b, n, d = x.shape

        # project to qkv
        qkv = self.linear_qkv(x)  # (b, n, 3*dim_qkv)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)  # (b, n, dim_qkv)

        if self.use_rmsnorm:
            q = self.rmsnorm_q(q)  # (b, n, dim_qkv)
            k = self.rmsnorm_k(k)  # (b, n, dim_qkv)

        q = q.reshape(b, n, self.num_heads, self.dim_head)  # (b, n, h, dim_head)
        k = k.reshape(b, n, self.num_heads, self.dim_head)  # (b, n, h, dim_head)
        v = v.reshape(b, n, self.num_heads, self.dim_head)  # (b, n, h, dim_head)

        # attention
        out = structural_memory_efficient_attention(
            query=q,  # (b, n, h, dhead)
            key=k,  # (b, n, h, dhead)
            value=v,  # (b, n, h, dhead)
            p=self.dropout_prob,
            structural_attn_dict=structural_attn_dict,
        )  # (b, n, h, dhead)
        out = self.linear_out(out.reshape(b, n, self.dim_qkv))  # (b, n, dim_in)

        return out


class CrossAttentionLayer(nn.Module):
    """
    Multihead cross attention.
    Convert q_tokens to q and kv_tokens to k and v.  Perform dot product attention.
    Convert concat head output to output tokens.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        dim_qkv: int,
        num_heads: int = 4,
        dropout_prob: float = 0.0,
        use_rmsnorm: bool = True,
        add_bias: bool = True,
        packed_kv: bool = False,
    ):
        """
        Args:
            dim_q:
                dimension of the input query tokens
            dim_kv:
                dimension of the input kv tokens (tokens that will be attended to)
            dim_qkv:
                feature dimension of qkv.
                Since we want to use flash attention, force them to be the same.
            num_heads:
                number of heads in the multihead attention.
            dropout_prob:
                dropout probability.
            use_rmsnorm:
                whether to use rms norm.
            add_bias:
                whether to add bias in linear qkv and output
            packed_kv:
                whether the key and value are in packed format, ie (n1+n2+..._nb, d)
        """
        super().__init__()
        assert dim_qkv % num_heads == 0
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.dim_qkv = dim_qkv
        self.num_heads = num_heads
        self.use_rmsnorm = use_rmsnorm
        self.dim_head = self.dim_qkv // self.num_heads
        self.dropout_prob = dropout_prob
        self.add_bias = add_bias
        self.packed_kv = packed_kv

        # linear projection
        self.linear_q = _ops().Linear(
            in_features=self.dim_q,
            out_features=self.dim_qkv,
            bias=self.add_bias,
        )
        self.linear_kv = _ops().Linear(
            in_features=self.dim_kv,
            out_features=2 * self.dim_qkv,
            bias=self.add_bias,
        )
        self.linear_out = _ops().Linear(
            in_features=self.dim_qkv,
            out_features=self.dim_q,
            bias=self.add_bias,
        )
        if self.use_rmsnorm:
            self.rmsnorm_q = RMSNorm(self.dim_qkv)
            self.rmsnorm_k = RMSNorm(self.dim_qkv)

        # pre layer normalization
        self.layernorm_q = _ops().LayerNorm(self.dim_q)
        self.layernorm_kv = _ops().LayerNorm(self.dim_kv)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        structural_attn_dict: T.Dict[str, T.Any] = None,
        packed_kv_coord: ppoint.PackedPoint = None,
    ) -> torch.Tensor:
        """
        Args:
            query:
                (b, n, dim_q)
            key_value:
                (b, m, dim_kv) or (bm, dim_kv)
            structural_attn_dict:
                structural attention constructed.
                None: all queries attend to all outputs
            packed_kv_coord:
                packed coordinate of key_value, needed if packed_kv is True

        Returns:
            (b, n, dim_q)
        """

        b, n, _dim_q = query.shape

        # pre layer norm
        query = self.layernorm_q(query)  # (b, n, dq)
        key_value = self.layernorm_kv(key_value)  # (b, m, dkv) or (bm, dkv)

        # linear projection
        query = self.linear_q(query)  # (b, n, dim_qkv)
        key_value = self.linear_kv(key_value)  # (b, m, 2 * dim_qkv) or (bm, 2 * dim_qkv)
        key, value = torch.chunk(key_value, chunks=2, dim=-1)  # (b, m, dim_qkv) or (bm, dim_qkv)

        if self.use_rmsnorm:
            query = self.rmsnorm_q(query)
            key = self.rmsnorm_k(key)

        query = query.reshape(b, n, self.num_heads, self.dim_head)  # (b, n, h, dim_head)
        if not self.packed_kv:
            _b, m, _dim_kv = key_value.shape
            key = key.reshape(b, m, self.num_heads, self.dim_head)  # (b, m, h, dim_head)
            value = value.reshape(b, m, self.num_heads, self.dim_head)  # (b, m, h, dim_head)
            # attention
            out = structural_memory_efficient_attention(
                query,  # (b, n, h, dim_head)
                key,  # (b, m, h, dim_head)
                value,  # (b, m, h, dim_head)
                p=self.dropout_prob,
                structural_attn_dict=structural_attn_dict,
            )  # (b, n, h, dim_head)
        else:
            # packed format
            assert packed_kv_coord is not None
            out = ppoint.cross_softmax_attention_with_packed_kv(
                query=query,  # (b, n, h, dim_head)
                packed_kv_coord=packed_kv_coord,
                packed_key=key.reshape(key.size(0), self.num_heads, self.dim_head),  # (bm, h, dim_head)
                packed_value=value.reshape(key.size(0), self.num_heads, self.dim_head),  # (bm, h, dim_head),
                save_to_cache=True,
            )  # (b, n, h, dim_head)

        out = self.linear_out(out.reshape(b, n, self.dim_qkv))  # (b, n, dim_q)
        return out


class SelfAttentionBlock(torch.nn.Module):
    """
    Each block contains
    1. self attention
    2. mlp
    """

    def __init__(
        self,
        dim: int,
        dim_qkv: int,
        num_self_heads: int = 8,
        dropout_prob: float = 0.0,
        use_rmsnorm: bool = True,
        mlp_ratio: float = 4,
        mlp_type: str = "timm",
        linear_in_attn_add_bias: bool = True,
        mlp_add_bias: bool = True,
    ):
        super().__init__()

        self.ln1 = _ops().LayerNorm(dim, eps=1e-6)
        self.ln2 = _ops().LayerNorm(dim, eps=1e-6)

        self.sa_layer = SelfAttentionLayer(
            dim_in=dim,
            dim_qkv=dim_qkv,
            num_heads=num_self_heads,
            dropout_prob=dropout_prob,
            use_rmsnorm=use_rmsnorm,
            add_bias=linear_in_attn_add_bias,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        if mlp_type == "timm":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
                bias=mlp_add_bias,
            )
        elif mlp_type == "swiglu":
            self.mlp = _SwiGLU(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                out_features=None,
                bias=mlp_add_bias,
            )
        else:
            raise NotImplementedError

    def _forward(
        self,
        x: torch.Tensor,
        structural_attn_dict: T.Dict[str, T.Any] = None,
    ):
        """
        Args:
            x:
                (b, seq_len, dim)

        Returns:
            (b, seq_len, dim)
        """
        x = x + self.sa_layer(
            x=self.ln1(x),
            structural_attn_dict=structural_attn_dict,
        )  # (b, seq_len, dim)
        x = x + self.mlp(self.ln2(x))  # (b, seq_len, dim)

        return x  # (b, seq_len, dim)

    def forward(
        self,
        x: torch.Tensor,
        structural_attn_dict: T.Dict[str, T.Any] = None,
        use_grad_checkpointing: bool = False,
    ):
        """
        Args:
            x:
                (b, seq_len, dim)

        Returns:
            (b, seq_len, dim)
        """
        if use_grad_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                x,
                structural_attn_dict,
                use_reentrant=False,
            )
        else:
            return self._forward(
                x=x,
                structural_attn_dict=structural_attn_dict,
            )


class PointwiseResnet(torch.nn.Module):
    """
    Resnet containing only 2 layers of linear layers.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_hidden: int = None,
        bias: bool = True,
        activation_fn: str = "gelu",
        add_init_activation: bool = True,
    ):
        super().__init__()

        if dim_out is None:
            dim_out = dim_in

        if dim_hidden is None:
            dim_hidden = min(dim_in, dim_out)

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.activation_fn = activation_fn
        self.add_init_activation = add_init_activation

        self.linear_in = self.linear_out = self.ffn_swiglu = None
        if self.activation_fn == "gelu":
            self.nonlinearity = torch.nn.GELU(approximate="tanh")
            self.linear_in = _ops().Linear(self.dim_in, self.dim_hidden, bias=bias)
            self.linear_out = _ops().Linear(self.dim_hidden, self.dim_out, bias=bias)
        elif self.activation_fn == "silu":
            self.nonlinearity = torch.nn.SiLU()
            self.linear_in = _ops().Linear(self.dim_in, self.dim_hidden, bias=bias)
            self.linear_out = _ops().Linear(self.dim_hidden, self.dim_out, bias=bias)
        elif self.activation_fn == "swiglu":
            self.ffn_swiglu = _SwiGLU(
                in_features=dim_in,
                hidden_features=dim_hidden,
                out_features=dim_out,
                bias=bias,
            )
            self.nonlinearity = torch.nn.SiLU()
        else:
            raise NotImplementedError

        if self.dim_in == self.dim_out:
            self.skip_linear = None
        else:
            self.skip_linear = _ops().Linear(self.dim_in, self.dim_out, bias=False)

        self._init_parameteres()

    def _init_parameteres(self):
        if self.linear_in is not None:
            torch.nn.init.xavier_uniform_(self.linear_in.weight)
            if self.linear_in.bias is not None:
                torch.nn.init.constant_(self.linear_in.bias, 0)
        if self.linear_out is not None:
            torch.nn.init.xavier_uniform_(self.linear_out.weight)
            if self.linear_out.bias is not None:
                torch.nn.init.constant_(self.linear_out.bias, 0)

        if self.skip_linear is not None:
            torch.nn.init.xavier_uniform_(self.skip_linear.weight)
            if self.skip_linear.bias is not None:
                torch.nn.init.constant_(self.skip_linear.bias, 0)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                (*, dim_in)

        Returns:
            (*, dim_out)
        """
        if self.skip_linear is not None:
            x_s = self.skip_linear(x)
        else:
            x_s = x

        if self.add_init_activation:
            x = self.nonlinearity(x)

        if self.activation_fn in ["gelu", "silu"]:
            dx = self.linear_out(self.nonlinearity(self.linear_in(x)))
        elif self.activation_fn == "swiglu":
            dx = self.ffn_swiglu(x.contiguous())
        else:
            raise NotImplementedError

        return x_s + dx

    def forward(
        self,
        x: torch.Tensor,
        use_grad_checkpointing: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                (*, dim_in)

        Returns:
            (*, dim_out)
        """
        if use_grad_checkpointing:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class FinalMLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: T.Union[int, T.List[int]],
        dim_out: int,
        num_layers: int,
        dim_cond_feature: int = 0,
        eps: float = 1e-6,
        mlp_type: str = "swiglu",
        mlp_add_bias: bool = False,
    ):
        super().__init__()
        if isinstance(dim_hidden, int):
            dim_hidden = [dim_hidden] * num_layers

        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.dim_cond_feature = dim_cond_feature

        self.blocks = []
        current_dim = self.dim_in
        for i in range(self.num_layers - 1):
            block_dict = dict()
            block_dict["norm_layer"] = _ops().LayerNorm(
                current_dim,
                elementwise_affine=(self.dim_cond_feature == 0),
                eps=eps,
            )

            if self.dim_cond_feature > 0:
                adaLN_modulation = nn.Sequential(
                    _ops().Linear(self.dim_cond_feature, self.dim_cond_feature, bias=True),
                    nn.SiLU(),
                    _ops().Linear(self.dim_cond_feature, 2 * current_dim, bias=True),
                )
            else:
                adaLN_modulation = None
            block_dict["adaLN_modulation"] = adaLN_modulation

            if mlp_type == "timm":
                approx_gelu = lambda: torch.nn.GELU(approximate="tanh")
                mlp = Mlp(
                    in_features=current_dim,
                    hidden_features=min(dim_hidden[i], current_dim),
                    act_layer=approx_gelu,
                    drop=0,
                    bias=mlp_add_bias,
                    out_features=dim_hidden[i],
                )
            elif mlp_type == "swiglu":
                mlp = _SwiGLU(
                    in_features=current_dim,
                    hidden_features=min(dim_hidden[i], current_dim),
                    out_features=dim_hidden[i],
                    bias=mlp_add_bias,
                )
            elif mlp_type.startswith("resnet_"):
                # 'resnet_gelu', 'resnet_silu', 'resnet_swiglu'
                activation_fn = mlp_type.split("resnet_", 1)[1]
                mlp = PointwiseResnet(
                    dim_in=current_dim,
                    dim_hidden=min(dim_hidden[i], current_dim),
                    dim_out=dim_hidden[i],
                    bias=mlp_add_bias,
                    activation_fn=activation_fn,
                    add_init_activation=False,
                )
            else:
                raise NotImplementedError
            block_dict["mlp"] = mlp
            block_dict = torch.nn.ModuleDict(block_dict)
            self.blocks.append(block_dict)

            current_dim = dim_hidden[i]

        self.blocks = nn.ModuleList(self.blocks)

        self.final_layer = FinalLayer(
            dim_input=current_dim,
            dim_output=dim_out,
            dim_cond_feature=self.dim_cond_feature,
            eps=eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        cond_feature: torch.Tensor = None,
    ):
        """
        Args:
            x:
                (b, n, dim_input)
            cond_feature:
                (b, dim_cond_feature)

        Returns:
            (b, n, dim_output)
        """
        for i in range(len(self.blocks)):
            block_dict = self.blocks[i]
            if cond_feature is not None:
                shift, scale = block_dict["adaLN_modulation"](cond_feature).chunk(2, dim=1)
                x = modulate(block_dict["norm_layer"](x), shift, scale)
            else:
                x = block_dict["norm_layer"](x)
            x = block_dict["mlp"](x)

        x = self.final_layer(x=x, cond_feature=cond_feature)
        return x


class FinalLayer(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        dim_cond_feature: int = 0,
        eps: float = 1e-6,
        force_fp32: bool = False,
    ):
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_cond_feature = dim_cond_feature
        self.norm_final = _ops().LayerNorm(dim_input, elementwise_affine=(self.dim_cond_feature == 0), eps=eps)
        self.linear = _ops().Linear(dim_input, dim_output, bias=True)
        if self.dim_cond_feature > 0:
            self.adaLN_modulation = nn.Sequential(
                _ops().Linear(dim_cond_feature, dim_cond_feature, bias=True),
                nn.SiLU(),
                _ops().Linear(dim_cond_feature, 2 * dim_input, bias=True),
            )
        else:
            self.adaLN_modulation = None
        self.force_fp32 = force_fp32

    def forward(
        self,
        x: torch.Tensor,
        cond_feature: torch.Tensor = None,
    ):
        """
        Args:
            x:
                (b, n, dim_input)
            cond_feature:
                (b, dim_cond_feature)

        Returns:
            (b, n, dim_output)
        """
        if cond_feature is not None:
            shift, scale = self.adaLN_modulation(cond_feature).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
        else:
            x = self.norm_final(x)

        if not self.force_fp32:
            x = self.linear(x)
        else:
            with _nullcontext():
                x = self.linear(x.float())
        return x


class RMSNorm(nn.Module):
    def __init__(
        self,
        d: int,
        p: float = -1.0,
        eps: float = 1e-8,
        bias: bool = False,
    ):
        """
        Root Mean Square Layer Normalization.
        Given the input (*, d), normalize the feature dimension (ie, d)
        to unit norm then shift and scale.

        Args
            d:
                feature dimension
            p:
                partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
            eps:
                epsilon value, default 1e-8
            bias:
                whether use bias term for RMSNorm, disabled by
                default because RMSNorm doesn't enforce re-centering invariance.
        """
        super().__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))

    def _normalize(self, x):
        """x: (*, d)"""
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                (*, d)

        Returns:
            (*, d)
        """
        if self.p < 0.0 or self.p > 1.0:
            # always use p = 2
            # norm_x = x.norm(2, dim=-1, keepdim=True, dtype=x.dtype)  # (*, 1)
            # d_x = self.d
            # rms_x = norm_x * d_x ** (-1.0 / 2)
            # x_normed = x / (rms_x + self.eps)

            x_normed = self._normalize(x.float()).type_as(x)
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True, dtype=x.dtype)
            d_x = partial_size

            rms_x = norm_x * d_x ** (-1.0 / 2)
            x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset
        else:
            return self.scale * x_normed


class SwiGLU(nn.Module):
    """Pure-PyTorch drop-in replacement for ``xformers.ops.SwiGLU``.

    Used as a fallback when xformers is not installed (e.g. on macOS).
    Matches the ``xformers.ops.SwiGLU`` interface and weight naming exactly:

    - ``w1``: in_features -> hidden_features (value branch)
    - ``w2``: in_features -> hidden_features (gate branch)
    - ``w3``: hidden_features -> out_features (output projection)
    - forward: ``w3(silu(w1(x)) * w2(x))``

    When ``_pack_weights=True``, ``w1`` / ``w2`` are merged into a single
    ``w12`` linear (matching xformers' default).

    Args:
        in_features: Input dimension.
        hidden_features: Hidden dimension (used directly, no reduction).
        out_features: Output dimension. If None, same as ``in_features``.
        bias: Whether to add bias to all three linear layers.
        _pack_weights: Whether to pack w1 and w2 into a single w12 linear
            (matches xformers' default).
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: T.Optional[int] = None,
        bias: bool = True,
        _pack_weights: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w12: T.Optional[nn.Linear]
        if _pack_weights:
            self.w12 = _ops().Linear(in_features, 2 * hidden_features, bias=bias)
        else:
            self.w12 = None
            self.w1 = _ops().Linear(in_features, hidden_features, bias=bias)
            self.w2 = _ops().Linear(in_features, hidden_features, bias=bias)
        self.w3 = _ops().Linear(hidden_features, out_features, bias=bias)

        self.hidden_features = hidden_features
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor. (*, in_features)

        Returns:
            Output tensor. (*, out_features)
        """
        if self.w12 is not None:
            x12 = self.w12(x)  # (*, 2 * hidden_features)
            return self.w3(
                torch.nn.functional.silu(x12[..., : self.hidden_features]) * x12[..., self.hidden_features :]
            )
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))


# Late-bind the fallback once the SwiGLU class above is available.
if _SwiGLU is None:
    _SwiGLU = SwiGLU


class SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: T.Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)  # see https://arxiv.org/pdf/2002.05202 (to maintain # parameters)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = _ops().Linear(dim, hidden_dim, bias=False)
        self.w2 = _ops().Linear(hidden_dim, dim, bias=False)
        self.w3 = _ops().Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class WriteBackLayer(torch.nn.Module):
    """Cross attention -> residual."""

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        dim_qkv: int,
        num_heads: int = 4,
        dropout_prob: float = 0.0,
        use_rmsnorm: bool = True,
    ):
        super().__init__()

        # cross attention layer
        self.ca_layer = CrossAttentionLayer(
            dim_q=dim_q,
            dim_kv=dim_kv,
            dim_qkv=dim_qkv,
            num_heads=num_heads,
            dropout_prob=dropout_prob,
            use_rmsnorm=use_rmsnorm,
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        structural_attn_dict: T.Dict[str, T.Any] = None,
    ):
        """
        Args:
            query:
                (b, n, dim_q)
            key_value:
                (b, m, dim_kv)

        Returns:
            (b, n, dim_q)
        """

        # cross attention
        ca_out = self.ca_layer(
            query=query,
            key_value=key_value,
            structural_attn_dict=structural_attn_dict,
        )  # (b, n, dim_q)
        return ca_out + query  # (b, n, dim_q)


def get_pos_enc_cls(cls_name: str):
    if cls_name == "fourier":
        return FourierEmbed
    elif cls_name == "cube":
        return PosEncCube
    elif cls_name == "cube_fixed_init":
        return PosEncCubeFixedInit
    elif cls_name == "learnable_fourier":
        return PosEncLearnableFourier
    else:
        raise ValueError(f"{cls_name=}")


class FourierEmbed(torch.nn.Module):
    """sin/cos positional encodings"""

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
        self.min_freq_log2 = min_freq_log2
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.create_embedding_fn()

    def create_embedding_fn(self):
        d = self.dim_pos
        dim_out = 0
        if self.include_input:
            dim_out += d

        min_freq = self.min_freq_log2
        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs

        if self.log_sampling:
            # print(torch.linspace(min_freq, max_freq, steps=N_freqs))
            # print(2.0 **  torch.linspace(min_freq, max_freq, steps=N_freqs))
            freq_bands = 2.0 ** torch.linspace(min_freq, max_freq, steps=N_freqs)  # (nf,)
        else:
            freq_bands = torch.linspace(2.0**min_freq, 2.0**max_freq, steps=N_freqs)  # (nf,)

        assert freq_bands.isfinite().all(), f"nan: {freq_bands.isnan().any()} inf: {freq_bands.isinf().any()}"

        self.register_buffer("freq_bands", freq_bands)  # (nf,)
        self.dim_out = dim_out + d * self.freq_bands.numel() * 2

        # print(
        #     f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
        #     f"self.freq_bands.dtype: {self.freq_bands.dtype}\n"
        #     f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        # )

    def forward(
        self,
        pos: torch.Tensor,
        debug: bool = False,
    ):
        """
        Get the positional encoding for each coordinate.

        Args:
            pos:
                (*, dim_pos)

        Returns:
            out:
                (*, dim_positional_encoding)
        """

        out = []
        if self.include_input:
            out = [pos]  # (*, dim_pos)

        if debug:
            assert self.freq_bands.isfinite().all(), (
                f"nan: {self.freq_bands.isnan().any()} inf: {self.freq_bands.isinf().any()}"
            )

        # print(
        #     f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
        #     f"pos.dtype: {pos.dtype}\n"
        #     f"self.freq_bands.dtype: {self.freq_bands.dtype}\n"
        #     f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        # )

        pos = pos.unsqueeze(-1) * self.freq_bands  # (*b, d, nf)

        if debug:
            assert pos.isfinite().all(), f"nan: {pos.isnan().any()} inf: {pos.isinf().any()}"

        out += [
            torch.sin(pos).flatten(start_dim=-2),  # (*b, d*nf)
            torch.cos(pos).flatten(start_dim=-2),  # (*b, d*nf)
        ]

        # for ii in range(len(out)):
        #     print(f'out[{ii}].dtype: {out[ii].dtype}')

        out = torch.cat(out, dim=-1)  # (*b, 2 * dim_pos * nf (+ dim_pos))

        return out


class PosEncCube(torch.nn.Module):
    # https://github.com/Roblox/cube/blob/d799be4101b0cb67315498befe0b8d83e05af68d/cube3d/model/autoencoder/embedder.py

    def __init__(
        self,
        dim_pos: int,
        num_freqs: int,
        with_normal_pos_enc: bool = True,
        include_input: bool = True,
        # for legacy issue, unused
        min_freq_log2: float | None = None,
        max_freq_log2: float | None = None,
        log_sampling: bool | None = None,
    ):
        """
        Initializes the PhaseModulatedFourierEmbedder class.
        Args:
            num_freqs (int): The number of frequencies to be used.
            input_dim (int, optional): The dimension of the input. Defaults to 3.
        Attributes:
            weight (torch.nn.Parameter): The weight parameter initialized with random values.
            carrier (torch.Tensor): The carrier frequencies calculated based on the Nyquist-Shannon sampling theorem.
            out_dim (int): The output dimension calculated based on the input dimension and number of frequencies.
        """

        super().__init__()

        input_dim = dim_pos

        # This is not Eq. (1) in https://arxiv.org/abs/2503.15475
        self.weight = nn.Parameter(torch.randn(input_dim, num_freqs) * math.sqrt(0.5 * num_freqs))

        # NOTE this is the highest frequency we can get (2 for peaks, 2 for zeros, and 4 for interpolation points), see also https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem
        carrier = (num_freqs / 8) ** torch.linspace(1, 0, num_freqs)
        carrier = (carrier + torch.linspace(0, 1, num_freqs)) * 2 * torch.pi
        self.register_buffer("carrier", carrier, persistent=False)

        self.include_input = include_input
        self.dim_out = input_dim * num_freqs * 2
        if self.include_input:
            self.dim_out += input_dim

        self.with_normal_pos_enc = with_normal_pos_enc

    def forward(self, x):
        """
        Perform the forward pass of the embedder model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ..., input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, ..., output_dim) where
                          output_dim = input_dim + 2 * input_dim.
        """

        m = x.float().unsqueeze(-1)
        fm = (m * self.weight).view(*x.shape[:-1], -1)
        pm = (m * 0.5 * torch.pi + self.carrier).view(*x.shape[:-1], -1)

        embedding = []
        if self.include_input:
            embedding.append(x)

        if self.with_normal_pos_enc:
            embedding.extend([fm.cos() + pm.cos(), fm.sin() + pm.sin()])
        else:
            embedding.extend([pm.cos(), pm.sin()])

        embedding = torch.cat(embedding, dim=-1)

        return embedding


class PosEncCubeFixedInit(PosEncCube):
    def __init__(
        self,
        dim_pos: int,
        num_freqs: int,
        with_normal_pos_enc: bool = True,
        include_input: bool | None = None,
        # for legacy issue, unused
        min_freq_log2: float | None = None,
        max_freq_log2: float | None = None,
        log_sampling: bool | None = None,
    ):
        super().__init__(dim_pos, num_freqs, with_normal_pos_enc=with_normal_pos_enc, include_input=include_input)

        self.weight = nn.Parameter(torch.linspace(0, 0.5, num_freqs) * math.sqrt(0.5 * num_freqs))


class PosEncLearnableFourier(nn.Module):
    """Learnable Fourier from https://arxiv.org/abs/2106.02795 (Algorithm 1).

    Note, we follow the original paper to use the shared Wr and MLP for different groups.

    Modified from https://github.com/willGuimont/learnable_fourier_positional_encoding/blob/1fe87e28b41616bad875d2eeb3a488dfe3fbf698/src/learnable_fourier_positional_encoding/learnable_fourier_pos_encoding.py.
    """

    def __init__(
        self,
        *,
        dim_pos: int,
        num_freqs: int,
        hidden_dim: int = 32,
        num_groups: int = 1,
        D: int | None = None,
        gamma: float = 4.0,
        use_mlp: bool = True,
        include_input: bool = False,
        # for legacy issue, unused
        min_freq_log2: float | None = None,
        max_freq_log2: float | None = None,
        log_sampling: bool | None = None,
    ):
        """
        Args:
            input_dim (int):
                Total input dimension. Each input will be splitted into num_groups.
            num_groups (int):
                positional groups (positions in different groups are independent)
            num_freqs (int):
                depth of the Fourier feature dimension (for each sin and cos).
            H_dim (int):
                Defaults to 32 (see Sec. 4.1 and Sec. D).
                Hidden layer dimension.
                However, according to the Johnson–Lindenstrauss lemma (https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma),
                we may need large enough hidden dimension to keep the isometric properties.
            D (int | None, optional):
                Defaults to None.
                positional encoding dimension.
            gamma (float, optional):
                Defaults to 4.0.
                Parameter to initialize Wr and as a result, controlling the standard deviation of the
                gaussian kernel used to compute the similarity (std = gamma/sqrt(2)).
                (See Eq. (5) and Tab. 6 in https://arxiv.org/abs/2106.02795).
        """
        super().__init__()

        assert dim_pos % num_groups == 0, f"{dim_pos=}, {num_groups=}"
        self.input_dim = dim_pos
        self.G = num_groups
        self.M = self.input_dim // self.G
        self.F_dim = num_freqs * self.M  # 2 for cos and sin for each input dimension
        self.H_dim = hidden_dim
        self.gamma = gamma
        self.include_input = include_input

        # Projection matrix on learned lines (used in eq. 2)
        self.Wr = _ops().Linear(self.M, self.F_dim, bias=False)
        # MLP (GeLU(F @ W1 + B1) @ W2 + B2 (eq. 6)

        self.init_weights(self.Wr)

        if use_mlp:
            self.D = 2 * num_freqs * self.input_dim if D is None else D

            # for layernorm, see Sec. D
            self.mlp = nn.Sequential(
                # _ops().LayerNorm(2 * self.F_dim, elementwise_affine=False, eps=1e-6),
                _ops().Linear(2 * self.F_dim, self.H_dim, bias=True),
                nn.GELU(),
                # _ops().LayerNorm(self.H_dim, elementwise_affine=False, eps=1e-6),
                _ops().Linear(self.H_dim, self.D // self.G),
            )  # 2 for cos and sin

            self.mlp.apply(self.init_weights)
        else:
            self.D = 2 * num_freqs * self.input_dim
            self.mlp = None

        self.dim_out = self.D
        if self.include_input:
            self.dim_out += self.input_dim

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0, std=self.gamma**-2)  # Mean 0.0, standard deviation 0.02
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, raw_x):
        """
        Produce positional encodings from x
        :param x: tensor of shape [N, G, M] that represents N positions where each position is in the shape of [G, M],
                  where G is the positional group and each group has M-dimensional positional values.
                  Positions in different positional groups are independent
        :return: positional encoding for X
        """
        # N, G, M = x.shape
        assert raw_x.shape[-1] == self.input_dim, f"{raw_x.shape=}, {self.input_dim=}"
        # print(f"\n{raw_x.shape=}")
        x = raw_x.reshape(*raw_x.shape[:-1], self.G, self.M)
        # print(f"{x.shape=}")

        # Step 1. Compute Fourier features (eq. 2)
        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        # print(f"{projected.shape=}, {cosines.shape=}, {sines.shape=}")

        F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
        # print(f"{F.shape=}, {self.F_dim=}")

        if self.mlp is None:
            Y = F
        else:
            # Step 2. Compute projected Fourier features (eq. 6)
            Y = self.mlp(F)
        # print(f"{Y.shape=}")

        # Step 3. Reshape to x's shape
        PEx = Y.reshape(*raw_x.shape[:-1], self.D)

        if self.include_input:
            PEx = torch.cat((raw_x, PEx), dim=-1)

        # print(f"\n{PEx.shape=}\n")

        return PEx


class FourierGaussianEmbed(torch.nn.Module):
    """
    sin/cos positional encodings and
    the integrated positional encoding in a gaussian from mipnerf"""

    def __init__(
        self,
        dim_pos: int,
        include_input: bool,
        min_freq_log2: float,
        max_freq_log2: float,
        num_freqs: int,
        log_sampling: bool,
        return_fourier: bool,
    ):
        super().__init__()
        self.dim_pos = dim_pos
        self.include_input = include_input
        self.min_freq_log2 = min_freq_log2
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.create_embedding_fn()
        self.return_fourier = return_fourier

    def create_embedding_fn(self):
        min_freq = self.min_freq_log2
        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(min_freq, max_freq, steps=N_freqs)  # (nf,)
        else:
            freq_bands = torch.linspace(2.0**min_freq, 2.0**max_freq, steps=N_freqs)  # (nf,)

        assert freq_bands.isfinite().all(), f"nan: {freq_bands.isnan().any()} inf: {freq_bands.isinf().any()}"

        # # for simplicity, we create the P matrix in mipnerf paper
        # P = []
        # for i in range(N_freqs):
        #     P.append(freq_bands[i] * torch.eye(self.dim_pos))  # (dim_pos, dim_pos)
        # P = torch.cat(P, dim=1)  # (dim_pos, dim_pos * N_freq)
        # self.register_buffer("P", P)  # (dim_pos, dim_pos * N_freq)

        self.register_buffer("freq_bands", freq_bands)  # (N_freq,)
        self.dim_out = self.dim_pos * N_freqs * 2

        if self.include_input:
            self.dim_out += self.dim_pos

    def forward(
        self,
        pos: torch.Tensor,
        cov: torch.Tensor,
        debug: bool = False,
    ):
        """
        Get the positional encoding for each coordinate.

        Args:
            pos:
                (*b, dim_pos), mean of the gaussian
            cov:
                (*b, dim_pos, dim_pos) covariance of the gaussian

        Returns:
            fourier:
                (*, dim_positional_encoding) or None if return_fourier is False
            gaussian:
                (*, dim_positional_encoding)
        """
        ori_pos = pos
        if debug:
            assert pos.isfinite().all(), f"nan: {pos.isnan().any()} inf: {pos.isinf().any()}"

        # (*, 1, dim_pos) @  self.P (dim_pos, dim_out) -> (*, 1dim_out)
        # pos = linalg_utils.matmul(pos.unsqueeze(-2), self.P).squeeze(-2)  # (*b, dim_out)
        # # compute diag(P @ cov @ P^T)
        # diag_cov = (self.P * linalg_utils.matmul(cov, self.P)).sum(dim=-2)  # (*b, dim_out)

        # (xyz, 2xyz, 4xyz, ...)
        pos = (pos.unsqueeze(-2) * self.freq_bands.unsqueeze(-1)).flatten(start_dim=-2)  # (*b, dim_out)

        diag_cov = torch.diagonal(cov, dim1=-2, dim2=-1)  # (*b, dim_pos)
        # (xyz, 4xyz, 8xyz, ...)
        diag_cov = (diag_cov.unsqueeze(-2) * (self.freq_bands**2).unsqueeze(-1)).flatten(start_dim=-2)

        # compute exp(-0.5 * diag_cov)
        exp_diag_cov = torch.exp(-0.5 * diag_cov)  # (*b, dim_out)
        sin_pos = torch.sin(pos)
        cos_pos = torch.cos(pos)

        if self.return_fourier:
            if self.include_input:
                fourier_out = [
                    ori_pos,
                    sin_pos,  # (*b, dim_out)
                    cos_pos,  # (*b, dim_out)
                ]
            else:
                fourier_out = [
                    sin_pos,  # (*b, dim_out)
                    cos_pos,  # (*b, dim_out)
                ]
            fourier_out = torch.cat(fourier_out, dim=-1)  # (*b, 2 * dim_out) or (*b, 2 * dim_out + 3)
        else:
            fourier_out = None

        if self.include_input:
            gaussian_out = [
                ori_pos,
                sin_pos * exp_diag_cov,  # (*b, dim_out)
                cos_pos * exp_diag_cov,  # (*b, dim_out)
            ]
        else:
            gaussian_out = [
                sin_pos * exp_diag_cov,  # (*b, dim_out)
                cos_pos * exp_diag_cov,  # (*b, dim_out)
            ]
        gaussian_out = torch.cat(gaussian_out, dim=-1)  # (*b, 2 * dim_out) or (*b, 2 * dim_out + 3)

        return dict(
            fourier=fourier_out,  # (*b, 2 * dim_out) or None
            gaussian=gaussian_out,  # (*b, 2 * dim_out)
        )


class TimeEmbedder(nn.Module):
    """
    Embeds scalar timestamps into vector representations.
    It is composed of fourier positional encoding followed
    by MLP.
    """

    def __init__(
        self,
        dim_output: int,
        max_timestamp: float = 1.0,
        num_freqs: int = 16,
        mlp_dim_feature: int = 64,
    ):
        super().__init__()
        self.dim_output = dim_output
        self.max_timestamp = max_timestamp
        self.min_timestamp = 0

        # compute the base freq (min_freq)
        time_range = self.max_timestamp - self.min_timestamp  # always from [0, max_timestamp]
        min_omega = 2 * np.pi / time_range
        max_omega = min(min_omega * (2 ** (num_freqs - 1)), 2**16)  # 2 * np.pi / (time_range / num_freqs)
        min_omega_log2 = np.log2(min_omega)
        max_omega_log2 = np.log2(max_omega)

        self.fourier_embedder = FourierEmbed(
            dim_pos=1,
            include_input=False,  # we will add normalized t manually
            min_freq_log2=min_omega_log2,
            max_freq_log2=max_omega_log2,
            num_freqs=num_freqs,
            log_sampling=True,
        )

        self.mlp = nn.Sequential(
            _ops().Linear(self.fourier_embedder.dim_out + 1, mlp_dim_feature, bias=True),
            nn.SiLU(),
            _ops().Linear(mlp_dim_feature, self.dim_output, bias=True),
        )

    def forward(self, t: torch.Tensor, debug: bool = False):
        """
        Args:
            t:
                (*,)

        Returns:
            (*, dim_output)
        """
        t = t.unsqueeze(-1)  # (*, 1)
        t_encoded = self.fourier_embedder(t, debug=debug)  # (*, d)

        if debug:
            assert t_encoded.isfinite().all(), f"nan: {t_encoded.isnan().any()} inf: {t_encoded.isinf().any()}"

        # concat the orignal t (but normalize it based on max/min timestamp)
        t_normalized = (t - self.min_timestamp) / max(self.max_timestamp - self.min_timestamp, 1e-6)  # (*, 1)

        if debug:
            assert t_normalized.isfinite().all(), f"nan: {t_normalized.isnan().any()} inf: {t_normalized.isinf().any()}"

        t_normalized = t_normalized * 2 - 1  # range [-1, 1]
        t_input = torch.cat(
            [
                t_normalized,  # (*, 1)
                t_encoded,
            ],
            dim=-1,
        )  # (*, d+1)

        t_emb = self.mlp(t_input)  # (*, dim_output)

        if debug:
            assert t_emb.isfinite().all(), f"nan: {t_emb.isnan().any()} inf: {t_emb.isinf().any()}"

        return t_emb


class PluckerEmbed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dim_out = 6

    def forward(
        self,
        ray_origin_direction_w: torch.Tensor,
    ):
        """
        Get the positional encoding for each coordinate.

        Args:
            ray_origin_direction_w
                (*, 6), the first 3 dimension is ray origin, the next 3 dimension is ray direction.

        Returns:
            out:
                (*, 6)
        """
        roxrd = torch.cross(ray_origin_direction_w[..., :3], ray_origin_direction_w[..., 3:6], dim=-1)  # (*, 3)
        plucker = torch.cat([roxrd, ray_origin_direction_w[..., 3:6]], dim=-1)  # (*, 6)
        return plucker
