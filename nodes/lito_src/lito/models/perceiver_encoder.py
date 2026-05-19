#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements general perceiver encoder.
import math
import typing as T

from lito.models._comfy_mlp import Mlp

from lito.models.layers import SwiGLU as _SwiGLU

import torch
from torch import nn

from lito.models.layers import CrossAttentionLayer, SelfAttentionLayer, WriteBackLayer
from plibs import ppoint
_OPS = None
def _ops():
    global _OPS
    if _OPS is None:
        import comfy.ops
        _OPS = comfy.ops.manual_cast
    return _OPS



class PerceiverEncoderBlock(torch.nn.Module):
    """
    Each perceiver encoder block is composed of
    1. cross attention (latent -> tokens)
    2. self attention (latent -> latent)
    3. mlp
    """

    def __init__(
        self,
        dim_latent: int,
        dim_token: int,
        dim_qkv: int,
        num_self_attn: int = 2,
        num_self_heads: int = 4,
        num_cross_heads: int = 4,
        dropout_prob: float = 0.0,
        use_rmsnorm: bool = True,
        mlp_ratio: float = 2,
        mlp_type: str = "timm",
        linear_in_attn_add_bias: bool = True,
        mlp_add_bias: bool = True,
        use_layernorm_scaling: bool = False,  # see https://www.arxiv.org/abs/2502.05795
        layer_idx: int = None,  # [0, ... L-1]
        packed_kv: bool = False,
        add_kv_linear: bool = False,
    ):
        super().__init__()
        self.add_kv_linear = add_kv_linear
        self.packed_kv = packed_kv
        self.use_layernorm_scaling = use_layernorm_scaling
        self.layer_idx = layer_idx
        if self.use_layernorm_scaling:
            assert self.layer_idx is not None

        if self.add_kv_linear:
            self.kv_linear = _ops().Linear(
                in_features=dim_token,
                out_features=dim_token,
                bias=False,  # followed by layernorm
            )
        else:
            self.kv_linear = None

        self.ca_ln = _ops().LayerNorm(dim_latent, eps=1e-6)

        self.ca_layer = CrossAttentionLayer(
            dim_q=dim_latent,
            dim_kv=dim_token,
            dim_qkv=dim_qkv,
            num_heads=num_cross_heads,
            dropout_prob=dropout_prob,
            use_rmsnorm=use_rmsnorm,
            add_bias=linear_in_attn_add_bias,
            packed_kv=self.packed_kv,
        )

        mlp_hidden_dim = int(dim_latent * mlp_ratio)
        if mlp_type == "timm":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.ca_mlp = Mlp(
                in_features=dim_latent,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
                bias=mlp_add_bias,
            )
        elif mlp_type == "swiglu":
            self.ca_mlp = _SwiGLU(
                in_features=dim_latent,
                hidden_features=mlp_hidden_dim,
                out_features=None,
                bias=mlp_add_bias,
            )
        else:
            raise NotImplementedError

        # self attention blocks
        _self_attention_layers = []
        _mlp_layers = []
        _ln1_layers = []
        _ln2_layers = []
        for _ in range(num_self_attn):
            ln1 = _ops().LayerNorm(dim_latent, eps=1e-6)
            ln2 = _ops().LayerNorm(dim_latent, eps=1e-6)
            _ln1_layers.append(ln1)
            _ln2_layers.append(ln2)

            sa_layer = SelfAttentionLayer(
                dim_in=dim_latent,
                dim_qkv=dim_qkv,
                num_heads=num_self_heads,
                dropout_prob=dropout_prob,
                use_rmsnorm=use_rmsnorm,
                add_bias=linear_in_attn_add_bias,
            )
            _self_attention_layers.append(sa_layer)

            mlp_hidden_dim = int(dim_latent * mlp_ratio)
            if mlp_type == "timm":
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                mlp_layer = Mlp(
                    in_features=dim_latent,
                    hidden_features=mlp_hidden_dim,
                    act_layer=approx_gelu,
                    drop=0,
                    bias=mlp_add_bias,
                )
            elif mlp_type == "swiglu":
                mlp_layer = _SwiGLU(
                    in_features=dim_latent,
                    hidden_features=mlp_hidden_dim,
                    out_features=None,
                    bias=mlp_add_bias,
                )
            else:
                raise NotImplementedError
            _mlp_layers.append(mlp_layer)

        self.ln1_layers = nn.ModuleList(_ln1_layers)
        self.ln2_layers = nn.ModuleList(_ln2_layers)
        self.sa_layers = nn.ModuleList(_self_attention_layers)
        self.mlp_layers = nn.ModuleList(_mlp_layers)

    def forward(
        self,
        latents: torch.Tensor,
        input_tokens: torch.Tensor,
        self_structural_attn_dict: T.Union[T.List[T.Dict[str, T.Any]], T.Dict[str, T.Any]] = None,
        cross_structural_attn_dict: T.Dict[str, T.Any] = None,
        packed_kv_coord: ppoint.PackedPoint = None,
    ):
        """
        Args:
            latents:
                (b, n, dim_latent)
            input_tokens:
                (b, m, dim_token)
            self_structural_attn_dict:
                attn_bias used during self attention. eg, LowerTriangularMask.
                It can be None (no attn bias), AttentionBias, which indicates
                all self attention layers use the same attn_bias,
                or a list of attn_bias, one for each self attention layer.
            cross_structural_attn_dict:
                structural attention dict for cross attention
            packed_kv_coord:
                packed coordinate of key_value, needed if packed_kv is True

        Returns:
            latents:
                (b, n, dim_latent)
        """
        if not isinstance(self_structural_attn_dict, (list, tuple)):
            self_structural_attn_dict = [self_structural_attn_dict] * len(self.sa_layers)

        if self.kv_linear is not None:
            input_tokens = self.kv_linear(input_tokens)  # (b, m, dim_token)

        # cross attention (latent -> input token)
        latents = latents + self.ca_layer(
            query=latents,
            key_value=input_tokens,
            structural_attn_dict=cross_structural_attn_dict,
            packed_kv_coord=packed_kv_coord,
        )  # (b, n, dim_latent)
        latents = latents + self.ca_mlp(self.ca_ln(latents))  # (b, n, dim_latent)

        layernorm_scaler = math.sqrt(1.0 / (self.layer_idx + 1))

        # self attention (latent -> latent)
        for ln1, sa_layer, ln2, mlp_layer, sdict in zip(
            self.ln1_layers,
            self.sa_layers,
            self.ln2_layers,
            self.mlp_layers,
            self_structural_attn_dict,
        ):
            latents = latents + sa_layer(
                x=ln1(latents) if not self.use_layernorm_scaling else layernorm_scaler * ln1(latents),
                structural_attn_dict=sdict,
            )  # (b, n, dim_latent)
            latents = latents + mlp_layer(
                ln2(latents) if not self.use_layernorm_scaling else layernorm_scaler * ln2(latents)
            )  # (b, n, dim_latent)

        return latents  # (b, n, dim_latent)


class PerceiverEncoder(nn.Module):
    """
    General perceiver encoder that takes a set of input tokens and
    a set of latent tokens. For each layer, the latent tokens use
    cross attention to gather info from input tokens, then
    perform self attention among latents.
    """

    def __init__(
        self,
        dim_latent: int,
        dim_token: int,
        num_blocks: int,
        dim_qkv: int,
        # dim_cond_feature: int = None,
        # dim_cond_tokens: int = None,
        num_self_attn: int = 2,
        num_self_heads: int = 4,
        num_cross_heads: int = 4,
        dropout_prob: float = 0.0,
        use_rmsnorm: bool = True,
        mlp_ratio: float = 2,
        add_write_back: bool = False,
        keep_block_bug: bool = True,
        mlp_type: str = "timm",
        linear_in_attn_add_bias: bool = True,
        mlp_add_bias: bool = True,
        add_kv_linear: bool = False,
        use_layernorm_scaling: bool = False,
        layer_idx: int = None,
        packed_kv: bool = False,
    ):
        """

        Args:
            dim_latent:
                feature dimension of the latent vectors
            dim_token:
                feature dimension of the input tokens
            num_blocks:
                number of encoder blocks to use
            dim_qkv:
                dimension of the qkv used in cross and self attention
            # dim_cond_feature:
            #     if not None, the dimension of the conditional feature (e.g., time, text)
            # dim_cond_tokens:
            #     if not None, dimension of the conditional tokens that will be the
            #     key and value tokens of the cross attention at the 0-th, 2-th, ..., blocks.
            num_self_attn:
                number of self attention in each encoder block
            num_self_heads:
                number of self attention heads in each encoder block
            num_cross_heads
                number of cross attention heads in each encoder block
            dropout_prob:
                dropout prob
            use_rmsnorm:
                whether to use rmsnorm (normalize the mean and std during cross and
                self attention at the output of the Wq, Wk, Wv)
            mlp_ratio:
                mlp expansion ratio
            add_write_back:
                whether to write info back to input tokens
            keep_block_bug:
                We had a bug of not updating the latents. To be backward compatible,
                we keep the option.  To avoid the bug, set to False.
            packed_kv:
                whether the key and value are in packed format, ie (n1+n2+..._nb, d)
            add_kv_linear:
                whether to add a linear layer to input tokens before each block
        """
        super().__init__()
        self.dim_latent = dim_latent
        self.dim_token = dim_token
        self.add_write_back = add_write_back
        self.num_blocks = num_blocks
        self.keep_block_bug = keep_block_bug
        self.use_layernorm_scaling = use_layernorm_scaling
        self.layer_idx = layer_idx
        if self.layer_idx is None:
            self.layer_idx = 0
        self.packed_kv = packed_kv

        if self.packed_kv:
            assert not self.add_write_back, f"not implemented"

        # encoder blocks
        self.blocks = nn.ModuleList(
            [
                PerceiverEncoderBlock(
                    dim_latent=dim_latent,
                    dim_token=dim_token,
                    dim_qkv=dim_qkv,
                    num_self_attn=num_self_attn,
                    num_self_heads=num_self_heads,
                    num_cross_heads=num_cross_heads,
                    dropout_prob=dropout_prob,
                    use_rmsnorm=use_rmsnorm,
                    mlp_ratio=mlp_ratio,
                    mlp_type=mlp_type,
                    linear_in_attn_add_bias=linear_in_attn_add_bias,
                    mlp_add_bias=mlp_add_bias,
                    use_layernorm_scaling=use_layernorm_scaling,
                    layer_idx=_layer_idx + self.layer_idx,
                    packed_kv=packed_kv,
                    add_kv_linear=add_kv_linear,
                )
                for _layer_idx in range(num_blocks)
            ]
        )

        # write back
        if self.add_write_back:
            self.write_back_blocks = []

            for _ in range(num_blocks - 1):
                layer = WriteBackLayer(
                    dim_q=dim_token,
                    dim_kv=dim_latent,
                    dim_qkv=dim_qkv,
                    num_heads=num_cross_heads,
                    dropout_prob=dropout_prob,
                    use_rmsnorm=use_rmsnorm,
                )
                self.write_back_blocks.append(layer)

            self.write_back_blocks = torch.nn.ModuleList(self.write_back_blocks)
        else:
            self.write_back_blocks = None

    def forward(
        self,
        input_tokens: torch.Tensor,
        latent_tokens: torch.Tensor,
        # cond_feature: T.Optional[torch.Tensor] = None,
        # cond_tokens: T.Optional[torch.Tensor] = None,
        structural_attn_dicts: T.List[T.Dict[str, T.Dict[str, T.Any]]] = None,
        return_all_layers: bool = False,
        packed_kv_coord: ppoint.PackedPoint = None,
    ) -> T.Dict[str, T.List[torch.Tensor]]:
        r"""
        Args:
            input_tokens:
                (b, m, dim_token) or (bm, dim_token) if packed_kv is true
            latent_tokens:
                (b, n, dim_latent)
            structural_attn_dicts:
                a list of dict(str, structural_attn_dict), one for each layer.
                each dict contains
                    cross: structural_attn_dict to be used for cross attention.
                    self: list of structural_attn_dict to be used for each layer of self attention in the block.
                    writeback: structural_attn_dict to be used for writeback.
            packed_kv_coord:
                packed coordinate of key_value, needed if packed_kv is True

        Returns:
            if return_all_layers == True:
                all_layer_latents:
                    a list of num_blocks, each is (b, n, dim_latent)
                    all_layer_latents[i] is the output latent of the i-th block
                all_layer_tokens:
                    a list of num_blocks, each is (b, m, dim_token)
                    all_layer_tokens[i] is the resulted input tokens of the i-th block.
            else:
                latent_tokens:
                    (b, n, dim_latent) final layer's latent output
        """

        # b, m, _dim_token = input_tokens.shape
        # _b, n, _dim_latent = latent_tokens.shape

        if structural_attn_dicts is not None:
            assert len(structural_attn_dicts) == len(self.blocks)

        all_layer_latents = []
        all_layer_tokens = []
        for i, block in enumerate(self.blocks):
            if structural_attn_dicts is not None:
                self_struct_attn_dict = structural_attn_dicts[i].get("self", None)
                cross_struct_attn_dict = structural_attn_dicts[i].get("cross", None)
            else:
                self_struct_attn_dict = None
                cross_struct_attn_dict = None

            # run through the block
            latents = block(
                latents=latent_tokens,
                input_tokens=input_tokens,
                self_structural_attn_dict=self_struct_attn_dict,
                cross_structural_attn_dict=cross_struct_attn_dict,
                packed_kv_coord=packed_kv_coord,
            )
            if return_all_layers:
                all_layer_latents.append(latents)

            # update the latent tokens
            if not self.keep_block_bug:
                latent_tokens = latents

            # write back
            if self.add_write_back and i < (len(self.blocks) - 1):
                if structural_attn_dicts is not None:
                    writeback_struct_attn_dict = structural_attn_dicts[i].get("writeback", None)
                else:
                    writeback_struct_attn_dict = None

                layer = self.write_back_blocks[i]
                input_tokens = layer(
                    query=input_tokens,
                    key_value=latents,
                    structural_attn_dict=writeback_struct_attn_dict,
                )
            if return_all_layers:
                all_layer_tokens.append(input_tokens)

        if return_all_layers:
            return dict(
                all_layer_latents=all_layer_latents,
                all_layer_tokens=all_layer_tokens,
            )
        else:  # added after bug fixed
            return latent_tokens
