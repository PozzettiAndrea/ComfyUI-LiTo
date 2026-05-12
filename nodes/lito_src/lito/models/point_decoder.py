#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements point decoders that take input tokens and
# shape latents as conditioning and output the target tokens.


import math
import traceback
import typing as T

from lito.models._comfy_mlp import Mlp

from lito.models.layers import SwiGLU as _SwiGLU

import torch
import torch.nn.functional

from lito.models.layers import PointwiseResnet
from plibs import linalg_utils, sh_utils, utils
from plibs.ppoint import PackedPoint

# ruff: isort: off
# Trellis integration removed for the ComfyUI wrapper (mesh decoding only)
try:
    import lito.integrations.trellis.representations.mesh as trellis_mesh
    import trellis.models.structured_latent_vae.decoder_mesh as slat_vae_mesh
    import trellis.modules.sparse as sp

    TRELLIS_IMPORTED = True
except ImportError:
    # Optional on macOS / non-CUDA installs: TRELLIS's mesh decoder pulls in
    # flash_attn which isn't available everywhere. Code paths that need these
    # modules (mesh decoder) will fail at runtime; the rest of the file works.
    print(f"\n{__file__=}")
    print("\n\nTrellis not imported, need to do `bash env/scripts/setup_trellis.sh`. Traceback is")
    traceback.print_exc()
    print("\n\n")
    trellis_mesh = None
    slat_vae_mesh = None
    sp = None
    TRELLIS_IMPORTED = False
# ruff: isort: on

# try:
#     import third_party.TRELLIS.trellis.models.structured_latent_vae.decoder_mesh as slat_vae_mesh
#     import third_party.TRELLIS.trellis.modules.sparse as sp
#     import third_party.TRELLIS.trellis.representations.mesh as trellis_mesh
#
#     TRELLIS_IMPORTED = True
# except:
#     print(f"\n{__file__=}")
#     print("\n\nTrellis not imported, need to do `bash environment/setup_trellis.sh`. Traceback is")
#     traceback.print_exc()
#     print("\n\n")
#     TRELLIS_IMPORTED = False


from lito.models.layers import CrossAttentionLayer, FinalLayer, SelfAttentionLayer, get_pos_enc_cls, modulate
from lito.models.linear import StackedLinearLayers
from lito.models.perceiver_encoder import PerceiverEncoder
from lito.models.spoint_encoder import SPointPerceiverEncoder
from lito.models.vector_decoder import VectorDecoder
from contextlib import nullcontext as _nullcontext
_OPS = None
def _ops():
    global _OPS
    if _OPS is None:
        import comfy.ops
        _OPS = comfy.ops.manual_cast
    return _OPS



class VelocityMLPDecoder(torch.nn.Module):
    """
    Velocity mlp decoder that is composed of a mlp. The input of the mlp
    is composed of an input point and the flattened shape latent.
    """

    def __init__(
        self,
        num_latent: int,
        dim_latent: int,
        dim_point: int,
        # mlp
        mlp_num_layers: int,
        mlp_dim_features: T.Union[int, T.List[int]],
        mlp_activation: str = "silu",
        mlp_add_layernorm: bool = True,
        # positional encoding
        min_freq_log2: float = 0,
        max_freq_log2: float = 12,
        num_freqs: int = 32,
        dim_cond_feature: int = None,
        dim_output: int = None,
        pos_enc_name_xyz: str = "fourier",
    ):
        """
        Args:
            num_latent:
                number of latent per shape
            dim_latent:
                dimension of each latent
            dim_point:
                dimension of the input point (3xyz + ...)
            dim_cond_feature:
                dimension of additional conditioning feature vector, None if not needed
            mlp_num_layers:
                number of layers in the mlp
            mlp_dim_features
                int or (mlp_num_layers-1,) dimension of the hidden layers of the mlp
            mlp_activation:
                mlp activation function
            mlp_add_layernorm:
                whether to add layernorm in mlp
            dim_output:
                output dimension of the model, if None, the same as dim_point
        """

        super().__init__()
        self.num_latent = num_latent
        self.dim_latent = dim_latent
        self.dim_point = dim_point
        self.dim_cond_feature = dim_cond_feature
        self.dim_output = dim_output if dim_output is not None else dim_point

        # position encoding for point
        self.xyz_pos_encoder = get_pos_enc_cls(pos_enc_name_xyz)(
            dim_pos=3,  # 3xyz
            include_input=False,  # we concat ourselves
            max_freq_log2=max_freq_log2,
            min_freq_log2=min_freq_log2,
            num_freqs=num_freqs,
            log_sampling=True,
        )

        # calculate the point token dimension
        self.dim_point_token = (
            self.dim_point + self.xyz_pos_encoder.dim_out + self.dim_cond_feature + self.num_latent * self.dim_latent
        )

        # mlp
        self.mlp = StackedLinearLayers(
            num_layers=mlp_num_layers,
            dim_input=self.dim_point_token,
            dim_output=self.dim_output,
            dim_features=mlp_dim_features,
            nonlinearity=mlp_activation,
            add_norm_layer=mlp_add_layernorm,
        )

    def forward(
        self,
        input_point_cloud: torch.Tensor,  # (b, m, dim_point)
        latent_tokens: torch.Tensor,  # (b, num_latent, dim_latent)
        cond_feature: torch.Tensor = None,  # (b, dim_cond_feature)
    ):
        """
        Args:
            input_point_cloud:
                (b, m, dim_point)  The first 3 dimension is xyz, then it can be rgb, normal, etc
            latent_tokens:
                (b, num_latent, dim_latent)  The latent representing the shape
            cond_feature:
                (b, dim_cond_feature)

        Returns:
            output_velocity:
                (b, m, dim_point)  output velocity at each point
        """

        b, m, _dim_point = input_point_cloud.shape
        _b, _num_latent, _dim_latent = latent_tokens.shape

        # flatten latent
        latent_tokens = latent_tokens.reshape(b, 1, self.num_latent * self.dim_latent).expand(
            -1, m, -1
        )  # (b, m, num_latent * dim_latent)

        # position encode the points
        encoded_xyz = self.xyz_pos_encoder(input_point_cloud[..., :3])  # (b, m, dim_encoded_xyz)
        input_tokens = [
            input_point_cloud,  # (b, m, dim_point)
            encoded_xyz,  # (b, m, dim_encoded_xyz)
            latent_tokens,  # (b, m, num_latent * dim_latent)
        ]  # (b, m, dim_point_token)

        if cond_feature is not None:
            assert cond_feature.shape == (b, self.dim_cond_feature)
            cond_feature = cond_feature.reshape(b, 1, self.dim_cond_feature).expand(
                -1, m, -1
            )  # (b, m, dim_cond_feature)
            input_tokens.append(cond_feature)

        input_tokens = torch.cat(input_tokens, dim=-1)
        assert input_tokens.size(-1) == self.dim_point_token

        # mlp
        out = self.mlp(input_tokens)  # (b, m, dim_point)
        return out


class DecoderBlock(torch.nn.Module):
    def __init__(
        self,
        dim_latent: int,
        dim_token: int,
        dim_qkv: int,
        dim_cond_feature: int,
        num_self_attn: int = 0,
        num_self_heads: int = 4,
        num_cross_heads: int = 4,
        dropout_prob: float = 0.0,
        use_rmsnorm: bool = True,
        mlp_ratio: float = 4,
        eps: float = 1e-6,
        mlp_type: str = "timm",
        linear_in_attn_add_bias: bool = True,
        mlp_add_bias: bool = True,
        use_cross_attn_layernorm1: bool = True,
        use_layernorm_scaling: bool = False,
        layer_idx: int = None,  # [0 ... L-1]
        packed_kv: bool = False,
    ):
        super().__init__()
        self.dim_latent = dim_latent
        self.dim_token = dim_token
        self.dim_qkv = dim_qkv
        self.dim_cond_feature = dim_cond_feature
        self.num_self_attn = num_self_attn
        self.dropout_prob = dropout_prob
        self.use_rmsnorm = use_rmsnorm
        self.eps = eps
        self.use_layernorm_scaling = use_layernorm_scaling
        self.layer_idx = layer_idx
        self.packed_kv = packed_kv

        if self.use_layernorm_scaling:
            assert self.layer_idx is not None

        # cond_feature -> adaln modulators
        if self.dim_cond_feature is not None and self.dim_cond_feature > 0:
            self.ca_modulator = torch.nn.Sequential(
                _ops().Linear(
                    self.dim_cond_feature,
                    self.dim_cond_feature,
                    bias=True,
                ),
                torch.nn.SiLU(),
                _ops().Linear(
                    self.dim_cond_feature,
                    6 * self.dim_token,
                    bias=True,
                ),
            )
        else:
            self.ca_modulator = None

        if use_cross_attn_layernorm1:
            self.ca_ln1 = _ops().LayerNorm(
                self.dim_token,
                elementwise_affine=(self.ca_modulator is None),
                eps=self.eps,
            )
        else:
            self.ca_ln1 = torch.nn.Identity()
        self.ca_ln2 = _ops().LayerNorm(
            self.dim_token,
            elementwise_affine=(self.ca_modulator is None),
            eps=self.eps,
        )

        # cross attention (input token -> latent)
        self.ca_layer = CrossAttentionLayer(
            dim_q=self.dim_token,
            dim_kv=self.dim_latent,
            dim_qkv=self.dim_qkv,
            num_heads=num_cross_heads,
            dropout_prob=dropout_prob,
            use_rmsnorm=self.use_rmsnorm,
            add_bias=linear_in_attn_add_bias,
            packed_kv=self.packed_kv,
        )  # output dimension is dim_token

        mlp_hidden_dim = int(self.dim_token * mlp_ratio)
        if mlp_type == "timm":
            approx_gelu = lambda: torch.nn.GELU(approximate="tanh")
            self.ca_mlp = Mlp(
                in_features=self.dim_token,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
                bias=mlp_add_bias,
            )  # v_channels -> v_channels
        elif mlp_type == "swiglu":
            self.ca_mlp = _SwiGLU(
                in_features=self.dim_token,
                hidden_features=mlp_hidden_dim,
                out_features=None,
                bias=mlp_add_bias,
            )
        elif mlp_type.startswith("resnet_"):
            # 'resnet_gelu', 'resnet_silu', 'resnet_swiglu'
            activation_fn = mlp_type.split("resnet_", 1)[1]
            self.ca_mlp = PointwiseResnet(
                dim_in=self.dim_token,
                dim_hidden=mlp_hidden_dim,
                dim_out=self.dim_token,
                bias=mlp_add_bias,
                activation_fn=activation_fn,
                add_init_activation=False,
            )
        else:
            raise NotImplementedError

        # self attention
        _self_attention_layers = []
        _mlp_layers = []
        _sa_modulator_layers = []
        _ln1_layers = []
        _ln2_layers = []
        for _ in range(self.num_self_attn):
            if self.dim_cond_feature > 0:
                mod_layer = torch.nn.Sequential(
                    _ops().Linear(self.dim_cond_feature, self.dim_cond_feature, bias=True),
                    torch.nn.SiLU(),
                    _ops().Linear(self.dim_cond_feature, 6 * self.dim_token, bias=True),
                )

            else:
                mod_layer = None
            _sa_modulator_layers.append(mod_layer)

            ln1 = _ops().LayerNorm(self.dim_token, elementwise_affine=(mod_layer is None), eps=self.eps)
            ln2 = _ops().LayerNorm(self.dim_token, elementwise_affine=(mod_layer is None), eps=self.eps)
            _ln1_layers.append(ln1)
            _ln2_layers.append(ln2)

            sa_layer = SelfAttentionLayer(
                dim_in=self.dim_token,
                dim_qkv=self.dim_qkv,
                num_heads=num_self_heads,
                dropout_prob=dropout_prob,
                use_rmsnorm=use_rmsnorm,
                add_bias=linear_in_attn_add_bias,
            )
            _self_attention_layers.append(sa_layer)

            mlp_hidden_dim = int(self.dim_token * mlp_ratio)
            if mlp_type == "timm":
                approx_gelu = lambda: torch.nn.GELU(approximate="tanh")
                mlp_layer = Mlp(
                    in_features=self.dim_token,
                    hidden_features=mlp_hidden_dim,
                    act_layer=approx_gelu,
                    drop=0,
                    bias=mlp_add_bias,
                )
            elif mlp_type == "swiglu":
                mlp_layer = _SwiGLU(
                    in_features=self.dim_token,
                    hidden_features=mlp_hidden_dim,
                    out_features=None,
                    bias=mlp_add_bias,
                )
            else:
                raise NotImplementedError
            _mlp_layers.append(mlp_layer)

        self.ln1_layers = torch.nn.ModuleList(_ln1_layers)
        self.ln2_layers = torch.nn.ModuleList(_ln2_layers)
        self.sa_modulator_layers = torch.nn.ModuleList(_sa_modulator_layers)
        self.sa_layers = torch.nn.ModuleList(_self_attention_layers)
        self.mlp_layers = torch.nn.ModuleList(_mlp_layers)

    def forward(
        self,
        latents: torch.Tensor,  # key/value
        input_tokens: torch.Tensor,  # query
        cond_feature: torch.Tensor = None,
        self_structural_attn_dict: T.Union[T.List[T.Dict[str, T.Any]], T.Dict[str, T.Any]] = None,
        cross_structural_attn_dict: T.Dict[str, T.Any] = None,
        debug: bool = False,
        latent_coord: T.Optional[PackedPoint] = None,
    ):
        """
        Args:
            latents:
                (b, n, dim_latent)
            input_tokens:
                (b, m, dim_token)
            cond_feature:
                (b, dim_cond_feature)
            self_structural_attn_dict:
            cross_structural_attn_dict:

        Returns:
            (b, m, dim_token)
        """

        current_layer_idx = self.layer_idx if self.layer_idx is not None else 0
        current_layer_idx += 1
        layernorm_scaler = math.sqrt(1.0 / current_layer_idx)

        if cond_feature is not None:
            # compute modulators
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ca_modulator(cond_feature).chunk(
                6, dim=1
            )  # list of (b, dim_token), https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L119

            if debug:
                assert shift_msa.isfinite().all(), f"nan: {shift_msa.isnan().any()}, inf: {shift_msa.isinf().any()}"
                assert scale_msa.isfinite().all(), f"nan: {scale_msa.isnan().any()}, inf: {scale_msa.isinf().any()}"
                assert gate_msa.isfinite().all(), f"nan: {gate_msa.isnan().any()}, inf: {gate_msa.isinf().any()}"
                assert shift_mlp.isfinite().all(), f"nan: {shift_mlp.isnan().any()}, inf: {shift_mlp.isinf().any()}"
                assert scale_mlp.isfinite().all(), f"nan: {scale_mlp.isnan().any()}, inf: {scale_mlp.isinf().any()}"
                assert gate_mlp.isfinite().all(), f"nan: {gate_mlp.isnan().any()}, inf: {gate_mlp.isinf().any()}"

            normed_input_tokens = self.ca_ln1(input_tokens)
            if self.use_layernorm_scaling:
                normed_input_tokens = layernorm_scaler * normed_input_tokens

            modulated_input_tokens = modulate(x=normed_input_tokens, shift=shift_msa, scale=scale_msa)

            if debug:
                assert normed_input_tokens.isfinite().all(), (
                    f"nan: {normed_input_tokens.isnan().any()}, inf: {normed_input_tokens.isinf().any()}"
                )
                assert modulated_input_tokens.isfinite().all(), (
                    f"nan: {modulated_input_tokens.isnan().any()}, inf: {modulated_input_tokens.isinf().any()}"
                )

            input_tokens = input_tokens + gate_msa.unsqueeze(1) * self.ca_layer(
                query=modulated_input_tokens,
                key_value=latents,
                structural_attn_dict=cross_structural_attn_dict,
                packed_kv_coord=latent_coord,
            )

            normed_input_tokens = self.ca_ln2(input_tokens)
            if self.use_layernorm_scaling:
                normed_input_tokens = layernorm_scaler * normed_input_tokens
            modulated_input_tokens = modulate(normed_input_tokens, shift_mlp, scale_mlp)

            if debug:
                assert normed_input_tokens.isfinite().all(), (
                    f"nan: {normed_input_tokens.isnan().any()}, inf: {normed_input_tokens.isinf().any()}"
                )
                assert modulated_input_tokens.isfinite().all(), (
                    f"nan: {modulated_input_tokens.isnan().any()}, inf: {modulated_input_tokens.isinf().any()}"
                )

            input_tokens = input_tokens + gate_mlp.unsqueeze(1) * self.ca_mlp(modulated_input_tokens)
        else:
            normed_input_tokens = self.ca_ln1(input_tokens)
            if self.use_layernorm_scaling:
                normed_input_tokens = layernorm_scaler * normed_input_tokens
            if debug:
                assert normed_input_tokens.isfinite().all(), (
                    f"nan: {normed_input_tokens.isnan().any()}, inf: {normed_input_tokens.isinf().any()}"
                )

            input_tokens = input_tokens + self.ca_layer(
                query=normed_input_tokens,
                key_value=latents,
                structural_attn_dict=cross_structural_attn_dict,
                packed_kv_coord=latent_coord,
            )

            normed_input_tokens = self.ca_ln2(input_tokens)
            if self.use_layernorm_scaling:
                normed_input_tokens = layernorm_scaler * normed_input_tokens
            if debug:
                assert normed_input_tokens.isfinite().all(), (
                    f"nan: {normed_input_tokens.isnan().any()}, inf: {normed_input_tokens.isinf().any()}"
                )

            input_tokens = input_tokens + self.ca_mlp(normed_input_tokens)

        # self attention
        if not isinstance(self_structural_attn_dict, (list, tuple)):
            self_structural_attn_dict = [self_structural_attn_dict] * self.num_self_attn

        for mod_layer, ln1, sa_layer, ln2, mlp_layer, s_attn_dict in zip(
            self.sa_modulator_layers,
            self.ln1_layers,
            self.sa_layers,
            self.ln2_layers,
            self.mlp_layers,
            self_structural_attn_dict,
        ):
            if cond_feature is not None:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_layer(cond_feature).chunk(6, dim=1)

                if debug:
                    assert shift_msa.isfinite().all(), f"nan: {shift_msa.isnan().any()}, inf: {shift_msa.isinf().any()}"
                    assert scale_msa.isfinite().all(), f"nan: {scale_msa.isnan().any()}, inf: {scale_msa.isinf().any()}"
                    assert gate_msa.isfinite().all(), f"nan: {gate_msa.isnan().any()}, inf: {gate_msa.isinf().any()}"
                    assert shift_mlp.isfinite().all(), f"nan: {shift_mlp.isnan().any()}, inf: {shift_mlp.isinf().any()}"
                    assert scale_mlp.isfinite().all(), f"nan: {scale_mlp.isnan().any()}, inf: {scale_mlp.isinf().any()}"
                    assert gate_mlp.isfinite().all(), f"nan: {gate_mlp.isnan().any()}, inf: {gate_mlp.isinf().any()}"

                normed_input_tokens = ln1(input_tokens)
                if self.use_layernorm_scaling:
                    normed_input_tokens = layernorm_scaler * normed_input_tokens
                modulated_input_tokens = modulate(x=normed_input_tokens, shift=shift_msa, scale=scale_msa)

                if debug:
                    assert normed_input_tokens.isfinite().all(), (
                        f"nan: {normed_input_tokens.isnan().any()}, inf: {normed_input_tokens.isinf().any()}"
                    )
                    assert modulated_input_tokens.isfinite().all(), (
                        f"nan: {modulated_input_tokens.isnan().any()}, inf: {modulated_input_tokens.isinf().any()}"
                    )

                input_tokens = input_tokens + (
                    gate_msa.unsqueeze(1)
                    * sa_layer(
                        x=modulated_input_tokens,
                        structural_attn_dict=s_attn_dict,
                    )
                )

                normed_input_tokens = ln2(input_tokens)
                if self.use_layernorm_scaling:
                    normed_input_tokens = layernorm_scaler * normed_input_tokens
                modulated_input_tokens = modulate(x=normed_input_tokens, shift=shift_mlp, scale=scale_mlp)

                if debug:
                    assert normed_input_tokens.isfinite().all(), (
                        f"nan: {normed_input_tokens.isnan().any()}, inf: {normed_input_tokens.isinf().any()}"
                    )
                    assert modulated_input_tokens.isfinite().all(), (
                        f"nan: {modulated_input_tokens.isnan().any()}, inf: {modulated_input_tokens.isinf().any()}"
                    )

                input_tokens = input_tokens + gate_mlp.unsqueeze(1) * mlp_layer(modulated_input_tokens)
            else:
                normed_input_tokens = ln1(input_tokens)
                if self.use_layernorm_scaling:
                    normed_input_tokens = layernorm_scaler * normed_input_tokens

                if debug:
                    assert normed_input_tokens.isfinite().all(), (
                        f"nan: {normed_input_tokens.isnan().any()}, inf: {normed_input_tokens.isinf().any()}"
                    )

                input_tokens = input_tokens + sa_layer(
                    x=normed_input_tokens,
                    structural_attn_dict=s_attn_dict,
                )

                normed_input_tokens = ln2(input_tokens)
                if self.use_layernorm_scaling:
                    normed_input_tokens = layernorm_scaler * normed_input_tokens

                if debug:
                    assert normed_input_tokens.isfinite().all(), (
                        f"nan: {normed_input_tokens.isnan().any()}, inf: {normed_input_tokens.isinf().any()}"
                    )

                input_tokens = input_tokens + mlp_layer(normed_input_tokens)

            # update layernorm_scaler
            current_layer_idx += 1
            layernorm_scaler = math.sqrt(1.0 / current_layer_idx)

        return input_tokens


class SelfDecoderBlock(torch.nn.Module):
    def __init__(
        self,
        dim_token: int,
        dim_qkv: int,
        num_self_attn: int = 1,
        upsample_ratio: int = 1,
        num_self_heads: int = 4,
        dropout_prob: float = 0.0,
        use_rmsnorm: bool = True,
        mlp_ratio: float = 4,
        eps: float = 1e-6,
        mlp_type: str = "timm",
        linear_in_attn_add_bias: bool = True,
        mlp_add_bias: bool = True,
        dim_cond_feature: int = 0,
        use_layernorm_scaling: bool = False,
        layer_idx: int = None,  # [0 ... L-1]
    ):
        super().__init__()
        self.dim_token = dim_token
        self.dim_qkv = dim_qkv
        self.num_self_attn = num_self_attn
        self.upsample_ratio = upsample_ratio
        self.dropout_prob = dropout_prob
        self.use_rmsnorm = use_rmsnorm
        self.eps = eps
        self.dim_cond_feature = dim_cond_feature
        self.use_layernorm_scaling = use_layernorm_scaling
        self.layer_idx = layer_idx
        if self.use_layernorm_scaling:
            assert self.layer_idx is not None

        # self attention
        _self_attention_layers = []
        _mlp_layers = []
        _sa_modulator_layers = []
        _ln1_layers = []
        _ln2_layers = []
        for layer_idx in range(self.num_self_attn):
            if self.dim_cond_feature > 0:
                mod_layer = torch.nn.Sequential(
                    _ops().Linear(self.dim_cond_feature, self.dim_cond_feature, bias=True),
                    torch.nn.SiLU(),
                    _ops().Linear(self.dim_cond_feature, 6 * self.dim_token, bias=True),
                )
            else:
                mod_layer = None
            _sa_modulator_layers.append(mod_layer)

            ln1 = _ops().LayerNorm(self.dim_token, elementwise_affine=(mod_layer is None), eps=self.eps)
            ln2 = _ops().LayerNorm(self.dim_token, elementwise_affine=(mod_layer is None), eps=self.eps)
            _ln1_layers.append(ln1)
            _ln2_layers.append(ln2)

            sa_layer = SelfAttentionLayer(
                dim_in=self.dim_token,
                dim_qkv=self.dim_qkv,
                num_heads=num_self_heads,
                dropout_prob=dropout_prob,
                use_rmsnorm=use_rmsnorm,
                add_bias=linear_in_attn_add_bias,
            )
            _self_attention_layers.append(sa_layer)

            mlp_hidden_dim = int(self.dim_token * mlp_ratio)
            if mlp_type == "timm":
                approx_gelu = lambda: torch.nn.GELU(approximate="tanh")
                mlp_layer = Mlp(
                    in_features=self.dim_token,
                    hidden_features=mlp_hidden_dim,
                    act_layer=approx_gelu,
                    drop=0,
                    bias=mlp_add_bias,
                    out_features=None,
                )
            elif mlp_type == "swiglu":
                mlp_layer = _SwiGLU(
                    in_features=self.dim_token,
                    hidden_features=mlp_hidden_dim,
                    out_features=None,
                    bias=mlp_add_bias,
                )
            elif mlp_type.startswith("resnet_"):
                # 'resnet_gelu', 'resnet_silu', 'resnet_swiglu'
                activation_fn = mlp_type.split("resnet_", 1)[1]
                mlp_layer = PointwiseResnet(
                    dim_in=self.dim_token,
                    dim_hidden=mlp_hidden_dim,
                    dim_out=self.dim_token,
                    bias=mlp_add_bias,
                    activation_fn=activation_fn,
                    add_init_activation=False,
                )
            else:
                raise NotImplementedError
            _mlp_layers.append(mlp_layer)

        self.ln1_layers = torch.nn.ModuleList(_ln1_layers)
        self.ln2_layers = torch.nn.ModuleList(_ln2_layers)
        self.sa_modulator_layers = torch.nn.ModuleList(_sa_modulator_layers)
        self.sa_layers = torch.nn.ModuleList(_self_attention_layers)
        self.mlp_layers = torch.nn.ModuleList(_mlp_layers)

        # final layer
        if self.upsample_ratio > 1:
            assert isinstance(self.upsample_ratio, int)
            mlp_hidden_dim = int(self.dim_token * mlp_ratio)
            out_dim = self.dim_token * upsample_ratio
            if mlp_type == "timm":
                approx_gelu = lambda: torch.nn.GELU(approximate="tanh")
                self.upsample_mlp = Mlp(
                    in_features=self.dim_token,
                    hidden_features=mlp_hidden_dim,
                    act_layer=approx_gelu,
                    drop=0,
                    bias=mlp_add_bias,
                    out_features=out_dim,
                )
            elif mlp_type == "swiglu":
                self.upsample_mlp = _SwiGLU(
                    in_features=self.dim_token,
                    hidden_features=mlp_hidden_dim,
                    out_features=out_dim,
                    bias=mlp_add_bias,
                )
            else:
                raise NotImplementedError
        else:
            self.upsample_mlp = None

    def forward(
        self,
        input_tokens: torch.Tensor,
        cond_feature: torch.Tensor = None,
        self_structural_attn_dict: T.Union[T.List[T.Dict[str, T.Any]], T.Dict[str, T.Any]] = None,
        debug: bool = False,
    ):
        """
        Args:
            input_tokens:
                (b, m, dim_token)
            cond_feature:
                (b, dim_cond_feature)
            self_structural_attn_dict:
            cross_structural_attn_dict:

        Returns:
            (b, m, dim_token) or (b, u*m, dim_token)
        """

        b, m, dim_token = input_tokens.shape

        # self attention
        if not isinstance(self_structural_attn_dict, (list, tuple)):
            self_structural_attn_dict = [self_structural_attn_dict] * self.num_self_attn

        current_layer_idx = self.layer_idx if self.layer_idx is not None else 0
        for mod_layer, ln1, sa_layer, ln2, mlp_layer, s_attn_dict in zip(
            self.sa_modulator_layers,
            self.ln1_layers,
            self.sa_layers,
            self.ln2_layers,
            self.mlp_layers,
            self_structural_attn_dict,
        ):
            current_layer_idx += 1
            layernorm_scaler = math.sqrt(1.0 / current_layer_idx)

            if cond_feature is not None:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_layer(cond_feature).chunk(6, dim=1)

                if debug:
                    assert shift_msa.isfinite().all(), f"nan: {shift_msa.isnan().any()}, inf: {shift_msa.isinf().any()}"
                    assert scale_msa.isfinite().all(), f"nan: {scale_msa.isnan().any()}, inf: {scale_msa.isinf().any()}"
                    assert gate_msa.isfinite().all(), f"nan: {gate_msa.isnan().any()}, inf: {gate_msa.isinf().any()}"
                    assert shift_mlp.isfinite().all(), f"nan: {shift_mlp.isnan().any()}, inf: {shift_mlp.isinf().any()}"
                    assert scale_mlp.isfinite().all(), f"nan: {scale_mlp.isnan().any()}, inf: {scale_mlp.isinf().any()}"
                    assert gate_mlp.isfinite().all(), f"nan: {gate_mlp.isnan().any()}, inf: {gate_mlp.isinf().any()}"

                normed_input_tokens = ln1(input_tokens)
                if self.use_layernorm_scaling:
                    normed_input_tokens = layernorm_scaler * normed_input_tokens

                modulated_input_tokens = modulate(x=normed_input_tokens, shift=shift_msa, scale=scale_msa)

                if debug:
                    assert normed_input_tokens.isfinite().all(), (
                        f"nan: {normed_input_tokens.isnan().any()}, inf: {normed_input_tokens.isinf().any()}"
                    )
                    assert modulated_input_tokens.isfinite().all(), (
                        f"nan: {modulated_input_tokens.isnan().any()}, inf: {modulated_input_tokens.isinf().any()}"
                    )

                input_tokens = input_tokens + (
                    gate_msa.unsqueeze(1)
                    * sa_layer(
                        x=modulated_input_tokens,
                        structural_attn_dict=s_attn_dict,
                    )
                )

                normed_input_tokens = ln2(input_tokens)
                if self.use_layernorm_scaling:
                    normed_input_tokens = layernorm_scaler * normed_input_tokens

                modulated_input_tokens = modulate(x=normed_input_tokens, shift=shift_mlp, scale=scale_mlp)

                if debug:
                    assert normed_input_tokens.isfinite().all(), (
                        f"nan: {normed_input_tokens.isnan().any()}, inf: {normed_input_tokens.isinf().any()}"
                    )
                    assert modulated_input_tokens.isfinite().all(), (
                        f"nan: {modulated_input_tokens.isnan().any()}, inf: {modulated_input_tokens.isinf().any()}"
                    )

                input_tokens = input_tokens + gate_mlp.unsqueeze(1) * mlp_layer(modulated_input_tokens)
            else:
                normed_input_tokens = ln1(input_tokens)
                if self.use_layernorm_scaling:
                    normed_input_tokens = layernorm_scaler * normed_input_tokens

                if debug:
                    assert normed_input_tokens.isfinite().all(), (
                        f"nan: {normed_input_tokens.isnan().any()}, inf: {normed_input_tokens.isinf().any()}"
                    )

                input_tokens = input_tokens + sa_layer(
                    x=normed_input_tokens,
                    structural_attn_dict=s_attn_dict,
                )

                normed_input_tokens = ln2(input_tokens)
                if self.use_layernorm_scaling:
                    normed_input_tokens = layernorm_scaler * normed_input_tokens

                if debug:
                    assert normed_input_tokens.isfinite().all(), (
                        f"nan: {normed_input_tokens.isnan().any()}, inf: {normed_input_tokens.isinf().any()}"
                    )

                input_tokens = input_tokens + mlp_layer(normed_input_tokens)  # (b, m, dim_token)

        if self.upsample_mlp is not None:
            input_tokens = self.upsample_mlp(input_tokens)  # (b, m, dim_token * u)
            input_tokens = input_tokens.reshape(b, -1, self.dim_token)  # (b, um, dim_token)

        return input_tokens


class VelocityCrossAttnDecoder(torch.nn.Module):
    """
    Velocity decoder that relies only on cross attention, ie,
    individual points are independent to each other.
    """

    def __init__(
        self,
        num_latent: T.Optional[int],  # not really needed
        dim_latent: int,
        dim_point: int,
        # block
        num_blocks: int,
        dim_qkv: int,
        dim_cond_feature: int = None,
        num_self_attn: int = 0,
        num_self_heads: int = 4,
        num_cross_heads: int = 4,
        dropout_prob: float = 0.0,
        use_rmsnorm: bool = True,
        mlp_ratio: float = 2,
        # output
        dim_output: int = None,
        # input linear
        dim_net: int = None,
        # positional encoding for XYZ
        pos_enc_name_xyz: str = "fourier",
        min_freq_log2: float = 0,
        max_freq_log2: float = 12,
        num_freqs: int = 32,
        pos_enc_xyz_extra_kwargs: T.Dict[str, T.Any] = {},
        # positional encoding for RGB
        pos_enc_name_rgb: str = "fourier",
        use_rgb_pos_encoder: bool = False,
        min_rgb_freq_log2: float = 0.0,
        max_rgb_freq_log2: float = 8.0,
        num_rgb_freqs: int = 8,
        pos_enc_rgb_extra_kwargs: T.Dict[str, T.Any] = {},
        # normal
        use_normal_pos_encoder: bool = False,
        # packed latent
        packed_latent: bool = False,
    ):
        """
        Args:
            num_latent:
                number of latent per shape.  Not really needed. Can be None
            dim_latent:
                dimension of each latent
            dim_point:
                dimension of the input point (3xyz + ...)
            dim_output:
                output dimension of the model, if None, the same as dim_point
            dim_net:
                main dimension of the network, if None, same as dim_point_token

            packed_latent:
                whether the latent will be provided in packed format
        """

        super().__init__()
        self.num_latent = num_latent
        self.dim_latent = dim_latent
        self.dim_point = dim_point
        self.dim_output = dim_output if dim_output is not None else dim_point
        self.dim_cond_feature = dim_cond_feature
        self.dim_net = dim_net
        self.use_rgb_pos_encoder = use_rgb_pos_encoder
        self.packed_latent = packed_latent

        # position encoding for point
        self.xyz_pos_encoder = get_pos_enc_cls(pos_enc_name_xyz)(
            dim_pos=3,  # 3xyz
            include_input=False,  # we concat ourselves
            min_freq_log2=min_freq_log2,
            max_freq_log2=max_freq_log2,
            num_freqs=num_freqs,
            log_sampling=True,
            **pos_enc_xyz_extra_kwargs,
        )

        # calculate the point token dimension
        self.dim_point_token = self.dim_point + self.xyz_pos_encoder.dim_out

        # position encoding for rgb
        if self.use_rgb_pos_encoder:
            self.rgb_pos_encoder = get_pos_enc_cls(pos_enc_name_rgb)(
                dim_pos=3,  # 3rgb [-1, 1]
                include_input=False,  # we concat ourselves
                min_freq_log2=min_rgb_freq_log2,
                max_freq_log2=max_rgb_freq_log2,
                num_freqs=num_rgb_freqs,
                log_sampling=True,
                **pos_enc_rgb_extra_kwargs,
            )
            # calculate the point token dimension
            self.dim_point_token += self.rgb_pos_encoder.dim_out
        else:
            self.rgb_pos_encoder = None

        if self.dim_net is None or self.dim_net == self.dim_point_token:
            self.dim_net = self.dim_point_token
            self.input_linear = None
        else:
            self.input_linear = _ops().Linear(
                in_features=self.dim_point_token,
                out_features=self.dim_net,
            )

        # self attn blocks
        self.blocks = torch.nn.ModuleList(
            [
                DecoderBlock(
                    dim_latent=dim_latent,
                    dim_token=self.dim_net,
                    dim_qkv=dim_qkv,
                    dim_cond_feature=dim_cond_feature,
                    num_self_attn=num_self_attn,
                    num_self_heads=num_self_heads,
                    num_cross_heads=num_cross_heads,
                    dropout_prob=dropout_prob,
                    use_rmsnorm=use_rmsnorm,
                    mlp_ratio=mlp_ratio,
                    packed_kv=packed_latent,
                )
                for _ in range(num_blocks)
            ]
        )

        # final layer
        self.final_layer = FinalLayer(
            dim_input=self.dim_net,
            dim_output=self.dim_output,
            dim_cond_feature=dim_cond_feature,
        )

    def forward(
        self,
        input_point_cloud: torch.Tensor,  # (b, m, dim_point)
        latent_tokens: torch.Tensor,  # (b, num_latent, dim_latent) or (bl, dim_latent) packed
        cond_feature: torch.Tensor = None,  # (b, dim_cond_feature)
        structural_attn_dicts: T.List[T.Dict[str, T.Dict[str, T.Any]]] = None,
        debug: bool = False,
        latent_coord: T.Optional[PackedPoint] = None,
    ):
        """
        Args:
            input_point_cloud:
                (b, m, dim_point)  The first 3 dimension is xyz, then it can be rgb, normal, etc
            latent_tokens:
                (b, num_latent, dim_latent) or (bl, dim_latent)  The latent representing the shape
            cond_feature:
                (b, dim_cond_feature)
            structural_attn_dicts:
                a list of dict(str, structural_attn_dict), one for each layer.
                each dict contains
                    cross: structural_attn_dict to be used for cross attention.
                    self: list of structural_attn_dict to be used for each layer of self attention in the block.
                    writeback: structural_attn_dict to be used for writeback.
            latent_coord:
                (bl, dn) needed if latent tokens are in packed format.

        Returns:
            output_velocity:
                (b, m, dim_point)  output velocity at each point
        """

        b, m, _dim_point = input_point_cloud.shape

        if self.packed_latent:
            assert latent_coord is not None

        if structural_attn_dicts is not None:
            assert len(structural_attn_dicts) == len(self.blocks)

        # position encode the points
        encoded_xyz = self.xyz_pos_encoder(input_point_cloud[..., :3])  # (b, m, dim_encoded_xyz)
        if debug:
            assert encoded_xyz.isfinite().all(), f"nan: {encoded_xyz.isnan().any()}, inf: {encoded_xyz.isinf().any()}"
        # input_tokens.append(encoded_xyz)

        input_tokens = [input_point_cloud, encoded_xyz]

        if self.use_rgb_pos_encoder:
            # we assume rgb is in the input from dim = 3 to 6
            dim_rgb_start = 3
            dim_rgb_end = 6
            encoded_rgb = self.rgb_pos_encoder(
                input_point_cloud[..., dim_rgb_start:dim_rgb_end]
            )  # (b, m, dim_encoded_rgb)
            input_tokens.append(encoded_rgb)

        input_tokens = torch.cat(input_tokens, dim=-1)  # (b, m, dim_point_token)
        assert input_tokens.size(-1) == self.dim_point_token, f"{input_tokens.size(-1)} {self.dim_point_token}"

        if debug:
            assert input_tokens.isfinite().all(), (
                f"nan: {input_tokens.isnan().any()}, inf: {input_tokens.isinf().any()}"
            )

        # input linear
        if self.input_linear is not None:
            input_tokens = self.input_linear(input_tokens)  # (b, m, dim_net)

        assert input_tokens.size(-1) == self.dim_net

        # block
        for i, block in enumerate(self.blocks):
            if structural_attn_dicts is not None:
                self_struct_attn_dict = structural_attn_dicts[i].get("self", None)
                cross_struct_attn_dict = structural_attn_dicts[i].get("cross", None)
            else:
                self_struct_attn_dict = None
                cross_struct_attn_dict = None

            # run through the block
            input_tokens = block(
                latents=latent_tokens,
                input_tokens=input_tokens,
                cond_feature=cond_feature,
                self_structural_attn_dict=self_struct_attn_dict,
                cross_structural_attn_dict=cross_struct_attn_dict,
                debug=debug,
                latent_coord=latent_coord,
            )

            if debug:
                assert input_tokens.isfinite().all(), (
                    f"nan: {input_tokens.isnan().any()}, inf: {input_tokens.isinf().any()}"
                )

        # final linear layer
        out = self.final_layer(x=input_tokens, cond_feature=cond_feature)  # (b, m, dim_point)

        if debug:
            assert out.isfinite().all(), f"nan: {out.isnan().any()}, inf: {out.isinf().any()}"

        return out


def get_output_mlp(
    dim_in: int,
    dim_hidden: int,
    dim_out: int,
    num_layers: int,
    mlp_type: str,
    mlp_add_bias: bool,
    contract: bool,
    force_fp32_for_final_layer: bool = False,
):
    # dim_in -> dim_hidden -> dim_in -> dim_hidden -> dim_in -> dim_out
    layers = []
    current_dim = dim_in
    for layer_idx in range(num_layers):
        layers.append(
            _ops().LayerNorm(
                normalized_shape=current_dim,
                eps=1e-6,
            )
        )
        if mlp_type == "timm":
            approx_gelu = lambda: torch.nn.GELU(approximate="tanh")
            layers.append(
                Mlp(
                    in_features=current_dim,
                    hidden_features=dim_hidden,
                    act_layer=approx_gelu,
                    drop=0,
                    bias=mlp_add_bias,
                    out_features=dim_in if contract else dim_hidden,
                )
            )
        elif mlp_type == "swiglu":
            layers.append(
                _SwiGLU(
                    in_features=current_dim,
                    hidden_features=dim_hidden,
                    out_features=dim_in if contract else dim_hidden,
                    bias=mlp_add_bias,
                )
            )
        else:
            raise NotImplementedError

        current_dim = dim_in if contract else dim_hidden

    layers.append(
        FinalLayer(
            dim_input=current_dim,
            dim_output=dim_out,
            force_fp32=force_fp32_for_final_layer,
        )
    )
    return torch.nn.Sequential(*layers)


class GaussianDecoderXv(torch.nn.Module):
    """
    Gaussian decoder estimates 3d gaussians given shape token.

    latent (packed) + init query (packed) -> perceiver -> shape mlp + color mlp -> gaussians
    """

    def __init__(
        self,
        dim_latent: int,
        given_point_inputs: T.List[str],  # 'xyz', 'xyz_encoded'
        num_given_mlp_layers: int,
        # perceiver
        perceiver_dim: int,
        perceiver_num_blocks: int,
        perceiver_num_self_attn: int,
        perceiver_cross_attn_type: str,  # 'global', 'localized_knn', 'localized_voxel'
        perceiver_self_attn_type: str,  # 'global', 'localized_knn', 'localized_voxel'
        # gs
        gs_expansion_ratio: int,
        gs_num_output_mlp_layers: int,
        gs_shape_outputs: T.List[str],
        gs_color_outputs: T.List[str],
        # perceiver attention settings
        perceiver_num_in_cluster: T.Union[int, T.List[int]] = None,
        perceiver_num_clusters: T.Union[int, T.List[int]] = None,
        perceiver_cross_cell_widths: T.Union[float, T.List[float]] = None,
        perceiver_self_cell_widths: T.Union[float, T.List[float]] = None,
        # region properties
        region_scaling: float = 2 / 64.0,  # radius of a region voxel
        # gaussian properties
        rgb_sh_degree: int = 0,  # 0-4
        bound_output_rgb: bool = True,
        fixed_bound_output_rgb_bug: bool = False,
        # properties
        use_unit_opacity: bool = False,
        opacity_logit_bias: float = 0.0,
        opacity_logit_scale: float = 1.0,
        min_opacity: float = None,
        max_opacity: float = None,
        min_opacity_clip: float = None,
        max_opacity_clip: float = None,
        max_scaling: float = None,
        scaling_activation: str = "sigmoid",
        scaling_scalar: float = 1.0,
        scaling_logit_bias: float = 0.0,
        min_scaling: float = 0.0,
        num_heads: int = 8,
        mlp_type: str = "timm",
        linear_in_attn_add_bias: bool = True,
        mlp_add_bias: bool = True,
        # positional encoding for point
        pos_enc_name_xyz: str = "fourier",
        pos_enc_config: T.Dict[str, T.Any] = dict(
            dim_pos=3,
            include_input=False,
            min_freq_log2=0,
            max_freq_log2=12,
            num_freqs=32,
            log_sampling=True,
        ),
        use_fp32_for_final_layer: bool = False,
    ):
        super().__init__()
        self.dim_latent = dim_latent
        self.gaussian_expansion_ratio = gs_expansion_ratio
        self.gs_expansion_ratio = gs_expansion_ratio
        self.given_point_inputs = given_point_inputs

        self.init_gaussian_bookkeep(
            region_scaling=region_scaling,
            rgb_sh_degree=rgb_sh_degree,
            bound_output_rgb=bound_output_rgb,
            fixed_bound_output_rgb_bug=fixed_bound_output_rgb_bug,
            use_unit_opacity=use_unit_opacity,
            opacity_logit_bias=opacity_logit_bias,
            opacity_logit_scale=opacity_logit_scale,
            min_opacity=min_opacity,
            max_opacity=max_opacity,
            min_opacity_clip=min_opacity_clip,
            max_opacity_clip=max_opacity_clip,
            max_scaling=max_scaling,
            scaling_activation=scaling_activation,
            scaling_scalar=scaling_scalar,
            scaling_logit_bias=scaling_logit_bias,
            min_scaling=min_scaling,
        )

        # perceiver
        self.xyz_encoding = get_pos_enc_cls(pos_enc_name_xyz)(
            **pos_enc_config,
        )
        assert self.given_point_inputs is not None
        self.given_point_start_dim_dict = dict()
        self.given_point_num_dim_dict = dict()
        current_dim = 0
        for name in self.given_point_inputs:
            if name == "xyz":
                self.given_point_start_dim_dict[name] = current_dim
                self.given_point_num_dim_dict[name] = 3
                current_dim += 3
            elif name == "xyz_encoded":
                self.given_point_start_dim_dict[name] = current_dim
                self.given_point_num_dim_dict[name] = self.xyz_encoding.dim_out
                current_dim += self.xyz_encoding.dim_out
            else:
                raise NotImplementedError
        self.point_linear_dim_in = current_dim
        self.point_linear = _ops().Linear(
            in_features=self.point_linear_dim_in,
            out_features=perceiver_dim,
        )
        if num_given_mlp_layers is not None and num_given_mlp_layers > 0:
            self.point_mlp = get_output_mlp(
                dim_in=perceiver_dim,
                dim_hidden=perceiver_dim,
                dim_out=perceiver_dim,
                num_layers=num_given_mlp_layers,
                mlp_type=mlp_type,
                mlp_add_bias=mlp_add_bias,
                contract=False,
            )
        else:
            self.point_mlp = None

        self.perceiver = SPointPerceiverEncoder(
            dim_latent=perceiver_dim,  # query
            dim_token=self.dim_latent,  # kv
            num_blocks=perceiver_num_blocks,
            dim_qkv=perceiver_dim,
            cross_attn_type=perceiver_cross_attn_type,
            self_attn_type=perceiver_self_attn_type,
            num_in_cluster=perceiver_num_in_cluster,
            num_clusters=perceiver_num_clusters,
            cross_cell_widths=perceiver_cross_cell_widths,
            self_cell_widths=perceiver_self_cell_widths,
            num_self_attn=perceiver_num_self_attn,
            num_self_heads=num_heads,
            num_cross_heads=num_heads,
            dropout_prob=0,
            use_rmsnorm=True,
            mlp_ratio=4,
            mlp_type=mlp_type,
            linear_in_attn_add_bias=linear_in_attn_add_bias,
            mlp_add_bias=mlp_add_bias,
            add_kv_linear=True,
        )

        # gs output mlp
        self.gs_shape_outputs = gs_shape_outputs
        self.gs_color_outputs = gs_color_outputs
        self.gs_shape_info = self.get_gs_info(output_types=self.gs_shape_outputs)
        self.gs_color_info = self.get_gs_info(output_types=self.gs_color_outputs)

        self.gs_output_shape_mlp = get_output_mlp(
            dim_in=perceiver_dim,
            dim_hidden=perceiver_dim,
            dim_out=self.gs_shape_info["total_dim"] * self.gs_expansion_ratio,
            num_layers=gs_num_output_mlp_layers,
            mlp_type=mlp_type,
            mlp_add_bias=mlp_add_bias,
            contract=False,
            force_fp32_for_final_layer=use_fp32_for_final_layer,
        )
        self.gs_output_color_mlp = get_output_mlp(
            dim_in=perceiver_dim,
            dim_hidden=perceiver_dim,
            dim_out=self.gs_color_info["total_dim"] * self.gs_expansion_ratio,
            num_layers=gs_num_output_mlp_layers,
            mlp_type=mlp_type,
            mlp_add_bias=mlp_add_bias,
            contract=False,
            force_fp32_for_final_layer=use_fp32_for_final_layer,
        )

    def init_gaussian_bookkeep(
        self,
        region_scaling: float,
        rgb_sh_degree: int,  # 0-4
        bound_output_rgb: bool,
        fixed_bound_output_rgb_bug: bool = False,
        use_unit_opacity: bool = False,
        opacity_logit_bias: float = 0.0,
        opacity_logit_scale: float = 1.0,
        min_opacity: float = None,
        max_opacity: float = None,
        min_opacity_clip: float = None,
        max_opacity_clip: float = None,
        max_scaling: float = None,
        scaling_activation: str = "softplus",
        scaling_scalar: float = 1.0,
        scaling_logit_bias: float = 0.0,
        min_scaling: float = 0.0,
    ):
        self.region_scaling = region_scaling
        self.rgb_sh_degree = rgb_sh_degree
        self.bound_output_rgb = bound_output_rgb
        self.fixed_bound_output_rgb_bug = fixed_bound_output_rgb_bug
        self.use_unit_opacity = use_unit_opacity
        self.opacity_logit_bias = opacity_logit_bias
        self.opacity_logit_scale = opacity_logit_scale
        self.min_opacity = min_opacity
        self.max_opacity = max_opacity
        self.min_opacity_clip = min_opacity_clip
        self.max_opacity_clip = max_opacity_clip
        self.max_scaling = max_scaling
        self.scaling_activation = scaling_activation
        self.scaling_scalar = scaling_scalar
        self.scaling_logit_bias = scaling_logit_bias
        self.min_scaling = min_scaling

        self.dim_rgb = ((self.rgb_sh_degree + 1) ** 2) * 3
        dim_scaling = 3

        # we set opacity to be 0.99 directly
        if self.use_unit_opacity:
            dim_opacity = 0
        else:
            dim_opacity = 1

        # individual dimension
        self.output_num_dim_dict = dict(
            xyz_w=3,  # (3,)
            opacity_logit=dim_opacity,  # (1,)
            scaling_logit=dim_scaling,  # (3,)
            quaternion_prenorm=4,  # (4,)
            rgb_sh=self.dim_rgb,  # (sh+1)**2 * 3
            normal_w=3,  # (3,)
            albedo=3,  # (3,)
            roughness_metallic=2,  # (2,)
        )

    def get_gs_info(self, output_types: T.List[str]):
        total_dim = 0
        start_dim_dict = dict()
        num_dim_dict = dict()
        for i in range(len(output_types)):
            assert output_types[i] in self.output_num_dim_dict
            start_dim_dict[output_types[i]] = total_dim
            num_dim_dict[output_types[i]] = self.output_num_dim_dict[output_types[i]]
            total_dim += self.output_num_dim_dict[output_types[i]]

        return dict(
            total_dim=total_dim,
            start_dim_dict=start_dim_dict,
            num_dim_dict=num_dim_dict,
        )

    def decode_gs(
        self,
        out: torch.Tensor,
        info: T.Dict[str, T.Any],
        scaling_logit_bias: float,
        scaling_scalar: float,
        min_scaling: float = None,
        max_scaling: float = None,
    ) -> T.Dict[str, torch.Tensor]:
        """
        Extract feature from out.

        Args:
            out:
                (*, d)
            info:
                total_dim:
                start_dim_dict:  key -> start_dim
                num_dim_dict:   key -> ndim
            scaling_scalar:


        Returns:

        """
        assert "total_dim" in info
        assert out.size(-1) == info["total_dim"]
        out_dict = dict()
        for key in info["start_dim_dict"]:
            arr = out[..., info["start_dim_dict"][key] : info["start_dim_dict"][key] + info["num_dim_dict"][key]]

            if key == "xyz_w":
                out_dict["xyz_w"] = arr  # (*, 3)

            elif key == "quaternion_prenorm":
                quaternion = torch.nn.functional.normalize(arr, dim=-1)  # (*, 4)
                out_dict["quaternion"] = quaternion  # (*, 4)

            elif key == "scaling_logit":
                scaling_logit = arr
                scaling_logit = scaling_logit + scaling_logit_bias
                if self.scaling_activation == "exp":
                    scaling = linalg_utils.exp(scaling_logit)  # (*, 2 or 3)
                elif self.scaling_activation == "softplus":
                    scaling = torch.nn.functional.softplus(scaling_logit)  # (*, 2 or 3)
                elif self.scaling_activation == "sigmoid":
                    scaling = scaling_logit.sigmoid()  # (*, 2 or 3)
                else:
                    raise NotImplementedError
                scaling = scaling * scaling_scalar

                if min_scaling is not None and min_scaling > 1e-8:
                    scaling = (scaling**2 + min_scaling**2).sqrt()

                # if self.use_2d_gaussians:
                #     scaling_z = 1e-6 * torch.ones(
                #         *scaling.shape[:-1], 1, dtype=scaling.dtype, device=scaling.device
                #     )  # (b, num_gaussian, 1)
                #     scaling = torch.cat(
                #         [
                #             scaling,  # (b, num_gaussian, 2)
                #             scaling_z,  # (b, num_gaussian, 1)
                #         ],
                #         dim=-1,
                #     )  # (b, num_gaussian, 3)

                if max_scaling is not None:
                    scaling = torch.clamp(scaling, max=max_scaling)

                out_dict["scaling"] = scaling

            elif key == "rgb_sh":
                if self.bound_output_rgb:
                    assert self.rgb_sh_degree == 0, f"{self.rgb_sh_degree=}"
                    arr = sh_utils.RGB2SH(arr.sigmoid())
                arr = utils.reshape(
                    arr,
                    start=-1,
                    end=-1,
                    shape=[(self.rgb_sh_degree + 1) ** 2, 3],
                )
                out_dict["rgb_sh"] = arr  # (*, (sh+1)**2, 3)

            elif key == "opacity_logit":
                if self.bound_output_rgb and not self.fixed_bound_output_rgb_bug:
                    arr = sh_utils.RGB2SH(arr.sigmoid())

                arr = self.opacity_logit_scale * arr + self.opacity_logit_bias
                opacity = arr.sigmoid()  # (*, 1)

                if self.min_opacity is not None or self.max_opacity is not None:
                    min_opacity = self.min_opacity if self.min_opacity is not None else 0
                    max_opacity = self.max_opacity if self.max_opacity is not None else 1
                    opacity = opacity * (max_opacity - min_opacity) + min_opacity

                if self.min_opacity_clip is not None or self.max_opacity_clip is not None:
                    opacity = torch.clamp(opacity, min=self.min_opacity_clip, max=self.max_opacity_clip)

                out_dict["opacity"] = opacity  # (*, 1)

            elif key == "normal_w":
                normal_w = torch.nn.functional.normalize(arr, dim=-1)  # (*, 3)

                # flip normal_w so it is on the same hemisphere as (0, 0, 1)
                ref_dir = torch.tensor([0.0, 0.0, 1.0], dtype=normal_w.dtype, device=normal_w.device)  # (3,)
                with torch.no_grad():
                    _sign = ((normal_w.detach() * ref_dir).sum(dim=-1) >= 0).to(
                        dtype=normal_w.dtype
                    ) * 2 - 1  # (*,) {1, -1}
                normal_w = normal_w * _sign.unsqueeze(-1)  # (*, 3)

                out_dict["normal_w"] = normal_w  # (*, 3)

            elif key == "albedo":
                out_dict["albedo"] = arr  # (*, 3rgb)

            elif key == "roughness_metallic":
                out_dict["roughness_metallic"] = arr  # (*, 2_roughness_metallic)

            else:
                raise NotImplementedError

        return out_dict

    def forward(
        self,
        latent_coord: PackedPoint,  # (n1+n2+...+nb, dn)
        latent: torch.Tensor,  # (n1+n2+...+nb, dim_latent)
        given_region_coord: PackedPoint,  # (m1+m2+...+mb, dn)
        use_grad_checkpointing: bool = False,
    ):
        """
        Args:
            latent_coord:
                (n1+n2+...+nb, dn), packed latent coordinate to indicate sample boundary,
                ie, only latent_coord.seq_lens is important, the coordinate can be dummy.
            latent:
                (n1+n2+...+nb, d)
            given_region_coord:
                (m1+m2+...+mb, dn)

        Returns:
            (b,) list of dict containing
                xyz_w:
                    (num_occ_cells, num_gaussian_per_cell, 3xyz_w) [-1, 1]
                scaling:
                    (num_occ_cells, num_gaussian_per_cell, 3xyz), after activation
                quaternion:
                    (num_occ_cells, num_gaussian_per_cell, 4xyzw), normalized
                opacity:
                    (num_occ_cells, num_gaussian_per_cell, 1), [0, 1]
                rgb_sh:
                    (num_occ_cells, (sh_degree+1)**2, 3rgb), raw
        """

        init_query = []
        for name in self.given_point_inputs:
            if name == "xyz":
                init_query.append(given_region_coord.coord)  # (bm, 3)
            elif name == "xyz_encoded":
                assert self.xyz_encoding is not None
                init_query.append(self.xyz_encoding(given_region_coord.coord))  # (bm, d)
            else:
                raise NotImplementedError
        init_query = torch.cat(init_query, dim=-1) if len(init_query) > 1 else init_query[0]  # (bm, d)
        init_query = self.point_linear(init_query)  # (bm, d)
        if self.point_mlp is not None:
            init_query = self.point_mlp(init_query)  # (bm, d)

        # perceiver
        query_latent = self.perceiver(
            input_tokens=latent,  # kv  (bn, dim_latent)
            latent_tokens=init_query,  # query  # (bm, d)
            coord_input_tokens=latent_coord,  # (bn, dn)
            coord_latents=given_region_coord,  # (bm, dn)
            use_grad_checkpointing=use_grad_checkpointing,
        )  # (bm, d)

        # gs output mlp
        bm, d = query_latent.shape
        shape_out = self.gs_output_shape_mlp(query_latent).reshape(
            bm, self.gs_expansion_ratio, -1
        )  # (bm, k, dim_shape)
        shape_out_dict = self.decode_gs(
            shape_out,
            info=self.gs_shape_info,
            scaling_logit_bias=self.scaling_logit_bias,
            scaling_scalar=self.scaling_scalar,
            min_scaling=self.min_scaling,
            max_scaling=self.max_scaling,
        )  # dict containing (bm, k, dim_shape)
        color_out = self.gs_output_color_mlp(query_latent).reshape(
            bm, self.gs_expansion_ratio, -1
        )  # (bm, k, dim_color)
        color_out_dict = self.decode_gs(
            color_out,
            info=self.gs_color_info,
            scaling_logit_bias=self.scaling_logit_bias,
            scaling_scalar=self.scaling_scalar,
            min_scaling=self.min_scaling,
            max_scaling=self.max_scaling,
        )  # dict containing (bm, k, dim_shape)

        _bm, k, _d = shape_out.shape

        # combine shape and color
        shape_out_dict.update(color_out_dict)

        if self.use_unit_opacity:
            opacity = torch.ones(bm, k, 1, dtype=color_out.dtype, device=color_out.device)  # (bm, k, 1)
            shape_out_dict["opacity"] = opacity

        # merge info from shape and region
        shape_out_dict["xyz_w"] = (
            (shape_out_dict["xyz_w"].sigmoid() * 2 - 1)  # (bm, k, 3)  [-1, 1]
            * self.region_scaling
        )  # (bm, k, 3) [-r, r]
        shape_out_dict["xyz_w"] = shape_out_dict["xyz_w"] + given_region_coord.coord.unsqueeze(-2)  # (bm, k, 3)

        # convert to dict of list
        # gs_dict = dict()
        # current_idx = 0
        # for ib in range(given_region_coord.batch_size):
        #     seq_len = given_region_coord.seq_lens[ib]
        #     end_idx = current_idx + seq_len
        #     for key in ["xyz_w", "scaling", "quaternion", "opacity", "rgb_sh"]:
        #         if key not in gs_dict:
        #             gs_dict[key] = []
        #         gs_dict[key].append(shape_out_dict[key][current_idx:end_idx])  # (num_occ_cells, num_gaussian_per_cell, d)
        #
        #     current_idx = end_idx

        # convert to list of gs_dict
        gs_dicts = []
        current_idx = 0
        for ib in range(given_region_coord.batch_size):
            seq_len = given_region_coord.seq_lens[ib]
            end_idx = current_idx + seq_len
            gs_dict = dict()
            for key in [
                "xyz_w",
                "scaling",
                "quaternion",
                "opacity",
                "rgb_sh",
                "normal_w",
                "albedo",
                "roughness_metallic",
            ]:
                if shape_out_dict.get(key, None) is None:
                    continue
                gs_dict[key] = shape_out_dict[key][current_idx:end_idx]  # (num_occ_cells, num_gaussian_per_cell, d)
            gs_dicts.append(gs_dict)
            current_idx = end_idx

        return gs_dicts


class SSLatentDecoder(torch.nn.Module):
    """
    Sparse Structure latent decoder estimates 3d gaussians given shape token.
    """

    def __init__(
        self,
        num_latent: int,
        dim_latent: int,
        # ss_latent
        res_ss_latent: int,  # 16
        dim_ss_latent: int,  # 8
        # block
        num_blocks: int,
        dim_perceiver: int,
        dim_qkv: int,
        num_self_attn: int = 0,
        num_self_heads: int = 4,
        num_cross_heads: int = 4,
        dropout_prob: float = 0.0,
        use_rmsnorm: bool = True,
        mlp_ratio: int = 2,
        mlp_type: str = "timm",
        linear_in_attn_add_bias: bool = True,
        mlp_add_bias: bool = True,
        # input linear
        add_input_linear: bool = False,
        dim_input_linear: int = -1,
    ):
        super().__init__()

        self.num_latent = num_latent
        self.dim_latent = dim_latent
        self.res_ss_latent = res_ss_latent
        self.dim_ss_latent = dim_ss_latent
        self.add_input_linear = add_input_linear
        self.dim_input_linear = dim_input_linear

        # input linear (applied to shape tokens)
        if self.add_input_linear:
            self.input_linear = _ops().Linear(
                in_features=self.dim_latent,
                out_features=self.dim_input_linear,
            )
            self.dim_actual_token = self.dim_input_linear
        else:
            self.input_linear = None
            self.dim_actual_token = self.dim_latent

        self.net = VectorDecoder(
            dim_input_token=self.dim_actual_token,
            init_query_q=self.res_ss_latent,
            init_query_h=self.res_ss_latent,
            init_query_w=self.res_ss_latent,
            init_method="learned_randn+poszyx",
            dim_output=self.dim_actual_token,
            dim_perceiver=dim_perceiver,
            perceiver_num_blocks=num_blocks,
            perceiver_dim_qkv=dim_qkv,
            perceiver_num_self_attn=num_self_attn,
            perceiver_num_self_heads=num_self_heads,
            perceiver_num_cross_heads=num_cross_heads,
            perceiver_dropout_prob=dropout_prob,
            perceiver_use_rmsnorm=use_rmsnorm,
            perceiver_mlp_ratio=mlp_ratio,
            perceiver_add_write_back=False,
            perceiver_mlp_type=mlp_type,
            perceiver_mlp_add_bias=mlp_add_bias,
            perceiver_linear_in_attn_add_bias=linear_in_attn_add_bias,
            upsample_num_blocks=0,
            upsample_kernel_size=1,
        )

        # final layer
        self.final_layer = FinalLayer(
            dim_input=self.dim_actual_token,
            dim_output=self.dim_ss_latent,
            dim_cond_feature=0,
        )

    def forward(
        self,
        latent_tokens: torch.Tensor,  # (b, num_latent, dim_latent)
    ):
        """
        Args:
            latent_tokens:
                (b, num_latent, dim_latent)  The latent representing the shape

        Returns:
            ss_latent:
                (b, dim_ss_latent, res_z, res_y, res_x)
        """

        b, _num_latent, _dim_latent = latent_tokens.shape

        # input linear on shape token
        if self.input_linear is not None:
            latent_tokens = self.input_linear(latent_tokens)  # (b, num_latent, dim_actual_token)

        # main network
        out = self.net(input_tokens=latent_tokens)  # (b, q, h, w, d)
        assert out.size(-1) == self.dim_actual_token

        # final layer
        out = self.final_layer(out)  # (b, q, h, w, dim_ss_latent)
        out = out.permute(0, 4, 1, 2, 3)  # (b, dim_ss_latent, q, h, w)
        assert out.shape == (b, self.dim_ss_latent, self.res_ss_latent, self.res_ss_latent, self.res_ss_latent)

        out_dict = dict(
            ss_latent=out,
        )

        return out_dict


class SSLatentDecoder_simplified(torch.nn.Module):
    """
    Sparse Structure latent decoder estimates 3d gaussians given shape token.
    Compared to above, it directly calls perceiver (without a wrapper of vector decoder)
    and directly use the given positional encoding as initial query.
    This allows the positional encoding to be the same used during training the tokenizer.

    query: voxel center xyz (16x16x16), kv: shape token -> ss_latent (16x16x16)
    """

    def __init__(
        self,
        dim_latent: int,
        # ss_latent
        res_ss_latent: int,  # 16
        dim_ss_latent: int,  # 8
        # perceiver block
        num_blocks: int,
        dim_perceiver: int,
        dim_qkv: int,
        num_self_attn: int = 0,
        num_self_heads: int = 4,
        num_cross_heads: int = 4,
        dropout_prob: float = 0.0,
        use_rmsnorm: bool = True,
        mlp_ratio: int = 2,
        mlp_type: str = "timm",
        linear_in_attn_add_bias: bool = True,
        mlp_add_bias: bool = True,
        # input linear (on shape latent)
        add_input_linear: bool = False,
        dim_input_linear: int = -1,
        # point mlp (on voxel center)
        num_voxel_mlp_layers: int = 1,
        # output mlp
        num_output_mlp_layers: int = 1,
        # positional encoding
        pos_enc_name_xyz: str = "fourier",
        pos_enc_config: T.Dict[str, T.Any] = dict(
            dim_pos=3,
            include_input=True,
            min_freq_log2=0,
            max_freq_log2=12,
            num_freqs=32,
            log_sampling=True,
        ),
    ):
        super().__init__()

        self.dim_latent = dim_latent
        self.res_ss_latent = res_ss_latent
        self.dim_ss_latent = dim_ss_latent
        self.add_input_linear = add_input_linear
        self.dim_input_linear = dim_input_linear

        # use voxel center as query
        self.xyz_encoding = get_pos_enc_cls(pos_enc_name_xyz)(
            **pos_enc_config,
        )
        self.point_linear_dim_in = self.xyz_encoding.dim_out
        self.point_linear = _ops().Linear(
            in_features=self.point_linear_dim_in,
            out_features=dim_perceiver,
        )
        if num_voxel_mlp_layers is not None and num_voxel_mlp_layers > 0:
            self.point_mlp = get_output_mlp(
                dim_in=dim_perceiver,
                dim_hidden=dim_perceiver,
                dim_out=dim_perceiver,
                num_layers=num_voxel_mlp_layers,
                mlp_type=mlp_type,
                mlp_add_bias=mlp_add_bias,
                contract=False,
            )
        else:
            self.point_mlp = None

        # input linear (applied to shape tokens)
        if self.add_input_linear:
            self.input_linear = _ops().Linear(
                in_features=self.dim_latent,
                out_features=self.dim_input_linear,
            )
            self.dim_actual_token = self.dim_input_linear
        else:
            self.input_linear = None
            self.dim_actual_token = self.dim_latent

        self.perceiver = PerceiverEncoder(
            dim_latent=dim_perceiver,  # query
            dim_token=self.dim_actual_token,  # kv
            num_blocks=num_blocks,
            dim_qkv=dim_qkv,
            num_self_attn=num_self_attn,
            num_self_heads=num_self_heads,
            num_cross_heads=num_cross_heads,
            dropout_prob=dropout_prob,
            use_rmsnorm=use_rmsnorm,
            mlp_ratio=mlp_ratio,
            add_write_back=False,
            keep_block_bug=False,
            mlp_type=mlp_type,
            linear_in_attn_add_bias=linear_in_attn_add_bias,
            mlp_add_bias=mlp_add_bias,
        )

        # final layer
        self.output_mlp = get_output_mlp(
            dim_in=dim_perceiver,
            dim_hidden=dim_perceiver,
            dim_out=self.dim_ss_latent,
            num_layers=num_output_mlp_layers,
            mlp_type=mlp_type,
            mlp_add_bias=mlp_add_bias,
            contract=False,
        )

    def get_init_voxel_xyz_w(
        self,
        min_xyz_w: float = -1,
        max_xyz_w: float = 1,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Get the low-res voxel center xyz_w.

        Returns:
            (res_ss_latent, res_ss_latent, res_ss_latent, 3xyz_w)
        """

        cw = (max_xyz_w - min_xyz_w) / self.res_ss_latent
        xs = (torch.arange(self.res_ss_latent, dtype=torch.float, device=device) + 0.5) * cw
        z, y, x = torch.meshgrid(
            xs,  # z
            xs,  # y
            xs,  # x
            indexing="ij",
        )  # (res_z, res_y, res_x)
        xyz_w = torch.stack([x, y, z], dim=-1)  # (res_z, res_y, res_x, 3xyz_w)
        return xyz_w

    def forward(
        self,
        latent_tokens: torch.Tensor,  # (b, num_latent, dim_latent)
    ):
        """
        Args:
            latent_tokens:
                (b, num_latent, dim_latent)  The latent representing the shape

        Returns:
            ss_latent:
                (b, num_frames, dim_ss_latent, res_z, res_y, res_x)
        """

        b, _num_latent, _dim_latent = latent_tokens.shape

        # input linear on shape token
        if self.input_linear is not None:
            latent_tokens = self.input_linear(latent_tokens)  # (b, num_latent, dim_actual_token)

        # get init voxel center
        voxel_xyz_w = self.get_init_voxel_xyz_w(device=latent_tokens.device)  # (rz, ry, rx, 3xyz_w)
        voxel_xyz_w = self.xyz_encoding(voxel_xyz_w)  # (rz, ry, rx, d)

        # process each frame separately
        init_query = voxel_xyz_w.unsqueeze(0).expand(
            b,
            self.res_ss_latent,
            self.res_ss_latent,
            self.res_ss_latent,
            -1,
        )  # (b, res_z, res_y, res_x, d)

        init_query = self.point_linear(init_query)  # (b, res_z, res_y, res_x, d)
        if self.point_mlp is not None:
            init_query = self.point_mlp(init_query)  # (b, res_z, res_y, res_x, d)

        # perceiver
        out = self.perceiver(
            input_tokens=latent_tokens,  # kv
            latent_tokens=init_query.reshape(b, self.res_ss_latent**3, -1),  # q
        )  # (b, res_z * res_y * res_x, d)

        # final layer
        out = self.output_mlp(out)  # (b, res_z * res_y * res_x, d)
        out = out.permute(0, 2, 1)  # (b, d, res_z * res_y * res_x)
        out = out.reshape(b, self.dim_ss_latent, self.res_ss_latent, self.res_ss_latent, self.res_ss_latent)
        assert out.shape == (b, self.dim_ss_latent, self.res_ss_latent, self.res_ss_latent, self.res_ss_latent)

        out_dict = dict(
            ss_latent=out,
        )
        return out_dict


class MeshDecoder_v2(torch.nn.Module):
    """
    Compared to MeshDecoder, the v2 version uses our own
    spoint perceiver instead of trellis's sparse transformer.
    """

    def __init__(
        self,
        dim_latent: int,
        grid_size: int,  # 64
        model_vertex_normal: bool,  # whether to predict vertex normal as network output
        # mlp on the input occ
        num_given_mlp_layers: int,
        # perceiver (q: occ, kv: shape token)
        perceiver_dim: int,
        perceiver_num_blocks: int,
        perceiver_num_self_attn: int,
        perceiver_self_attn_type: str,  # 'global', 'localized_knn', 'localized_voxel'
        # perceiver attention settings
        perceiver_num_in_cluster: T.Union[int, T.List[int]] = None,
        perceiver_num_clusters: T.Union[int, T.List[int]] = None,
        num_heads: int = 8,
        mlp_type: str = "timm",
        linear_in_attn_add_bias: bool = True,
        mlp_add_bias: bool = True,
        # positional encoding for point
        pos_enc_name_xyz: str = "fourier",
        pos_enc_config: T.Dict[str, T.Any] = dict(
            dim_pos=3,
            include_input=True,
            min_freq_log2=0,
            max_freq_log2=12,
            num_freqs=32,
            log_sampling=True,
        ),
    ):
        super().__init__()
        self.dim_latent = dim_latent
        self.grid_size = grid_size
        self.model_vertex_normal = model_vertex_normal

        # positional encoding for occ_grid center xyz_w
        self.xyz_encoding = get_pos_enc_cls(pos_enc_name_xyz)(
            **pos_enc_config,
        )

        # linear layer for query (occ_xyz_w)
        self.point_linear_dim_in = self.xyz_encoding.dim_out
        self.point_linear = _ops().Linear(
            in_features=self.point_linear_dim_in,
            out_features=perceiver_dim,
        )
        if num_given_mlp_layers is not None and num_given_mlp_layers > 0:
            self.point_mlp = get_output_mlp(
                dim_in=perceiver_dim,
                dim_hidden=perceiver_dim,
                dim_out=perceiver_dim,
                num_layers=num_given_mlp_layers,
                mlp_type=mlp_type,
                mlp_add_bias=mlp_add_bias,
                contract=False,
            )
        else:
            self.point_mlp = None

        # create perceiver (q: occ_grid, kv: shape token)
        self.perceiver = SPointPerceiverEncoder(
            dim_latent=perceiver_dim,  # query
            dim_token=self.dim_latent,  # kv
            num_blocks=perceiver_num_blocks,
            dim_qkv=perceiver_dim,
            cross_attn_type="global",
            self_attn_type=perceiver_self_attn_type,
            num_in_cluster=perceiver_num_in_cluster,
            num_clusters=perceiver_num_clusters,
            cross_cell_widths=None,
            self_cell_widths=(2.0 / grid_size) * 8,
            num_self_attn=perceiver_num_self_attn,
            num_self_heads=num_heads,
            num_cross_heads=num_heads,
            dropout_prob=0,
            use_rmsnorm=True,
            mlp_ratio=4,
            mlp_type=mlp_type,
            linear_in_attn_add_bias=linear_in_attn_add_bias,
            mlp_add_bias=mlp_add_bias,
            add_kv_linear=True,
        )

        # marching cube settings
        self.upsample_ratio = 4
        self.resolution = grid_size
        self.mesh_extractor_device = "cpu"
        self.create_mesh_extractor(device="cpu")
        self.out_channels = int(self.mesh_extractor.feats_channels)

        # upsampling sparse conv3d
        self.upsample = torch.nn.ModuleList(
            [
                slat_vae_mesh.SparseSubdivideBlock3d(
                    channels=perceiver_dim,
                    resolution=grid_size,
                    out_channels=perceiver_dim // 4,
                ),
                slat_vae_mesh.SparseSubdivideBlock3d(
                    channels=perceiver_dim // 4,
                    resolution=grid_size * 2,
                    out_channels=perceiver_dim // 8,
                ),
            ]
        )
        self.out_layer = sp.SparseLinear(perceiver_dim // 8, self.out_channels)

        # # NOTE: zero-initialization is really bad based on overfitting experiments
        # self.initialize_weights()

    def create_mesh_extractor(self, device: str):
        self.mesh_extractor = trellis_mesh.SparseFeatures2Mesh(
            device=device,
            res=self.resolution * self.upsample_ratio,
            use_color=self.model_vertex_normal,  # 6rgb,nxnynz
            full_width=2.0,  # [-1, 1]
        )
        self.mesh_extractor_device = device

    # def initialize_weights(self) -> None:
    #     # Zero-out output layers (make sure init sdf = 0)
    #     torch.nn.init.constant_(self.out_layer.weight, 0)
    #     torch.nn.init.constant_(self.out_layer.bias, 0)

    def to_representation(
        self,
        x: "sp.SparseTensor",
    ) -> T.List["trellis_mesh.MeshExtractResult"]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        if torch.device(self.mesh_extractor_device) != x.device:
            print(f"creating mesh extractor on device: {str(x.device)}")
            self.create_mesh_extractor(device=str(x.device))

        ret = []
        for i in range(x.shape[0]):
            mesh = self.mesh_extractor(
                x[i],
                training=self.training,
            )
            ret.append(mesh)
        return ret

    def forward(
        self,
        latent_token: torch.Tensor,  # (b, num_tokens, dim_tokens)
        occ_bijk: torch.Tensor,  # (n, 4bijk) int32
        grid_min_xyz_w: float = -1.0,
        grid_max_xyz_w: float = 1.0,
        use_grad_checkpointing: bool = False,
    ) -> T.List[T.Dict[str, T.Any]]:
        """
        Args:
            latent_token:
                (b, num_tokens, dim_tokens)
            occ_bijk:
                (n, 4bijk) int32, occupied cell's index
            grid_min_xyz_w:
                float, where the grid boundardy starts
            grid_max_xyz_w:
                float, where the grid boundary ends

        Returns:
            list of (b,) containing a dict
                vertex_xyz_w:
                    (n, 3xyz_w)  [-1, 1], the vertex xyz coordinates
                triangles:
                    (num_triangles, 3idx)  long
                vertex_rgb:
                    (n, 3)  real valued
                vertex_normal_w:
                    (n, 3)  real valued, not normalized
                grid_size:
                    int, number of cells per side
                success:
                    bool, whether the extraction is successful
        """
        b, num_latent, dim_latent = latent_token.shape
        occ_bidx = occ_bijk[:, 0]  # (bn,)
        ii = torch.argsort(occ_bidx, stable=True)  # (bn,)
        occ_bijk = occ_bijk[ii]  # (bn, 4bijk)
        del ii
        seq_lens = torch.bincount(occ_bidx, minlength=b)  # (b,)

        # pack occupied cell's center xyz_w into feats, coords
        cw = (grid_max_xyz_w - grid_min_xyz_w) / self.resolution
        occ_xyz = (occ_bijk[:, 1:4].float() + 0.5) * cw + grid_min_xyz_w  # (bn, 3xyz_w)

        coord_query = PackedPoint(
            coord=occ_xyz,  # (bn, 3xyz_w)
            seq_lens=seq_lens,  # (b,)
        )

        # positional encoding
        init_query = self.xyz_encoding(occ_xyz)  # (bn, d)
        init_query = self.point_linear(init_query)  # (bm, d)
        if self.point_mlp is not None:
            init_query = self.point_mlp(init_query)  # (bm, d)

        coord_kv = PackedPoint(
            coord=torch.zeros(
                b * num_latent, 3, device=occ_xyz.device
            ),  # (bm, 3xyz_w)  not important, since we use global attn
            seq_lens=torch.ones(b, dtype=torch.long, device=occ_xyz.device) * num_latent,  # (b,)
        )

        # perceiver
        query_latent = self.perceiver(
            input_tokens=latent_token.reshape(b * num_latent, dim_latent),  # kv  (bn, dim_latent)
            latent_tokens=init_query,  # query  # (bn, d)
            coord_input_tokens=coord_kv,  # (bm, dn)
            coord_latents=coord_query,  # (bn, dn)
            use_grad_checkpointing=use_grad_checkpointing,
        )  # (bm, d)

        # create sparse tensor
        h_sp = sp.SparseTensor(
            feats=query_latent,  # (bm, d)
            coords=occ_bijk,  # (bn, 4bijk)
        )

        # upsample the grid
        with _nullcontext():
            # spconv does not support bfloat16
            ori_dtype = h_sp.dtype
            for block in self.upsample:
                h_sp = block(h_sp.float())
            h_sp = h_sp.to(dtype=ori_dtype)

        # output layer
        h_sp = self.out_layer(h_sp)  # sparsetensor, (b, d=101)

        # diff marching cube
        with _nullcontext():
            meshing_results: T.List[trellis_mesh.MeshExtractResult] = self.to_representation(
                h_sp.to(dtype=torch.float32),
            )  # (b,)

        mesh_dicts = []
        for ib in range(len(meshing_results)):
            mdict = dict(
                vertex_xyz_w=meshing_results[ib].vertices,  # (n, 3xyz_w)  [-1, 1]
                triangles=meshing_results[ib].faces,  # (num_triangles, 3idx)  long
                vertex_rgb=meshing_results[ib].vertex_attrs[..., :3]
                if meshing_results[ib].vertex_attrs is not None
                else None,  # (n, 3rgb)   real valued
                vertex_normal_w=meshing_results[ib].vertex_attrs[..., 3:6]
                if meshing_results[ib].vertex_attrs is not None
                else None,  # (n, 3)  real valued
                grid_size=meshing_results[ib].res,  # int
                success=meshing_results[ib].success,  # bool
                # tsdf_v=meshing_results[ib].tsdf_v,
                # tsdf_s=meshing_results[ib].tsdf_s,
                reg_loss=meshing_results[ib].reg_loss,
                reg_sdf_loss=meshing_results[ib].reg_sdf_loss,
            )
            mesh_dicts.append(mdict)

            # print(f"\n\n{mdict['success']=}\n\n")

        return mesh_dicts


class MeshDecoderOverfitLearnableParams(torch.nn.Module):
    def __init__(
        self,
        # num_latent: int,
        dim_latent: int,
        # from trellis
        resolution: int = 64,
        # model_channels: int = 1024,
        # num_blocks: int = 6,
        # num_heads: T.Optional[int] = 16,
        # num_head_channels: T.Optional[int] = None,
        # mlp_ratio: float = 4,
        # attn_mode: T.Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        # window_size: int = 8,
        # use_fp16: bool = False,
        # use_checkpoint: bool = False,
        # qk_rms_norm: bool = False,
        # qk_rms_norm_cross: bool = False,
        representation_config: dict = None,
        # cross_first: bool = False,
        # pos_enc_name_xyz: str = "fourier",
    ):
        super().__init__()

        self.resolution = resolution

        self.rep_config = representation_config

        self.upsample_ratio = 2

        self.create_mesh_extractor(device="cpu")
        self.out_channels = int(self.mesh_extractor.feats_channels)

        self.learnable_params = torch.nn.Parameter(
            torch.randn((self.resolution * self.upsample_ratio,) * 3 + (self.out_channels,)), requires_grad=True
        )

        print(f"\n\n{self.learnable_params.shape=}\n\n")

        self.need_fpoint_latent = False

        from third_party.TRELLIS.trellis.modules.sparse import spatial as trellis_sparse_spatial

        self.sparse_subdivider = trellis_sparse_spatial.SparseSubdivide()  # each call will subdivide the input 2x

    def create_mesh_extractor(self, device: str):
        self.mesh_extractor = trellis_mesh.SparseFeatures2Mesh(
            device=device,
            res=self.resolution * self.upsample_ratio,
            use_color=self.rep_config.get("use_color", False),  # 6rgb,nxnynz
            full_width=2.0,  # [-1, 1]
        )
        self.mesh_extractor_device = device

    def to_representation(
        self,
        x: "sp.SparseTensor",
    ) -> T.List["trellis_mesh.MeshExtractResult"]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        if torch.device(self.mesh_extractor_device) != x.device:
            print(f"creating mesh extractor on device: {str(x.device)}")
            self.create_mesh_extractor(device=str(x.device))

        ret = []
        for i in range(x.shape[0]):
            mesh = self.mesh_extractor(
                x[i],
                training=self.training,
            )
            ret.append(mesh)
        return ret

    def forward(
        self,
        latent_token: torch.Tensor,  # (b, num_tokens, dim_tokens)
        occ_bijk: torch.Tensor,  # (n, 4bijk)
        grid_min_xyz_w: float = -1.0,
        grid_max_xyz_w: float = 1.0,
    ) -> T.List[T.Dict[str, T.Any]]:
        """
        Args:
            latent_token:
                (b, num_tokens, dim_tokens)
            occ_bijk:
                (n, 4bijk) occupied cell's index
            grid_min_xyz_w:
                float, where the grid boundardy starts
            grid_max_xyz_w:
                float, where the grid boundary ends

        Returns:
            list of (b,) containing a dict
                vertex_xyz_w:
                    (n, 3xyz_w)  [-1, 1], the vertex xyz coordinates
                triangles:
                    (num_triangles, 3idx)  long
                vertex_rgb:
                    (n, 3)  real valued
                vertex_normal_w:
                    (n, 3)  real valued, not normalized
                grid_size:
                    int, number of cells per side
                success:
                    bool, whether the extraction is successful
        """
        # pack occupied cell's center xyz_w into feats, coords
        cw = (grid_max_xyz_w - grid_min_xyz_w) / self.resolution
        occ_xyz = (occ_bijk[..., 1:4].float() + 0.5) * cw + grid_min_xyz_w  # (n, 3xyz_w)

        if False:
            # NOTE: DEBUG. save occupancy grid
            import trimesh

            print(f"\n\n{occ_xyz.shape=}\n\n")
            debug_pcd = trimesh.PointCloud(vertices=occ_xyz.cpu(), process=False)
            _ = debug_pcd.export("/mnt/test/code/shape_tokenization/debug/mesh_overfit/occ_xyz.ply")

            import sys

            sys.exit(1)

        # make sure all elements in the batch are the same as this class is for overfitting
        unique_b = torch.unique(occ_bijk[:, 0])
        bs = unique_b.numel()

        ref_occ_ijk = None
        for tmp_b in unique_b:
            tmp_occ_ijk = occ_bijk[occ_bijk[:, 0] == tmp_b, 1:4]
            if ref_occ_ijk is None:
                ref_occ_ijk = tmp_occ_ijk
            else:
                try:
                    assert torch.all(ref_occ_ijk == tmp_occ_ijk)
                except AssertionError:
                    tmp_diff = torch.sum(torch.abs(ref_occ_ijk.float() - tmp_occ_ijk.float()))
                    raise RuntimeError(f"\n{tmp_b=}, {tmp_diff=}\n")

        assert (self.upsample_ratio == 1) or (self.upsample_ratio % 2 == 0), f"{self.upsample_ratio=}"

        occ_grid = sp.SparseTensor(
            feats=torch.zeros((occ_bijk.shape[0], self.learnable_params.shape[-1]), device=occ_bijk.device),
            coords=occ_bijk,
        )

        for tmp_i in range(self.upsample_ratio // 2):
            occ_grid = self.sparse_subdivider(occ_grid)

        full_occ_feats = self.learnable_params.expand(bs, -1, -1, -1, -1)
        coords = occ_grid.coords
        occ_feats = full_occ_feats[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]]

        # create sparse tensor
        h = sp.SparseTensor(
            feats=occ_feats,
            coords=occ_grid.coords,
        )

        # # main network
        # h = self.sparse_transformer(x, context=latent_token)  # sparsetensor, (b, d=768)
        # for block in self.upsample:
        #     h = block(h)
        # h = h.type(x.dtype)  # sparsetensor, (b, d=96)
        # h = self.out_layer(h)  # sparsetensor, (b, d=101)

        # print(
        #     f"\n\n{occ_bijk.shape=}, {torch.sum(occ_bijk.float())=}, {h.coords.shape=}, {torch.sum(h.coords.abs())=}\n\n"
        # )

        with _nullcontext():
            meshing_results: T.List[trellis_mesh.MeshExtractResult] = self.to_representation(
                h.to(dtype=torch.float32),
            )  # (b,)

        mesh_dicts = []
        for ib in range(len(meshing_results)):
            mdict = dict(
                vertex_xyz_w=meshing_results[ib].vertices,  # (n, 3xyz_w)  [-1, 1]
                triangles=meshing_results[ib].faces,  # (num_triangles, 3idx)  long
                vertex_rgb=meshing_results[ib].vertex_attrs[..., :3],  # (n, 3rgb)   real valued
                vertex_normal_w=meshing_results[ib].vertex_attrs[..., 3:6],  # (n, 3)  real valued
                grid_size=meshing_results[ib].res,  # int
                success=meshing_results[ib].success,  # bool
                # tsdf_v=meshing_results[ib].tsdf_v,
                # tsdf_s=meshing_results[ib].tsdf_s,
                reg_loss=meshing_results[ib].reg_loss,
                reg_sdf_loss=meshing_results[ib].reg_sdf_loss,
            )
            mesh_dicts.append(mdict)

        return mesh_dicts
