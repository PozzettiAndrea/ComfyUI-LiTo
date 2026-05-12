#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements a simple decoder that converts shape tokens
# to a vector (or a set of vector).
import copy
import math
import typing as T

from lito.models._comfy_mlp import Mlp

import torch

from lito.models import layers, perceiver_encoder, resnet
from lito.models.layers import FinalLayer, SwiGLU as _SwiGLU
from plibs import utils
from plibs.flash_utils import create_block_diagonal_attn_bias_from_seq_lens
_OPS = None
def _ops():
    global _OPS
    if _OPS is None:
        import comfy.ops
        _OPS = comfy.ops.manual_cast
    return _OPS



class Upsample2DLayer(torch.nn.Module):
    """
    Upsample by nearest neighbor, followed by a conv layer.
    """

    def __init__(
        self,
        scale_factor: int,
        in_channels: int,
        out_channels: int,
        add_conv: bool = True,
    ):
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.add_conv = add_conv

        if self.scale_factor % 2 == 0:
            kernel_size = self.scale_factor + 1
        else:
            kernel_size = self.scale_factor

        padding = (kernel_size - 1) // 2

        if self.add_conv:
            self.conv = _ops().Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )
        else:
            self.conv = None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x:
                (b, c, h, w)

        Returns:
            (b, c, h * s, w * s)
        """

        x = torch.nn.functional.interpolate(
            input=x,  # (b, c, h, w)
            scale_factor=self.scale_factor,
            mode="nearest-exact",
            align_corners=None,
            antialias=False,
        )  # (b, c, h', w')

        if self.add_conv:
            x = self.conv(x)

        return x


class VectorDecoderSlow(torch.nn.Module):
    """
    Architecture overview:

    We first use a perceiver encoder (learned z (queries) -> shape token (key, value))
    to gather information from shape tokens. Then we gradually upsample z.


    Input: shape token s

    Let z be a set of learned init query. The shape of z is (q, ho, wo)
    z <- perceiver_encoder(q=z, kv=s)

    for i in range(L):
      z <- upsample(z)    # (q, h, w) -> (q, 2h, 2w)
      z <- block_diagonal_attention_within(q, ho, wo)
      z <- block_diagonal_attention_within(q, ho, wo, random_shift_with_(ho, wo))

    z <- final_linear(z)
    """

    def __init__(
        self,
        dim_input_token: int,
        init_query_q: int,
        init_query_h: int,
        init_query_w: int,
        init_method: str,
        dim_output: int,
        # perceiver
        dim_perceiver: int,
        perceiver_num_blocks: int,
        perceiver_dim_qkv: int,
        perceiver_num_self_attn: int,
        perceiver_num_self_heads: int,
        perceiver_num_cross_heads: int,
        perceiver_dropout_prob: float,
        perceiver_use_rmsnorm: bool,
        perceiver_mlp_ratio: int,
        perceiver_add_write_back: bool,
        # super-res
        upsample_num_blocks: int,
        upsample_kernel_size: int,  # = 1
        upsample_use_random_shift: bool,
    ):
        super().__init__()
        self.dim_input_token = dim_input_token
        self.init_shape = (init_query_q, init_query_h, init_query_w)
        self.dim_perceiver = dim_perceiver
        self.init_method = init_method
        self.dim_output = dim_output
        self.upsample_use_random_shift = upsample_use_random_shift

        # init query
        if self.init_method == "learned_randn":
            self.init_query = torch.nn.Parameter(torch.randn(*self.init_shape, self.dim_perceiver))  # (q, h, w, d)
        elif self.init_method == "learned_randn+poshw":
            assert self.dim_perceiver % 4 == 0
            self.hw_pos_encoder = layers.FourierEmbed(
                dim_pos=2,  # 2hw
                include_input=False,
                min_freq_log2=0,
                max_freq_log2=math.log2(max(self.init_shape[1:])) + 1,
                num_freqs=self.dim_perceiver // 4,
                log_sampling=True,
            )  # output dim = 2hw * num_freqs * 2sin_cos
            self.q_pos_encoder = _ops().Embedding(
                num_embeddings=self.init_shape[0],
                embedding_dim=self.dim_perceiver,
            )
            self.dim_pos_output = self.hw_pos_encoder.dim_out
            assert self.dim_perceiver == self.dim_pos_output, f"{self.dim_perceiver}, {self.dim_pos_output}"
            self.init_query = torch.nn.Parameter(torch.randn(*self.init_shape, self.dim_perceiver))  # (q, h, w, d)
        elif self.init_method == "learned_randn+posqhw":
            assert self.dim_perceiver % 6 == 0
            self.qhw_pos_encoder = layers.FourierEmbed(
                dim_pos=3,  # 3qhw
                include_input=False,
                min_freq_log2=0,
                max_freq_log2=math.log2(max(self.init_shape)) + 1,
                num_freqs=self.dim_perceiver // 6,
                log_sampling=True,
            )  # output dim = 3qhw * num_freqs * 2sin_cos
            self.dim_pos_output = self.qhw_pos_encoder.dim_out
            assert self.dim_perceiver == self.dim_pos_output, f"{self.dim_perceiver}, {self.dim_pos_output}"
            self.init_query = torch.nn.Parameter(torch.randn(*self.init_shape, self.dim_perceiver))  # (q, h, w, d)
        else:
            raise NotImplementedError

        # perceiver encoder
        self.encoder = perceiver_encoder.PerceiverEncoder(
            dim_latent=self.dim_perceiver,
            dim_token=self.dim_input_token,
            num_blocks=perceiver_num_blocks,
            dim_qkv=perceiver_dim_qkv,
            num_self_attn=perceiver_num_self_attn,
            num_self_heads=perceiver_num_self_heads,
            num_cross_heads=perceiver_num_cross_heads,
            dropout_prob=perceiver_dropout_prob,
            use_rmsnorm=perceiver_use_rmsnorm,
            mlp_ratio=perceiver_mlp_ratio,
            add_write_back=perceiver_add_write_back,
            keep_block_bug=False,
        )
        # output is dim_perceiver

        # upsample layer (b, q*h*w, d) -> (b, q*2h*2w, d)
        blocks = []
        current_shape = [s for s in self.init_shape]
        current_dim = self.dim_perceiver
        for block_idx in range(upsample_num_blocks):
            block_dict = dict()

            print(f"block_idx: {block_idx}, current_dim: {current_dim}")

            # we use conv2d with stride for the upsampling:
            # (b, q*h*w, d) -> (bq, d, h, w) -> (bq, d, 2h, 2w) -> (bq, 2h, 2w, d) -> (b, q*2h*2w, d)
            assert current_dim // 2 >= 1
            block_dict["conv"] = Upsample2DLayer(
                scale_factor=upsample_kernel_size,
                in_channels=current_dim,
                out_channels=current_dim // 2,
            )
            current_dim = current_dim // 2
            current_shape = [
                current_shape[0],
                current_shape[1] * upsample_kernel_size,
                current_shape[2] * upsample_kernel_size,
            ]

            print(f"  after upsample dim: {current_dim}")
            print(f"  current_shape: {current_shape}")

            # block diagonal self attention (centered)
            block_dict["self_center"] = layers.SelfAttentionLayer(
                dim_in=current_dim,
                dim_qkv=current_dim * 2,
                num_heads=perceiver_num_self_heads,
                dropout_prob=perceiver_dropout_prob,
                use_rmsnorm=perceiver_use_rmsnorm,
            )

            # block diagonal with random shift (or centered again)
            block_dict["self_shift"] = layers.SelfAttentionLayer(
                dim_in=current_dim,
                dim_qkv=current_dim * 2,
                num_heads=perceiver_num_self_heads,
                dropout_prob=perceiver_dropout_prob,
                use_rmsnorm=perceiver_use_rmsnorm,
            )

            block_dict = torch.nn.ModuleDict(block_dict)
            blocks.append(block_dict)

        self.blocks = torch.nn.ModuleList(blocks)
        self.output_shape = current_shape  # (3qhw,)
        self.dim_upsample_output = current_dim

        # final linear layer
        if self.dim_upsample_output != self.dim_output:
            self.final_linear = _ops().Linear(
                in_features=self.dim_upsample_output,
                out_features=self.dim_output,
            )
        else:
            self.final_linear = None

    def run_self_attn_with_shift(
        self,
        self_attn_layer: layers.SelfAttentionLayer,
        latents: torch.Tensor,
        block_h: int,
        block_w: int,
        shift_h: T.Union[torch.Tensor, int],
        shift_w: T.Union[torch.Tensor, int],
    ) -> torch.Tensor:
        """
        Create the attention bias for block diagonal attention with a given shift.
        Each block is (block_h, block_w).

        Args:
            self_attn_layer:
                the self attention layer
            latents:
                (b, q, h, w, d)
            qhw_idxs:
                (q, h, w, 3qhw)
            shift_h:
                (b, q) or int.  shift 0 toward bottom (larger h).  [0, .., block_h-1]
            shift_w:
                (b, q) or int.  shift 0 toward right (larger w). [0, .., block_w-1]

        Returns:
            output_latent:
                (b, q, h, w, d)
        """

        b, q, h, w, d = latents.shape
        bqhw = b * q * h * w

        bidx = torch.arange(b, device=latents.device).reshape(b, 1, 1, 1)  # (b, 1q, 1h, 1w)
        qidx = torch.arange(q, device=latents.device).reshape(1, q, 1, 1)
        hidx = torch.arange(h, device=latents.device).reshape(1, 1, h, 1).expand(b, q, h, 1)  # (b, q, h, 1)
        widx = torch.arange(w, device=latents.device).reshape(1, 1, 1, w).expand(b, q, 1, w)  # (b, q, 1, w)

        if isinstance(shift_h, torch.Tensor):
            assert shift_h.shape == (b, q), f"{shift_h.shape}"
            shift_h = shift_h.reshape(b, q, 1, 1)  # (b, q, 1h, 1w)
            # assert torch.logical_and(
            #     shift_h >= 0,
            #     shift_h < block_h
            # ).all()
        elif isinstance(shift_h, int):
            # assert 0 <= shift_h < block_h, f'{shift_h}'
            pass
        else:
            raise NotImplementedError

        if isinstance(shift_w, torch.Tensor):
            assert shift_w.shape == (b, q), f"{shift_w.shape}"
            shift_w = shift_w.reshape(b, q, 1, 1)  # (b, q, 1h, 1w)
            # assert torch.logical_and(
            #     shift_w >= 0,
            #     shift_w < block_w
            # ).all()
        elif isinstance(shift_w, int):
            # assert 0 <= shift_w < block_w, f'{shift_w}'
            pass
        else:
            raise NotImplementedError

        # h -> v, w -> u
        # print(f'hidx: {hidx}, shift_h: {shift_h}, block_h: {block_h}')
        vidx = (hidx - shift_h) // block_h  # (b, q, h, 1)
        uidx = (widx - shift_w) // block_w  # (b, q, 1, w)

        # we directly create bqhw_idxs, since we need to sort and count,
        # might as well use unique directly
        bqhw_idxs = torch.cat(
            [
                bidx.expand(b, q, h, w),
                qidx.expand(b, q, h, w),
                vidx.expand(b, q, h, w),
                uidx.expand(b, q, h, w),
            ],
            dim=-1,
        )  # (b, q, h, w, 4bqhw)

        # use unique to create index and count
        _, unique_idxs, unique_idx_counts = torch.unique(
            bqhw_idxs.reshape(bqhw, 4),
            sorted=True,
            return_inverse=True,
            return_counts=True,
            dim=0,
        )  # (num_block, 4), (bqhw,) (num_block,)
        num_blocks = unique_idx_counts.size(0)

        # sort with unique_idxs
        _, sort_idx = torch.sort(
            input=unique_idxs,
            dim=0,
        )  # (bqhw,)
        latents = latents.reshape(bqhw, latents.size(-1))  # (bqhw, d)
        latents = latents[sort_idx]  # (bqhw, d)

        # create attn_bias
        chunk_size = 65535  # flash attn supports max 65535 blocks
        num_chunks = (num_blocks + chunk_size - 1) // chunk_size
        if num_chunks == 1:
            attn_bias = create_block_diagonal_attn_bias_from_seq_lens(
                q_seqlen=unique_idx_counts.tolist(),  # (num_block,)
                kv_seqlen=unique_idx_counts.tolist(),  # (num_block,)
            )
            latents = self_attn_layer(
                x=latents.unsqueeze(0),  # (1, bqhw, d)
                structural_attn_dict=dict(
                    mode="flash_varlen",
                    attn_bias=attn_bias,
                ),
            ).squeeze(0)  # (bqhw, d)
        else:
            start_idx_1 = torch.cumsum(unique_idx_counts, dim=0)  # (num_blocks,)
            out = []
            for chunk_idx in range(num_chunks):
                cidx_start = chunk_idx * chunk_size
                cidx_end = (chunk_idx + 1) * chunk_size
                sidx = start_idx_1[cidx_start - 1] if chunk_idx >= 1 else 0
                eidx = start_idx_1[cidx_end - 1]

                attn_bias = create_block_diagonal_attn_bias_from_seq_lens(
                    q_seqlen=unique_idx_counts[cidx_start:cidx_end].tolist(),  # (num_block,)
                    kv_seqlen=unique_idx_counts[cidx_start:cidx_end].tolist(),  # (num_block,)
                )
                _latents = self_attn_layer(
                    x=latents[sidx:eidx].unsqueeze(0),  # (1, n, d)
                    structural_attn_dict=dict(
                        mode="flash_varlen",
                        attn_bias=attn_bias,
                    ),
                ).squeeze(0)  # (n, d)
                out.append(_latents)
            latents = torch.cat(out, dim=0)  # (bqhw, d)

        latents = latents.reshape(b, q, h, w, latents.size(-1))  # (b, q, h, w, d)
        return latents

    def forward(
        self,
        input_tokens: torch.Tensor,  # (b, m, dim_in)
    ):
        b, m, dim_in = input_tokens.shape
        q, h, w = self.init_shape
        qhw = q * h * w

        if self.init_method in ["learned_randn+poshw", "learned_randn+posqhw"]:
            qidxs, hidxs, widxs = torch.meshgrid(
                torch.arange(q, device=input_tokens.device),
                torch.arange(h, device=input_tokens.device),
                torch.arange(w, device=input_tokens.device),
                indexing="ij",
            )  # (q, h, w)
            qhw_idxs = torch.stack([qidxs, hidxs, widxs], dim=-1)  # (q, h, w, 3qhw)
        else:
            qhw_idxs = None

        # get init query
        if self.init_method == "learned_randn":
            latents = self.init_query.expand(b, q, h, w, self.dim_perceiver)  # (b, q, h, w, d)
        elif self.init_method == "learned_randn+poshw":
            latents = self.init_query.expand(b, q, h, w, self.dim_perceiver)  # (b, q, h, w, d)
            hw_pose = self.hw_pos_encoder(qhw_idxs[..., 1:].expand(b, q, h, w, 2))  # (b, q, h, w, dim_pose)
            q_pose = (
                self.q_pos_encoder(qhw_idxs[..., 0]).unsqueeze(0).expand(b, q, h, w, self.dim_perceiver)
            )  # (q, h, w, dim_pose)
            latents = latents + hw_pose + q_pose
        elif self.init_method == "learned_randn+posqhw":
            latents = self.init_query.expand(b, q, h, w, self.dim_perceiver)  # (b, q, h, w, d)
            qhw_pose = self.qhw_pos_encoder(qhw_idxs.expand(b, q, h, w, 3))  # (b, q, h, w, dim_pose)
            latents = latents + qhw_pose
        else:
            raise NotImplementedError

        # perceiver encoder
        latents = self.encoder(
            input_tokens=input_tokens,  # (b, m, dim_in)
            latent_tokens=latents.reshape(b, qhw, self.dim_perceiver),  # (b, qhw, d)
            return_all_layers=False,
        )  # (b, qhw, d)
        latents = latents.reshape(b, q, h, w, latents.size(-1))  # (b, q, h, w, d)

        print(f"b, q, h, w: {b}, {self.init_shape}")
        print(f"after encoder, latent shape: {latents.shape}")

        # upsample
        for block_idx, block in enumerate(self.blocks):
            conv = block["conv"]
            self_center = block["self_center"]
            self_shift = block["self_shift"]

            print(f"block_idx = {block_idx}:")

            # conv upsampling
            latents = conv(
                latents.flatten(start_dim=0, end_dim=1).permute(0, 3, 1, 2)  # (bq, d, h, w)
            )  # (bq, du, hu, wu)
            _bq, du, hu, wu = latents.shape
            latents = latents.permute(0, 2, 3, 1).reshape(b, q, hu, wu, du)  # (b, q, hu, wu, du)

            print(f"  after conv, latent shape: {latents.shape}")

            # self_attn_center
            latents = self.run_self_attn_with_shift(
                self_attn_layer=self_center,
                latents=latents,  # (b, q, hu, wu, du)
                block_h=self.init_shape[1],
                block_w=self.init_shape[2],
                shift_h=0,
                shift_w=0,
            )  # (b, q, hu, wu, du)

            print(f"  after 1st, latent shape: {latents.shape}")

            # self_attn_shift
            # random sample a shift
            if self.upsample_use_random_shift:
                random_shift_h = torch.randint(
                    low=0, high=self.init_shape[1], size=(b, q), device=latents.device
                )  # (b, q)
                random_shift_w = torch.randint(
                    low=0, high=self.init_shape[2], size=(b, q), device=latents.device
                )  # (b, q)
            else:
                random_shift_h = random_shift_w = 0
            latents = self.run_self_attn_with_shift(
                self_attn_layer=self_shift,
                latents=latents,  # (b, q, hu, wu, du)
                block_h=self.init_shape[1],
                block_w=self.init_shape[2],
                shift_h=random_shift_h,  # (b, q)
                shift_w=random_shift_w,  # (b, q)
            )  # (b, q, hu, wu, du)

            print(f"  after 2nd, latent shape: {latents.shape}")

        # final linear
        if self.final_linear is not None:
            latents = self.final_linear(latents)  # (b, q, hu, wu, du)

        return latents  # (b, q, hu, wu, du)


class VectorDecoder(torch.nn.Module):
    """
    Architecture overview:

    We first use a perceiver encoder (learned z (queries) -> shape token (key, value))
    to gather information from shape tokens. Then we gradually upsample z.


    Input: shape token s

    Let z be a set of learned init query. The shape of z is (q, ho, wo)
    z <- perceiver_encoder(q=z, kv=s)

    for i in range(L):
      z <- upsample(z)    # (q, h, w) -> (q, 2h, 2w)
      z <- block_diagonal_attention_within(q, ho, wo)

    z <- final_linear(z)
    """

    def __init__(
        self,
        dim_input_token: int,
        init_query_q: int,
        init_query_h: int,
        init_query_w: int,
        init_method: str,
        dim_output: int,
        # perceiver
        dim_perceiver: int,
        perceiver_num_blocks: int,
        perceiver_dim_qkv: int,
        perceiver_num_self_attn: int,
        perceiver_num_self_heads: int,
        perceiver_num_cross_heads: int,
        perceiver_dropout_prob: float,
        perceiver_use_rmsnorm: bool,
        perceiver_mlp_ratio: int,
        perceiver_add_write_back: bool,
        # super-res
        upsample_num_blocks: int,
        upsample_kernel_size: int,
        #
        perceiver_mlp_type: str = "timm",
        perceiver_linear_in_attn_add_bias: bool = True,
        perceiver_mlp_add_bias: bool = True,
    ):
        super().__init__()
        self.dim_input_token = dim_input_token
        self.init_shape = (init_query_q, init_query_h, init_query_w)
        self.dim_perceiver = dim_perceiver
        self.init_method = init_method
        self.dim_output = dim_output

        # init query
        if self.init_method == "learned_randn":
            self.init_query = torch.nn.Parameter(torch.randn(*self.init_shape, self.dim_perceiver))  # (q, h, w, d)
        elif self.init_method == "learned_randn+poshw":
            assert self.dim_perceiver % 4 == 0
            self.hw_pos_encoder = layers.FourierEmbed(
                dim_pos=2,  # 2hw
                include_input=False,
                min_freq_log2=0,
                max_freq_log2=math.log2(max(self.init_shape[1:])) + 1,
                num_freqs=self.dim_perceiver // 4,
                log_sampling=True,
            )  # output dim = 2hw * num_freqs * 2sin_cos
            self.q_pos_encoder = _ops().Embedding(
                num_embeddings=self.init_shape[0],
                embedding_dim=self.dim_perceiver,
            )
            self.dim_pos_output = self.hw_pos_encoder.dim_out
            assert self.dim_perceiver == self.dim_pos_output, f"{self.dim_perceiver}, {self.dim_pos_output}"
            self.init_query = torch.nn.Parameter(torch.randn(*self.init_shape, self.dim_perceiver))  # (q, h, w, d)
        elif self.init_method == "learned_randn+poszyx":
            assert self.dim_perceiver % 4 == 0
            self.zyx_pos_encoder = layers.FourierEmbed(
                dim_pos=3,  # 3zyx
                include_input=True,
                min_freq_log2=0,
                max_freq_log2=math.log2(max(self.init_shape[1:])) + 1,
                num_freqs=self.dim_perceiver // 4,
                log_sampling=True,
            )  # output dim = 3qhw * num_freqs * 2sin_cos + 3xyz
            self.dim_pos_output = self.zyx_pos_encoder.dim_out
            self.init_query_linear = _ops().Linear(
                in_features=self.dim_pos_output,
                out_features=self.dim_perceiver,
            )
            self.init_query = torch.nn.Parameter(torch.randn(*self.init_shape, self.dim_perceiver))  # (q, h, w, d)

        else:
            raise NotImplementedError

        # perceiver encoder
        self.encoder = perceiver_encoder.PerceiverEncoder(
            dim_latent=self.dim_perceiver,
            dim_token=self.dim_input_token,
            num_blocks=perceiver_num_blocks,
            dim_qkv=perceiver_dim_qkv,
            num_self_attn=perceiver_num_self_attn,
            num_self_heads=perceiver_num_self_heads,
            num_cross_heads=perceiver_num_cross_heads,
            dropout_prob=perceiver_dropout_prob,
            use_rmsnorm=perceiver_use_rmsnorm,
            mlp_ratio=perceiver_mlp_ratio,
            add_write_back=perceiver_add_write_back,
            keep_block_bug=False,
            mlp_type=perceiver_mlp_type,
            linear_in_attn_add_bias=perceiver_linear_in_attn_add_bias,
            mlp_add_bias=perceiver_mlp_add_bias,
        )
        # output is dim_perceiver

        # upsample layer (b, q*h*w, d) -> (b, q*2h*2w, d)
        blocks = []
        current_shape = [s for s in self.init_shape]
        current_dim = self.dim_perceiver
        for block_idx in range(upsample_num_blocks):
            block_dict = dict()

            # print(f'block_idx: {block_idx}, current_dim: {current_dim}')

            # we use conv2d with stride for the upsampling:
            # (b, q*h*w, d) -> (bq, d, h, w) -> (bq, d, 2h, 2w) -> (bq, 2h, 2w, d) -> (b, q*2h*2w, d)
            assert current_dim // 2 >= 1
            block_dict["conv"] = Upsample2DLayer(
                scale_factor=upsample_kernel_size,
                in_channels=current_dim,
                out_channels=current_dim // 2,
            )
            current_dim = current_dim // 2
            current_shape = [
                current_shape[0],
                current_shape[1] * upsample_kernel_size,
                current_shape[2] * upsample_kernel_size,
            ]

            # print(f'  after upsample dim: {current_dim}')
            # print(f'  current_shape: {current_shape}')

            # block diagonal self attention (centered)
            block_dict["self_center"] = layers.SelfAttentionLayer(
                dim_in=current_dim,
                dim_qkv=current_dim * 2,
                num_heads=perceiver_num_self_heads,
                dropout_prob=perceiver_dropout_prob,
                use_rmsnorm=perceiver_use_rmsnorm,
            )

            block_dict = torch.nn.ModuleDict(block_dict)
            blocks.append(block_dict)

        self.blocks = torch.nn.ModuleList(blocks)
        self.output_shape = current_shape  # (3qhw,)
        self.dim_upsample_output = current_dim

        # final linear layer
        if self.dim_upsample_output != self.dim_output:
            self.final_linear = _ops().Linear(
                in_features=self.dim_upsample_output,
                out_features=self.dim_output,
            )
        else:
            self.final_linear = None

    def forward(
        self,
        input_tokens: torch.Tensor,
    ):
        """
        Args:
            input_tokens:
                (b, num_tokens, dim_input_token)

        Returns:
            output_map:
                (b, q, hu, wu, dim_output)
        """

        b, m, dim_in = input_tokens.shape
        q, h, w = self.init_shape
        qhw = q * h * w

        if self.init_method in ["learned_randn+poshw", "learned_randn+posqhw"]:
            qidxs, hidxs, widxs = torch.meshgrid(
                torch.arange(q, device=input_tokens.device),
                torch.arange(h, device=input_tokens.device),
                torch.arange(w, device=input_tokens.device),
                indexing="ij",
            )  # (q, h, w)
            qhw_idxs = torch.stack([qidxs, hidxs, widxs], dim=-1)  # (q, h, w, 3qhw)
        elif self.init_method in ["learned_randn+poszyx"]:
            qidxs, hidxs, widxs = torch.meshgrid(
                (torch.arange(q, device=input_tokens.device) + 0.5) * (2.0 / q) - 1,
                (torch.arange(h, device=input_tokens.device) + 0.5) * (2.0 / h) - 1,
                (torch.arange(w, device=input_tokens.device) + 0.5) * (2.0 / w) - 1,
                indexing="ij",
            )  # (q, h, w)
            qhw_idxs = torch.stack([qidxs, hidxs, widxs], dim=-1)  # (q, h, w, 3qhw)
        else:
            qhw_idxs = None

        # get init query
        if self.init_method == "learned_randn":
            latents = self.init_query.expand(b, q, h, w, self.dim_perceiver)  # (b, q, h, w, d)
        elif self.init_method == "learned_randn+poshw":
            latents = self.init_query.expand(b, q, h, w, self.dim_perceiver)  # (b, q, h, w, d)
            hw_pose = self.hw_pos_encoder(qhw_idxs[..., 1:].expand(b, q, h, w, 2))  # (b, q, h, w, dim_pose)
            q_pose = (
                self.q_pos_encoder(qhw_idxs[..., 0]).unsqueeze(0).expand(b, q, h, w, self.dim_perceiver)
            )  # (q, h, w, dim_pose)
            latents = latents + hw_pose + q_pose
        elif self.init_method == "learned_randn+posqhw":
            latents = self.init_query.expand(b, q, h, w, self.dim_perceiver)  # (b, q, h, w, d)
            qhw_pose = self.qhw_pos_encoder(qhw_idxs.expand(b, q, h, w, 3))  # (b, q, h, w, dim_pose)
            latents = latents + qhw_pose
        elif self.init_method == "learned_randn+poszyx":
            latents = self.init_query.expand(b, q, h, w, self.dim_perceiver)  # (b, q, h, w, d)
            qhw_pose = self.zyx_pos_encoder(qhw_idxs.expand(b, q, h, w, 3))  # (b, q, h, w, dim_pose)
            latents = latents + self.init_query_linear(qhw_pose)
        else:
            raise NotImplementedError

        # perceiver encoder
        latents = self.encoder(
            input_tokens=input_tokens,  # (b, m, dim_in)
            latent_tokens=latents.reshape(b, qhw, self.dim_perceiver),  # (b, qhw, d)
            return_all_layers=False,
        )  # (b, qhw, d)
        latents = latents.reshape(b * q, h, w, latents.size(-1))  # (bq, h, w, d)
        latents = latents.permute(0, 3, 1, 2)  # (bq, d, h, w)

        # print(f'b, q, h, w: {b}, {self.init_shape}')
        # print(f'after encoder, latent shape: {latents.shape}')

        # upsample
        block_h = self.init_shape[1]
        block_w = self.init_shape[2]
        for block_idx, block in enumerate(self.blocks):
            # latents: (bq, d, h, w)

            conv = block["conv"]
            self_center = block["self_center"]

            # print(f'block_idx = {block_idx}:')

            # conv upsampling
            latents = conv(latents)  # (bq, du, hu, wu)
            # print(f'  after conv, latent shape: {latents.shape}')

            # self_attn
            _bq, du, hu, wu = latents.shape
            nh = hu // block_h
            nw = wu // block_w
            latents = latents.reshape(_bq, du, nh, block_h, nw, block_w)
            latents = latents.permute(0, 2, 4, 3, 5, 1)  # (_bq, nh, nw, block_h, block_w, du)
            latents = latents.reshape(_bq * nh * nw, block_h * block_w, du)  # (_bq * nh * nw, block_h * block_w, du)

            latents = self_center(latents)  # (_bq * nh * nw, block_h * block_w, du)
            du = latents.size(-1)
            latents = latents.reshape(_bq, nh, nw, block_h, block_w, du)
            latents = latents.permute(0, 5, 1, 3, 2, 4)  # (bq, du, nh, block_h, nw, block_w)
            latents = latents.reshape(_bq, du, hu, wu)  # (bq, du, hu, wu)

            # print(f'  after 1st, latent shape: {latents.shape}')

        # latents: (bq, d, h, w) -> (b, q, h, w, d)
        _bq, du, hu, wu = latents.shape
        latents = latents.permute(0, 2, 3, 1).reshape(b, q, hu, wu, du)  # (b, q, hu, wu, du)

        # final linear
        if self.final_linear is not None:
            latents = self.final_linear(latents)  # (b, q, hu, wu, du)

        return latents  # (b, q, hu, wu, du)


def get_mlp(
    dim_in: int,
    dim_hidden: int,
    dim_out: int,
    num_layers: int,
    mlp_type: str,
    mlp_add_bias: bool,
    contract: bool,
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
        )
    )
    return torch.nn.Sequential(*layers)


class VectorDecoder2(torch.nn.Module):
    """
    Architecture overview:

    We first use a perceiver encoder to gather information from shape/color tokens.
    The queries can have two types:
    1) (q, h, w): this corresponds to multiple feature maps
    2) (m,): this corresponds to free tokens

    After perceiver, we will use a conv2d network to process (1).

    Then all the tokens (qhw, d) and (m, d) are processed by self-attention layers.

    Then the output corresponding to (1) are processed by a conv2d, potentially with upsampling.
    """

    def __init__(
        self,
        dim_input_token: int,
        dim_output: int,
        init_query_map_q: int,
        init_query_map_h: int,
        init_query_map_w: int,
        init_map_method: str,
        init_query_map_given_dim: T.Optional[int],  # only used when given init
        init_query_token_k: int,
        init_token_method: str,
        init_query_token_given_dim: T.Optional[int],  # only used when given init
        num_adapter_layers: int,
        # perceiver
        dim_perceiver: int,
        perceiver_num_blocks: int,
        perceiver_dim_qkv: int,
        perceiver_num_self_attn: int,
        perceiver_num_self_heads: int,
        perceiver_num_cross_heads: int,
        perceiver_dropout_prob: float,
        perceiver_use_rmsnorm: bool,
        perceiver_mlp_ratio: int,
        perceiver_add_write_back: bool,
        # super-res
        upsample_num_blocks: int,
        upsample_kernel_size: int,
        num_residual_layers_per_block: int,
    ):
        super().__init__()
        self.dim_input_token = dim_input_token
        self.init_map_shape = (init_query_map_q, init_query_map_h, init_query_map_w)
        self.init_num_token = init_query_token_k
        self.init_map_method = init_map_method
        self.init_query_map_given_dim = init_query_map_given_dim
        self.init_token_method = init_token_method
        self.init_query_token_given_dim = init_query_token_given_dim
        self.dim_perceiver = dim_perceiver
        self.dim_output = dim_output
        self.num_adapter_layers = num_adapter_layers

        # init query for map
        self.use_map = math.prod(self.init_map_shape) >= 1
        if self.use_map:
            if self.init_map_method == "learned_randn":
                self.init_query_map = torch.nn.Parameter(
                    torch.randn(*self.init_map_shape, self.dim_perceiver)
                )  # (q, h, w, d)
            elif self.init_map_method == "learned_zeros+poshw":
                assert self.dim_perceiver % 4 == 0
                self.hw_pos_encoder = layers.FourierEmbed(
                    dim_pos=2,  # 2hw
                    include_input=False,
                    min_freq_log2=0,
                    max_freq_log2=math.log2(max(self.init_map_shape[1:])) + 1,
                    num_freqs=self.dim_perceiver // 4,
                    log_sampling=True,
                )  # output dim = 2hw * num_freqs * 2sin_cos
                self.q_pos_encoder = _ops().Embedding(
                    num_embeddings=self.init_map_shape[0],
                    embedding_dim=self.dim_perceiver,
                )
                self.dim_pos_output = self.hw_pos_encoder.dim_out
                assert self.dim_perceiver == self.dim_pos_output, f"{self.dim_perceiver}, {self.dim_pos_output}"
                self.init_query_map = torch.nn.Parameter(
                    torch.zeros(*self.init_map_shape, self.dim_perceiver)
                )  # (q, h, w, d)
            elif self.init_map_method == "given":
                assert self.init_query_map_given_dim is not None and self.init_query_map_given_dim > 0
                if self.init_query_map_given_dim != self.dim_perceiver:
                    self.init_linear_map = _ops().Linear(
                        in_features=self.init_query_map_given_dim,
                        out_features=self.dim_perceiver,
                    )
                else:
                    self.init_linear_map = None
            else:
                raise NotImplementedError

            self.init_map_mlp = get_mlp(
                dim_in=self.dim_perceiver,
                dim_hidden=self.dim_perceiver,
                dim_out=self.dim_perceiver,
                num_layers=self.num_adapter_layers,
                mlp_type="swiglu",
                mlp_add_bias=False,
                contract=False,
            )

        # init query for free tokens
        self.use_token = self.init_num_token >= 1
        if self.use_token:
            if self.init_token_method == "learned_randn":
                self.init_query_token = torch.nn.Parameter(
                    torch.randn(self.init_num_token, self.dim_perceiver)
                )  # (k, d)
            elif self.init_token_method == "given":
                assert self.init_query_token_given_dim is not None and self.init_query_token_given_dim > 0
                if self.init_query_token_given_dim != self.dim_perceiver:
                    self.init_linear_token = _ops().Linear(
                        in_features=self.init_query_token_given_dim,
                        out_features=self.dim_perceiver,
                    )
                else:
                    self.init_linear_token = None
            else:
                raise NotImplementedError

            self.init_token_mlp = get_mlp(
                dim_in=self.dim_perceiver,
                dim_hidden=self.dim_perceiver,
                dim_out=self.dim_perceiver,
                num_layers=self.num_adapter_layers,
                mlp_type="swiglu",
                mlp_add_bias=False,
                contract=False,
            )

        # perceiver encoder (we will concat the query from map and tokens)
        self.perceiver = perceiver_encoder.PerceiverEncoder(
            dim_latent=self.dim_perceiver,
            dim_token=self.dim_input_token,
            num_blocks=perceiver_num_blocks,
            dim_qkv=perceiver_dim_qkv,
            num_self_attn=perceiver_num_self_attn,
            num_self_heads=perceiver_num_self_heads,
            num_cross_heads=perceiver_num_cross_heads,
            dropout_prob=perceiver_dropout_prob,
            use_rmsnorm=perceiver_use_rmsnorm,
            mlp_ratio=perceiver_mlp_ratio,
            add_write_back=perceiver_add_write_back,
            keep_block_bug=False,
        )
        # output is dim_perceiver

        # upsample layer (b, q*h*w, d) -> (b, q*2h*2w, d)
        if self.use_map:
            blocks = []
            current_shape = [s for s in self.init_map_shape]
            current_dim = self.dim_perceiver
            for block_idx in range(upsample_num_blocks):
                block_dict = dict()

                # (b, q*h*w, d) -> (bq, d, h, w) -> (bq, d, 2h, 2w) -> (bq, 2h, 2w, d) -> (b, q*2h*2w, d)
                assert current_dim // 2 >= 1
                if upsample_kernel_size <= 1:
                    block_dict["upsampler"] = torch.nn.Identity()
                else:
                    block_dict["upsampler"] = Upsample2DLayer(
                        scale_factor=upsample_kernel_size,
                        in_channels=current_dim,
                        out_channels=current_dim,
                        add_conv=False,
                    )
                current_shape = [
                    current_shape[0],
                    current_shape[1] * upsample_kernel_size,
                    current_shape[2] * upsample_kernel_size,
                ]

                # residual
                out_channels = current_dim // 2 if upsample_kernel_size > 1 else current_dim
                block_dict["conv"] = resnet.ResidualBlocks(
                    in_channels=current_dim,
                    out_channels=out_channels,
                    num_layers=num_residual_layers_per_block,
                    kernel_size=3,
                    stride=1,
                    activation=torch.nn.SiLU(),
                )
                current_dim = out_channels

                block_dict = torch.nn.ModuleDict(block_dict)
                blocks.append(block_dict)

            self.blocks = torch.nn.ModuleList(blocks)
            self.output_shape = current_shape  # (3qhw,)
            self.dim_upsample_output = current_dim

            # final linear layer
            if self.dim_upsample_output != self.dim_output:
                self.final_linear_map = FinalLayer(
                    dim_input=self.dim_upsample_output,
                    dim_output=self.dim_output,
                )
            else:
                self.final_linear_map = None

        if self.use_token:
            # final linear layer
            if self.dim_perceiver != self.dim_output:
                self.final_linear_token = FinalLayer(
                    dim_input=self.dim_perceiver,
                    dim_output=self.dim_output,
                )
            else:
                self.final_linear_token = None

    def forward(
        self,
        input_tokens: torch.Tensor,
        given_init_map: torch.Tensor = None,
        given_init_token: torch.Tensor = None,
    ):
        """
        Args:
            input_tokens:
                (b, num_tokens, dim_input_token)
            given_init_map:
                (b, init_query_map_q, init_query_map_h, init_query_map_w, init_query_map_given_dim)
            given_init_token:
                (b, init_query_token_k, init_query_token_given_dim)

        Returns:
            output_map:
                (b, q, hu, wu, dim_output)
            output_token:
                (b, k, dim_output)
        """

        b, m, dim_in = input_tokens.shape
        q, h, w = self.init_map_shape
        qhw = q * h * w

        # get init query map
        init_tokens = []
        if self.use_map:
            if self.init_map_method == "learned_randn":
                init_query_map = self.init_query_map.expand(b, q, h, w, self.dim_perceiver)  # (b, q, h, w, d)
            elif self.init_map_method == "learned_zeros+poshw":
                qidxs, hidxs, widxs = torch.meshgrid(
                    torch.arange(q, device=input_tokens.device),
                    torch.arange(h, device=input_tokens.device),
                    torch.arange(w, device=input_tokens.device),
                    indexing="ij",
                )  # (q, h, w)
                qhw_idxs = torch.stack([qidxs, hidxs, widxs], dim=-1)  # (q, h, w, 2hw)
                init_query_map = self.init_query_map.expand(b, q, h, w, self.dim_perceiver)  # (b, q, h, w, d)
                hw_pose = self.hw_pos_encoder(qhw_idxs[..., 1:]).expand(
                    b, q, h, w, self.dim_perceiver
                )  # (b, q, h, w, dim_perceiver)
                q_pose = (
                    self.q_pos_encoder(qhw_idxs[..., 0]).unsqueeze(0).expand(b, q, h, w, self.dim_perceiver)
                )  # (q, h, w, dim_pose)
                init_query_map = init_query_map + hw_pose + q_pose
            elif self.init_map_method == "given":
                assert given_init_map is not None
                _b, q, h, w, dim_init_map = given_init_map.shape
                qhw = q * h * w
                assert dim_init_map == self.init_query_map_given_dim
                init_query_map = given_init_map
                if self.init_linear_map is not None:
                    init_query_map = self.init_linear_map(init_query_map)  # (b, q, h, w, dim_perceiver)
            else:
                raise NotImplementedError

            init_query_map = self.init_map_mlp(init_query_map)
            init_query_map = init_query_map.reshape(b, qhw, self.dim_perceiver)  # (b, qhw, dim_perceiver)
            init_tokens.append(init_query_map)
        else:
            init_query_map = None

        if self.use_token:
            if self.init_token_method == "learned_randn":
                init_query_token = self.init_query_token.expand(b, self.init_num_token, self.dim_perceiver)
                k = self.init_linear_token
            elif self.init_token_method == "given":
                assert given_init_token is not None
                _b, k, dim_init_token = given_init_token.shape
                assert dim_init_token == self.init_query_token_given_dim
                init_query_token = given_init_token
                if self.init_linear_token is not None:
                    init_query_token = self.init_linear_token(init_query_token)  # (b, k, dim_perceiver)
            else:
                raise NotImplementedError

            init_query_token = self.init_token_mlp(init_query_token)  # (b, k, dim_perceiver)
            init_tokens.append(init_query_token)
        else:
            init_query_token = None
            k = 0

        if len(init_tokens) > 1:
            init_tokens = torch.cat(init_tokens, dim=1)  # (b, qhw+k, d)
        else:
            init_tokens = init_tokens[0]  # (b, qhw, d) or (b, k, d)

        # perceiver encoder
        latents = self.perceiver(
            input_tokens=input_tokens,  # (b, m, dim_in)
            latent_tokens=init_tokens,  # (b, qhw, d) or (b, k, d)
            return_all_layers=False,
        )  # (b, qhw, d) or (b, k, d)

        current_idx = 0
        if self.use_map:
            latents_map = latents[:, current_idx : (current_idx + qhw)]
            latents_map_before_upsample = latents_map.reshape(b, q, h, w, self.dim_perceiver)
            latents_map = latents_map.reshape(b * q, h, w, self.dim_perceiver)
            latents_map = latents_map.permute(0, 3, 1, 2)  # (bq, d, h, w)
            current_idx += qhw
        else:
            latents_map = None
            latents_map_before_upsample = None

        if self.use_token:
            latents_token = latents[:, current_idx : (current_idx + k)]  # (b, k, d)
            current_idx += k
        else:
            latents_token = None

        # upsample the map
        if latents_map is not None:
            for block_idx, block in enumerate(self.blocks):
                # latents_map: (bq, d, h, w)
                upsampler = block["upsampler"]
                conv = block["conv"]
                latents_map = upsampler(latents_map)  # (bq, d, hu, wu)
                latents_map = conv(latents_map)  # (bq, du, hu, wu)

            # latents: (bq, d, h, w) -> (b, q, h, w, d)
            _bq, du, hu, wu = latents_map.shape
            latents_map = latents_map.permute(0, 2, 3, 1).reshape(b, q, hu, wu, du)  # (b, q, hu, wu, du)

            # final linear
            if self.final_linear_map is not None:
                latents_map = self.final_linear_map(latents_map)  # (b, q, hu, wu, do)

        # token
        if latents_token is not None:
            # final linear
            if self.final_linear_token is not None:
                latents_token = self.final_linear_token(latents_token)  # (b, k, do)

        return dict(
            latents_map_before_upsample=latents_map_before_upsample,  # (b, q, h, w, dim_perceiver) or None
            latents_map=latents_map,  # (b, q, hu, wu, do) or None
            latents_token=latents_token,  # (b, k, do) or None
        )
