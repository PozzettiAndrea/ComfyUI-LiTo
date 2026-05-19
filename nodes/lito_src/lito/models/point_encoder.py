#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements the encoder that converts point cloud to shape latent.

import typing as T

import numpy as np

import torch
from torch import nn

from lito.models import perceiver_encoder, pointnet_utils
from lito.models.layers import FourierEmbed
from plibs.flash_utils import create_block_diagonal_attn_bias_from_seq_lens
_OPS = None
def _ops():
    global _OPS
    if _OPS is None:
        import comfy.ops
        _OPS = comfy.ops.manual_cast
    return _OPS



class ShapeLatent(torch.nn.Module):
    def __init__(
        self,
        num_latent: int,
        dim_latent: int,
        init_mode: str = "randn",
    ):
        """
        Args:
            num_latent:
                number of latent tokens used to encode a point cloud
            dim_latent:
                dimension of each latent token
            init_mode:
                'randn'
                'zeros'
        """
        super().__init__()
        self.num_latent = num_latent
        self.dim_latent = dim_latent

        if init_mode == "randn":
            self.latents = nn.Parameter(torch.randn(self.num_latent, self.dim_latent))  # (n, d)
        elif init_mode == "zeros":
            self.latents = nn.Parameter(torch.zeros(self.num_latent, self.dim_latent))  # (n, d)
        else:
            raise NotImplementedError

        # print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
        #       f"self.latent.dtype: {self.latents.dtype}\n"
        #       f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    def forward(self, batch_size: int):
        return self.latents.expand(batch_size, -1, -1)  # (b, n, d)


class PointEncoder(torch.nn.Module):
    """
    Point encoder takes a point cloud as input, and it constructs
    shape latents that represent the shape the point cloud was sampled from.
    """

    def __init__(
        self,
        num_latent: int,
        dim_latent: int,
        dim_point: int,
        # latent construction
        # perceiver encoder
        num_blocks: int,
        dim_qkv: int,
        num_self_attn: int = 2,
        num_self_heads: int = 4,
        num_cross_heads: int = 4,
        dropout_prob: float = 0.0,
        use_rmsnorm: bool = True,
        mlp_ratio: float = 2,
        add_write_back: bool = False,
        # positional encoding
        min_freq_log2: float = 0,
        max_freq_log2: float = 12,
        num_freqs: int = 32,
        # vnet (feature: rgb, pos_emb(xyz))
        use_vnet: bool = False,
        vnet_out_channel: int = 512,
        vnet_cell_widths: T.List[float] = (4 / 512.0, 8 / 512.0, 16 / 512.0),
        vnet_mlps_base: T.List[int] = (64, 64, 128),
        vnet_width_mult: int = 1,
        vnet_layer_mult: T.Union[T.Union[int, float], T.List[T.Union[int, float]]] = 2,
        vnet_norm_type: str = "batchnorm",
        vnet_point_aggregation_method: str = "mean",
        vnet_feature_aggregation_method: str = "amax",
        # decouple latent dimension and perceiver dimension
        dim_perceiver: int = None,
        # bug in perceiver encoder
        keep_perceiver_block_bug: bool = True,  # set to False to avoid bug
        # rgb pos encoding
        use_rgb_pos_encoder: bool = False,
        min_rgb_freq_log2: float = 0.0,
        max_rgb_freq_log2: float = 8.0,
        num_rgb_freqs: int = 8,
    ):
        super().__init__()
        self.num_latent = num_latent
        self.dim_latent = dim_latent
        self.dim_point = dim_point
        self.dim_perceiver = dim_perceiver
        if self.dim_perceiver is None:
            self.dim_perceiver = self.dim_latent
        self.keep_perceiver_block_bug = keep_perceiver_block_bug
        self.use_rgb_pos_encoder = use_rgb_pos_encoder

        # latent construction
        self.latent_constructor = ShapeLatent(
            num_latent=self.num_latent,
            dim_latent=self.dim_perceiver,
        )

        # position encoding for point
        self.xyz_pos_encoder = FourierEmbed(
            dim_pos=3,  # 3xyz
            include_input=False,  # we concat ourselves
            min_freq_log2=min_freq_log2,
            max_freq_log2=max_freq_log2,
            num_freqs=num_freqs,
            log_sampling=True,
        )

        # calculate the point token dimension
        self.dim_point_token = self.dim_point + self.xyz_pos_encoder.dim_out  # concat

        # position encoding for rgb
        if self.use_rgb_pos_encoder:
            self.rgb_pos_encoder = FourierEmbed(
                dim_pos=3,  # 3rgb [-1, 1]
                include_input=False,  # we concat ourselves
                min_freq_log2=min_rgb_freq_log2,
                max_freq_log2=max_rgb_freq_log2,
                num_freqs=num_rgb_freqs,
                log_sampling=True,
            )
            # calculate the point token dimension
            self.dim_point_token += self.rgb_pos_encoder.dim_out
        else:
            self.rgb_pos_encoder = None

        # vnet
        self.use_vnet = use_vnet
        if self.use_vnet:
            self.vnet = pointnet_utils.VNet(
                in_channel=self.dim_point_token,
                out_channel=vnet_out_channel,
                cell_widths=vnet_cell_widths,
                mlps_base=vnet_mlps_base,
                width_mult=vnet_width_mult,
                layer_mult=vnet_layer_mult,
                norm_type=vnet_norm_type,
                point_aggregation_method=vnet_point_aggregation_method,
                feature_aggregation_method=vnet_feature_aggregation_method,
            )
            self.dim_encoder_input = (
                self.vnet.out_channel
                + 3  # xyz
                + self.xyz_pos_encoder.dim_out  # encoded_xyz
            )
        else:
            self.vnet = None
            self.dim_encoder_input = self.dim_point_token

        # encoder
        self.encoder = perceiver_encoder.PerceiverEncoder(
            dim_latent=self.dim_perceiver,
            dim_token=self.dim_encoder_input,
            num_blocks=num_blocks,
            dim_qkv=dim_qkv,
            num_self_attn=num_self_attn,
            num_self_heads=num_self_heads,
            num_cross_heads=num_cross_heads,
            dropout_prob=dropout_prob,
            use_rmsnorm=use_rmsnorm,
            mlp_ratio=mlp_ratio,
            add_write_back=add_write_back,
            keep_block_bug=self.keep_perceiver_block_bug,
        )

        # output layer
        if self.dim_perceiver != self.dim_latent:
            self.final_layer = _ops().Linear(
                in_features=self.dim_perceiver,
                out_features=self.dim_latent,
            )
        else:
            self.final_layer = None

    def forward(
        self,
        # input_point_cloud: torch.Tensor,  # (b, m, dim_point)
        xyz_w: torch.Tensor,  # (b, m, 3)
        rgb: T.Optional[torch.Tensor],  # (b, m, 3)  [-1, 1]
        normal_w: T.Optional[torch.Tensor],  # (b, m, 3)
    ):
        """
        Args:
            input_point_cloud:
                (b, m, dim_point)  The first 3 dimension is xyz, then it can be rgb, normal, etc
            xyz_w:
                (b, m, 3)
            rgb:
                (b, m, 3rgb) or None. [-1, 1]
            normal_w:
                (b, m, 3xyz)

        Returns:
            latent:
                (b, num_latent, dim_latent)
        """

        b, m, _3xyz = xyz_w.shape
        latent_tokens = self.latent_constructor(batch_size=b)  # (b, num_latent, dim_perceiver)

        input_tokens = [xyz_w]
        if rgb is not None:
            input_tokens.append(rgb)
        if normal_w is not None:
            input_tokens.append(normal_w)

        # position encode the points
        encoded_xyz = self.xyz_pos_encoder(xyz_w)  # (b, m, dim_encoded_xyz)
        input_tokens.append(encoded_xyz)

        if self.use_rgb_pos_encoder:
            assert rgb is not None
            encoded_rgb = self.rgb_pos_encoder(rgb)  # (b, m, dim_encoded_rgb)
            input_tokens.append(encoded_rgb)

        input_tokens = torch.cat(input_tokens, dim=-1)  # (b, m, dim_point_token)
        assert input_tokens.size(-1) == self.dim_point_token

        if not self.use_vnet:
            # perceiver encoder
            out_latent_tokens = self.encoder(
                input_tokens=input_tokens,  # (b, m, dim_point_token)
                latent_tokens=latent_tokens,  # (b, num_latent, dim_perceiver)
                structural_attn_dicts=None,
                return_all_layers=False,
            )  # (b, num_latent, dim_perceiver)
            # out_latent_tokens = out_dict['all_layer_latents'][-1]  # (b, num_latent, dim_perceiver)

        else:
            # vnet -> perceiver encoder

            # run vnet
            out_dict = self.vnet(
                xyz=xyz_w,  # (b, m, 3)
                feature=input_tokens,  # (b, m, dim_point_token)
                max_m=None,
                bidx=None,
                b=b,
                input_format="batch",
                output_format="packed",
            )
            new_xyz_w = out_dict["xyz"]  # (bm', 3)
            new_input_tokens = out_dict["feature"]  # (bm', d)
            input_token_bidx = out_dict["bidx"]  # (bm',)

            # group same bidx together (to use block diagonal attn bias)
            input_token_bidx, ii = torch.sort(input_token_bidx, descending=False, stable=True)  # (bm',),  (bm',)
            _, input_token_bidx_counts = torch.unique_consecutive(input_token_bidx, return_counts=True)  # (b,)
            # our algorithm below assumes every b is represented
            assert input_token_bidx_counts.size(0) == b
            # input_token_first_idxs = torch.cat([
            #     input_token_bidx_counts.new(1).fill_(0),
            #     input_token_bidx_counts.cumsum(dim=0)[:-1]
            # ], dim=0)  # (b,)
            new_xyz_w = new_xyz_w[ii]  # (bm', 3)
            new_input_tokens = new_input_tokens[ii]  # (bm', d)

            # concat xyz and input_tokens
            new_input_tokens = torch.cat(
                [
                    new_xyz_w,  # (bm', 3)
                    self.xyz_pos_encoder(new_xyz_w),  # (bm', dim_pos)
                    new_input_tokens,  # (bm', d)
                ],
                dim=-1,
            )

            # since after voxel downsample, each point cloud will have
            # different number of points, we need to operate perceiver encoder
            # in packed format
            out_dict = pointnet_utils.batch_to_packed(
                arr=latent_tokens,
            )
            latent_tokens = out_dict["arr"]  # (bl, dim_perceiver)
            # latent_bidx = out_dict['bidx']  # (bl,)

            # construct the attn_bias
            # For cross attn:
            #   each latent_b (num_latents of them) can attend to
            #   all input token with the same bidx (input_token_bidx_counts of them).
            # For self attn:
            #   each latent_b (num_latents of them) can attend to
            #   latents with same latent_b (num_latents of them).
            # For writeback attn:
            #   each input token  (input_token_bidx_counts of them) can attend to
            #   latents with same bidx (num_latents of them).
            # Since we do not change the topology in perceiver_encoder,
            # the attn_bias can be reused for all layers.

            # flash attention supports max 65535 blocks
            assert b <= 65535
            input_token_bidx_counts_list = input_token_bidx_counts.detach().cpu().tolist()  # (b,)
            _ntoken = np.array(input_token_bidx_counts_list)
            # print(f'num_input_token_after_vnet: '
            #       f'avg={_ntoken.mean()}, '
            #       f'std={_ntoken.std()}, '
            #       f'max={_ntoken.max()}, '
            #       f'min={_ntoken.min()}, '
            #       f'{input_token_bidx_counts_list}')

            cross_attn_bias = create_block_diagonal_attn_bias_from_seq_lens(
                q_seqlen=[self.num_latent] * b,  # (b,)
                kv_seqlen=input_token_bidx_counts_list,  # (b,)
            )
            self_attn_bias = create_block_diagonal_attn_bias_from_seq_lens(
                q_seqlen=[self.num_latent] * b,  # (b,)
                kv_seqlen=[self.num_latent] * b,  # (b,)
            )
            writeback_attn_bias = create_block_diagonal_attn_bias_from_seq_lens(
                q_seqlen=input_token_bidx_counts_list,  # (b,)
                kv_seqlen=[self.num_latent] * b,  # (b,)
            )

            structural_attn_dicts = [
                dict(
                    cross=dict(
                        mode="flash_varlen",
                        attn_bias=cross_attn_bias,
                    ),
                    self=dict(
                        mode="flash_varlen",
                        attn_bias=self_attn_bias,
                    ),
                    writeback=dict(
                        mode="flash_varlen",
                        attn_bias=writeback_attn_bias if self.encoder.add_write_back else None,
                    ),
                )
            ] * self.encoder.num_blocks

            # run perceiver encoder with struct attention
            out_latent_tokens = self.encoder(
                input_tokens=new_input_tokens.unsqueeze(0),  # (1, bm, dim_point_token)
                latent_tokens=latent_tokens.unsqueeze(0),  # (1, b*num_latent, dim_perceiver)
                structural_attn_dicts=structural_attn_dicts,
                return_all_layers=False,
            )  # (1, b*num_latent, dim_perceiver)
            # out_latent_tokens = out_dict['all_layer_latents'][-1]  # (1, b*num_latent, dim_perceiver)

            # convert packed latent back to batch format
            # normally we call packed_to_batch, but in this case it is simple
            out_latent_tokens = out_latent_tokens.reshape(b, self.num_latent, self.dim_perceiver)

        # final layer
        if self.final_layer is not None:
            out_latent_tokens = self.final_layer(out_latent_tokens)  # (b, num_latent, dim_latent)

        return out_latent_tokens
