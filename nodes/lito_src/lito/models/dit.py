#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements the Diffusion Transformer that can take conditioning tokens as input.

import typing as T

import einops
from timm.models.vision_transformer import DropPath
from lito.models._comfy_mlp import Mlp

import torch
from torch import nn
import torch.nn.functional as F

from lito.models.layers import CrossAttentionLayer, FinalLayer, SelfAttentionLayer
from lito.script_utils import config_utils
_OPS = None
def _ops():
    global _OPS
    if _OPS is None:
        import comfy.ops
        _OPS = comfy.ops.manual_cast
    return _OPS

def _comfy_device():
    import comfy.model_management
    return comfy.model_management.get_torch_device()


def pixart_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ConditionEmbedder(nn.Module):
    """
    Embeds condition tokens into vector representations. Also handles dropout for classifier-free guidance.
    """

    def __init__(
        self,
        dim_cond_token,
        dim_hidden,
        cond_drop_prob,
    ):
        super().__init__()
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.y_proj = Mlp(
            in_features=dim_cond_token,
            hidden_features=dim_hidden,
            out_features=dim_hidden,
            act_layer=approx_gelu,
            drop=0.0,
        )
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(dim_cond_token) / dim_cond_token**0.5))
        self.cond_drop_prob = cond_drop_prob

    def token_drop(self, cond, force_drop_ids=None):
        """
        Drops cond to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(cond.shape[0]).to(_comfy_device()) < self.cond_drop_prob
        else:
            drop_ids = force_drop_ids == 1
        cond = torch.where(drop_ids[:, None, None], self.y_embedding, cond)
        return cond

    def forward(self, cond, train, force_drop_ids=None):
        if train:
            assert cond.shape[-1] == self.y_embedding.shape[-1]
        use_dropout = self.cond_drop_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            cond = self.token_drop(cond, force_drop_ids)
        cond = self.y_proj(cond)
        return cond


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of=256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = _ops().Linear(dim, hidden_dim, bias=False)
        self.w2 = _ops().Linear(hidden_dim, dim, bias=True)
        self.w3 = _ops().Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim_hidden,
        dim_cond_token,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        use_rmsnorm=True,
        use_swiglu=False,
        **block_kwargs,
    ):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.dim_cond_token = dim_cond_token
        self.attn = SelfAttentionLayer(
            dim_in=dim_hidden,
            dim_qkv=dim_hidden,
            num_heads=num_heads,
            use_rmsnorm=use_rmsnorm,
        )
        self.norm1 = _ops().LayerNorm(dim_hidden, elementwise_affine=False, eps=1e-6)

        # add conditional tokens with cross attention
        if self.dim_cond_token is not None:
            self.cross_attn = CrossAttentionLayer(
                dim_q=dim_hidden,
                dim_kv=dim_cond_token,
                dim_qkv=dim_hidden,
                num_heads=num_heads,
                use_rmsnorm=use_rmsnorm,
            )
            self.norm2 = _ops().LayerNorm(dim_hidden, elementwise_affine=False, eps=1e-6)

        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            self.mlp = SwiGLUFeedForward(dim_hidden, int(dim_hidden * mlp_ratio))
        else:
            self.mlp = Mlp(
                in_features=dim_hidden, hidden_features=int(dim_hidden * mlp_ratio), act_layer=approx_gelu, drop=0
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim_hidden) / dim_hidden**0.5)

    def forward(self, x, y, t, self_struct_attn_dict=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        x = x + self.drop_path(
            gate_msa
            * self.attn(
                pixart_modulate(self.norm1(x), shift_msa, scale_msa), structural_attn_dict=self_struct_attn_dict
            ).reshape(B, N, C)
        )
        if y is not None and self.dim_cond_token is not None:
            x = x + self.cross_attn(x, y)
            x = self.norm2(x)
        x = x + self.drop_path(gate_mlp * self.mlp(pixart_modulate(x, shift_mlp, scale_mlp)))

        return x


class DiffusionTransformer(nn.Module):
    """
    Velocity decoder that takes (noisy) latents and additional
    conditioning tokens as input, and outputs the estimated velocity.

    The conditioning tokens are attended by the latents with cross attention
    at the beginning of each block.
    """

    def __init__(
        self,
        # timestep embedding
        time_embedder_config,
        num_latent: int,
        dim_latent: int,
        dim_cond_token: int = None,
        patch_size: int = 1,
        cond_drop_prob: float = 0.1,
        init_pos_emb_dim: int = None,
        # block
        num_blocks: int = 28,
        dim_hidden: int = 1152,
        num_self_heads: int = 4,
        drop_path: float = 0.0,
        use_rmsnorm: bool = True,
        use_swiglu: bool = False,
        mlp_ratio: float = 4,
        # positional encoding
        init_positional_encoding: torch.Tensor = None,
        learn_positional_encoding: bool = True,
        # output
        dim_output: int = None,
    ):
        """
        Args:
            num_latent:
                number of latent per shape
            dim_latent:
                dimension of each latent
            dim_cond_token:
                dimension of the conditioning tokens
            dim_output:
                output dimension of the model, if None, the same as dim_point
            init_positional_encoding:
                (num_latent, dim_latent).  if None, it will be set to all 0.
        """

        super().__init__()
        self.num_latent = num_latent
        assert num_latent % patch_size == 0, f"num_latent {num_latent} must be divisible by patch_size {patch_size}"
        self.dim_latent = dim_latent
        self.num_latent = num_latent // patch_size
        self.dim_latent_in = dim_latent * patch_size
        self.patch_size = patch_size
        self.dim_hidden = dim_hidden
        self.dim_cond_token = dim_cond_token
        self.dim_output = dim_output if dim_output is not None else dim_latent
        self.learn_positional_encoding = learn_positional_encoding
        self.init_pos_emb_dim = init_pos_emb_dim if init_pos_emb_dim is not None else dim_hidden

        # # position encoding for latent
        # if init_positional_encoding is None:
        #     init_positional_encoding = torch.zeros(self.num_latent, self.dim_hidden)
        # assert init_positional_encoding.shape == (self.num_latent, self.dim_hidden)
        # if self.learn_positional_encoding:
        #     self.pos_mtx = torch.nn.Parameter(init_positional_encoding)
        # else:
        #     self.register_buffer('pos_mtx', init_positional_encoding)

        # token projection
        self.z_proj = _ops().Linear(self.dim_latent_in, dim_hidden)
        self.z_proj_ln = _ops().LayerNorm(dim_hidden, eps=1e-6)

        # positional embedding projection
        if init_pos_emb_dim != dim_hidden:
            self.pos_proj = _ops().Linear(self.init_pos_emb_dim, dim_hidden)

        # timestep embedding
        self.time_embedder_config = time_embedder_config
        self.t_embedder = config_utils.instantiate_from_config(self.time_embedder_config)
        self.t_proj = nn.Sequential(
            _ops().Linear(self.t_embedder.dim_out, self.dim_hidden, bias=True),
            nn.SiLU(),
            _ops().Linear(self.dim_hidden, self.dim_hidden, bias=True),
        )
        self.t0_proj = nn.Sequential(
            nn.SiLU(),
            _ops().Linear(self.dim_hidden, 6 * self.dim_hidden, bias=True),
        )

        if self.dim_cond_token is not None:
            # approx_gelu = lambda: nn.GELU(approximate="tanh")
            # self.y_proj = Mlp(
            #     in_features=self.dim_cond_token,
            #     hidden_features=self.dim_hidden,
            #     out_features=self.dim_hidden,
            #     act_layer=approx_gelu,
            #     drop=0.0,
            # )
            self.cond_embedder = ConditionEmbedder(
                dim_cond_token=self.dim_cond_token,
                dim_hidden=self.dim_hidden,
                cond_drop_prob=cond_drop_prob,
            )
            self.dim_cond_block = self.dim_hidden
        else:
            self.dim_cond_block = None

        # DiT blocks
        # self attn on latents, cross attn on cond_tokens
        blocks = []
        for _ in range(num_blocks):
            blck = DiTBlock(
                dim_hidden=self.dim_hidden,
                dim_cond_token=self.dim_cond_block,
                num_heads=num_self_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
                use_rmsnorm=use_rmsnorm,
                use_swiglu=use_swiglu,
            )
            blocks.append(blck)
        self.blocks = torch.nn.ModuleList(blocks)

        # final layer
        self.final_layer = FinalLayer(
            dim_input=self.dim_hidden,
            dim_output=self.dim_output * patch_size,
            dim_cond_feature=self.dim_hidden,
        )

        self.initialize_weights()

    def init_positional_embedding(self, pos_mtx=None):
        if self.learn_positional_encoding:
            if pos_mtx is not None:
                self.pos_mtx = torch.nn.Parameter(pos_mtx, requires_grad=True)
                print(f"Initialized positional encoding with shape {self.pos_mtx.shape}")
            else:
                init_positional_encoding = torch.zeros(self.num_latent, self.dim_hidden)
                self.pos_mtx = torch.nn.Parameter(init_positional_encoding, requires_grad=True)
        else:
            init_positional_encoding = torch.zeros(self.num_latent, self.dim_hidden)
            self.register_buffer("pos_mtx", init_positional_encoding)

    def patchify(self, x):
        bsz, n, c = x.shape
        p = self.patch_size
        x = einops.rearrange(x, "b (p t) c -> b t (p c)", p=p)
        return x

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        x = einops.rearrange(x, "b t (p c) -> b (p t) c", p=p)
        return x

    def forward(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor = None,
        structural_attn_dicts: T.List[T.Dict[str, T.Dict[str, T.Any]]] = None,
        cond_drop_ids: torch.Tensor = None,
        cond_use_grad_checkpointing: bool = False,
        debug: bool = False,
    ):
        """
        Args:
            tokens:
                (b, M_t, D_t)  The tokens representing the shape
            t:
                (b, ) The timestep
            cond:
                (b, M_c, D_c)  The conditional tokens (img patches)
            structural_attn_dicts:
                a list of dict(str, structural_attn_dict), one for each layer.
                each dict contains
                    cross: list of structural_attn_dict to be used for each layer of cross attention in the block.
                    self: list of structural_attn_dict to be used for each layer of self attention in the block.
            cond_drop_ids:
                (b, )  The ids of the conditional tokens to be dropped

        Returns:
            output_velocity:
                (b, M_t, D_o)  output velocity at each point
        """

        b, num_latent, dim_latent = tokens.shape
        assert (num_latent, dim_latent) == (
            self.num_latent,
            self.dim_latent,
        ), f"{(num_latent, dim_latent)}, {(self.num_latent, self.dim_latent)}"

        if cond is not None:
            _b, num_cond_token, dim_cond_token = cond.shape
            assert (_b, dim_cond_token) == (b, self.dim_cond_token), f"{cond.shape=}, {b=}, {self.dim_cond_token=}"

        if structural_attn_dicts is not None:
            assert len(structural_attn_dicts) == len(self.blocks)

        # timestep embedding
        t = self.t_embedder(t.unsqueeze(-1))
        t = self.t_proj(t)  # (b, D_h)
        t0 = self.t0_proj(t)  # (b, 6 * D_h)

        # conditional token embedding
        if cond is not None and self.dim_cond_token is not None:
            if cond_use_grad_checkpointing:
                cond = torch.utils.checkpoint.checkpoint(
                    self.cond_embedder,
                    cond,
                    self.training,
                    cond_drop_ids,
                    use_reentrant=False,
                )  # (b, num_extra + phpw, d)
            else:
                cond = self.cond_embedder(cond, self.training, force_drop_ids=cond_drop_ids)  # (b, M_c, D_h)

        # group tokens into latents
        if self.patch_size > 1:
            latents = self.patchify(tokens)  # (b, (p * M_l), D_t) -> (b, M_l, (p * D_t))
        else:
            latents = tokens  # (b, M_t, D_t)
        latents = self.z_proj(latents)  # (b, M_l, D_h), M_l = M_t / p
        latents = self.z_proj_ln(latents)

        # positional encoding
        if hasattr(self, "pos_proj"):
            pos_embed = self.pos_proj(self.pos_mtx)
        else:
            pos_embed = self.pos_mtx
        latents = latents + pos_embed.unsqueeze(0)  # (b, M, D_h)
        # a quick hack to support various number of latents
        # latents = latents + pos_embed[:latents.size(1)].unsqueeze(0)  # (b, M, D_h)
        # latents = latents + self.pos_mtx.unsqueeze(0)  # (b, M, D_h)

        # block
        for i, block in enumerate(self.blocks):
            if structural_attn_dicts is not None:
                self_struct_attn_dict = structural_attn_dicts[i].get("self", None)
            else:
                self_struct_attn_dict = None

            latents = block(
                x=latents,
                y=cond,
                t=t0,
                self_struct_attn_dict=self_struct_attn_dict,
            )  # (b, M_l, D_h)

        # final linear layer
        out = self.final_layer(x=latents, cond_feature=t)  # (b, M_l, (D * p_o))

        # ungroup latents into tokens
        out = self.unpatchify(out)  # (b, M_l, (D * p_o)) -> (b, (p * M_l), D_o)

        if debug:
            assert out.isfinite().all(), f"nan: {out.isnan().any()}, inf: {out.isinf().any()}"

        return out  # (b, M_t, D_o)

    def forward_with_cfg(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor = None,
        structural_attn_dicts: T.List[T.Dict[str, T.Dict[str, T.Any]]] = None,
        cfg_scale: float = 1.0,
        cond_drop_ids: torch.Tensor = None,
        debug: bool = False,
    ):
        """
        Batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = tokens[: len(tokens) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, cond, structural_attn_dicts, cond_drop_ids, debug)
        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
        # cond_eps = self.forward(tokens, t, cond, structural_attn_dicts, debug)
        # uncond_eps = self.forward(tokens, t, None, structural_attn_dicts, debug)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_proj[0].weight, std=0.02)
        nn.init.normal_(self.t_proj[2].weight, std=0.02)
        nn.init.normal_(self.t0_proj[1].weight, std=0.02)

        # Initialize cond embedding MLP:
        if self.dim_cond_token is not None:
            nn.init.normal_(self.cond_embedder.y_proj.fc1.weight, std=0.02)
            nn.init.normal_(self.cond_embedder.y_proj.fc2.weight, std=0.02)

            # Zero-out adaLN modulation layers in PixArt blocks:
            for block in self.blocks:
                nn.init.constant_(block.cross_attn.linear_out.weight, 0)
                nn.init.constant_(block.cross_attn.linear_out.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
