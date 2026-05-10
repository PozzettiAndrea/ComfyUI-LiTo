#
# Copyright (C) 2026 Apple Inc. All rights reserved.
#
# MLX implementation of the Diffusion Transformer.
# Inference only — mirrors src/lito/models/dit.py

import typing as T

from mlx import nn
import mlx.core as mx

from lito.mlx.models.layers import (
    CrossAttentionLayer,
    FinalLayer,
    FourierEmbed,
    Mlp,
    SelfAttentionLayer,
    SwiGLUFeedForward,
)


def pixart_modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    """PixArt-style adaptive layer norm modulation.

    Args:
        x: Input. (b, n, d)
        shift: Shift vector. (b, 1, d)
        scale: Scale vector. (b, 1, d)

    Returns:
        Modulated output. (b, n, d)
    """
    return x * (1 + scale) + shift


class ConditionEmbedder(nn.Module):
    """Embeds condition tokens into vector representations.

    Handles token dropout for classifier-free guidance.

    Args:
        dim_cond_token: Dimension of input conditioning tokens.
        dim_hidden: Hidden / output dimension.
    """

    def __init__(self, dim_cond_token: int, dim_hidden: int):
        super().__init__()
        self.y_proj = Mlp(
            in_features=dim_cond_token,
            hidden_features=dim_hidden,
            out_features=dim_hidden,
        )
        self.y_embedding = mx.zeros((dim_cond_token,))  # placeholder, loaded from weights

    def token_drop(self, cond: mx.array, force_drop_ids: mx.array) -> mx.array:
        """Drop conditioning tokens for classifier-free guidance.

        Args:
            cond: Conditioning tokens. (b, m, dim_cond_token)
            force_drop_ids: Boolean mask indicating which samples to drop. (b,)

        Returns:
            Conditioning tokens with dropped samples replaced by y_embedding. (b, m, dim_cond_token)
        """
        drop_ids = force_drop_ids.astype(mx.bool_)  # (b,)
        # Broadcast: (b, 1, 1) * (dim_cond_token,) -> replace entire sequence for dropped samples
        return mx.where(drop_ids[:, None, None], self.y_embedding, cond)  # (b, m, dim_cond_token)

    def __call__(
        self,
        cond: mx.array,
        force_drop_ids: T.Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            cond: Conditioning tokens. (b, m, dim_cond_token)
            force_drop_ids: Boolean mask for CFG dropout. (b,)

        Returns:
            Embedded conditioning tokens. (b, m, dim_hidden)
        """
        if force_drop_ids is not None:
            cond = self.token_drop(cond, force_drop_ids)  # (b, m, dim_cond_token)
        return self.y_proj(cond)  # (b, m, dim_hidden)


class DiTBlock(nn.Module):
    """Diffusion Transformer block with adaLN modulation.

    Contains self-attention, optional cross-attention, and MLP with
    PixArt-style adaptive layer norm modulation from timestep embedding.

    Args:
        dim_hidden: Hidden dimension.
        dim_cond_token: Dimension of conditioning tokens for cross-attention. None to disable.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim ratio.
        use_rmsnorm: Whether to use RMSNorm in attention.
        use_swiglu: Whether to use SwiGLU MLP instead of standard MLP.
    """

    def __init__(
        self,
        dim_hidden: int,
        dim_cond_token: T.Optional[int],
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_rmsnorm: bool = True,
        use_swiglu: bool = False,
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
        self.norm1 = nn.LayerNorm(dim_hidden, affine=False, eps=1e-6)

        if dim_cond_token is not None:
            self.cross_attn = CrossAttentionLayer(
                dim_q=dim_hidden,
                dim_kv=dim_cond_token,
                dim_qkv=dim_hidden,
                num_heads=num_heads,
                use_rmsnorm=use_rmsnorm,
            )
            self.norm2 = nn.LayerNorm(dim_hidden, affine=False, eps=1e-6)

        if use_swiglu:
            self.mlp = SwiGLUFeedForward(dim_hidden, int(dim_hidden * mlp_ratio))
        else:
            self.mlp = Mlp(
                in_features=dim_hidden,
                hidden_features=int(dim_hidden * mlp_ratio),
                out_features=dim_hidden,
            )

        self.scale_shift_table = mx.zeros((6, dim_hidden))  # placeholder, loaded from weights

    def __call__(self, x: mx.array, y: T.Optional[mx.array], t: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Latent tokens. (b, n, dim_hidden)
            y: Conditioning tokens. (b, m, dim_cond_token) or None
            t: Timestep modulation features. (b, 6 * dim_hidden)

        Returns:
            Updated latent tokens. (b, n, dim_hidden)
        """
        b, n, c = x.shape

        # Extract 6 modulation vectors from scale_shift_table + timestep
        mod = self.scale_shift_table[None] + t.reshape(b, 6, -1)  # (b, 6, dim_hidden)
        shift_msa = mod[:, 0:1, :]  # (b, 1, d)
        scale_msa = mod[:, 1:2, :]  # (b, 1, d)
        gate_msa = mod[:, 2:3, :]  # (b, 1, d)
        shift_mlp = mod[:, 3:4, :]  # (b, 1, d)
        scale_mlp = mod[:, 4:5, :]  # (b, 1, d)
        gate_mlp = mod[:, 5:6, :]  # (b, 1, d)

        # Self-attention with adaLN modulation
        x = x + gate_msa * self.attn(pixart_modulate(self.norm1(x), shift_msa, scale_msa))  # (b, n, dim_hidden)

        # Cross-attention (if conditioning tokens)
        if y is not None and self.dim_cond_token is not None:
            x = x + self.cross_attn(x, y)  # (b, n, dim_hidden)
            x = self.norm2(x)  # (b, n, dim_hidden)

        # MLP with adaLN modulation
        x = x + gate_mlp * self.mlp(pixart_modulate(x, shift_mlp, scale_mlp))  # (b, n, dim_hidden)

        return x


class DiffusionTransformer(nn.Module):
    """Velocity decoder for flow matching diffusion.

    Takes (noisy) latent tokens and conditioning tokens, outputs estimated velocity.
    Conditioning tokens are attended via cross attention in each block.

    Args:
        num_latent: Number of latent tokens per shape.
        dim_latent: Dimension of each latent token.
        dim_hidden: Hidden dimension of the transformer.
        dim_cond_token: Dimension of conditioning tokens. None to disable.
        patch_size: Token patchification factor.
        num_blocks: Number of DiT blocks.
        num_self_heads: Number of self-attention heads.
        use_rmsnorm: Whether to use RMSNorm in attention.
        use_swiglu: Whether to use SwiGLU MLP.
        mlp_ratio: MLP hidden dim ratio.
        dim_output: Output dimension. If None, same as dim_latent.
        fourier_embed_config: Config dict for FourierEmbed (dim_pos, include_input, etc.).
        has_pos_proj: Whether to include a positional embedding projection layer.
        init_pos_emb_dim: Dimension of the positional embedding before projection.
    """

    def __init__(
        self,
        num_latent: int,
        dim_latent: int,
        dim_hidden: int,
        dim_cond_token: T.Optional[int] = None,
        patch_size: int = 1,
        num_blocks: int = 28,
        num_self_heads: int = 4,
        use_rmsnorm: bool = True,
        use_swiglu: bool = False,
        mlp_ratio: float = 4.0,
        dim_output: T.Optional[int] = None,
        fourier_embed_config: T.Optional[T.Dict[str, T.Any]] = None,
        has_pos_proj: bool = False,
        init_pos_emb_dim: T.Optional[int] = None,
    ):
        super().__init__()
        self.dim_latent = dim_latent
        self.patch_size = patch_size
        self.num_latent = num_latent // patch_size
        self.dim_latent_in = dim_latent * patch_size
        self.dim_hidden = dim_hidden
        self.dim_cond_token = dim_cond_token
        self.dim_output = dim_output if dim_output is not None else dim_latent

        # Token projection
        self.z_proj = nn.Linear(self.dim_latent_in, dim_hidden)
        self.z_proj_ln = nn.LayerNorm(dim_hidden, eps=1e-6)

        # Positional embedding projection (optional)
        self.has_pos_proj = has_pos_proj
        if has_pos_proj:
            _init_pos_emb_dim = init_pos_emb_dim if init_pos_emb_dim is not None else dim_hidden
            self.pos_proj = nn.Linear(_init_pos_emb_dim, dim_hidden)

        # Positional embedding — placeholder, loaded from weights
        self.pos_mtx = mx.zeros((self.num_latent, dim_hidden))

        # Timestep embedding
        if fourier_embed_config is not None:
            self.t_embedder = FourierEmbed(**fourier_embed_config)
        else:
            self.t_embedder = FourierEmbed(
                dim_pos=1, include_input=False, min_freq_log2=0, max_freq_log2=12, num_freqs=32, log_sampling=True
            )
        self.t_proj_linear1 = nn.Linear(self.t_embedder.dim_out, dim_hidden, bias=True)
        self.t_proj_linear2 = nn.Linear(dim_hidden, dim_hidden, bias=True)
        self.t0_proj_linear = nn.Linear(dim_hidden, 6 * dim_hidden, bias=True)

        # Conditioning embedder
        if dim_cond_token is not None:
            self.cond_embedder = ConditionEmbedder(
                dim_cond_token=dim_cond_token,
                dim_hidden=dim_hidden,
            )
            dim_cond_block = dim_hidden
        else:
            dim_cond_block = None

        # DiT blocks
        self.blocks = [
            DiTBlock(
                dim_hidden=dim_hidden,
                dim_cond_token=dim_cond_block,
                num_heads=num_self_heads,
                mlp_ratio=mlp_ratio,
                use_rmsnorm=use_rmsnorm,
                use_swiglu=use_swiglu,
            )
            for _ in range(num_blocks)
        ]

        # Final layer
        self.final_layer = FinalLayer(
            dim_input=dim_hidden,
            dim_output=self.dim_output * patch_size,
            dim_cond_feature=dim_hidden,
        )

    def patchify(self, x: mx.array) -> mx.array:
        """Group tokens into patches.

        Args:
            x: Input tokens. (b, p * t, c)

        Returns:
            Patchified tokens. (b, t, p * c)
        """
        b, n, c = x.shape
        p = self.patch_size
        t = n // p
        return x.reshape(b, t, p * c)  # (b, t, p * c)

    def unpatchify(self, x: mx.array) -> mx.array:
        """Ungroup patches back to tokens.

        Args:
            x: Patchified tokens. (b, t, p * c)

        Returns:
            Unpatchified tokens. (b, p * t, c)
        """
        b, t, pc = x.shape
        p = self.patch_size
        c = pc // p
        return x.reshape(b, t * p, c)  # (b, p * t, c)

    def __call__(
        self,
        tokens: mx.array,
        t: mx.array,
        cond: T.Optional[mx.array] = None,
        cond_drop_ids: T.Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            tokens: Noisy latent tokens. (b, num_latent * patch_size, dim_latent)
            t: Timestep. (b,)
            cond: Conditioning tokens. (b, m, dim_cond_token) or None
            cond_drop_ids: Boolean mask for CFG dropout. (b,) or None

        Returns:
            Estimated velocity. (b, num_latent * patch_size, dim_output)
        """
        # Timestep embedding
        t_emb = self.t_embedder(mx.expand_dims(t, axis=-1))  # (b, t_emb_dim)
        t_emb = nn.silu(self.t_proj_linear1(t_emb))  # (b, dim_hidden)
        t_emb = self.t_proj_linear2(t_emb)  # (b, dim_hidden)
        t0 = nn.silu(t_emb)  # (b, dim_hidden)
        t0 = self.t0_proj_linear(t0)  # (b, 6 * dim_hidden)

        # Conditional token embedding
        if cond is not None and self.dim_cond_token is not None:
            cond = self.cond_embedder(cond, force_drop_ids=cond_drop_ids)  # (b, m, dim_hidden)

        # Patchify
        if self.patch_size > 1:
            latents = self.patchify(tokens)  # (b, num_latent, dim_latent_in)
        else:
            latents = tokens  # (b, num_latent, dim_latent)

        latents = self.z_proj(latents)  # (b, num_latent, dim_hidden)
        latents = self.z_proj_ln(latents)  # (b, num_latent, dim_hidden)

        # Positional encoding
        if self.has_pos_proj:
            pos_embed = self.pos_proj(self.pos_mtx)  # (num_latent, dim_hidden)
        else:
            pos_embed = self.pos_mtx  # (num_latent, dim_hidden)
        latents = latents + pos_embed[None]  # (b, num_latent, dim_hidden)

        # DiT blocks
        for block in self.blocks:
            latents = block(x=latents, y=cond, t=t0)  # (b, num_latent, dim_hidden)

        # Final layer
        out = self.final_layer(x=latents, cond_feature=t_emb)  # (b, num_latent, dim_output * patch_size)

        # Unpatchify
        out = self.unpatchify(out)  # (b, num_latent * patch_size, dim_output)

        return out

    def forward_with_cfg(
        self,
        tokens: mx.array,
        t: mx.array,
        cond: T.Optional[mx.array] = None,
        cfg_scale: float = 1.0,
        cond_drop_ids: T.Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass with classifier-free guidance.

        Batches the conditional and unconditional forward passes.

        Args:
            tokens: Noisy latent tokens (doubled batch: [cond, uncond]). (2b, n, d)
            t: Timestep (doubled batch). (2b,)
            cond: Conditioning tokens (doubled batch). (2b, m, dim_cond_token)
            cfg_scale: Classifier-free guidance scale.
            cond_drop_ids: Boolean mask (doubled: [False..., True...]). (2b,)

        Returns:
            CFG-interpolated velocity (doubled batch). (2b, n, dim_output)
        """
        half = tokens[: tokens.shape[0] // 2]  # (b, n, d)
        combined = mx.concatenate([half, half], axis=0)  # (2b, n, d)
        model_out = self(combined, t, cond, cond_drop_ids)  # (2b, n, dim_output)

        mid = model_out.shape[0] // 2
        cond_eps = model_out[:mid]  # (b, n, dim_output)
        uncond_eps = model_out[mid:]  # (b, n, dim_output)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)  # (b, n, dim_output)

        return mx.concatenate([half_eps, half_eps], axis=0)  # (2b, n, dim_output)
