#
# Copyright (C) 2026 Apple Inc. All rights reserved.
#
# PyTorch to MLX weight conversion for the DiffusionTransformer.

import typing as T

import mlx.core as mx

import torch

from lito.mlx.models.dit import DiffusionTransformer as MLXDiffusionTransformer
from lito.models.dit import DiffusionTransformer as TorchDiffusionTransformer


def torch_to_mx(tensor: torch.Tensor) -> mx.array:
    """Convert a PyTorch tensor to an MLX array.

    Args:
        tensor: PyTorch tensor (any device/dtype).

    Returns:
        MLX array in float32.
    """
    return mx.array(tensor.detach().cpu().float().numpy())


def build_mlx_dit(torch_model: TorchDiffusionTransformer) -> MLXDiffusionTransformer:
    """Build an MLX DiffusionTransformer from a PyTorch DiffusionTransformer.

    Extracts architecture parameters and weights from the PyTorch model,
    constructs the MLX equivalent, and loads the converted weights.

    Args:
        torch_model: A PyTorch ``lito.models.dit.DiffusionTransformer`` instance.

    Returns:
        An MLX DiffusionTransformer with the same weights.
    """
    # Extract architecture params from the PyTorch model
    num_latent_orig = torch_model.num_latent * torch_model.patch_size  # undo the patch_size division
    dim_latent = torch_model.dim_latent
    dim_hidden = torch_model.dim_hidden
    dim_cond_token = torch_model.dim_cond_token
    patch_size = torch_model.patch_size
    num_blocks = len(torch_model.blocks)
    num_self_heads = torch_model.blocks[0].attn.num_heads
    dim_output = torch_model.dim_output

    # Detect use_rmsnorm from first block
    use_rmsnorm = hasattr(torch_model.blocks[0].attn, "rmsnorm_q")

    # Detect use_swiglu from first block
    use_swiglu = hasattr(torch_model.blocks[0].mlp, "w1")

    # Detect mlp_ratio from first block
    if use_swiglu:
        # SwiGLU: hidden_dim was computed as multiple_of * ceil(int(2/3 * dim * mlp_ratio) / multiple_of)
        # We can recover it from w1 weight shape
        _swiglu_hidden = torch_model.blocks[0].mlp.w1.weight.shape[0]
        # Approximate mlp_ratio: hidden_dim ≈ 2/3 * dim * mlp_ratio (before rounding)
        # We need to pass it so the MLX model creates the same hidden dim
        mlp_ratio = (_swiglu_hidden * 3 / 2) / dim_hidden
        # But since SwiGLU rounds up, we need to find the mlp_ratio that gives the same result
        # Try common values
        for candidate in [4.0, 3.0, 2.0, 8 / 3, 6.0]:
            test_hidden = int(2 * int(dim_hidden * candidate) / 3)
            test_hidden = 256 * ((test_hidden + 256 - 1) // 256)
            if test_hidden == _swiglu_hidden:
                mlp_ratio = candidate
                break
    else:
        # Standard Mlp: hidden_features = int(dim_hidden * mlp_ratio)
        mlp_ratio = torch_model.blocks[0].mlp.fc1.weight.shape[0] / dim_hidden

    # Detect FourierEmbed config
    t_emb = torch_model.t_embedder
    fourier_embed_config = dict(
        dim_pos=t_emb.dim_pos,
        include_input=t_emb.include_input,
        min_freq_log2=t_emb.min_freq_log2,
        max_freq_log2=t_emb.max_freq_log2,
        num_freqs=t_emb.num_freqs,
        log_sampling=t_emb.log_sampling,
    )

    # Detect pos_proj
    has_pos_proj = hasattr(torch_model, "pos_proj")
    init_pos_emb_dim = torch_model.init_pos_emb_dim if has_pos_proj else None

    # Construct MLX model
    mlx_model = MLXDiffusionTransformer(
        num_latent=num_latent_orig,
        dim_latent=dim_latent,
        dim_hidden=dim_hidden,
        dim_cond_token=dim_cond_token,
        patch_size=patch_size,
        num_blocks=num_blocks,
        num_self_heads=num_self_heads,
        use_rmsnorm=use_rmsnorm,
        use_swiglu=use_swiglu,
        mlp_ratio=mlp_ratio,
        dim_output=dim_output,
        fourier_embed_config=fourier_embed_config,
        has_pos_proj=has_pos_proj,
        init_pos_emb_dim=init_pos_emb_dim,
    )

    # Convert weights
    weight_pairs = _convert_state_dict(torch_model, use_swiglu=use_swiglu, dim_cond_token=dim_cond_token)
    mlx_model.load_weights(weight_pairs)
    mx.eval(mlx_model.parameters())

    return mlx_model


# ---- Key mapping for nn.Sequential → explicit named layers ----
#
# PyTorch nn.Sequential indices:
#   t_proj:   0=Linear, 1=SiLU, 2=Linear   → t_proj_linear1, t_proj_linear2
#   t0_proj:  0=SiLU, 1=Linear              → t0_proj_linear
#   adaLN_modulation: 0=Linear, 1=SiLU, 2=Linear → adaLN_linear1, adaLN_linear2

_SEQUENTIAL_MAP = {
    "t_proj.0.": "t_proj_linear1.",
    "t_proj.2.": "t_proj_linear2.",
    "t0_proj.1.": "t0_proj_linear.",
}

_FINAL_LAYER_SEQ_MAP = {
    "adaLN_modulation.0.": "adaLN_linear1.",
    "adaLN_modulation.2.": "adaLN_linear2.",
}


def _remap_key(key: str) -> T.Optional[str]:
    """Remap a PyTorch state_dict key to the corresponding MLX weight name.

    Returns None if the key should be skipped (e.g., parameterless layers).
    """
    # Skip SiLU / GELU / Dropout / DropPath / Identity (no params)
    # These are represented by sequential indices that we don't map
    for skip_pattern in ["t_proj.1.", "t0_proj.0.", "adaLN_modulation.1."]:
        if skip_pattern in key:
            return None

    # Skip norm1, norm2 (elementwise_affine=False) — they have no weight/bias in state_dict
    # but if they somehow appear, skip
    for norm_pat in [".norm1.weight", ".norm1.bias", ".norm2.weight", ".norm2.bias"]:
        if norm_pat in key and "final_layer" not in key:
            return None

    # Top-level sequential remapping
    for pt_prefix, mlx_prefix in _SEQUENTIAL_MAP.items():
        if key.startswith(pt_prefix):
            return mlx_prefix + key[len(pt_prefix) :]

    # FinalLayer sequential remapping
    if "final_layer.adaLN_modulation." in key:
        for pt_prefix, mlx_prefix in _FINAL_LAYER_SEQ_MAP.items():
            full_pt = "final_layer." + pt_prefix
            if key.startswith(full_pt):
                return "final_layer." + mlx_prefix + key[len(full_pt) :]
        return None  # Skip unknown sequential indices

    # norm_final with elementwise_affine=False has no params — skip if it appears
    if key == "final_layer.norm_final.weight" or key == "final_layer.norm_final.bias":  # noqa: PLR1714
        # Only skip if FinalLayer was constructed with dim_cond_feature > 0 (affine=False)
        # We handle this in _convert_state_dict
        pass

    return key


def _convert_state_dict(
    torch_model: TorchDiffusionTransformer,
    use_swiglu: bool,
    dim_cond_token: T.Optional[int],
) -> T.List[T.Tuple[str, mx.array]]:
    """Convert PyTorch state_dict to MLX weight pairs.

    Args:
        torch_model: PyTorch DiffusionTransformer.
        use_swiglu: Whether SwiGLU MLP is used.
        dim_cond_token: Conditioning token dimension (None if no conditioning).

    Returns:
        List of (name, mx.array) pairs for mlx_model.load_weights().
    """
    weight_pairs = []

    # Collect both parameters and buffers
    state = {}
    for name, param in torch_model.named_parameters():
        state[name] = param
    for name, buf in torch_model.named_buffers():
        state[name] = buf

    # FinalLayer norm_final has affine=False when dim_cond_feature > 0
    # In that case, norm_final.weight/bias don't exist in state_dict
    final_layer_has_affine = torch_model.final_layer.dim_cond_feature == 0

    for pt_key, pt_tensor in state.items():
        mlx_key = _remap_key(pt_key)
        if mlx_key is None:
            continue

        # Skip norm_final weight/bias if affine=False
        if not final_layer_has_affine and mlx_key in ("final_layer.norm_final.weight", "final_layer.norm_final.bias"):
            continue

        weight_pairs.append((mlx_key, torch_to_mx(pt_tensor)))

    return weight_pairs


def build_mlx_model(torch_model):
    """Build an MLX model matching the torch model's architecture.

    Dispatches by torch model type. Currently only ``DiffusionTransformer``
    (cross-attention conditioning) is supported in the public release.

    Args:
        torch_model: The PyTorch source model.

    Returns:
        Corresponding MLX model with weights loaded.
    """
    if isinstance(torch_model, TorchDiffusionTransformer):
        return build_mlx_dit(torch_model)
    raise NotImplementedError(f"No MLX builder for {type(torch_model).__name__}")
