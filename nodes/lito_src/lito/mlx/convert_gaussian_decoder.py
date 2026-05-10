# Copyright (C) 2026 Apple Inc. All rights reserved.
#
# PyTorch to MLX weight conversion for the GaussianDecoderXv.

import typing as T

from mlx import nn
import mlx.core as mx

from lito.mlx.convert import torch_to_mx
from lito.mlx.models.gaussian_decoder import (
    MLXGaussianDecoderXv,
    MLXOutputMLP,
    MLXSPointPerceiverEncoder,
    MLXSPointPerceiverEncoderBlock,
)
from lito.mlx.models.layers import FourierEmbed
from lito.models.point_decoder import GaussianDecoderXv


def _remap_sequential_key(key: str, prefix: str, num_mlp_layers: int) -> T.Optional[str]:
    """Remap an ``nn.Sequential``-indexed key to named-attribute form.

    PyTorch ``get_output_mlp`` produces an ``nn.Sequential`` with this layout::

        0: LayerNorm       -> norms.0
        1: SwiGLU/Mlp      -> mlps.0
        2: LayerNorm       -> norms.1   (if num_mlp_layers > 1)
        3: SwiGLU/Mlp      -> mlps.1   (if num_mlp_layers > 1)
        ...
        2*N: FinalLayer    -> final_layer

    Args:
        key: Full PyTorch state-dict key, e.g.
            ``"gs_output_shape_mlp.2.weight"``.
        prefix: The sequential prefix to match, e.g.
            ``"gs_output_shape_mlp"``.
        num_mlp_layers: Number of (LayerNorm, MLP) pairs before the
            ``FinalLayer``.

    Returns:
        Remapped key for the MLX model, or ``None`` if the key does not
        start with *prefix*.
    """
    if not key.startswith(prefix + "."):
        return None

    # Strip the prefix: e.g. "2.weight"
    suffix = key[len(prefix) + 1 :]
    # First token is the sequential index
    dot_pos = suffix.find(".")
    if dot_pos == -1:
        return None
    idx_str = suffix[:dot_pos]
    rest = suffix[dot_pos + 1 :]
    idx = int(idx_str)

    final_idx = 2 * num_mlp_layers
    if idx == final_idx:
        # FinalLayer
        return f"{prefix}.final_layer.{rest}"
    elif idx < final_idx and idx % 2 == 0:
        # LayerNorm
        return f"{prefix}.norms.{idx // 2}.{rest}"
    elif idx < final_idx and idx % 2 == 1:
        # MLP (SwiGLU or timm Mlp)
        return f"{prefix}.mlps.{idx // 2}.{rest}"
    else:
        # Should not happen with well-formed state dicts
        return None


def _detect_mlp_type(module: "torch.nn.Module") -> str:  # noqa: F821
    """Detect whether a PyTorch MLP module is ``"swiglu"`` or ``"timm"``.

    Args:
        module: A SwiGLU (has ``.w1`` or ``.w12``) or timm Mlp (has ``.fc1``) instance.

    Returns:
        ``"swiglu"`` or ``"timm"``.

    Raises:
        ValueError: If the module type cannot be determined.
    """
    # Pure-PyTorch SwiGLU has separate w1, w2, w3
    if hasattr(module, "w1"):
        return "swiglu"
    # xformers.ops.SwiGLU fuses w1+w2 into w12
    if hasattr(module, "w12"):
        return "swiglu"
    if hasattr(module, "fc1"):
        return "timm"
    raise ValueError(f"Cannot determine mlp_type from {type(module)}")


def _count_output_mlp_layers(sequential: "torch.nn.Sequential") -> int:  # noqa: F821
    """Count the number of (LayerNorm, MLP) pairs in a ``get_output_mlp`` Sequential.

    The layout is ``(LayerNorm, MLP) x N, FinalLayer``, so the total number
    of children is ``2*N + 1``.

    Args:
        sequential: A PyTorch ``nn.Sequential`` produced by ``get_output_mlp``.

    Returns:
        The number of MLP layers *N*.
    """
    return (len(sequential) - 1) // 2


def build_mlx_gaussian_decoder(torch_model: GaussianDecoderXv) -> MLXGaussianDecoderXv:
    """Build an MLX ``MLXGaussianDecoderXv`` from a PyTorch ``GaussianDecoderXv``.

    Extracts architecture parameters and weights from the PyTorch model,
    constructs the MLX equivalent, and loads the converted weights.

    Args:
        torch_model: A PyTorch ``lito.models.point_decoder.GaussianDecoderXv``
            instance.

    Returns:
        An ``MLXGaussianDecoderXv`` with the same weights.
    """
    # ------------------------------------------------------------------
    # 1. Extract architecture parameters from the PyTorch model
    # ------------------------------------------------------------------
    given_point_inputs: T.List[str] = torch_model.given_point_inputs
    perceiver_dim: int = torch_model.perceiver.dim_latent
    dim_latent: int = torch_model.dim_latent
    num_blocks: int = len(torch_model.perceiver.blocks)

    block0 = torch_model.perceiver.blocks[0]
    num_self_attn: int = len(block0.sa_layers)
    num_self_heads: int = block0.sa_layers[0].num_heads
    num_cross_heads: int = block0.ca_layer.num_heads
    cross_attn_type: str = block0.cross_attn_type
    self_attn_type: str = block0.self_attn_type
    use_rmsnorm: bool = hasattr(block0.ca_layer, "rmsnorm_q")
    add_bias: bool = block0.ca_layer.linear_q.bias is not None
    add_kv_linear: bool = block0.kv_linear is not None

    # MLP type and ratio (from the cross-attention MLP)
    mlp_type: str = _detect_mlp_type(block0.ca_mlp)
    if mlp_type == "swiglu":
        if hasattr(block0.ca_mlp, "w1"):
            # Pure-PyTorch SwiGLU: separate w1, w2, w3
            mlp_add_bias: bool = block0.ca_mlp.w1.bias is not None
            mlp_ratio: float = block0.ca_mlp.w1.weight.shape[0] / perceiver_dim
        else:
            # xformers SwiGLU: fused w12 (out_features = 2 * hidden), w3
            mlp_add_bias = block0.ca_mlp.w12.bias is not None
            mlp_ratio = block0.ca_mlp.w12.weight.shape[0] / 2 / perceiver_dim
    else:
        mlp_add_bias = block0.ca_mlp.fc1.bias is not None
        mlp_ratio = block0.ca_mlp.fc1.weight.shape[0] / perceiver_dim

    gs_expansion_ratio: int = torch_model.gs_expansion_ratio

    # point_mlp
    if torch_model.point_mlp is not None:
        num_given_mlp_layers: int = _count_output_mlp_layers(torch_model.point_mlp)
    else:
        num_given_mlp_layers = 0

    # xyz_encoding config
    xyz_enc = torch_model.xyz_encoding
    xyz_encoding_config = dict(
        dim_pos=xyz_enc.dim_pos,
        include_input=xyz_enc.include_input,
        min_freq_log2=xyz_enc.min_freq_log2,
        max_freq_log2=xyz_enc.max_freq_log2,
        num_freqs=xyz_enc.num_freqs,
        log_sampling=xyz_enc.log_sampling,
    )

    # Output MLP layer count and output dims
    gs_num_output_mlp_layers: int = _count_output_mlp_layers(torch_model.gs_output_shape_mlp)
    shape_final = torch_model.gs_output_shape_mlp[-1]  # FinalLayer
    color_final = torch_model.gs_output_color_mlp[-1]  # FinalLayer
    dim_shape_out: int = shape_final.linear.weight.shape[0]
    dim_color_out: int = color_final.linear.weight.shape[0]

    # ------------------------------------------------------------------
    # 2. Build the MLX model
    # ------------------------------------------------------------------
    xyz_encoding_mlx = FourierEmbed(**xyz_encoding_config)

    # Compute point_linear input dimension
    point_linear_dim_in = 0
    for name in given_point_inputs:
        if name == "xyz":
            point_linear_dim_in += 3
        elif name == "xyz_encoded":
            point_linear_dim_in += xyz_encoding_mlx.dim_out
        else:
            raise NotImplementedError(f"given_point_input={name}")

    point_linear = nn.Linear(point_linear_dim_in, perceiver_dim)

    # point_mlp
    point_mlp: T.Optional[MLXOutputMLP] = None
    if num_given_mlp_layers > 0:
        point_mlp = MLXOutputMLP(
            num_layers=num_given_mlp_layers,
            dim_in=perceiver_dim,
            dim_hidden=perceiver_dim,
            dim_out=perceiver_dim,
            mlp_type=mlp_type,
            mlp_add_bias=mlp_add_bias,
            contract=False,
        )

    # Perceiver blocks
    blocks: T.List[MLXSPointPerceiverEncoderBlock] = []
    for _ in range(num_blocks):
        blocks.append(
            MLXSPointPerceiverEncoderBlock(
                dim_latent=perceiver_dim,
                dim_token=dim_latent,
                dim_qkv=perceiver_dim,
                cross_attn_type=cross_attn_type,
                self_attn_type=self_attn_type,
                num_self_attn=num_self_attn,
                num_self_heads=num_self_heads,
                num_cross_heads=num_cross_heads,
                use_rmsnorm=use_rmsnorm,
                mlp_ratio=mlp_ratio,
                mlp_type=mlp_type,
                linear_in_attn_add_bias=add_bias,
                mlp_add_bias=mlp_add_bias,
                add_kv_linear=add_kv_linear,
            )
        )

    # Determine hidden dim for output MLPs.  The output MLPs from
    # get_output_mlp use contract=False, so dim_hidden = perceiver_dim
    # (the same as dim_in).
    perceiver_enc = MLXSPointPerceiverEncoder(blocks=blocks)

    gs_output_shape_mlp = MLXOutputMLP(
        num_layers=gs_num_output_mlp_layers,
        dim_in=perceiver_dim,
        dim_hidden=perceiver_dim,
        dim_out=dim_shape_out,
        mlp_type=mlp_type,
        mlp_add_bias=mlp_add_bias,
        contract=False,
    )
    gs_output_color_mlp = MLXOutputMLP(
        num_layers=gs_num_output_mlp_layers,
        dim_in=perceiver_dim,
        dim_hidden=perceiver_dim,
        dim_out=dim_color_out,
        mlp_type=mlp_type,
        mlp_add_bias=mlp_add_bias,
        contract=False,
    )

    mlx_model = MLXGaussianDecoderXv(
        xyz_encoding=xyz_encoding_mlx,
        point_linear=point_linear,
        point_mlp=point_mlp,
        perceiver=perceiver_enc,
        gs_output_shape_mlp=gs_output_shape_mlp,
        gs_output_color_mlp=gs_output_color_mlp,
        gs_expansion_ratio=gs_expansion_ratio,
        given_point_inputs=given_point_inputs,
    )

    # ------------------------------------------------------------------
    # 3. Convert state dict
    # ------------------------------------------------------------------
    weight_pairs = _convert_gaussian_decoder_state_dict(
        torch_model,
        num_given_mlp_layers=num_given_mlp_layers,
        gs_num_output_mlp_layers=gs_num_output_mlp_layers,
    )

    # ------------------------------------------------------------------
    # 4. Load weights
    # ------------------------------------------------------------------
    mlx_model.load_weights(weight_pairs)
    mx.eval(mlx_model.parameters())

    return mlx_model


# Prefixes that correspond to nn.Sequential output MLPs from get_output_mlp
_SEQUENTIAL_PREFIXES = ("gs_output_shape_mlp", "gs_output_color_mlp", "point_mlp")


def _convert_gaussian_decoder_state_dict(
    torch_model: GaussianDecoderXv,
    num_given_mlp_layers: int,
    gs_num_output_mlp_layers: int,
) -> T.List[T.Tuple[str, mx.array]]:
    """Convert a PyTorch GaussianDecoderXv state dict to MLX weight pairs.

    Handles the remapping of ``nn.Sequential`` indices in output MLPs to
    the named ``norms`` / ``mlps`` / ``final_layer`` attributes used by
    ``MLXOutputMLP``, while passing perceiver block keys through
    unchanged.

    Args:
        torch_model: PyTorch ``GaussianDecoderXv`` instance.
        num_given_mlp_layers: Number of MLP layers in the ``point_mlp``
            sequential (0 if ``point_mlp`` is None).
        gs_num_output_mlp_layers: Number of MLP layers in each of
            ``gs_output_shape_mlp`` and ``gs_output_color_mlp``.

    Returns:
        List of ``(name, mx.array)`` pairs for
        ``mlx_model.load_weights()``.
    """
    weight_pairs: T.List[T.Tuple[str, mx.array]] = []

    # Collect both parameters and buffers (freq_bands is a buffer)
    state: T.Dict[str, "torch.Tensor"] = {}  # noqa: F821
    for name, param in torch_model.named_parameters():
        state[name] = param
    for name, buf in torch_model.named_buffers():
        state[name] = buf

    # Build a mapping from sequential prefix to its num_mlp_layers
    seq_layer_counts = {
        "gs_output_shape_mlp": gs_num_output_mlp_layers,
        "gs_output_color_mlp": gs_num_output_mlp_layers,
    }
    if num_given_mlp_layers > 0:
        seq_layer_counts["point_mlp"] = num_given_mlp_layers

    for pt_key, pt_tensor in state.items():
        # Handle xformers fused w12 → split into w1 + w2
        if ".w12." in pt_key:
            suffix = pt_key.split(".w12.")[-1]  # "weight" or "bias"
            prefix = pt_key.split(".w12.")[0]

            # Remap the prefix if it's inside an nn.Sequential
            test_key_w1 = f"{prefix}.w1.{suffix}"
            remapped_w1 = _remap_gaussian_decoder_key(test_key_w1, seq_layer_counts)
            if remapped_w1 is None:
                # Try direct prefix mapping
                remapped_w1 = (
                    test_key_w1
                    if any(test_key_w1.startswith(dp) for dp in ("perceiver.", "point_linear.", "xyz_encoding."))
                    else None
                )
            if remapped_w1 is None:
                continue
            remapped_w2 = remapped_w1.replace(".w1.", ".w2.")

            # Split fused tensor: w12 has shape (2*hidden, in) for weight, (2*hidden,) for bias
            half = pt_tensor.shape[0] // 2
            w1_tensor = pt_tensor[:half]
            w2_tensor = pt_tensor[half:]
            weight_pairs.append((remapped_w1, torch_to_mx(w1_tensor)))
            weight_pairs.append((remapped_w2, torch_to_mx(w2_tensor)))
            continue

        mlx_key = _remap_gaussian_decoder_key(pt_key, seq_layer_counts)
        if mlx_key is None:
            continue
        weight_pairs.append((mlx_key, torch_to_mx(pt_tensor)))

    return weight_pairs


def _remap_gaussian_decoder_key(
    key: str,
    seq_layer_counts: T.Dict[str, int],
) -> T.Optional[str]:
    """Remap a single PyTorch state-dict key to the MLX weight name.

    Keys from perceiver blocks, ``point_linear``, and ``xyz_encoding``
    map 1:1.  Keys from ``nn.Sequential`` output MLPs need index-to-name
    remapping.

    Args:
        key: PyTorch state-dict key.
        seq_layer_counts: Mapping from sequential prefix (e.g.
            ``"gs_output_shape_mlp"``) to its number of (norm, mlp)
            pairs.

    Returns:
        The corresponding MLX key, or ``None`` if the key should be
        skipped (e.g. non-weight bookkeeping buffers).
    """
    # Try sequential remapping for output MLPs
    for prefix, num_layers in seq_layer_counts.items():
        remapped = _remap_sequential_key(key, prefix, num_layers)
        if remapped is not None:
            return remapped

    # Keys that map 1:1 without modification
    # perceiver.blocks.*, point_linear.*, xyz_encoding.freq_bands
    direct_prefixes = ("perceiver.", "point_linear.", "xyz_encoding.")
    for dp in direct_prefixes:
        if key.startswith(dp):
            return key

    # Unrecognized keys (bookkeeping tensors, etc.) are skipped
    return None
