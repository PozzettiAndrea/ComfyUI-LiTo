"""LiToImageTo3D node - DiT sampling + Gaussian decoding."""

import logging
import time
from typing import Any

import torch
import torch.nn.functional as F
import comfy.model_management as mm
from comfy_api.latest import io

log = logging.getLogger("comfyui-lito")

IMG_RESOLUTION = 518


def _compose_cond_rgba(image: torch.Tensor, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Combine ComfyUI IMAGE + MASK into LiTo's (1, 1, H, W, 4rgba) conditioning tensor.

    LiTo expects 518x518 RGBA with straight (not premultiplied) alpha in [0, 1].
    """
    # IMAGE: (B, H, W, 3) [0,1] RGB. Take first.
    rgb = image[0]  # (H, W, 3)
    # MASK: (B, H, W) [0,1]. Take first.
    if mask.ndim == 4:
        # Some upstream nodes emit (B, H, W, 1)
        alpha = mask[0, ..., 0]
    else:
        alpha = mask[0]  # (H, W)

    H, W = rgb.shape[:2]
    if alpha.shape != (H, W):
        # Resize mask to match image
        alpha = F.interpolate(
            alpha.unsqueeze(0).unsqueeze(0).float(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    rgba = torch.cat([rgb, alpha.unsqueeze(-1)], dim=-1)  # (H, W, 4)

    # Resize to LiTo's expected 518x518
    if H != IMG_RESOLUTION or W != IMG_RESOLUTION:
        rgba = F.interpolate(
            rgba.permute(2, 0, 1).unsqueeze(0),  # (1, 4, H, W)
            size=(IMG_RESOLUTION, IMG_RESOLUTION),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0].permute(1, 2, 0)  # (518, 518, 4)

    rgba = rgba.clamp(0.0, 1.0).to(device=device).float()
    return rgba.unsqueeze(0).unsqueeze(0)  # (1, 1, 518, 518, 4)

# Cache for loaded models to avoid reloading on each run
_model_cache = {}


def _get_dtype(precision: str) -> torch.dtype:
    """Map precision string to torch dtype."""
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[precision]


def _load_and_cache_model(checkpoint_path: str, compile: bool, device: torch.device):
    """Load model from checkpoint, cache for reuse."""
    cache_key = (checkpoint_path, compile)
    if cache_key in _model_cache:
        log.info("Using cached model")
        return _model_cache[cache_key]

    log.info("Loading model from %s...", checkpoint_path)
    from lito.eval_scripts.st_model_utils import load_model

    mdict = load_model(
        checkpoint_url=checkpoint_path,
        download_dir_root="",  # Already local
        overwrite=False,
        dtype=torch.float,
        device=device,
        load_params=True,
    )
    model = mdict["model"]
    model.to(device=device)
    model.eval()
    model.freeze()

    if compile:
        log.info("Compiling model with torch.compile (this may take a few minutes on first run)...")
        model = torch.compile(model)

    st_model = model.pretrained_tokenizer

    result = {"model": model, "st_model": st_model}
    _model_cache[cache_key] = result
    log.info("Model loaded successfully")
    return result


class LiToImageTo3D(io.ComfyNode):
    """
    Generate 3D Gaussian Splats from a preprocessed image.

    Runs the LiTo DiT flow-matching model to sample latent tokens,
    then decodes them into 3D Gaussians (~524K Gaussians).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LiToImageTo3D",
            display_name="LiTo Image to 3D",
            category="LiTo",
            description="Generate 3D Gaussians from image. ~4.7s on H100 (compiled), ~15s uncompiled.",
            inputs=[
                io.Custom("LITO_MODEL").Input("model", tooltip="Model from LiToLoadModel"),
                io.Image.Input("image", tooltip="Input image (will be resized to 518x518)"),
                io.Mask.Input("mask", tooltip="Foreground mask (1=object, 0=background)"),
                io.Int.Input(
                    "sampling_steps",
                    default=20,
                    min=5,
                    max=100,
                    step=1,
                    tooltip="Number of ODE sampling steps (more = better quality, slower)",
                ),
                io.Float.Input(
                    "cfg_scale",
                    default=3.0,
                    min=1.0,
                    max=10.0,
                    step=0.5,
                    tooltip="Classifier-free guidance scale",
                ),
                io.Combo.Input(
                    "sampling_method",
                    options=["heun", "euler"],
                    default="heun",
                    tooltip="ODE solver. Heun is higher quality (2x NFE), Euler is faster.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2**31 - 1,
                    tooltip="Random seed for reproducibility",
                ),
            ],
            outputs=[
                io.Custom("LITO_GAUSSIAN").Output(
                    display_name="gaussians",
                    tooltip="3D Gaussian dict (xyz, quaternion, scale, opacity, SH coefficients)",
                ),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        model: Any,
        image: torch.Tensor,
        mask: torch.Tensor,
        sampling_steps: int = 20,
        cfg_scale: float = 3.0,
        sampling_method: str = "heun",
        seed: int = 0,
    ):
        device = mm.get_torch_device()
        precision = model["precision"]
        dtype = _get_dtype(precision)

        # Load model into GPU
        models = _load_and_cache_model(
            model["checkpoint_path"],
            model["compile"],
            device,
        )
        dit_model = models["model"]
        st_model = models["st_model"]

        # Set seed for reproducibility
        torch.manual_seed(seed)

        # Compose IMAGE + MASK into LiTo's (1, 1, 518, 518, 4rgba) conditioning tensor
        cond_rgba = _compose_cond_rgba(image, mask, device)

        # Step 1: Sample latent tokens via DiT
        log.info("Sampling latent tokens (%d steps, %s, cfg=%.1f)...", sampling_steps, sampling_method, cfg_scale)
        t0 = time.time()

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            out_dict = dit_model.inference_sample_latent(
                cond_rgba=cond_rgba,
                ode_sampling_method=sampling_method,
                ode_num_steps=sampling_steps,
                cfg_scale=cfg_scale,
                use_ema=True,
            )

        t_sample = time.time() - t0
        log.info("Sampling done in %.1fs", t_sample)

        # Step 2: Decode latents to 3D Gaussians
        log.info("Decoding Gaussians...")
        t0 = time.time()

        # Determine init coordinate source
        if st_model.voxel_decoder is not None:
            init_coord_src = "voxel_decoder"
        else:
            init_coord_src = "sample_xyz"

        with torch.autocast(device_type="cuda", enabled=True):
            gs_dicts = st_model.inference_estimate_gaussians(
                fpoint_latent=out_dict["unnormalized_latent"],
                init_coord_src=init_coord_src,
                steps_for_sample_xyz=50,
            )
            gs_dict = gs_dicts[0]

        t_decode = time.time() - t0
        log.info("Decoding done in %.1fs (total: %.1fs)", t_decode, t_sample + t_decode)

        # Move to CPU to free VRAM
        gs_output = {
            "xyz_w": gs_dict["xyz_w"].cpu(),
            "rgb_sh": gs_dict["rgb_sh"].cpu(),
            "scaling": gs_dict["scaling"].cpu(),
            "quaternion": gs_dict["quaternion"].cpu(),
            "opacity": gs_dict["opacity"].cpu(),
        }

        return io.NodeOutput(gs_output)
