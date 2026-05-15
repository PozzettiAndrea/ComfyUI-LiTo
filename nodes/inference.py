"""LiToImageTo3D node - DiT sampling + Gaussian decoding."""

import contextlib
import logging
import time
from typing import Any

import torch
import torch.nn.functional as F
from comfy_api.latest import io

# `comfy.model_management` and `comfy.utils` are imported lazily inside the
# methods that use them. See the same lazy-import idiom in lito_src/plibs/.
# Top-level `import comfy.model_management` triggers its CUDA probe at module
# load and crashes on CPU-only torch (e.g. the Windows mock-CUDA CI runner).

log = logging.getLogger("comfyui-lito")


def _say(stage: str, msg: str) -> None:
    """Print + log a stage marker. print() ensures it shows up in the worker log
    regardless of the logger config; log.info() keeps things tidy when run
    outside the comfy-env worker pipe."""
    line = f"[LiTo] {stage}: {msg}"
    print(line, flush=True)
    log.info(line)


def _peak_vram_gb() -> float:
    """Return CUDA peak allocation since the last reset, in GiB. 0.0 on
    non-CUDA devices so the log line stays printable."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def _reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@contextlib.contextmanager
def _progress_through_tqdm(total: int, label: str):
    """Patch tqdm.tqdm so each iteration also bumps a comfy ProgressBar (which
    powers the ComfyUI node progress UI) and emits a one-line worker-log update
    every ~10% with elapsed + eta."""
    import tqdm as _tqdm_mod
    try:
        import lito.odelibs.ode_solvers as _ode_mod
    except Exception:
        _ode_mod = None

    import comfy.utils
    pbar = comfy.utils.ProgressBar(total)
    orig_tqdm = _tqdm_mod.tqdm
    orig_ode_tqdm = getattr(_ode_mod, "tqdm", None) if _ode_mod is not None else None
    start = time.time()

    class _PatchedTqdm:
        def __init__(self_inner, iterable=None, *a, **kw):
            self_inner._wrapped = orig_tqdm(iterable, *a, **kw)
            self_inner._step = 0
            self_inner._n_total = getattr(self_inner._wrapped, "total", total) or total
        def __iter__(self_inner):
            milestone = max(1, self_inner._n_total // 10)
            for item in self_inner._wrapped:
                yield item
                self_inner._step += 1
                pbar.update(1)
                if self_inner._step % milestone == 0 or self_inner._step == self_inner._n_total:
                    elapsed = time.time() - start
                    rate = self_inner._step / elapsed if elapsed > 0 else 0
                    eta = (self_inner._n_total - self_inner._step) / rate if rate > 0 else 0
                    _say(label, f"{self_inner._step}/{self_inner._n_total} "
                                f"({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)")
        def __getattr__(self_inner, name):
            return getattr(self_inner._wrapped, name)
        def __enter__(self_inner):
            self_inner._wrapped.__enter__()
            return self_inner
        def __exit__(self_inner, *exc):
            return self_inner._wrapped.__exit__(*exc)

    _tqdm_mod.tqdm = _PatchedTqdm
    if _ode_mod is not None and orig_ode_tqdm is not None:
        _ode_mod.tqdm = _PatchedTqdm
    try:
        yield pbar
    finally:
        _tqdm_mod.tqdm = orig_tqdm
        if _ode_mod is not None and orig_ode_tqdm is not None:
            _ode_mod.tqdm = orig_ode_tqdm

IMG_RESOLUTION = 518


def _compose_cond_rgba(image: torch.Tensor, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Combine ComfyUI IMAGE + MASK into LiTo's (1, 1, H, W, 4rgba) conditioning tensor.

    LiTo expects 518x518 RGBA with straight (not premultiplied) alpha in [0, 1].
    """
    # IMAGE: (B, H, W, C) [0,1]. Take first; force 3-channel - the MASK input
    # is the authoritative alpha, so drop any alpha that came in on the IMAGE.
    rgb = image[0, ..., :3]  # (H, W, 3)
    # MASK: (B, H, W) [0,1]. Take first.
    if mask.ndim == 4:
        # Some upstream nodes emit (B, H, W, 1)
        alpha = mask[0, ..., 0]
    else:
        alpha = mask[0]  # (H, W)
    alpha = alpha.clamp(0.0, 1.0)

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


def _load_and_cache_model(checkpoint_path: str, compile: bool, device: torch.device, dtype: torch.dtype):
    """Load model from checkpoint, cache for reuse.

    Wraps the loaded module in `comfy.model_patcher.ModelPatcher` so
    ComfyUI's memory manager is aware of our ~3 GB DiT+decoder and can
    evict competing models to make room (and evict OURS for downstream
    nodes). Now that LiToDiTTrainer is a plain nn.Module (Lightning
    removed), the ModelPatcher's `self.model.device = ...` works.

    Note: `st_model = model.pretrained_tokenizer` is a submodule of the
    DiT trainer, so one patcher covers both — they share physical weights.
    """
    cache_key = (checkpoint_path, compile, dtype)
    if cache_key in _model_cache:
        log.info("Using cached model")
        return _model_cache[cache_key]

    log.info("Loading model from %s...", checkpoint_path)
    from lito.eval_scripts.st_model_utils import load_model
    import comfy.model_management as mm
    from comfy.model_patcher import ModelPatcher

    mdict = load_model(
        checkpoint_url=checkpoint_path,
        download_dir_root="",  # Already local
        overwrite=False,
        dtype=dtype,
        device=device,
        load_params=True,
    )
    model = mdict["model"]
    model.to(device=device, dtype=dtype)
    model.eval()
    model.freeze()

    if compile:
        log.info("Compiling model with torch.compile (this may take a few minutes on first run)...")
        model = torch.compile(model)

    st_model = model.pretrained_tokenizer

    patcher = ModelPatcher(
        model,
        load_device=device,
        offload_device=mm.unet_offload_device(),
    )

    result = {"model": model, "st_model": st_model, "patcher": patcher}
    _model_cache[cache_key] = result
    log.info(
        "Model loaded successfully (size %.2f GB on %s, offload device %s)",
        patcher.model_size() / (1024 ** 3), device, mm.unet_offload_device(),
    )
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
        import comfy.model_management as mm
        device = mm.get_torch_device()
        precision = model["precision"]
        dtype = _get_dtype(precision)

        t_total = time.time()
        _say("start", f"device={device}, dtype={precision}, steps={sampling_steps}, "
                     f"method={sampling_method}, cfg={cfg_scale}, seed={seed}")

        # -- load --------------------------------------------------------------
        _reset_peak_vram()
        t0 = time.time()
        _say("load", f"checkpoint={model['checkpoint_path']}, compile={model['compile']}")
        models = _load_and_cache_model(
            model["checkpoint_path"],
            model["compile"],
            device,
            dtype,
        )
        dit_model = models["model"]
        st_model = models["st_model"]
        patcher = models["patcher"]
        # Hand ourselves to Comfy's memory manager. It will evict competing
        # workflow models to make room; on subsequent runs this is a no-op
        # if the model is still resident. 1.3x covers DiT activations.
        mm.load_models_gpu(
            [patcher],
            memory_required=int(patcher.model_size() * 1.3),
        )
        _say("load", f"ok in {time.time() - t0:.1f}s, peak VRAM {_peak_vram_gb():.2f} GB")

        # -- prep --------------------------------------------------------------
        torch.manual_seed(seed)
        cond_rgba = _compose_cond_rgba(image, mask, device).to(dtype=dtype)

        # -- sample latent tokens (DiT) ----------------------------------------
        # Heun does 2 NFE per outer step but the tqdm in ode_solvers wraps the
        # outer loop, so sampling_steps is the right total.
        _reset_peak_vram()
        _say("sample", f"DiT, {sampling_steps} {sampling_method} steps")
        t0 = time.time()
        with _progress_through_tqdm(sampling_steps, label="sample"):
            out_dict = dit_model.inference_sample_latent(
                cond_rgba=cond_rgba,
                ode_sampling_method=sampling_method,
                ode_num_steps=sampling_steps,
                cfg_scale=cfg_scale,
                use_ema=True,
            )
        t_sample = time.time() - t0
        _say("sample", f"done in {t_sample:.1f}s, peak VRAM {_peak_vram_gb():.2f} GB")

        # Drop DiT activations + fragmentation before the (memory-hungry)
        # decoder pass. Without this, allocator holds onto blocks even
        # after Python refs go away, inflating peak for the decode stage.
        mm.soft_empty_cache()

        # -- decode latents to Gaussians ---------------------------------------
        init_coord_src = "voxel_decoder" if st_model.voxel_decoder is not None else "sample_xyz"
        decode_steps = 50  # used only when init_coord_src == "sample_xyz"
        _reset_peak_vram()
        _say("decode", f"Gaussians via {init_coord_src}"
                       + (f" ({decode_steps} init steps)" if init_coord_src == "sample_xyz" else ""))
        t0 = time.time()
        decode_total = decode_steps if init_coord_src == "sample_xyz" else 1
        with _progress_through_tqdm(decode_total, label="decode"):
            gs_dicts = st_model.inference_estimate_gaussians(
                fpoint_latent=out_dict["unnormalized_latent"],
                init_coord_src=init_coord_src,
                steps_for_sample_xyz=decode_steps,
            )
        gs_dict = gs_dicts[0]
        t_decode = time.time() - t0
        num_gauss = gs_dict["xyz_w"].shape[0] if gs_dict["xyz_w"].dim() == 2 else gs_dict["xyz_w"].numel() // 3
        _say("decode", f"done in {t_decode:.1f}s, {num_gauss} Gaussians, "
                       f"peak VRAM {_peak_vram_gb():.2f} GB")

        # -- pack outputs (move to CPU to free VRAM) ---------------------------
        gs_output = {
            "xyz_w": gs_dict["xyz_w"].cpu(),
            "rgb_sh": gs_dict["rgb_sh"].cpu(),
            "scaling": gs_dict["scaling"].cpu(),
            "quaternion": gs_dict["quaternion"].cpu(),
            "opacity": gs_dict["opacity"].cpu(),
        }
        # Final cache flush so VRAM reported to user / downstream nodes
        # reflects only the resident model, not lingering decode tensors.
        mm.soft_empty_cache()
        _say("done", f"total {time.time() - t_total:.1f}s "
                    f"(load + sample {t_sample:.1f}s + decode {t_decode:.1f}s)")
        return io.NodeOutput(gs_output)
