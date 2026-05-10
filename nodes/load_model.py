"""LiToLoadModel node - downloads checkpoint and loads model."""

import logging
import os
from pathlib import Path

import torch
import comfy.model_management as mm
from comfy_api.latest import io

log = logging.getLogger("comfyui-lito")

# Apple CDN checkpoint URLs
CHECKPOINT_URLS = {
    "lito_dit_rgba (recommended)": "https://ml-site.cdn-apple.com/models/lito/lito_dit_rgba.ckpt",
    "lito_dit (paper)": "https://ml-site.cdn-apple.com/models/lito/lito_dit.ckpt",
}

# Expected approximate file sizes for verification
EXPECTED_SIZES = {
    "lito_dit_rgba.ckpt": 3_000_000_000,  # ~3GB estimated
    "lito_dit.ckpt": 3_000_000_000,
}


def _comfy_tqdm():
    """tqdm that shows download progress in ComfyUI's UI."""
    try:
        import comfy.utils
        import tqdm as _tqdm_mod
    except ImportError:
        return None
    holder = {"pbar": None, "total": 0, "done": 0}

    class _T(_tqdm_mod.tqdm):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if self.total and self.total > 0 and holder["pbar"] is None:
                holder["total"] = self.total
                holder["done"] = 0
                holder["pbar"] = comfy.utils.ProgressBar(self.total)

        def update(self, n=1):
            ret = super().update(n)
            if n and holder["pbar"] and holder["total"] > 0:
                holder["done"] = min(holder["done"] + n, holder["total"])
                holder["pbar"].update_absolute(holder["done"], holder["total"])
            return ret

    return _T


try:
    from .comfy_utils import get_lito_models_path
except ImportError:
    from comfy_utils import get_lito_models_path


class LiToLoadModel(io.ComfyNode):
    """
    Load LiTo image-to-3D model.

    Downloads checkpoint from Apple CDN if needed and loads the generative model
    and tokenizer. Actual model loading happens in the isolated subprocess.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LiToLoadModel",
            display_name="(Down)Load LiTo Model",
            category="LiTo",
            description="Load LiTo image-to-3D model. Downloads checkpoint (~3GB) on first use.",
            inputs=[
                io.Combo.Input(
                    "checkpoint",
                    options=list(CHECKPOINT_URLS.keys()),
                    default="lito_dit_rgba (recommended)",
                    tooltip="Model checkpoint to use. 'recommended' includes bug fixes over paper version.",
                ),
                io.Boolean.Input(
                    "compile",
                    default=False,
                    tooltip="Enable torch.compile for faster inference (slow first run, then ~4.7s on H100).",
                ),
                io.Combo.Input(
                    "precision",
                    options=["auto", "bf16", "fp16", "fp32"],
                    default="auto",
                    tooltip="Model precision. auto: bf16 on Ampere+, fp16 on older GPUs.",
                ),
            ],
            outputs=[
                io.Custom("LITO_MODEL").Output(
                    display_name="model",
                    tooltip="LiTo model (generative DiT + tokenizer)",
                ),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(cls, checkpoint: str, compile: bool, precision: str = "auto"):
        log.info("Loading LiTo model...")

        # Resolve precision
        if precision == "auto":
            device = mm.get_torch_device()
            if mm.should_use_bf16(device):
                precision = "bf16"
            elif mm.should_use_fp16(device):
                precision = "fp16"
            else:
                precision = "fp32"
        log.info("Precision: %s", precision)

        # Get checkpoint URL and download if needed
        url = CHECKPOINT_URLS[checkpoint]
        models_dir = get_lito_models_path()
        checkpoint_path = cls._get_or_download_checkpoint(url, models_dir)

        log.info("LiTo model checkpoint ready: %s", checkpoint_path)

        model_config = {
            "checkpoint_path": str(checkpoint_path),
            "compile": compile,
            "precision": precision,
        }
        return io.NodeOutput(model_config)

    @staticmethod
    def _get_or_download_checkpoint(url: str, models_dir: Path) -> Path:
        """Download checkpoint if not already cached."""
        from lito.eval_scripts.st_model_utils import download_checkpoint

        filename = os.path.basename(url)
        local_path = models_dir / filename

        if local_path.exists():
            log.info("Using cached checkpoint: %s", local_path)
            return local_path

        log.info("Downloading checkpoint from %s ...", url)
        # Use vendored download function (streams with tqdm)
        result_path = download_checkpoint(
            url=url,
            download_dir_root=str(models_dir),
            overwrite=False,
        )
        return Path(result_path)
