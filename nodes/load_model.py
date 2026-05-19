"""LiToLoadModel node - downloads checkpoint and loads model."""

import logging
import os
import threading
import time
from pathlib import Path

import torch
from comfy_api.latest import io

# `comfy.model_management` and `comfy.utils` are imported lazily inside the
# methods that use them — see same pattern in lito_src/plibs/linalg_utils.py.
# Top-level import would trigger comfy.model_management's module-load CUDA
# probe, which crashes on CPU-only environments (e.g. the Windows mock-CUDA
# CI runner).

log = logging.getLogger("comfyui-lito")

# Apple CDN checkpoint URLs
CHECKPOINT_URLS = {
    "lito_dit_rgba (recommended)": "https://ml-site.cdn-apple.com/models/lito/lito_dit_rgba.ckpt",
    "lito_dit (paper)": "https://ml-site.cdn-apple.com/models/lito/lito_dit.ckpt",
}


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
            description="Load LiTo image-to-3D model. Downloads checkpoint (~7GB) on first use.",
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
            import comfy.model_management as mm
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
        """Fast parallel download via hf_transfer (Rust, HTTP Range parallelism).

        Apple's CDN advertises accept-ranges: bytes; hf_transfer fans out 16
        concurrent range requests, typically saturating a 1 Gbps NIC. Downloads
        to <name>.partial then renames on success so a Ctrl+C never leaves a
        truncated file at the cached path.
        """
        import hf_transfer
        import requests

        filename = os.path.basename(url)
        local_path = models_dir / filename

        if local_path.exists():
            log.info("Using cached checkpoint: %s", local_path)
            return local_path

        # Probe size for the progress bar
        head = requests.head(url, allow_redirects=True, timeout=30)
        head.raise_for_status()
        total_size = int(head.headers["content-length"])

        partial = local_path.with_suffix(local_path.suffix + ".partial")
        if partial.exists():
            partial.unlink()  # hf_transfer doesn't resume; start clean

        import comfy.utils
        pbar = comfy.utils.ProgressBar(total_size)
        done = 0
        lock = threading.Lock()

        def on_chunk(chunk_size: int) -> None:
            nonlocal done
            with lock:
                done += chunk_size
                pbar.update_absolute(done, total_size)

        log.info(
            "Downloading %s (%.2f GB) via hf_transfer (16 parallel ranges)...",
            url, total_size / 1e9,
        )
        t0 = time.time()
        hf_transfer.download(
            url=url,
            filename=str(partial),
            max_files=16,
            chunk_size=10 * 1024 * 1024,
            parallel_failures=3,
            max_retries=5,
            callback=on_chunk,
        )
        elapsed = time.time() - t0
        log.info(
            "Downloaded %.2f GB in %.1fs (%.1f MB/s) -> %s",
            total_size / 1e9, elapsed, total_size / elapsed / 1e6, local_path,
        )

        os.rename(partial, local_path)
        return local_path
