"""Utility functions for ComfyUI-LiTo nodes."""

import os
from pathlib import Path
import folder_paths

# Register model folder with ComfyUI's folder_paths system
_lito_models_dir = os.path.join(folder_paths.models_dir, "lito")
os.makedirs(_lito_models_dir, exist_ok=True)
folder_paths.add_model_folder_path("lito", _lito_models_dir)


def get_lito_models_path() -> Path:
    """Get the path to LiTo models directory within ComfyUI models folder."""
    models_dir = Path(folder_paths.models_dir) / "lito"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir
