"""ComfyUI-LiTo Prestartup Script."""

from pathlib import Path
from comfy_env import setup_env, copy_files
from comfy_3d_viewers import copy_viewer

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy Gaussian splat viewer for 3D preview
copy_viewer("pointcloud_vtk", SCRIPT_DIR / "web")

# Copy example assets
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input")
