"""LiToPreviewPointCloud node - 3D point cloud preview via VTK viewer."""

import logging
import os

from comfy_api.latest import io

log = logging.getLogger("comfyui-lito")


class LiToPreviewPointCloud(io.ComfyNode):
    """
    Preview a PLY point cloud / Gaussian splat in the ComfyUI viewer.

    Uses the VTK.js-based viewer for interactive 3D visualization.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LiToPreviewPointCloud",
            display_name="LiTo Preview Point Cloud",
            category="LiTo",
            description="Preview 3D Gaussian PLY in the browser.",
            is_output_node=True,
            inputs=[
                io.String.Input("file_path", force_input=True, tooltip="Path to PLY file from LiToExportPLY"),
            ],
            outputs=[],
        )

    @classmethod
    def fingerprint_inputs(cls, file_path: str):
        """Ensure cache invalidation when file changes."""
        if file_path and os.path.exists(file_path):
            stat = os.stat(file_path)
            return {"file_path": file_path, "mtime": stat.st_mtime, "size": stat.st_size}
        return {"file_path": file_path}

    @classmethod
    def execute(cls, file_path: str):
        if not file_path or not os.path.exists(file_path):
            log.warning("PLY file not found: %s", file_path)
            return {"ui": {"file_path": [""]}}

        log.info("Preview: %s", file_path)
        return {"ui": {"file_path": [file_path]}}
