"""LiToExportPLY node - save Gaussians to PLY file."""

import logging
import math
import os
from typing import Any

import torch
from comfy_api.latest import io
import folder_paths

log = logging.getLogger("comfyui-lito")


class LiToExportPLY(io.ComfyNode):
    """
    Export LiTo Gaussians to a PLY file.

    Saves the 3D Gaussian Splat representation as a standard PLY file
    compatible with Gaussian Splatting viewers.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LiToExportPLY",
            display_name="LiTo Export PLY",
            category="LiTo",
            description="Save 3D Gaussians to PLY file.",
            inputs=[
                io.Custom("LITO_GAUSSIAN").Input("gaussians", tooltip="Gaussians from LiToImageTo3D"),
                io.String.Input(
                    "filename",
                    default="lito_output",
                    tooltip="Output filename (without extension)",
                ),
            ],
            outputs=[
                io.String.Output(display_name="ply_filepath", tooltip="Path to saved PLY file"),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(cls, gaussians: Any, filename: str = "lito_output"):
        from plibs import gs_utils, sh_utils

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        # Ensure unique filename
        ply_path = os.path.join(output_dir, f"{filename}.ply")
        counter = 1
        while os.path.exists(ply_path):
            ply_path = os.path.join(output_dir, f"{filename}_{counter:04d}.ply")
            counter += 1

        # Reshape Gaussians to flat (N, ...) format
        ngs = math.prod(gaussians["xyz_w"].shape[:-1])
        _sh_degree = sh_utils.get_sh_degree_from_total_dim(gaussians["rgb_sh"].size(-2))

        gs = gs_utils.Gaussians(
            sh_degree=_sh_degree,
            xyz_w=gaussians["xyz_w"].reshape(ngs, 3),
            rgb_sh=gaussians["rgb_sh"].reshape(ngs, -1, 3),
            rgb_sh_dc=None,
            rgb_sh_rest=None,
            scaling_logit=None,
            quaternion_prenorm=None,
            opacity_logit=None,
            scaling=gaussians["scaling"].reshape(ngs, 3),
            quaternion=gaussians["quaternion"].reshape(ngs, 4),
            opacity=gaussians["opacity"].reshape(ngs, 1),
            min_scaling=0,
            scaling_activation_type="none",
        )
        gs.save_ply(filename=ply_path)

        log.info("Saved PLY: %s (%d Gaussians)", ply_path, ngs)
        return io.NodeOutput(ply_path)
