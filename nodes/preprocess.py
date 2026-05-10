"""LiToPreprocess node - background removal + crop, outputs IMAGE + MASK."""

import logging

import numpy as np
from PIL import Image, ImageOps
import torch
from comfy_api.latest import io

log = logging.getLogger("comfyui-lito")

IMG_RESOLUTION = 518


class LiToPreprocess(io.ComfyNode):
    """
    Preprocess an image for LiTo image-to-3D generation.

    Removes background (optional), crops/pads to center the object,
    and resizes to 518x518. Outputs a standard ComfyUI IMAGE + MASK pair
    so downstream nodes can mix and match (you can also skip this node and
    pass an IMAGE + MASK from any other source directly to LiToImageTo3D).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LiToPreprocess",
            display_name="LiTo Preprocess Image",
            category="LiTo",
            description="Background removal + crop + resize to 518x518. Outputs IMAGE + MASK.",
            inputs=[
                io.Image.Input("image", tooltip="Input image (RGB or RGBA)"),
                io.Boolean.Input(
                    "remove_bg",
                    default=True,
                    tooltip="Remove background using rembg",
                ),
                io.Boolean.Input(
                    "crop",
                    default=True,
                    tooltip="Crop and center the object in frame",
                ),
                io.Float.Input(
                    "fill_ratio",
                    default=0.8,
                    min=0.3,
                    max=1.0,
                    step=0.05,
                    tooltip="Target ratio of object size to canvas size",
                ),
                io.Boolean.Input(
                    "keep_optical_axis",
                    default=True,
                    tooltip="Keep the optical axis centered (better 3D alignment)",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image", tooltip="Cropped/resized RGB image (518x518)"),
                io.Mask.Output(display_name="mask", tooltip="Foreground alpha mask (518x518)"),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        image: torch.Tensor,
        remove_bg: bool = True,
        crop: bool = True,
        fill_ratio: float = 0.8,
        keep_optical_axis: bool = True,
    ):
        # comfy-env's worker replaces builtins.print with a wrapper named
        # "_forwarded_print"; numba (pulled in via rembg -> pymatting) does
        # getattr(__main__, print.__name__) at import time and crashes if
        # __main__ doesn't carry that attr. Register it before importing rembg.
        import sys, builtins
        _main = sys.modules.get("__main__")
        if _main is not None and not hasattr(_main, builtins.print.__name__):
            setattr(_main, builtins.print.__name__, builtins.print)

        import rembg
        from lito.eval_scripts import st_paper_utils

        # ComfyUI images are (B, H, W, C) float [0,1] RGB
        # Take first image from batch
        img_tensor = image[0]  # (H, W, C)
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)

        # Convert to PIL
        if img_np.shape[-1] == 4:
            pil_image = Image.fromarray(img_np, "RGBA")
        else:
            pil_image = Image.fromarray(img_np, "RGB")

        # Apply EXIF orientation
        pil_image = ImageOps.exif_transpose(pil_image)

        # Background removal
        if remove_bg:
            has_alpha = False
            if pil_image.mode == "RGBA":
                alpha_in = np.array(pil_image)[:, :, 3]
                if not np.all(alpha_in == 255):
                    has_alpha = True

            if has_alpha:
                output = pil_image
            else:
                input_rgb = pil_image.convert("RGB")
                output = rembg.remove(input_rgb)

            output_np = np.array(output)  # (h, w, 4) uint8
            alpha = output_np[:, :, 3]
        else:
            if pil_image.mode != "RGBA":
                pil_image = pil_image.convert("RGBA")
            output_np = np.array(pil_image)
            output_np[:, :, 3] = 255
            alpha = output_np[:, :, 3]

        # Crop and pad
        if crop:
            cdict = st_paper_utils.determine_crop_and_pad(
                alpha=torch.from_numpy(alpha).float() / 255.0,
                keep_optical_axis=keep_optical_axis,
                fill_ratio=fill_ratio,
                th_alpha=0.8,
                pad_x_ratio=0.5,
                pad_y_ratio=0.5,
            )
            crop_x1 = cdict["crop_x1"]
            crop_y1 = cdict["crop_y1"]
            crop_x2 = cdict["crop_x2"]
            crop_y2 = cdict["crop_y2"]
            pad_left = cdict["pad_left"]
            pad_right = cdict["pad_right"]
            pad_top = cdict["pad_top"]
            pad_bottom = cdict["pad_bottom"]

            rgba = torch.from_numpy(output_np)
            rgba = rgba[crop_y1:crop_y2, crop_x1:crop_x2].clone()

            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                rgba = torch.nn.functional.pad(
                    rgba,
                    (0, 0, pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=0,
                )

            output_np = rgba.detach().cpu().numpy().astype(np.uint8)
        else:
            h, w = output_np.shape[:2]
            if remove_bg:
                max_dim = max(h, w)
                pad_h = max_dim - h
                pad_w = max_dim - w
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                output_np = np.pad(
                    output_np,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
            else:
                min_dim = min(h, w)
                start_y = (h - min_dim) // 2
                start_x = (w - min_dim) // 2
                output_np = output_np[start_y:start_y + min_dim, start_x:start_x + min_dim]

        # Resize to target resolution (LANCZOS works on RGBA in one pass)
        output = Image.fromarray(output_np.astype(np.uint8), mode="RGBA")
        output = output.resize((IMG_RESOLUTION, IMG_RESOLUTION), Image.Resampling.LANCZOS)
        output_float = np.array(output).astype(np.float32) / 255.0  # (H, W, 4) [0, 1]

        # ComfyUI conventions:
        # - IMAGE: (B, H, W, 3) float [0,1] RGB
        # - MASK:  (B, H, W)    float [0,1]  (1 = foreground)
        image_out = torch.from_numpy(output_float[:, :, :3]).unsqueeze(0).float()
        mask_out = torch.from_numpy(output_float[:, :, 3]).unsqueeze(0).float()

        return io.NodeOutput(image_out, mask_out)
