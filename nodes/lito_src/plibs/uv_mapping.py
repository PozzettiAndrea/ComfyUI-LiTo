#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class UVMap:
    def __init__(
        self,
        texture: np.ndarray,
        mode: str = "wrap",
    ):
        """
        Args:
            texture:
                (h, w, dim)  for example, an rgb image, a displacement map, a bump map, etc
            mode:
                'wrap': used when 1 <= uv or uv <= 0.
                'edge': used when no wrapping is needed
        """
        self.texture = texture
        self.texture_height = self.texture.shape[0]
        self.texture_width = self.texture.shape[1]
        self.mode = mode

        # handle padding
        pad_widths = [[0, 0]] * self.texture.ndim
        pad_widths[0] = [1, 1]
        pad_widths[1] = [1, 1]
        padded_texture = np.pad(self.texture, pad_width=pad_widths, mode=mode)

        # create interpolator for the texture
        self.grid_ys = np.linspace(-1, self.texture_height, self.texture_height + 2)  # 0, 1, ..., h-1
        self.grid_xs = np.linspace(-1, self.texture_width, self.texture_width + 2)  # 0, 1, ..., w-1
        # yg, xg = np.meshgrid(ys, xs, indexing='ij')
        self.interpolator = RegularGridInterpolator(
            (self.grid_ys, self.grid_xs), padded_texture, method="linear", bounds_error=True
        )
        # image grid defined on 0..h-1

    def __call__(self, uv: np.ndarray):
        """
        query the texture map at locations uv
        Args:
            uv: (*, 2)  u is in the x/width direction, v is in the y/height direction,

        Returns:
            (*, dim)
        """
        if isinstance(uv, (list, tuple)):
            uv = np.array(uv)

        # in case want to tile the texture map
        uv = np.mod(uv, 1)

        # convert uv to yx
        y = uv[..., 1:2] * self.texture_height - 0.5  # (*, 1)
        x = uv[..., 0:1] * self.texture_width - 0.5  # (*, 1)

        # we need to mark out-of-boundary UVs and manually set them to zeros later
        mask_invalid_y = np.logical_or(y < 0, y > self.texture_height - 1)
        mask_invalid_x = np.logical_or(x < 0, x > self.texture_width - 1)
        # [..., 1] -> [...,], reduce the last dimesnion
        mask_invalid = np.logical_or(mask_invalid_x, mask_invalid_y)[..., 0]

        clipped_y = np.clip(y, 0, self.texture_height - 1)
        clipped_x = np.clip(x, 0, self.texture_width - 1)

        clipped_yx = np.concatenate((clipped_y, clipped_x), axis=-1)

        # reduce_axis_list = tuple(np.arange(len(clipped_yx.shape) - 1).tolist())
        # min_clipped_yx = np.min(clipped_yx, axis=reduce_axis_list)
        # max_clipped_yx = np.max(clipped_yx, axis=reduce_axis_list)

        # print(
        #     f"\n{min_clipped_yx=}, {max_clipped_yx=}, {self.texture_height=}, {self.texture_width=}, "
        #     f"{self.grid_ys=}, {self.grid_xs=}\n"
        # )

        ret_val = self.interpolator(clipped_yx)

        if np.any(mask_invalid):
            fill_val = np.zeros((1, ret_val.shape[-1]), dtype=ret_val.dtype)
            ret_val[mask_invalid] = fill_val  # [..., texture_dimension]

        return ret_val
