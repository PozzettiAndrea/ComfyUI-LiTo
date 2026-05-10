#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements utils to use pytorch3d


import typing as T

from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.utils import opencv_from_cameras_projection
import torch

from plibs import rigid_motion


def get_intrinsic_and_H_c2w(
    camera: PerspectiveCameras,
    width_px: int,
    height_px: int,
) -> T.Dict[str, T.Any]:
    """
    Compute the 3x3 intrinsic matrix and 4x4 H_c2w from the camera object.

    Args:
        camera:
            (b,)
        width_px:
            number of pixels of the image in width
        height_px:
            number of pixels of the image in height

    Returns:
        intrinsic:
            (b, 3, 3)
        H_c2w:
            (b, 4, 4)
    """

    dtype = camera.R.dtype
    device = camera.R.device
    n = camera.R.size(0)

    if isinstance(width_px, int):
        image_size = torch.tensor([height_px, width_px], dtype=dtype, device=device)
    else:
        image_size = torch.cat([height_px, width_px], dim=-1)
    image_size = image_size.expand(n, 2)

    R_w2c, t_w2c, intrinsic = opencv_from_cameras_projection(
        cameras=camera,
        image_size=image_size,
    )  # (n, 3, 3)  (n, 3)  (n, 3, 3)
    H_w2c = torch.zeros(n, 4, 4, dtype=dtype, device=device)  # (n, 4, 4)
    H_w2c[:, :3, :3] = R_w2c
    H_w2c[:, :3, 3] = t_w2c
    H_c2w = rigid_motion.inv_homogeneous_tensors(H_w2c)  # (b, 4, 4)

    return dict(
        H_c2w=H_c2w,
        intrinsic=intrinsic,
        width_px=width_px,
        height_px=height_px,
    )


def get_intrinsic_and_H_c2w_bug(
    camera: PerspectiveCameras,
    width_px: int,
    height_px: int,
) -> T.Dict[str, T.Any]:
    """
    Compute the 3x3 intrinsic matrix and 4x4 H_c2w from the camera object.

    Args:
        camera:
            (b,)
        width_px:
            number of pixels of the image in width
        height_px:
            number of pixels of the image in height

    Returns:
        intrinsic:
            (b, 3, 3)
        H_c2w:
            (b, 4, 4)

    Note:
        The function does not account for the fact that pytorch's ndc is (+X left, +Y up)
        whereas the screen/image space is (+X right, +Y down, origin top left).
    """

    if camera.in_ndc():
        fx_ndc = camera.focal_length[:, 0]  # (b,)
        fy_ndc = camera.focal_length[:, 1]  # (b,)
        ps = camera.get_principal_point()  # (b, 2)
        px_ndc = ps[:, 0]  # (b,)
        py_ndc = ps[:, 1]  # (b,)
        s = min(width_px, height_px)

        fx_screen = fx_ndc / 2.0 * s  # (b, )
        fy_screen = fy_ndc / 2.0 * s  # (b, )
        px_screen = -1 * px_ndc / 2.0 * s + width_px / 2  # (b, )
        py_screen = -1 * py_ndc / 2.0 * s + height_px / 2  # (b, )

    else:
        fx_screen = camera.focal_length[:, 0]  # (b,)
        fy_screen = camera.focal_length[:, 1]  # (b,)
        ps = camera.get_principal_point()  # (b, 2)
        px_screen = ps[:, 0]  # (b,)
        py_screen = ps[:, 1]  # (b,)

    b = fx_screen.size(0)
    intrinsic = torch.zeros(b, 3, 3, dtype=fx_screen.dtype, device=fx_screen.device)  # (b, 3, 3)
    intrinsic[:, 0, 0] = fx_screen
    intrinsic[:, 1, 1] = fy_screen
    intrinsic[:, 0, 2] = px_screen
    intrinsic[:, 1, 2] = py_screen
    intrinsic[:, 2, 2] = 1

    # get H_c2w
    w2c_trans = camera.get_world_to_view_transform()
    H_c2w = w2c_trans.inverse().get_matrix().transpose(-1, -2)  # (b, 4, 4)

    return dict(
        H_c2w=H_c2w,
        intrinsic=intrinsic,
        width_px=width_px,
        height_px=height_px,
    )
