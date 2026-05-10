#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements helper functions to visualization.

import json
import os
import typing as T

import imageio.v3 as iio
import numpy as np

import pytorch3d.transforms
import torch

from blender_rendering import blender_open3d_utils, blender_plib_utils, utils as blender_rendering_utils
from plibs import exr_utils, gs_utils, json_utils, render as p_render, rigid_motion, structures, utils


def colormap_xyz_w(xyz_w: torch.Tensor):
    """
    (From LION), the colormap that converts xyz to rgb.
    Args:
        xyz_w:
            (n, 3xyz_w) [-1, 1]

    Returns:
        rgb:
            (n, 3rgb) [0, 1]
    """
    rgb = (xyz_w + 1) * 0.5  # (n, 3)
    rgb[:, -1] -= 0.0125
    rgb = torch.clamp(rgb, min=0.001, max=1.0)
    norm = torch.sqrt(torch.sum(rgb**2, dim=-1, keepdim=True))  # (n, 1)
    rgb /= norm
    return rgb


def render_point_cloud_with_blender(
    out_dir: str,
    xyz_w: torch.Tensor,
    point_radius: float = 0.02,
    rgb: T.Optional[torch.Tensor] = None,
    H_c2w_o3d: T.Optional[torch.Tensor] = None,
    intrinsic_o3d: T.Optional[torch.Tensor] = None,
    width_px: int = 1024,
    height_px: int = 1024,
    rotate_around_xaxis: bool = False,
    save_blender_file: bool = False,
):
    """
    Render point cloud in blender with area light, ground plane and global illumination.

    Args:
        xyz_w:
            (n, 3) [-1, 1].  See notes.
        point_radius:
            radius of each point in blender
        rgb:
            (n, 3) rgb [0, 1]
        H_c2w_o3d:
            (q, 4, 4).  If None, render 12 images at 45 degree angle
            This is assumed to be in the o3d coordinate.
        intrinsic_o3d:
            (q, 4, 4).  If None, fov = 45 degrees.
            This is assumed to be in the o3d coordinate.
        width_px:
        height_px:
        rotate_around_xaxis:
            whether to rotate xyz_w 90 degree around the x-axis

    Returns:

    Notes:
        We assume the ground plane is parallel to the xy-plane of the world coordinate used by xyz_w.
        In other words, +z is toward top.
    """

    os.makedirs(out_dir, exist_ok=True)

    if rotate_around_xaxis:
        R_w2n = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).float()  # (3, 3)
        xyz_w = (R_w2n.unsqueeze(0) @ xyz_w.unsqueeze(-1)).squeeze(-1)  # (n, 3)
        del R_w2n

    if rgb is None:
        rgb = colormap_xyz_w(xyz_w)  # (n, 3)

    # save point cloud into an npz
    npz_filename = os.path.join(out_dir, "pcd.npz")
    np.savez(
        npz_filename,
        xyz_w=xyz_w.detach().cpu().float().numpy(),
        rgba=rgb.detach().cpu().float().numpy(),
    )

    if H_c2w_o3d is None:
        # generate camera poses at 45 degree angle looking toward the origin
        H_c2w_o3d = []
        r = 3
        z = 3
        n = 12
        for i in range(n):
            H_c2w = rigid_motion.get_H_c2w_lookat(
                pinhole_location_w=torch.tensor(
                    [r * np.cos(2 * np.pi / n * i), r * np.sin(2 * np.pi / n * i), z]
                ).float(),
                look_at_w=(0.0, 0.0, 0.0),
                up_w=(0.0, 0.0, 1.0),
                invert_y=True,
            )  # (4, 4)
            H_c2w_o3d.append(H_c2w)
        H_c2w_o3d = torch.stack(H_c2w_o3d, dim=0)  # (n, 4, 4)

    if intrinsic_o3d is None:
        fov = np.ones(H_c2w_o3d.shape[0]) * 40.0  # (n,)
        intrinsic_o3d = torch.from_numpy(
            p_render.derive_camera_intrinsics(
                width_px=width_px,
                height_px=height_px,
                fov=fov,
            )
        )  # (n, 3, 3)

    # construct the json config dict for the scene
    mesh_dicts = []
    pcd_dicts = []
    mdict = dict(
        name="pcd",
        filename=npz_filename,
        radius=point_radius,  # meter
        metallic=0,
        roughness=1.0,
    )
    pcd_dicts.append(mdict)

    camera_dicts = []
    for ii in range(H_c2w_o3d.shape[0]):
        mdict = blender_open3d_utils.convert_open3d_camera_to_blender(
            H_c2w=H_c2w_o3d[ii],
            intrinsic=intrinsic_o3d[ii],
            width_px=width_px,
            height_px=height_px,
        )
        camera_dicts.append(mdict)

    light_dicts = []

    # the default light is square, 0.25 meter in full width, at (0,0,0), looking at -z (like a blender camera)
    H_c2w_light_o3d = rigid_motion.get_H_c2w_lookat(
        pinhole_location_w=[-5, -5, 20.0],
        look_at_w=(0.0, 0.0, 0.0),
        up_w=(0.0, 0.0, 1),
        invert_y=True,
    )  # (4, 4)
    assert H_c2w_light_o3d.shape == (4, 4)
    H_c2w_light_o3d = H_c2w_light_o3d.cpu().numpy()
    H_c2w_light_blender = blender_open3d_utils.convert_open3d_camera_to_blender_H_c2w(H_c2w_light_o3d)

    mdict = dict(
        name="light 1",
        light_type="AREA",
        H_c2w=H_c2w_light_blender,
        size=20.0,
        size_y=20.0,
        energy=9000.0,
        use_shadow=True,
        color=[1.0, 1.0, 1.0, 1.0],
    )
    light_dicts.append(mdict)

    plane_dicts = []
    # default plane is on xy-plane
    H_c2w_plane_o3d = torch.eye(4)
    H_c2w_plane_o3d[:3, 3] = torch.tensor([0.0, 0.0, -1.5])
    H_c2w_plane_blender = (
        H_c2w_plane_o3d  # blender_open3d_utils.convert_open3d_camera_to_blender_H_c2w(H_c2w_light_o3d)
    )
    mdict = dict(
        name="ground",
        H_c2w=H_c2w_plane_blender,
        length_x=15.0,
        length_y=15.0,
        rgba=[1.0, 1.0, 1.0, 1.0],
        metallic=0.3,
        roughness=1.0,
    )
    plane_dicts.append(mdict)

    config_dict = dict(
        meshes=mesh_dicts,
        cameras=camera_dicts,
        lighting=light_dicts,
        point_clouds=pcd_dicts,
        planes=plane_dicts,
    )

    json_filename = os.path.join(out_dir, "scene.json")
    with open(json_filename, "w") as f:
        json.dump(config_dict, f, indent=2, cls=json_utils.NumpyJsonEncoder)

    # render with blender
    render_out_dir = os.path.join(out_dir, "render")

    blender_cmd = blender_rendering_utils.get_blender_exe()
    blender_script = blender_rendering_utils.get_blender_utils_path()
    cmd = (
        f"{blender_cmd} --background --python {blender_script} -- "
        f"--filename {json_filename} --out_dir {render_out_dir} --debug {int(save_blender_file)} "
    )
    print(cmd)
    os.system(cmd)

    # read resutls

    all_results = []
    for ii in range(H_c2w_o3d.shape[0]):
        # exr
        ddict = dict()
        for key in ["rgb", "depth", "normal", "obj_id"]:
            filename = os.path.join(out_dir, f"{ii:04d}_{key}.exr")
            arr = exr_utils.read_exr(filename)  # (h, w, c)
            assert arr.shape[:2] == (h, w)
            ddict[key] = arr

        # srgb
        filename = os.path.join(out_dir, f"{ii:04d}_srgb.png")
        arr = iio.imread(filename)
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255
        elif arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 65535
        else:
            raise NotImplementedError
        assert arr.shape[:2] == (h, w)
        ddict["srgb"] = arr

        # camera
        filename = os.path.join(out_dir, f"{ii:04d}_camera.json")
        with open(filename, "r") as f:
            cam_info = json.load(f)
        ddict["cam_info"] = cam_info

        # # # check
        assert np.allclose(cam_info["H_c2w_open3d"], H_c2w_o3d[ii], rtol=1e-6, atol=1e-6), (
            f"{cam_info['H_c2w_open3d']} \n\n {H_c2w_o3d[ii]}"
        )
        assert np.allclose(cam_info["intrinsic_open3d"], intrinsic_o3d[ii], rtol=1e-6, atol=1e-6), (
            f"{cam_info['intrinsic_open3d']} \n\n {intrinsic_o3d[ii]}"
        )

        all_results.append(ddict)

    # create rgbd image from all_results
    rgbds = []
    for ii in range(len(all_results)):
        ddict = all_results[ii]
        hit_map = torch.from_numpy(ddict["rgb"][:, :, 3]) > 0.5  # (h, w)
        rgb = torch.from_numpy(ddict["srgb"][:, :, :3]).float()
        rgb = rgb * hit_map.unsqueeze(-1).float()
        normal_w = torch.from_numpy(ddict["normal"][:, :, :3]).float()
        normal_w = normal_w * hit_map.unsqueeze(-1).float()
        depth = torch.from_numpy(ddict["depth"]).float()
        depth = depth * hit_map.unsqueeze(-1).float()

        assert (depth > -1e-9).all()
        cam_info = ddict["cam_info"]
        camera = structures.Camera(
            H_c2w=torch.tensor(cam_info["H_c2w_open3d"]).float().reshape(1, 1, 4, 4),
            intrinsic=torch.tensor(cam_info["intrinsic_open3d"]).float().reshape(1, 1, 3, 3),
            width_px=cam_info["width_px"],
            height_px=cam_info["height_px"],
        )

        rgbd = structures.RGBDImage(
            rgb=rgb.reshape(1, 1, h, w, 3),
            depth=depth.reshape(1, 1, h, w),
            normal_w=normal_w.reshape(1, 1, h, w, 3),
            hit_map=hit_map.reshape(1, 1, h, w),
            camera=camera,
        )

        rgbds.append(rgbd)

    rgbd = structures.RGBDImage.cat(rgbds, dim=1)  # (b=1, q, h, w)

    return rgbd
