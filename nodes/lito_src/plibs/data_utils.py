#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements the script to sample point clouds and images
# from a mesh.

from collections import OrderedDict
import contextlib
import copy
import enum
import gc
import glob
import json
import os
import pathlib
import random
import resource
import shutil
import subprocess
import sys
import tempfile
from timeit import default_timer as timer
import typing as T
import zipfile

import cv2
import numpy as np
import numpy.typing as nptyping
import open3d as o3d
import psutil
from scipy.spatial import cKDTree
import tqdm
import trimesh

import torch

from blender_rendering import blender_open3d_utils, blender_plib_utils, utils as blender_rendering_utils
from plibs import exr_utils, json_utils, linalg_utils, mesh_utils, render, rigid_motion, structures, utils
from plibs.byte_dict_utils import load_file_from_byte_dict, load_single_rgbd_file_from_byte_dict

if (sys.version_info.major == 3) and (sys.version_info.minor >= 11):
    # StrEnum is added into enum in 3.11
    from enum import StrEnum
else:
    from strenum import StrEnum


REPO_ROOT = pathlib.Path(__file__).parent.parent
DEBUG_ROOT = REPO_ROOT / "debug"
DEBUG_ROOT.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # This is very important to ensure full reproducibility for Open3D, especially for sampling point cloud.
    o3d.utility.random.seed(seed)


class FuncType(StrEnum):
    UNKNOWN = enum.auto()
    SAMPLE_PCD_RGBD_FROM_MESH = enum.auto()
    SAMPLE_PCD_RGBD_FROM_MESH_WITH_BLENDER = enum.auto()
    SAMPLE_PCD_RGBD_FROM_MESH_WITH_BLENDER_AND_O3D = enum.auto()
    SAMPLE_PCD_RGBD_FROM_MULTIPLE_MESHES_WITH_BLENDER = enum.auto()


def get_process_func(func_type: FuncType):
    if func_type == FuncType.SAMPLE_PCD_RGBD_FROM_MESH:
        return sample_pcd_rgbd_from_mesh
    elif func_type == FuncType.SAMPLE_PCD_RGBD_FROM_MESH_WITH_BLENDER:
        return sample_pcd_rgbd_from_mesh_with_blender
    elif func_type == FuncType.SAMPLE_PCD_RGBD_FROM_MULTIPLE_MESHES_WITH_BLENDER:
        return sample_pcd_rgbd_from_multiple_meshes_with_blender
    elif func_type == FuncType.SAMPLE_PCD_RGBD_FROM_MESH_WITH_BLENDER_AND_O3D:
        return sample_pcd_rgbd_from_mesh_with_blender_and_o3d
    else:
        raise NotImplementedError(str(func_type))


def sample_pcd_from_mesh(
    mesh_filename: str,
    out_dir: str,
    out_dir_rgbd: str = None,
    pcd_sample_method: str = "uniform",
    num_points: int = 100_000,
    raise_error_if_no_color: bool = True,
    overwrite: bool = False,
    max_time_to_sample_pcd: float = None,
    mesh_scale: T.Optional[float] = 1.0,
    mesh_center_w: T.Optional[T.List[float]] = [0.0, 0.0, 0.0],
    preprocess_mesh: bool = True,
    compute_raycasting_scene: bool = True,
):
    """
    Sample point cloud from a mesh."""
    if not overwrite:
        assert (not os.path.exists(out_dir)) or (not os.listdir(out_dir))

    if out_dir_rgbd is None:
        out_dir_rgbd = out_dir

    odict = mesh_utils.load_mesh_using_trimesh(
        filename=mesh_filename,
        raise_error_if_no_color=raise_error_if_no_color,
    )
    o3d_mesh = odict["o3d_mesh"]
    has_color_texture = odict["has_color_texture"]

    st_mesh = structures.Mesh(
        mesh=o3d_mesh,
        scale=mesh_scale,
        center_w=mesh_center_w,
        preprocess_mesh=preprocess_mesh,
        compute_raycasting_scene=compute_raycasting_scene,
    )

    # sample point cloud
    stime = timer()
    out = st_mesh.sample_point_cloud(
        num_points=num_points * 2,  # 2x points just in case
        method=pcd_sample_method,
    )  # note that it might not get exactly N points, so we sample more
    point_cloud: structures.PointCloud = out["point_cloud"]  # (b=1, n, 3xyz)
    total_time = timer() - stime
    print(f"Sampling {num_points} points with {pcd_sample_method} takes {total_time:.3f} secs.")
    print(f"before removing bad points, num points: {point_cloud.xyz_w.size(1)}")

    if max_time_to_sample_pcd is not None and max_time_to_sample_pcd > 0:
        if total_time > max_time_to_sample_pcd:
            raise RuntimeError(f"Sampling point cloud takes {total_time:.3f}. Raise error")

    # remove points that are outside [-1, 1] aabb, and are not finite
    xyz_w = point_cloud.extract_valid_attr(
        arr=point_cloud.xyz_w,
        bidx=0,
    )  # (n, 3)
    n, _3xyz = xyz_w.shape

    bbox_eps = 0.01
    vmask = torch.logical_and(
        (xyz_w >= -1 - bbox_eps).all(dim=-1),  # (n,)
        (xyz_w <= 1 + bbox_eps).all(dim=-1),  # (n,)
    )  # (n,)
    # print(f'after xyz_w, vmask: {vmask.shape} ({vmask.float().mean()})')
    normal_w = point_cloud.extract_valid_attr(
        arr=point_cloud.normal_w,
        bidx=0,
    )  # (n, 3)
    assert xyz_w.size(0) == normal_w.size(0)
    vmask = torch.logical_and(
        vmask,  # (n,)
        normal_w.isfinite().all(dim=-1),  # (n,)
    )  # (n,)
    # print(f'after normal, vmask: {vmask.shape} ({vmask.float().mean()})')
    rgb = point_cloud.extract_valid_attr(
        arr=point_cloud.rgb,
        bidx=0,
    )  # (n, 3)
    assert xyz_w.size(0) == rgb.size(0)
    vmask = torch.logical_and(
        vmask,  # (n,)
        rgb.isfinite().all(dim=-1),  # (n,)
    )  # (n,)
    # print(f'after rgb, vmask: {vmask.shape} ({vmask.float().mean()})')
    # print(f'vmask.shape = {vmask.shape} (total = {vmask.sum()})')
    # print(f'rgb.shape = {rgb.shape} ({rgb[vmask].shape})')
    # create point cloud with only valid points
    point_cloud = structures.PointCloud(
        xyz_w=xyz_w[vmask].reshape(1, -1, 3),
        rgb=rgb[vmask].reshape(1, -1, 3),
        normal_w=normal_w[vmask].reshape(1, -1, 3),
    )  # (b=1, n, 3)
    print(f"after removing bad points, num points: {point_cloud.xyz_w.size(1)}")

    return {"st_mesh": st_mesh, "point_cloud": point_cloud, "has_color_texture": has_color_texture}


def save_sampled_pcd_dynamic(
    *,
    pcd_save_version: int,
    out_dir: str,
    index_dict: T.Dict[str, T.Any],
    xyz_w: torch.Tensor,  # (n, 3)
    rgb: torch.Tensor,  # (n, 3)
    normal_w: torch.Tensor,  # (n, 3)
    save_np_dtype,
    save_chunk_size=100_000,
    internal_folder_name="",
    overwrite=False,
    view_dir: T.Optional[torch.Tensor] = None,  # (n, 3)
):
    """
    save point xyz, rgb, and normal as npy files.

    Args:
        pcd_save_version:
            1:
                use a single file to save all points
            2:
                chunk points and save as separate npy files (better)
        out_dir:
            root_dir
        index_dict:
            index dict of the whole (not pcd_index_dict)
        xyz_w:
            (n, 3)
        rgb:
            (n, 3) [0, 1]
        normal_w:
            (n, 3)
        view_dir:
            (n, 3)
        save_np_dtype:
            np.float32
        save_chunk_size:
            used by
        internal_folder_name:
            will save xyz_w/{internal_folder_name}/file.npy instead, ignored if folder is ""

    Returns:

    """
    if pcd_save_version == 1:
        # save pcd as a single huge npz file containing all of xyz_w, rgb, normal_w
        pcd_filename = os.path.join(out_dir, "pcd.npz")
        if view_dir is not None:
            np.savez(
                pcd_filename,
                xyz_w=xyz_w.detach().cpu().numpy(),  # (n, 3xyz_w)
                rgb=rgb.detach().cpu().numpy(),  # (n, 3rgb) [0, 1]
                normal_w=normal_w.detach().cpu().numpy(),  # (n, 3xyz_w)
            )
        else:
            np.savez(
                pcd_filename,
                xyz_w=xyz_w.detach().cpu().numpy(),  # (n, 3xyz_w)
                rgb=rgb.detach().cpu().numpy(),  # (n, 3rgb) [0, 1]
                normal_w=normal_w.detach().cpu().numpy(),  # (n, 3xyz_w)
                view_dir=view_dir.detach().cpu().numpy(),  # (n, 3xyz_w)
            )
        index_dict["pcd_filename"] = os.path.relpath(pcd_filename, start=out_dir)
    elif pcd_save_version == 2:
        # save pcd as a separate huge npy file containing one of xyz_w, rgb, normal_w,
        # and each xyz_w / rgb / normal_w is split into multiple files (max number of points = 100k)
        # chunk_size = 100_000
        if save_chunk_size is None or save_chunk_size < 0:
            save_chunk_size = xyz_w.size(0)

        num_chunks = (xyz_w.size(0) + save_chunk_size - 1) // save_chunk_size
        fdict = dict(chunk_size=save_chunk_size, num_chunks=num_chunks)
        for chunk_idx in range(num_chunks):
            name_arr_lists = [
                ["xyz_w", xyz_w.detach().cpu().numpy().astype(save_np_dtype)],
                ["normal_w", normal_w.detach().cpu().numpy().astype(save_np_dtype)],
                ["rgb", rgb.detach().cpu().numpy().astype(save_np_dtype)],
            ]
            if view_dir is not None:
                name_arr_lists.append(["view_dir", view_dir.detach().cpu().numpy().astype(save_np_dtype)])

            for name, arr in name_arr_lists:
                _arr = arr[chunk_idx * save_chunk_size : (chunk_idx + 1) * save_chunk_size]  # (m, 3)

                # if internal_folder_name is not ""
                filename = os.path.join(out_dir, name, internal_folder_name, f"{name}_{chunk_idx}.npy")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                np.save(filename, _arr)
                if name not in fdict:
                    fdict[name] = []
                fdict[name].append(os.path.relpath(filename, start=out_dir))

        pcd_index_filename = os.path.join(out_dir, "pcd_index.json")
        # with open(pcd_index_filename, "w") as f:
        #     json.dump(fdict, f, indent=2)
        # Check if file exists

        if os.path.exists(pcd_index_filename) and not overwrite:
            # Load the existing data
            with open(pcd_index_filename, "r") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = None
        else:
            existing_data = None

        if existing_data is None:
            # If file does not exist or is invalid, initialize with new data
            data_to_write = fdict
        else:
            # If existing_data is a list, add fdict to the list
            if isinstance(existing_data, list):
                existing_data.append(fdict)
                data_to_write = existing_data
            # If existing_data is a dict, convert it to a list and add fdict
            elif isinstance(existing_data, dict):
                data_to_write = [existing_data, fdict]
            else:
                # Fallback: if it's an unexpected type, start a new list with the new dict
                data_to_write = [fdict]

        with open(pcd_index_filename, "w") as f:
            json.dump(data_to_write, f, indent=2)
        index_dict["pcd_index_filename"] = os.path.relpath(pcd_index_filename, start=out_dir)
        index_dict["pcd_save_version"] = pcd_save_version
    else:
        raise NotImplementedError
    return index_dict


def save_sampled_pcd(
    *,
    pcd_save_version: int,
    out_dir: str,
    index_dict: T.Dict[str, T.Any],
    xyz_w: torch.Tensor,  # (n, 3)
    rgb: torch.Tensor,  # (n, 3)
    normal_w: torch.Tensor,  # (n, 3)
    save_np_dtype,
    save_chunk_size=100_000,
    other_attrs: T.Dict[str, torch.Tensor] = {},
):
    """
    save point xyz, rgb, and normal as npy files.

    Args:
        pcd_save_version:
            1:
                use a single file to save all points
            2:
                chunk points and save as separate npy files (better)
        out_dir:
            root_dir
        index_dict:
            index dict of the whole (not pcd_index_dict)
        xyz_w:
            (n, 3)
        rgb:
            (n, 3) [0, 1]
        normal_w:
            (n, 3)
        save_np_dtype:
            np.float32
        save_chunk_size:
            used by
        other_attrs:
            other attributes that we want to save besides [xyz_w, rgb, normal_w]

    Returns:

    """
    if pcd_save_version == 1:
        # save pcd as a single huge npz file containing all of xyz_w, rgb, normal_w
        pcd_filename = os.path.join(out_dir, "pcd.npz")
        np.savez(
            pcd_filename,
            xyz_w=xyz_w.detach().cpu().numpy(),  # (n, 3xyz_w)
            rgb=rgb.detach().cpu().numpy(),  # (n, 3rgb) [0, 1]
            normal_w=normal_w.detach().cpu().numpy(),  # (n, 3xyz_w)
        )
        index_dict["pcd_filename"] = os.path.relpath(pcd_filename, start=out_dir)
    elif pcd_save_version == 2:
        # save pcd as a separate huge npy file containing one of xyz_w, rgb, normal_w,
        # and each xyz_w / rgb / normal_w is split into multiple files (max number of points = 100k)
        # chunk_size = 100_000
        num_chunks = (xyz_w.size(0) + save_chunk_size - 1) // save_chunk_size
        fdict = dict(chunk_size=save_chunk_size, num_chunks=num_chunks, xyz_w=[], normal_w=[], rgb=[])
        for tmp_k in other_attrs:
            fdict[tmp_k] = []

        for chunk_idx in range(num_chunks):
            for name, arr in [
                ["xyz_w", xyz_w.detach().cpu().numpy().astype(save_np_dtype)],
                ["normal_w", normal_w.detach().cpu().numpy().astype(save_np_dtype)],
                ["rgb", rgb.detach().cpu().numpy().astype(save_np_dtype)],
            ] + [[tmp_k, tmp_v.detach().cpu().numpy().astype(save_np_dtype)] for tmp_k, tmp_v in other_attrs.items()]:
                _arr = arr[chunk_idx * save_chunk_size : (chunk_idx + 1) * save_chunk_size]  # (m, 3)
                filename = os.path.join(out_dir, name, f"{name}_{chunk_idx}.npy")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                np.save(filename, _arr)
                fdict[name].append(os.path.relpath(filename, start=out_dir))

        pcd_index_filename = os.path.join(out_dir, "pcd_index.json")
        with open(pcd_index_filename, "w") as f:
            json.dump(fdict, f, indent=2)
        index_dict["pcd_index_filename"] = os.path.relpath(pcd_index_filename, start=out_dir)
        index_dict["pcd_save_version"] = pcd_save_version
    else:
        raise NotImplementedError
    return index_dict


def sample_pcd_rgbd_from_mesh(
    *,
    mesh_filename: str,
    out_dir: str,
    out_dir_rgbd: str = None,
    out_dir_pcd_rgbd_visibility_dict: T.Union[T.Dict[float, str], None] = None,
    pcd_sample_method: str = "uniform",
    num_points: int = 100_000,
    width_px: int = 448,  # 14 x 32
    height_px: int = 448,  # 14 x 32
    # circular camera
    num_regular_images: int = 10,
    fov: float = 40.0,  # degree
    circular_radius: float = 3.5,  # meter
    # random camera
    num_random_images: int = 30,
    min_fov: float = 40.0,  # degree
    max_fov: float = 60.0,  # degree
    min_random_radius: float = 3,
    max_random_radius: float = 4,
    random_lookat_r: float = 0.25,
    mesh_rel_dir: str = None,
    background_color: float = 1,
    raise_error_if_no_color: bool = True,
    overwrite: bool = False,
    save_attr_names: T.List[str] = None,
    max_time_to_sample_pcd: float = None,
    pcd_save_version: int = 2,
    save_np_dtype: np.dtype = np.float32,
    seed: int = 0,
    pcd_save_chunk_size: int = 100_000,
    flag_debug: bool = False,
    flag_save_space: bool = False,
    **kwargs,
):
    """
    Sample point cloud from a mesh and capture rgbd images.

    out_dir_rgbd:
        if not None, rgbd will be saved in the folder instead of out_dir

    pcd_save_version:
            1: single npz file saving xyz_w (n, 3), rgb (n, 3), normal_w (n, 3)
            2: multiple npz files, each saving a subset of xyz_w (m, 3) or rgb (n, 3), or normal_w (n, 3),
                xyz, rgb, normal are saved separately
    """
    _set_seed(seed)

    sample_dict = sample_pcd_from_mesh(
        mesh_filename=mesh_filename,
        out_dir=out_dir,
        out_dir_rgbd=out_dir_rgbd,
        pcd_sample_method=pcd_sample_method,
        num_points=num_points,
        raise_error_if_no_color=raise_error_if_no_color,
        overwrite=overwrite,
        max_time_to_sample_pcd=max_time_to_sample_pcd,
    )

    st_mesh: structures.Mesh = sample_dict["st_mesh"]
    point_cloud: structures.PointCloud = sample_dict["point_cloud"]
    has_color_texture = sample_dict["has_color_texture"]

    # generate cameras on a circle on xz-plane
    q = num_regular_images
    fov = fov
    w = width_px
    h = height_px
    r = circular_radius

    # determine intrinsic
    intrinsic = torch.from_numpy(
        render.derive_camera_intrinsics(
            width_px=w,
            height_px=h,
            fov=fov,
        )
    )  # (3, 3)

    # xz-plane
    H_c2w = rigid_motion.generate_circular_camera_poses(
        n=q,
        r=r,
        # normal_w=[1., 0., 0.],  # (yz-plane)
        normal_w=[0.0, 1.0, 0.0],  # (xz-plane)
        # normal_w=[0., 0., 1.],  # (xy-plane)
    )
    camera = structures.Camera(
        H_c2w=H_c2w.unsqueeze(0),  # (1, q, 4, 4)
        intrinsic=intrinsic.expand(1, q, 3, 3),  # (1, q, 3, 3)
        width_px=w,
        height_px=h,
    )
    # capture images
    rgbd_image_xz = st_mesh.get_rgbd_image(
        camera=camera,  # (1, q)
        render_method="ray_cast",
    )

    # yz-plane
    H_c2w = rigid_motion.generate_circular_camera_poses(
        n=q,
        r=r,
        normal_w=[1.0, 0.0, 0.0],  # (yz-plane)
        # normal_w=[0., 1., 0.],  # (xz-plane)
        # normal_w=[0., 0., 1.],  # (xy-plane)
    )
    camera = structures.Camera(
        H_c2w=H_c2w.unsqueeze(0),  # (1, q, 4, 4)
        intrinsic=intrinsic.expand(1, q, 3, 3),  # (1, q, 3, 3)
        width_px=w,
        height_px=h,
    )
    # capture images
    rgbd_image_yz = st_mesh.get_rgbd_image(
        camera=camera,  # (1, q)
        render_method="ray_cast",
    )

    # xy-plane
    H_c2w = rigid_motion.generate_circular_camera_poses(
        n=q,
        r=r,
        # normal_w=[1., 0., 0.],  # (yz-plane)
        # normal_w=[0., 1., 0.],  # (xz-plane)
        normal_w=[0.0, 0.0, 1.0],  # (xy-plane)
    )
    camera = structures.Camera(
        H_c2w=H_c2w.unsqueeze(0),  # (1, q, 4, 4)
        intrinsic=intrinsic.expand(1, q, 3, 3),  # (1, q, 3, 3)
        width_px=w,
        height_px=h,
    )
    # capture images
    rgbd_image_xy = st_mesh.get_rgbd_image(
        camera=camera,  # (1, q)
        render_method="ray_cast",
    )

    # random images
    H_c2w = rigid_motion.generate_random_camera_poses_lookat(
        n=num_random_images,
        pinhole_min_r=min_random_radius,
        pinhole_max_r=max_random_radius,
        lookat_r=random_lookat_r,
    )  # (q, 4, 4)

    random_fovs = np.random.rand(num_random_images) * (max_fov - min_fov) + min_fov  # (q,) degree
    intrinsic = torch.from_numpy(
        render.derive_camera_intrinsics(
            width_px=width_px,
            height_px=height_px,
            fov=random_fovs,
        )
    )  # (q, 3, 3)
    camera = structures.Camera(
        H_c2w=H_c2w.unsqueeze(0),  # (1, q, 4, 4)
        intrinsic=intrinsic.unsqueeze(0),  # (1, q, 3, 3)
        width_px=width_px,
        height_px=height_px,
    )
    # capture images
    random_rgbd_image = st_mesh.get_rgbd_image(
        camera=camera,  # (1, q)
        render_method="ray_cast",
    )

    # save results
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_rgbd, exist_ok=True)
    index_dict = dict()

    # get mesh relative dir
    if mesh_rel_dir is not None:
        fn = os.path.relpath(mesh_filename, start=mesh_rel_dir)
    else:
        fn = mesh_filename
    index_dict["mesh_filename"] = fn
    index_dict["mesh_has_color_texture"] = has_color_texture

    # shuffle point cloud before saving (added when pcd_save_version = 2 is added)
    ridxs = torch.randperm(point_cloud.xyz_w.size(1), device=point_cloud.xyz_w.device)
    xyz_w = point_cloud.xyz_w[0][ridxs]  # (n, 3)
    normal_w = point_cloud.normal_w[0][ridxs]  # (n, 3)
    rgb = point_cloud.rgb[0][ridxs]  # (n, 3)

    index_dict = save_sampled_pcd(
        pcd_save_version=pcd_save_version,
        out_dir=out_dir,
        index_dict=index_dict,
        xyz_w=xyz_w,
        rgb=rgb,
        normal_w=normal_w,
        save_np_dtype=save_np_dtype,
        save_chunk_size=pcd_save_chunk_size,
    )

    # save rgbd image
    for rgbd, name in [
        [rgbd_image_xz, "rgbd_xz"],
        [rgbd_image_yz, "rgbd_yz"],
        [rgbd_image_xy, "rgbd_xy"],
        [random_rgbd_image, "rgbd_random"],
    ]:
        sub_dir = os.path.join(out_dir_rgbd, name)
        rgbd: structures.RGBDImage
        rgbd = rgbd.remove_invalid(
            min_depth=0,
            max_depth=1e4,
            background_color=background_color,
        )
        _, sub_index_filename = rgbd.save_as(
            out_dir=sub_dir,
            overwrite=overwrite,
            mode="png",
            background_color=background_color,
            save_attr_names=save_attr_names,
            flag_save_space=flag_save_space,
        )
        index_dict[name] = dict(
            index_filename=os.path.relpath(sub_index_filename, start=out_dir_rgbd),
            q=rgbd.rgb.size(1),
            h=rgbd.rgb.size(2),
            w=rgbd.rgb.size(3),
        )

    # save json
    json_filename = os.path.join(out_dir, "index.json")
    with open(json_filename, "w") as f:
        json.dump(index_dict, f, indent=2)

    if out_dir_rgbd != out_dir:
        _json_filename = os.path.join(out_dir_rgbd, "index.json")
        with open(_json_filename, "w") as f:
            json.dump(index_dict, f, indent=2)

    return index_dict, json_filename


def get_two_different_sphere_cameras(
    *,
    num_views: int,
    r: float,
    fov: float,
    width_px: int,
    height_px: int,
    up_method: str = "z",
    invert_y: bool = True,
):
    """
    Create two.

    Args:
        num_views:
            number of views for each set of sphere cameras
        r:
            radius of the regular images
        fov:
            fov of the regular images
        width_px / height_px:
            resolution of the rendering
        up_method:
            axis for the up direction, can be chosen from {'y', 'z'}
        invert_y:
            if True, we invert the +y axis to be down (since image coordinate is +x to right, +y to down)

    Returns:
        camera_1:
            structure.Camera
        camera_2:
            structure.Camera
    """
    intrinsic = torch.from_numpy(
        render.derive_camera_intrinsics(
            width_px=width_px,
            height_px=height_px,
            fov=fov,
        )
    ).float()  # (3, 3)

    H_c2w_1 = rigid_motion.generate_uniform_camera_poses_with_golden_spiral(
        n=num_views, r=r, up_method=up_method, invert_y=True
    )  # (n, 4, 4)  tensor

    camera_1 = structures.Camera(
        H_c2w=H_c2w_1.float().unsqueeze(0),  # (1, q, 4, 4)
        intrinsic=intrinsic.expand(1, H_c2w_1.shape[0], 3, 3),  # (1, q, 3, 3)
        width_px=width_px,
        height_px=height_px,
    )

    H_c2w_2 = rigid_motion.generate_uniform_camera_poses_with_sphere_hammersley_sequence(
        n=num_views, r=r, up_method=up_method, invert_y=invert_y, offset=(0, 0), allow_trellis_cam_dist_skew=False
    )  # (n, 4, 4)  tensor

    camera_2 = structures.Camera(
        H_c2w=H_c2w_2.float().unsqueeze(0),  # (1, q, 4, 4)
        intrinsic=intrinsic.expand(1, H_c2w_2.shape[0], 3, 3),  # (1, q, 3, 3)
        width_px=width_px,
        height_px=height_px,
    )

    ret_dict = dict(
        camera_1=camera_1,
        camera_2=camera_2,
    )

    return ret_dict


def get_regular_and_random_cameras(
    *,
    num_regular_images: int,
    r: float,
    fov: float,
    width_px: int,
    height_px: int,
    num_random_images: int,
    min_random_radius: float,
    max_random_radius: float,
    random_lookat_r: float,
    max_fov: float,
    min_fov: float,
    regular_camera_sampling_type: str = "circular",
    regular_camera_sampling_type_func: str = "ours",
    extra_kwargs_regular_cameras: T.Dict[str, T.Any] = {},
    extra_kwargs_random_cameras: T.Dict[str, T.Any] = {},
):
    """
    Create random and regular cameras in o3d camera format.

    Args:
        num_regular_images:
            number of regular images
        r:
            radius of the regular images (which lies on circle or sphere)
        fov:
            fov of the regular images
        width_px:
        height_px:
        num_random_images:
            number of random images that lie within a shell
        min_random_radius:
            min radius of the shell
        max_random_radius:
            max radius of the shell
        random_lookat_r:
            where the random cameras look at
        max_fov:
            max fov in degree of the random camera
        min_fov:
            min fov in degree of the random camera
        regular_camera_sampling_type:
            'circular', 'sphere'
        regular_camera_sampling_type_func:
            'ours', 'trellis'
        extra_kwargs_regular_cameras:
            Any additional arguments that will be fed to functions for sampling regular cameras.
            This provides some flexibilities to use each function.
        extra_kwargs_random_cameras:
            Any additional arguments that will be fed to functions for sampling random cameras.
            This provides some flexibilities to use each function.
        num_regular_for_geometry_images:
            number of regular for geometry images
        regular_camera_for_geometry_sampling_type:
            if not None, we will sample cameras for geometry-related evaluation purposes.
        regular_camera_for_geometry_sampling_type_func:
            if not None, we will use this type of cameras for geometry-related evaluation purposes.

    Returns:
        camera_regular:
            structure.Camera
        camera_random:
            structure.Camera
        camera_regular_for_geometry:
            None or structure.Camera
    """

    camera_regular = None
    camera_random = None

    # create regular camera
    if num_regular_images > 0:
        # intrinsic for all regular camera
        intrinsic_regular = torch.from_numpy(
            render.derive_camera_intrinsics(
                width_px=width_px,
                height_px=height_px,
                fov=fov,
            )
        ).float()  # (3, 3)
        if regular_camera_sampling_type == "circular":
            n_images_per_plane = int(np.ceil(num_regular_images / 3))

            # generate cameras on a circle on a plane
            for tmp_plane_name, tmp_normal_w in (
                ("xz", [0.0, 1.0, 0.0]),  # xz-plane
                ("yz", [1.0, 0.0, 0.0]),  # yz-plane
                ("xy", [0.0, 0.0, 1.0]),  # xy-plane
            ):
                H_c2w = rigid_motion.generate_circular_camera_poses(
                    n=n_images_per_plane, r=r, normal_w=tmp_normal_w, **extra_kwargs_regular_cameras
                )  # (q, 4, 4) tensor
        elif regular_camera_sampling_type == "sphere":
            if regular_camera_sampling_type_func == "ours":
                H_c2w = rigid_motion.generate_uniform_camera_poses_with_golden_spiral(
                    n=num_regular_images, r=r, **extra_kwargs_regular_cameras
                )  # (n, 4, 4)  tensor
            elif regular_camera_sampling_type_func == "trellis":
                H_c2w = rigid_motion.generate_uniform_camera_poses_with_sphere_hammersley_sequence(
                    n=num_regular_images, r=r, **extra_kwargs_regular_cameras
                )  # (n, 4, 4)  tensor
            else:
                raise ValueError(f"{regular_camera_sampling_type=}")
        else:
            raise NotImplementedError(f"{regular_camera_sampling_type=}")

        camera_regular = structures.Camera(
            H_c2w=H_c2w.float().unsqueeze(0),  # (1, q, 4, 4)
            intrinsic=intrinsic_regular.expand(1, H_c2w.shape[0], 3, 3),  # (1, q, 3, 3)
            width_px=width_px,
            height_px=height_px,
        )

    # random images
    if num_random_images > 0:
        H_c2w = rigid_motion.generate_random_camera_poses_lookat(
            n=num_random_images,
            pinhole_min_r=min_random_radius,
            pinhole_max_r=max_random_radius,
            lookat_r=random_lookat_r,
            **extra_kwargs_random_cameras,
        )  # (n, 4, 4)

        random_fovs = np.random.rand(num_random_images) * (max_fov - min_fov) + min_fov  # (n,) degree
        intrinsic_random = torch.from_numpy(
            render.derive_camera_intrinsics(
                width_px=width_px,
                height_px=height_px,
                fov=random_fovs,
            )
        ).float()  # (n, 3, 3)

        camera_random = structures.Camera(
            H_c2w=H_c2w.unsqueeze(0),  # (1, q, 4, 4)
            intrinsic=intrinsic_random.unsqueeze(0),  # (1, q, 3, 3)
            width_px=width_px,
            height_px=height_px,
        )

    # # regular camera for geometry
    # if num_regular_for_geometry_images > 0:
    #     if regular_camera_for_geometry_sampling_type == "sphere":
    #         # geometry should have different camera sampling type
    #         if regular_camera_sampling_type == regular_camera_for_geometry_sampling_type:
    #             assert regular_camera_for_geometry_sampling_type_func != regular_camera_sampling_type_func, (
    #                 f"{regular_camera_for_geometry_sampling_type_func=}, {regular_camera_sampling_type_func=}"
    #             )

    #         if regular_camera_for_geometry_sampling_type_func == "ours":
    #             H_c2w_for_geo = rigid_motion.generate_uniform_camera_poses_with_golden_spiral(
    #                 n=num_regular_for_geometry_images, r=r, **extra_kwargs_regular_cameras
    #             )  # (n, 4, 4)  tensor
    #         elif regular_camera_for_geometry_sampling_type_func == "trellis":
    #             # no skew
    #             extra_kwargs_regular_cameras["offset"] = (0, 0)
    #             extra_kwargs_regular_cameras["allow_trellis_cam_dist_skew"] = False

    #             H_c2w_for_geo = rigid_motion.generate_uniform_camera_poses_with_sphere_hammersley_sequence(
    #                 n=num_regular_for_geometry_images, r=r, **extra_kwargs_regular_cameras
    #             )  # (n, 4, 4)  tensor
    #         else:
    #             raise ValueError(f"{regular_camera_for_geometry_sampling_type=}")
    #     else:
    #         raise NotImplementedError(f"{regular_camera_for_geometry_sampling_type=}")

    #     camera_regular_for_geometry = structures.Camera(
    #         H_c2w=H_c2w_for_geo.float().unsqueeze(0),  # (1, q, 4, 4)
    #         intrinsic=intrinsic_regular.expand(1, H_c2w_for_geo.shape[0], 3, 3),  # (1, q, 3, 3)
    #         width_px=width_px,
    #         height_px=height_px,
    #     )
    # else:
    #     camera_regular_for_geometry = None

    return dict(
        camera_regular=camera_regular,
        camera_random=camera_random,
        # camera_regular_for_geometry=camera_regular_for_geometry,
    )


def scale_camera_for_img_cond_ldm_cam_transformation(
    *, xyz_w: torch.Tensor, H_c2w: torch.Tensor, img_cond_ldm_cam_transformation: str = "z_up"
):
    """Adjust camera's H_c2w for generative model training.

    During image conditioning generative model training, we will rotate the input point cloud and camera
    such that the camera is always at the same pose. This could cause the point cloud go out of boundary of [-1, 1]^3.
    We need to rescale the c2w such that point cloud is always within boundary.
    """
    assert H_c2w.shape == (4, 4), f"{H_c2w.shape=}"
    if img_cond_ldm_cam_transformation == "y_up":
        # rotate the world coordinate so that the first camera's camera pose
        # is at diag([1, -1, -1]), ie, the world is y-up
        R_w2c = H_c2w[:3, :3].t()  # (3, 3)
        R_c2b = torch.tensor(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ],
            dtype=R_w2c.dtype,
            device=R_w2c.device,
        )  # (3, 3)
        R_w2b = R_c2b @ R_w2c
        H_w2b = torch.eye(4)  # (4, 4)
        H_w2b[:3, :3] = R_w2b
    elif img_cond_ldm_cam_transformation == "z_up":
        # rotate the world coordinate so that the first camera's camera pose
        # is below and the pinhole is at (r, 0, 0), ie, the world is z-up
        R_w2c = H_c2w[:3, :3].t()  # (3, 3))
        R_c2b = torch.tensor(
            [
                [0, 0, -1],
                [1, 0, 0],
                [0, -1, 0],
            ],
            dtype=R_w2c.dtype,
            device=R_w2c.device,
        )  # (3, 3)
        R_w2b = R_c2b @ R_w2c
        H_w2b = torch.eye(4)  # (4, 4)
        H_w2b[:3, :3] = R_w2b
    elif img_cond_ldm_cam_transformation == "identity_rotation":
        # rotate the world coordinate so that the first camera's oriention aligns with the world.
        # The pinhole is at (0, 0, -r)
        R_w2c = H_c2w[:3, :3].t()  # (3, 3)
        R_c2b = torch.eye(3)  # (3, 3)
        R_w2b = R_c2b @ R_w2c
        H_w2b = torch.eye(4)  # (4, 4)
        H_w2b[:3, :3] = R_w2b
    else:
        raise ValueError(f"{img_cond_ldm_cam_transformation=}")

    xyz_w = linalg_utils.matmul(R_w2b, xyz_w.unsqueeze(-1)).squeeze(-1)  # (n, 3)

    # NOTE: after rotation, xyz_w could be out of bounding box of [-1, 1].
    # If that happens, we need to scale it.
    xyz_w_max_abs = torch.abs(xyz_w).max()
    if xyz_w_max_abs <= 1.0 + 1e-6:
        scene_renormalize_scale = 1.0
        flag_renormalize = False
    else:
        scene_renormalize_scale = (1 - 1e-6) / xyz_w_max_abs
        flag_renormalize = True

    if flag_renormalize:
        # NOTE: since we will down-scale H_c2w during generative model training,
        # to maintain the same camera radius, we should up-scale H_c2w here.
        H_c2w[:3, 3] = H_c2w[:3, 3] / scene_renormalize_scale

    return H_c2w


def generate_trellis_cameras_for_img_cond_ldm(
    *,
    width_px: int,
    height_px: int,
    n_views: int = 24,
    up_method: str = "z",
    invert_y: bool = True,
    device: torch.device = torch.device("cpu"),
    offset: T.Tuple[float, float] = (0, 0),
    allow_trellis_cam_dist_skew: bool = False,
    # xyz_w_for_cam_transformation: torch.Tensor | None = None,
    rng: np.random.Generator | None = None,
):
    """Creates the camera poses for image-conditioning generative model training.
    Note, the FOV and radius are bonded.

    See https://github.com/microsoft/TRELLIS/blob/6b0d64751ad54d9c32d7b05fec482eb29178f56f/dataset_toolkits/render_cond.py#L30-L46

    Further, TRELLIS uses 24 views by default:
    - https://github.com/microsoft/TRELLIS/blob/6b0d64751ad54d9c32d7b05fec482eb29178f56f/dataset_toolkits/render_cond.py#L75
    - https://github.com/microsoft/TRELLIS/blob/6b0d64751ad54d9c32d7b05fec482eb29178f56f/DATASET.md#step-9-render-image-conditions

    Args:
        width_px / height_px:
            number of pixels for width / height
        n_views:
            number of cameras to sample
        up_method:
            'y': up = (0, 1, 0)
        invert_y:
            whether to invert the y axis (since image coordinate is x to right y to down)
        allow_trellis_cam_dist_skew:
            bool. If True, this skews the camera layout to focus more on the upper hemisphere
            (but this also affects the rest two axis).
        xyz_w_for_cam_transformation:
            torch.Tensor, a set of points to be used for scaling camera pose

    Returns:
        Camera:
            (n, 4, 4)
    """
    yaws = []
    pitches = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(n_views):
        y, p = rigid_motion.sphere_hammersley_sequence(
            i, n_views, offset, allow_trellis_cam_dist_skew=allow_trellis_cam_dist_skew
        )
        yaws.append(y)
        pitches.append(p)

    fov_min, fov_max = 10, 70

    # converts the FOV limits into corresponding radii (distance from the camera to the object/scene).
    # The formula is based on the geometry of projecting an cube
    # (since sqrt(3) / 2 is half the diagonal of a cube with unit length ).
    #
    # NOTE: originally, TRELLIS uses sqrt(3) / 2 as the length to optimize since its object is normalized in [-0.5, 0.5].
    # However, we normalize the object into [-1, 1]. Thus, we need to optimize the length of sqrt(3).
    length_to_use = np.sqrt(3)
    radius_min = length_to_use / np.sin(fov_max / 2 / 180 * np.pi)
    radius_max = length_to_use / np.sin(fov_min / 2 / 180 * np.pi)

    # ensures the resulting radii are distributed uniformly in solid angle
    # (important for unbiased random sampling in spherical coordinates).
    k_min = 1 / radius_max**2
    k_max = 1 / radius_min**2

    if rng is None:
        rng = np.random.default_rng()
    ks = rng.uniform(low=k_min, high=k_max, size=(n_views,))

    radius = [1 / np.sqrt(k) for k in ks]
    fov = [2 * np.arcsin(length_to_use / r) for r in radius]

    # convert
    # views = [{"yaw": y, "pitch": p, "radius": r, "fov": f} for y, p, r, f in zip(yaws, pitches, radius, fov)]

    yaws = torch.FloatTensor(yaws)  # (n,)
    pitches = torch.FloatTensor(pitches)  # (n,)
    radius = torch.FloatTensor(radius)  # (n,)
    fov = torch.FloatTensor(fov)  # (n,)

    # https://github.com/microsoft/TRELLIS/blob/f17fdf12d8f17a6a09225f01756d141285dc848f/dataset_toolkits/blender_script/render.py#L459-L463
    x = torch.cos(pitches) * torch.cos(yaws)
    y = torch.cos(pitches) * torch.sin(yaws)
    z = torch.sin(pitches)

    pinhole_location_w = radius[:, None] * torch.stack((x, y, z), dim=-1)  # (n, 3)
    assert pinhole_location_w.shape[1] == 3, f"{pinhole_location_w.shape=}"

    H_c2w = rigid_motion.sphere_camera_poses_sampling_postprocessing(
        pinhole_location_w=pinhole_location_w,
        up_method=up_method,
        invert_y=invert_y,
    )  # (n, 4, 4)

    assert H_c2w.shape == (n_views, 4, 4), f"{H_c2w.shape=}"

    # if xyz_w_for_cam_transformation is not None:
    #     H_c2w_list = []
    #     for tmp_i in range(n_views):
    #         tmp_H_c2w = scale_camera_for_img_cond_ldm_cam_transformation(
    #             xyz_w=xyz_w_for_cam_transformation, H_c2w=H_c2w[tmp_i]
    #         )
    #         H_c2w_list.append(tmp_H_c2w)
    #     H_c2w = torch.stack(H_c2w_list, dim=0)

    intrinsic = render.derive_camera_intrinsics(
        width_px=width_px,
        height_px=height_px,
        fov=fov / np.pi * 180,  # radian -> degree
    ).float()  # (n, 3, 3)

    camera = structures.Camera(
        H_c2w=H_c2w.unsqueeze(0),  # (1, q, 4, 4)
        intrinsic=intrinsic.unsqueeze(0),  # (1, q, 3, 3)
        width_px=width_px,
        height_px=height_px,
    )

    return camera


def generate_trellis_eval_cameras_for_img_cond_ldm(
    *,
    width_px: int,
    height_px: int,
    up_method="z",
    invert_y: bool = True,
    purpose: str = "fid",
    radius: float = 4.0,
    device: torch.device = torch.device("cpu"),
):
    """
    See Sec. C.1 in https://arxiv.org/abs/2412.01506
    - for conditioning: one random view
    - for evaluation: depending on "purpose"

    Args:
        width_px / height_px:
            number of pixels for width / height
        up_method:
            'z': up = (0, 0, 1)
        invert_y:
            whether to invert the y axis (since image coordinate is x to right y to down)
        purpose:
            choices: [fid, clip].
            - if purpose == "fid": we produce 4 cameras with yaws {0, 90, 180, 270}, and a pitch angle of 30.
            - if purpose == "clip": we produce 8 cameras with yaws equally placed in [0, 360] with interval 45,
                                    and a pitch angle of 30.

    Returns:
        Camera:
            (n, 4, 4)
    """

    if purpose == "fid":
        yaws = [0, 90, 180, 270]
    elif purpose == "clip":
        yaws = np.linspace(0, 360, 9)[:-1].tolist()
    else:
        raise ValueError(f"{purpose=}")

    radius = [radius for _ in yaws]
    fov = [40 for _ in yaws]

    n_views = len(yaws)

    yaws = torch.FloatTensor(yaws) / 180 * np.pi  # (n,)
    radius = torch.FloatTensor(radius)  # (n,)
    fov = torch.FloatTensor(fov)  # (n,)

    intrinsic = render.derive_camera_intrinsics(
        width_px=width_px,
        height_px=height_px,
        fov=fov,  # degree
    ).float()  # (n, 3, 3)

    cam_dict = {}

    # - we use sphere cameras that are placed at pitch=0 as conditioning image
    # - we use the cameras specified in TRELLIS paper as evalaution cameras
    # We reload the mode name of "camera_regular" and "camera_random" to re-use functions
    for tmp_mode, tmp_pitch in [("regular", 0), ("random", 30)]:
        tmp_pitches = [tmp_pitch for _ in yaws]
        tmp_pitches = torch.FloatTensor(tmp_pitches) / 180 * np.pi  # (n,)

        # https://github.com/microsoft/TRELLIS/blob/f17fdf12d8f17a6a09225f01756d141285dc848f/dataset_toolkits/blender_script/render.py#L459-L463
        tmp_x = torch.cos(tmp_pitches) * torch.cos(yaws)
        tmp_y = torch.cos(tmp_pitches) * torch.sin(yaws)
        tmp_z = torch.sin(tmp_pitches)

        tmp_pinhole_location_w = radius[:, None] * torch.stack((tmp_x, tmp_y, tmp_z), dim=-1)  # (n, 3)
        assert tmp_pinhole_location_w.shape[1] == 3, f"{tmp_pinhole_location_w.shape=}"

        tmp_H_c2w = rigid_motion.sphere_camera_poses_sampling_postprocessing(
            pinhole_location_w=tmp_pinhole_location_w,
            up_method=up_method,
            invert_y=invert_y,
        )  # (n, 4, 4)

        assert tmp_H_c2w.shape == (n_views, 4, 4), f"{tmp_H_c2w.shape=}"

        tmp_camera = structures.Camera(
            H_c2w=tmp_H_c2w.unsqueeze(0),  # (1, q, 4, 4)
            intrinsic=intrinsic.unsqueeze(0),  # (1, q, 3, 3)
            width_px=width_px,
            height_px=height_px,
        )

        cam_dict[f"num_images_{tmp_mode}"] = n_views
        cam_dict[f"camera_{tmp_mode}"] = tmp_camera

    return cam_dict


def sample_positions_on_circle_around_pivot(
    *,
    pivot: T.Union[torch.Tensor, list, tuple],
    angle_to_pivot: float,
    n_views: int,
    radius: float = 1.0,
    input_degree: bool = False,
    start_angle: float = 0.0,
) -> torch.Tensor:
    """
    Sample n_views points uniformly along the small circle on a sphere of radius,
    consisting of points whose angle to pivot is angle_to_pivot.

    Essentially, for each returned sampled point X, it has the following constraints:
        normalized(X) \cdot normalized(pivot) = cos(angle_to_pivot).

    Geometrically, we will have a cone whose
    - apex is the origin;
    - foot of the perpendicular is the pivot;
    - half-vertical angle is angle_to_pivot.

    The intersection between this cone and the sphere formulates a circle. We uniformly sample from this circle.

    Args:
        pivot :
            (3,) torch.Tensor or array-like
            Camera position vector (need not be unit; only its direction is used).
            Device/dtype are taken from this if it's a tensor.
        angle_to_pivot:
            float
            Angle between C and each sampled point. Radians by default, set degrees=True if in degrees.
        n_views:
            int
            Number of points to sample (N >= 1).
        radius:
            float
            Sphere radius (default 1.0).
        input_degree:
            bool
            If True, interpret X (and start_angle) in degrees.
        start_angle:
            float
            Phase offset for the sampling parameter θ (same units as pivot).

    Returns:
        sampled_pos:
            (N, 3) torch.Tensor
            Sampled 3D points on the sphere.
    """
    # Prepare tensor & dtype/device
    if not torch.is_tensor(pivot):
        pivot = torch.tensor(pivot, dtype=torch.get_default_dtype())
    pivot = pivot.to(dtype=torch.get_default_dtype())
    if pivot.numel() != 3:
        raise ValueError("C must be a 3D vector.")
    device = pivot.device

    # Normalize pivot direction
    pivot_normalized = torch.nn.functional.normalize(pivot, p=2.0, dim=-1)

    # Angle handling
    if input_degree:
        angle_to_pivot = torch.deg2rad(torch.tensor(angle_to_pivot, dtype=pivot.dtype, device=device)).item()
        start_angle = torch.deg2rad(torch.tensor(start_angle, dtype=pivot.dtype, device=device)).item()
    angle_to_pivot = torch.tensor(angle_to_pivot, dtype=pivot.dtype, device=device)
    start_angle = torch.tensor(start_angle, dtype=pivot.dtype, device=device)

    eps = torch.tensor(1e-12, dtype=pivot.dtype, device=device)
    if torch.abs(torch.sin(angle_to_pivot)) < eps:
        # Degenerate cases: small circle collapses to a point
        sign = 1.0 if torch.cos(angle_to_pivot) > 0 else -1.0
        sampled_pos_unit_sphere = sign * pivot_normalized
        sampled_pos = (radius * sampled_pos_unit_sphere).expand(n_views, 3)  # (n, 3)
    else:
        # Build orthonormal basis {u, v, c_hat}
        up = torch.tensor([0.0, 0.0, 1.0], dtype=pivot.dtype, device=device)
        if torch.abs(torch.dot(up, pivot_normalized)) > 0.999:
            # parallel
            up = torch.tensor([0.0, 1.0, 0.0], dtype=pivot.dtype, device=device)

        # project 'up' onto plane that is perpendicular to pivot)
        up_dot_pivot = torch.dot(up, pivot_normalized)  # projection of up vector to the pivot line
        u = up - up_dot_pivot * pivot_normalized  # the projection of up vector that is perpendicular to the pivot line
        u = u / torch.linalg.norm(u)

        # v = c_hat x u
        # (u, v) is the basis vector in the plane that is perpendicular to the pivto line
        v = torch.linalg.cross(pivot_normalized, u)  # unit if u and c_hat are unit & orthogonal

        # Uniform arclength samples via uniform in [0, 2 pi]
        thetas_for_plane = (
            start_angle + 2.0 * torch.pi * torch.arange(n_views, dtype=pivot.dtype, device=device) / n_views
        )
        cos_on_unit_plane = torch.cos(thetas_for_plane).unsqueeze(1)  # (N, 1)
        sin_on_unit_plane = torch.sin(thetas_for_plane).unsqueeze(1)  # (N, 1)
        pos_on_unit_plane = cos_on_unit_plane * u + sin_on_unit_plane * v

        # these marks that for a unit sphere, for any point on the contour circle,
        # 1) length along the pivot line;
        # 2) length prependicular the pivot line, i.e., the radius of the perpendicular plane.
        radius_plane = torch.sin(angle_to_pivot)
        len_on_pivot = torch.cos(angle_to_pivot)

        sampled_pos_unit_sphere = radius_plane * pos_on_unit_plane + len_on_pivot * pivot_normalized
        sampled_pos = radius * sampled_pos_unit_sphere  # (n, 3)

    return sampled_pos


def generate_our_eval_cameras_for_img_cond_ldm(
    *,
    width_px: int,
    height_px: int,
    up_method="z",
    invert_y: bool = True,
    # include_eval_clip: bool = False,
    device: torch.device = torch.device("cpu"),
):
    """
    See Sec. C.1 in https://arxiv.org/abs/2412.01506
    - for conditioning: one random view
    - for evaluation: depending on "purpose"

    Args:
        width_px / height_px:
            number of pixels for width / height
        up_method:
            'z': up = (0, 0, 1)
        invert_y:
            whether to invert the y axis (since image coordinate is x to right y to down)
        purpose:
            choices: [fid, clip].
            - if purpose == "fid": we produce 4 cameras with yaws {0, 90, 180, 270}, and a pitch angle of 30.
            - if purpose == "clip": we produce 8 cameras with yaws equally placed in [0, 360] with interval 45,
                                    and a pitch angle of 30.

    Returns:
        a dict of Camera:
            each camera is of (n, 4, 4), with the following camera types:
            - regular: we use sphere/regular cameras that are placed at pitch=0 as conditioning image
            - random camera: we use randomly-sampled camera as the conditioning
            - eval_fid_point: we use follow
            - eval_fid: we use the cameras specified in TRELLIS paper as FID evalaution cameras
            - eval_contour_XXX: cameras to be evaluated.
              The angles between these cameras and the conditioning cameras are XXX degress.
    """

    common_radius = 4.0
    common_fov = 40

    # intrinsic = render.derive_camera_intrinsics(
    #     width_px=width_px,
    #     height_px=height_px,
    #     fov=common_fov,  # degree
    # )  # (n, 3, 3)
    # intrinsic = torch.FloatTensor(intrinsic)  # (n, 3, 3)

    cam_dict = {}

    # get regular conditioning camera and eval_fid cameras
    for tmp_mode in ["regular", "random", "eval_fid", "eval_clip"]:
        if tmp_mode in ["eval_fid"]:
            tmp_yaws = [0, 90, 180, 270]
            tmp_pitches = [30 for _ in tmp_yaws]
        elif tmp_mode in ["eval_clip"]:
            tmp_yaws = np.linspace(0, 360, 9)[:-1].tolist()
            tmp_pitches = [30 for _ in tmp_yaws]
        elif tmp_mode in ["regular"]:
            tmp_yaws = [0, 90, 180, 270]
            tmp_pitches = [0 for _ in tmp_yaws]
        elif tmp_mode in ["random"]:
            tmp_yaws = np.random.uniform(low=0, high=1, size=(1,)) * 360
            tmp_pitches = np.random.uniform(low=0, high=1, size=tmp_yaws.shape[0]) * 90  # upper hemisphere
        else:
            raise ValueError(f"{tmp_mode=}")

        tmp_n_views = len(tmp_yaws)

        tmp_yaws = torch.FloatTensor(tmp_yaws) / 180 * np.pi  # (n,)
        tmp_radius = torch.FloatTensor([common_radius for _ in tmp_yaws])  # (n,)
        tmp_fov = torch.FloatTensor([common_fov for _ in tmp_yaws])  # (n,)
        tmp_pitches = torch.FloatTensor(tmp_pitches) / 180 * np.pi  # (n,)

        tmp_intrinsic = render.derive_camera_intrinsics(
            width_px=width_px,
            height_px=height_px,
            fov=tmp_fov,  # degree
        ).float()  # (n, 3, 3)

        # https://github.com/microsoft/TRELLIS/blob/f17fdf12d8f17a6a09225f01756d141285dc848f/dataset_toolkits/blender_script/render.py#L459-L463
        tmp_x = torch.cos(tmp_pitches) * torch.cos(tmp_yaws)
        tmp_y = torch.cos(tmp_pitches) * torch.sin(tmp_yaws)
        tmp_z = torch.sin(tmp_pitches)

        tmp_pinhole_location_w = tmp_radius[:, None] * torch.stack((tmp_x, tmp_y, tmp_z), dim=-1)  # (n, 3)
        assert tmp_pinhole_location_w.shape[1] == 3, f"{tmp_pinhole_location_w.shape=}"

        tmp_H_c2w = rigid_motion.sphere_camera_poses_sampling_postprocessing(
            pinhole_location_w=tmp_pinhole_location_w,
            up_method=up_method,
            invert_y=invert_y,
        )  # (n, 4, 4)

        assert tmp_H_c2w.shape == (tmp_n_views, 4, 4), f"{tmp_H_c2w.shape=}"
        assert tmp_intrinsic.shape == (tmp_n_views, 3, 3), f"{tmp_intrinsic.shape=}"

        tmp_camera = structures.Camera(
            H_c2w=tmp_H_c2w.unsqueeze(0),  # (1, q, 4, 4)
            intrinsic=tmp_intrinsic.unsqueeze(0),  # (1, q, 3, 3)
            width_px=width_px,
            height_px=height_px,
        )

        cam_dict[f"num_images_{tmp_mode}"] = tmp_n_views
        cam_dict[f"camera_{tmp_mode}"] = tmp_camera

    # cameras on the contour
    n_views_contour = 8
    for tmp_contour_deg in [5, 10, 20, 30]:
        for tmp_mode in ["regular", "random"]:
            tmp_pivot_H_c2w = cam_dict[f"camera_{tmp_mode}"].H_c2w
            assert (tmp_pivot_H_c2w.ndim == 4) and (tmp_pivot_H_c2w.shape[0] == 1), (
                f"{tmp_mode=}, {tmp_pivot_H_c2w.shape=}"
            )

            tmp_pivot = tmp_pivot_H_c2w[0, 0][:3, 3]
            tmp_sampled_pos = sample_positions_on_circle_around_pivot(
                pivot=tmp_pivot,
                angle_to_pivot=tmp_contour_deg,
                n_views=n_views_contour,
                radius=common_radius,
                input_degree=True,
                start_angle=0.0,
            )
            assert tmp_sampled_pos.shape == (n_views_contour, 3), f"{tmp_sampled_pos.shape=}"

            tmp_H_c2w = rigid_motion.sphere_camera_poses_sampling_postprocessing(
                pinhole_location_w=tmp_sampled_pos,
                up_method=up_method,
                invert_y=invert_y,
            )  # (n, 4, 4)

            tmp_intrinsic = render.derive_camera_intrinsics(
                width_px=width_px,
                height_px=height_px,
                fov=torch.FloatTensor([common_fov for _ in range(n_views_contour)]),  # degree
            ).float()  # (n, 3, 3)

            assert tmp_H_c2w.shape == (n_views_contour, 4, 4), f"{tmp_H_c2w.shape=}"
            assert tmp_intrinsic.shape == (tmp_n_views, 3, 3), f"{tmp_intrinsic.shape=}"

            tmp_camera = structures.Camera(
                H_c2w=tmp_H_c2w.unsqueeze(0),  # (1, q, 4, 4)
                intrinsic=tmp_intrinsic.unsqueeze(0),  # (1, q, 3, 3)
                width_px=width_px,
                height_px=height_px,
            )

            tmp_cam_type = f"eval_cond_{tmp_mode}_contour_{tmp_contour_deg}"
            cam_dict[f"num_images_{tmp_cam_type}"] = n_views_contour
            cam_dict[f"camera_{tmp_cam_type}"] = tmp_camera

    # for Point-FID evaluations
    n_views_eval_fid_point = 20
    # We overload this camera type to re-use functions in create_data_udf_for_train
    eval_fid_point_cam_type = "sphere_for_geometry"
    eval_fid_point_cam_dict = get_two_different_sphere_cameras(
        num_views=n_views_eval_fid_point,
        r=3.5,
        fov=40,
        width_px=width_px,
        height_px=height_px,
        up_method=up_method,
        invert_y=invert_y,
    )
    eval_fid_point_purpose = "gt"
    if eval_fid_point_purpose == "gt":
        eval_fid_point_cam: structures.Camera = eval_fid_point_cam_dict["camera_1"]
    elif eval_fid_point_purpose == "pred":
        eval_fid_point_cam: structures.Camera = eval_fid_point_cam_dict["camera_2"]
    else:
        raise ValueError(f"{eval_fid_point_purpose=}")
    cam_dict[f"num_images_{eval_fid_point_cam_type}"] = n_views_eval_fid_point
    cam_dict[f"camera_{eval_fid_point_cam_type}"] = eval_fid_point_cam

    return cam_dict


def generate_trellis_eval_cameras_for_img_cond_ldm_2(*, width_px: int, height_px: int, up_method="z"):
    """
    See Sec. C.1 in https://arxiv.org/abs/2412.01506
    - for conditioning: one random view
    - for evaluation: yaw angles of {0, 90, 180, 270}, and a pitch angle of 30.
    """
    #  See Sec. C.1 in https://arxiv.org/abs/2412.01506
    # - for conditioning: one random view
    # - for evaluation: yaw angles of {0, 90, 180, 270}, and a pitch angle of 30.

    yaw_degs = np.array([0, 90, 180, 270])
    pitch_deg = 30

    radius = 4.0
    fov_deg = 40

    if up_method == "z":
        # This definition of [x, y, z] differs from what TRELLIS uses.
        # This will not have any effects for now since the yaw is equally distributed in [0, 360].
        # However, it will produce different results when the distribution is not uniform.
        #
        # See https://github.com/microsoft/TRELLIS/blob/f17fdf12d8f17a6a09225f01756d141285dc848f/dataset_toolkits/blender_script/render.py#L459-L463
        pinhole_location = np.stack(
            [
                radius * np.sin((90 - pitch_deg) / 180 * np.pi) * np.sin(yaw_degs / 180 * np.pi),
                radius * np.sin((90 - pitch_deg) / 180 * np.pi) * np.cos(yaw_degs / 180 * np.pi),
                radius * np.cos((90 - pitch_deg) / 180 * np.pi) * np.ones_like(yaw_degs),
            ],
            axis=-1,
        ).astype(np.float32)
        up_w = np.repeat(np.array([[0, 0, 1]]), yaw_degs.shape[0], axis=0).astype(np.float32)
    else:
        raise NotImplementedError(f"{up_method=}")
    look_at = np.repeat(np.zeros((1, 3)), yaw_degs.shape[0], axis=0).astype(np.float32)
    H_c2w = torch.FloatTensor(rigid_motion.get_H_c2w_lookat(pinhole_location, look_at_w=look_at, up_w=up_w))

    intrinsic = render.derive_camera_intrinsics(width_px=width_px, height_px=height_px, fov=fov_deg)
    intrinsic = torch.FloatTensor(np.repeat(intrinsic[None], yaw_degs.shape[0], axis=0))

    camera = structures.Camera(
        H_c2w=H_c2w.unsqueeze(0),  # (1, q, 4, 4)
        intrinsic=intrinsic.unsqueeze(0),  # (1, q, 3, 3)
        width_px=width_px,
        height_px=height_px,
    )

    return camera


def get_camera_dicts_for_blender(
    *,
    num_regular_images,
    r,
    fov,
    width_px,
    height_px,
    num_random_images,
    min_random_radius,
    max_random_radius,
    random_lookat_r,
    max_fov,
    min_fov,
    regular_camera_sampling_type: str = "circular",
):
    def _udpate_camera_dicts(
        *,
        camera_dicts: T.List[T.Dict[str, T.Any]],  # list of (q,) cam_dict
        H_c2w: torch.Tensor,  # (q, 4, 4)
        intrinsic: torch.Tensor,  # (q, 3, 3)
        width_px: int,
        height_px: int,
    ) -> T.List[T.Dict[str, T.Any]]:
        """Given (q, 4, 4), convert o3d camera to blender camera"""
        n = H_c2w.shape[0]
        odict = blender_open3d_utils.convert_open3d_camera_to_blender(
            H_c2w=H_c2w.detach().cpu().numpy(),  # (n, 4, 4)
            intrinsic=intrinsic.expand(n, 3, 3).detach().cpu().numpy(),  # (n, 3, 3)
            width_px=width_px,
            height_px=height_px,
        )
        for iq in range(n):
            cdict = dict(
                H_c2w=odict["H_c2w"][iq],  # (4, 4)
                intrinsic=odict["intrinsic"][iq],  # (3, 3)
                width_px=width_px,
                height_px=height_px,
            )
            camera_dicts.append(cdict)
        return camera_dicts

    intrinsic = torch.from_numpy(
        render.derive_camera_intrinsics(
            width_px=width_px,
            height_px=height_px,
            fov=fov,
        )
    )  # (3, 3)

    camera_dicts = []
    cam_name_dict = OrderedDict()  # name -> num_q (int)

    if regular_camera_sampling_type == "circular":
        n_images_per_plane = int(np.ceil(num_regular_images / 3))

        # generate cameras on a circle on a plane
        for tmp_plane_name, tmp_normal_w in (
            ("xz", [0.0, 1.0, 0.0]),  # xz-plane
            ("yz", [1.0, 0.0, 0.0]),  # yz-plane
            ("xy", [0.0, 0.0, 1.0]),  # xy-plane
        ):
            H_c2w = rigid_motion.generate_circular_camera_poses(
                n=n_images_per_plane, r=r, normal_w=tmp_normal_w
            )  # (n, 4, 4)
            camera_dicts = _udpate_camera_dicts(
                camera_dicts=camera_dicts, H_c2w=H_c2w, intrinsic=intrinsic, width_px=width_px, height_px=height_px
            )
            cam_name_dict[tmp_plane_name] = int(H_c2w.shape[0])
    elif regular_camera_sampling_type == "sphere":
        # num_regular_images = num_regular_images
        if False:
            # Even with solid angle, the camera poses are not evenly distributed on the sphere.
            H_c2w = rigid_motion.generate_uniform_camera_poses_wrt_sphere_solid_angle(
                n=num_regular_images, r=r
            )  # (n, 4, 4)
        else:
            H_c2w = rigid_motion.generate_uniform_camera_poses_with_golden_spiral(
                n=num_regular_images, r=r
            )  # (n, 4, 4)

        camera_dicts = _udpate_camera_dicts(
            camera_dicts=camera_dicts, H_c2w=H_c2w, intrinsic=intrinsic, width_px=width_px, height_px=height_px
        )
        cam_name_dict[regular_camera_sampling_type] = int(H_c2w.shape[0])
    else:
        raise NotImplementedError(f"{regular_camera_sampling_type=}")

    # random images
    H_c2w = (
        rigid_motion.generate_random_camera_poses_lookat(
            n=num_random_images,
            pinhole_min_r=min_random_radius,
            pinhole_max_r=max_random_radius,
            lookat_r=random_lookat_r,
        )
        .detach()
        .cpu()
        .numpy()
    )  # (n, 4, 4)

    random_fovs = np.random.rand(num_random_images) * (max_fov - min_fov) + min_fov  # (n,) degree
    intrinsic = render.derive_camera_intrinsics(
        width_px=width_px,
        height_px=height_px,
        fov=random_fovs,
    )  # (n, 3, 3)
    for ii in range(H_c2w.shape[0]):
        mdict = blender_open3d_utils.convert_open3d_camera_to_blender(
            H_c2w=H_c2w[ii],
            intrinsic=intrinsic[ii],
            width_px=width_px,
            height_px=height_px,
        )
        camera_dicts.append(mdict)
    cam_name_dict["random"] = H_c2w.shape[0]

    return camera_dicts, cam_name_dict


def sample_pcd_rgbd_from_mesh_with_blender(
    *,
    mesh_filename: str,
    out_dir: str,
    out_dir_rgbd: str = None,
    out_dir_pcd_rgbd_visibility_dict: T.Union[T.Dict[float, str], None] = None,
    light_type: str = "SUN",
    num_lights: int = 8,
    min_light_energy: float = 0.0,  # remember to adjust when num_lights change
    max_light_energy: float = 3.0,  # remember to adjust when num_lights change
    num_cells: int = 512,  # number of cells per side for voxel sampling
    width_px: int = 448,  # 14 x 32
    height_px: int = 448,  # 14 x 32
    # circular camera
    num_regular_images: int = 10,
    fov: float = 40.0,  # degree
    circular_radius: float = 3.5,  # meter
    # random camera
    num_random_images: int = 30,
    min_fov: float = 40.0,  # degree
    max_fov: float = 60.0,  # degree
    min_random_radius: float = 3,
    max_random_radius: float = 4,
    random_lookat_r: float = 0.25,
    mesh_rel_dir: str = None,
    background_color: float = 1,
    overwrite: bool = False,
    save_attr_names: T.List[str] = None,
    pcd_save_version: int = 2,
    save_np_dtype: np.dtype = np.float32,
    normalized_mesh_fname: str = "blender_normalized_mesh.ply",
    normalization_info_fname: str = "config_after_blender_normalization.json",
    seed: int = 0,
    pcd_save_chunk_size: int = 100_000,
    flag_debug: bool = False,
    flag_return_xyz_w: bool = False,
    regular_camera_sampling_type: str = "circular",
    flag_save_space: bool = False,
    **kwargs,
):
    """
    Render rgbd images with blender then resample point cloud from the
    rgbd point cloud.

    Strategy:
    1. normalize the mesh to fit [-1, 1] bounding box
    2. randomly add lighting in the scene
    3. sample randomly the camera in a sphere to avoid occusion
    4. sample camera on multi-plane circles
    5. sample camera in a spherical shell

    Args:
        out_dir_rgbd:
            if not None, rgbd will be saved in the folder instead of out_dir

        pcd_save_version:
            1: single npz file saving xyz_w (n, 3), rgb (n, 3), normal_w (n, 3)
            2: multiple npz files, each saving a subset of xyz_w (m, 3) or rgb (n, 3), or normal_w (n, 3),
                xyz, rgb, normal are saved separately
    """

    _set_seed(seed)

    if not overwrite:
        assert (not os.path.exists(out_dir)) or (not os.listdir(out_dir))

    if out_dir_rgbd is None:
        out_dir_rgbd = out_dir

    # compile json file (mesh, lighting, camera)
    scene_dict = dict()

    # mesh
    mdict = dict(
        name="mesh",
        filename=mesh_filename,
        normalize_first=True,  # [-1, 1] aabb box
        H_c2w=np.eye(4),  # no rotation after normalization
        scale=np.array([1.0, 1.0, 1.0]),  # no scaling after normalization
    )
    scene_dict["meshes"] = [mdict]

    # lighting
    light_dicts = []
    for il in range(num_lights):
        if light_type == "SUN":
            # note that the light is toward -z
            # but since we just wnat random light direction, we do not care
            H_c2w = rigid_motion.get_H_c2w_lookat(
                pinhole_location_w=(0, 0, 0.0),
                look_at_w=rigid_motion.get_random_direction().astype(np.float32),  # (3,)
                up_w=(0, 1, 0.0),
                invert_y=False,
            )  # (4, 4)
            energy = float(np.random.rand()) * (max_light_energy - min_light_energy) + min_light_energy

            mdict = dict(
                name=f"light {il}",
                light_type=light_type,
                H_c2w=H_c2w.tolist(),
                energy=energy,
                use_shadow=False,
                specular_factor=1.0,
            )
        elif light_type == "diffuse":
            mdict = dict(
                name=f"light {il}",
                light_type=light_type,
                color=[1.0, 1.0, 1.0, 1.0],
                strength=1.0,
            )
        else:
            raise NotImplementedError
        light_dicts.append(mdict)
    scene_dict["lighting"] = light_dicts

    # camera
    camera_dicts, cam_name_dict = get_camera_dicts_for_blender(
        num_regular_images=num_regular_images,
        r=circular_radius,
        fov=fov,
        width_px=width_px,
        height_px=height_px,
        num_random_images=num_random_images,
        min_random_radius=min_random_radius,
        max_random_radius=max_random_radius,
        random_lookat_r=random_lookat_r,
        max_fov=max_fov,
        min_fov=min_fov,
        regular_camera_sampling_type=regular_camera_sampling_type,
    )

    scene_dict["cameras"] = camera_dicts

    # save the scene config
    os.makedirs(out_dir, exist_ok=True)
    json_filename = os.path.join(out_dir, "scene_config.json")
    with open(json_filename, "w") as f:
        json.dump(scene_dict, f, indent=2, cls=json_utils.NumpyJsonEncoder)

    # render
    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        # if True:
        #     tmp_dir = 'tmp_blender'
        #     print(f'tmp_dir = {tmp_dir}')

        tmp_dir = os.path.abspath(tmp_dir)

        # save config to tmp file (again) just to keep config
        json_filename = os.path.join(tmp_dir, "config.json")
        with open(json_filename, "w") as f:
            json.dump(scene_dict, f, indent=2, cls=json_utils.NumpyJsonEncoder)

        # render with blender
        tmp_out_dir = os.path.join(tmp_dir, "out")
        blender_cmd = blender_rendering_utils.get_blender_exe()
        blender_script = blender_rendering_utils.get_blender_utils_path()
        blender_log_fname = f"blender_{pathlib.Path(mesh_filename).stem}.log"
        blender_log_f = os.path.join(tmp_dir, blender_log_fname)
        cmd = (
            f"{blender_cmd} --background --log-level 1 --python {blender_script} -- "
            f"--filename {json_filename} --out_dir {tmp_out_dir} "
            f"--normalized_mesh_fname {normalized_mesh_fname} "
            f"--normalization_info_fname {normalization_info_fname} "
            f"--debug 0 > {blender_log_fname}"
        )
        print(cmd)
        os.system(cmd)

        # copy normlization information to disk
        out_dir = pathlib.Path(out_dir)
        print(f"\nCopying data from {tmp_out_dir=} to {out_dir=}\n")
        shutil.copyfile(pathlib.Path(tmp_out_dir) / normalized_mesh_fname, out_dir / normalized_mesh_fname)
        shutil.copyfile(pathlib.Path(tmp_out_dir) / normalization_info_fname, out_dir / normalization_info_fname)

        # read and save the output
        rgbd_dict = dict()  # name -> rgbd
        from_idx = 0
        for name in cam_name_dict:
            rgbd = blender_plib_utils.read_blender_results_to_rgbd(
                result_dir=tmp_out_dir,
                from_idx=from_idx,
                to_idx=from_idx + cam_name_dict[name],
                use_srgb=True,
                flag_save_space=flag_save_space,
                debug=flag_debug,
            )  # (b=1, q, h, w)
            assert rgbd.rgb.size(1) == cam_name_dict[name]
            rgbd_dict[name] = rgbd  # (b=1, q, h, w)
            from_idx += cam_name_dict[name]

        if flag_debug:
            # check average processing time

            # tmp_tgt_log_f = pathlib.Path(os.getcwd()) / blender_log_fname
            # if tmp_tgt_log_f.exists():
            #     os.remove(tmp_tgt_log_f)
            # shutil.copyfile(blender_log_f, tmp_tgt_log_f)

            with open(blender_log_fname, "r") as f:
                tmp_log_lines = [_.strip() for _ in f.readlines()]
            # e.g., Time: 00:01.82 (Saving: 00:00.00)
            tmp_time_lines = [_ for _ in tmp_log_lines if "(Saving:" in _]
            # print(f"\n\n{len(tmp_time_lines)=}\n\n")

            def _time_line_to_float(time_line):
                # e.g., Time: 00:01.82 (Saving: 00:00.00)
                time_str = time_line.split("(Saving:")[0].strip().split("Time:")[1].strip()
                # print(f"\n{time_line=}, {time_str=}\n")
                minutes, seconds = time_str.split(":")
                return float(minutes) + float(seconds)

            tmp_time_floats = [_time_line_to_float(_) for _ in tmp_time_lines]
            tmp_time_mean = np.mean(tmp_time_floats)
            tmp_time_std = np.std(tmp_time_floats)
            print(
                f"\n\nBlender rendering time {tmp_time_mean} +- {tmp_time_std}, "
                f"averaged over {len(tmp_time_floats)} renderings.\n\n"
            )

        # save blender log file for future check
        os.makedirs(out_dir_rgbd, exist_ok=True)
        save_log_f = pathlib.Path(out_dir_rgbd) / blender_log_fname
        print(f"\nmoving {blender_log_fname=} to {save_log_f=}\n")
        shutil.move(blender_log_fname, save_log_f)

    # construct point cloud by combining rgbd images
    pcd_dict = dict()  # name -> pcd
    for name in rgbd_dict:
        # print(f'{name}: {rgbd_dict[name].hit_map.float().mean()}')
        assert rgbd_dict[name].depth.size(0) == 1
        pcd = rgbd_dict[name].get_pcd()
        # print(f'{name} pcd: {pcd.xyz_w.shape}')
        pcd_dict[name] = pcd  # (b=1, n)
    pcd = structures.PointCloud.cat(list(pcd_dict.values()), dim=1)  # (b=1, n)
    # print(f'ori xyz_w.shape: {pcd.xyz_w.shape}')

    # remove points that are outside [-1, 1] aabb, and are not finite
    xyz_w = pcd.extract_valid_attr(
        arr=pcd.xyz_w,
        bidx=0,
    )  # (n, 3)
    n, _3xyz = xyz_w.shape

    bbox_eps = 0.01
    vmask = torch.logical_and(
        (xyz_w >= -1 - bbox_eps).all(dim=-1),  # (n,)
        (xyz_w <= 1 + bbox_eps).all(dim=-1),  # (n,)
    )  # (n,)
    # print(f'after xyz_w, vmask: {vmask.shape} ({vmask.float().mean()})')
    normal_w = pcd.extract_valid_attr(
        arr=pcd.normal_w,
        bidx=0,
    )  # (n, 3)
    assert xyz_w.size(0) == normal_w.size(0)
    vmask = torch.logical_and(
        vmask,  # (n,)
        normal_w.isfinite().all(dim=-1),  # (n,)
    )  # (n,)
    # print(f'after normal, vmask: {vmask.shape} ({vmask.float().mean()})')
    rgb = pcd.extract_valid_attr(
        arr=pcd.rgb,
        bidx=0,
    )  # (n, 3)
    assert xyz_w.size(0) == rgb.size(0)
    vmask = torch.logical_and(
        vmask,  # (n,)
        rgb.isfinite().all(dim=-1),  # (n,)
    )  # (n,)
    # print(f'after rgb, vmask: {vmask.shape} ({vmask.float().mean()})')
    # print(f'vmask.shape = {vmask.shape} (total = {vmask.sum()})')
    # print(f'rgb.shape = {rgb.shape} ({rgb[vmask].shape})')
    # create point cloud with only valid points
    pcd = structures.PointCloud(
        xyz_w=xyz_w[vmask].reshape(1, -1, 3),
        rgb=rgb[vmask].reshape(1, -1, 3),
        normal_w=normal_w[vmask].reshape(1, -1, 3),
    )  # (b=1, n, 3)

    # resample the point cloud to account for uneven sampling rate (due to rand cam pose)
    ori_num_points = pcd.xyz_w.size(1)
    print(f"before voxel downsampling, number of points = {ori_num_points}")
    cw = 2.0 / num_cells
    pcd = pcd.voxel_downsampling_random(
        cell_width=cw,
        drop_features=True,
        bidx=0,
        printout=False,
    )  # (b=1, n')
    print(
        f"after voxel downsampling, number of points = "
        f"{pcd.xyz_w.size(1)} ({pcd.xyz_w.size(1) / ori_num_points * 100.0:.2f} %)"
    )

    if pcd.xyz_w.size(1) < 10000:
        raise RuntimeError(f"number of points too few ({pcd.xyz_w.size(1)})")

    # save results
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_rgbd, exist_ok=True)
    index_dict = dict()

    # get mesh relative dir
    if mesh_rel_dir is not None:
        fn = os.path.relpath(mesh_filename, start=mesh_rel_dir)
    else:
        fn = mesh_filename
    index_dict["mesh_filename"] = fn

    # save point cloud
    xyz_w = pcd.extract_valid_attr(
        arr=pcd.xyz_w,
        bidx=0,
    )  # (n, 3)
    normal_w = pcd.extract_valid_attr(
        arr=pcd.normal_w,
        bidx=0,
    )  # (n, 3)
    rgb = pcd.extract_valid_attr(
        arr=pcd.rgb,
        bidx=0,
    )  # (n, 3)
    assert xyz_w.size(0) == normal_w.size(0)
    assert xyz_w.size(0) == rgb.size(0)

    # shuffle
    ridxs = torch.randperm(xyz_w.size(0), device=xyz_w.device)
    xyz_w = xyz_w[ridxs]
    normal_w = normal_w[ridxs]
    rgb = rgb[ridxs]

    index_dict = save_sampled_pcd(
        pcd_save_version=pcd_save_version,
        out_dir=out_dir,
        index_dict=index_dict,
        xyz_w=xyz_w,
        rgb=rgb,
        normal_w=normal_w,
        save_np_dtype=save_np_dtype,
        save_chunk_size=pcd_save_chunk_size,
    )

    # save rgbd image
    for sub_name, rgbd in rgbd_dict.items():
        name = f"rgbd_{sub_name}"  # rgbd_xy, rgbd_yz, rgbd_xz, rgbd_random
        sub_dir = os.path.join(out_dir_rgbd, name)
        rgbd: structures.RGBDImage
        rgbd = rgbd.remove_invalid(
            min_depth=0,
            max_depth=1e4,
            background_color=background_color,
        )
        _, sub_index_filename = rgbd.save_as(
            out_dir=sub_dir,
            overwrite=overwrite,
            mode="png",  # 'exr',  # exr is more efficient than npy, png is more efficient than exr
            background_color=background_color,
            save_attr_names=save_attr_names,
            flag_save_space=flag_save_space,
        )
        index_dict[name] = dict(
            index_filename=os.path.relpath(sub_index_filename, start=out_dir_rgbd),
            q=rgbd.rgb.size(1),
            h=rgbd.rgb.size(2),
            w=rgbd.rgb.size(3),
        )

    # save json
    json_filename = os.path.join(out_dir, "index.json")
    with open(json_filename, "w") as f:
        json.dump(index_dict, f, indent=2)

    if out_dir_rgbd != out_dir:
        _json_filename = os.path.join(out_dir_rgbd, "index.json")
        with open(_json_filename, "w") as f:
            json.dump(index_dict, f, indent=2)

    ret_dict = {"index_dict": index_dict, "json_filename": json_filename, "cam_name_dict": cam_name_dict}

    if flag_return_xyz_w:
        return ret_dict, xyz_w
    else:
        return ret_dict


def sample_pcd_rgbd_from_multiple_meshes_with_blender(
    *,
    mesh_filenames: T.List[str],
    scales: T.List[float],
    H_c2ws: T.List[np.ndarray],
    out_dir: str,
    out_dir_rgbd: str = None,
    out_dir_pcd_rgbd_visibility_dict: T.Union[T.Dict[float, str], None] = None,
    light_type: str = "SUN",
    num_lights: int = 8,
    min_light_energy: float = 0.0,  # remember to adjust when num_lights change
    max_light_energy: float = 3.0,  # remember to adjust when num_lights change
    num_cells: int = 512,  # number of cells per side for voxel sampling
    width_px: int = 448,  # 14 x 32
    height_px: int = 448,  # 14 x 32
    # circular camera
    num_regular_images: int = 10,
    fov: float = 40.0,  # degree
    circular_radius: float = 3.5,  # meter
    # random camera
    num_random_images: int = 30,
    min_fov: float = 40.0,  # degree
    max_fov: float = 60.0,  # degree
    min_random_radius: float = 3,
    max_random_radius: float = 4,
    random_lookat_r: float = 0.25,
    mesh_rel_dirs: T.List[str] = None,
    background_color: float = 1,
    overwrite: bool = False,
    save_attr_names: T.List[str] = None,
    pcd_save_version: int = 2,
    save_np_dtype: np.dtype = np.float32,
    seed: int = 0,
    pcd_save_chunk_size: int = 100_000,
    flag_debug: bool = False,
    regular_camera_sampling_type: str = "circular",
    flag_save_space: bool = False,
    **kwargs,
):
    """
    Render rgbd images with blender then resample point cloud from the
    rgbd point cloud.

    Strategy:
    1. normalize the mesh to fit [-1, 1] bounding box
    2. randomly add lighting in the scene
    3. sample randomly the camera in a sphere to avoid occusion
    4. sample camera on multi-plane circles
    5. sample camera in a spherical shell

    Args:
        pcd_save_version:
            1: single npz file saving xyz_w (n, 3), rgb (n, 3), normal_w (n, 3)
            2: multiple npz files, each saving a subset of xyz_w (m, 3) or rgb (n, 3), or normal_w (n, 3),
                xyz, rgb, normal are saved separately
    """
    _set_seed(seed)

    assert isinstance(mesh_filenames, (list, tuple))
    assert mesh_rel_dirs is None or isinstance(mesh_rel_dirs, (list, tuple))
    n = len(mesh_filenames)
    assert len(scales) == n
    assert len(H_c2ws) == n

    if not overwrite:
        assert (not os.path.exists(out_dir)) or (not os.listdir(out_dir))

    # compile json file (mesh, lighting, camera)
    scene_dict = dict()

    # mesh
    scene_dict["meshes"] = []
    for ii in range(len(mesh_filenames)):
        mdict = dict(
            name="mesh",
            filename=mesh_filenames[ii],
            normalize_first=True,  # [-1, 1] aabb box
            H_c2w=H_c2ws[ii],  # no rotation after normalization
            scale=np.array([scales[ii]] * 3),  # no scaling after normalization
            cut_aabb_center=[0.0, 0.0, 0.0],
            cut_aabb_radius=[1.0, 1.0, 1.0],
        )
        scene_dict["meshes"].append(mdict)

    # lighting
    light_dicts = []
    for il in range(num_lights):
        if light_type == "SUN":
            # note that the light is toward -z
            # but since we just wnat random light direction, we do not care
            H_c2w = rigid_motion.get_H_c2w_lookat(
                pinhole_location_w=(0, 0, 0.0),
                look_at_w=rigid_motion.get_random_direction().astype(np.float32),  # (3,)
                up_w=(0, 1, 0.0),
                invert_y=False,
            )  # (4, 4)
            energy = float(np.random.rand()) * (max_light_energy - min_light_energy) + min_light_energy

            mdict = dict(
                name=f"light {il}",
                light_type=light_type,
                H_c2w=H_c2w.tolist(),
                energy=energy,
                use_shadow=False,
                specular_factor=1.0,
            )
        elif light_type == "diffuse":
            mdict = dict(
                name=f"light {il}",
                light_type=light_type,
                color=[1.0, 1.0, 1.0, 1.0],
                strength=1.0,
            )
        else:
            raise NotImplementedError
        light_dicts.append(mdict)
    scene_dict["lighting"] = light_dicts

    # camera
    camera_dicts, cam_name_dict = get_camera_dicts_for_blender(
        num_regular_images=num_regular_images,
        r=circular_radius,
        fov=fov,
        width_px=width_px,
        height_px=height_px,
        num_random_images=num_random_images,
        min_random_radius=min_random_radius,
        max_random_radius=max_random_radius,
        random_lookat_r=random_lookat_r,
        max_fov=max_fov,
        min_fov=min_fov,
        regular_camera_sampling_type=regular_camera_sampling_type,
    )

    scene_dict["cameras"] = camera_dicts

    # save the scene config
    os.makedirs(out_dir, exist_ok=True)
    json_filename = os.path.join(out_dir, "scene_config.json")
    with open(json_filename, "w") as f:
        json.dump(scene_dict, f, indent=2, cls=json_utils.NumpyJsonEncoder)

    # render
    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        # if True:
        #     tmp_dir = 'tmp_blender'
        #     print(f'tmp_dir = {tmp_dir}')

        tmp_dir = os.path.abspath(tmp_dir)

        # save config to tmp file (again) just to keep config
        json_filename = os.path.join(tmp_dir, "config.json")
        with open(json_filename, "w") as f:
            json.dump(scene_dict, f, indent=2, cls=json_utils.NumpyJsonEncoder)

        # render with blender
        tmp_out_dir = os.path.join(tmp_dir, "out")
        blender_cmd = blender_rendering_utils.get_blender_exe()
        blender_script = blender_rendering_utils.get_blender_utils_path()
        cmd = (
            f"{blender_cmd} --background --log-level 1 --python {blender_script} -- "
            f"--filename {json_filename} --out_dir {tmp_out_dir} --debug 0 > blender.log"
        )
        print(cmd)
        os.system(cmd)

        # read and save the output
        rgbd_dict = dict()  # name -> rgbd
        from_idx = 0
        for name in cam_name_dict:
            rgbd = blender_plib_utils.read_blender_results_to_rgbd(
                result_dir=tmp_out_dir,
                from_idx=from_idx,
                to_idx=from_idx + cam_name_dict[name],
                use_srgb=True,
                flag_save_space=flag_save_space,
                debug=flag_debug,
            )  # (b=1, q, h, w)
            assert rgbd.rgb.size(1) == cam_name_dict[name]
            rgbd_dict[name] = rgbd  # (b=1, q, h, w)
            from_idx += cam_name_dict[name]

    # construct point cloud by combining rgbd images
    pcd_dict = dict()  # name -> pcd
    for name in rgbd_dict:
        # print(f'{name}: {rgbd_dict[name].hit_map.float().mean()}')
        pcd = rgbd_dict[name].get_pcd()
        # print(f'{name} pcd: {pcd.xyz_w.shape}')
        pcd_dict[name] = pcd  # (b=1, n)
    pcd = structures.PointCloud.cat(list(pcd_dict.values()), dim=1)  # (b=1, n)
    # print(f'ori xyz_w.shape: {pcd.xyz_w.shape}')

    # remove points that are outside [-1, 1] aabb, and are not finite
    xyz_w = pcd.extract_valid_attr(
        arr=pcd.xyz_w,
        bidx=0,
    )  # (n, 3)
    n, _3xyz = xyz_w.shape

    bbox_eps = 0.01
    vmask = torch.logical_and(
        (xyz_w >= -1 - bbox_pes).all(dim=-1),  # (n,)
        (xyz_w <= 1 + bbox_eps).all(dim=-1),  # (n,)
    )  # (n,)
    # print(f'after xyz_w, vmask: {vmask.shape} ({vmask.float().mean()})')
    normal_w = pcd.extract_valid_attr(
        arr=pcd.normal_w,
        bidx=0,
    )  # (n, 3)
    assert xyz_w.size(0) == normal_w.size(0)
    vmask = torch.logical_and(
        vmask,  # (n,)
        normal_w.isfinite().all(dim=-1),  # (n,)
    )  # (n,)
    # print(f'after normal, vmask: {vmask.shape} ({vmask.float().mean()})')
    rgb = pcd.extract_valid_attr(
        arr=pcd.rgb,
        bidx=0,
    )  # (n, 3)
    assert xyz_w.size(0) == rgb.size(0)
    vmask = torch.logical_and(
        vmask,  # (n,)
        rgb.isfinite().all(dim=-1),  # (n,)
    )  # (n,)
    # print(f'after rgb, vmask: {vmask.shape} ({vmask.float().mean()})')
    # print(f'vmask.shape = {vmask.shape} (total = {vmask.sum()})')
    # print(f'rgb.shape = {rgb.shape} ({rgb[vmask].shape})')
    # create point cloud with only valid points
    pcd = structures.PointCloud(
        xyz_w=xyz_w[vmask].reshape(1, -1, 3),
        rgb=rgb[vmask].reshape(1, -1, 3),
        normal_w=normal_w[vmask].reshape(1, -1, 3),
    )  # (b=1, n, 3)

    # resample the point cloud to account for uneven sampling rate (due to rand cam pose)
    ori_num_points = pcd.xyz_w.size(1)
    print(f"before voxel downsampling, number of points = {ori_num_points}")
    cw = 2.0 / num_cells
    pcd = pcd.voxel_downsampling_random(
        cell_width=cw,
        drop_features=True,
        bidx=0,
        printout=False,
    )  # (b=1, n')
    print(
        f"after voxel downsampling, number of points = "
        f"{pcd.xyz_w.size(1)} ({pcd.xyz_w.size(1) / ori_num_points * 100.0:.2f} %)"
    )

    if pcd.xyz_w.size(1) < 10000:
        raise RuntimeError(f"number of points too few ({pcd.xyz_w.size(1)})")

    # save results
    os.makedirs(out_dir, exist_ok=True)
    json_filename = os.path.join(out_dir, "index.json")
    index_dict = dict()

    # get mesh relative dir
    if mesh_rel_dirs is not None:
        fns = [os.path.relpath(mesh_filenames[i], start=mesh_rel_dirs[i]) for i in range(len(mesh_filenames))]
    else:
        fns = mesh_filenames
    index_dict["mesh_filenames"] = fns
    index_dict["scales"] = scales
    index_dict["H_c2ws"] = H_c2ws

    # save point cloud
    pcd_filename = os.path.join(out_dir, "pcd.npz")
    xyz_w = pcd.extract_valid_attr(
        arr=pcd.xyz_w,
        bidx=0,
    )  # (n, 3)
    normal_w = pcd.extract_valid_attr(
        arr=pcd.normal_w,
        bidx=0,
    )  # (n, 3)
    rgb = pcd.extract_valid_attr(
        arr=pcd.rgb,
        bidx=0,
    )  # (n, 3)
    assert xyz_w.size(0) == normal_w.size(0)
    assert xyz_w.size(0) == rgb.size(0)

    # shuffle
    ridxs = torch.randperm(xyz_w.size(0), device=xyz_w.device)
    xyz_w = xyz_w[ridxs]
    normal_w = normal_w[ridxs]
    rgb = rgb[ridxs]

    index_dict = save_sampled_pcd(
        pcd_save_version=pcd_save_version,
        out_dir=out_dir,
        index_dict=index_dict,
        xyz_w=xyz_w,
        rgb=rgb,
        normal_w=normal_w,
        save_np_dtype=save_np_dtype,
        save_chunk_size=pcd_save_chunk_size,
    )

    # save rgbd image
    for sub_name, rgbd in rgbd_dict.items():
        name = f"rgbd_{sub_name}"  # rgbd_xy, rgbd_yz, rgbd_xz, rgbd_random
        sub_dir = os.path.join(out_dir, name)
        rgbd: structures.RGBDImage
        _, sub_index_filename = rgbd.save_as(
            out_dir=sub_dir,
            overwrite=overwrite,
            mode="png",  # 'exr' is more efficient than npy
            background_color=background_color,
            save_attr_names=save_attr_names,
            flag_save_space=flag_save_space,
        )
        index_dict[name] = dict(
            index_filename=os.path.relpath(sub_index_filename, start=out_dir),
            q=rgbd.rgb.size(1),
            h=rgbd.rgb.size(2),
            w=rgbd.rgb.size(3),
        )

    # save json
    with open(json_filename, "w") as f:
        json.dump(index_dict, f, indent=2, cls=json_utils.NumpyJsonEncoder)

    return index_dict, json_filename


def load_all(
    index_filename: str,
):
    with open(index_filename, "r") as f:
        index_dict = json.load(f)

    root_dir = os.path.dirname(index_filename)

    mesh_filename = index_dict["mesh_filename"]

    # load pcd
    pcd_dict = np.load(
        os.path.join(root_dir, index_dict["pcd_filename"]),
        allow_pickle=True,
    )
    pcd_dict = {key: pcd_dict[key] for key in pcd_dict}
    pcd_dict = utils.to_tensor(pcd_dict)

    # rgbd_xy
    sub_index_dict = index_dict["rgbd_xy"]
    rgbd_xy = structures.RGBDImage.load_from(
        index_filename=os.path.join(root_dir, sub_index_dict["index_filename"]),
    )

    # rgbd_xz
    sub_index_dict = index_dict["rgbd_xz"]
    rgbd_xz = structures.RGBDImage.load_from(
        index_filename=os.path.join(root_dir, sub_index_dict["index_filename"]),
    )

    # rgbd_yz
    sub_index_dict = index_dict["rgbd_yz"]
    rgbd_yz = structures.RGBDImage.load_from(
        index_filename=os.path.join(root_dir, sub_index_dict["index_filename"]),
    )

    # rgbd_random
    sub_index_dict = index_dict["rgbd_random"]
    rgbd_random = structures.RGBDImage.load_from(
        index_filename=os.path.join(root_dir, sub_index_dict["index_filename"]),
    )

    out_dict = dict(
        mesh_filename=mesh_filename,
        **pcd_dict,
        rgbd_xy=rgbd_xy,
        rgbd_xz=rgbd_xz,
        rgbd_yz=rgbd_yz,
        rgbd_random=rgbd_random,
    )

    return out_dict


def load_partial(
    index_filename: str,
    num_xy: int,  # -1: load all
    num_yz: int,
    num_xz: int,
    num_random: int,
    num_sphere: int = 0,
    attr_names: T.List[str] = None,
    load_pcd_xyz: bool = True,
    load_pcd_rgb: bool = True,
    load_pcd_normal: bool = True,
    load_pcd_plucker: bool = False,
    load_pcd_ray_origin: bool = False,
    load_pcd_ray_direction: bool = False,
    num_points_needed: int = None,  # if None, load one chunk no matter what
    rng: np.random.RandomState = None,
    zip_f: str | None = None,
    candidate_qidxs: T.Optional[T.List[int]] = None,
    printout: bool = False,
):
    """
    - when zip_f is None, index_filename is full path to a file on disk
    - when zip_f is not None, index_filename is relative path to the root of the content in the zip file
    """
    with zipfile.ZipFile(zip_f) if zip_f is not None else contextlib.nullcontext() as zipfile_obj:
        return load_partial_core(
            zipfile_obj=zipfile_obj,
            index_filename=index_filename,
            num_xy=num_xy,  # -1: load all
            num_yz=num_yz,
            num_xz=num_xz,
            num_random=num_random,
            num_sphere=num_sphere,
            attr_names=attr_names,
            load_pcd_xyz=load_pcd_xyz,
            load_pcd_rgb=load_pcd_rgb,
            load_pcd_normal=load_pcd_normal,
            load_pcd_plucker=load_pcd_plucker,
            load_pcd_ray_origin=load_pcd_ray_origin,
            load_pcd_ray_direction=load_pcd_ray_direction,
            num_points_needed=num_points_needed,  # if None, load one chunk no matter what
            candidate_qidxs=candidate_qidxs,
            rng=rng,
            printout=printout,
        )


def _load_json_file(filepath, zipfile_obj: zipfile.ZipFile | None = None):
    """Helper to load JSON file from disk or zipfile"""
    if zipfile_obj is not None:
        with zipfile_obj.open(filepath) as f:
            return json.load(f)
    else:
        with open(filepath, "r") as f:
            return json.load(f)


def _load_numpy_file(filepath, allow_pickle: bool = True, zipfile_obj: zipfile.ZipFile | None = None):
    """Helper to load numpy file from disk or zipfile"""
    if zipfile_obj is not None:
        with zipfile_obj.open(filepath) as f:
            return np.load(f, allow_pickle=allow_pickle)
    else:
        return np.load(filepath, allow_pickle=allow_pickle)


def load_partial_core(
    *,
    index_filename: str,
    num_xy: int,  # -1: load all
    num_yz: int,
    num_xz: int,
    num_random: int,
    num_sphere: int = 0,
    attr_names: T.List[str] = None,
    load_pcd_xyz: bool = True,
    load_pcd_rgb: bool = True,
    load_pcd_normal: bool = True,
    load_pcd_plucker: bool = False,
    load_pcd_ray_origin: bool = False,
    load_pcd_ray_direction: bool = False,
    num_points_needed: int = None,  # if None, load one chunk no matter what
    candidate_qidxs: T.Optional[T.List[int]] = None,
    rng: np.random.RandomState = None,
    zipfile_obj: zipfile.ZipFile | None = None,
    printout: bool = False,
):
    """
    - if zipfile_obj is None, this function reads data from disk
    - if zipfile_obj is not None, this function reads data from zip
    """
    if rng is None:
        rng = np.random

    # with open(index_filename, "r") as f:
    #     index_dict = json.load(f)
    index_dict = _load_json_file(index_filename, zipfile_obj=zipfile_obj)

    root_dir = os.path.dirname(index_filename)

    load_pcd = (
        load_pcd_xyz
        or load_pcd_rgb
        or load_pcd_normal
        or load_pcd_plucker
        or load_pcd_ray_origin
        or load_pcd_ray_direction
    )

    time_dict = dict()

    if load_pcd:
        # load pcd
        pcd_save_version = index_dict.get("pcd_save_version", 1)
        stime = timer()
        if pcd_save_version == 1:
            assert "pcd_filename" in index_dict, f"{list(index_dict.keys())=}"

            # pcd_dict = np.load(
            #     os.path.join(root_dir, index_dict["pcd_filename"]),
            #     allow_pickle=True,
            # )
            pcd_dict = _load_numpy_file(
                os.path.join(root_dir, index_dict["pcd_filename"]), allow_pickle=True, zipfile_obj=zipfile_obj
            )

            # pcd_dict = {key: pcd_dict[key] for key in pcd_dict}
            pdict = dict()
            if load_pcd_xyz:
                pdict["xyz_w"] = pcd_dict["xyz_w"]
            if load_pcd_rgb:
                pdict["rgb"] = pcd_dict["rgb"]
            if load_pcd_normal:
                pdict["normal_w"] = pcd_dict["normal_w"]
            if load_pcd_plucker:
                raise NotImplementedError
            if load_pcd_ray_origin:
                raise NotImplementedError
            if load_pcd_ray_direction:
                raise NotImplementedError
            pcd_dict = utils.to_tensor(pdict, dtype=torch.float)
        elif pcd_save_version == 2:
            assert "pcd_index_filename" in index_dict
            pcd_index_filename = os.path.join(root_dir, index_dict["pcd_index_filename"])
            # with open(pcd_index_filename, "r") as f:
            #     pcd_index = json.load(f)
            pcd_index = _load_json_file(pcd_index_filename, zipfile_obj=zipfile_obj)

            num_chunks = pcd_index["num_chunks"]
            chunk_size = pcd_index["chunk_size"]
            assert num_chunks >= 1
            if num_points_needed is None:
                num_chunks_needed = 1
            elif num_points_needed >= 0:
                num_chunks_needed = (num_points_needed + chunk_size - 1) // chunk_size
                num_chunks_needed = min(num_chunks_needed, num_chunks - 1)
            else:
                # negative value
                assert num_points_needed < 0, f"{num_points_needed=}"
                num_chunks_needed = num_chunks - 1  # we handle the last chunk differently

            # we first select from 0..num_chuck-1 chunks that have the "chunk_size" points
            # then if not enough of points, we include the last chunk
            ridxs = rng.permutation(num_chunks - 1)
            pcd_dict = dict()
            current_num_points = 0
            for cidx in range(num_chunks_needed):
                chunk_idx = ridxs[cidx]
                added = False
                if load_pcd_xyz:
                    filename = pcd_index["xyz_w"][chunk_idx]
                    # arr = np.load(os.path.join(root_dir, filename))  # (m, 3)
                    arr = _load_numpy_file(os.path.join(root_dir, filename), zipfile_obj=zipfile_obj)  # (m, 3)

                    if "xyz_w" not in pcd_dict:
                        pcd_dict["xyz_w"] = []
                    pcd_dict["xyz_w"].append(arr)
                    if not added:
                        current_num_points += arr.shape[0]
                        added = True
                if load_pcd_rgb:
                    filename = pcd_index["rgb"][chunk_idx]
                    # arr = np.load(os.path.join(root_dir, filename))  # (m, 3)
                    arr = _load_numpy_file(os.path.join(root_dir, filename), zipfile_obj=zipfile_obj)  # (m, 3)

                    if "rgb" not in pcd_dict:
                        pcd_dict["rgb"] = []
                    pcd_dict["rgb"].append(arr)
                    if not added:
                        current_num_points += arr.shape[0]
                        added = True
                if load_pcd_normal:
                    filename = pcd_index["normal_w"][chunk_idx]
                    # arr = np.load(os.path.join(root_dir, filename))  # (m, 3)
                    arr = _load_numpy_file(os.path.join(root_dir, filename), zipfile_obj=zipfile_obj)  # (m, 3)

                    arr_norm = np.linalg.norm(arr, ord=2, axis=1, keepdims=True)
                    arr = arr / (arr_norm + 1e-9)
                    if "normal_w" not in pcd_dict:
                        pcd_dict["normal_w"] = []
                    pcd_dict["normal_w"].append(arr)
                    if not added:
                        current_num_points += arr.shape[0]
                        added = True
                if load_pcd_plucker:
                    filename = pcd_index["plucker"][chunk_idx]
                    # arr = np.load(os.path.join(root_dir, filename))  # (m, 6)
                    arr = _load_numpy_file(os.path.join(root_dir, filename), zipfile_obj=zipfile_obj)  # (m, 6)

                    if "plucker" not in pcd_dict:
                        pcd_dict["plucker"] = []
                    pcd_dict["plucker"].append(arr)
                    if not added:
                        current_num_points += arr.shape[0]
                        added = True
                if load_pcd_ray_origin:
                    filename = pcd_index["ray_o"][chunk_idx]
                    # arr = np.load(os.path.join(root_dir, filename))  # (m, 3)
                    arr = _load_numpy_file(os.path.join(root_dir, filename), zipfile_obj=zipfile_obj)  # (m, 3)

                    arr_norm = np.linalg.norm(arr, ord=2, axis=1, keepdims=True)
                    arr = arr / (arr_norm + 1e-9)
                    if "ray_o" not in pcd_dict:
                        pcd_dict["ray_o"] = []
                    pcd_dict["ray_o"].append(arr)
                    if not added:
                        current_num_points += arr.shape[0]
                        added = True
                if load_pcd_ray_direction:
                    filename = pcd_index["ray_d"][chunk_idx]
                    # arr = np.load(os.path.join(root_dir, filename))  # (m, 3)
                    arr = _load_numpy_file(os.path.join(root_dir, filename), zipfile_obj=zipfile_obj)  # (m, 3)

                    arr_norm = np.linalg.norm(arr, ord=2, axis=1, keepdims=True)
                    arr = arr / (arr_norm + 1e-9)
                    if "ray_d" not in pcd_dict:
                        pcd_dict["ray_d"] = []
                    pcd_dict["ray_d"].append(arr)
                    if not added:
                        current_num_points += arr.shape[0]
                        added = True

            # load the last chunk if needed
            if (num_points_needed < 0) or (
                (num_points_needed is not None) and (current_num_points < num_points_needed)
            ):
                chunk_idx = num_chunks - 1
                added = False
                if load_pcd_xyz:
                    filename = pcd_index["xyz_w"][chunk_idx]
                    # arr = np.load(os.path.join(root_dir, filename))  # (m, 3)
                    arr = _load_numpy_file(os.path.join(root_dir, filename), zipfile_obj=zipfile_obj)  # (m, 3)

                    if "xyz_w" not in pcd_dict:
                        pcd_dict["xyz_w"] = []
                    pcd_dict["xyz_w"].append(arr)
                    if not added:
                        current_num_points += arr.shape[0]
                        added = True
                if load_pcd_rgb:
                    filename = pcd_index["rgb"][chunk_idx]
                    # arr = np.load(os.path.join(root_dir, filename))  # (m, 3)
                    arr = _load_numpy_file(os.path.join(root_dir, filename), zipfile_obj=zipfile_obj)  # (m, 3)

                    if "rgb" not in pcd_dict:
                        pcd_dict["rgb"] = []
                    pcd_dict["rgb"].append(arr)
                    if not added:
                        current_num_points += arr.shape[0]
                        added = True
                if load_pcd_normal:
                    filename = pcd_index["normal_w"][chunk_idx]
                    # arr = np.load(os.path.join(root_dir, filename))  # (m, 3)
                    arr = _load_numpy_file(os.path.join(root_dir, filename), zipfile_obj=zipfile_obj)  # (m, 3)

                    if "normal_w" not in pcd_dict:
                        pcd_dict["normal_w"] = []
                    pcd_dict["normal_w"].append(arr)
                    if not added:
                        current_num_points += arr.shape[0]
                        added = True
                if load_pcd_plucker:
                    filename = pcd_index["plucker"][chunk_idx]
                    # arr = np.load(os.path.join(root_dir, filename))  # (m, 6)
                    arr = _load_numpy_file(os.path.join(root_dir, filename), zipfile_obj=zipfile_obj)  # (m, 6)

                    if "plucker" not in pcd_dict:
                        pcd_dict["plucker"] = []
                    pcd_dict["plucker"].append(arr)
                    if not added:
                        current_num_points += arr.shape[0]
                        added = True
                if load_pcd_ray_origin:
                    filename = pcd_index["ray_o"][chunk_idx]
                    # arr = np.load(os.path.join(root_dir, filename))  # (m, 3)
                    arr = _load_numpy_file(os.path.join(root_dir, filename), zipfile_obj=zipfile_obj)  # (m, 3)

                    arr_norm = np.linalg.norm(arr, ord=2, axis=1, keepdims=True)
                    arr = arr / (arr_norm + 1e-9)
                    if "ray_o" not in pcd_dict:
                        pcd_dict["ray_o"] = []
                    pcd_dict["ray_o"].append(arr)
                    if not added:
                        current_num_points += arr.shape[0]
                        added = True
                if load_pcd_ray_direction:
                    filename = pcd_index["ray_d"][chunk_idx]
                    # arr = np.load(os.path.join(root_dir, filename))  # (m, 3)
                    arr = _load_numpy_file(os.path.join(root_dir, filename), zipfile_obj=zipfile_obj)  # (m, 3)

                    arr_norm = np.linalg.norm(arr, ord=2, axis=1, keepdims=True)
                    arr = arr / (arr_norm + 1e-9)
                    if "ray_d" not in pcd_dict:
                        pcd_dict["ray_d"] = []
                    pcd_dict["ray_d"].append(arr)
                    if not added:
                        current_num_points += arr.shape[0]
                        added = True

            # concat all loaded chunks
            for key in pcd_dict:
                pcd_dict[key] = np.concatenate(pcd_dict[key], axis=0)  # (n, 3)

            pcd_dict = utils.to_tensor(pcd_dict, dtype=torch.float)
        else:
            raise NotImplementedError
        time_dict["open_pcd_npz"] = timer() - stime
    else:
        pcd_dict = dict()

    # rgbd_xy
    stime = timer()
    if num_xy > 0:
        sub_index_dict = index_dict["rgbd_xy"]
        if candidate_qidxs is None:
            q = sub_index_dict["q"]
            qidxs = rng.permutation(q)[:num_xy]
        else:
            _ridx = rng.permutation(len(candidate_qidxs))[:num_xy]
            qidxs = [candidate_qidxs[ii] for ii in _ridx]
        rgbd_xy = structures.RGBDImage.load_from(
            index_filename=os.path.join(root_dir, sub_index_dict["index_filename"]),
            bidxs=None,
            qidxs=[qidxs],
            attr_names=attr_names,
            zipfile_obj=zipfile_obj,
        )
    elif num_xy == -1:
        # load all
        sub_index_dict = index_dict["rgbd_xy"]
        if candidate_qidxs is None:
            q = sub_index_dict["q"]
            qidxs = np.arange(q)
        else:
            qidxs = candidate_qidxs
        rgbd_xy = structures.RGBDImage.load_from(
            index_filename=os.path.join(root_dir, sub_index_dict["index_filename"]),
            bidxs=None,
            qidxs=[qidxs],
            attr_names=attr_names,
            zipfile_obj=zipfile_obj,
        )
    else:
        rgbd_xy = None
    time_dict["read_xy"] = timer() - stime

    # rgbd_xz
    stime = timer()
    if num_xz > 0:
        sub_index_dict = index_dict["rgbd_xz"]
        if candidate_qidxs is None:
            q = sub_index_dict["q"]
            qidxs = rng.permutation(q)[:num_xz]
        else:
            _ridx = rng.permutation(len(candidate_qidxs))[:num_xz]
            qidxs = [candidate_qidxs[ii] for ii in _ridx]
        rgbd_xz = structures.RGBDImage.load_from(
            index_filename=os.path.join(root_dir, sub_index_dict["index_filename"]),
            bidxs=None,
            qidxs=[qidxs],
            attr_names=attr_names,
            zipfile_obj=zipfile_obj,
        )
    elif num_xz == -1:
        # load all
        sub_index_dict = index_dict["rgbd_xz"]
        if candidate_qidxs is None:
            q = sub_index_dict["q"]
            qidxs = np.arange(q)
        else:
            qidxs = candidate_qidxs
        rgbd_xz = structures.RGBDImage.load_from(
            index_filename=os.path.join(root_dir, sub_index_dict["index_filename"]),
            bidxs=None,
            qidxs=[qidxs],
            attr_names=attr_names,
            zipfile_obj=zipfile_obj,
        )
    else:
        rgbd_xz = None
    time_dict["read_xz"] = timer() - stime

    # rgbd_yz
    stime = timer()
    if num_yz > 0:
        sub_index_dict = index_dict["rgbd_yz"]
        if candidate_qidxs is None:
            q = sub_index_dict["q"]
            qidxs = rng.permutation(q)[:num_yz]
        else:
            _ridx = rng.permutation(len(candidate_qidxs))[:num_yz]
            qidxs = [candidate_qidxs[ii] for ii in _ridx]
        rgbd_yz = structures.RGBDImage.load_from(
            index_filename=os.path.join(root_dir, sub_index_dict["index_filename"]),
            bidxs=None,
            qidxs=[qidxs],
            attr_names=attr_names,
            zipfile_obj=zipfile_obj,
        )
    elif num_yz == -1:
        # load all
        sub_index_dict = index_dict["rgbd_yz"]
        if candidate_qidxs is None:
            q = sub_index_dict["q"]
            qidxs = np.arange(q)
        else:
            qidxs = candidate_qidxs
        rgbd_yz = structures.RGBDImage.load_from(
            index_filename=os.path.join(root_dir, sub_index_dict["index_filename"]),
            bidxs=None,
            qidxs=[qidxs],
            attr_names=attr_names,
            zipfile_obj=zipfile_obj,
        )
    else:
        rgbd_yz = None
    time_dict["read_yz"] = timer() - stime

    # rgbd_random
    stime = timer()
    if num_random > 0:
        sub_index_dict = index_dict["rgbd_random"]
        if candidate_qidxs is None:
            q = sub_index_dict["q"]
            qidxs = rng.permutation(q)[:num_random]
        else:
            _ridx = rng.permutation(len(candidate_qidxs))[:num_random]
            qidxs = [candidate_qidxs[ii] for ii in _ridx]
        rgbd_random = structures.RGBDImage.load_from(
            index_filename=os.path.join(root_dir, sub_index_dict["index_filename"]),
            bidxs=None,
            qidxs=[qidxs],
            attr_names=attr_names,
            zipfile_obj=zipfile_obj,
        )
    elif num_random == -1:
        # load all
        sub_index_dict = index_dict["rgbd_random"]
        if candidate_qidxs is None:
            q = sub_index_dict["q"]
            qidxs = np.arange(q)
        else:
            qidxs = candidate_qidxs
        rgbd_random = structures.RGBDImage.load_from(
            # index_filename=os.path.join(root_dir, "rgbd_random", "index.json"),
            index_filename=os.path.join(root_dir, sub_index_dict["index_filename"]),
            bidxs=None,
            qidxs=[qidxs],
            attr_names=attr_names,
            zipfile_obj=zipfile_obj,
        )
    else:
        rgbd_random = None
    time_dict["read_random"] = timer() - stime

    # rgbd_sphere
    stime = timer()
    if num_sphere > 0:
        if printout:
            print(f"loading rgbd_sphere: {num_sphere}", flush=True)
        sub_index_dict = index_dict["rgbd_sphere"]
        if candidate_qidxs is None:
            q = sub_index_dict["q"]
            qidxs = rng.permutation(q)[:num_sphere]
        else:
            _ridx = rng.permutation(len(candidate_qidxs))[:num_sphere]
            qidxs = [candidate_qidxs[ii] for ii in _ridx]
        rgbd_sphere = structures.RGBDImage.load_from(
            index_filename=os.path.join(root_dir, sub_index_dict["index_filename"]),
            bidxs=None,
            qidxs=[qidxs],
            attr_names=attr_names,
            printout=printout,
            zipfile_obj=zipfile_obj,
        )
        if printout:
            print(f"finished loading rgbd_sphere: {num_sphere}", flush=True)
    elif num_sphere == -1:
        # load all
        sub_index_dict = index_dict["rgbd_sphere"]
        if candidate_qidxs is None:
            q = sub_index_dict["q"]
            qidxs = np.arange(q)
        else:
            qidxs = candidate_qidxs
        rgbd_sphere = structures.RGBDImage.load_from(
            index_filename=os.path.join(root_dir, sub_index_dict["index_filename"]),
            bidxs=None,
            qidxs=[qidxs],
            attr_names=attr_names,
            printout=printout,
            zipfile_obj=zipfile_obj,
        )
    else:
        rgbd_sphere = None
    time_dict["read_sphere"] = timer() - stime

    # for tmp_k, tmp_v in pcd_dict.items():
    #     print(f"\n\n{tmp_k=}, {tmp_v.shape=}\n\n")

    out_dict = dict(
        # mesh_filename=mesh_filename,
        **pcd_dict,  # tensor
        rgbd_xy=rgbd_xy,
        rgbd_xz=rgbd_xz,
        rgbd_yz=rgbd_yz,
        rgbd_random=rgbd_random,
        rgbd_sphere=rgbd_sphere,
        time_dict=time_dict,
    )

    return out_dict


def load_specific(
    index_filename: str,
    mode: str,
    attr_names: T.List[str] = None,
    rng: np.random.RandomState = None,
) -> T.Dict[str, T.Any]:
    """
    Load specific qidx in the rgbd images.

    Args:
        index_filename:
        mode:
            'cube': cubemap (ie, 6 images)
            'random_xy': randomly select one image from xy
            'random_yz': randomly select one image from yz
            'random_xz': randomly select one image from xz
            'frontal': [xy[0], xy[5], yz[0], yz[5]]  #TODO: check if they are y up
        attr_names:

    Returns:

    """
    if rng is None:
        rng = np.random.RandomState()

    with open(index_filename, "r") as f:
        index_dict = json.load(f)

    root_dir = os.path.dirname(index_filename)

    if mode == "cube":
        # Note: there is a bug, the selected two camera poses
        # from xz is actually on x-axis, so duplicated with xy

        # 2 on the xy plane (separated uniformly)
        sub_index_dict = index_dict["rgbd_xy"]
        q = sub_index_dict["q"]
        n = 2
        step = max(1, q // n)
        qidxs = [i * step for i in range(n)]
        rgbd_xy = structures.RGBDImage.load_from(
            index_filename=os.path.join(root_dir, "rgbd_xy", "index.json"),
            bidxs=None,
            qidxs=[qidxs],
            attr_names=attr_names,
        )

        # 2 on the yz plane (separated uniformly)
        sub_index_dict = index_dict["rgbd_yz"]
        q = sub_index_dict["q"]
        n = 2
        step = max(1, q // n)
        qidxs = [i * step for i in range(n)]
        rgbd_yz = structures.RGBDImage.load_from(
            index_filename=os.path.join(root_dir, "rgbd_yz", "index.json"),
            bidxs=None,
            qidxs=[qidxs],
            attr_names=attr_names,
        )

        # 2 on the xz plane
        sub_index_dict = index_dict["rgbd_xz"]
        q = sub_index_dict["q"]
        n = 2
        step = max(1, q // n)
        qidxs = [i * step for i in range(n)]
        rgbd_xz = structures.RGBDImage.load_from(
            index_filename=os.path.join(root_dir, "rgbd_xz", "index.json"),
            bidxs=None,
            qidxs=[qidxs],
            attr_names=attr_names,
        )
        # concat along q
        rgbd = structures.RGBDImage.cat(
            [
                rgbd_xy,
                rgbd_yz,
                rgbd_xz,
            ],
            dim=1,
        )

        out_dict = dict(
            rgbd=rgbd,
        )
    elif mode == "cube_v2":
        # Note: this is not exactly cube

        # 2 on the xy plane (separated uniformly)
        sub_index_dict = index_dict["rgbd_xy"]
        q = sub_index_dict["q"]
        n = 2
        step = max(1, q // n)
        qidxs = [i * step for i in range(n)]
        rgbd_xy = structures.RGBDImage.load_from(
            index_filename=os.path.join(root_dir, "rgbd_xy", "index.json"),
            bidxs=None,
            qidxs=[qidxs],
            attr_names=attr_names,
        )

        # 2 on the yz plane (separated uniformly)
        sub_index_dict = index_dict["rgbd_yz"]
        q = sub_index_dict["q"]
        n = 2
        step = max(1, q // n)
        qidxs = [i * step for i in range(n)]
        rgbd_yz = structures.RGBDImage.load_from(
            index_filename=os.path.join(root_dir, "rgbd_yz", "index.json"),
            bidxs=None,
            qidxs=[qidxs],
            attr_names=attr_names,
        )

        # 2 on the xz plane
        sub_index_dict = index_dict["rgbd_xz"]
        q = sub_index_dict["q"]
        n = 2
        step = max(1, q // n)
        # need to get the 90 degree
        delta_angle = 360.0 / q
        delta_i = round(90 / delta_angle)
        qidxs = [delta_i + i * step for i in range(n)]
        rgbd_xz = structures.RGBDImage.load_from(
            index_filename=os.path.join(root_dir, "rgbd_xz", "index.json"),
            bidxs=None,
            qidxs=[qidxs],
            attr_names=attr_names,
        )
        # concat along q
        rgbd = structures.RGBDImage.cat(
            [
                rgbd_xy,
                rgbd_yz,
                rgbd_xz,
            ],
            dim=1,
        )

        out_dict = dict(
            rgbd=rgbd,
        )
    elif mode in ["random_xy", "random_yz", "random_xz"]:
        # 1 on the plane
        plane_name = mode[-2:]
        sub_index_dict = index_dict[f"rgbd_{plane_name}"]
        q = sub_index_dict["q"]
        qidxs = rng.randint(0, q)
        rgbd = structures.RGBDImage.load_from(
            index_filename=os.path.join(root_dir, f"rgbd_{plane_name}", "index.json"),
            bidxs=None,
            qidxs=[qidxs],
            attr_names=attr_names,
        )  # (b=1, q=1, h, w)
        out_dict = dict(
            rgbd=rgbd,
        )
    elif mode == "frontal":
        # randomly choose one camera from xy[0], xy[5]

        rgbd_type = "rgbd_xy"

        sub_index_dict = index_dict[rgbd_type]
        q = sub_index_dict["q"]
        n = 2
        step = max(1, q // n)
        qidxs = [i * step for i in range(n)]
        qidx = rng.choice(
            qidxs,
            size=1,
        ).tolist()[0]

        rgbd = structures.RGBDImage.load_from(
            index_filename=os.path.join(root_dir, rgbd_type, "index.json"),
            bidxs=None,
            qidxs=[qidx],
            attr_names=attr_names,
        )

        out_dict = dict(
            rgbd=rgbd,
        )

    else:
        raise NotImplementedError

    return out_dict


def load_pcd_rgbd_from_byte_dict(
    byte_dict: T.Dict[str, T.Any],
    pcd_chunk_idxs: T.Optional[T.List[int]],
    rgbd_qidxs_dict: T.Dict[str, T.List[int]],
    pcd_attr_names: T.List[str] = ("xyz_w", "rgb", "normal_w"),
    rgbd_attr_names: T.List[str] = ("rgb", "depth", "hit_map", "normal_w"),
):
    """
    Load from the byte dict returned by webdataset.

    Args:
        byte_dict:
            key:
                the filename relative to index.json, folder structure ("/") is replaced by "-"
                e.g., "xyz_w-xyz_w_0.npy", 'rgbd_sphere-000000-rgb_000192.png'
            value:
                the byte content of the file
        pcd_chunk_idxs:
            indexes of the chunks to load.
            None to disable loading pcd.
        rgbd_qidxs_dict:
            mode (str) -> qidxs (list of int).
            e.g., 'sphere' -> [0, 4, 6]
            e.g., 'random' -> [3, 5]
        pcd_attr_names:
            'xyz_w', 'rgb', 'normal_w', attributes to load for pcd
        rgbd_attr_names:
            'rgb', 'depth', 'hit_map', 'normal_w', attributes to load for rgbd images

    Returns:
        xyz_w:
            (b=1, n, 3) tensor
        rgb:
            (b=1, n, 3) [0, 1]  tensor
        normal_w:
            (b=1, n, 3)  tensor
        rgbd_{name}:
            (b=1, q, h, w)  tensor
    """
    time_dict = dict()

    # get index_dict
    index_start_path = "."
    index_dict = load_file_from_byte_dict(
        byte_dict=byte_dict,
        filename="index.json",
        start_path=index_start_path,
    )

    # load pcd
    if pcd_chunk_idxs is None or len(pcd_chunk_idxs) == 0:
        pcd_dict = dict()
    else:
        pcd_save_version = index_dict.get("pcd_save_version", 1)
        stime = timer()
        if pcd_save_version == 1:
            # pcd_save_version=1 does not support chunking, load all points
            assert "pcd_filename" in index_dict
            pcd_dict = load_file_from_byte_dict(
                byte_dict=byte_dict,
                filename=index_dict["pcd_filename"],
                start_path=index_start_path,
            )
            # pcd_dict = {key: pcd_dict[key] for key in pcd_dict}

            pdict = dict()
            for attr_name in pcd_attr_names:
                assert attr_name in pcd_dict, f"{attr_name} not in pcd_dict {pcd_dict.keys()}"
                pdict[attr_name] = pcd_dict[attr_name]
            pcd_dict = utils.to_tensor(pdict, dtype=torch.float)
        elif pcd_save_version == 2:
            assert "pcd_index_filename" in index_dict
            pcd_index = load_file_from_byte_dict(
                byte_dict=byte_dict,
                filename=index_dict["pcd_index_filename"],
                start_path=index_start_path,
            )
            pcd_index_start = os.path.dirname(os.path.join(index_start_path, index_dict["pcd_index_filename"]))
            num_chunks = pcd_index["num_chunks"]

            pcd_dict = dict()
            for attr_name in pcd_attr_names:
                assert attr_name in pcd_index, f"{attr_name} not in pcd_index {pcd_index.keys()}"
                pcd_dict[attr_name] = []
                for pcd_chunk_idx in pcd_chunk_idxs:
                    assert pcd_chunk_idx < num_chunks, f"{pcd_chunk_idx}, {num_chunks}"
                    arr = load_file_from_byte_dict(
                        byte_dict=byte_dict,
                        filename=os.path.join(pcd_index_start, pcd_index[attr_name][pcd_chunk_idx]),
                        start_path=index_start_path,
                    )  # (m, 3)
                    arr = torch.from_numpy(arr).float()
                    pcd_dict[attr_name].append(arr)

            for attr_name in pcd_dict:
                pcd_dict[attr_name] = torch.cat(pcd_dict[attr_name], dim=0)  # (m, 3)
        else:
            raise NotImplementedError
        time_dict["open_pcd_npz"] = timer() - stime

    # load rgbd
    rgbd_dict = dict()  # rgbd_name -> rgbd_image
    for name, qidxs in rgbd_qidxs_dict.items():
        rgbd_name = f"rgbd_{name}"
        rgbd_index_filename = os.path.join(index_start_path, rgbd_name, "index.json")
        rgbd_start_path = os.path.dirname(rgbd_index_filename)
        rgbd_index_dict = load_file_from_byte_dict(
            byte_dict=byte_dict,
            filename=rgbd_index_filename,
            start_path=index_start_path,
        )
        bidx = 0
        sub_index_dict = rgbd_index_dict["sub_index_dicts"][bidx]  # attr_name -> (q,) filename (rel to rgbd_index)

        # load camera
        stime = timer()
        cam_filename = os.path.join(rgbd_start_path, rgbd_index_dict["camera"])
        camera_state_dict = load_file_from_byte_dict(
            byte_dict=byte_dict,
            filename=cam_filename,
            start_path=index_start_path,
        )
        H_c2w = np.stack([camera_state_dict["H_c2w"][bidx][qidx] for qidx in qidxs], axis=0)  # (q, 4, 4)
        H_c2w = torch.from_numpy(H_c2w).unsqueeze(0)  # (b=1, q, 4, 4)
        intrinsic = np.stack([camera_state_dict["intrinsic"][bidx][qidx] for qidx in qidxs], axis=0)  # (q, 3, 3)
        intrinsic = torch.from_numpy(intrinsic).unsqueeze(0)  # (b=1, q, 3, 3)
        if camera_state_dict.get("timestamp", None) is not None:
            timestamp = np.stack([camera_state_dict["timestamp"][bidx][qidx] for qidx in qidxs], axis=0)
            timestamp = torch.from_numpy(timestamp).unsqueeze(0)  # (b=1, q)
        else:
            timestamp = None
        camera = structures.Camera(
            H_c2w=H_c2w,  # (b=1, q, 4, 4)
            intrinsic=intrinsic,  # (b=1, q, 3, 3)
            width_px=camera_state_dict["width_px"],
            height_px=camera_state_dict["height_px"],
            timestamp=timestamp,  # (b=1, q) or None
        )  # (b=1, q)
        time_dict["read_camera"] = timer() - stime

        # load each attribute
        rgbd = dict()  # attr_name -> tensor (b, q, h, w)
        for attr_name in rgbd_attr_names:
            stime = timer()
            rgbd[attr_name] = []
            for iq, qidx in enumerate(qidxs):
                filename = os.path.join(rgbd_start_path, sub_index_dict[attr_name][qidx])
                arr = load_single_rgbd_file_from_byte_dict(
                    byte_dict=byte_dict,
                    filename=filename,
                    attr_name=attr_name,
                    start_path=index_start_path,
                )  # (h, w) or (h, w, d) torch.Tensor
                rgbd[attr_name].append(arr)
            time_dict[f"read_{rgbd_name}_{attr_name}"] = timer() - stime

            stime = timer()
            rgbd[attr_name] = torch.stack(rgbd[attr_name], dim=0).unsqueeze(0)  # (b=1, q, h, w, *)
            time_dict[f"stack_{rgbd_name}_{attr_name}"] = timer() - stime

        rgbd = structures.RGBDImage(**rgbd, camera=camera)  # (b=1, q, h, w)
        rgbd_dict[rgbd_name] = rgbd

    out_dict = dict(
        **pcd_dict,  # tensor
        **rgbd_dict,
        time_dict=time_dict,
    )

    return out_dict


def load_pcd_from_byte_dict(
    byte_dict: T.Dict[str, T.Any],
    bidxs: T.List[int] = None,
    # pcd_chunk_idxs: T.Optional[T.List[int]],
    attr_names: T.List[str] = ("xyz_w", "rgb", "normal_w"),
):
    """
    Load from the byte dict returned by webdataset.

    Args:
        byte_dict:
            key:
                the filename relative to index.json, folder structure ("/") is replaced by "-"
                e.g., "xyz_w-xyz_w_0.npy", 'rgbd_sphere-000000-rgb_000192.png'
            value:
                the byte content of the file
        bidxs:
            (list of int) for time indexes to fetch
        attr_names:
            'xyz_w', 'rgb', 'normal_w', attributes to load for pcd

    Returns:
        xyz_w:
            (b=len(bidxs), n, 3) tensor
        rgb:
            (b=len(bidxs), n, 3) [0, 1]  tensor
        normal_w:
            (b=len(bidxs), n, 3)  tensor
    """
    time_dict = dict()

    # get index_dict
    index_start_path = "."
    index_dict = load_file_from_byte_dict(
        byte_dict=byte_dict,
        filename="index.json",
        start_path=index_start_path,
    )
    assert "pcd_index_filename" in index_dict
    pcd_index = load_file_from_byte_dict(
        byte_dict=byte_dict,
        filename=index_dict["pcd_index_filename"],
        start_path=index_start_path,
    )

    pcd_index_start = os.path.dirname(os.path.join(index_start_path, index_dict["pcd_index_filename"]))

    pcd_dict = dict()
    for attr_name in attr_names:
        pcd_dict[attr_name] = []
        for ib in bidxs:
            assert attr_name in pcd_index[ib], f"{attr_name} not in pcd_index {pcd_index[ib].keys()}"

            arr = load_file_from_byte_dict(
                byte_dict=byte_dict,
                filename=os.path.join(pcd_index_start, pcd_index[ib][attr_name][0]),
                start_path=index_start_path,
            )  # (m, 3)
            arr = torch.from_numpy(arr).float()
            pcd_dict[attr_name].append(arr)

    for attr_name in pcd_dict:
        pcd_dict[attr_name] = torch.stack(pcd_dict[attr_name], dim=0)  # (b, m, 3)

    return pcd_dict


def sample_pcd_rgbd_from_mesh_with_blender_and_o3d(
    *,
    mesh_filename: str,
    out_dir: str,
    out_dir_rgbd: str = None,
    out_dir_pcd_rgbd_visibility_dict: T.Union[T.Dict[float, str], None] = None,
    light_type: str = "SUN",
    num_lights: int = 8,
    min_light_energy: float = 0.0,  # remember to adjust when num_lights change
    max_light_energy: float = 3.0,  # remember to adjust when num_lights change
    num_cells: int = 512,  # number of cells per side for voxel sampling
    width_px: int = 448,  # 14 x 32
    height_px: int = 448,  # 14 x 32
    # circular camera
    num_regular_images: int = 10,
    fov: float = 40.0,  # degree
    circular_radius: float = 3.5,  # meter
    # random camera
    num_random_images: int = 30,
    min_fov: float = 40.0,  # degree
    max_fov: float = 60.0,  # degree
    min_random_radius: float = 3,
    max_random_radius: float = 4,
    random_lookat_r: float = 0.25,
    mesh_rel_dir: str = None,
    background_color: float = 1,
    overwrite: bool = False,
    save_attr_names: T.List[str] = None,
    pcd_save_version: int = 2,
    save_np_dtype: np.dtype = np.float32,
    pcd_sample_method: str = "uniform",
    num_points: int = 100_000,
    max_time_to_sample_pcd: float = None,
    verbose: bool = True,
    seed: int = 0,
    pcd_save_chunk_size: int = 100_000,
    flag_debug: bool = False,
    flag_save_data_from_o3d: bool = False,
    flag_save_space: bool = True,
    pcd_rgbd_visibility_err_rtol_list: T.List[float] = [0],
    n_o3d_rendering_for_align_check: int = -1,
    **kwargs,
):
    """This function conducts the following steps:
    1. Use Blender to normalize objects and render RGB images;
    2. Use Open3D to directly sample 3D points ob normalized objects from Blender.
    """
    _set_seed(seed)

    out_dir = pathlib.Path(out_dir).absolute()
    out_dir_rgbd = pathlib.Path(out_dir_rgbd).absolute()
    out_dir_pcd_rgbd_visibility_dict = {
        k: pathlib.Path(v).absolute() for k, v in out_dir_pcd_rgbd_visibility_dict.items()
    }

    # run Blender rendering
    out_dir_blender = out_dir / "blender"
    out_dir_rgbd_blender = out_dir_rgbd / "blender"

    normalized_mesh_fname: str = "blender_normalized_mesh.ply"
    normalization_info_fname: str = "config_after_blender_normalization.json"

    blender_sample_ret_info, xyz_w_blender = sample_pcd_rgbd_from_mesh_with_blender(
        seed=seed,
        mesh_filename=mesh_filename,
        out_dir=out_dir_blender,
        out_dir_rgbd=out_dir_rgbd_blender,
        light_type=light_type,
        num_lights=num_lights,
        min_light_energy=min_light_energy,
        max_light_energy=max_light_energy,
        num_cells=num_cells,
        width_px=width_px,
        height_px=height_px,
        # circular camera
        num_regular_images=num_regular_images,
        fov=fov,  # degree
        circular_radius=circular_radius,  # meter
        # random camera
        num_random_images=num_random_images,
        min_fov=min_fov,  # degree
        max_fov=max_fov,  # degree
        min_random_radius=min_random_radius,
        max_random_radius=max_random_radius,
        random_lookat_r=random_lookat_r,
        mesh_rel_dir=mesh_rel_dir,
        background_color=background_color,
        overwrite=overwrite,
        save_attr_names=save_attr_names,
        pcd_save_version=pcd_save_version,
        save_np_dtype=save_np_dtype,
        normalized_mesh_fname=normalized_mesh_fname,
        normalization_info_fname=normalization_info_fname,
        flag_debug=flag_debug,
        regular_camera_sampling_type="sphere",
        flag_return_xyz_w=True,
        pcd_save_chunk_size=pcd_save_chunk_size,
        flag_save_space=flag_save_space,
    )

    index_dict_blender = blender_sample_ret_info["index_dict"]
    cam_name_dict: OrderedDict = blender_sample_ret_info["cam_name_dict"]

    normalized_mesh_f = out_dir_blender / normalized_mesh_fname
    normalization_info_f = out_dir_blender / normalization_info_fname

    with open(normalization_info_f, "r") as f:
        normalization_info = json.load(f)

    # NOTE: these cameras are for Blender / OpenGL instead of OpenCV
    camera_dicts = normalization_info["camera_dicts"]

    n_cameras = sum(list(cam_name_dict.values()))
    assert len(camera_dicts) == n_cameras, f"{len(camera_dicts)=}, {n_cameras=}"

    # run Open3D sampling
    out_dir_o3d = out_dir / "o3d"
    out_dir_rgbd_o3d = out_dir_rgbd / "o3d"

    # assert not flag_save_data_from_o3d, f"We currently use point cloud from depth maps rendered by Blender."

    if flag_save_data_from_o3d:
        out_dir_o3d.mkdir(parents=True, exist_ok=True)

        sample_dict_o3d = sample_pcd_from_mesh(
            mesh_filename=normalized_mesh_f,
            out_dir=out_dir_o3d,
            out_dir_rgbd=out_dir_rgbd_o3d,
            pcd_sample_method=pcd_sample_method,
            num_points=num_points,
            raise_error_if_no_color=False,
            overwrite=overwrite,
            max_time_to_sample_pcd=max_time_to_sample_pcd,
            # NOTE: it is important to set the scale and center_w to None
            # as this avoids Open3D from re-normalizing the mesh.
            # For some reasons, Open3D's bounding box computation is not the same as Blender.
            # Thus, even though we have already normalized the mesh in Blender,
            # Open3D will re-do it, causing discrepancies.
            mesh_scale=None,
            mesh_center_w=None,
            preprocess_mesh=True,
            compute_raycasting_scene=True,
        )

        # has_color_texture = sample_dict["has_color_texture"]
        st_mesh_o3d: structures.Mesh = sample_dict_o3d["st_mesh"]
        point_cloud_o3d: structures.PointCloud = sample_dict_o3d["point_cloud"]

        # re-save mesh from open3d for double check
        o3d_mesh_fname = "o3d_resaved_mesh.ply"
        o3d.io.write_triangle_mesh(out_dir_o3d / o3d_mesh_fname, st_mesh_o3d.mesh, write_ascii=True)

        assert (point_cloud_o3d.xyz_w.ndim == 3) and (point_cloud_o3d.xyz_w.shape[0] == 1), (
            f"{sample_pcd_from_mesh.shape=}"
        )
        xyz_w_for_visibility = point_cloud_o3d.xyz_w[0]
        out_dir_pcd_rgbd_visibility_dict = {k: v / "o3d" for k, v in out_dir_pcd_rgbd_visibility_dict.items()}

    else:
        odict = mesh_utils.load_mesh_using_trimesh(
            filename=normalized_mesh_f,
            raise_error_if_no_color=False,
        )
        o3d_mesh = odict["o3d_mesh"]
        # has_color_texture = odict["has_color_texture"]

        st_mesh_o3d = structures.Mesh(
            mesh=o3d_mesh,
            scale=None,
            center_w=None,
            preprocess_mesh=True,
            compute_raycasting_scene=True,
        )

        xyz_w_for_visibility = xyz_w_blender
        out_dir_pcd_rgbd_visibility_dict = {k: v / "blender" for k, v in out_dir_pcd_rgbd_visibility_dict.items()}

    def _check_aligned_rendering(
        f1: T.Union[str, pathlib.Path],
        f2: T.Union[str, pathlib.Path, None],
        data_type: str,
        mask: nptyping.NDArray | torch.Tensor | None = None,
        input_img2: nptyping.NDArray | torch.Tensor | None = None,
        real_check: bool = True,
    ):
        img1 = structures.RGBDImage.load_single_file(str(f1), data_type).detach().cpu().numpy()
        img2 = None
        flag_align = None
        diff_metric = None

        if real_check:
            if f2 is None:
                assert input_img2 is not None
                img2 = input_img2
            else:
                img2 = structures.RGBDImage.load_single_file(str(f2), data_type).detach().cpu().numpy()

            if data_type == "hit_map":
                assert img2.dtype in [bool, torch.bool], f"{img2.dtype=}"
                img2 = img2.astype(np.float32)
                assert img1.dtype == img2.dtype, f"{f1}, {img1.dtype=}, {img2.dtype=}"
            else:
                assert img1.dtype == img2.dtype, f"{f1}, {img1.dtype=}, {img2.dtype=}"
            assert np.all(img1.shape == img2.shape), f"{f1}, {img1.shape=}, {img2.shape=}"

            if mask is None:
                assert data_type == "hit_map", f"{data_type=}"
                img1_bool = img1.astype(bool)
                img2_bool = img2.astype(bool)
                img_intersect = np.logical_and(img1_bool, img2_bool)
                n_intersect = np.sum(img_intersect.astype(float))
                img_union = np.logical_or(img1_bool, img2_bool)
                n_union = np.sum(img_union.astype(float))
                iou = n_intersect / (n_union + np.finfo(np.float32).eps)
                # print(f"\n\n{iou=}, {img_intersect=}, {img_union=}\n\n")
                flag_align = bool(iou > 1 - 1e-2)
                diff_metric = {"iou": iou}

                mask = img_intersect
                if mask.ndim == 3:
                    assert mask.shape[2] == 1, f"{mask.shape=}"
                    mask = mask[..., 0]
                assert mask.ndim == 2, f"{mask.shape=}"
                assert mask.dtype == bool, f"{mask.dtype=}"
            else:
                masked_img1 = img1[mask]
                masked_img2 = img2[mask]
                diff_per_masked_pix = float(np.mean(np.abs(masked_img1 - masked_img2)))
                # assert np.allclose(img1, img2, rtol=1e-3, atol=1e-4), f"{diff=}, {f1=}, {f2=}"
                flag_align = bool(diff_per_masked_pix < 1e-2)
                diff_metric = {"diff_per_masked_pix": diff_per_masked_pix}
            if not flag_align:
                print(f"\n{diff_metric=}, {f1=}, {f2=}\n")

        return mask, flag_align, diff_metric, {str(f1): img1, str(f2): img2}

    def _save_visibility_mask(
        pcd_visibility_mask_dict: T.Dict[float, nptyping.NDArray],
        out_dir_dict: T.Dict[float, T.Union[str, pathlib.Path]],
        cam_set_name: str,
    ):
        num_chunks = (xyz_w_for_visibility.size(0) + pcd_save_chunk_size - 1) // pcd_save_chunk_size
        for err_rtol, pcd_visibility_mask in pcd_visibility_mask_dict.items():
            fdict = dict(chunk_size=pcd_save_chunk_size, num_chunks=num_chunks)
            tmp_out_dir = out_dir_dict[err_rtol] / cam_set_name
            tmp_out_dir.mkdir(parents=True, exist_ok=True)
            for chunk_idx in range(num_chunks):
                for tmp_arr_name_raw, tmp_arr in [
                    ["visibility", pcd_visibility_mask.detach().cpu().numpy()],
                ]:
                    tmp_arr_name = f"{tmp_arr_name_raw}_err_rtol_{err_rtol}"
                    _tmp_arr = tmp_arr[
                        chunk_idx * pcd_save_chunk_size : (chunk_idx + 1) * pcd_save_chunk_size
                    ]  # (m, 3)
                    filename = os.path.join(tmp_out_dir, tmp_arr_name, f"{tmp_arr_name_raw}_{chunk_idx}.npy")
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    np.save(filename, _tmp_arr)
                    if tmp_arr_name not in fdict:
                        fdict[tmp_arr_name] = []
                    fdict[tmp_arr_name].append(os.path.relpath(filename, start=tmp_out_dir))
            pcd_index_filename = os.path.join(tmp_out_dir, "pcd_rgbd_visibility_index.json")
            with open(pcd_index_filename, "w") as f:
                json.dump(fdict, f, indent=2)

    def _render_o3d_with_poses_from_blender(
        *,
        st_mesh_o3d: structures.Mesh,
        cam_blender_list: T.List[nptyping.NDArray],
        cam_mat_dict: T.Dict[str, T.List[nptyping.NDArray]],
        cam_set_name: T.Dict[str, int],
        background_color: nptyping.NDArray,
        n_renderings: int,
    ):
        cam_o3d_list = []

        for cam_blender_dict in tqdm.tqdm(
            cam_blender_list, disable=not verbose, desc=f"cam_name_dict | {cam_set_name}"
        ):
            tmp_cam_o3d_dict = blender_open3d_utils.convert_blender_camera_to_open3d(
                H_c2w=cam_blender_dict["H_c2w"],  # (4, 4)
                intrinsic=cam_blender_dict["intrinsic"],  # (3, 3)
                width_px=cam_blender_dict["width_px"],
                height_px=cam_blender_dict["height_px"],
            )
            cam_o3d_list.append(tmp_cam_o3d_dict)

        cam_set_name_cam_dict = utils.list_of_dicts_to_dict_of_lists(cam_o3d_list)
        cam_key = f"rgbd_{cam_set_name}"
        assert cam_key not in cam_mat_dict, f"{cam_key=}, {list(cam_mat_dict.keys())=}"
        cam_mat_dict[cam_key] = cam_set_name_cam_dict

        tmp_cam = structures.Camera(
            H_c2w=torch.FloatTensor(np.array(cam_set_name_cam_dict["H_c2w"]))[None, :n_renderings, ...],  # (1, n, 4, 4)
            intrinsic=torch.FloatTensor(np.array(cam_set_name_cam_dict["intrinsic"]))[
                None, :n_renderings, ...
            ],  # (1, 1, 3, 3)
            width_px=cam_set_name_cam_dict["width_px"][0],
            height_px=cam_set_name_cam_dict["height_px"][0],
        )

        # capture images
        cam_set_name_rgbd_image = st_mesh_o3d.get_rgbd_image(
            camera=tmp_cam,  # (1, q)
            render_method="ray_cast",
        )

        # rgbd: structures.RGBDImage
        cam_set_name_rgbd_image = cam_set_name_rgbd_image.remove_invalid(
            min_depth=0,
            max_depth=1e4,
            background_color=background_color,
        )
        return cam_set_name_rgbd_image, cam_mat_dict

    if flag_save_data_from_o3d:
        index_dict_o3d = {}
        index_dict_o3d["mesh_filename"] = str(normalized_mesh_f)

    from_idx = 0
    opencv_cam_mat_dict = {}

    # check that Blender and Open3D renderings are aligned
    flag_align = True
    align_dict = {"agg": {}}

    for tmp_set_name in tqdm.tqdm(cam_name_dict, disable=not verbose, desc="cam_name_dict"):
        tmp_n = cam_name_dict[tmp_set_name]
        tmp_cam_blender_list = camera_dicts[from_idx : (from_idx + tmp_n)]

        from_idx += cam_name_dict[tmp_set_name]

        # render images with Open3D
        tmp_o3d_render_results = _render_o3d_with_poses_from_blender(
            st_mesh_o3d=st_mesh_o3d,
            cam_blender_list=tmp_cam_blender_list,
            cam_mat_dict=opencv_cam_mat_dict,
            cam_set_name=tmp_set_name,
            background_color=background_color,
            n_renderings=n_o3d_rendering_for_align_check,
        )
        tmp_name_rgbd_image_o3d: structures.RGBDImage = tmp_o3d_render_results[0]
        opencv_cam_mat_dict: T.Dict[str, nptyping.NDArray] = tmp_o3d_render_results[1]

        tmp_arr_for_shape_check = tmp_name_rgbd_image_o3d.hit_map
        assert (tmp_arr_for_shape_check.ndim == 4) and (tmp_arr_for_shape_check.shape[0] == 1), (
            f"we assume batch size is 1 but got {tmp_arr_for_shape_check.shape=}."
        )
        assert tmp_arr_for_shape_check.shape[1] == min(tmp_n, n_o3d_rendering_for_align_check), (
            f"{tmp_arr_for_shape_check.shape=}, {tmp_n=}, {n_o3d_rendering_for_align_check=}"
        )

        # check alignment between renderings from Open3D and Blender
        tmp_name = f"rgbd_{tmp_set_name}"
        align_dict[tmp_name] = {}

        tmp_name_data_dict_blender = {}

        # tmp_name_rgbd_dir_o3d = out_dir_rgbd_o3d / tmp_name
        tmp_name_rgbd_dir_blender = out_dir_rgbd_blender / tmp_name

        tmp_name_rgbd_json_f = out_dir_rgbd_blender / index_dict_blender[tmp_name]["index_filename"]
        with open(tmp_name_rgbd_json_f, "r") as f:
            tmp_name_rgbd_json = json.load(f)

        tmp_sub_index_dicts = tmp_name_rgbd_json["sub_index_dicts"]
        assert len(tmp_sub_index_dicts) == 1, f"{len(tmp_sub_index_dicts)=}"

        if n_o3d_rendering_for_align_check < 0:
            tmp_n_from_blender = len(tmp_sub_index_dicts[0]["hit_map"])
            assert tmp_name_rgbd_image_o3d.hit_map.shape[1] == tmp_n_from_blender, (
                f"{tmp_name_rgbd_image_o3d.hit_map.shape=}, {tmp_n_from_blender=}"
            )

        for tmp_i in range(tmp_n):
            if n_o3d_rendering_for_align_check <= 0:
                tmp_flag_real_check = False
            else:
                tmp_flag_real_check = tmp_i < n_o3d_rendering_for_align_check

            tmp_mask = None

            # hit_map must be the 1st one in order to get the maks
            for tmp_render_type in ["hit_map", "depth"]:
                if tmp_render_type not in tmp_name_data_dict_blender:
                    tmp_name_data_dict_blender[tmp_render_type] = []

                if tmp_render_type not in align_dict["agg"]:
                    align_dict["agg"][tmp_render_type] = True
                if tmp_render_type not in align_dict[tmp_name]:
                    align_dict[tmp_name][tmp_render_type] = {}

                tmp_render_fname = tmp_sub_index_dicts[0][tmp_render_type][tmp_i]

                if tmp_flag_real_check:
                    tmp_render_o3d = getattr(tmp_name_rgbd_image_o3d, tmp_render_type)[0, tmp_i].cpu().numpy()
                else:
                    tmp_render_o3d = None

                # tmp_render_f_o3d = tmp_name_rgbd_dir_o3d / tmp_render_fname
                tmp_render_f_blender = tmp_name_rgbd_dir_blender / tmp_render_fname
                tmp_mask, tmp_flag_align, tmp_diff_metric, tmp_single_data_dict = _check_aligned_rendering(
                    tmp_render_f_blender,
                    None,
                    tmp_render_type,
                    mask=tmp_mask,
                    input_img2=tmp_render_o3d,
                    real_check=tmp_flag_real_check,
                )

                if tmp_flag_real_check:
                    align_dict["agg"][tmp_render_type] = align_dict["agg"][tmp_render_type] and tmp_flag_align
                    if not tmp_flag_align:
                        flag_align = False
                    tmp_rel_path = tmp_render_f_blender.relative_to(out_dir_rgbd)
                    align_dict[tmp_name][tmp_render_type][str(tmp_rel_path)] = {
                        "align": tmp_flag_align,
                        "diff_metric": tmp_diff_metric,
                    }

                tmp_name_data_dict_blender[tmp_render_type].append(tmp_single_data_dict[str(tmp_render_f_blender)])

        # We project points onto rendered images, interpolate RGB, and then average across visible views.
        assert (xyz_w_for_visibility.ndim == 2) and (xyz_w_for_visibility.shape[1] == 3), (
            f"{xyz_w_for_visibility.shape=}"
        )
        tmp_depth_blender = torch.FloatTensor(tmp_name_data_dict_blender["depth"])
        tmp_mask_blender = torch.FloatTensor(tmp_name_data_dict_blender["hit_map"]).bool()
        tmp_H_c2w = torch.FloatTensor(opencv_cam_mat_dict[tmp_name]["H_c2w"])
        tmp_intrinsic = torch.FloatTensor(opencv_cam_mat_dict[tmp_name]["intrinsic"])

        # print(
        #     f"\n{tmp_depth_blender.shape=}, {tmp_depth_blender.dtype=}, "
        #     f"{tmp_mask_blender.shape=}, {tmp_mask_blender.dtype=}, "
        #     f"{tmp_H_c2w.shape=}, {tmp_H_c2w.dtype=}, "
        #     f"{tmp_intrinsic.shape=}, {tmp_intrinsic.dtype=}, "
        # )

        pcd_visibility_mask_dict = utils.compute_visibility_mask_for_pcd_with_depth(
            xyz_w=xyz_w_for_visibility[None, ...],  # (b, n, 3xyz_w)
            z_c=tmp_depth_blender[None, ...],  # (b, q, h, w)
            H_c2w=tmp_H_c2w[None, ...],  # (b, q, 4, 4)
            intrinsic=tmp_intrinsic[None, ...],  # (b, q, 3, 3)
            hit_map=tmp_mask_blender[None, ...],  # (b, q, h, w)
            err_rtol_list=pcd_rgbd_visibility_err_rtol_list,
        )
        pcd_visibility_mask_dict = {
            k: v[0, ...].permute(1, 0).contiguous() for k, v in pcd_visibility_mask_dict.items()
        }  # [#img, #point] -> [#point, #img], bool

        _save_visibility_mask(pcd_visibility_mask_dict, out_dir_pcd_rgbd_visibility_dict, tmp_name)

        if flag_save_data_from_o3d:
            _, tmp_sub_index_f = tmp_name_rgbd_image_o3d.save_as(
                out_dir=out_dir_rgbd_o3d / f"rgbd_{tmp_name}",
                overwrite=overwrite,
                mode="png",
                background_color=background_color,
                save_attr_names=save_attr_names,
                flag_save_space=flag_save_space,
            )
            index_dict_o3d[f"rgbd_{tmp_name}"] = dict(
                index_filename=os.path.relpath(tmp_sub_index_f, start=out_dir_rgbd_o3d),
                q=tmp_name_rgbd_image_o3d.rgb.size(1),
                h=tmp_name_rgbd_image_o3d.rgb.size(2),
                w=tmp_name_rgbd_image_o3d.rgb.size(3),
            )

    if flag_save_data_from_o3d:
        # shuffle point cloud before saving (added when pcd_save_version = 2 is added)
        assert (point_cloud_o3d.xyz_w.ndim == 3) and (point_cloud_o3d.xyz_w.shape[0] == 1), (
            f"{point_cloud_o3d.xyz_w.shape=}"
        )
        ridxs_o3d = torch.randperm(point_cloud_o3d.xyz_w.size(1), device=point_cloud_o3d.xyz_w.device)
        xyz_w_o3d = point_cloud_o3d.xyz_w[0][ridxs_o3d]  # (n, 3)
        normal_w_o3d = point_cloud_o3d.normal_w[0][ridxs_o3d]  # (n, 3)
        rgb_o3d = point_cloud_o3d.rgb[0][ridxs_o3d]  # (n, 3)

        index_dict_o3d = save_sampled_pcd(
            pcd_save_version=pcd_save_version,
            out_dir=str(out_dir_o3d),
            index_dict=index_dict_o3d,
            xyz_w=xyz_w_o3d,
            rgb=rgb_o3d,
            normal_w=normal_w_o3d,
            save_np_dtype=save_np_dtype,
            save_chunk_size=pcd_save_chunk_size,
        )

        # save json
        json_filename = os.path.join(out_dir_o3d, "index.json")
        with open(json_filename, "w") as f:
            json.dump(index_dict_o3d, f, indent=2)

        if out_dir_rgbd_o3d != out_dir_o3d:
            _json_filename = os.path.join(out_dir_rgbd_o3d, "index.json")
            with open(_json_filename, "w") as f:
                json.dump(index_dict_o3d, f, indent=2)

        if flag_save_space:  # and (not flag_debug):
            print(f"\ndelete {out_dir_rgbd_o3d=}\n")
            shutil.rmtree(out_dir_rgbd_o3d)

    align_dict["align"] = flag_align
    with open(out_dir_rgbd / "align.json", "w") as f:
        json.dump(align_dict, f, indent=2, sort_keys=True)


'''
def sample_pcd_rgbd_from_mesh_with_blender_and_o3d_v1_old(
    *,
    mesh_filename: str,
    out_dir: str,
    out_dir_rgbd: str = None,
    out_dir_pcd_rgbd_visibility_dict: T.Dict[float, str] | None = None,
    light_type: str = "SUN",
    num_lights: int = 8,
    min_light_energy: float = 0.0,  # remember to adjust when num_lights change
    max_light_energy: float = 3.0,  # remember to adjust when num_lights change
    num_cells: int = 512,  # number of cells per side for voxel sampling
    width_px: int = 448,  # 14 x 32
    height_px: int = 448,  # 14 x 32
    # circular camera
    num_regular_images: int = 10,
    fov: float = 40.0,  # degree
    circular_radius: float = 3.5,  # meter
    # random camera
    num_random_images: int = 30,
    min_fov: float = 40.0,  # degree
    max_fov: float = 60.0,  # degree
    min_random_radius: float = 3,
    max_random_radius: float = 4,
    random_lookat_r: float = 0.25,
    mesh_rel_dir: str = None,
    background_color: float = 1,
    overwrite: bool = False,
    save_attr_names: T.List[str] = None,
    pcd_save_version: int = 2,
    save_np_dtype: np.dtype = np.float32,
    pcd_sample_method: str = "uniform",
    num_points: int = 100_000,
    max_time_to_sample_pcd: float = None,
    verbose: bool = True,
    seed: int = 0,
    pcd_save_chunk_size: int = 100_000,
    flag_debug: bool = False,
    flag_save_data_from_o3d: bool = False,
    flag_save_space: bool = False,
):
    """This function conducts the following steps:
    1. Use Blender to normalize objects and render RGB images;
    2. Use Open3D to directly sample 3D points ob normalized objects from Blender.

    This old version first saves Open3D results and then load them to compare, 
    which is not that efficient.
    """
    _set_seed(seed)

    out_dir = pathlib.Path(out_dir).absolute()
    out_dir_rgbd = pathlib.Path(out_dir_rgbd).absolute()
    out_dir_pcd_rgbd_visibility = pathlib.Path(out_dir_pcd_rgbd_visibility).absolute()

    # run Blender rendering
    out_dir_blender = out_dir / "blender"
    out_dir_rgbd_blender = out_dir_rgbd / "blender"

    normalized_mesh_fname: str = "blender_normalized_mesh.ply"
    normalization_info_fname: str = "config_after_blender_normalization.json"

    blender_sample_ret_info, xyz_w_blender = sample_pcd_rgbd_from_mesh_with_blender(
        seed=seed,
        mesh_filename=mesh_filename,
        out_dir=out_dir_blender,
        out_dir_rgbd=out_dir_rgbd_blender,
        light_type=light_type,
        num_lights=num_lights,
        min_light_energy=min_light_energy,
        max_light_energy=max_light_energy,
        num_cells=num_cells,
        width_px=width_px,
        height_px=height_px,
        # circular camera
        num_regular_images=num_regular_images,
        fov=fov,  # degree
        circular_radius=circular_radius,  # meter
        # random camera
        num_random_images=num_random_images,
        min_fov=min_fov,  # degree
        max_fov=max_fov,  # degree
        min_random_radius=min_random_radius,
        max_random_radius=max_random_radius,
        random_lookat_r=random_lookat_r,
        mesh_rel_dir=mesh_rel_dir,
        background_color=background_color,
        overwrite=overwrite,
        save_attr_names=save_attr_names,
        pcd_save_version=pcd_save_version,
        save_np_dtype=save_np_dtype,
        normalized_mesh_fname=normalized_mesh_fname,
        normalization_info_fname=normalization_info_fname,
        flag_debug=flag_debug,
        regular_camera_sampling_type="sphere",
        flag_return_xyz_w=True,
        pcd_save_chunk_size=pcd_save_chunk_size,
        flag_save_space=flag_save_space,
    )

    cam_name_dict: OrderedDict = blender_sample_ret_info["cam_name_dict"]

    normalized_mesh_f = out_dir_blender / normalized_mesh_fname
    normalization_info_f = out_dir_blender / normalization_info_fname

    with open(normalization_info_f, "r") as f:
        normalization_info = json.load(f)

    # NOTE: these cameras are for Blender / OpenGL instead of OpenCV
    camera_dicts = normalization_info["camera_dicts"]

    n_cameras = sum(list(cam_name_dict.values()))
    assert len(camera_dicts) == n_cameras, f"{len(camera_dicts)=}, {n_cameras=}"

    # run Open3D sampling
    out_dir_o3d = out_dir / "o3d"
    out_dir_rgbd_o3d = out_dir_rgbd / "o3d"

    assert not flag_save_data_from_o3d, f"We currently use point cloud from depth maps rendered by Blender."

    if flag_save_data_from_o3d:
        out_dir_o3d.mkdir(parents=True, exist_ok=True)

        sample_dict = sample_pcd_from_mesh(
            mesh_filename=normalized_mesh_f,
            out_dir=out_dir_o3d,
            out_dir_rgbd=out_dir_rgbd_o3d,
            pcd_sample_method=pcd_sample_method,
            num_points=num_points,
            raise_error_if_no_color=False,
            overwrite=overwrite,
            max_time_to_sample_pcd=max_time_to_sample_pcd,
            # NOTE: it is important to set the scale and center_w to None
            # as this avoids Open3D from re-normalizing the mesh.
            # For some reasons, Open3D's bounding box computation is not the same as Blender.
            # Thus, even though we have already normalized the mesh in Blender,
            # Open3D will re-do it, causing discrepancies.
            mesh_scale=None,
            mesh_center_w=None,
            preprocess_mesh=True,
            compute_raycasting_scene=True,
        )

        # has_color_texture = sample_dict["has_color_texture"]
        st_mesh_o3d: structures.Mesh = sample_dict["st_mesh"]
        point_cloud_o3d: structures.PointCloud = sample_dict["point_cloud"]

        # re-save mesh from open3d for double check
        o3d_mesh_fname = "o3d_resaved_mesh.ply"
        o3d.io.write_triangle_mesh(out_dir_o3d / o3d_mesh_fname, st_mesh.mesh, write_ascii=True)

        xyz_w_for_visibility = point_cloud_o3d.xyz_w
        out_dir_pcd_rgbd_visibility = out_dir_pcd_rgbd_visibility / "o3d"
    else:
        odict = mesh_utils.load_mesh_using_trimesh(
            filename=normalized_mesh_f,
            raise_error_if_no_color=False,
        )
        o3d_mesh = odict["o3d_mesh"]
        # has_color_texture = odict["has_color_texture"]

        st_mesh_o3d = structures.Mesh(
            mesh=o3d_mesh,
            scale=None,
            center_w=None,
            preprocess_mesh=True,
            compute_raycasting_scene=True,
        )

        xyz_w_for_visibility = xyz_w_blender
        out_dir_pcd_rgbd_visibility = out_dir_pcd_rgbd_visibility / "blender"

    index_dict = {}
    index_dict["mesh_filename"] = str(normalized_mesh_f)

    from_idx = 0
    opencv_cam_mat_dict = {}

    for tmp_name in tqdm.tqdm(cam_name_dict, disable=not verbose, desc="cam_name_dict"):
        tmp_n = cam_name_dict[tmp_name]
        tmp_cam_blender_list = camera_dicts[from_idx : (from_idx + tmp_n)]

        from_idx += cam_name_dict[tmp_name]

        tmp_cam_o3d_list = []

        for tmp_cam_blender_dict in tqdm.tqdm(
            tmp_cam_blender_list, disable=not verbose, desc=f"cam_name_dict | {tmp_name}"
        ):
            tmp_cam_o3d_dict = blender_open3d_utils.convert_blender_camera_to_open3d(
                H_c2w=tmp_cam_blender_dict["H_c2w"],  # (4, 4)
                intrinsic=tmp_cam_blender_dict["intrinsic"],  # (3, 3)
                width_px=tmp_cam_blender_dict["width_px"],
                height_px=tmp_cam_blender_dict["height_px"],
            )
            tmp_cam_o3d_list.append(tmp_cam_o3d_dict)

        tmp_name_cam_dict = utils.list_of_dicts_to_dict_of_lists(tmp_cam_o3d_list)
        opencv_cam_mat_dict[f"rgbd_{tmp_name}"] = tmp_name_cam_dict

        tmp_cam = structures.Camera(
            H_c2w=torch.FloatTensor(np.array(tmp_name_cam_dict["H_c2w"]))[None, ...],  # (1, n, 4, 4)
            intrinsic=torch.FloatTensor(np.array(tmp_name_cam_dict["intrinsic"]))[None, ...],  # (1, 1, 3, 3)
            width_px=tmp_name_cam_dict["width_px"][0],
            height_px=tmp_name_cam_dict["height_px"][0],
        )

        # capture images
        tmp_name_rgbd_image_o3d = st_mesh_o3d.get_rgbd_image(
            camera=tmp_cam,  # (1, q)
            render_method="ray_cast",
        )

        # rgbd: structures.RGBDImage
        tmp_name_rgbd_image_o3d = tmp_name_rgbd_image_o3d.remove_invalid(
            min_depth=0,
            max_depth=1e4,
            background_color=background_color,
        )

        _, tmp_sub_index_f = tmp_name_rgbd_image_o3d.save_as(
            out_dir=out_dir_rgbd_o3d / f"rgbd_{tmp_name}",
            overwrite=overwrite,
            mode="png",
            background_color=background_color,
            save_attr_names=save_attr_names,
            flag_save_space=flag_save_space,
        )
        index_dict[f"rgbd_{tmp_name}"] = dict(
            index_filename=os.path.relpath(tmp_sub_index_f, start=out_dir_rgbd_o3d),
            q=tmp_name_rgbd_image_o3d.rgb.size(1),
            h=tmp_name_rgbd_image_o3d.rgb.size(2),
            w=tmp_name_rgbd_image_o3d.rgb.size(3),
        )

    if flag_save_data_from_o3d:
        # shuffle point cloud before saving (added when pcd_save_version = 2 is added)
        assert (point_cloud_o3d.xyz_w.ndim == 3) and (point_cloud_o3d.xyz_w.shape[0] == 1), (
            f"{point_cloud_o3d.xyz_w.shape=}"
        )
        ridxs_o3d = torch.randperm(point_cloud_o3d.xyz_w.size(1), device=point_cloud_o3d.xyz_w.device)
        xyz_w_o3d = point_cloud_o3d.xyz_w[0][ridxs_o3d]  # (n, 3)
        normal_w_o3d = point_cloud_o3d.normal_w[0][ridxs_o3d]  # (n, 3)
        rgb_o3d = point_cloud_o3d.rgb[0][ridxs_o3d]  # (n, 3)

        index_dict = save_sampled_pcd(
            pcd_save_version=pcd_save_version,
            out_dir=str(out_dir_o3d),
            index_dict=index_dict,
            xyz_w=xyz_w_o3d,
            rgb=rgb_o3d,
            normal_w=normal_w_o3d,
            save_np_dtype=save_np_dtype,
            save_chunk_size=pcd_save_chunk_size,
        )

        # save json
        json_filename = os.path.join(out_dir_o3d, "index.json")
        with open(json_filename, "w") as f:
            json.dump(index_dict, f, indent=2)

    if out_dir_rgbd_o3d != out_dir_o3d:
        _json_filename = os.path.join(out_dir_rgbd_o3d, "index.json")
        with open(_json_filename, "w") as f:
            json.dump(index_dict, f, indent=2)

    def _check_aligned_rendering(
        f1: str | pathlib.Path, f2: str | pathlib.Path, data_type: str, mask: nptyping.NDArray | None = None
    ):
        img1 = structures.RGBDImage.load_single_file(str(f1), data_type)
        img2 = structures.RGBDImage.load_single_file(str(f2), data_type)

        if mask is None:
            assert data_type == "hit_map", f"{data_type=}"
            img1_bool = img1.astype(bool)
            img2_bool = img2.astype(bool)
            img_intersect = np.logical_and(img1_bool, img2_bool)
            n_intersect = np.sum(img_intersect.astype(float))
            img_union = np.logical_or(img1_bool, img2_bool)
            n_union = np.sum(img_union.astype(float))
            iou = n_intersect / (n_union + np.finfo(np.float32).eps)
            # print(f"\n\n{iou=}, {img_intersect=}, {img_union=}\n\n")
            flag_align = bool(iou > 1 - 1e-2)
            diff_metric = {"iou": iou}

            mask = img_intersect
            if mask.ndim == 3:
                assert mask.shape[2] == 1, f"{mask.shape=}"
                mask = mask[..., 0]
            assert mask.ndim == 2, f"{mask.shape=}"
            assert mask.dtype == bool, f"{mask.dtype=}"
        else:
            masked_img1 = img1[mask]
            masked_img2 = img2[mask]
            diff_per_masked_pix = float(np.mean(np.abs(masked_img1 - masked_img2)))
            # assert np.allclose(img1, img2, rtol=1e-3, atol=1e-4), f"{diff=}, {f1=}, {f2=}"
            flag_align = bool(diff_per_masked_pix < 1e-2)
            diff_metric = {"diff_per_masked_pix": diff_per_masked_pix}
        if not flag_align:
            print(f"\n{diff_metric=}, {f1=}, {f2=}\n")

        return mask, flag_align, diff_metric, {str(f1): img1, str(f2): img2}

    def _save_visibility_mask(pcd_visibility_mask_dict: T.Dict[float, nptyping.NDArray], out_dir: str | pathlib.Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        num_chunks = (xyz_w_for_visibility.size(0) + pcd_save_chunk_size - 1) // pcd_save_chunk_size
        fdict = dict(chunk_size=pcd_save_chunk_size, num_chunks=num_chunks)
        for err_rtol, pcd_visibility_mask in pcd_visibility_mask_dict.items():
            for chunk_idx in range(num_chunks):
                for tmp_arr_name_raw, tmp_arr in [
                    ["visibility", pcd_visibility_mask.detach().cpu().numpy()],
                ]:
                    tmp_arr_name = f"{tmp_arr_name_raw}_err_rtol_{err_rtol}"
                    _tmp_arr = tmp_arr[
                        chunk_idx * pcd_save_chunk_size : (chunk_idx + 1) * pcd_save_chunk_size
                    ]  # (m, 3)
                    filename = os.path.join(out_dir, tmp_arr_name, f"{tmp_arr_name_raw}_{chunk_idx}.npy")
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    np.save(filename, _tmp_arr)
                    if tmp_arr_name not in fdict:
                        fdict[tmp_arr_name] = []
                    fdict[tmp_arr_name].append(os.path.relpath(filename, start=out_dir))
        pcd_index_filename = os.path.join(out_dir, "pcd_rgbd_visibility_index.json")
        with open(pcd_index_filename, "w") as f:
            json.dump(fdict, f, indent=2)

    # check that Blender and Open3D renderings are aligned
    flag_align = True
    align_dict = {"agg": {}}

    for tmp_name in tqdm.tqdm(cam_name_dict, disable=not verbose, desc="cam_name_dict align check"):
        tmp_name = f"rgbd_{tmp_name}"
        align_dict[tmp_name] = {}

        tmp_name_data_dict_blender = {}

        tmp_name_rgbd_dir_o3d = out_dir_rgbd_o3d / tmp_name
        tmp_name_rgbd_dir_blender = out_dir_rgbd_blender / tmp_name

        tmp_name_rgbd_json_f = out_dir_rgbd_o3d / index_dict[tmp_name]["index_filename"]
        with open(tmp_name_rgbd_json_f, "r") as f:
            tmp_name_rgbd_json = json.load(f)

        tmp_sub_index_dicts = tmp_name_rgbd_json["sub_index_dicts"]
        assert len(tmp_sub_index_dicts) == 1, f"{len(tmp_sub_index_dicts)=}"

        tmp_n = len(tmp_sub_index_dicts[0]["hit_map"])

        for tmp_i in range(tmp_n):
            tmp_mask = None

            # hit_map must be the 1st one in order to get the maks
            for tmp_render_type in ["hit_map", "depth"]:
                if tmp_render_type not in tmp_name_data_dict_blender:
                    tmp_name_data_dict_blender[tmp_render_type] = []

                if tmp_render_type not in align_dict["agg"]:
                    align_dict["agg"][tmp_render_type] = True
                if tmp_render_type not in align_dict[tmp_name]:
                    align_dict[tmp_name][tmp_render_type] = {}

                tmp_render_fname = tmp_sub_index_dicts[0][tmp_render_type][tmp_i]

                tmp_render_f_o3d = tmp_name_rgbd_dir_o3d / tmp_render_fname
                tmp_render_f_blender = tmp_name_rgbd_dir_blender / tmp_render_fname
                tmp_mask, tmp_flag_align, tmp_diff_metric, tmp_single_data_dict = _check_aligned_rendering(
                    tmp_render_f_o3d,
                    tmp_render_f_blender,
                    tmp_render_type,
                    mask=tmp_mask,
                )

                tmp_name_data_dict_blender[tmp_render_type].append(tmp_single_data_dict[str(tmp_render_f_blender)])

                align_dict["agg"][tmp_render_type] = align_dict["agg"][tmp_render_type] and tmp_flag_align
                if not tmp_flag_align:
                    flag_align = False
                tmp_rel_path = tmp_render_f_o3d.relative_to(out_dir_rgbd)
                align_dict[tmp_name][tmp_render_type][str(tmp_rel_path)] = {
                    "align": tmp_flag_align,
                    "diff_metric": tmp_diff_metric,
                }

        # We project points onto rendered images, interpolate RGB, and then average across visible views.
        assert (xyz_w_for_visibility.ndim == 2) and (xyz_w_for_visibility.shape[1] == 3), (
            f"{xyz_w_for_visibility.shape=}"
        )
        tmp_depth_blender = torch.FloatTensor(tmp_name_data_dict_blender["depth"])
        tmp_mask_blender = torch.FloatTensor(tmp_name_data_dict_blender["hit_map"]).bool()
        tmp_H_c2w = torch.FloatTensor(opencv_cam_mat_dict[tmp_name]["H_c2w"])
        tmp_intrinsic = torch.FloatTensor(opencv_cam_mat_dict[tmp_name]["intrinsic"])

        # print(
        #     f"\n{tmp_depth_blender.shape=}, {tmp_depth_blender.dtype=}, "
        #     f"{tmp_mask_blender.shape=}, {tmp_mask_blender.dtype=}, "
        #     f"{tmp_H_c2w.shape=}, {tmp_H_c2w.dtype=}, "
        #     f"{tmp_intrinsic.shape=}, {tmp_intrinsic.dtype=}, "
        # )

        pcd_visibility_mask_dict = utils.compute_visibility_mask_for_pcd_with_depth(
            xyz_w=xyz_w_for_visibility[None, ...],  # (b, n, 3xyz_w)
            z_c=tmp_depth_blender[None, ...],  # (b, q, h, w)
            H_c2w=tmp_H_c2w[None, ...],  # (b, q, 4, 4)
            intrinsic=tmp_intrinsic[None, ...],  # (b, q, 3, 3)
            hit_map=tmp_mask_blender[None, ...],  # (b, q, h, w)
            err_rtol_list=[0, 0.001, 0.005, 0.01, 0.05],
        )
        pcd_visibility_mask_dict = {
            k: v[0, ...].permute(1, 0).contiguous() for k, v in pcd_visibility_mask_dict.items()
        }  # [#img, #point] -> [#point, #img], bool

        _save_visibility_mask(pcd_visibility_mask_dict, out_dir_pcd_rgbd_visibility / tmp_name)

    if (not flag_save_data_from_o3d) and (flag_save_space):  # and (not flag_debug):
        print(f"\ndelete {out_dir_rgbd_o3d=}\n")
        shutil.rmtree(out_dir_rgbd_o3d)

    align_dict["align"] = flag_align
    with open(out_dir_rgbd / "align.json", "w") as f:
        json.dump(align_dict, f, indent=2, sort_keys=True)
'''


def run_blender(
    *,
    blender_run_dir: str,
    mesh_filename: str,
    json_filename: str,
    normalized_mesh_fname: str,
    normalization_info_fname: str,
    debug: bool,
    printout: bool,
    blender_version: str = "4.2.0",
    blender_script_filename: str = "blender_utils.py",
    blender_file_to_be_before_python_script: T.Optional[str] = None,
    blender_download_from_s3: bool = False,
    blender_save_scene_format: str | None = None,
    blender_resume_info_f: str | None = None,
    device: str = "CPU",
):
    """This function run Blender to render images/depths etc according to json_filename.

    Args:
        blender_run_dir (str):
            directory for saving results
        mesh_filename (str):
            3D asset file, this is only used for configuring log filename
        json_filename (str):
            json file that contains configuration for Blender rendering
        normalized_mesh_fname (str):
            filename for saved normalized mesh
        normalization_info_fname (str):
            filename for saved normalization information from Blender
        printout (bool):
            if True, we print out Blender logging instead of saving it
        blender_resume_info_f (str):
            if not None, we will append this to the command line argument and let Blender script re-use information
            in the file, e.g., normalization and lightings
        device:
            'CPU', 'GPU'
    """
    os.makedirs(blender_run_dir, exist_ok=True)
    blender_cmd = blender_rendering_utils.get_blender_exe(version=blender_version)
    blender_script = blender_rendering_utils.get_blender_utils_path(blender_script_filename=blender_script_filename)
    blender_log_fname = f"blender_{pathlib.Path(mesh_filename).stem}.log"
    blender_log_f = os.path.join(blender_run_dir, blender_log_fname)
    normalized_mesh_f: str = os.path.join(blender_run_dir, normalized_mesh_fname)
    normalization_info_f: str = os.path.join(blender_run_dir, normalization_info_fname)

    if blender_file_to_be_before_python_script is None:
        cmd = f"{blender_cmd} "
    else:
        # https://github.com/microsoft/TRELLIS/blob/f17fdf12d8f17a6a09225f01756d141285dc848f/dataset_toolkits/render.py#L52-L53
        cmd = f"{blender_cmd} {blender_file_to_be_before_python_script} "
    cmd += (
        # f"{blender_cmd} --background --log-level 1 --python {blender_script} -- "
        f"--background --log-level 1 --python {blender_script} -- "
        f"--filename {json_filename} --out_dir {blender_run_dir} "
        f"--normalized_mesh_fname {normalized_mesh_fname} "
        f"--normalization_info_fname {normalization_info_fname} "
        f"--device {device} "
    )
    if blender_resume_info_f is not None:
        # Notice the leading space
        cmd += f" --resume_info_f {str(blender_resume_info_f)}"
    if blender_save_scene_format is not None:
        # Notice the leading space
        cmd += f" --save_scene_format {str(blender_save_scene_format)}"
    if debug:
        # Notice the leading space
        cmd += " --debug 1 "
    if not printout:
        # Notice the leading space
        cmd += f" > {blender_log_f}"

    print(cmd)
    os.system(cmd)

    return {
        "blender_log_fname": None if printout else blender_log_f,
        "normalized_mesh_fname": normalized_mesh_f,
        "normalization_info_fname": normalization_info_f,
        "scene_config.json": json_filename,
        "object_metadata.json": os.path.join(blender_run_dir, "metadata.json"),
    }


def render_rgbd_from_mesh_with_blender(
    *,
    blender_render_dir: str,
    mesh_filename: str,
    # camera
    H_c2w: torch.Tensor,  # (q, 4, 4) o3d camera
    intrinsic: torch.Tensor,  # (q, 3, 3) o3d camera
    width_px: int = 448,  # 14 x 32
    height_px: int = 448,  # 14 x 32
    # misc
    printout: bool = False,
    normalize_mesh: bool = True,
    normalized_mesh_fname: str = "blender_normalized_mesh.ply",
    normalization_info_fname: str = "config_after_blender_normalization.json",
    rerender_depth_from_noramlized_mesh: bool = False,
    blender_version: str = "4.2.0",
    rng: int | np.random.Generator | None = None,
    blender_download_from_s3: bool = False,
    blender_resume_info_f: str | None = None,
    blender_save_scene_format: str | None = None,
    extra_kwargs: T.Dict[str, T.Any] = dict(
        # normalize_mesh=True,  # bool
        mesh_H_c2w=None,  # T.Optional[torch.Tensor], (4, 4) or None
        mesh_scale=None,  # T.Optional[float] =None,
        blender_cycles_config=dict(
            rendering_samples_per_pixel=128,
            enable_indirect_illumination=False,
            filter_type="BLACKMAN_HARRIS",
            filter_width=0.01,
            enforce_texture_alpha_to_opaque=False,
            view_layer_pass_alpha_threshold=0.0,
            ablate_film_exposure_kwargs=dict(
                film_exposure_list=[1.0, 1.1],
                camera_dicts=[
                    dict(H_c2w=np.eye(4).tolist(), intrinsic=np.eye(3).tolist(), width_px=128, height_px=128)
                ],
                over_exposure_threshold=235 / 255,
                under_exposure_threshold=20 / 255,
            ),
        ),
        flag_align_attrs_with_rgb_mask=False,
        light_info_dict=dict(
            light_type="random",
            light_type_pool=["AREA", "POINT", "SPOT"],
            num_lights=None,
            min_num_lights=3,
            max_num_lights=10,
            min_light_energy=200.0,
            max_light_energy=2000.0,
            light_up_method="z",
            min_random_radius=7.5,
            max_random_radius=10.5,
            random_lookat_r=0.25,
            light_type_pool_kwargs={
                "AREA": {"size": {"min": 5.0, "max": 20.0}},
                "SPOT": {"shadow_soft_size": {"min": 0.0, "max": 5.0}},
            },
        ),
    ),
    blender_device: str = "CPU",
    debug: bool = False,
):
    """This function runs the Blender rendering script and saved the rendered RGB-D image as well as the normalized mesh.

    Args:
        blender_resume_info_f:
            str, if not None, we will append this to the command line argument and let Blender script re-use information
            in the file, e.g., normalization and lightings
        blender_save_scene_format:
            str or None. if not None, Blender will save the 3D asset to the specified format, e.g., GLB.
        extra_kwargs:
            a dict that contains information for blender_rendering/blender_utils.py
            It contains the following entries:

            - normalize_mesh:
                if True, we will normalize the mesh in blender rendering
            - mesh_H_c2w:
                (4, 4) or None, camera-to-world transformation for the mesh
            - mesh_scale:
                float, scale for rescaling the mesh
            - blender_cycles_config:
                dict, it contains information mainly for setting up Blender Cycles. It has the following entries:
                - rendering_samples_per_pixel:
                    int, samples per ray
                - enable_indirect_illumination:
                    bool, if True, we enable indirect illumination
                - filter_type:
                    str, antialiasing type to be applied for Blender
                - filter_width:
                    float, pixel width to be used with antialiasing
                - enforce_texture_alpha_to_opaque:
                    bool. if True, we will enforce all textures to be opaque in Blender
                - view_layer_pass_alpha_threshold:
                    float. see definition of "pass_alpha_threshold" in Blender
                - ablate_film_exposure_kwargs:
                    dict, contains information needed for ablating exposure values
                    See the arguments definition for find_suitable_film_exposure() in blender_rendering/blender_utils.py
            - flag_align_attrs_with_rgb_mask:
                WARNING!!!
                if True, we manually align all attributes mask with RGB.
                This is for experiment purpose only. We should not call it in most cases.
            - light_info_dict:
                dict, contains information to generate reasonable lighting setups for Blender rendering.
                It has the following entries:
                - light_type:
                    str, choose from ["diffuse", "random"]
                - light_type_pool:
                    list of str. It specifies the set of lighting types that we can use if we use "random" lighting
                - num_lights:
                    int or None, the number of lights to use with "random" lighting.
                    If it is None, we will randomly sample the number of lights
                - min_num_lights / max_num_lights:
                    int, if num_lights is None, we will sample the number of lights from [min_num_lights, max_num_lights]
                - min_light_energy / max_light_energy:
                    float, this specifies the TOTAL energy we will distrbute across all lights.
                    We will sample the energey from [min_light_energy, max_light_energy]
                - light_up_method:
                    str, the direction for the "UP"
                - min_random_radius / max_random_radius:
                    float, the lights will be placed at a sampled raidus from [min_random_radius, max_random_radius]
                - random_lookat_r:
                    float, we will randomly look at a position on a sphere with radius sampled from [0, random_lookat_r]
                - light_type_pool_kwargs:
                    dict, it contains specific attributes to be used in Blender.
                    Please check Blender's document for the attributes we can set for each light.
                    It is used in load_light() function in blender_rendering/blender_utils.py

        blender_device:
            'CPU', 'GPU'
    """

    rng_np: np.random.Generator = utils.get_np_rng(rng)

    blender_script_filename: str = "blender_utils.py"

    run_dir = os.path.abspath(blender_render_dir)
    assert os.path.exists(run_dir), f"{run_dir=}"
    # os.makedirs(run_dir, exist_ok=True)

    # mesh-related
    # normalize_mesh: bool = extra_kwargs["normalize_mesh"]
    mesh_H_c2w: T.Optional[torch.Tensor] = extra_kwargs["mesh_H_c2w"]  # (4, 4) or None
    mesh_scale: T.Optional[float] = extra_kwargs["mesh_scale"]
    # lighting
    light_info_dict = extra_kwargs["light_info_dict"]
    light_type: str = light_info_dict["light_type"]
    if light_type not in ["use_light_in_3d_file"]:
        # NOTE: we only need to specify lights ourselves if we decide to not use lights in the original 3D file
        light_type_pool: T.List[str] = light_info_dict.get("light_type_pool", None)
        light_type_pool_kwargs: T.List[str] = light_info_dict.get("light_type_pool_kwargs", None)
        num_lights: int = light_info_dict.get("num_lights", None)
        if num_lights is None:
            min_num_lights: int | None = light_info_dict.get("min_num_lights", None)
            max_num_lights: int | None = light_info_dict.get("max_num_lights", None)
            assert min_num_lights is not None
            assert max_num_lights is not None
            num_lights: int = rng_np.integers(low=min_num_lights, high=max_num_lights, size=None)

        if light_type in ["random"]:
            min_total_light_energy: float | None = light_info_dict["min_light_energy"]
            max_total_light_energy: float | None = light_info_dict["max_light_energy"]
            assert min_total_light_energy is not None
            assert max_total_light_energy is not None
            total_light_energy = (
                float(rng_np.uniform()) * (max_total_light_energy - min_total_light_energy) + min_total_light_energy
            )
            total_light_energy_dist = rng_np.dirichlet(np.ones(num_lights), size=1)[0]
        else:
            total_light_energy = None
            total_light_energy_dist = None

    blender_resume_info = None
    if blender_resume_info_f is not None:
        # NOTE: we currently only resume, exposure, mesh, and lighting
        assert pathlib.Path(blender_resume_info_f).suffix in [".json"], f"{blender_resume_info_f=}"
        with open(blender_resume_info_f, "r") as f:
            blender_resume_info = json.load(f)

    # compile json file (mesh, lighting, camera)
    scene_dict = dict()

    # Blender cycles
    if (blender_resume_info is None) or (blender_resume_info.get("render_dicts", None) is None):
        render_dicts: T.Dict[str, int | float | str | bool] = extra_kwargs["blender_cycles_config"]
    else:
        render_dicts = blender_resume_info["render_dicts"]
        if render_dicts.get("film_exposure", None) is not None:
            render_dicts["given_best_film_exposure"] = render_dicts["film_exposure"]
        else:
            print("\n\nHere legacy\n\n")
            # TODO: to remove
            # for legacy-compatible
            resume_ablate_film_exposure_f_list = list(
                pathlib.Path(blender_resume_info_f).parent.glob("search_exposure*.json")
            )
            assert len(resume_ablate_film_exposure_f_list) in [0, 1], f"{len(resume_ablate_film_exposure_f_list)=}"
            if len(resume_ablate_film_exposure_f_list) > 0:
                resume_ablate_film_exposure_f = resume_ablate_film_exposure_f_list[0]
            with open(resume_ablate_film_exposure_f, "r") as f:
                ablate_film_exposure_info = json.load(f)
            render_dicts["given_best_film_exposure"] = ablate_film_exposure_info["best_film_exposure"]
    scene_dict["render"] = render_dicts

    # mesh
    mesh_dicts = [
        dict(
            name="mesh",
            filename=mesh_filename,
            normalize_first=normalize_mesh,  # [-1, 1] aabb box
            H_c2w=np.eye(4)
            if mesh_H_c2w is None
            else mesh_H_c2w.detach().cpu().float().numpy(),  # no rotation after normalization
            scale=np.array([1.0, 1.0, 1.0])
            if mesh_scale is None
            else np.array([1.0, 1.0, 1.0]) * mesh_scale,  # no scaling after normalization
        )
    ]
    if (blender_resume_info is not None) and (blender_resume_info.get("mesh_dicts", None) is not None):
        previous_mesh_dicts = blender_resume_info["mesh_dicts"]
        assert len(previous_mesh_dicts) == 1, f"{len(previous_mesh_dicts)=}"
        previous_mesh_filename = previous_mesh_dicts[0]["filename"]
        assert pathlib.Path(previous_mesh_filename).name == pathlib.Path(mesh_filename).name, (
            f"{previous_mesh_filename=}, {mesh_filename=}"
        )
        mesh_dicts[0]["given_normalize_scale"] = previous_mesh_dicts[0]["normalize_scale"]
        mesh_dicts[0]["given_normalize_c2w_trans_offset_after_scale"] = previous_mesh_dicts[0][
            "normalize_c2w_trans_offset_after_scale"
        ]
    scene_dict["meshes"] = mesh_dicts

    # lighting
    if (blender_resume_info is None) or (blender_resume_info.get("light_dicts", None) is None):
        light_dicts = []

        if light_type not in ["use_light_in_3d_file"]:
            for il in range(num_lights):
                if light_type in ["random"]:
                    if False:
                        # NOTE: This is not reasonable as the SUN is placed at the origin and orients random directions.
                        # In most cases, this will place the SUN inside an object, making its effect minimal.
                        """
                        # that the light is toward -z
                        # but since we just wnat random light direction, we do not care
                        _H_c2w = rigid_motion.get_H_c2w_lookat(
                            pinhole_location_w=(0, 0, 0.0),
                            look_at_w=rigid_motion.get_random_direction(rng=rng).astype(np.float32),  # (3,)
                            up_w=(0, 1, 0.0),
                            invert_y=False,
                        )  # (4, 4)
                        """
                        raise NotImplementedError
                    else:
                        _H_c2w = rigid_motion.generate_random_camera_poses_lookat(
                            n=1,
                            pinhole_min_r=light_info_dict.get("min_random_radius", None),
                            pinhole_max_r=light_info_dict.get("max_random_radius", None),
                            lookat_r=light_info_dict.get("random_lookat_r", None),
                            up_method=light_info_dict.get("light_up_method", None),
                        )  # (1, 4, 4)
                        assert (_H_c2w.ndim == 3) and (_H_c2w.shape == (1, 4, 4)), f"{_H_c2w.shape=}"
                        _H_c2w = blender_open3d_utils.convert_open3d_camera_to_blender_H_c2w(_H_c2w[0])

                    tmp_energy = total_light_energy * total_light_energy_dist[il]
                    assert light_type_pool is not None
                    tmp_light_type = str(rng_np.choice(light_type_pool, 1, replace=False)[0])

                    tmp_extra_setups = {}
                    if (light_type_pool_kwargs is not None) and (tmp_light_type in light_type_pool_kwargs):
                        tmp_light_type_kwargs = light_type_pool_kwargs[tmp_light_type]
                        for tmp_k, tmp_k_dict in tmp_light_type_kwargs.items():
                            tmp_min = tmp_k_dict["min"]
                            tmp_max = tmp_k_dict["max"]
                            tmp_v = float(rng_np.uniform()) * (tmp_max - tmp_min) + tmp_min
                            tmp_extra_setups[tmp_k] = tmp_v

                    mdict = dict(
                        name=f"light {il}",
                        light_type=tmp_light_type,
                        H_c2w=_H_c2w.tolist(),
                        energy=tmp_energy,
                        use_shadow=False,
                        specular_factor=1.0,
                        total_energy=total_light_energy,
                        extra_setups=tmp_extra_setups,
                    )
                elif light_type == "diffuse":
                    mdict = dict(
                        name=f"light {il}",
                        light_type=light_type,
                        color=[1.0, 1.0, 1.0, 1.0],
                        strength=1.0,
                    )
                else:
                    raise NotImplementedError
                light_dicts.append(mdict)
    else:
        light_dicts = blender_resume_info["light_dicts"]
    scene_dict["lighting"] = light_dicts

    # get camera dict
    cdict = blender_open3d_utils.convert_open3d_camera_to_blender(
        H_c2w=H_c2w.detach().cpu().numpy(),  # (q, 4, 4)
        intrinsic=intrinsic.expand(H_c2w.size(0), 3, 3).detach().cpu().numpy(),  # (q, 3, 3)
        width_px=width_px,
        height_px=height_px,
    )
    camera_dicts = []
    for iq in range(H_c2w.shape[0]):
        _cdict = dict(
            H_c2w=cdict["H_c2w"][iq],  # (4, 4)
            intrinsic=cdict["intrinsic"][iq],  # (3, 3)
            width_px=width_px,
            height_px=height_px,
        )
        camera_dicts.append(_cdict)
    scene_dict["cameras"] = camera_dicts

    # save config to tmp file (again) just to keep config
    json_filename = os.path.join(run_dir, "config.json")
    with open(json_filename, "w") as f:
        json.dump(scene_dict, f, indent=2, cls=json_utils.NumpyJsonEncoder)

    # render with blender
    out_dir = os.path.join(run_dir, "out")

    blender_ret_dict = run_blender(
        blender_run_dir=out_dir,
        mesh_filename=mesh_filename,
        json_filename=json_filename,
        normalized_mesh_fname=normalized_mesh_fname,
        normalization_info_fname=normalization_info_fname,
        blender_version=blender_version,
        blender_script_filename=blender_script_filename,
        blender_download_from_s3=blender_download_from_s3,
        blender_resume_info_f=None,  # we have already updated the render config
        blender_save_scene_format=blender_save_scene_format,
        device=blender_device,
        debug=debug,
        printout=printout,
    )
    saved_normalized_mesh_f = blender_ret_dict["normalized_mesh_fname"]

    if rerender_depth_from_noramlized_mesh:
        # Since raw 3D asset file can have transparent materials, rendered depth can go through the transparnet materials.
        # It is also complicated to fix the depth renderings in Blender with those transparent materials.
        # Thus, we turn to re-render depths from the saved normalized mesh in a 2nd run.

        scene_dict_rerender = copy.deepcopy(scene_dict)
        assert len(scene_dict_rerender["meshes"]) == 1, f"{len(scene_dict_rerender['meshes'])=}"
        scene_dict_rerender["meshes"][0]["filename"] = blender_ret_dict["normalized_mesh_fname"]
        scene_dict_rerender["meshes"][0]["normalize_first"] = False  # NOTE: important, we should not normalize again
        json_filename_rerender = os.path.join(run_dir, "config_rerender.json")
        with open(json_filename_rerender, "w") as f:
            json.dump(scene_dict_rerender, f, indent=2, cls=json_utils.NumpyJsonEncoder)

        tmp_rerender_dir = os.path.join(run_dir, "re_render")

        blender_ret_dict_rerender = run_blender(
            blender_run_dir=tmp_rerender_dir,
            mesh_filename=mesh_filename,
            json_filename=json_filename_rerender,
            normalized_mesh_fname=normalized_mesh_fname,
            normalization_info_fname=normalization_info_fname,
            blender_version=blender_version,
            blender_script_filename=blender_script_filename,
            blender_download_from_s3=blender_download_from_s3,
            blender_resume_info_f=None,  # we have already updated the render config
            device=blender_device,
            printout=printout,
            debug=debug,
        )

        for tmp_k in blender_ret_dict_rerender:
            blender_ret_dict[f"rerender_{tmp_k}"] = blender_ret_dict_rerender[tmp_k]

        # move depth renderings to original rendering directory
        out_dir = pathlib.Path(out_dir)
        tmp_raw_depth_dir = pathlib.Path(run_dir) / "raw_depth"
        tmp_raw_depth_dir.mkdir(parents=True, exist_ok=True)
        tmp_raw_depth_f_list = sorted(list(out_dir.glob("*_depth.exr")))
        assert len(tmp_raw_depth_f_list) == H_c2w.shape[0], f"{len(tmp_raw_depth_f_list)=}, {H_c2w.shape=}"
        for tmp_raw_detph_f in tmp_raw_depth_f_list:
            shutil.move(tmp_raw_detph_f, tmp_raw_depth_dir / f"{tmp_raw_detph_f.name}")

        # move re-rendered depth
        tmp_rerender_dir = pathlib.Path(tmp_rerender_dir)
        tmp_rerender_depth_f_list = sorted(list(tmp_rerender_dir.glob("*_depth.exr")))
        assert len(tmp_rerender_depth_f_list) == H_c2w.shape[0], f"{len(tmp_rerender_depth_f_list)=}, {H_c2w.shape=}"
        for tmp_rerender_detph_f in tmp_rerender_depth_f_list:
            shutil.move(tmp_rerender_detph_f, out_dir / tmp_rerender_detph_f.name)
        tmp_raw_depth_f_list_2 = sorted(list(out_dir.glob("*_depth.exr")))
        assert len(tmp_raw_depth_f_list_2) == H_c2w.shape[0], f"{len(tmp_raw_depth_f_list_2)=}, {H_c2w.shape=}"

        out_dir = str(out_dir)

    ret_dict = dict(
        out_dir=out_dir,
        saved_normalized_mesh_f=saved_normalized_mesh_f,
        blender_ret_dict=blender_ret_dict,
    )

    return ret_dict


def render_rgbd_from_mesh_with_trellis_blender(
    *,
    blender_render_dir: str,
    mesh_filename: str,
    # camera
    H_c2w: torch.Tensor,  # (q, 4, 4) o3d camera
    intrinsic: torch.Tensor,  # (q, 3, 3) o3d camera
    width_px: int = 448,  # 14 x 32
    height_px: int = 448,  # 14 x 32
    # misc
    normalize_mesh: bool = True,
    printout: bool = False,
    normalized_mesh_fname: str = "blender_normalized_mesh.ply",
    normalization_info_fname: str = "config_after_blender_normalization.json",
    rerender_depth_from_noramlized_mesh: bool = False,
    blender_version: str = "4.2.0",
    rng: int | np.random.Generator | None = None,
    blender_download_from_s3: bool = False,
    blender_resume_info_f: str | None = None,
    blender_save_scene_format: str | None = None,
    extra_kwargs: T.Dict[str, T.Any] = dict(
        flag_align_attrs_with_rgb_mask=False,
        kwargs_for_script=dict(
            resolution=512,
            engine="CYCLES",
            geo_mode=False,
            save_rgb_exr=True,
            save_depth=True,
            save_normal=True,
            save_albedo=True,
            save_mist=True,
            save_obj_id=True,
            split_normal=False,
            save_mesh=True,
            filter_width=1.0,
            view_layer_pass_alpha_threshold=0.5,
        ),
    ),
    debug: bool = False,
):
    """This function runs the Blender rendering script and saved the rendered RGB-D image as well as the normalized mesh.

    Args:
        blender_resume_info_f:
            str, if not None, we will append this to the command line argument and let Blender script re-use information
            in the file, e.g., normalization and lightings
        blender_save_scene_format:
            str or None. if not None, Blender will save the 3D asset to the specified format, e.g., GLB.
        extra_kwargs:
            a dict that contains information for blender_rendering/blender_utils_trellis.py
            It contains the following entries:

            - flag_align_attrs_with_rgb_mask:
                WARNING!!!
                if True, we manually align all attributes mask with RGB.
                This is for experiment purpose only. We should not call it in most cases.
            - kwargs_for_script:
                dict, contains arguments used in blender_rendering/blender_utils_trellis.py
                It contains the following keys:

                - resolution:
                    int, the resolution to be used for Blender rendering
                - engine:
                    str, the rendering engine to be used with Blender
                - geo_mode:
                    bool, if True, remove all materials and only render grey geometry of the 3D assets
                - save_rgb_exr:
                    bool, if True, we save RGB (in EXR mode)
                - save_depth:
                    bool, if True, we save depth
                - save_normal:
                    bool, if True, we save normal
                - save_albedo:
                    bool, if True, we save diffuse color (this is not albedo actually)
                - save_mist:
                    bool, if True, we save the mist pass
                - save_obj_id:
                    bool, if True, we save OBJ_ID
                - split_normal:
                    bool, if True, we split the normal for a shared vertex.
                    Namely, if the vertex is shared across faces,
                    we make the normals for vertex on different faces different.
                    See split_mesh_normal() function in blender_rendering/blender_utils_trellis.py
                - save_mesh:
                    bool, if True, we save the mesh
                - filter_width:
                    float, pixel width to be used with antialiasing
                - view_layer_pass_alpha_threshold:
                    float. see definition of "pass_alpha_threshold" in Blender
    """

    blender_script_filename: str = "blender_utils_trellis.py"

    run_dir = os.path.abspath(blender_render_dir)
    assert os.path.exists(run_dir), f"{run_dir=}"
    # os.makedirs(run_dir, exist_ok=True)

    # compile json file (mesh, lighting, camera)
    scene_dict = dict()

    assert "kwargs_for_script" in extra_kwargs, f"{list(extra_kwargs.keys())=}"
    scene_dict["kwargs_for_script"] = extra_kwargs["kwargs_for_script"]

    if mesh_filename.endswith(".blend"):
        # https://github.com/microsoft/TRELLIS/blob/f17fdf12d8f17a6a09225f01756d141285dc848f/dataset_toolkits/render.py#L52-L53
        blender_file_to_be_before_python_script = mesh_filename
    else:
        blender_file_to_be_before_python_script = None

    scene_dict["mesh_filename"] = mesh_filename
    scene_dict["mesh_normalize_first"] = normalize_mesh

    # get camera dict
    cdict = blender_open3d_utils.convert_open3d_camera_to_blender(
        H_c2w=H_c2w.detach().cpu().numpy(),  # (q, 4, 4)
        intrinsic=intrinsic.expand(H_c2w.size(0), 3, 3).detach().cpu().numpy(),  # (q, 3, 3)
        width_px=width_px,
        height_px=height_px,
    )
    camera_dicts = []
    for iq in range(H_c2w.shape[0]):
        _cdict = dict(
            H_c2w=cdict["H_c2w"][iq],  # (4, 4)
            intrinsic=cdict["intrinsic"][iq],  # (3, 3)
            width_px=width_px,
            height_px=height_px,
        )
        camera_dicts.append(_cdict)
    scene_dict["cameras"] = camera_dicts

    # save config to tmp file (again) just to keep config
    json_filename = os.path.join(run_dir, "config.json")
    with open(json_filename, "w") as f:
        json.dump(scene_dict, f, indent=2, cls=json_utils.NumpyJsonEncoder)

    # render with blender
    out_dir = os.path.join(run_dir, "out")

    blender_ret_dict = run_blender(
        blender_run_dir=out_dir,
        mesh_filename=mesh_filename,
        json_filename=json_filename,
        normalized_mesh_fname=normalized_mesh_fname,
        normalization_info_fname=normalization_info_fname,
        blender_version=blender_version,
        blender_script_filename=blender_script_filename,
        blender_file_to_be_before_python_script=blender_file_to_be_before_python_script,
        blender_download_from_s3=blender_download_from_s3,
        blender_resume_info_f=blender_resume_info_f,
        blender_save_scene_format=blender_save_scene_format,
        debug=debug,
        printout=printout,
    )
    saved_normalized_mesh_f = blender_ret_dict["normalized_mesh_fname"]

    if rerender_depth_from_noramlized_mesh:
        # Since raw 3D asset file can have transparent materials, rendered depth can go through the transparnet materials.
        # It is also complicated to fix the depth renderings in Blender with those transparent materials.
        # Thus, we turn to re-render depths from the saved normalized mesh in a 2nd run.

        scene_dict_rerender = copy.deepcopy(scene_dict)
        scene_dict_rerender["mesh_filename"] = blender_ret_dict["normalized_mesh_fname"]
        scene_dict["mesh_normalize_first"] = False  # NOTE: important, we should not normalize again
        json_filename_rerender = os.path.join(run_dir, "config_rerender.json")
        with open(json_filename_rerender, "w") as f:
            json.dump(scene_dict_rerender, f, indent=2, cls=json_utils.NumpyJsonEncoder)

        tmp_rerender_dir = os.path.join(run_dir, "re_render")

        blender_ret_dict_rerender = run_blender(
            blender_run_dir=tmp_rerender_dir,
            mesh_filename=mesh_filename,
            json_filename=json_filename_rerender,
            normalized_mesh_fname=normalized_mesh_fname,
            normalization_info_fname=normalization_info_fname,
            blender_version=blender_version,
            blender_script_filename=blender_script_filename,
            blender_file_to_be_before_python_script=None,  # the saved mesh file is not a Blender file
            blender_download_from_s3=blender_download_from_s3,
            blender_resume_info_f=blender_resume_info_f,
            debug=debug,
            printout=printout,
        )

        for tmp_k in blender_ret_dict_rerender:
            blender_ret_dict[f"rerender_{tmp_k}"] = blender_ret_dict_rerender[tmp_k]

        # move depth renderings to original rendering directory
        out_dir = pathlib.Path(out_dir)
        tmp_raw_depth_dir = pathlib.Path(run_dir) / "raw_depth"
        tmp_raw_depth_dir.mkdir(parents=True, exist_ok=True)
        tmp_raw_depth_f_list = sorted(list(out_dir.glob("*_depth.exr")))
        assert len(tmp_raw_depth_f_list) == H_c2w.shape[0], f"{len(tmp_raw_depth_f_list)=}, {H_c2w.shape=}"
        for tmp_raw_detph_f in tmp_raw_depth_f_list:
            shutil.move(tmp_raw_detph_f, tmp_raw_depth_dir / f"{tmp_raw_detph_f.name}")

        # move re-rendered depth
        tmp_rerender_dir = pathlib.Path(tmp_rerender_dir)
        tmp_rerender_depth_f_list = sorted(list(tmp_rerender_dir.glob("*_depth.exr")))
        assert len(tmp_rerender_depth_f_list) == H_c2w.shape[0], f"{len(tmp_rerender_depth_f_list)=}, {H_c2w.shape=}"
        for tmp_rerender_detph_f in tmp_rerender_depth_f_list:
            shutil.move(tmp_rerender_detph_f, out_dir / tmp_rerender_detph_f.name)
        tmp_raw_depth_f_list_2 = sorted(list(out_dir.glob("*_depth.exr")))
        assert len(tmp_raw_depth_f_list_2) == H_c2w.shape[0], f"{len(tmp_raw_depth_f_list_2)=}, {H_c2w.shape=}"

        out_dir = str(out_dir)

    ret_dict = dict(
        out_dir=out_dir,
        saved_normalized_mesh_f=saved_normalized_mesh_f,
        blender_ret_dict=blender_ret_dict,
    )

    return ret_dict


def render_rgbd_from_mesh_with_blender_and_pcd_with_open3d_given_cameras(
    *,
    mesh_filename: str,
    num_points: int,
    # camera
    H_c2w: torch.Tensor,  # (q, 4, 4) o3d camera
    intrinsic: torch.Tensor,  # (q, 3, 3) o3d camera
    width_px: int = 448,  # 14 x 32
    height_px: int = 448,  # 14 x 32
    # misc
    background_color: float = 1.0,
    blender_render_dir: str = None,  # if None, use tempdir for rendering
    flag_save_space: bool = True,  # whether to keep hdr_map, obj_id_map, etc
    printout: bool = False,
    normalized_mesh_fname: str = "blender_normalized_mesh.ply",
    normalization_info_fname: str = "config_after_blender_normalization.json",
    rerender_depth_from_noramlized_mesh: bool = False,
    blender_version: str = "4.2.0",
    blender_script_filename: str = "blender_utils.py",
    blender_extra_kwargs: T.Dict[str, T.Any] = {},
    rng: int | np.random.Generator | None = None,
    debug: bool = False,
    **kwargs,
):
    """
    Given a mesh and a bunch of cameras, we
    1. normalize the mesh to fit [-1, 1] bounding box (if normalize_mesh is True)
    2. add lighting in the scene
    3. render the scene in blender with given camera (which already assumes [-1, 1] scene)
    4. open the mesh in trimesh/open3d, sample point cloud from the mesh uniformly on surface
    5. color each point in o3d_pcd with the closest point in the backprojected points from rendered images

    Returns:
        rgbd:
            structure.RGBDImage (1, q, h, w)
        pcd:
            structure.PointCloud (1, n)

    """

    del kwargs

    if blender_script_filename == "blender_utils.py":
        blender_render_func = render_rgbd_from_mesh_with_blender
    elif blender_script_filename == "blender_utils_trellis.py":
        blender_render_func = render_rgbd_from_mesh_with_trellis_blender
    else:
        raise ValueError(f"{blender_script_filename=}")

    # render
    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        blender_render_ret_dict = blender_render_func(
            blender_render_dir=tmp_dir if blender_render_dir is None else blender_render_dir,
            mesh_filename=mesh_filename,
            # camera
            H_c2w=H_c2w,  # (q, 4, 4) o3d camera
            intrinsic=intrinsic,  # (q, 3, 3) o3d camera
            width_px=width_px,  # 14 x 32
            height_px=height_px,  # 14 x 32
            # misc
            printout=printout,
            normalized_mesh_fname=normalized_mesh_fname,
            normalization_info_fname=normalization_info_fname,
            rerender_depth_from_noramlized_mesh=rerender_depth_from_noramlized_mesh,
            blender_version=blender_version,
            extra_kwargs=blender_extra_kwargs,
            rng=rng,
            debug=debug,
        )

        tmp_out_dir = blender_render_ret_dict["out_dir"]
        saved_normalized_mesh_f = blender_render_ret_dict["saved_normalized_mesh_f"]
        blender_ret_dict = blender_render_ret_dict["blender_ret_dict"]

        # read the rendered rgbd image
        rgbd = blender_plib_utils.read_blender_results_to_rgbd(
            result_dir=tmp_out_dir,
            from_idx=0,
            to_idx=None,
            use_srgb=True,
            flag_save_space=flag_save_space,
            flag_align_attrs_with_rgb_mask=blender_extra_kwargs["flag_align_attrs_with_rgb_mask"],
            debug=debug,
        )  # (b=1, q, h, w)
        assert rgbd.rgb.size(1) == H_c2w.size(0), f"{rgbd.rgb.shape=}, {H_c2w.shape=}"

        rgbd = rgbd.remove_invalid(
            min_depth=0,
            max_depth=1e4,
            background_color=background_color,
        )

        # sample point cloud from the saved ply file
        o3d_mesh = o3d.io.read_triangle_mesh(saved_normalized_mesh_f)
        # sample points uniformly on surface
        o3d_pcd = o3d_mesh.sample_points_uniformly(
            number_of_points=num_points,
        )
        xyz_w = torch.tensor(np.asarray(o3d_pcd.points), dtype=torch.float)  # (n, 3)
        normal_w = torch.tensor(np.asarray(o3d_pcd.normals), dtype=torch.float)  # (n, 3)
        del o3d_pcd

        # backproject pixel from rgbd images
        st_pcd = rgbd.get_pcd(
            subsample=1,
            remove_background=True,
            keep_img_idxs=False,
            compute_ray_feature=False,
        )

        # color the o3d_pcd with nearest backprojected point
        tree = cKDTree(st_pcd.xyz_w[0].detach().cpu().float().numpy())
        _, knn_midx = tree.query(xyz_w.detach().cpu().float().numpy(), k=1, workers=-1)
        # knn_midx: (n, 1)
        knn_midx = torch.from_numpy(knn_midx).reshape(xyz_w.size(0))  # (n,)
        assert knn_midx.shape[0] == xyz_w.size(0)
        rgb = st_pcd.rgb[0][knn_midx]  # (n, 3)

        # remove problematic points that are not finite
        n, _3xyz = xyz_w.shape
        assert normal_w.shape == xyz_w.shape
        vmask = xyz_w.isfinite().all(dim=-1)  # (n,)
        vmask = torch.logical_and(
            vmask,  # (n,)
            normal_w.isfinite().all(dim=-1),  # (n,)
        )  # (n,)
        assert rgb.shape == xyz_w.shape
        vmask = torch.logical_and(
            vmask,  # (n,)
            rgb.isfinite().all(dim=-1),  # (n,)
        )  # (n,)

        # remove invalid points
        xyz_w = xyz_w[vmask]  # (n, 3)
        rgb = rgb[vmask]  # (n, 3)
        normal_w = normal_w[vmask]  # (n, 3)

        # shuffle
        ridxs = torch.randperm(xyz_w.size(0), device=xyz_w.device)
        xyz_w = xyz_w[ridxs]
        normal_w = normal_w[ridxs]
        rgb = rgb[ridxs]

        # create point cloud with only valid points
        pcd = structures.PointCloud(
            xyz_w=xyz_w.reshape(1, -1, 3),
            rgb=rgb.reshape(1, -1, 3),
            normal_w=normal_w.reshape(1, -1, 3),
        )  # (b=1, n, 3)

    return dict(
        rgbd=rgbd,  # (1, q, h, w)
        pcd=pcd,  # (1, n)
        blender_ret_dict=blender_ret_dict,
    )


def save_pcd(
    *,
    out_dir: str,
    pcd: structures.PointCloud,
    pcd_save_version: int,
    pcd_save_chunk_size: int,
    save_np_dtype: np.dtype,
    other_attr_names: T.List[str] = [],
    max_n_pts: int = -1,
):
    # remove points that are outside [-1, 1] aabb, and are not finite
    xyz_w = pcd.extract_valid_attr(
        arr=pcd.xyz_w,
        bidx=0,
    )  # (n, 3)
    n, _3xyz = xyz_w.shape

    bbox_eps = 0.01
    vmask = torch.logical_and(
        (xyz_w >= -1 - bbox_eps).all(dim=-1),  # (n,)
        (xyz_w <= 1 + bbox_eps).all(dim=-1),  # (n,)
    )  # (n,)
    # print(f'after xyz_w, vmask: {vmask.shape} ({vmask.float().mean()})')
    normal_w = pcd.extract_valid_attr(
        arr=pcd.normal_w,
        bidx=0,
    )  # (n, 3)
    assert xyz_w.size(0) == normal_w.size(0)
    vmask = torch.logical_and(
        vmask,  # (n,)
        normal_w.isfinite().all(dim=-1),  # (n,)
    )  # (n,)
    # print(f'after normal, vmask: {vmask.shape} ({vmask.float().mean()})')
    rgb = pcd.extract_valid_attr(
        arr=pcd.rgb,
        bidx=0,
    )  # (n, 3)
    assert xyz_w.size(0) == rgb.size(0)
    vmask = torch.logical_and(
        vmask,  # (n,)
        rgb.isfinite().all(dim=-1),  # (n,)
    )  # (n,)

    xyz_w_masked = xyz_w[vmask].reshape(1, -1, 3)
    n_pts = xyz_w_masked.shape[1]

    masked_other_attrs = {
        _: pcd.extract_valid_attr(arr=getattr(pcd, _, None), bidx=0)[vmask].reshape((1, n_pts, -1))
        for _ in other_attr_names
    }

    # create point cloud with only valid points
    pcd = structures.PointCloud(
        xyz_w=xyz_w_masked,
        rgb=rgb[vmask].reshape(1, -1, 3),
        normal_w=normal_w[vmask].reshape(1, -1, 3),
        **masked_other_attrs,
    )  # (b=1, n, 3)

    # save point cloud
    xyz_w = pcd.extract_valid_attr(
        arr=pcd.xyz_w,
        bidx=0,
    )  # (n, 3)
    normal_w = pcd.extract_valid_attr(
        arr=pcd.normal_w,
        bidx=0,
    )  # (n, 3)
    rgb = pcd.extract_valid_attr(
        arr=pcd.rgb,
        bidx=0,
    )  # (n, 3)
    assert xyz_w.size(0) == normal_w.size(0)
    assert xyz_w.size(0) == rgb.size(0)

    # shuffle
    ridxs = torch.randperm(xyz_w.size(0), device=xyz_w.device)
    if max_n_pts > 0:
        ridxs = ridxs[:max_n_pts]
    xyz_w = xyz_w[ridxs]
    normal_w = normal_w[ridxs]
    rgb = rgb[ridxs]

    other_attrs = {}
    for tmp_k in other_attr_names:
        tmp_v = getattr(pcd, tmp_k, None)
        assert tmp_v is not None, f"{tmp_k=}"
        tmp_v = pcd.extract_valid_attr(arr=tmp_v, bidx=0)
        other_attrs[tmp_k] = tmp_v[ridxs]

    index_dict = save_sampled_pcd(
        pcd_save_version=pcd_save_version,
        out_dir=out_dir,
        index_dict={},
        xyz_w=xyz_w,
        rgb=rgb,
        normal_w=normal_w,
        save_np_dtype=save_np_dtype,
        save_chunk_size=pcd_save_chunk_size,
        other_attrs=other_attrs,
    )
    return index_dict


def sample_rgbd_from_mesh_with_blender_and_pcd_with_open3d(
    *,
    mesh_filename: str,
    out_dir: str,  # main dir to save (where index.json and point cloud would be)
    out_dir_rgbd: str = None,  # where rgbd will be saved
    num_points: int = 3_000_000,
    width_px: int = 448,  # 14 x 32
    height_px: int = 448,  # 14 x 32
    # circular camera
    num_regular_images: int = 10,
    fov: float = 40.0,  # degree
    circular_radius: float = 3.5,  # meter
    # random camera
    num_random_images: int = 30,
    min_fov: float = 40.0,  # degree
    max_fov: float = 60.0,  # degree
    min_random_radius: float = 3,
    max_random_radius: float = 4,
    random_lookat_r: float = 0.25,
    mesh_rel_dir: str = None,  # whether we want to simplify mesh_filename when saving it to index.json
    background_color: float = 1,
    overwrite: bool = False,
    save_attr_names: T.List[str] = None,
    pcd_save_version: int = 2,
    save_np_dtype: np.dtype = np.float32,
    normalized_mesh_fname: str = "blender_normalized_mesh.ply",
    normalization_info_fname: str = "config_after_blender_normalization.json",
    seed: int = 0,
    pcd_save_chunk_size: int = 500_000,
    regular_camera_sampling_type: str = "sphere",
    regular_camera_sampling_type_func: str = "ours",
    flag_save_space: bool = True,  # whether to keep hdr_map, obj_id_map
    printout: bool = False,
    rerender_depth_from_noramlized_mesh: bool = False,
    debug: bool = False,
    blender_version: str = "4.2.0",
    blender_script_filename: str = "blender_utils.py",
    blender_extra_kwargs: T.Dict[str, T.Any] = dict(
        normalize_mesh=True,  # bool
        mesh_H_c2w=None,  # T.Optional[torch.Tensor], (4, 4) or None
        mesh_scale=None,  # T.Optional[float] =None,
        # lighting
        light_type="SUN",
        num_lights=8,
        min_light_energy=0.0,  # remember to adjust when num_lights change
        max_light_energy=3.0,  # remember to adjust when num_lights change
        blender_cycles_config={},  # T.Dict[str, int | float | str | bool]
    ),
    extra_kwargs_regular_cameras: T.Dict[str, T.Any] = {},
    extra_kwargs_random_cameras: T.Dict[str, T.Any] = {},
    **kwargs,
):
    """
    Render rgbd images with blender then resample point cloud from the surface of the mesh.

    Strategy:
    1. normalize the mesh to fit [-1, 1] bounding box
    2. randomly add lighting in the scene
    3. sample randomly the camera in a sphere to avoid occusion
    4. sample camera randomly in a shell
    5. sample camera in a spherical shell

    Args:
        out_dir_rgbd:
            if not None, rgbd will be saved in the folder instead of out_dir

    For legacy, here we list the blender_extra_kwargs default values before:
    blender_extra_kwargs = dict(
        normalize_mesh=True,  # bool
        mesh_H_c2w=None,  # T.Optional[torch.Tensor], (4, 4) or None
        mesh_scale=None,  # T.Optional[float] =None,
        # lighting
        light_type="SUN",
        num_lights=8,
        min_light_energy=0.0,  # remember to adjust when num_lights change
        max_light_energy=3.0,  # remember to adjust when num_lights change
        blender_cycles_config={},  # T.Dict[str, int | float | str | bool]
    )
    """

    del kwargs

    _set_seed(seed)

    if not overwrite:
        assert (not os.path.exists(out_dir)) or (not os.listdir(out_dir))

    if out_dir_rgbd is None:
        out_dir_rgbd = out_dir

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_rgbd, exist_ok=True)

    # create camera
    cam_dict = get_regular_and_random_cameras(
        num_regular_images=num_regular_images,
        r=circular_radius,
        fov=fov,
        width_px=width_px,
        height_px=height_px,
        num_random_images=num_random_images,
        min_random_radius=min_random_radius,
        max_random_radius=max_random_radius,
        random_lookat_r=random_lookat_r,
        max_fov=max_fov,
        min_fov=min_fov,
        regular_camera_sampling_type=regular_camera_sampling_type,
        regular_camera_sampling_type_func=regular_camera_sampling_type_func,
        extra_kwargs_regular_cameras=extra_kwargs_regular_cameras,
        extra_kwargs_random_cameras=extra_kwargs_random_cameras,
    )
    camera_regular: structures.Camera = cam_dict["camera_regular"]
    camera_random: structures.Camera = cam_dict["camera_random"]

    q_regular = camera_regular.H_c2w.size(1)
    q_random = camera_random.H_c2w.size(1)
    camera = structures.Camera.cat([camera_regular, camera_random], dim=1)  # (b=1, q)

    # render in a temp folder
    with (
        contextlib.nullcontext(DEBUG_ROOT)
        if debug
        else tempfile.TemporaryDirectory(dir=REPO_ROOT) as blender_render_dir
    ):
        blender_render_dir = os.path.join(blender_render_dir, "blender")
        if os.path.exists(blender_render_dir):
            shutil.rmtree(blender_render_dir)
        os.makedirs(blender_render_dir, exist_ok=True)

        rdict = render_rgbd_from_mesh_with_blender_and_pcd_with_open3d_given_cameras(
            mesh_filename=mesh_filename,
            num_points=num_points,
            H_c2w=camera.H_c2w.squeeze(0),  # (q, 4, 4)
            intrinsic=camera.intrinsic.squeeze(0),  # (q, 3, 3)
            width_px=camera.width_px,
            height_px=camera.height_px,
            background_color=background_color,
            blender_render_dir=blender_render_dir,
            flag_save_space=flag_save_space,
            printout=printout,
            normalized_mesh_fname=normalized_mesh_fname,
            normalization_info_fname=normalization_info_fname,
            rerender_depth_from_noramlized_mesh=rerender_depth_from_noramlized_mesh,
            rng=seed,
            debug=debug,
            blender_version=blender_version,
            blender_script_filename=blender_script_filename,
            blender_extra_kwargs=blender_extra_kwargs,
        )
        # the normalized mesh will be at {blender_render_dir}/out/blender_normalized_mesh.ply
        # {blender_render_dir}/out/config_after_blender_normalization.json
        # {blender_render_dir}/out/config.json
        # {blender_render_dir}/out/metadata.json
        # for src_filename, dest_filename in [
        #     [
        #         os.path.join(blender_render_dir, "out", "blender_normalized_mesh.ply"),  # src
        #         os.path.join(out_dir, normalized_mesh_fname),  # dest
        #     ],
        #     [
        #         os.path.join(blender_render_dir, "out", "config_after_blender_normalization.json"),  # src
        #         os.path.join(out_dir, normalization_info_fname),  # dest
        #     ],
        #     [
        #         os.path.join(blender_render_dir, "out", "config.json"),  # src
        #         os.path.join(out_dir, "scene_config.json"),  # dest
        #     ],
        #     [
        #         os.path.join(blender_render_dir, "out", "metadata.json"),  # src
        #         os.path.join(out_dir, "object_metadata.json"),  # dest
        #     ],
        # ]:
        for src_f in rdict["blender_ret_dict"].values():
            if (src_f is not None) and os.path.exists(src_f):
                dest_f = os.path.join(out_dir, pathlib.Path(src_f).name)
                if os.path.exists(dest_f):
                    os.remove(dest_f)
                shutil.copy(src_f, dest_f)

    # gather rendered results into rgbd images
    rgbd: structures.RGBDImage = rdict["rgbd"]  # (b=1, q, h, w)
    pcd: structures.PointCloud = rdict["pcd"]  # (b=1, n)

    rgbd_regular = rgbd.index_select(
        dim=1,
        index=torch.arange(q_regular),
    )
    rgbd_random = rgbd.index_select(
        dim=1,
        index=torch.arange(q_random) + q_regular,
    )
    rgbd_dict = dict()
    rgbd_dict[regular_camera_sampling_type] = rgbd_regular
    rgbd_dict["random"] = rgbd_random

    # save results
    index_dict = dict()

    # get mesh relative dir
    if mesh_rel_dir is not None:
        fn = os.path.relpath(mesh_filename, start=mesh_rel_dir)
    else:
        fn = mesh_filename
    index_dict["mesh_filename"] = fn

    pcd_index_dict = save_pcd(
        out_dir=out_dir,
        pcd=pcd,
        pcd_save_version=pcd_save_version,
        pcd_save_chunk_size=pcd_save_chunk_size,
        save_np_dtype=save_np_dtype,
    )
    pcd_index_dict_intersection = set(pcd_index_dict).intersection(set(index_dict.keys()))
    assert len(pcd_index_dict_intersection) == 0, f"{pcd_index_dict_intersection=}"
    index_dict.update(pcd_index_dict)

    # save rgbd image
    for sub_name, rgbd in rgbd_dict.items():
        name = f"rgbd_{sub_name}"  # rgbd_random, rgbd_sphere
        sub_dir = os.path.join(out_dir_rgbd, name)
        rgbd: structures.RGBDImage
        rgbd = rgbd.remove_invalid(
            min_depth=0,
            max_depth=1e4,
            background_color=background_color,
        )
        _, sub_index_filename = rgbd.save_as(
            out_dir=sub_dir,
            overwrite=overwrite,
            mode="png",
            background_color=background_color,
            save_attr_names=save_attr_names,
            flag_save_space=flag_save_space,
        )
        index_dict[name] = dict(
            index_filename=os.path.relpath(sub_index_filename, start=out_dir_rgbd),
            q=rgbd.rgb.size(1),
            h=rgbd.rgb.size(2),
            w=rgbd.rgb.size(3),
        )

    # save json
    json_filename = os.path.join(out_dir, "index.json")
    with open(json_filename, "w") as f:
        json.dump(index_dict, f, indent=2)

    if out_dir_rgbd != out_dir:
        _json_filename = os.path.join(out_dir_rgbd, "index.json")
        with open(_json_filename, "w") as f:
            json.dump(index_dict, f, indent=2)

    ret_dict = {"index_dict": index_dict, "json_filename": json_filename}
    return ret_dict


def render_rgbd_dynamic_given_cameras(
    *,
    mesh_filename: str,
    normalize_mesh: bool,
    num_points: int,
    # camera
    H_c2w: torch.Tensor,  # (q, 4, 4) o3d camera
    intrinsic: torch.Tensor,  # (q, 3, 3) o3d camera
    width_px: int = 448,  # 14 x 32
    height_px: int = 448,  # 14 x 32
    # lighting
    light_type: str = "SUN",
    num_lights: int = 8,
    min_light_energy: float = 0.0,  # remember to adjust when num_lights change
    max_light_energy: float = 3.0,  # remember to adjust when num_lights change
    # misc
    background_color: float = 1.0,
    blender_render_dir: str = None,  # if None, use tempdir for rendering
    printout: bool = False,
    dynamic=False,
    num_frames=-1,
    out_dir_mesh=None,
):
    """
    Given a mesh and a bunch of cameras, we
    1. normalize the mesh to fit [-1, 1] bounding box (if normalize_mesh is True)
    2. add lighting in the scene
    3. render the scene in blender with given camera (which already assumes [-1, 1] scene)
    4. open the mesh in trimesh/open3d, sample point cloud from the mesh uniformly on surface
    5. color each point in o3d_pcd with the closest point in the backprojected points from rendered images

    Returns:
        rgbd:
            structure.RGBDImage (1, q, h, w)
        pcd:
            structure.PointCloud (1, n)

    """

    # compile json file (mesh, lighting, camera)
    scene_dict = dict()

    # mesh
    mdict = dict(
        name="mesh",
        filename=mesh_filename,
        normalize_first=normalize_mesh,  # [-1, 1] aabb box
        H_c2w=np.eye(4),  # no rotation after normalization
        scale=np.array([1.0, 1.0, 1.0]),  # no scaling after normalization
    )
    scene_dict["meshes"] = [mdict]

    # lighting
    light_dicts = []
    for il in range(num_lights):
        if light_type == "SUN":
            # note that the light is toward -z
            # but since we just wnat random light direction, we do not care
            _H_c2w = rigid_motion.get_H_c2w_lookat(
                pinhole_location_w=(0, 0, 0.0),
                look_at_w=rigid_motion.get_random_direction().astype(np.float32),  # (3,)
                up_w=(0, 1, 0.0),
                invert_y=False,
            )  # (4, 4)
            energy = float(np.random.rand()) * (max_light_energy - min_light_energy) + min_light_energy

            mdict = dict(
                name=f"light {il}",
                light_type=light_type,
                H_c2w=_H_c2w.tolist(),
                energy=energy,
                use_shadow=False,
                specular_factor=1.0,
            )
        elif light_type == "diffuse":
            mdict = dict(
                name=f"light {il}",
                light_type=light_type,
                color=[1.0, 1.0, 1.0, 1.0],
                strength=1.0,
            )
        else:
            raise NotImplementedError
        light_dicts.append(mdict)

    scene_dict["lighting"] = light_dicts

    # get camera dict
    cdict = blender_open3d_utils.convert_open3d_camera_to_blender(
        H_c2w=H_c2w.detach().cpu().numpy(),  # (q, 4, 4)
        intrinsic=intrinsic.expand(H_c2w.size(0), 3, 3).detach().cpu().numpy(),  # (q, 3, 3)
        width_px=width_px,
        height_px=height_px,
    )
    camera_dicts = []

    for iq in range(H_c2w.shape[0]):
        _cdict = dict(
            H_c2w=cdict["H_c2w"][iq],  # (4, 4)
            intrinsic=cdict["intrinsic"][iq],  # (3, 3)
            width_px=width_px,
            height_px=height_px,
        )
        camera_dicts.append(_cdict)
    scene_dict["cameras"] = camera_dicts

    # render
    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        if blender_render_dir is None:
            tmp_dir = os.path.abspath(tmp_dir)
        else:
            tmp_dir = blender_render_dir
            os.makedirs(tmp_dir, exist_ok=True)

        # save config to tmp file (again) just to keep config
        json_filename = os.path.join(tmp_dir, "config.json")
        with open(json_filename, "w") as f:
            json.dump(scene_dict, f, indent=2, cls=json_utils.NumpyJsonEncoder)

        # render with blender
        tmp_out_dir = os.path.join(tmp_dir, "out")
        blender_cmd = blender_rendering_utils.get_blender_exe()
        blender_script = blender_rendering_utils.get_blender_utils_path()
        blender_log_fname = f"blender_{pathlib.Path(mesh_filename).stem}.log"
        blender_log_f = os.path.join(tmp_dir, blender_log_fname)
        normalized_mesh_fname: str = "blender_normalized_mesh.ply"
        normalization_info_fname: str = "config_after_blender_normalization.json"

        cmd = (
            f"{blender_cmd} --background --log-level 1 --python {blender_script} -- "
            f"--filename {json_filename} --out_dir {tmp_out_dir} "
            f"--normalized_mesh_fname {normalized_mesh_fname} "
            f"--normalization_info_fname {normalization_info_fname} "
            f"--dynamic {int(dynamic)} "
            f"--num_frames {num_frames} "
        )
        if printout:
            cmd += " --debug 1 "
        else:
            cmd += f" > {blender_log_fname}"
        print(cmd)
        os.system(cmd)

        # read the rendered rgbd image
        rgbd = blender_plib_utils.read_blender_results_to_rgbd(
            result_dir=tmp_out_dir, from_idx=0, to_idx=None, use_srgb=True, dynamic=dynamic
        )  # (b=num_frames, q, h, w)

        assert rgbd.rgb.size(1) == H_c2w.size(0)

        rgbd = rgbd.remove_invalid(
            min_depth=0,
            max_depth=1e4,
            background_color=background_color,
        )

        num_frames = rgbd.shape[0]
        # backproject pixel from rgbd images
        st_pcd = rgbd.get_pcd(
            subsample=1,
            remove_background=True,
            keep_img_idxs=False,
            compute_ray_feature=False,
        )
        pcds = []
        for frame in range(num_frames):
            prefix = f"{frame:04d}_" if dynamic else ""
            saved_normalized_mesh_filename = os.path.join(tmp_out_dir, prefix + normalized_mesh_fname)
            saved_normalization_json_filename = os.path.join(tmp_out_dir, normalization_info_fname)

            # sample point cloud from the saved ply file
            o3d_mesh = o3d.io.read_triangle_mesh(saved_normalized_mesh_filename)
            # sample points uniformly on surface
            o3d_pcd = o3d_mesh.sample_points_uniformly(
                number_of_points=num_points,
            )
            xyz_w = torch.tensor(np.asarray(o3d_pcd.points), dtype=torch.float)  # (n, 3)
            normal_w = torch.tensor(np.asarray(o3d_pcd.normals), dtype=torch.float)  # (n, 3)
            del o3d_pcd

            # color the o3d_pcd with nearest backprojected point
            tree = cKDTree(st_pcd.xyz_w[frame].detach().cpu().float().numpy())
            _, knn_midx = tree.query(xyz_w.detach().cpu().float().numpy(), k=1, workers=-1)
            # knn_midx: (n, 1)
            knn_midx = torch.from_numpy(knn_midx).reshape(xyz_w.size(0))  # (n,)
            assert knn_midx.shape[0] == xyz_w.size(0)
            rgb = st_pcd.rgb[frame][knn_midx]  # (n, 3)

            # remove problematic points that are not finite
            n, _3xyz = xyz_w.shape
            assert normal_w.shape == xyz_w.shape
            vmask = xyz_w.isfinite().all(dim=-1)  # (n,)
            vmask = torch.logical_and(
                vmask,  # (n,)
                normal_w.isfinite().all(dim=-1),  # (n,)
            )  # (n,)
            assert rgb.shape == xyz_w.shape
            vmask = torch.logical_and(
                vmask,  # (n,)
                rgb.isfinite().all(dim=-1),  # (n,)
            )  # (n,)

            # remove invalid points
            xyz_w = xyz_w[vmask]  # (n, 3)
            rgb = rgb[vmask]  # (n, 3)
            normal_w = normal_w[vmask]  # (n, 3)

            # shuffle
            ridxs = torch.randperm(xyz_w.size(0), device=xyz_w.device)
            xyz_w = xyz_w[ridxs]
            normal_w = normal_w[ridxs]
            rgb = rgb[ridxs]

            # create point cloud with only valid points
            pcd = structures.PointCloud(
                xyz_w=xyz_w.reshape(1, -1, 3),
                rgb=rgb.reshape(1, -1, 3),
                normal_w=normal_w.reshape(1, -1, 3),
            )  # (b=1, n, 3)
            pcds.append(pcd)
            if out_dir_mesh is not None:
                os.makedirs(pathlib.Path(out_dir_mesh) / f"{frame:06d}", exist_ok=True)
                shutil.copyfile(
                    pathlib.Path(tmp_out_dir) / saved_normalized_mesh_filename,
                    pathlib.Path(out_dir_mesh) / os.path.join(f"{frame:06d}", normalized_mesh_fname),
                )
                shutil.copyfile(
                    pathlib.Path(tmp_out_dir) / saved_normalization_json_filename,
                    pathlib.Path(out_dir_mesh) / os.path.join(f"{frame:06d}", normalization_info_fname),
                )

    pcd = structures.PointCloud.cat(pcds, 0)

    return dict(
        rgbd=rgbd,  # (num_frames, q=num_views, h, w)
        pcd=pcd,  # (num_frames, n)
    )


def render_rgbd_dynamic_without_cameras(
    *,
    mesh_filename: str,
    normalize_mesh: bool,
    num_points: int,
    # camera
    width_px: int = 448,  # 14 x 32
    height_px: int = 448,  # 14 x 32
    # lighting
    light_type: str = "SUN",
    num_lights: int = 8,
    min_light_energy: float = 0.0,  # remember to adjust when num_lights change
    max_light_energy: float = 3.0,  # remember to adjust when num_lights change
    # misc
    background_color: float = 1.0,
    blender_render_dir: str = None,  # if None, use tempdir for rendering
    printout: bool = False,
    dynamic=False,
    num_frames=-1,
    out_dir_mesh=None,
    # circular camera
    num_regular_images: int = 10,
    fov: float = 40.0,  # degree
    circular_radius: float = 3.5,  # meter
    # random camera
    num_random_images: int = 30,
    min_fov: float = 40.0,  # degree
    max_fov: float = 60.0,  # degree
    min_random_radius: float = 3,
    max_random_radius: float = 4,
    random_lookat_r: float = 0.25,
    regular_camera_sampling_type: str = "sphere",
    out_dir="/mnt/dynamic_tokenization/blender_rendering/out_pcd",
    out_dir_rgbd="/mnt/dynamic_tokenization/blender_rendering/out_rgbd",
    save_chunk_size=300_000,
    mesh_rel_dir="objaverse_10/glbs/000-012",
    tmp_out_dir="/mnt/dynamic_tokenization/blender_rendering/out",
    overwrite=True,
    save_attr_names=["rgb", "depth", "hit_map", "normal_w"],
    flag_save_space=False,
    frame_start=0,
    animation_number=0,
):
    """
    Given a mesh and a bunch of cameras, we
    1. normalize the mesh to fit [-1, 1] bounding box (if normalize_mesh is True)
    2. add lighting in the scene
    3. render the scene in blender with given camera (which already assumes [-1, 1] scene)
    4. open the mesh in trimesh/open3d, sample point cloud from the mesh uniformly on surface
    5. color each point in o3d_pcd with the closest point in the backprojected points from rendered images

    Returns:
        rgbd:
            structure.RGBDImage (1, q, h, w)
        pcd:
            structure.PointCloud (1, n)

    """

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_rgbd, exist_ok=True)

    index_dict = dict()

    # compile json file (mesh, lighting, camera)
    scene_dict = dict()

    # mesh
    mdict = dict(
        name="mesh",
        filename=mesh_filename,
        normalize_first=normalize_mesh,  # [-1, 1] aabb box
        H_c2w=np.eye(4),  # no rotation after normalization
        scale=np.array([1.0, 1.0, 1.0]),  # no scaling after normalization
    )
    scene_dict["meshes"] = [mdict]

    # lighting
    light_dicts = []
    for il in range(num_lights):
        if light_type == "SUN":
            # note that the light is toward -z
            # but since we just wnat random light direction, we do not care
            _H_c2w = rigid_motion.get_H_c2w_lookat(
                pinhole_location_w=(0, 0, 0.0),
                look_at_w=rigid_motion.get_random_direction().astype(np.float32),  # (3,)
                up_w=(0, 1, 0.0),
                invert_y=False,
            )  # (4, 4)
            energy = float(np.random.rand()) * (max_light_energy - min_light_energy) + min_light_energy

            mdict = dict(
                name=f"light {il}",
                light_type=light_type,
                H_c2w=_H_c2w.tolist(),
                energy=energy,
                use_shadow=False,
                specular_factor=1.0,
            )
        elif light_type == "diffuse":
            mdict = dict(
                name=f"light {il}",
                light_type=light_type,
                color=[1.0, 1.0, 1.0, 1.0],
                strength=1.0,
            )
        else:
            raise NotImplementedError
        light_dicts.append(mdict)

    scene_dict["lighting"] = light_dicts

    # # # camera
    # camera_dicts1, cam_name_dict = get_camera_dicts_for_blender(
    #     num_regular_images=num_regular_images,
    #     r=circular_radius,
    #     fov=fov,
    #     width_px=width_px,
    #     height_px=height_px,
    #     num_random_images=num_random_images,
    #     min_random_radius=min_random_radius,
    #     max_random_radius=max_random_radius,
    #     random_lookat_r=random_lookat_r,
    #     max_fov=max_fov,
    #     min_fov=min_fov,
    #     regular_camera_sampling_type=regular_camera_sampling_type,
    # )

    camera_dicts = get_regular_and_random_cameras(
        num_regular_images=num_regular_images,
        r=circular_radius,
        fov=fov,
        width_px=width_px,
        height_px=height_px,
        num_random_images=num_random_images,
        min_random_radius=min_random_radius,
        max_random_radius=max_random_radius,
        random_lookat_r=random_lookat_r,
        max_fov=max_fov,
        min_fov=min_fov,
        regular_camera_sampling_type=regular_camera_sampling_type,
        regular_camera_sampling_type_func="trellis",
        extra_kwargs_regular_cameras={"offset": (0, 0), "up_method": "z"},
        extra_kwargs_random_cameras={"up_method": "z"},
    )

    def cam_classes_to_list(dict_of_cameras):
        camera_dicts = []
        cam_name_dict = OrderedDict()
        # for cam_name in dict_of_cameras.keys():
        if "camera_random" in dict_of_cameras.keys():
            cam_name = "camera_random"
            cam_class = dict_of_cameras[cam_name]
            num_frames = cam_class.H_c2w.shape[1]
            # if cam_name == "camera_regular": name = "sphere"
            # else: name = "random"
            name = "random"
            cam_name_dict[name] = num_frames
            for i in range(num_frames):
                mdict = blender_open3d_utils.convert_open3d_camera_to_blender(
                    H_c2w=cam_class.H_c2w[0, i],
                    intrinsic=cam_class.intrinsic[0, i],
                    width_px=cam_class.width_px,
                    height_px=cam_class.height_px,
                )
                # mdict["intrinsic"] = mdict["intrinsic"][:3, :3]
                camera_dicts.append(mdict)

        if "camera_regular" in dict_of_cameras.keys():
            cam_name = "camera_regular"
            cam_class = dict_of_cameras[cam_name]
            num_frames = cam_class.H_c2w.shape[1]
            # if cam_name == "camera_regular": name = "sphere"
            # else: name = "random"
            name = "sphere"
            cam_name_dict[name] = num_frames
            for i in range(num_frames):
                mdict = blender_open3d_utils.convert_open3d_camera_to_blender(
                    H_c2w=cam_class.H_c2w[0, i],
                    intrinsic=cam_class.intrinsic[0, i],
                    width_px=cam_class.width_px,
                    height_px=cam_class.height_px,
                )
                # mdict["intrinsic"] = mdict["intrinsic"][:3, :3]
                camera_dicts.append(mdict)

        return camera_dicts, cam_name_dict

    camera_dicts, cam_name_dict = cam_classes_to_list(camera_dicts)
    scene_dict["cameras"] = camera_dicts

    # render
    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
        if blender_render_dir is None:
            tmp_dir = os.path.abspath(tmp_dir)
        else:
            tmp_dir = blender_render_dir
            os.makedirs(tmp_dir, exist_ok=True)

        # save config to tmp file (again) just to keep config
        json_filename = os.path.join(tmp_dir, "config.json")
        with open(json_filename, "w") as f:
            json.dump(scene_dict, f, indent=2, cls=json_utils.NumpyJsonEncoder)

        # render with blender
        tmp_out_dir = os.path.join(tmp_dir, "out")
        blender_cmd = blender_rendering_utils.get_blender_exe()
        blender_script = blender_rendering_utils.get_blender_utils_path()
        blender_log_fname = f"blender_{pathlib.Path(mesh_filename).stem}.log"
        blender_log_f = os.path.join(tmp_dir, blender_log_fname)
        normalized_mesh_fname: str = "blender_normalized_mesh.ply"
        normalization_info_fname: str = "config_after_blender_normalization.json"

        print(f"Rendering data.... {out_dir} ")
        cmd = (
            f"{blender_cmd} --background --log-level 1 --python {blender_script} -- "
            f"--filename {json_filename} --out_dir {tmp_out_dir} "
            f"--normalized_mesh_fname {normalized_mesh_fname} "
            f"--normalization_info_fname {normalization_info_fname} "
            f"--dynamic {int(dynamic)} "
            f"--num_frames {num_frames} "
            f"--frame_start {frame_start} "
            f"--animation_number {animation_number} "
        )
        if printout:
            cmd += " --debug 1 "
        else:
            cmd += f" > {blender_log_fname}"
        print(cmd)
        os.system(cmd)

        from_idx = 0
        fns = glob.glob(os.path.join(tmp_out_dir, "*_0000_srgb.png"))
        num_frames = len(fns)

        for name in cam_name_dict:
            for frame in range(num_frames):
                print(f"Collecting data for {name}")
                rgbd = blender_plib_utils.read_blender_results_to_rgbd(
                    result_dir=tmp_out_dir,
                    from_idx=from_idx,
                    to_idx=from_idx + cam_name_dict[name],
                    from_bidx=frame,
                    to_bidx=frame + 1,
                    use_srgb=True,
                    flag_save_space=flag_save_space,
                    dynamic=dynamic,
                )  # (b=1, q, h, w)

                assert rgbd.rgb.size(1) == cam_name_dict[name]

                save_name = f"rgbd_{name}"  # rgbd_xy, rgbd_yz, rgbd_xz, rgbd_random
                sub_dir = os.path.join(out_dir_rgbd, save_name)
                rgbd: structures.RGBDImage
                rgbd = rgbd.remove_invalid(
                    min_depth=0,
                    max_depth=1e4,
                    background_color=background_color,
                )

                _, sub_index_filename = rgbd.save_as(
                    out_dir=sub_dir,
                    overwrite=overwrite,
                    mode="png",  # 'exr',  # exr is more efficient than npy, png is more efficient than exr
                    background_color=background_color,
                    save_attr_names=save_attr_names,
                    flag_save_space=flag_save_space,
                    ib_filename_offset=frame,
                    concatenate_along_b=True,
                )
                index_dict[save_name] = dict(
                    index_filename=os.path.relpath(sub_index_filename, start=out_dir_rgbd),
                    q=rgbd.rgb.size(1),
                    h=rgbd.rgb.size(2),
                    w=rgbd.rgb.size(3),
                )

                # rgbd = rgbd.remove_invalid(
                #     min_depth=0,
                #     max_depth=1e4,
                #     background_color=background_color,
                # )
                # num_frames = rgbd.shape[0]
                # print(f"Scene has {num_frames} many frames ")

                # # backproject pixel from rgbd images
                # for frame in range(num_frames):
                if name == "sphere":
                    st_pcd = rgbd.get_pcd(
                        subsample=1,
                        remove_background=True,
                        keep_img_idxs=False,
                        compute_ray_feature=False,
                    )

                    prefix = f"{frame:04d}_" if dynamic else ""
                    saved_normalized_mesh_filename = os.path.join(tmp_out_dir, prefix + normalized_mesh_fname)
                    saved_normalization_json_filename = os.path.join(tmp_out_dir, normalization_info_fname)

                    if out_dir_mesh is not None:
                        os.makedirs(pathlib.Path(out_dir_mesh) / f"{frame:06d}", exist_ok=True)
                        shutil.copyfile(
                            pathlib.Path(tmp_out_dir) / saved_normalized_mesh_filename,
                            pathlib.Path(out_dir_mesh) / os.path.join(f"{frame:06d}", normalized_mesh_fname),
                        )
                        shutil.copyfile(
                            pathlib.Path(tmp_out_dir) / saved_normalization_json_filename,
                            pathlib.Path(out_dir_mesh) / os.path.join(f"{frame:06d}", normalization_info_fname),
                        )
                        shutil.copyfile(
                            os.path.join(tmp_dir, "config.json"),
                            pathlib.Path(out_dir_mesh) / os.path.join(f"{frame:06d}", "config.json"),
                        )

                    try:
                        o3d_mesh = o3d.io.read_triangle_mesh(saved_normalized_mesh_filename)
                        # sample points uniformly on surface
                        o3d_pcd = o3d_mesh.sample_points_uniformly(
                            number_of_points=num_points,
                        )
                    except:
                        mesh = trimesh.load(saved_normalized_mesh_filename, process=False)
                        points, face_indices = trimesh.sample.sample_surface(mesh, count=num_points)
                        normals = mesh.face_normals[face_indices]
                        o3d_pcd = o3d.geometry.PointCloud()
                        o3d_pcd.points = o3d.utility.Vector3dVector(points)
                        o3d_pcd.normals = o3d.utility.Vector3dVector(normals)

                    xyz_w = torch.tensor(np.asarray(o3d_pcd.points), dtype=torch.float)  # (n, 3)
                    normal_w = torch.tensor(np.asarray(o3d_pcd.normals), dtype=torch.float)  # (n, 3)
                    del o3d_pcd

                    # color the o3d_pcd with nearest backprojected point
                    tree = cKDTree(st_pcd.xyz_w[0].detach().cpu().float().numpy())
                    _, knn_midx = tree.query(xyz_w.detach().cpu().float().numpy(), k=1, workers=-1)
                    # knn_midx: (n, 1)
                    knn_midx = torch.from_numpy(knn_midx).reshape(xyz_w.size(0))  # (n,)
                    assert knn_midx.shape[0] == xyz_w.size(0)
                    rgb = st_pcd.rgb[0][knn_midx]  # (n, 3)

                    # remove problematic points that are not finite
                    n, _3xyz = xyz_w.shape
                    assert normal_w.shape == xyz_w.shape
                    vmask = xyz_w.isfinite().all(dim=-1)  # (n,)
                    vmask = torch.logical_and(
                        vmask,  # (n,)
                        normal_w.isfinite().all(dim=-1),  # (n,)
                    )  # (n,)
                    assert rgb.shape == xyz_w.shape
                    vmask = torch.logical_and(
                        vmask,  # (n,)
                        rgb.isfinite().all(dim=-1),  # (n,)
                    )  # (n,)

                    # remove invalid points
                    xyz_w = xyz_w[vmask]  # (n, 3)
                    rgb = rgb[vmask]  # (n, 3)
                    normal_w = normal_w[vmask]  # (n, 3)

                    # shuffle
                    ridxs = torch.randperm(xyz_w.size(0), device=xyz_w.device)
                    xyz_w = xyz_w[ridxs]
                    normal_w = normal_w[ridxs]
                    rgb = rgb[ridxs]

                    index_dict = save_sampled_pcd_dynamic(
                        pcd_save_version=2,
                        out_dir=out_dir,
                        index_dict=index_dict,
                        xyz_w=xyz_w,
                        rgb=rgb,
                        normal_w=normal_w,
                        save_np_dtype=np.float32,
                        save_chunk_size=save_chunk_size,
                        internal_folder_name=f"{frame:06d}",
                    )

                    if out_dir_mesh is not None:
                        os.makedirs(pathlib.Path(out_dir_mesh) / f"{frame:06d}", exist_ok=True)
                        shutil.copyfile(
                            pathlib.Path(tmp_out_dir) / saved_normalized_mesh_filename,
                            pathlib.Path(out_dir_mesh) / os.path.join(f"{frame:06d}", normalized_mesh_fname),
                        )
                        shutil.copyfile(
                            pathlib.Path(tmp_out_dir) / saved_normalization_json_filename,
                            pathlib.Path(out_dir_mesh) / os.path.join(f"{frame:06d}", normalization_info_fname),
                        )

            from_idx += cam_name_dict[name]

    # save results
    num_frames = rgbd.shape[0]

    # get mesh relative dir
    if mesh_rel_dir is not None:
        fn = os.path.relpath(mesh_filename, start=mesh_rel_dir)
    else:
        fn = mesh_filename

    index_dict["mesh_filename"] = fn
    index_dict["num_frames"] = num_frames

    # save blender log file for future check
    blender_log_fname = f"blender_{fn[:-4]}.log"
    save_log_f = pathlib.Path(out_dir_rgbd) / blender_log_fname
    print(f"\nmoving {blender_log_fname=} to {save_log_f=}\n")
    shutil.move(blender_log_fname, save_log_f)

    # save json
    json_filename = os.path.join(out_dir, "index.json")
    with open(json_filename, "w") as f:
        json.dump(index_dict, f, indent=2)

    if out_dir_rgbd != out_dir:
        _json_filename = os.path.join(out_dir_rgbd, "index.json")
        with open(_json_filename, "w") as f:
            json.dump(index_dict, f, indent=2)

    out_dict = dict(
        index_dict=index_dict,
        json_filename=json_filename,
        cam_name_dict=cam_name_dict,
    )
    return out_dict


def get_mesh_animation_info(
    mesh_filename: str,
    normalize_mesh: bool,
    return_mesh_xyz_ws: bool,
    out_dir: str = None,  # if None, use tempdir for rendering
    printout: bool = False,
):
    """
    Given a mesh:
    1. normalize the mesh to fit [-1, 1] bounding box (if normalize_mesh is True)
    2. identify animation sequences in the mesh
    3. check motion in the animation sequences and determine candidate short clips
    4. randomly select a short clip
    5. add lighting in the scene
    6. determine the camera position for creating input point cloud (no anti-alias) and target (with anti-alias)

    Returns:
        a dict, which contains:
        animation_names:
            (m,) list of str, the name of the animation
        animation_start_frame_dict:
            dict, animation_name -> start frame index (included)
        animation_ending_frame_dict:
            dict, animation_name -> end frame index (included)
        all_mesh_xyz_ws:
            (m,) list of (T, n, 3xyz_w)

    """
    # compile json file (mesh, lighting, camera)
    scene_dict = dict()

    # mesh
    mdict = dict(
        name="mesh",
        filename=mesh_filename,
        normalize_first=normalize_mesh,  # [-1, 1] aabb box
        H_c2w=np.eye(4),  # no rotation after normalization
        scale=np.array([1.0, 1.0, 1.0]),  # no scaling after normalization
    )
    scene_dict["meshes"] = [mdict]

    # gather information about the animation
    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
        if out_dir is None:
            tmp_dir = os.path.abspath(tmp_dir)
        else:
            tmp_dir = out_dir
            os.makedirs(tmp_dir, exist_ok=True)

        # save config to tmp file (again) just to keep config
        json_filename = os.path.join(tmp_dir, "config.json")
        with open(json_filename, "w") as f:
            json.dump(scene_dict, f, indent=2, cls=json_utils.NumpyJsonEncoder)

        # get animation infos
        blender_cmd = blender_rendering_utils.get_blender_exe()
        blender_script = blender_rendering_utils.get_blender_utils_v2_path()
        blender_log_fname = f"blender_{pathlib.Path(mesh_filename).stem}.log"
        blender_err_fname = f"blender_{pathlib.Path(mesh_filename).stem}.err.log"

        print(f"gathering animation infomation....{tmp_dir} ")
        cmd = (
            f"{blender_cmd} --background --log-level 1 --python {blender_script} -- "
            f"--filename {json_filename} --mode get_animation_info --out_dir {tmp_dir} "
            f"--get_info_return_mesh_xyz_ws {int(return_mesh_xyz_ws)} "
        )

        if printout:
            cmd += " --debug 1 "
        else:
            cmd += f" > {blender_log_fname} 2> {blender_err_fname}"
        print(cmd)
        os.system(cmd)

        animation_json_filename = os.path.join(tmp_dir, "all_animation_info.json")
        assert os.path.exists(animation_json_filename), f"{animation_json_filename} does not exist"
        with open(animation_json_filename, "r") as f:
            animation_infos = json.load(f)  # list of dict

        animation_info = animation_infos[0]  # one mesh
        animation_names = animation_info["animation_names"]  # (m,)
        animation_start_frame_dict = animation_info["animation_start_frame_dict"]
        animation_ending_frame_dict = animation_info["animation_ending_frame_dict"]

        if return_mesh_xyz_ws:
            mesh_xyz_ws_filenames = animation_info.get("mesh_xyz_ws_filenames", None)  # (m,)
            assert mesh_xyz_ws_filenames is not None

            all_mesh_xyz_ws = []
            for i in range(len(mesh_xyz_ws_filenames)):
                filename = os.path.join(tmp_dir, mesh_xyz_ws_filenames[i])
                mesh_xyz_ws = np.load(filename)
                all_mesh_xyz_ws.append(mesh_xyz_ws)
        else:
            all_mesh_xyz_ws = None

    return dict(
        animation_idxs=list(range(len(animation_names))),
        animation_names=animation_names,  # (m,)
        animation_start_frame_dict=animation_start_frame_dict,
        animation_ending_frame_dict=animation_ending_frame_dict,
        all_mesh_xyz_ws=all_mesh_xyz_ws,  # (m,)
    )


def get_prob_of_good_animaitons(
    mesh_filename: str,
    num_frames: int,
    animation_number: T.Optional[int],
    frame_start: T.Optional[int],
    # animation filtering
    animation_min_num_frames: int = 16,  # min number of frames that has motion
    animation_th_velocity_norm: float = 0.01,  # wrt normalized based on first frame [-1, 1]
    animation_th_motion_ratio: float = 0,  # [0, 1]
    animation_th_max_velocity_norm: float = 0.1,  # wrt normalized based on first frame [-1, 1]
    animation_frame_skips: T.List[int] = [1, 2],
    # misc
    out_dir: str = None,  # if None, use tempdir for rendering
    raise_error_if_not_finite: bool = False,
    printout: bool = False,
):
    """
    Compute the selection probabilities of each frame based on
    how good the animation is starting from that frame.

    Args:
        mesh_filename:
        num_frames:
            select `num_frames` duration
        animation_number:
            if given, only consider this animation_number
        frame_start:
            if given, only consider frame starting from this index
        animation_min_num_frames:
            min number of frames that has motion
        animation_th_velocity_norm:
            threshold used to determine is a point moves, wrt normalized based on first frame [-1, 1]
        animation_th_motion_ratio:
            threshold for ratio of vertices, used to determine if a frame has movement,  [0, 1]
        animation_th_max_velocity_norm:
            threshold for max velocity, wrt normalized based on first frame [-1, 1]
        animation_frame_skips:
            candidates of frame skips (ie, 1x, 2x speed)
        out_dir:
        printout:

    Returns:
        all_probs:
            (nn,) all the frame canidates as 1d array, probabiity
        all_animation_idxs:
            (nn,) all the frame canidates as 1d array, animation index
        all_animation_fskips:
            (nn,) all the frame canidates as 1d array, frame skip
        all_start_idxs:
            (nn,) all the frame canidates as 1d array, start index in all_probs
        animation_info:
    """

    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
        if out_dir is None:
            tmp_dir = os.path.abspath(tmp_dir)
        else:
            tmp_dir = out_dir
            os.makedirs(tmp_dir, exist_ok=True)

        animation_info = get_mesh_animation_info(
            mesh_filename=mesh_filename,
            normalize_mesh=True,
            return_mesh_xyz_ws=True,
            out_dir=tmp_dir,
            printout=printout,
        )

        animation_names = animation_info["animation_names"]  # (m,)
        animation_idxs = animation_info["animation_idxs"]  # (m,)
        animation_start_frame_dict = animation_info["animation_start_frame_dict"]
        animation_ending_frame_dict = animation_info["animation_ending_frame_dict"]
        all_mesh_xyz_ws = animation_info["all_mesh_xyz_ws"]  # (m,) of (t, n, 3xyz_w) numpy

    if animation_number is None:
        candidate_animation_idxs = animation_idxs  # (m,)
    else:
        candidate_animation_idxs = [animation_number]

    sidx = 0 if frame_start is None else frame_start

    # label all good animation durations
    all_probs = []  # (m * num_frame_skips,), each is of length (t-1,) start frame
    all_animation_idxs = []  # (m * num_frame_skips,)
    all_animation_fskips = []  # (m * num_frame_skips,)
    all_start_idxs = []  # (m * num_frame_skips,)

    all_valid_velocity_norms = []
    all_valid_motion_ratios = []
    all_valid_num_frames = []

    current_idx = 0
    for ii, aidx in enumerate(candidate_animation_idxs):
        mesh_xyz_ws = all_mesh_xyz_ws[aidx]  # (t, n, 3)
        mesh_xyz_ws = torch.from_numpy(mesh_xyz_ws).float()  # (t, n, 3)

        if len(mesh_xyz_ws) == 0:
            # all_probs.append([torch.zeros(0)] * len(animation_frame_skips))  # (num_skips,) of (0,)
            continue

        if animation_min_num_frames is not None and mesh_xyz_ws.shape[0] < animation_min_num_frames:
            # all_probs.append([torch.zeros(mesh_xyz_ws.size(0)-1)] * len(animation_frame_skips))  # (num_skips,) of (t-1,)
            continue

        # normalize based on the first frame
        _min_xyz_w, _ = torch.min(mesh_xyz_ws[sidx], dim=0)  # (3,)
        _max_xyz_w, _ = torch.max(mesh_xyz_ws[sidx], dim=0)  # (3,)
        _center_xyz_w = (_min_xyz_w + _max_xyz_w) * 0.5  # (3,)
        _scale = torch.max(_max_xyz_w - _min_xyz_w)  # (,)
        mesh_xyz_ws = (mesh_xyz_ws - _center_xyz_w) * (1.999 / _scale)  # (t, n, 3)

        if raise_error_if_not_finite:
            assert torch.all(torch.isfinite(mesh_xyz_ws))
            # check 10000 as we only normalize based on first frame
            assert torch.all(torch.logical_and(mesh_xyz_ws >= -10000, mesh_xyz_ws <= 10000)), (
                f"{mesh_xyz_ws.max()} {mesh_xyz_ws.min()}"
            )

        for frame_skip in animation_frame_skips:
            _mesh_xyz_ws = mesh_xyz_ws[::frame_skip]  # (t, n, 3)
            # compute velocity in the normalized mesh_xyz_ws
            velocity_xyz_w = _mesh_xyz_ws[1:] - _mesh_xyz_ws[:-1]  # (t-1, n, 3)
            velocity_norms = torch.linalg.vector_norm(velocity_xyz_w, dim=-1)  # (t-1, n)

            # mask out too fast motion
            valid_velocity_norms = torch.all(velocity_norms <= animation_th_max_velocity_norm, dim=-1)  # (t-1,) bool

            # mask out motion ratio
            has_motion = velocity_norms >= animation_th_velocity_norm  # (t-1, n) bool
            ratio_motion_frame_mean = torch.mean(has_motion.float(), dim=-1)  # (t-1,)
            valid_motion_ratio = ratio_motion_frame_mean >= animation_th_motion_ratio  # (t-1,) bool

            # aggregate
            valid_mask = torch.logical_and(valid_velocity_norms, valid_motion_ratio)  # (t-1,) bool

            if not valid_mask.any():
                max_velocity_norms_per_frame, _ = torch.max(velocity_norms, dim=-1)  # (t-1,)
                avg_velocity_norms_per_frame = torch.mean(velocity_norms, dim=-1)  # (t-1,)
                std_velocity_norms_per_frame = torch.std(velocity_norms, dim=-1)  # (t-1,)
                avg_ratio_motion_per_frame = ratio_motion_frame_mean  # (t-1,)

                aa = torch.stack(
                    [
                        avg_velocity_norms_per_frame,
                        std_velocity_norms_per_frame,
                        max_velocity_norms_per_frame,
                        avg_ratio_motion_per_frame,
                    ],
                    dim=-1,
                )  # (t-1, 4)

                print(
                    f"{mesh_filename}: aidx {aidx}, fskip {frame_skip}, "
                    f"animation_th_max_velocity_norm {animation_th_max_velocity_norm},"
                    f"animation_th_motion_ratio {animation_th_motion_ratio}:\n "
                    f"avg_v, std_v, max_v, ratio_motion:\n"
                    f"{aa}"
                )

            if animation_min_num_frames is not None and animation_min_num_frames > 0:
                kernel = torch.ones(max(num_frames, animation_min_num_frames))  # (num_frames,)
                # we pad 0 at the end, as we can render beyond the animation end frame idx (they are just still)
                window_sums = torch.nn.functional.conv1d(
                    torch.nn.functional.pad(
                        valid_mask.view(1, 1, -1).float(),  # (b=1, c=1, t-1)
                        pad=[0, kernel.size(0) - 1],
                        mode="constant",
                        value=0,
                    ),  # (b=1, c=1, t-1 + w-1)
                    kernel.view(1, 1, -1),  # (out_channels, in_channels, kernel_size)
                ).view(-1)  # (t-1,)
                assert len(window_sums) == valid_mask.size(0)

                # print(f"num_frames: {num_frames}")
                # print(f"valid_mask: {valid_mask}")
                # print(f"window_sums: {window_sums}")

                valid_num_frames = window_sums >= animation_min_num_frames  # (t-1,)
                valid_mask = torch.logical_and(valid_num_frames, valid_mask)  # (t-1,)
            else:
                valid_num_frames = torch.ones_like(valid_mask)  # (t-1,)

            probs = valid_mask.float()  # (t-1,)
            all_probs.append(probs)  # (t-1,)
            all_animation_idxs.append(torch.ones_like(probs, dtype=torch.long) * aidx)  # (t-1,)
            all_animation_fskips.append(torch.ones_like(probs, dtype=torch.long) * frame_skip)  # (t-1,)
            all_start_idxs.append(torch.ones_like(probs, dtype=torch.long) * current_idx)  # (t-1,)
            current_idx = current_idx + len(probs)

            all_valid_velocity_norms.append(valid_velocity_norms)  # (t-1,)
            all_valid_motion_ratios.append(valid_motion_ratio)  # (t-1,)
            all_valid_num_frames.append(valid_num_frames)  # (t-1,)

    all_probs = torch.cat(all_probs, dim=0) if len(all_probs) > 1 else all_probs[0]  # (m * num_frame_skips,)
    all_animation_idxs = (
        torch.cat(all_animation_idxs, dim=0) if len(all_animation_idxs) > 1 else all_animation_idxs[0]
    )  # (m * num_frame_skips,)
    all_animation_fskips = (
        torch.cat(all_animation_fskips, dim=0) if len(all_animation_fskips) > 1 else all_animation_fskips[0]
    )  # (m * num_frame_skips,)
    all_start_idxs = (
        torch.cat(all_start_idxs, dim=0) if len(all_start_idxs) > 1 else all_start_idxs[0]
    )  # (m * num_frame_skips,)

    all_valid_velocity_norms = (
        torch.cat(all_valid_velocity_norms, dim=0) if len(all_valid_velocity_norms) > 1 else all_valid_velocity_norms[0]
    )  # (m * num_frame_skips,)
    all_valid_motion_ratios = (
        torch.cat(all_valid_motion_ratios, dim=0) if len(all_valid_motion_ratios) > 1 else all_valid_motion_ratios[0]
    )  # (m * num_frame_skips,)
    all_valid_num_frames = (
        torch.cat(all_valid_num_frames, dim=0) if len(all_valid_num_frames) > 1 else all_valid_num_frames[0]
    )  # (m * num_frame_skips,)

    return dict(
        # for selection
        all_probs=all_probs,  # (nn,)
        all_animation_idxs=all_animation_idxs,  # (nn,)
        all_animation_fskips=all_animation_fskips,  # (nn,)
        all_start_idxs=all_start_idxs,  # (nn,)
        # debug
        all_valid_velocity_norms=all_valid_velocity_norms,  # (nn,)
        all_valid_motion_ratios=all_valid_motion_ratios,  # (nn,)
        all_valid_num_frames=all_valid_num_frames,  # (nn,)
        # info
        animation_info=animation_info,
    )


def randomly_select_good_animation(
    mesh_filename: str,
    num_frames: int,
    animation_number: T.Optional[int],
    frame_start: T.Optional[int],
    # animation filtering
    animation_min_num_frames: int = 16,  # min number of frames that has motion
    animation_th_velocity_norm: float = 0.01,  # wrt normalized based on first frame [-1, 1]
    animation_th_motion_ratio: float = 0,  # [0, 1]
    animation_th_max_velocity_norm: float = 0.1,  # wrt normalized based on first frame [-1, 1]
    animation_frame_skips: T.List[int] = [1, 2],
    animation_strategy: str = "raise_error",
    # misc
    out_dir: str = None,  # if None, use tempdir for rendering
    raise_error_if_not_finite: bool = False,
    printout: bool = False,
):
    """
    Compute the selection probabilities of each frame based on
    how good the animation is starting from that frame,
    and then select a good animation_number and / or frame_start.

    Args:
        mesh_filename:
        num_frames:
            select `num_frames` duration
        animation_number:
            if given, only consider this animation_number
        frame_start:
            if given, only consider frame starting from this index
        animation_min_num_frames:
            min number of frames that has motion
        animation_th_velocity_norm:
            threshold used to determine is a point moves, wrt normalized based on first frame [-1, 1]
        animation_th_motion_ratio:
            threshold for ratio of vertices, used to determine if a frame has movement,  [0, 1]
        animation_th_max_velocity_norm:
            threshold for max velocity, wrt normalized based on first frame [-1, 1]
        animation_frame_skips:
            candidates of frame skips (ie, 1x, 2x speed)
        animation_strategy:
            'raise_error': raise an error if no good animation satisfy criteria
            'search_again_with_all_animations_then_raise_error': if animation_number was not None, try again with None
            'search_again_with_all_animations_then_rand_start': use random frame start
            'rand_start': use random frame start
        out_dir:
        printout:

    Returns:
        animation_number:
            int
        frame_start:
            int
        animation_frame_skip:
            int

        all_probs:
            (nn,) all the frame canidates as 1d array, probabiity
        all_animation_idxs:
            (nn,) all the frame canidates as 1d array, animation index
        all_animation_fskips:
            (nn,) all the frame canidates as 1d array, frame skip
        all_start_idxs:
            (nn,) all the frame canidates as 1d array, start index in all_probs
        animation_info:
    """

    out_dict = get_prob_of_good_animaitons(
        mesh_filename=mesh_filename,
        num_frames=num_frames,
        animation_number=animation_number,
        frame_start=frame_start,
        # animation filtering
        animation_min_num_frames=animation_min_num_frames,
        animation_th_velocity_norm=animation_th_velocity_norm,
        animation_th_motion_ratio=animation_th_motion_ratio,
        animation_th_max_velocity_norm=animation_th_max_velocity_norm,
        animation_frame_skips=animation_frame_skips,
        # misc
        out_dir=out_dir,
        raise_error_if_not_finite=raise_error_if_not_finite,
        printout=printout,
    )
    all_probs = out_dict["all_probs"]  # (nn,)
    all_animation_idxs = out_dict["all_animation_idxs"]  # (nn,)
    all_animation_fskips = out_dict["all_animation_fskips"]  # (nn,)
    all_start_idxs = out_dict["all_start_idxs"]  # (nn,)

    # randomly select based on all probs
    if all_probs.sum() > 1e-6:
        idx = torch.multinomial(
            input=all_probs,  # (m * num_frame_skips,)
            num_samples=1,
        )
        animation_number = all_animation_idxs[idx].item()
        animation_frame_skip = all_animation_fskips[idx].item()
        if frame_start is None:
            frame_start = (idx - all_start_idxs[idx]) * animation_frame_skip
            frame_start = frame_start.item()
        print(f"selected idx: {idx}, frame_start = {frame_start}")

        return dict(
            animation_number=animation_number,
            frame_start=frame_start,
            animation_frame_skip=animation_frame_skip,
            **out_dict,
        )

    else:
        # No valid frame nor animation!

        all_valid_velocity_norms = out_dict["all_valid_velocity_norms"]  # (nn,)
        all_valid_motion_ratios = out_dict["all_valid_motion_ratios"]  # (nn,)
        all_valid_num_frames = out_dict["all_valid_num_frames"]  # (nn,)
        aa = torch.stack(
            [
                all_probs.long(),
                all_animation_idxs,
                all_animation_fskips,
                all_start_idxs,
                all_valid_velocity_norms.long(),
                all_valid_motion_ratios.long(),
                all_valid_num_frames.long(),
            ],
            dim=1,
        )  # (nn, 4)

        if animation_strategy == "raise_error":
            raise RuntimeError(f"No valid frame nor animation for {mesh_filename}!!\n num_frame = {num_frames}\n{aa}")

        elif animation_strategy in [
            "search_again_with_all_animations_then_raise_error",
            "search_again_with_all_animations_then_rand_start",
        ]:
            if animation_strategy == "search_again_with_all_animations_then_raise_error":
                next_animation_strategy = "raise_error"
            elif animation_strategy == "search_again_with_all_animations_then_rand_start":
                next_animation_strategy = "rand_start"
            else:
                raise NotImplementedError(f"Unknown animation strategy {animation_strategy}")

            if animation_number is not None or frame_start is not None:
                print(
                    f"No valid frame nor animation for {mesh_filename}!!\nnum_frame = {num_frames}\n{aa}, expand search"
                )
                return randomly_select_good_animation(
                    mesh_filename=mesh_filename,
                    num_frames=num_frames,
                    animation_number=None,
                    frame_start=None,
                    # animation filtering
                    animation_min_num_frames=animation_min_num_frames,  # min number of frames that has motion
                    animation_th_velocity_norm=animation_th_velocity_norm,  # wrt normalized based on first frame [-1, 1]
                    animation_th_motion_ratio=animation_th_motion_ratio,  # [0, 1]
                    animation_th_max_velocity_norm=animation_th_max_velocity_norm,  # wrt normalized based on first frame [-1, 1]
                    animation_frame_skips=animation_frame_skips,
                    animation_strategy=next_animation_strategy,
                    # misc
                    out_dir=out_dir,  # if None, use tempdir for rendering
                    printout=printout,
                )
            else:
                raise RuntimeError(
                    f"No valid frame nor animation for {mesh_filename}!!\n num_frame = {num_frames}\n{aa}"
                )

        elif animation_strategy == "rand_start":
            animation_info = out_dict["animation_info"]
            """
            animation_idxs=list(range(len(animation_names))),
            animation_names=animation_names,  # (m,)
            animation_start_frame_dict=animation_start_frame_dict,
            animation_ending_frame_dict=animation_ending_frame_dict,
            all_mesh_xyz_ws=all_mesh_xyz_ws,  # (m,)
            """

            if animation_number is None:
                # select the one that has the longest number of frames
                animation_num_frames = [
                    animation_info["animation_ending_frame_dict"][aname]
                    - animation_info["animation_start_frame_dict"][aname]
                    + 1
                    for aname in animation_info["animation_names"]
                ]  # (m,)
                animation_number = np.argmax(animation_num_frames).item()

            animation_frame_skip = animation_frame_skips[0]
            total_frame_needed = (num_frames - 1) * animation_frame_skip + 1

            aname = animation_info["animation_names"][animation_number]
            sframe = animation_info["animation_start_frame_dict"][aname]
            eframe = animation_info["animation_ending_frame_dict"][aname]  # included
            avail_frame = eframe - sframe + 1

            eframe = sframe + max(0, avail_frame - total_frame_needed)  # excluded
            frame_start = np.random.randint(low=sframe, high=eframe)

            print("using rand_start")
            return dict(
                animation_number=animation_number,
                frame_start=frame_start,
                animation_frame_skip=animation_frame_skip,
                **out_dict,
            )
        else:
            raise NotImplementedError(f"Unknown animation strategy {animation_strategy}")


def render_rgbd_dynamic_data_v2(
    mesh_filename: str,
    normalize_mesh: bool,
    num_points: int,  # num of points to keep per frame
    # camera
    width_px: int = 532,  # 14 x 37
    height_px: int = 532,  # 14 x 37
    # misc
    background_color: float = 1.0,
    blender_render_dir: str = None,  # if None, use tempdir for rendering
    printout: bool = False,
    dynamic: bool = True,
    num_frames: int = 24,
    out_dir_mesh: T.Optional[str] = None,
    # circular camera
    num_regular_images: int = 10,
    fov: float = 40.0,  # degree
    circular_radius: float = 3.5,  # meter
    # random camera
    num_random_images: int = 30,
    min_fov: float = 40.0,  # degree
    max_fov: float = 60.0,  # degree
    min_random_radius: float = 3,
    max_random_radius: float = 4,
    random_lookat_r: float = 0.25,
    out_dir: str = "/mnt/dynamic_tokenization/blender_rendering/out_pcd",
    save_chunk_size: int = 1_000_000,
    mesh_rel_dir: T.Optional[str] = None,  # used when saving input mesh filename in index_dict
    overwrite: bool = True,
    save_attr_names: T.List[str] = ("rgb", "depth", "hit_map", "normal_w", "alpha"),
    flag_save_space: bool = True,
    frame_start: T.Optional[int] = 0,
    animation_number: T.Optional[int] = 0,
    adjust_camera_pose_per_frame: bool = True,
    # animation filtering
    animation_min_num_frames: int = 16,  # min number of frames that has motion
    animation_th_velocity_norm: float = 0.01,  # wrt normalized based on first frame_idx [-1, 1]
    animation_th_motion_ratio: float = 0,  # [0, 1]
    animation_th_max_velocity_norm: float = 0.2,  # wrt normalized based on first frame_idx [-1, 1]
    animation_frame_skips: T.List[int] = (1, 2),
    animation_strategy: str = "error_out",
    given_camera_random: structures.Camera = None,
    rand_view_mode: str = "m_cont_frames_n_same_views",  # "m_frames_n_views",
    max_num_mesh_vertices: int = -1,
    normalize_bbox_mode: str = "render_clip",  # "whole_animation"
    blender_device: str = "CPU",  # "GPU"
):
    """
    Given a mesh:
    1. normalize the mesh to fit [-1, 1] bounding box (if normalize_mesh is True)
    2. identify animation sequences in the mesh
    3. check motion in the animation sequences and determine candidate short clips
    4. randomly select a short clip
    5. add lighting in the scene
    6. determine the camera position for creating input point cloud (no anti-alias) and target (with anti-alias)

    Args:
        animation_min_num_frames:
            min number of frames that has motion
        animation_th_velocity_norm:
            threshold used to determine is a point moves, wrt normalized based on first frame_idx [-1, 1]
        animation_th_motion_ratio:
            threshold for ratio of vertices, used to determine if a frame_idx has movement,  [0, 1]
        animation_th_max_velocity_norm:
            threshold for max velocity, wrt normalized based on first frame_idx [-1, 1]
        animation_frame_skips:
            candidates of frame_idx skips (ie, 1x, 2x speed)
        animation_strategy:
            'raise_error': raise an error if no good animation satisfy criteria
            'search_again_with_all_animations_then_raise_error': if animation_number was not None, try again with None
            'search_again_with_all_animations_then_rand_start': use random frame start
            'rand_start': use random frame start


    Returns:
        rgbd:
            structure.RGBDImage (1, q, h, w)
        pcd:
            structure.PointCloud (1, n)

    """
    assert isinstance(animation_frame_skips, (list, tuple))
    for fskip in animation_frame_skips:
        assert fskip > 0

    os.makedirs(out_dir, exist_ok=True)

    index_dict = dict()

    # compile json file (mesh, lighting, camera)
    scene_dict = dict()

    # mesh
    mdict = dict(
        name="mesh",
        filename=mesh_filename,
        normalize_first=normalize_mesh,  # [-1, 1] aabb box
        H_c2w=np.eye(4),  # no rotation after normalization
        scale=np.array([1.0, 1.0, 1.0]),  # no scaling after normalization
    )
    scene_dict["meshes"] = [mdict]

    if max_num_mesh_vertices is not None and max_num_mesh_vertices > 0:
        # load mesh
        o3d_mesh = o3d.io.read_triangle_mesh(mesh_filename)
        if not o3d_mesh.has_vertices():
            raise RuntimeError(f"{mesh_filename} has no vertices")
        elif len(o3d_mesh.vertices) > max_num_mesh_vertices:
            raise RuntimeError(
                f"{mesh_filename}: Number of vertices: = {len(o3d_mesh.vertices)} > {max_num_mesh_vertices}"
            )
        print(f"{mesh_filename}: Number of vertices: = {len(o3d_mesh.vertices)}")

    # gather information about the animation and determine rendering region if needed
    if frame_start is None or animation_number is None or len(animation_frame_skips) > 1:
        out_dict = randomly_select_good_animation(
            mesh_filename=mesh_filename,
            num_frames=num_frames,
            animation_number=animation_number,
            frame_start=frame_start,
            # animation filtering
            animation_min_num_frames=animation_min_num_frames,
            animation_th_velocity_norm=animation_th_velocity_norm,
            animation_th_motion_ratio=animation_th_motion_ratio,
            animation_th_max_velocity_norm=animation_th_max_velocity_norm,
            animation_frame_skips=animation_frame_skips,
            animation_strategy=animation_strategy,
            # misc
            out_dir=os.path.join(blender_render_dir, "animation") if blender_render_dir is not None else None,
            raise_error_if_not_finite=True,
            printout=printout,
        )
        frame_start = out_dict["frame_start"]
        animation_number = out_dict["animation_number"]
        animation_frame_skip = out_dict["animation_frame_skip"]

    else:
        ridx = np.random.randint(len(animation_frame_skips))
        animation_frame_skip = animation_frame_skips[ridx]

    assert frame_start is not None
    assert isinstance(frame_start, int), f"{type(frame_start)} is not int"
    assert num_frames is not None
    assert isinstance(num_frames, int), f"{type(num_frames)} is not int"
    assert animation_frame_skip is not None
    assert isinstance(animation_frame_skip, int), f"{type(animation_frame_skip)} is not int"
    assert animation_number is not None
    assert isinstance(animation_number, int), f"{type(animation_number)} is not int"

    print(
        f"===================================\n"
        f"Render with:\n"
        f"  frame_start: {frame_start}\n"
        f"  num_frames: {num_frames}\n"
        f"  animation_frame_skip: {animation_frame_skip}\n"
        f"  animation_number: {animation_number}\n"
        f"==================================="
    )

    index_dict["animation_number"] = animation_number
    index_dict["animation_frame_skip"] = animation_frame_skip
    index_dict["frame_start"] = frame_start

    # lighting
    light_dicts = []
    mdict = dict(
        name="light 0",
        light_type="diffuse",
        color=[1.0, 1.0, 1.0, 1.0],
        strength=1.0,
    )
    light_dicts.append(mdict)
    scene_dict["lighting"] = light_dicts

    # camera
    # get sphere camera (which will be used at every frame_idx)
    H_c2w = rigid_motion.generate_uniform_camera_poses_with_golden_spiral(
        n=num_regular_images,
        r=circular_radius,
        up_method="z",
        invert_y=True,
    )  # (q, 4, 4)  tensor
    intrinsic = (
        torch.from_numpy(
            render.derive_camera_intrinsics(
                width_px=width_px,
                height_px=height_px,
                fov=fov,
            )
        )
        .float()
        .expand(H_c2w.size(0), 3, 3)
    )  # (q, 3, 3)
    camera_sphere = structures.Camera(
        H_c2w=H_c2w.float().expand(num_frames, H_c2w.shape[0], 4, 4),  # (num_frames, q, 4, 4)
        intrinsic=intrinsic.expand(num_frames, H_c2w.shape[0], 3, 3),  # (num_frames, q, 3, 3)
        width_px=width_px,
        height_px=height_px,
    )

    # get random camera (which can be different every frame_idx)
    if given_camera_random is not None:
        camera_random = given_camera_random  # (num_frames, q)
        assert camera_random.H_c2w.size(0) == num_frames, f"camera random shape {camera_random.H_c2w.size()}"
    elif rand_view_mode == "m_frames_n_views":
        H_c2w = rigid_motion.generate_random_camera_poses_lookat(
            n=num_frames * num_random_images,
            pinhole_min_r=min_random_radius,
            pinhole_max_r=max_random_radius,
            lookat_r=random_lookat_r,
            up_method="z",
            invert_y=True,
        ).reshape(num_frames, num_random_images, 4, 4)  # (num_frames, q, 4, 4)

        random_fovs = (
            np.random.rand(num_frames * num_random_images) * (max_fov - min_fov) + min_fov
        )  # (num_frames*q,) degree
        intrinsic = (
            torch.from_numpy(
                render.derive_camera_intrinsics(
                    width_px=width_px,
                    height_px=height_px,
                    fov=random_fovs,  # (num_frames*q,)
                )
            )
            .reshape(num_frames, num_random_images, 3, 3)
            .float()
        )  # (num_frames, q, 3, 3)

        camera_random = structures.Camera(
            H_c2w=H_c2w,  # (num_frames, q, 4, 4)
            intrinsic=intrinsic,  # (num_frames, q, 3, 3)
            width_px=width_px,
            height_px=height_px,
        )
    elif rand_view_mode == "m_cont_frames_n_same_views":
        H_c2w = rigid_motion.generate_random_camera_poses_lookat(
            n=num_random_images,
            pinhole_min_r=min_random_radius,
            pinhole_max_r=max_random_radius,
            lookat_r=random_lookat_r,
            up_method="z",
            invert_y=True,
        ).expand(num_frames, num_random_images, 4, 4)  # (num_frames, q, 4, 4)

        random_fovs = np.random.rand(num_random_images) * (max_fov - min_fov) + min_fov  # (q,) degree
        intrinsic = (
            torch.from_numpy(
                render.derive_camera_intrinsics(
                    width_px=width_px,
                    height_px=height_px,
                    fov=random_fovs,  # (q,)
                )
            )
            .expand(num_frames, num_random_images, 3, 3)
            .float()
        )  # (num_frames, q, 3, 3)

        camera_random = structures.Camera(
            H_c2w=H_c2w,  # (num_frames, q, 4, 4)
            intrinsic=intrinsic,  # (num_frames, q, 3, 3)
            width_px=width_px,
            height_px=height_px,
        )
    else:
        raise NotImplementedError(f"{rand_view_mode} is not implemented")

    # construct camera_dicts, list of list, (num_frames, num_total_views)
    all_camera_dicts = []
    all_cam_name_start_idx_dicts = []
    all_cam_name_num_frame_dicts = []  # list of OrderedDict(), each is cam_name -> num_views

    for frame_idx in range(num_frames):
        camera_dicts = []
        cam_name_start_idx_dict = dict()
        cam_name_dict = dict()

        current_idx = 0
        for name, camera in [
            ["random", camera_random],
            ["sphere", camera_sphere],
        ]:
            cam_name_start_idx_dict[name] = current_idx  # name -> q
            cam_name_dict[name] = camera.H_c2w.size(1)  # name -> q
            current_idx += camera.H_c2w.size(1)
            for qidx in range(camera.H_c2w.size(1)):
                mdict = blender_open3d_utils.convert_open3d_camera_to_blender(
                    H_c2w=camera.H_c2w[frame_idx, qidx],
                    intrinsic=camera.intrinsic[frame_idx, qidx],
                    width_px=camera.width_px,
                    height_px=camera.height_px,
                )
                if name == "random":
                    mdict["filter_width"] = 1.0  # with antialiasing
                    mdict["use_denoising"] = True  # with antialiasing
                elif name == "sphere":
                    mdict["filter_width"] = 0.01  # no antialiasing
                    mdict["use_denoising"] = False  # no antialiasing
                camera_dicts.append(mdict)

        all_camera_dicts.append(camera_dicts)
        all_cam_name_start_idx_dicts.append(cam_name_start_idx_dict)
        all_cam_name_num_frame_dicts.append(cam_name_dict)

    scene_dict["cameras"] = all_camera_dicts  # list (frame_idx) of list (views)

    # render
    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
        if blender_render_dir is None:
            tmp_dir = os.path.abspath(tmp_dir)
        else:
            tmp_dir = blender_render_dir
            os.makedirs(tmp_dir, exist_ok=True)

        # save config to tmp file (again) just to keep config
        json_filename = os.path.join(tmp_dir, "config.json")
        with open(json_filename, "w") as f:
            json.dump(scene_dict, f, indent=2, cls=json_utils.NumpyJsonEncoder)

        # render with blender
        tmp_out_dir = os.path.join(tmp_dir, "out")
        blender_cmd = blender_rendering_utils.get_blender_exe()
        blender_script = blender_rendering_utils.get_blender_utils_v2_path()
        blender_log_fname = f"blender_{pathlib.Path(mesh_filename).stem}.log"
        normalized_mesh_fname: str = "blender_normalized_mesh.ply"

        print("Rendering data....")
        cmd = (
            f"{blender_cmd} --background --log-level 1 --python {blender_script} -- "
            f"--filename {json_filename} --out_dir {tmp_out_dir} "
            f"--normalized_mesh_fname {normalized_mesh_fname} "
            f"--dynamic {int(dynamic)} "
            f"--num_frames {num_frames} "
            f"--frame_start {frame_start} "
            f"--frame_skip {animation_frame_skip} "
            f"--animation_number {animation_number} "
            f"--adjust_camera_pose_per_frame {int(adjust_camera_pose_per_frame)} "
            f"--normalize_bbox_mode {normalize_bbox_mode} "
            f"--device {blender_device.upper()} "
        )
        if printout:
            cmd += " --debug 1 "
        else:
            cmd += f" > {blender_log_fname}"
        print(cmd)
        os.system(cmd)

        fns = glob.glob(os.path.join(tmp_out_dir, "*_0000_srgb.png"))
        print(f"Found {len(fns)} frames")
        assert len(fns) == num_frames, f"Expected {num_frames} frames, got {len(fns)}"

        # gather render images into rgbd images
        for frame_idx in range(num_frames):
            for name in all_cam_name_start_idx_dicts[frame_idx]:
                stime = timer()
                rgbd = blender_plib_utils.read_blender_results_to_rgbd(
                    result_dir=tmp_out_dir,
                    from_idx=all_cam_name_start_idx_dicts[frame_idx][name],
                    to_idx=all_cam_name_start_idx_dicts[frame_idx][name]
                    + all_cam_name_num_frame_dicts[frame_idx][name],
                    from_bidx=frame_idx,
                    to_bidx=frame_idx + 1,
                    use_srgb=True,
                    flag_save_space=flag_save_space,
                    dynamic=dynamic,
                )  # (b=1, q, h, w)
                print(f"reading blender result to rgbd takes {timer() - stime:.3f} seconds")

                assert rgbd.rgb.size(1) == all_cam_name_num_frame_dicts[frame_idx][name]

                save_name = f"rgbd_{name}"
                sub_dir = os.path.join(out_dir, save_name)
                rgbd: structures.RGBDImage

                stime = timer()
                _, sub_index_filename = rgbd.save_as(
                    out_dir=sub_dir,
                    overwrite=overwrite,
                    mode="png",  # 'exr',  # exr is more efficient than npy, png is more efficient than exr
                    background_color=background_color,
                    save_attr_names=save_attr_names,
                    flag_save_space=flag_save_space,
                    ib_filename_offset=frame_idx,
                    concatenate_along_b=True,
                )
                print(f"saving rgbd takes {timer() - stime:.3f} seconds")

                index_dict[save_name] = dict(
                    index_filename=os.path.relpath(sub_index_filename, start=out_dir),
                    q=rgbd.rgb.size(1),
                    h=rgbd.rgb.size(2),
                    w=rgbd.rgb.size(3),
                )

                # backproject pixel from rgbd_sphere images
                if name == "sphere":
                    # save mesh
                    prefix = f"{frame_idx:04d}_" if dynamic else ""
                    saved_normalized_mesh_filename = os.path.join(tmp_out_dir, prefix + normalized_mesh_fname)

                    if out_dir_mesh is not None:
                        os.makedirs(pathlib.Path(out_dir_mesh) / f"{frame_idx:06d}", exist_ok=True)
                        shutil.copyfile(
                            saved_normalized_mesh_filename,
                            pathlib.Path(out_dir_mesh) / os.path.join(f"{frame_idx:06d}", normalized_mesh_fname),
                        )
                        shutil.copyfile(
                            os.path.join(tmp_dir, "config.json"),
                            pathlib.Path(out_dir_mesh) / os.path.join(f"{frame_idx:06d}", "config.json"),
                        )

                    # backproejct pixel from this frame
                    st_pcd = rgbd.get_pcd(
                        subsample=1,
                        remove_background=True,
                        keep_img_idxs=False,
                        compute_ray_feature=True,
                    )  # (b=1, n, 3)
                    st_pcd = st_pcd.extract_valid_point_cloud(bidx=0)  # (b=1, n, 3)

                    # shuffle
                    ridxs = torch.randperm(st_pcd.xyz_w.size(1), device=st_pcd.xyz_w.device)[:num_points]
                    xyz_w = st_pcd.xyz_w[0, ridxs]  # (n, 3)
                    normal_w = st_pcd.normal_w[0, ridxs]  # (n, 3)
                    rgb = st_pcd.rgb[0, ridxs]  # (n, 3) [0, 1]
                    view_dir = st_pcd.captured_view_direction_w[0, ridxs]  # (n, 3)

                    index_dict = save_sampled_pcd_dynamic(
                        pcd_save_version=2,
                        out_dir=out_dir,
                        index_dict=index_dict,
                        xyz_w=xyz_w,
                        rgb=rgb,
                        normal_w=normal_w,
                        view_dir=view_dir,
                        save_np_dtype=np.float32,
                        save_chunk_size=save_chunk_size,
                        internal_folder_name=f"{frame_idx:06d}",
                    )

                    del st_pcd
                del rgbd
                gc.collect()

    # get mesh relative dir
    if mesh_rel_dir is not None:
        fn = os.path.relpath(mesh_filename, start=mesh_rel_dir)
    else:
        fn = mesh_filename

    index_dict["mesh_filename"] = fn
    index_dict["num_frames"] = num_frames

    # save json
    json_filename = os.path.join(out_dir, "index.json")
    with open(json_filename, "w") as f:
        json.dump(index_dict, f, indent=2)

    out_dict = dict(
        index_dict=index_dict,
        json_filename=json_filename,
    )
    return out_dict


def render_rgbd_dynamic_data_simplified(
    mesh_filename: str,
    normalize_mesh: bool,
    num_points: int,  # num of points to keep per frame
    # camera
    width_px: int = 532,  # 14 x 37
    height_px: int = 532,  # 14 x 37
    # misc
    background_color: float = 1.0,
    blender_render_dir: str = None,  # if None, use tempdir for rendering
    printout: bool = False,
    dynamic: bool = True,
    animation_number: int = 0,
    ref_frame_start: int = 0,  # relative to animation_start_frame_idx
    frame_skip: int = 1,
    num_frames: int = 24,
    out_dir_mesh: T.Optional[str] = None,
    # circular camera
    num_regular_images: int = 10,
    fov: float = 40.0,  # degree
    circular_radius: float = 3.5,  # meter
    # random camera
    num_random_images: int = 30,
    min_fov: float = 40.0,  # degree
    max_fov: float = 60.0,  # degree
    min_random_radius: float = 3,
    max_random_radius: float = 4,
    random_lookat_r: float = 0.25,
    out_dir: str = "/mnt/dynamic_tokenization/blender_rendering/out_pcd",
    save_chunk_size: int = 1_000_000,
    mesh_rel_dir: T.Optional[str] = None,  # used when saving input mesh filename in index_dict
    overwrite: bool = True,
    save_attr_names: T.List[str] = ("rgb", "depth", "hit_map", "normal_w", "alpha"),
    flag_save_space: bool = True,
    adjust_camera_pose_per_frame: bool = True,
    given_camera_random: structures.Camera = None,  # (num_frames, q)
    rand_view_mode: str = "m_cont_frames_n_same_views",  # "m_frames_n_views",
    mesh_post_H_c2w: T.Optional[torch.Tensor] = None,
    max_num_mesh_vertices: int = -1,
    normalize_bbox_mode: str = "render_clip",  # "whole_animation"
    error_out_if_not_enough_animated_frames: bool = True,  # we will raise error if the animation does not cover [frame_start, frame_start + num_frames]
    blender_device: str = "CPU",  # "GPU"
):
    """
    Given a mesh:
    1. normalize the mesh to fit [-1, 1] bounding box (if normalize_mesh is True)
    2. identify animation sequences in the mesh
    3. check motion in the animation sequences and determine candidate short clips
    4. randomly select a short clip
    5. add lighting in the scene
    6. determine the camera position for creating input point cloud (no anti-alias) and target (with anti-alias)

    Args:

    Returns:
        rgbd:
            structure.RGBDImage (1, q, h, w)
        pcd:
            structure.PointCloud (1, n)

    """
    assert ref_frame_start is not None
    assert animation_number is not None

    os.makedirs(out_dir, exist_ok=True)

    index_dict = dict()

    # compile json file (mesh, lighting, camera)
    scene_dict = dict()

    # mesh
    mdict = dict(
        name="mesh",
        filename=mesh_filename,
        normalize_first=normalize_mesh,  # [-1, 1] aabb box
        H_c2w=np.eye(4)
        if mesh_post_H_c2w is None
        else mesh_post_H_c2w.cpu().float().numpy(),  # no rotation after normalization
        scale=np.array([1.0, 1.0, 1.0]),  # no scaling after normalization
        post_normalization=mesh_post_H_c2w is not None,
    )
    scene_dict["meshes"] = [mdict]

    if max_num_mesh_vertices is not None and max_num_mesh_vertices > 0:
        # load mesh
        o3d_mesh = o3d.io.read_triangle_mesh(mesh_filename)
        if not o3d_mesh.has_vertices():
            raise RuntimeError(f"{mesh_filename} has no vertices")
        elif len(o3d_mesh.vertices) > max_num_mesh_vertices:
            raise RuntimeError(
                f"{mesh_filename}: Number of vertices: = {len(o3d_mesh.vertices)} > {max_num_mesh_vertices}"
            )
        print(f"{mesh_filename}: Number of vertices: = {len(o3d_mesh.vertices)}")

    # gather information about the animation
    animation_info = get_mesh_animation_info(
        mesh_filename=mesh_filename,
        normalize_mesh=True,
        return_mesh_xyz_ws=False,
        out_dir=os.path.join(blender_render_dir, "animation") if blender_render_dir is not None else None,
        printout=printout,
    )
    animation_names = animation_info["animation_names"]  # (num_animations,)
    animation_start_frame_dict = animation_info["animation_start_frame_dict"]  # animation_name -> int
    animation_ending_frame_dict = animation_info["animation_ending_frame_dict"]  # animation_name -> int, included

    animation_name = animation_names[animation_number]
    animation_start_frame_idx = animation_start_frame_dict[animation_name]
    animation_ending_frame_idx = animation_ending_frame_dict[animation_name]

    print(
        f"{mesh_filename}: Animation {animation_name} ({animation_number}): "
        f"from {animation_start_frame_idx} to {animation_ending_frame_idx}"
    )

    total_frame_needed = 1 + (num_frames - 1) * frame_skip
    if error_out_if_not_enough_animated_frames and (
        ref_frame_start + total_frame_needed > (animation_ending_frame_idx - animation_start_frame_idx + 1)
    ):
        raise RuntimeError(
            f"{mesh_filename}: {animation_start_frame_idx}--{animation_ending_frame_idx}, "
            f"does not cover {ref_frame_start} + {total_frame_needed} (frame_skip = {frame_skip})"
        )

    frame_start = animation_start_frame_idx + ref_frame_start
    animation_frame_skip = frame_skip
    assert frame_start is not None
    assert isinstance(frame_start, int), f"{type(frame_start)} is not int"
    assert num_frames is not None
    assert isinstance(num_frames, int), f"{type(num_frames)} is not int"
    assert animation_frame_skip is not None
    assert isinstance(animation_frame_skip, int), f"{type(animation_frame_skip)} is not int"
    assert animation_number is not None
    assert isinstance(animation_number, int), f"{type(animation_number)} is not int"

    print(
        f"===================================\n"
        f"Render with:\n"
        f"  frame_start: {frame_start}\n"
        f"  num_frames: {num_frames}\n"
        f"  animation_frame_skip: {animation_frame_skip}\n"
        f"  animation_number: {animation_number}\n"
        f"==================================="
    )

    index_dict["animation_number"] = animation_number
    index_dict["animation_frame_skip"] = animation_frame_skip
    index_dict["frame_start"] = frame_start

    # lighting
    light_dicts = []
    mdict = dict(
        name="light 0",
        light_type="diffuse",
        color=[1.0, 1.0, 1.0, 1.0],
        strength=1.0,
    )
    light_dicts.append(mdict)
    scene_dict["lighting"] = light_dicts

    # camera
    # get sphere camera (which will be used at every frame_idx)
    H_c2w = rigid_motion.generate_uniform_camera_poses_with_golden_spiral(
        n=num_regular_images,
        r=circular_radius,
        up_method="z",
        invert_y=True,
    )  # (q, 4, 4)  tensor
    intrinsic = (
        torch.from_numpy(
            render.derive_camera_intrinsics(
                width_px=width_px,
                height_px=height_px,
                fov=fov,
            )
        )
        .float()
        .expand(H_c2w.size(0), 3, 3)
    )  # (q, 3, 3)
    camera_sphere = structures.Camera(
        H_c2w=H_c2w.float().expand(num_frames, H_c2w.shape[0], 4, 4),  # (num_frames, q, 4, 4)
        intrinsic=intrinsic.expand(num_frames, H_c2w.shape[0], 3, 3),  # (num_frames, q, 3, 3)
        width_px=width_px,
        height_px=height_px,
    )

    # get random camera (which can be different every frame_idx)
    if given_camera_random is not None:
        camera_random = given_camera_random  # (num_frames, q)
        assert camera_random.H_c2w.size(0) == num_frames, f"camera random shape {camera_random.H_c2w.size()}"
    elif rand_view_mode == "m_frames_n_views":
        H_c2w = rigid_motion.generate_random_camera_poses_lookat(
            n=num_frames * num_random_images,
            pinhole_min_r=min_random_radius,
            pinhole_max_r=max_random_radius,
            lookat_r=random_lookat_r,
            up_method="z",
            invert_y=True,
        ).reshape(num_frames, num_random_images, 4, 4)  # (num_frames, q, 4, 4)

        random_fovs = (
            np.random.rand(num_frames * num_random_images) * (max_fov - min_fov) + min_fov
        )  # (num_frames*q,) degree
        intrinsic = (
            torch.from_numpy(
                render.derive_camera_intrinsics(
                    width_px=width_px,
                    height_px=height_px,
                    fov=random_fovs,  # (num_frames*q,)
                )
            )
            .reshape(num_frames, num_random_images, 3, 3)
            .float()
        )  # (num_frames, q, 3, 3)

        camera_random = structures.Camera(
            H_c2w=H_c2w,  # (num_frames, q, 4, 4)
            intrinsic=intrinsic,  # (num_frames, q, 3, 3)
            width_px=width_px,
            height_px=height_px,
        )
    elif rand_view_mode == "m_cont_frames_n_same_views":
        H_c2w = rigid_motion.generate_random_camera_poses_lookat(
            n=num_random_images,
            pinhole_min_r=min_random_radius,
            pinhole_max_r=max_random_radius,
            lookat_r=random_lookat_r,
            up_method="z",
            invert_y=True,
        ).expand(num_frames, num_random_images, 4, 4)  # (num_frames, q, 4, 4)

        random_fovs = np.random.rand(num_random_images) * (max_fov - min_fov) + min_fov  # (q,) degree
        intrinsic = (
            torch.from_numpy(
                render.derive_camera_intrinsics(
                    width_px=width_px,
                    height_px=height_px,
                    fov=random_fovs,  # (q,)
                )
            )
            .expand(num_frames, num_random_images, 3, 3)
            .float()
        )  # (num_frames, q, 3, 3)

        camera_random = structures.Camera(
            H_c2w=H_c2w,  # (num_frames, q, 4, 4)
            intrinsic=intrinsic,  # (num_frames, q, 3, 3)
            width_px=width_px,
            height_px=height_px,
        )
    else:
        raise NotImplementedError(f"{rand_view_mode} is not implemented")

    # construct camera_dicts, list of list, (num_frames, num_total_views)
    all_camera_dicts = []
    all_cam_name_start_idx_dicts = []
    all_cam_name_num_frame_dicts = []  # list of OrderedDict(), each is cam_name -> num_views

    for frame_idx in range(num_frames):
        camera_dicts = []
        cam_name_start_idx_dict = dict()
        cam_name_dict = dict()

        current_idx = 0
        for name, camera in [
            ["random", camera_random],
            ["sphere", camera_sphere],
        ]:
            cam_name_start_idx_dict[name] = current_idx  # name -> q
            cam_name_dict[name] = camera.H_c2w.size(1)  # name -> q
            current_idx += camera.H_c2w.size(1)
            for qidx in range(camera.H_c2w.size(1)):
                mdict = blender_open3d_utils.convert_open3d_camera_to_blender(
                    H_c2w=camera.H_c2w[frame_idx, qidx],
                    intrinsic=camera.intrinsic[frame_idx, qidx],
                    width_px=camera.width_px,
                    height_px=camera.height_px,
                )
                if name == "random":
                    mdict["filter_width"] = 1.0  # with antialiasing
                    mdict["use_denoising"] = True  # with antialiasing
                elif name == "sphere":
                    mdict["filter_width"] = 0.01  # no antialiasing
                    mdict["use_denoising"] = False  # no antialiasing
                camera_dicts.append(mdict)

        all_camera_dicts.append(camera_dicts)
        all_cam_name_start_idx_dicts.append(cam_name_start_idx_dict)
        all_cam_name_num_frame_dicts.append(cam_name_dict)

    scene_dict["cameras"] = all_camera_dicts  # list (frame_idx) of list (views)

    # render
    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
        if blender_render_dir is None:
            tmp_dir = os.path.abspath(tmp_dir)
        else:
            tmp_dir = blender_render_dir
            os.makedirs(tmp_dir, exist_ok=True)

        # save config to tmp file (again) just to keep config
        json_filename = os.path.join(tmp_dir, "config.json")
        with open(json_filename, "w") as f:
            json.dump(scene_dict, f, indent=2, cls=json_utils.NumpyJsonEncoder)

        # render with blender
        tmp_out_dir = os.path.join(tmp_dir, "out")
        blender_cmd = blender_rendering_utils.get_blender_exe()
        blender_script = blender_rendering_utils.get_blender_utils_v2_path()
        blender_log_fname = f"blender_{pathlib.Path(mesh_filename).stem}.log"
        normalized_mesh_fname: str = "blender_normalized_mesh.ply"

        print("Rendering data....")
        cmd = (
            f"{blender_cmd} --background --log-level 1 --python {blender_script} -- "
            f"--filename {json_filename} --out_dir {tmp_out_dir} "
            f"--normalized_mesh_fname {normalized_mesh_fname} "
            f"--dynamic {int(dynamic)} "
            f"--num_frames {num_frames} "
            f"--frame_start {frame_start} "
            f"--frame_skip {animation_frame_skip} "
            f"--animation_number {animation_number} "
            f"--adjust_camera_pose_per_frame {int(adjust_camera_pose_per_frame)} "
            f"--normalize_bbox_mode {normalize_bbox_mode} "
            f"--device {blender_device.upper()} "
        )
        if printout:
            cmd += " --debug 1 "
        else:
            cmd += f" > {blender_log_fname}"
        print(cmd)
        os.system(cmd)

        fns = glob.glob(os.path.join(tmp_out_dir, "*_0000_srgb.png"))
        print(f"Found {len(fns)} frames")
        assert len(fns) == num_frames, f"Expected {num_frames} frames, got {len(fns)}"

        # gather render images into rgbd images
        for frame_idx in range(num_frames):
            for name in all_cam_name_start_idx_dicts[frame_idx]:
                stime = timer()
                rgbd = blender_plib_utils.read_blender_results_to_rgbd(
                    result_dir=tmp_out_dir,
                    from_idx=all_cam_name_start_idx_dicts[frame_idx][name],
                    to_idx=all_cam_name_start_idx_dicts[frame_idx][name]
                    + all_cam_name_num_frame_dicts[frame_idx][name],
                    from_bidx=frame_idx,
                    to_bidx=frame_idx + 1,
                    use_srgb=True,
                    flag_save_space=flag_save_space,
                    dynamic=dynamic,
                )  # (b=1, q, h, w)
                print(f"reading blender result to rgbd takes {timer() - stime:.3f} seconds")

                assert rgbd.rgb.size(1) == all_cam_name_num_frame_dicts[frame_idx][name]

                save_name = f"rgbd_{name}"
                sub_dir = os.path.join(out_dir, save_name)
                rgbd: structures.RGBDImage

                stime = timer()
                _, sub_index_filename = rgbd.save_as(
                    out_dir=sub_dir,
                    overwrite=overwrite,
                    mode="png",  # 'exr',  # exr is more efficient than npy, png is more efficient than exr
                    background_color=background_color,
                    save_attr_names=save_attr_names,
                    flag_save_space=flag_save_space,
                    ib_filename_offset=frame_idx,
                    concatenate_along_b=True,
                )
                print(f"saving rgbd takes {timer() - stime:.3f} seconds")

                index_dict[save_name] = dict(
                    index_filename=os.path.relpath(sub_index_filename, start=out_dir),
                    q=rgbd.rgb.size(1),
                    h=rgbd.rgb.size(2),
                    w=rgbd.rgb.size(3),
                )

                # backproject pixel from rgbd_sphere images
                if name == "sphere":
                    # save mesh
                    prefix = f"{frame_idx:04d}_" if dynamic else ""
                    saved_normalized_mesh_filename = os.path.join(tmp_out_dir, prefix + normalized_mesh_fname)

                    if out_dir_mesh is not None:
                        os.makedirs(pathlib.Path(out_dir_mesh) / f"{frame_idx:06d}", exist_ok=True)
                        shutil.copyfile(
                            saved_normalized_mesh_filename,
                            pathlib.Path(out_dir_mesh) / os.path.join(f"{frame_idx:06d}", normalized_mesh_fname),
                        )
                        shutil.copyfile(
                            os.path.join(tmp_dir, "config.json"),
                            pathlib.Path(out_dir_mesh) / os.path.join(f"{frame_idx:06d}", "config.json"),
                        )

                    # backproejct pixel from this frame
                    st_pcd = rgbd.get_pcd(
                        subsample=1,
                        remove_background=True,
                        keep_img_idxs=False,
                        compute_ray_feature=True,
                    )  # (b=1, n, 3)
                    st_pcd = st_pcd.extract_valid_point_cloud(bidx=0)  # (b=1, n, 3)

                    # shuffle
                    ridxs = torch.randperm(st_pcd.xyz_w.size(1), device=st_pcd.xyz_w.device)[:num_points]
                    xyz_w = st_pcd.xyz_w[0, ridxs]  # (n, 3)
                    normal_w = st_pcd.normal_w[0, ridxs]  # (n, 3)
                    rgb = st_pcd.rgb[0, ridxs]  # (n, 3) [0, 1]
                    view_dir = st_pcd.captured_view_direction_w[0, ridxs]  # (n, 3)

                    index_dict = save_sampled_pcd_dynamic(
                        pcd_save_version=2,
                        out_dir=out_dir,
                        index_dict=index_dict,
                        xyz_w=xyz_w,
                        rgb=rgb,
                        normal_w=normal_w,
                        view_dir=view_dir,
                        save_np_dtype=np.float32,
                        save_chunk_size=save_chunk_size,
                        internal_folder_name=f"{frame_idx:06d}",
                    )

                    del st_pcd
                del rgbd
                gc.collect()

    # get mesh relative dir
    if mesh_rel_dir is not None:
        fn = os.path.relpath(mesh_filename, start=mesh_rel_dir)
    else:
        fn = mesh_filename

    index_dict["mesh_filename"] = fn
    index_dict["num_frames"] = num_frames

    # save json
    json_filename = os.path.join(out_dir, "index.json")
    with open(json_filename, "w") as f:
        json.dump(index_dict, f, indent=2)

    out_dict = dict(
        index_dict=index_dict,
        json_filename=json_filename,
    )
    return out_dict


def render_rgbd_static_data_simplified(
    mesh_filename: str,
    normalize_mesh: bool,
    num_points: int,  # num of points to save
    # camera
    width_px: int = 532,  # 14 x 37
    height_px: int = 532,  # 14 x 37
    # misc
    background_color: float = 1.0,
    blender_render_dir: str = None,  # if None, use tempdir for rendering
    printout: bool = False,
    out_dir_mesh: T.Optional[str] = None,
    # circular camera
    num_regular_images: int = 10,
    fov: float = 40.0,  # degree
    circular_radius: float = 3.5,  # meter
    # random camera
    num_random_images: int = 30,
    min_fov: float = 40.0,  # degree
    max_fov: float = 60.0,  # degree
    min_random_radius: float = 3,
    max_random_radius: float = 4,
    random_lookat_r: float = 0.25,
    out_dir: str = "/mnt/dynamic_tokenization/blender_rendering/out_pcd",
    save_chunk_size: int = 50_000_000,
    mesh_rel_dir: T.Optional[str] = None,  # used when saving input mesh filename in index_dict
    overwrite: bool = True,
    save_attr_names: T.List[str] = ("rgb", "depth", "hit_map", "normal_w", "alpha"),
    flag_save_space: bool = True,
    given_camera_random: structures.Camera = None,  # (num_frames, q)
    max_num_mesh_vertices: int = -1,
    keep_existing_lights: bool = False,
    add_diffuse_light: bool = True,
    keep_exact_structure: bool = False,
    blender_device: str = "CPU",  # "GPU"
):
    """
    Args:

    Returns:
        rgbd:
            structure.RGBDImage (1, q, h, w)
        pcd:
            structure.PointCloud (1, n)

    """

    os.makedirs(out_dir, exist_ok=True)

    index_dict = dict()

    # compile json file (mesh, lighting, camera)
    scene_dict = dict()

    # mesh
    mdict = dict(
        name="mesh",
        filename=mesh_filename,
        normalize_first=normalize_mesh,  # [-1, 1] aabb box
        H_c2w=np.eye(4),  # no rotation after normalization
        scale=np.array([1.0, 1.0, 1.0]),  # no scaling after normalization
    )
    scene_dict["meshes"] = [mdict]

    if max_num_mesh_vertices is not None and max_num_mesh_vertices > 0:
        # load mesh
        o3d_mesh = o3d.io.read_triangle_mesh(mesh_filename)
        if not o3d_mesh.has_vertices():
            raise RuntimeError(f"{mesh_filename} has no vertices")
        elif len(o3d_mesh.vertices) > max_num_mesh_vertices:
            raise RuntimeError(
                f"{mesh_filename}: Number of vertices: = {len(o3d_mesh.vertices)} > {max_num_mesh_vertices}"
            )
        print(f"{mesh_filename}: Number of vertices: = {len(o3d_mesh.vertices)}")

    # lighting
    light_dicts = []
    if add_diffuse_light:
        mdict = dict(
            name="light 0",
            light_type="diffuse",
            color=[1.0, 1.0, 1.0, 1.0],
            strength=1.0,
        )
        light_dicts.append(mdict)
    scene_dict["lighting"] = light_dicts

    # camera
    # get sphere camera (which will be used at every frame_idx)
    H_c2w = rigid_motion.generate_uniform_camera_poses_with_golden_spiral(
        n=num_regular_images,
        r=circular_radius,
        up_method="z",
        invert_y=True,
    )  # (q, 4, 4)  tensor
    intrinsic = (
        torch.from_numpy(
            render.derive_camera_intrinsics(
                width_px=width_px,
                height_px=height_px,
                fov=fov,
            )
        )
        .float()
        .expand(H_c2w.size(0), 3, 3)
    )  # (q, 3, 3)
    camera_sphere = structures.Camera(
        H_c2w=H_c2w.float().unsqueeze(0),  # (num_frames=1, q, 4, 4)
        intrinsic=intrinsic.unsqueeze(0),  # (num_frames=1, q, 3, 3)
        width_px=width_px,
        height_px=height_px,
    )

    # get random camera (which can be different every frame_idx)
    if given_camera_random is not None:
        camera_random = given_camera_random  # (num_frames=1, q)
        assert camera_random.H_c2w.size(0) == 1, f"camera random shape {camera_random.H_c2w.size()}"
    else:
        H_c2w = rigid_motion.generate_random_camera_poses_lookat(
            n=num_random_images,
            pinhole_min_r=min_random_radius,
            pinhole_max_r=max_random_radius,
            lookat_r=random_lookat_r,
            up_method="z",
            invert_y=True,
        ).reshape(1, num_random_images, 4, 4)  # (num_frames=1, q, 4, 4)

        random_fovs = np.random.rand(num_random_images) * (max_fov - min_fov) + min_fov  # (q,) degree
        intrinsic = (
            torch.from_numpy(
                render.derive_camera_intrinsics(
                    width_px=width_px,
                    height_px=height_px,
                    fov=random_fovs,  # (q,)
                )
            )
            .reshape(1, num_random_images, 3, 3)
            .float()
        )  # (num_frames=1, q, 3, 3)

        camera_random = structures.Camera(
            H_c2w=H_c2w,  # (num_frames=1, q, 4, 4)
            intrinsic=intrinsic,  # (num_frames=1, q, 3, 3)
            width_px=width_px,
            height_px=height_px,
        )

    # construct camera_dicts, list of list, (num_frames, num_total_views)
    all_camera_dicts = []
    all_cam_name_start_idx_dicts = []
    all_cam_name_num_frame_dicts = []  # list of OrderedDict(), each is cam_name -> num_views

    for frame_idx in range(1):
        camera_dicts = []
        cam_name_start_idx_dict = dict()
        cam_name_dict = dict()

        current_idx = 0
        for name, camera in [
            ["random", camera_random],
            ["sphere", camera_sphere],
        ]:
            cam_name_start_idx_dict[name] = current_idx  # name -> q
            cam_name_dict[name] = camera.H_c2w.size(1)  # name -> q
            current_idx += camera.H_c2w.size(1)
            for qidx in range(camera.H_c2w.size(1)):
                mdict = blender_open3d_utils.convert_open3d_camera_to_blender(
                    H_c2w=camera.H_c2w[frame_idx, qidx],
                    intrinsic=camera.intrinsic[frame_idx, qidx],
                    width_px=camera.width_px,
                    height_px=camera.height_px,
                )
                if name == "random":
                    mdict["filter_width"] = 1.0  # with antialiasing
                    mdict["use_denoising"] = True  # with antialiasing
                elif name == "sphere":
                    mdict["filter_width"] = 0.01  # no antialiasing
                    mdict["use_denoising"] = False  # no antialiasing
                camera_dicts.append(mdict)

        all_camera_dicts.append(camera_dicts)
        all_cam_name_start_idx_dicts.append(cam_name_start_idx_dict)
        all_cam_name_num_frame_dicts.append(cam_name_dict)

    scene_dict["cameras"] = all_camera_dicts  # list (frame_idx=1) of list (views)

    # render
    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
        if blender_render_dir is None:
            tmp_dir = os.path.abspath(tmp_dir)
        else:
            tmp_dir = blender_render_dir
            os.makedirs(tmp_dir, exist_ok=True)

        # save config to tmp file (again) just to keep config
        json_filename = os.path.join(tmp_dir, "config.json")
        with open(json_filename, "w") as f:
            json.dump(scene_dict, f, indent=2, cls=json_utils.NumpyJsonEncoder)

        # render with blender
        tmp_out_dir = os.path.join(tmp_dir, "out")
        blender_cmd = blender_rendering_utils.get_blender_exe()
        blender_script = blender_rendering_utils.get_blender_utils_v2_path()
        blender_log_fname = f"blender_{pathlib.Path(mesh_filename).stem}.log"
        normalized_mesh_fname: str = "blender_normalized_mesh.ply"

        print("Rendering data....")
        cmd = (
            f"{blender_cmd} --background --log-level 1 --python {blender_script} -- "
            f"--filename {json_filename} --out_dir {tmp_out_dir} "
            f"--normalized_mesh_fname {normalized_mesh_fname} "
            f"--dynamic {0} "
            f"--num_frames 1 "
            f"--frame_start {1} "
            f"--frame_skip {1} "
            f"--animation_number {0} "
            f"--adjust_camera_pose_per_frame {0} "
            f"--normalize_bbox_mode render_clip "
            f"--keep_existing_lights {int(keep_existing_lights)} "
            f"--keep_exact_structure {int(keep_exact_structure)} "
            f"--device {blender_device.upper()} "
        )
        if printout:
            cmd += " --debug 1 "
        else:
            cmd += f" > {blender_log_fname}"
        print(cmd)
        os.system(cmd)

        fns = glob.glob(os.path.join(tmp_out_dir, "*_srgb.png"))
        print(f"Found {len(fns)} srgb.png")
        assert len(fns) == (camera_random.H_c2w.shape[1] + camera_sphere.H_c2w.shape[1]), (
            f"Expected {camera_random.H_c2w.shape[1] + camera_sphere.H_c2w.shape[1]} frame, "
            f"got {len(fns)}, {glob.glob(os.path.join(tmp_out_dir, '*'))}"
        )

        # gather render images into rgbd images
        for frame_idx in range(1):
            for name in all_cam_name_start_idx_dicts[frame_idx]:
                stime = timer()
                rgbd = blender_plib_utils.read_blender_results_to_rgbd(
                    result_dir=tmp_out_dir,
                    from_idx=all_cam_name_start_idx_dicts[frame_idx][name],
                    to_idx=all_cam_name_start_idx_dicts[frame_idx][name]
                    + all_cam_name_num_frame_dicts[frame_idx][name],
                    from_bidx=frame_idx,
                    to_bidx=frame_idx + 1,
                    use_srgb=True,
                    flag_save_space=flag_save_space,
                    dynamic=False,
                )  # (b=1, q, h, w)
                print(f"reading blender result to rgbd takes {timer() - stime:.3f} seconds")

                assert rgbd.rgb.size(1) == all_cam_name_num_frame_dicts[frame_idx][name]

                save_name = f"rgbd_{name}"
                sub_dir = os.path.join(out_dir, save_name)
                rgbd: structures.RGBDImage

                stime = timer()
                _, sub_index_filename = rgbd.save_as(
                    out_dir=sub_dir,
                    overwrite=overwrite,
                    mode="png",  # 'exr',  # exr is more efficient than npy, png is more efficient than exr
                    background_color=background_color,
                    save_attr_names=save_attr_names,
                    flag_save_space=flag_save_space,
                    ib_filename_offset=frame_idx,
                    concatenate_along_b=True,
                )
                print(f"saving rgbd takes {timer() - stime:.3f} seconds")

                index_dict[save_name] = dict(
                    index_filename=os.path.relpath(sub_index_filename, start=out_dir),
                    q=rgbd.rgb.size(1),
                    h=rgbd.rgb.size(2),
                    w=rgbd.rgb.size(3),
                )

                # backproject pixel from rgbd_sphere images
                if name == "sphere":
                    # save mesh
                    saved_normalized_mesh_filename = os.path.join(tmp_out_dir, normalized_mesh_fname)

                    if out_dir_mesh is not None:
                        os.makedirs(pathlib.Path(out_dir_mesh) / f"{frame_idx:06d}", exist_ok=True)
                        shutil.copyfile(
                            saved_normalized_mesh_filename,
                            pathlib.Path(out_dir_mesh) / os.path.join(f"{frame_idx:06d}", normalized_mesh_fname),
                        )
                        shutil.copyfile(
                            os.path.join(tmp_dir, "config.json"),
                            pathlib.Path(out_dir_mesh) / os.path.join(f"{frame_idx:06d}", "config.json"),
                        )

                    # backproejct pixel from this frame
                    st_pcd = rgbd.get_pcd(
                        subsample=1,
                        remove_background=True,
                        keep_img_idxs=False,
                        compute_ray_feature=True,
                    )  # (b=1, n, 3)
                    st_pcd = st_pcd.extract_valid_point_cloud(bidx=0)  # (b=1, n, 3)

                    # shuffle
                    ridxs = torch.randperm(st_pcd.xyz_w.size(1), device=st_pcd.xyz_w.device)[:num_points]
                    xyz_w = st_pcd.xyz_w[0, ridxs]  # (n, 3)
                    normal_w = st_pcd.normal_w[0, ridxs]  # (n, 3)
                    rgb = st_pcd.rgb[0, ridxs]  # (n, 3) [0, 1]
                    view_dir = st_pcd.captured_view_direction_w[0, ridxs]  # (n, 3)

                    index_dict = save_sampled_pcd_dynamic(
                        pcd_save_version=2,
                        out_dir=out_dir,
                        index_dict=index_dict,
                        xyz_w=xyz_w,
                        rgb=rgb,
                        normal_w=normal_w,
                        view_dir=view_dir,
                        save_np_dtype=np.float32,
                        save_chunk_size=save_chunk_size,
                        internal_folder_name=f"{frame_idx:06d}",
                    )

                    del st_pcd
                del rgbd
                gc.collect()

    # get mesh relative dir
    if mesh_rel_dir is not None:
        fn = os.path.relpath(mesh_filename, start=mesh_rel_dir)
    else:
        fn = mesh_filename

    index_dict["mesh_filename"] = fn
    index_dict["num_frames"] = 1

    # save json
    json_filename = os.path.join(out_dir, "index.json")
    with open(json_filename, "w") as f:
        json.dump(index_dict, f, indent=2)

    out_dict = dict(
        index_dict=index_dict,
        json_filename=json_filename,
    )
    return out_dict


def get_ray_intersection(
    mesh,
    ray: structures.Ray,
    device: torch.device = torch.device("cpu"),
):
    """
    Intersect the mesh with rays to get ground truth

    Args:
        ray:
            (b, *m_shape)

    Returns:
        ray_rgbs:
            (b, *m_shape, 3)
        ray_ts:
            (b, *m_shape)
        surface_normals_w:
            (b, *m_shape, 3)  in the world coordinate
        hit_map:
            (b, *m_shape)  1: hit, 0: miss
    """

    torch_dtype = ray.origins_w.dtype
    b, *m_shape, _ = ray.origins_w.shape
    rays = torch.cat(
        (
            ray.origins_w,
            ray.directions_w,
        ),
        dim=-1,
    )  # (b, *m, 6)
    rays = rays.detach().cpu().float().numpy()

    # cast the rays, get the intersections
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    raycast_results = scene.cast_rays(rays)
    t_hits = raycast_results["t_hit"].numpy()  # (b, *m), inf if not hit the mesh
    hit_map = 1 - np.isinf(t_hits)  # (b, *m)  1 if hit a surface, 0 otherwise

    # note that primitive_normals is the normal of the triangle face
    # we can use uv map to interpolate vertex normal
    # interpolate surface normal using uv map to get better normal estimation
    if mesh.has_vertex_normals():
        surface_normals = render.interp_surface_normal_from_ray_tracing_results(
            mesh=mesh,
            raycast_results=raycast_results,
        )  # (b, *m, 3)
    else:
        surface_normals = raycast_results["primitive_normals"].numpy()  # (b, *m, 3)

    # if not hit a surface, set surface normal to (0, 0, 0)
    surface_normals = surface_normals * np.expand_dims(hit_map, axis=-1)  # (b, *m, 3)

    # normalize surface normal, and avoid dividing zero by zero
    # note that if not hit a surface, surface normal is set to (0, 0, 0)
    surface_normals_norm = np.linalg.norm(surface_normals, ord=2, axis=-1, keepdims=True)  # (b, *m, 1)
    surface_normals_norm = np.repeat(surface_normals_norm, 3, axis=-1)  # (b, *m, 3)
    valid_mask = surface_normals_norm != 0  # (b, *m, 3)
    surface_normals[valid_mask] = surface_normals[valid_mask] / surface_normals_norm[valid_mask]  # (b, *m, 3)

    # make sure surface normal points to the ray origin (opposite direction of ray_direction)
    ray_directions = ray.directions_w.detach().cpu().numpy()  # (b, *m, 3)
    surface_normals = surface_normals * (-1 * np.sign(np.sum(surface_normals * ray_directions, axis=-1, keepdims=True)))

    # convert to tensor
    ray_ts = torch.from_numpy(t_hits).to(dtype=torch.float, device=device)  # (b, *m)
    surface_normals = torch.from_numpy(surface_normals).to(dtype=torch.float, device=device)  # (b, *m, 3)
    hit_map = torch.from_numpy(hit_map).to(dtype=torch.bool, device=device)  # (b, *m) 1 if hit a surface, 0 otherwise

    return dict(
        ray_ts=ray_ts,
        surface_normals_w=surface_normals,
        hit_map=hit_map,
    )


def render_only_normals_dynamic(out_dir_mesh: str, out_dir_rgbd: str, device="cpu", background_color=1, mode="png"):
    cam_types = ["rgbd_random", "rgbd_sphere"]
    # check if normals exist
    for cam_type in cam_types:
        index_path = os.path.join(out_dir_rgbd, cam_type, "index.json")
        with open(index_path, "r") as f:
            scene_index_dict = json.load(f)

        num_frames = scene_index_dict["b"]
        num_views = scene_index_dict["q"]
        h = scene_index_dict["h"]
        w = scene_index_dict["w"]

        cameras = np.load(os.path.join(out_dir_rgbd, cam_type, "cameras.npz"))
        H_c2w = torch.from_numpy(cameras["H_c2w"])
        intrinsic = torch.from_numpy(cameras["intrinsic"])

        camera = structures.Camera(H_c2w=H_c2w, intrinsic=intrinsic, height_px=h, width_px=w)

        for num_frame in tqdm.tqdm(range(num_frames)):
            sub_dir = os.path.join(out_dir_rgbd, cam_type, f"{num_frame:06d}")
            frame_cameras = camera[num_frame]
            mesh_path = os.path.join(out_dir_mesh, "mesh", f"{num_frame:06}", "blender_normalized_mesh.ply")

            mesh = mesh_utils.load_mesh_using_trimesh(mesh_path)["o3d_mesh"]
            out_dict = get_ray_intersection(mesh, frame_cameras.generate_camera_rays(device=device), device=device)

            normal_w = out_dict["surface_normals_w"]
            hit_map = out_dict["hit_map"]
            normal_filenames = []
            for iq in range(num_views):
                os.makedirs(sub_dir, exist_ok=True)

                _normal_w = (
                    normal_w * hit_map.unsqueeze(-1).to(dtype=normal_w.dtype)
                    + (1 - hit_map.unsqueeze(-1).to(dtype=normal_w.dtype)) * background_color
                )
                if mode == "exr":
                    filename = os.path.join(sub_dir, f"normal_w_{iq:06d}.exr")
                    exr_utils.write_exr(filename, _normal_w[0, iq])  # (h, w, 3)
                elif mode == "npy":
                    filename = os.path.join(sub_dir, f"normal_w_{iq:06d}.npy")
                    np.save(filename, _normal_w[0, iq])  # (h, w, 3)
                elif mode == "png":
                    # since we know the max range of normal [-1, 1] we can save as uint16
                    # with limited loss
                    filename = os.path.join(sub_dir, f"normal_w_{iq:06d}.png")
                    _arr = (
                        (((_normal_w[0, iq] + 1) * 0.5) * 65535)
                        .detach()
                        .cpu()
                        .clamp(min=0, max=65535)
                        .numpy()
                        .astype(np.uint16)
                    )  # (h, w, 3xyz)
                    # since we use opencv (which takes images as bgr)
                    # _arr_bgr = cv2.cvtColor(_arr, cv2.COLOR_RGB2BGR)  # (h, w, 3zyx)
                    _arr_bgr = _arr[..., ::-1]  # (h, w, 3zyx)
                    rs = cv2.imwrite(filename, _arr_bgr)
                    assert rs == True
                else:
                    raise NotImplementedError
                normal_filenames.append(os.path.relpath(filename, start=os.path.join(out_dir_rgbd, cam_type)))
            scene_index_dict["sub_index_dicts"][num_frame]["normal_w"] = normal_filenames
        with open(index_path, "w") as f:
            json.dump(scene_index_dict, f, indent=2)


def get_normalized_meshes(
    mesh_filename: str,
    normalize_mesh: bool,
    blender_render_dir: str = None,  # if None, use tempdir for rendering
    printout: bool = False,
    dynamic: bool = True,
    animation_number: int = 0,
    frame_start: int = 0,  # relative to animation_start_frame_idx
    frame_skip: int = 1,
    num_frames: int = 24,
    out_dir_mesh: T.Optional[str] = None,
    out_dir: str = "/mnt/dynamic_tokenization/blender_rendering/out_pcd",
    mesh_rel_dir: T.Optional[str] = None,  # used when saving input mesh filename in index_dict
    overwrite: bool = True,
    bbox_method: str = "v3",  # "v1" "v2"
):
    """
    Given a mesh:
    1. normalize the mesh to fit [-1, 1] bounding box (if normalize_mesh is True)
    2. identify animation sequences in the mesh
    3. check motion in the animation sequences and determine candidate short clips
    4. randomly select a short clip
    5. add lighting in the scene
    6. determine the camera position for creating input point cloud (no anti-alias) and target (with anti-alias)

    Args:

    Returns:
        rgbd:
            structure.RGBDImage (1, q, h, w)
        pcd:
            structure.PointCloud (1, n)

    """
    assert animation_number is not None

    os.makedirs(out_dir, exist_ok=True)

    index_dict = dict()

    # compile json file (mesh, lighting, camera)
    scene_dict = dict()

    # mesh
    mdict = dict(
        name="mesh",
        filename=mesh_filename,
        normalize_first=normalize_mesh,  # [-1, 1] aabb box
        H_c2w=np.eye(4),  # no rotation after normalization
        scale=np.array([1.0, 1.0, 1.0]),  # no scaling after normalization
    )
    scene_dict["meshes"] = [mdict]

    animation_frame_skip = frame_skip
    assert frame_start is not None
    assert isinstance(frame_start, int), f"{type(frame_start)} is not int"
    assert num_frames is not None
    assert isinstance(num_frames, int), f"{type(num_frames)} is not int"
    assert animation_frame_skip is not None
    assert isinstance(animation_frame_skip, int), f"{type(animation_frame_skip)} is not int"
    assert animation_number is not None
    assert isinstance(animation_number, int), f"{type(animation_number)} is not int"

    print(
        f"===================================\n"
        f"Render with:\n"
        f"  frame_start: {frame_start}\n"
        f"  num_frames: {num_frames}\n"
        f"  animation_frame_skip: {animation_frame_skip}\n"
        f"  animation_number: {animation_number}\n"
        f"==================================="
    )

    index_dict["animation_number"] = animation_number
    index_dict["animation_frame_skip"] = animation_frame_skip
    index_dict["frame_start"] = frame_start

    # lighting
    light_dicts = []
    mdict = dict(
        name="light 0",
        light_type="diffuse",
        color=[1.0, 1.0, 1.0, 1.0],
        strength=1.0,
    )
    light_dicts.append(mdict)
    scene_dict["lighting"] = light_dicts

    # camera
    # get sphere camera (which will be used at every frame_idx)
    H_c2w = rigid_motion.generate_uniform_camera_poses_with_golden_spiral(
        n=1,
        r=3.5,
        up_method="z",
        invert_y=True,
    )  # (q, 4, 4)  tensor
    intrinsic = (
        torch.from_numpy(
            render.derive_camera_intrinsics(
                width_px=512,
                height_px=51,
                fov=40,
            )
        )
        .float()
        .expand(H_c2w.size(0), 3, 3)
    )  # (q, 3, 3)
    camera_sphere = structures.Camera(
        H_c2w=H_c2w.float().expand(num_frames, H_c2w.shape[0], 4, 4),  # (num_frames, q, 4, 4)
        intrinsic=intrinsic.expand(num_frames, H_c2w.shape[0], 3, 3),  # (num_frames, q, 3, 3)
        width_px=512,
        height_px=512,
    )

    # construct camera_dicts, list of list, (num_frames, num_total_views)
    all_camera_dicts = []
    all_cam_name_start_idx_dicts = []
    all_cam_name_num_frame_dicts = []  # list of OrderedDict(), each is cam_name -> num_views

    for frame_idx in range(num_frames):
        camera_dicts = []
        cam_name_start_idx_dict = dict()
        cam_name_dict = dict()

        current_idx = 0
        for name, camera in [
            ["sphere", camera_sphere],
        ]:
            cam_name_start_idx_dict[name] = current_idx  # name -> q
            cam_name_dict[name] = camera.H_c2w.size(1)  # name -> q
            current_idx += camera.H_c2w.size(1)
            for qidx in range(camera.H_c2w.size(1)):
                mdict = blender_open3d_utils.convert_open3d_camera_to_blender(
                    H_c2w=camera.H_c2w[frame_idx, qidx],
                    intrinsic=camera.intrinsic[frame_idx, qidx],
                    width_px=camera.width_px,
                    height_px=camera.height_px,
                )
                if name == "random":
                    mdict["filter_width"] = 1.0  # with antialiasing
                    mdict["use_denoising"] = True  # with antialiasing
                elif name == "sphere":
                    mdict["filter_width"] = 0.01  # no antialiasing
                    mdict["use_denoising"] = False  # no antialiasing
                camera_dicts.append(mdict)

        all_camera_dicts.append(camera_dicts)
        all_cam_name_start_idx_dicts.append(cam_name_start_idx_dict)
        all_cam_name_num_frame_dicts.append(cam_name_dict)

    scene_dict["cameras"] = all_camera_dicts  # list (frame_idx) of list (views)

    # render
    o3d_meshes = []
    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        os.makedirs(tmp_dir, exist_ok=True)
        if blender_render_dir is None:
            tmp_dir = os.path.abspath(tmp_dir)
        else:
            tmp_dir = blender_render_dir
            os.makedirs(tmp_dir, exist_ok=True)

        # save config to tmp file (again) just to keep config
        json_filename = os.path.join(tmp_dir, "config.json")
        with open(json_filename, "w") as f:
            json.dump(scene_dict, f, indent=2, cls=json_utils.NumpyJsonEncoder)

        # render with blender
        tmp_out_dir = os.path.join(tmp_dir, "out")
        blender_cmd = blender_rendering_utils.get_blender_exe()
        blender_script = blender_rendering_utils.get_blender_utils_v2_path()
        blender_log_fname = f"blender_{pathlib.Path(mesh_filename).stem}.log"
        normalized_mesh_fname: str = "blender_normalized_mesh.ply"

        print("Rendering data....")
        cmd = (
            f"{blender_cmd} --background --log-level 1 --python {blender_script} -- "
            f"--filename {json_filename} --out_dir {tmp_out_dir} "
            f"--normalized_mesh_fname {normalized_mesh_fname} "
            f"--dynamic {int(dynamic)} "
            f"--num_frames {num_frames} "
            f"--frame_start {frame_start} "
            f"--frame_skip {animation_frame_skip} "
            f"--animation_number {animation_number} "
            f"--bbox_method {bbox_method} "
            f"--mode get_normalized_meshes "
        )
        if printout:
            cmd += " --debug 1 "
        else:
            cmd += f" > {blender_log_fname}"
        print(cmd)
        os.system(cmd)

        # gather render images into rgbd images
        for frame_idx in range(num_frames):
            # save mesh
            prefix = f"{frame_idx:04d}_" if dynamic else ""
            saved_normalized_mesh_filename = os.path.join(tmp_out_dir, prefix + normalized_mesh_fname)
            o3d_mesh = o3d.io.read_triangle_mesh(saved_normalized_mesh_filename)
            o3d_meshes.append(o3d_mesh)

            if out_dir_mesh is not None:
                os.makedirs(pathlib.Path(out_dir_mesh) / f"{frame_idx:06d}", exist_ok=True)
                shutil.copyfile(
                    saved_normalized_mesh_filename,
                    pathlib.Path(out_dir_mesh) / os.path.join(f"{frame_idx:06d}", normalized_mesh_fname),
                )
                shutil.copyfile(
                    os.path.join(tmp_dir, "config.json"),
                    pathlib.Path(out_dir_mesh) / os.path.join(f"{frame_idx:06d}", "config.json"),
                )

    # get mesh relative dir
    if mesh_rel_dir is not None:
        fn = os.path.relpath(mesh_filename, start=mesh_rel_dir)
    else:
        fn = mesh_filename

    index_dict["mesh_filename"] = fn
    index_dict["num_frames"] = num_frames

    # save json
    json_filename = os.path.join(out_dir, "index.json")
    with open(json_filename, "w") as f:
        json.dump(index_dict, f, indent=2)

    out_dict = dict(
        index_dict=index_dict,
        json_filename=json_filename,
        bbox_method=bbox_method,
        o3d_meshes=o3d_meshes,  # list of (num_frames,)
    )
    return out_dict


def set_memory_limit(max_memory_gb: float):
    def _limit_memory():
        if max_memory_gb is not None and max_memory_gb > 0:
            memory_limit_byte = int(max_memory_gb * 1024 * 1024 * 1024)
            try:
                # not work on mac (raise error)
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit_byte, memory_limit_byte))
            except:
                pass

    return _limit_memory


def render_with_blender(
    out_dir: str,
    mesh_dicts: T.List[T.Dict[str, T.Any]],  # (num_meshes,)
    light_dicts: T.List[T.Dict[str, T.Any]],  # (num_lights,)
    cam_dicts: T.List[T.List[T.Dict[str, T.Any]]],  # (num_frames, num_views)
    cycles_settings: T.Dict[str, T.Any],
    view_layer_settings: T.Dict[str, T.Any],
    # animation settings
    frame_start: int = 0,
    frame_skip: int = 1,
    animation_number: int = 0,
    normalize_bbox_mode: str = "render_clip",
    # advanced camera setting:
    adjust_camera_pose_per_frame: bool = False,
    # other
    normalize_entire_scene: bool = True,
    keep_existing_lights: bool = False,
    # misc
    render_mode: str = "render_json",
    read_result_to_rgbd: bool = False,
    blender_version: str = "4.2.0",
    blender_device: str = "CPU",  # "CPU", "GPU"
    blender_exe: str = None,
    overwrite: bool = False,
    max_memory_gb: float = None,
    timeout: float = None,  # secs
    debug: bool = False,
):
    """
    A bridge function to call blender_utils_v3.

    Args:
        out_dir:
            rendered results will be saved under `out_dir`, eg,
            {frame_idx}_{view_idx}_srgb.png

        mesh_dicts:
            (num_meshes,), list of dict containing the arguments to `read_mesh`:
                name:
                    can be arbitrary
                filename:
                    filename of the mesh (ideally absoluate path)
                pre_H_c2w:
                    (4, 4) rotation and translation (no scale) before first normalization.
                    Blender uses x-right, y-far, z-up.
                pre_scale:
                    (3,) or null, before first normalization.
                normalize_first:
                    bool, whether to normalize the mesh to [-1, 1] before applying H_c2w and scale and
                    after pre_H_c2w and pre_scale
                H_c2w:
                    (4, 4) rotation and translation (no scale) after normalization.
                    Blender uses x-right, y-far, z-up.
                    We will not convert the H_c2w to blender format, as there is no need.
                scale:
                    (3xyz,) or null, after normalization
                post_normalization:
                    bool, after applying H_c2w and scale, whether to apply another normalization.
                cut_aabb_center:
                    (3,) center xyz_w of the cutting aabb. No cutting if None.
                cut_aabb_radius:
                    (3,), radius (half width) for xyz.  No cutting if None.

        light_dicts:
            (num_lights,), list of dict containing the arguments to `read_lighting`:
                name:
                    str, Name of the light object.
                light_type:
                    Literal["POINT", "SUN", "SPOT", "AREA"], Type of the light.
                H_c2w:
                    (4, 4). Blender uses x-right, y-far, z-up.
                    Treat it as open3d camera, Light is toward +z_c.
                    We will convert it to blender format (light toward -z_c).
                energy:
                    float, Energy of the light.
                use_shadow:
                    (bool, optional): Whether to use shadows. Defaults to False.
                specular_factor:
                    (float, optional): Specular factor of the light. Defaults to 1.0.

                # AREA
                size:
                    float, optional for area light, full width along x axis
                size_y:
                    float, optional for area light, full width along y axis. None: the same as size

                # SPOT
                shadow_soft_size:
                    radius of the spotlight itself (in meter). It is the point size.
                spot_size:
                    angle (in degree) of the spot size (max = 180)

                # SUN
                angle:
                    angular diameter of the sun as seen from the earth (in degree). max = 180

                # POINT
                shadow_soft_size:
                    radius of the point itself (in meter).

        cam_dicts:
            (num_frames, num_views,), list of list of dict containing the arguments to `read_camera`:
                intrinsic:
                    (3, 3), open3d format
                H_c2w:
                    (4, 4), open3d format
                width_px:
                    int
                height_px:
                    int

                # optionally
                film_exposure:
                    float. If None, use the defualt = 1.
                filter_width:
                    float, anti-aliasing filter width (min=0.01).  If None, use the setting in cycles_settings.
                use_denoising:
                    bool.  Should be false. If None, use the setting in cycles_settings.

        cycles_settings:
            dict. cycles_settings contains the arguments of `setup_blender_cycles`
            Any arguments not presented will use the default argument values.
            (Support by blender_utils_v3.py)

            samples:
                int, number of samples per pixel
            max_bounces:
                Maximum number of light bounces. For best quality, this should be set to the maximum.
                However, in practice, it may be good to set it to lower values for faster rendering.
                A value of 0 bounces results in direct lighting only.
            diffuse_bounces:
                int, max number of diffuse bounces.  Blender default = 4.
            glossy_bounces:
                int, number of glossy bounces.  Blender default = 4.
            transparent_max_bounces:
                int, number of transparent bounces.  Blender default = 8.
                Note, the maximum number of transparent bounces is controlled separately from other bounces.
            volume_bounces:
                Maximum number of volume scattering bounces. Blender default = 0.
            transmission_bounces:
                int, number of transmission bounces.  Blender default = 12.

            filter_width:
                float, anti-aliasing pixel fileter width
            use_denoising:
                bool, whether to use 2d image-based denoising. Should be False to avoid black boundary pixels.

        view_layer_settings:
            dict. view_layer_settings contains the arguments of `setup_blender_view_layers`
            Any arguments not presented will use the default argument values.
            (Support by blender_utils_v3.py)

            view_layer_pass_alpha_threshold:
                Probability of a ray pass through a (semi)-transparent surface.
                Blender's default is 0.5.
                Z, Index, normal, UV and vector passes are only affected by surfaces with alpha transparency equal to
                or higher than this threshold. With value 0.0 the first surface hit will always write to these passes,
                regardless of transparency. With higher values surfaces that are mostly transparent can be skipped
                until an opaque surface is encountered.
            save_srgb:
                bool, whether to save the tone-mapped srgb. Default = True.
            save_depth:
                bool, whether to save the depth maps (z_c). Default = True.
            save_normal:
                bool, whether to save the normal_map (in world coordinate). Default = True.
            save_albedo:
                bool, whether to save the diffuse component of the rendering. Default = True.
            save_obj_id:
                bool, whether to save the object id. Default = True.

        # animation settings
        frame_start:
            int, which frame to start the rendering
        frame_skip:
            int, controls the animation speed
        animation_number:
            int, animation number
        normalize_bbox_mode:
            'render_clip': normalize based on the rendering frames only
            'whole_animation': normalize based on the entire animation
            'first_frame': normalize based on the first frame

        # advanced camera setting:
        adjust_camera_pose_per_frame:
            bool, whether to consider the given camera pose is relative to
            the bbox center and scale of each frame.

        # other
        normalize_entire_scene:
            bool, whether to normalize again the scene after placing all meshes
            so the scene is bounded by [-1, 1]^3

        keep_existing_lights:
            bool, whether to keep the existing lights in the scene.
            If False, we remove all light components in the scene and meshes.

        read_result_to_rgbd:
            bool, if True, assume all rendered results to rgbd.
            It requires all frames have the same number of views, and
            all rendered images have the same resolution.

        max_memory_gb:
            float, in GB, limit the amount of cpu memoery blender can use. None: inf

        timeout:
            float, in secs.  Allows blender to run `timeout` secs. None: inf
            Otherwise it will raise TimeoutExpired

    Returns:
        config_dict:
            dict, the config dict to blender_utils_v3.py
        out_dir:
            str
        rgbd:
            (num_frames, num_view, h, w) or None

    Note:
        To read the rendered results, you can use below
        (assuming all frames have the same number of views, and
        all rendered images have the same resolution):

        ```
        rgbd = blender_plib_utils.read_blender_results_to_rgbd(
            result_dir=out_dir,
        )  # (num_frames, num_views, h, w)
        ```
    """
    json_filename = os.path.join(out_dir, "config.json")
    if os.path.exists(json_filename):
        if overwrite:
            print(f"removing existing rendering results in {out_dir}", flush=True)
            shutil.rmtree(out_dir)
        else:
            raise RuntimeError(f"{out_dir} not empty")

    # make sure cam_dicts is list of list (num_frames, num_views)
    assert isinstance(cam_dicts, (list, tuple))
    assert len(cam_dicts) > 0
    assert isinstance(cam_dicts[0], (list, tuple))
    num_frames = len(cam_dicts)

    # convert H_c2w of camera and light
    light_dicts = copy.deepcopy(light_dicts)
    for light_dict in light_dicts:
        if light_dict.get("H_c2w", None) is None:
            continue
        H_c2w = light_dict["H_c2w"]
        if isinstance(H_c2w, torch.Tensor):
            H_c2w = H_c2w.detach().cpu().float().numpy()
        light_dict["H_c2w"] = blender_open3d_utils.convert_open3d_H_c2w_to_blender(
            H_c2w=H_c2w,  # (4, 4)
        )  # (4, 4)

    cam_dicts = copy.deepcopy(cam_dicts)
    for cam_view_dicts in cam_dicts:
        for cam_dict in cam_view_dicts:
            if cam_dict.get("H_c2w", None) is None:
                continue
            H_c2w = cam_dict["H_c2w"]
            if isinstance(H_c2w, torch.Tensor):
                H_c2w = H_c2w.detach().cpu().float().numpy()
            cam_dict["H_c2w"] = blender_open3d_utils.convert_open3d_H_c2w_to_blender(
                H_c2w=H_c2w,  # (4, 4)
            )  # (4, 4)

    # compose json
    config_dict = dict(
        meshes=mesh_dicts,  # (num_meshes,)
        cameras=cam_dicts,  # (num_frames, num_views)
        lighting=light_dicts,  # (num_lights,)
        cycles_settings=cycles_settings,
        view_layer_settings=view_layer_settings,
    )

    os.makedirs(os.path.dirname(json_filename), exist_ok=True)
    with open(json_filename, "w") as f:
        json.dump(config_dict, f, indent=2, cls=json_utils.NumpyJsonEncoder)

    # render!
    out_dir = os.path.abspath(out_dir)
    if blender_exe is None:
        blender_exe = blender_rendering_utils.get_blender_exe(version=blender_version)
    else:
        assert os.path.exists(blender_exe), f"{blender_exe} not exists"
    blender_script_filename = blender_rendering_utils.get_blender_utils_v3_path()  # note that v3

    # cmd = (
    #     f"{blender_exe} --background --python {blender_script_filename} -- "
    #     f"--filename {json_filename} "
    #     f"--out_dir {out_dir} "
    #     f"--num_frames {num_frames} "
    #     f"--frame_start {frame_start} "
    #     f"--frame_skip {frame_skip} "
    #     f"--dynamic 1 "  # can always use dynamic
    #     f"--animation_number {int(animation_number)} "
    #     f"--device {blender_device.upper()} "
    #     f"--adjust_camera_pose_per_frame {int(adjust_camera_pose_per_frame)} "
    #     f"--normalize_bbox_mode {normalize_bbox_mode} "
    #     f"--normalize_entire_scene {int(normalize_entire_scene)} "
    #     f"--debug {int(debug)} "
    # )

    if blender_device.lower() == "cpu":
        num_threads = 0
    elif blender_device.lower() == "gpu":
        num_threads = min(16, os.cpu_count())
    else:
        raise NotImplementedError(blender_device)

    cmd = [
        f"{blender_exe}",
        "-t",
        f"{num_threads}",
        "--background",
        "--python",
        f"{blender_script_filename}",
        "--",
        "--mode",
        f"{render_mode}",
        "--filename",
        f"{json_filename}",
        "--out_dir",
        f"{out_dir}",
        "--num_frames",
        f"{num_frames}",
        "--frame_start",
        f"{frame_start}",
        "--frame_skip",
        f"{frame_skip}",
        "--dynamic",
        "1",  # can always use dynamic
        "--animation_number",
        f"{int(animation_number)}",
        "--device",
        f"{blender_device.upper()}",
        "--adjust_camera_pose_per_frame",
        f"{int(adjust_camera_pose_per_frame)}",
        "--normalize_bbox_mode",
        f"{normalize_bbox_mode}",
        "--normalize_entire_scene",
        f"{int(normalize_entire_scene)}",
        "--keep_existing_lights",
        f"{int(keep_existing_lights)}",
        "--debug",
        f"{int(debug)}",
    ]

    print(f"cmd: {' '.join(cmd)}")
    # os.system(' '.join(cmd))

    # make sure blender does not use all available memory
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    available_gb = min(60, available_gb)
    if max_memory_gb is None:
        max_memory_gb = available_gb * 0.8
    else:
        max_memory_gb = min(max_memory_gb, available_gb * 0.8)
    env = os.environ.copy()

    print(f"cmd: {' '.join(cmd)}, timeout = {timeout}, max_memory_gb = {max_memory_gb}")
    result = subprocess.run(
        cmd,
        env=env,
        timeout=timeout,  # secs
        preexec_fn=set_memory_limit(max_memory_gb),
        check=False,
    )
    if result.returncode == 0:
        pass
    else:
        raise RuntimeError(f"cmd: {' '.join(cmd)} failed with code {result.returncode}")

    if read_result_to_rgbd:
        rgbd = blender_plib_utils.read_blender_results_to_rgbd(
            result_dir=out_dir,
            from_idx=0,
            to_idx=None,
            from_bidx=0,
            to_bidx=None,
            use_srgb=True,
            flag_save_space=False,
            dynamic=None,  # auto-detect
            th_alpha=0.5,
            min_depth=0.0,
            max_depth=1.0e4,
            fps=24,
        )  # (num_frames, num_views, h, w)
    else:
        rgbd = None

    return dict(
        config_dict=config_dict,
        out_dir=out_dir,
        rgbd=rgbd,
    )


def construct_light_dicts_for_random_lighting(
    light_types: T.List[str] = ("POINT", "SPOT", "AREA"),
    num_lights: int = None,  # None: randomly sample number of lights using min_num_lights and max_num_lights
    min_num_lights: int = 3,
    max_num_lights: int = 10,
    min_total_light_energy: float = 200,
    max_total_light_energy: float = 2000,
    min_random_radius: float = 7,
    max_random_radius: float = 10.5,
    random_lookat_r: float = 0.25,
    light_type_settings: T.Dict[str, T.Dict[str, T.Any]] = dict(
        AREA=dict(
            size={"min": 5.0, "max": 20.0},
        ),
        SPOT=dict(
            shadow_soft_size={"min": 0.0, "max": 5.0},
        ),
    ),
    up_dir: str = "z",
    seed: int = None,
) -> T.List[T.Dict[str, T.Any]]:
    """
    This function takes a light_info_dict, as is commonly used in the rest of the file,
    and converts them into the light_dicts that our blender rendering pipeline expects.

    Args:
        light_types:
            list of str. It specifies the set of lighting types that we can use if we use "random" lighting
        num_lights:
            int or None, the number of lights to use with "random" lighting.
            If it is None, we will randomly sample the number of lights
        min_num_lights / max_num_lights:
            int, if num_lights is None, we will sample the number of lights from [min_num_lights, max_num_lights]
        min_total_light_energy / max_total_light_energy:
            float, this specifies the TOTAL energy we will distrbute across all lights.
            We will sample the energey from [min_light_energy, max_light_energy]
        min_random_radius / max_random_radius:
            float, the lights will be placed at a sampled raidus from [min_random_radius, max_random_radius]
        random_lookat_r:
            float, we will randomly look at a position on a sphere with radius sampled from [0, random_lookat_r]
        light_type_settings:
            dict of dict, it contains specific attributes to be used in Blender.
            Please check Blender's document for the attributes we can set for each light.
            It is used in load_light() function in blender_rendering/blender_utils.py
        up_dir:
            str, the direction for the "UP" direction.  can be "y" or "z"

    Returns:
        (num_lights,) list of light_dict to be sent to `render_with_blender`.
    """
    rng_np = np.random.RandomState(seed)

    if num_lights is None:
        assert min_num_lights is not None
        assert max_num_lights is not None
        num_lights: int = rng_np.randint(low=min_num_lights, high=max_num_lights + 1, size=None)

    # determine energy of each light
    assert min_total_light_energy is not None
    assert max_total_light_energy is not None
    total_light_energy = (
        float(rng_np.rand()) * (max_total_light_energy - min_total_light_energy) + min_total_light_energy
    )
    total_light_energy_dist = rng_np.dirichlet(np.ones(num_lights), size=1)[0]  # (num_lights,) sum to 1

    # create each light
    light_dicts = []
    for il in range(num_lights):
        _H_c2w = rigid_motion.generate_random_camera_poses_lookat(
            n=1,
            pinhole_min_r=min_random_radius,
            pinhole_max_r=max_random_radius,
            lookat_r=random_lookat_r,
            up_method=up_dir,
            invert_y=True,
        )  # (1, 4, 4)
        assert (_H_c2w.ndim == 3) and (_H_c2w.shape == (1, 4, 4)), f"{_H_c2w.shape=}"
        _H_c2w = _H_c2w[0]

        tmp_energy = total_light_energy * total_light_energy_dist[il]
        tmp_light_type = str(rng_np.choice(light_types, 1, replace=False)[0])

        tmp_extra_setups = {}
        if (light_type_settings is not None) and (tmp_light_type in light_type_settings):
            light_setting = light_type_settings[tmp_light_type]
            for tmp_k, tmp_k_dict in light_setting.items():
                tmp_min = tmp_k_dict["min"]
                tmp_max = tmp_k_dict["max"]
                tmp_v = float(rng_np.uniform()) * (tmp_max - tmp_min) + tmp_min
                tmp_extra_setups[tmp_k] = tmp_v

        mdict = dict(
            name=f"light {il}",
            light_type=tmp_light_type,
            H_c2w=_H_c2w.tolist(),
            energy=tmp_energy,
            use_shadow=False,
            specular_factor=1.0,
            total_energy=total_light_energy,
            extra_setups=tmp_extra_setups,
        )

        light_dicts.append(mdict)
    return light_dicts


def linear_to_srgb(c: torch.Tensor):
    """
    Converts from linear RGB to sRGB. Expects input range [0.0, 1.0] (everything outside the range is clipped)

    Args:
        c:
            (*,) [0, 1], (everything outside the range is clipped)

    Returns:
        (*,) [0, 1]
    """
    return torch.clamp(torch.where(c < 0.0031308, c * 12.92, 1.055 * torch.pow(c, 1.0 / 2.4) - 0.055), min=0, max=1)


def render_multi_mesh_sample(
    out_dir: str,
    mesh_filenames: T.List[str],
    min_mesh_scale: float = 0.8,
    max_mesh_scale: float = 1.2,
    # light
    light_mode: str = "diffuse",
    # circular camera (no anti-aliasing)
    num_regular_images: int = 150,
    fov: float = 40.0,  # degree
    circular_radius: float = 3.5,  # meter
    regular_width_px: int = 1036,
    regular_height_px: int = 1036,
    # random camera (with anti-aliasing)
    num_random_images: int = 100,
    min_fov: float = 20.0,  # degree
    max_fov: float = 40.0,  # degree
    min_random_radius: float = 1.5,
    max_random_radius: float = 3.5,
    random_lookat_r: float = 0.25,
    random_width_px: int = 1036,
    random_height_px: int = 1036,
    # genai camera (with anti-aliasing),
    render_gen: bool = True,
    num_cond_images: int = 8,
    cond_width_px: int = 1036,
    cond_height_px: int = 1036,
    num_gen_eval_images: int = 8,
    gen_eval_width_px: int = 1036,
    gen_eval_height_px: int = 1036,
    gen_eval_radius: float = 4,
    gen_eval_fov: float = 40,
    # misc
    read_result_to_rgbd: bool = False,
    overwrite: bool = True,
    blender_device: str = "CPU",  # "GPU"
    max_memory_gb: float = None,
    timeout: float = None,
    seed: int = None,
    blender_exe: str = None,
    rotate_objs: bool = True,
    adjust_exposure: bool = False,
    under_exposure_threshold: float = 5.0 / 255.0,
    over_exposure_threshold: float = 250.0 / 255.0,
    exposure_cam_type: str = "regular",
    exposure_cam_count: int = 8,
):
    """
    Given a few meshes
    1. randomly scale, rotate, and place them in the scene
    2. normalize the scene to be bounded by [-1, 1]^3
    3. render rgbd images from a) sphere/regular cameras and b) random cameras and optional c) image conditioning cameras
    4. save the raw blender rendering results

    Args:
        out_dir:
            where the blender rendering results will be saved
        mesh_filenames:
            (num_meshes,) filenames of the meshes to be added to the scene

        # light
        light_mode:
            "diffuse" or "random"

        # regular camera
        num_regular_images:
            int, number of views to place uniformly on sphere of radius `circular_radius`
        fov:
            float, fov in degree of the regular cameras
        circular_radius:
            float, radius of the sphere

        # random camera
        num_random_images:
            int, number of views to randomly place uniformly in a sphere shell
        min_fov:
            float, min fov
        max_fov:
            float, max fov
        min_random_radius:
            float, inner radius of the sphere shell
        max_random_radius:
            float, outer radius of the sphere shell
        random_lookat_r:
            float, the random cameras will look at a random point in the ball of this radius

        # gen ai camera
        render_gen:
            bool, whether to render genai cameras
        num_cond_images:
            int, number of conditioning views to render. selects random radii for a set of uniformly-distributed cameras
        num_gen_eval_images:
            int, number of evaluation images to render. stratified samples views along the horizon, with a small random pitch.
        gen_eval_radius:
            float, how far evaluation views should be from the origin.
        gen_eval_fov:
            float, fov of evaluation views.

        # misc
        read_result_to_rgbd:
            whether to read the rendered results to memory (as rgbd object)
        overwrite:
            whether to delete `out_dir` if not empty
        blender_device:
            'CPU', 'GPU':  blender rendering device

        max_memory_gb:
            float, in GB, limit the amount of cpu memoery blender can use. None: inf

        timeout:
            float, in secs.  Allows blender to run `timeout` secs. None: inf
            Otherwise it will raise TimeoutExpired

        rotate_objs:
            bool, whether to randomly rotate objects

        # exposure handling
        adjust_exposure:
            bool, whether to adjust exposure of captured images to avoid any super dark/super bright images.
        under_exposure_threshold:
            float, lower bound of "well-exposed" pixels
        over_exposure_threshold:
            float, upper bound of "well-exposed" pixels
        exposure_cam_type:
            str, camera type to use to calculate new exposure. "random" or "regular", "gen_eval" or "cond" are also valid if render_gen is True.
        exposure_cam_count:
            int, number of views to use to calculate new exposure.

    Returns:
        config_dict:
            dict, the config dict to blender_utils_v3.py
        out_dir:
            str
        rgbd:
            (num_frames, num_view, h, w) or None

    Notes:
        After imported into blender, blender converts all glb files (and other file formats)
        as z-up, x-right, y-far. We thus assumes the objects are actually z-up after imported.
    """

    if seed is None:
        rng = np.random
    else:
        rng = np.random.RandomState(seed)

    # Create mesh dicts
    # to randomly place the meshes, we rely on blender_utils_v3's
    # normalization and H_c2w and normalization.
    mesh_dicts = []
    for i in range(len(mesh_filenames)):
        if rotate_objs:
            # determine random center within [-1, 1] -- this is after mesh is normalized to [-1, 1]
            _t_w = rng.rand(3) * 2 - 1  # (3,)
            # determine another random point to look at
            # we try to maintain the object's up by only selecting points within [+z_min, 1]
            _l_w = rigid_motion.get_random_direction_on_sphere(
                n=1,
                z_max=1,
                z_min=0.25,
                rng=rng,
                method="random",
            )[0]  # (3,)
            _R_c2w = rigid_motion.get_min_R(
                v1=np.array([0, 0, 1.0]).astype(np.float32),
                v2=_l_w.astype(np.float32),
            )  # (3, 3)
            _H_c2w = np.eye(4)  # (4, 4)
            _H_c2w[:3, :3] = _R_c2w
            _H_c2w[:3, 3] = _t_w

            _scale = rng.rand(1).item() * (max_mesh_scale - min_mesh_scale) + min_mesh_scale  # float
            _scale = np.ones(3) * _scale  # (3,)
        else:
            _H_c2w = np.eye(4)
            _scale = np.ones(3)

        mesh_dict = dict(
            name=os.path.splitext(os.path.basename(mesh_filenames[i]))[0],
            filename=os.path.abspath(mesh_filenames[i]),
            normalize_first=True,
            H_c2w=_H_c2w,  # (4, 4)
            scale=_scale,  # (3,)
            post_normalization=False,  # not a bug, we normalize the whole scene using `normalize_entire_scene`
        )
        mesh_dicts.append(mesh_dict)

    # Create light dicts
    # support diffuse light for now, we can make it more complex later
    if light_mode == "diffuse":
        light_dict = dict(
            name="diffuse light",
            light_type=light_mode,
            color=[1.0, 1.0, 1.0, 1.0],
            strength=1.0,
        )
        light_dicts = [light_dict]
    elif light_mode == "random":
        circular_radius_for_light = 3.5
        light_dicts = construct_light_dicts_for_random_lighting(
            light_types=["AREA", "POINT", "SPOT"],
            num_lights=None,
            min_num_lights=3,
            max_num_lights=10,
            min_total_light_energy=200,
            max_total_light_energy=2000,
            min_random_radius=circular_radius_for_light * 2,  # 8.0 for TRELLIS cam, 7.0 for OUR cam
            max_random_radius=circular_radius_for_light * 3,  # 12.0 for TRELLIS cam, 10.5 for OUR cam
            random_lookat_r=0.25,
            light_type_settings={
                "AREA": {"size": {"min": 5.0, "max": 20.0}},
                "SPOT": {"shadow_soft_size": {"min": 0.0, "max": 5.0}},
            },
            seed=rng.randint(42, 4294967295),
        )
    else:
        raise NotImplementedError(f"{light_mode=}")

    # Create camera
    # sphere camera
    num_frames = 1  # static for now
    H_c2w = rigid_motion.generate_uniform_camera_poses_with_golden_spiral(
        n=num_regular_images,
        r=circular_radius,
        up_method="z",  # most objs in objaverse are z-up
        invert_y=True,
    )  # (q, 4, 4)  tensor
    intrinsic = (
        torch.from_numpy(
            render.derive_camera_intrinsics(
                width_px=regular_width_px,
                height_px=regular_height_px,
                fov=fov,
            )
        )
        .float()
        .expand(H_c2w.size(0), 3, 3)
    )  # (q, 3, 3)
    camera_sphere = structures.Camera(
        H_c2w=H_c2w.float().expand(num_frames, H_c2w.shape[0], 4, 4),  # (num_frames, q, 4, 4)
        intrinsic=intrinsic.expand(num_frames, H_c2w.shape[0], 3, 3),  # (num_frames, q, 3, 3)
        width_px=regular_width_px,
        height_px=regular_height_px,
    )

    # random camera (which can be different every frame_idx)
    H_c2w = rigid_motion.generate_random_camera_poses_lookat(
        n=num_frames * num_random_images,
        pinhole_min_r=min_random_radius,
        pinhole_max_r=max_random_radius,
        lookat_r=random_lookat_r,
        up_method="z",  # most objs in objaverse are z-up
        invert_y=True,
    ).reshape(num_frames, num_random_images, 4, 4)  # (num_frames, q, 4, 4)

    random_fovs = rng.random(num_frames * num_random_images) * (max_fov - min_fov) + min_fov  # (num_frames*q,) degree
    intrinsic = (
        torch.from_numpy(
            render.derive_camera_intrinsics(
                width_px=random_width_px,
                height_px=random_height_px,
                fov=random_fovs,  # (num_frames*q,)
            )
        )
        .reshape(num_frames, num_random_images, 3, 3)
        .float()
    )  # (num_frames, q, 3, 3)

    camera_random = structures.Camera(
        H_c2w=H_c2w,  # (num_frames, q, 4, 4)
        intrinsic=intrinsic,  # (num_frames, q, 3, 3)
        width_px=random_width_px,
        height_px=random_height_px,
    )

    if render_gen:
        # conditioning camera
        # create camera with coupled fov and distance to origin
        r_min = 1.75
        r_max = 5.0
        r_buffer = 0.5
        half_width = 1.25

        rs = rng.random(num_cond_images) * (r_max - r_min) + r_min  # (num_cond_views,)
        thetas = np.arctan2(half_width, rs - r_buffer) * 180 / np.pi  # degree  # (num_cond_views,)
        fovs = 2 * thetas  # degree  # (num_cond_views,)

        # determine the camera elevation and pinhole location
        yaws = []
        pitchs = []
        offset = (rng.random(), rng.random())
        for i in range(num_cond_images):
            y, p = rigid_motion.sphere_hammersley_sequence(i, num_cond_images, offset)
            # sphere_hammersley_sequence(i, num_cond_images, offset)
            yaws.append(y)
            pitchs.append(p)

        yaws = np.array(yaws)  # (num_cond_views,)
        pitchs = np.array(pitchs)  # (num_cond_views,)

        pinhole_x = rs * np.cos(pitchs) * np.cos(yaws)  # (num_cond_views,)
        pinhole_y = rs * np.cos(pitchs) * np.sin(yaws)  # (num_cond_views,)
        pinhole_z = rs * np.sin(pitchs)  # (num_cond_views,)

        pinhole_xyz_w = np.stack([pinhole_x, pinhole_y, pinhole_z], axis=-1)  # (num_cond_views, 3xyz_w)
        pinhole_xyz_w = torch.from_numpy(pinhole_xyz_w).float()  # (num_cond_views, 3xyz_w)

        # the random camera stays at the same location during the sequence
        camera_cond = structures.Camera(
            H_c2w=rigid_motion.get_H_c2w_lookat(
                pinhole_location_w=pinhole_xyz_w,  # (num_cond_views, 3xyz_w)
                look_at_w=[0.0, 0.0, 0.0],
                up_w=[0.0, 0.0, 1.0],
                invert_y=True,
            )
            .reshape(1, num_cond_images, 4, 4)
            .expand(num_frames, num_cond_images, 4, 4),
            intrinsic=torch.from_numpy(
                render.derive_camera_intrinsics(
                    width_px=cond_width_px,
                    height_px=cond_height_px,
                    fov=fovs,
                ),
            )
            .reshape(1, num_cond_images, 3, 3)
            .expand(num_frames, num_cond_images, 3, 3),
            width_px=cond_width_px,
            height_px=cond_height_px,
        )  # (num_frames, num_cond_views)

        # gen eval cameras
        # Note: this is different from the eval camera we used in the paper (which is constructed by `generate_our_eval_cameras_for_img_cond_ldm`)
        gen_eval_yaws_reg = np.linspace(0, 360, num_gen_eval_images + 1)[:-1]
        gen_eval_yaws_step = gen_eval_yaws_reg[1] - gen_eval_yaws_reg[0]
        gen_eval_yaws = (
            gen_eval_yaws_reg + rng.random(num_gen_eval_images) * gen_eval_yaws_step
        ).tolist()  # np.linspace(0, 360, num_gen_eval_images + 1)[:-1].tolist()
        gen_eval_pitches = rng.random(num_gen_eval_images) * 30  # [30 for _ in gen_eval_yaws]

        gen_eval_yaws = torch.tensor(gen_eval_yaws, dtype=torch.float) / 180 * torch.pi  # (n,)
        gen_eval_radii = torch.tensor([gen_eval_radius for _ in gen_eval_yaws], dtype=torch.float)  # (n,)
        gen_eval_fovs = torch.tensor([gen_eval_fov for _ in gen_eval_yaws], dtype=torch.float)  # (n,)
        gen_eval_pitches = torch.tensor(gen_eval_pitches, dtype=torch.float) / 180 * np.pi  # (n,)

        gen_eval_intrinsic = render.derive_camera_intrinsics(
            width_px=gen_eval_width_px,
            height_px=gen_eval_height_px,
            fov=gen_eval_fovs,  # degree
        ).float()  # (n, 3, 3)

        # https://github.com/microsoft/TRELLIS/blob/f17fdf12d8f17a6a09225f01756d141285dc848f/dataset_toolkits/blender_script/render.py#L459-L463
        tmp_x = torch.cos(gen_eval_pitches) * torch.cos(gen_eval_yaws)
        tmp_y = torch.cos(gen_eval_pitches) * torch.sin(gen_eval_yaws)
        tmp_z = torch.sin(gen_eval_pitches)

        tmp_pinhole_location_w = gen_eval_radii[:, None] * torch.stack((tmp_x, tmp_y, tmp_z), dim=-1)  # (n, 3)
        assert tmp_pinhole_location_w.shape[1] == 3, f"{tmp_pinhole_location_w.shape=}"

        # get H_c2w
        gen_eval_H_c2w = rigid_motion.sphere_camera_poses_sampling_postprocessing(
            pinhole_location_w=tmp_pinhole_location_w,
            up_method="z",
            invert_y=True,
        )  # (n, 4, 4)

        assert gen_eval_H_c2w.shape == (num_gen_eval_images, 4, 4), f"{gen_eval_H_c2w.shape=}"
        assert gen_eval_intrinsic.shape == (num_gen_eval_images, 3, 3), f"{gen_eval_intrinsic.shape=}"

        camera_gen_eval = structures.Camera(
            H_c2w=gen_eval_H_c2w.unsqueeze(0).expand(num_frames, num_gen_eval_images, 4, 4),  # (1, q, 4, 4)
            intrinsic=gen_eval_intrinsic.unsqueeze(0).expand(num_frames, num_gen_eval_images, 3, 3),  # (1, q, 3, 3)
            width_px=gen_eval_width_px,
            height_px=gen_eval_height_px,
        )

    # construct camera_dicts, list of list, (num_frames, num_total_views)
    all_camera_dicts = []  # (num_frames,) regular and sphere is concatenated along q
    cam_name_start_idx_dict = dict(
        regular=0,
        random=num_regular_images,
    )  # name -> qidx
    cam_name_num_frame_dict = dict(
        regular=num_regular_images,
        random=num_random_images,
    )  # name -> num_views

    camera_list = [
        ["regular", camera_sphere],
        ["random", camera_random],
    ]

    if render_gen:
        cam_name_start_idx_dict["cond"] = num_regular_images + num_random_images
        cam_name_num_frame_dict["cond"] = num_cond_images
        camera_list.append(["cond", camera_cond])

        cam_name_start_idx_dict["gen_eval"] = num_regular_images + num_random_images + num_cond_images
        cam_name_num_frame_dict["gen_eval"] = num_gen_eval_images
        camera_list.append(["gen_eval", camera_gen_eval])

    for frame_idx in range(num_frames):
        view_camera_dicts = []

        current_idx = 0
        for name, camera in camera_list:
            assert current_idx == cam_name_start_idx_dict[name]
            for qidx in range(camera.H_c2w.size(1)):
                # render_with_blender takes camera in plib coordinate system
                mdict = dict(
                    H_c2w=camera.H_c2w[frame_idx, qidx],
                    intrinsic=camera.intrinsic[frame_idx, qidx],
                    width_px=camera.width_px,
                    height_px=camera.height_px,
                )
                if name == "random" or name == "cond" or name == "gen_eval":
                    mdict["filter_width"] = 1.0  # with antialiasing
                    mdict["use_denoising"] = True  # with antialiasing can cautiously use it
                elif name == "regular":
                    mdict["filter_width"] = 0.01  # no antialiasing
                    mdict["use_denoising"] = False  # no antialiasing
                else:
                    raise NotImplementedError
                view_camera_dicts.append(mdict)  # (num_regular_views + num_random_views,)

            current_idx += camera.H_c2w.size(1)

        all_camera_dicts.append(view_camera_dicts)

    # cycles settings
    cycles_settings = dict()  # use default

    # view_layer_settings
    view_layer_settings = dict()  # use default

    out_dict = render_with_blender(
        out_dir=out_dir,
        mesh_dicts=mesh_dicts,  # (num_meshes,)
        light_dicts=light_dicts,  # (num_lights,)
        cam_dicts=all_camera_dicts,  # (num_frames, num_views)
        cycles_settings=cycles_settings,
        view_layer_settings=view_layer_settings,
        # animation settings (assume static for now, can support later)
        frame_start=0,
        frame_skip=1,
        animation_number=0,
        normalize_bbox_mode="render_clip",
        # advanced camera setting:
        adjust_camera_pose_per_frame=False,
        normalize_entire_scene=True,
        # misc
        read_result_to_rgbd=False,
        blender_version="4.2.0",
        blender_device=blender_device.upper(),  # "CPU", "GPU"
        blender_exe=blender_exe,
        overwrite=overwrite,
        max_memory_gb=max_memory_gb,
        timeout=timeout,
        debug=False,
    )

    output_rgbd = {}
    if read_result_to_rgbd:
        # First, load all renders in HDR
        for cam_name, _ in camera_list:
            rgbd_current = blender_plib_utils.read_blender_results_to_rgbd(
                result_dir=out_dir,
                from_idx=cam_name_start_idx_dict[cam_name],
                to_idx=cam_name_start_idx_dict[cam_name] + cam_name_num_frame_dict[cam_name],
                from_bidx=0,
                to_bidx=None,
                use_srgb=False,  # set to False, so rgbd_current.rgb is hdr
                flag_save_space=False,
                dynamic=None,  # auto-detect
                th_alpha=0.5,
                min_depth=0.0,
                max_depth=1.0e4,
                fps=24,
            )
            output_rgbd[f"rgbd_{cam_name}"] = rgbd_current

        if adjust_exposure:
            test_inds = torch.randperm(output_rgbd[f"rgbd_{exposure_cam_type}"].rgb.shape[1])[:exposure_cam_count]

            test_rgb = output_rgbd[f"rgbd_{exposure_cam_type}"].rgb[:, test_inds]
            test_hit_map = output_rgbd[f"rgbd_{exposure_cam_type}"].hit_map[:, test_inds]
            all_valid_rgb = test_rgb[test_hit_map]

            # Then, compute optimal exposure for each one
            ABLATE_FILM_EXPOSURE_LIST = sorted(
                list(set(np.linspace(0.01, 5.0, 20).tolist() + [1.0] + np.power(2, np.arange(3, 6.5, 0.5)).tolist()))
            )

            best_exposed_exposure = None
            best_exposure_count = -np.inf
            for exposure in ABLATE_FILM_EXPOSURE_LIST:
                scaled_rgb = linear_to_srgb(all_valid_rgb * exposure)
                gray = torch.sum(
                    scaled_rgb * torch.tensor([0.299, 0.587, 0.114], device=scaled_rgb.device), dim=-1
                )  # (n,)
                well_exposed_mask = (gray >= under_exposure_threshold) & (gray <= over_exposure_threshold)
                well_exposed_count = well_exposed_mask.sum()
                if well_exposed_count > best_exposure_count:
                    best_exposed_exposure = exposure
                    best_exposure_count = well_exposed_count
        else:
            best_exposed_exposure = 1.0

        for cam_name, _ in camera_list:
            scaled_rgb = output_rgbd[f"rgbd_{cam_name}"].rgb * best_exposed_exposure
            # save srgb to rgbd.rgb
            output_rgbd[f"rgbd_{cam_name}"].rgb = linear_to_srgb(scaled_rgb)
            # save hdr to other maps
            output_rgbd[f"rgbd_{cam_name}"].other_maps["rgb_hdr"] = scaled_rgb
            del output_rgbd[f"rgbd_{cam_name}"].other_maps["rgb_ldr"]

        out_dict["config_dict"]["exposure_scale"] = best_exposed_exposure

    else:
        for cam_name, _ in camera_list:
            output_rgbd[f"rgbd_{cam_name}"] = None

    return dict(config_dict=out_dict["config_dict"], out_dir=out_dict["out_dir"], **output_rgbd)
