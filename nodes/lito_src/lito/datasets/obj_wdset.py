#
# Copyright (C) 2025 Apple Inc. All rights reserved.
#
# The file implememts webdataset loader of prechunked tars.

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
sys.path.append(str(REPO_ROOT))

from functools import partial
import json
import os
import pathlib
import re
import time
from timeit import default_timer as timer
import traceback
import typing as T

import numpy as np
import webdataset as wds  # noqa

import torch
from torch.utils.data import IterableDataset

import apple_fsspec as af

from lito.datasets import base
from plibs import byte_dict_utils, data_utils, linalg_utils, o3d_utils, structures, utils, wds_utils


@torch.no_grad()
def expand_occ_grid_for_mesh(
    *,
    occ_grid: torch.Tensor,
    num_to_add: int,
    occ_expand_type: str,
    dtype: torch.dtype,
    device: torch.device,
):
    """
    This function expands the given occupancy grid with one of the following option:
    - dilation
    - add some occupancy with the total number equals to num_to_add

    Args:
        occ_grid:
            (b, 1, res_z, res_y, res_x), bool
        num_to_add:
            number of new voxels to be added
        occ_expand_type:
            - dilation
            - total
        dtype:
            torch.Tensor's dtype to be used when expanding
        device:
            torch.device to be operated on
    """
    if occ_expand_type == "dilate":
        kernel_size = 2 * num_to_add + 1
        expanded_occ_grid = torch.nn.functional.conv3d(
            input=occ_grid.to(dtype=dtype, device=device),  # (b, 1, res_z, res_y, res_x)
            weight=torch.ones(
                1,
                1,
                kernel_size,
                kernel_size,
                kernel_size,
                dtype=dtype,
                device=device,
            ),
            padding="same",
        )  # (b, 1, res_z, res_y, res_x) float
        expanded_occ_grid = expanded_occ_grid > 0.5  # (b, 1, res_z, res_y, res_x) bool
    elif occ_expand_type == "total":
        expanded_occ_grid = torch.nn.functional.conv3d(
            input=occ_grid.to(dtype=dtype, device=device),  # (b, 1, res_z, res_y, res_x)
            weight=torch.ones(1, 1, 3, 3, 3, dtype=dtype, device=device),
            padding="same",
        )  # (b, 1, res_z, res_y, res_x) float
        _r = torch.rand_like(expanded_occ_grid)  # (b, 1, res_z, res_y, res_x)
        new_occ_grid = torch.logical_xor(
            occ_grid,
            expanded_occ_grid > 0.5,
        )  # (b, 1, res_z, res_y, res_x)
        num_new = new_occ_grid.reshape(new_occ_grid.size(0), -1).sum(dim=-1)  # (b,)
        th = num_to_add / num_new
        _r = _r < th.reshape(-1, 1, 1, 1, 1)  # (b, 1, res_z, res_y, res_x)  bool
        expanded_occ_grid = torch.logical_or(
            occ_grid,
            torch.logical_and(_r, new_occ_grid),
        )
        del new_occ_grid
        del _r
    else:
        raise ValueError(f"{occ_expand_type=}")

    return expanded_occ_grid


def convert_camera_wrt_cond_img_space(
    *,
    dtype: torch.dtype,
    device: torch.device,
    H_c2w: T.Optional[torch.Tensor],
    H_c2w_for_cond_img: torch.Tensor,
    img_cond_ldm_cam_transformation: str = "z_up",
):
    """
    When rendering with given H_c2w for the generated Gaussians, we should transform those H_c2w to the world space
    defined by the H_c2w corresponding to the conditioning image.

    Args:
        H_c2w:
            (b, 4, 4), camera-to-world transformation
        H_c2w_for_cond_im:
            (4, 4), ground-truth c2w for the conditioning image.
        img_cond_ldm_cam_transformation:
            str, the strategy we used for transforming the conditioning image's camera pose.
    """
    if H_c2w is not None:
        bs = H_c2w.shape[0]
        assert H_c2w.shape == (bs, 4, 4), f"{H_c2w.shape=}, {bs=}"

    assert H_c2w_for_cond_img.shape == (4, 4), f"{H_c2w_for_cond_img.shape=}"

    if img_cond_ldm_cam_transformation == "y_up":
        # rotate the world coordinate so that the conditioning camera's camera pose
        # is at diag([1, -1, -1]), ie, the world is y-up
        R_w2c = H_c2w_for_cond_img[:3, :3].t()  # (3, 3)
        R_c2b = torch.tensor(
            [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ],
            dtype=dtype,
            device=device,
        )  # (3, 3)
        R_w2b = R_c2b @ R_w2c
        H_w2b = torch.eye(4, device=device)  # (4, 4)
        H_w2b[:3, :3] = R_w2b
    elif img_cond_ldm_cam_transformation == "z_up":
        # rotate the world coordinate so that the conditioning camera's camera pose
        # is below and the pinhole is at (r, 0, 0), ie, the world is z-up
        R_w2c = H_c2w_for_cond_img[:3, :3].t()  # (3, 3)
        R_c2b = torch.tensor(
            [
                [0, 0, -1],
                [1, 0, 0],
                [0, -1, 0],
            ],
            dtype=dtype,
            device=device,
        )  # (3, 3)
        R_w2b = R_c2b @ R_w2c
        H_w2b = torch.eye(4, device=device)  # (4, 4)
        H_w2b[:3, :3] = R_w2b
    elif img_cond_ldm_cam_transformation == "identity_rotation":
        # rotate the world coordinate so that the conditioning camera's oriention aligns with the world.
        # The pinhole is at (0, 0, -r)
        R_w2c = H_c2w_for_cond_img[:3, :3].t()  # (3, 3)
        R_c2b = torch.eye(3)  # (3, 3)
        R_w2b = R_c2b @ R_w2c
        H_w2b = torch.eye(4, device=device)  # (4, 4)
        H_w2b[:3, :3] = R_w2b
    else:
        raise ValueError(f"{img_cond_ldm_cam_transformation=}")

    if H_c2w is not None:
        H_c2w = H_w2b @ H_c2w

    ret_dict = dict(
        R_w2b=R_w2b,
        H_w2b=H_w2b,
        H_c2w=H_c2w,
    )
    return ret_dict


def extract_sample_from_byte_dict(
    byte_dict: T.Dict[str, T.Any],
    load_point: bool,
    load_rgbd_random: bool,
    load_rgbd_sphere: bool,
    load_rgbd_cond: bool,
    num_random_views: int,
    num_sphere_views: int,
    num_cond_views: int,
    max_num_voxels: int,
    printout: bool = False,
    forbidden_list: T.List = tuple(),
    min_num_points: int = -1,
    min_num_voxels: int = -1,
    target_width_px: int = None,
    target_height_px: int = None,
):
    """
    Args:
        byte_dict:
            sample_index.json
                'ori_uid': str
                'new_uid': str
                'num_points': int
                "num_rgbd_random_views": int
                "num_rgbd_sphere_views": int

                'point_info':
                    'xyz_w': 'point.xyz_w.npy'  [-1, 1]
                    'rgb': 'point.rgb.npy'   [0, 1]
                    "ray_origin_w": "point.ray_origin_w.npy",
                    "ray_direction_w": "point.ray_direction_w.npy"
                    "pinhole_idx": "point.pinhole_idx.npy",  (n, 1) uint8, index to "pinhole_w_for_points"
                    "pinhole_w_for_points": "pinhole_w_for_points.npy"  # (q, 3xyz_w)
                "rgbd_random_info":
                    "index": "rgbd_random.index.json",
                    "name": "rgbd_random",
                    "b": 1,
                    "q": 16,
                    "h": 518,
                    "w": 518
                "rgbd_sphere_info":
                    "index": "rgbd_sphere.index.json",
                    "name": "rgbd_sphere",
                    "b": 1,
                    "q": 16,
                    "h": 518,
                    "w": 518
            # rgbd random
            rgbd_random.index.json
            rgbd_random.camera.npz
            rgbd_random.{ib}-rgb_{iq}.png
            rgbd_random.{ib}-alpha_{iq}.png
            rgbd_random.{ib}-depth_{iq}.png
            rgbd_random.{ib}-normal_w_{iq}.png
            rgbd_random.{ib}-hit_map_{iq}.png
            # rgbd sphere
            rgbd_sphere.index.json
            rgbd_sphere.camera.npz
            rgbd_sphere.{ib}-rgb_{iq}.png
            rgbd_sphere.{ib}-alpha_{iq}.png
            rgbd_sphere.{ib}-depth_{iq}.png
            rgbd_sphere.{ib}-normal_w_{iq}.png
            rgbd_sphere.{ib}-hit_map_{iq}.png

    Returns:
        A dictionary containing extracted sample data with the following keys:
            uid:
                Unique identifier from byte_dict["__key__"]
            point_xyz_w:
                (n, 3) the point xyz_w
            point_rgb:
                (n, 3) or -1, the point rgb [0, 1]
            point_alpha:
                (n, 1) or -1, the point alpha [0, 1]
            point_ray_o:
                (n, xyz_w)
            point_ray_d:
                (n, 3xyz_w)

            rgbd_random (structures.RGBDImage, optional):
                (b=1, q, h, w) Multi-view RGBD data (if load_rgbd_random=True)
            rgbd_sphere (structures.RGBDImage, optional):
                (b=1, q, h, w) Fixed-view RGBD data (if load_rgbd_sphere=True)
            rgbd_cond (structures.RGBDImage, optional):
                (b=1, q, h, w) Fixed-view RGBD data (if load_rgbd_cond=True)
    """

    stime = timer()

    url = byte_dict.get("__url__", "")
    uid = byte_dict["__key__"]  # "./{uid}_{xxx}_{ooo}"

    if not uid.startswith("./"):
        uid = "./" + uid

    assert re.split(r"[/_]", uid)[1] not in forbidden_list, f"{uid=}, {forbidden_list=}"

    assert "sample_index.json" in byte_dict, f"sample_index.json not found in {uid}, {byte_dict['__url__']}"
    index_dict = data_utils.load_file_from_byte_dict(
        byte_dict=byte_dict,
        filename="sample_index.json",
        start_path=None,
    )

    out_dict = dict(
        uid=uid,
        shard_url=byte_dict.get("__url__", ""),
    )

    # load point
    if load_point:
        assert index_dict.get("point_info", None) is not None
        point_info = index_dict["point_info"]

        for key in [
            "xyz_w",
            "rgb",
            "normal_w",
            "ray_origin_w",
            "ray_direction_w",
            "alpha",
        ]:
            if key not in point_info:
                continue

            arr = torch.from_numpy(
                byte_dict_utils.load_file_from_byte_dict(
                    byte_dict=byte_dict,
                    filename=point_info[key],
                )
            )  # (n, d)
            if key == "normal_w":
                assert arr.dtype != torch.uint8, "not implemented yet"

            point_key = f"point_{key}"
            if arr.dtype == torch.uint8:
                out_dict[point_key] = arr.float() / 255  # [0, 1]
            else:
                out_dict[point_key] = arr.float()

        # check hit_map
        if min_num_points > 0:
            point_xyz_w = out_dict["point_xyz_w"]  # (n, 3)
            assert point_xyz_w.size(0) >= min_num_points, (
                f"{uid}, {out_dict['shard_url']}, only has {point_xyz_w.size(0)} points, skipped"
            )
            num_test_points = 16384
            _xyz_w = point_xyz_w[:num_test_points]  # (m, 3)
            _unique_xyz_w = torch.unique(_xyz_w, dim=0)  # (mm, 3)
            assert _unique_xyz_w.size(0) == _xyz_w.size(0), (
                f"{uid}, {out_dict['shard_url']}, {_unique_xyz_w.size(0)} points != {num_test_points}"
            )

        if "pinhole_idx" in point_info and "pinhole_w_for_points" in point_info:
            point_pinhole_qidx = torch.from_numpy(
                byte_dict_utils.load_file_from_byte_dict(
                    byte_dict=byte_dict,
                    filename=point_info["pinhole_idx"],
                )
            ).squeeze(-1)  # (n,) uint8

            pinhole_w_for_points = torch.from_numpy(
                byte_dict_utils.load_file_from_byte_dict(
                    byte_dict=byte_dict,
                    filename=point_info["pinhole_w_for_points"],
                )
            ).float()  # (q, 3xyz_w)

            if point_pinhole_qidx.dtype == torch.uint8:
                assert pinhole_w_for_points.size(0) <= 255, f"{pinhole_w_for_points.size(0)}"

            point_ray_origin_w = pinhole_w_for_points[point_pinhole_qidx.long()]  # (n, 3xyz_w)
            assert out_dict.get("point_xyz_w") is not None
            assert out_dict["point_xyz_w"].shape == point_ray_origin_w.shape, f""
            # from pinhole to point (lito convention)
            point_ray_direction_w = out_dict["point_xyz_w"] - point_ray_origin_w  # (n, 3xyz_w)
            point_ray_direction_w = torch.nn.functional.normalize(
                point_ray_direction_w,
                dim=-1,
            )  # (n, 3xyz_w)

            out_dict["point_ray_origin_direction_w"] = torch.cat(
                [
                    point_ray_origin_w,
                    point_ray_direction_w,
                ],
                dim=-1,
            )  # (n, 6)

        if (max_num_voxels is not None and max_num_voxels > 0) or (min_num_voxels is not None and min_num_voxels > 0):
            # compute number of voxels
            num_point_to_use = 100_000
            min_xyz_w = -1
            max_xyz_w = 1
            grid_size = 64
            cell_width = (max_xyz_w - min_xyz_w) / grid_size
            _xyz_w = out_dict["point_xyz_w"][:num_point_to_use]  # (nn, 3)
            occ_ijk = torch.floor((_xyz_w - min_xyz_w) / cell_width).long()  # (nn, 3ijk)
            occ_ijk = occ_ijk.clamp(min=0, max=grid_size - 1)
            occ_grid = torch.zeros(grid_size, grid_size, grid_size, dtype=torch.bool)  # (gz, gy, gx)
            occ_grid[occ_ijk[:, 2], occ_ijk[:, 1], occ_ijk[:, 0]] = True
            num_occ_cells = occ_grid.sum()
            if (max_num_voxels is not None and max_num_voxels > 0) and num_occ_cells > max_num_voxels:
                raise RuntimeError(
                    f"too many voxels {num_occ_cells}, skipped, uid: {out_dict['uid']}, url: {out_dict['shard_url']}"
                )
            if (min_num_voxels is not None and min_num_voxels > 0) and num_occ_cells < min_num_voxels:
                raise RuntimeError(
                    f"too few voxels {num_occ_cells}, skipped, uid: {out_dict['uid']}, url: {out_dict['shard_url']}"
                )

    # load rgbd_random
    if load_rgbd_random:
        multiview_info = index_dict.get("rgbd_random_info", None)
        if multiview_info is not None:
            max_q = multiview_info["q"]
            assert max_q > 0, f"{max_q=}"
            if num_random_views is None or num_random_views < 0:
                qidxs = None
            else:
                # load num_random_views
                qidxs = np.random.permutation(max_q)[:num_random_views]
                qidxs = qidxs.tolist()

            rgbd_random = structures.RGBDImage.load_from_byte_dict(
                byte_dict=byte_dict,
                prefix=multiview_info["name"],
                qidxs=qidxs,
            )

            _b, _q, _h, _w = rgbd_random.shape
            if target_height_px is None:
                target_height_px = _h
            if target_width_px is None:
                target_width_px = _w
            if target_height_px != _h or target_width_px != _w:
                rgbd_random = rgbd_random.resize(
                    new_width_px=target_width_px,
                    new_height_px=target_height_px,
                    make_hit_map_bool=True,
                    interpolation_mode="bilinear",
                )

            out_dict["rgbd_random"] = rgbd_random  # (b=1, q_random, h, w)

    # load rgbd_sphere
    if load_rgbd_sphere:
        fixedview_info = index_dict.get("rgbd_sphere_info", None)
        if fixedview_info is not None:
            max_q = fixedview_info["q"]
            assert max_q > 0, f"{max_q=}"
            if num_sphere_views is None or num_sphere_views < 0:
                qidxs = None
            else:
                # load num_random_views
                qidxs = np.random.permutation(max_q)[:num_sphere_views]
                qidxs = qidxs.tolist()

            rgbd_sphere = structures.RGBDImage.load_from_byte_dict(
                byte_dict=byte_dict,
                prefix=fixedview_info["name"],
                qidxs=qidxs,
            )

            _b, _q, _h, _w = rgbd_sphere.shape
            if target_height_px is None:
                target_height_px = _h
            if target_width_px is None:
                target_width_px = _w
            if target_height_px != _h or target_width_px != _w:
                rgbd_sphere = rgbd_sphere.resize(
                    new_width_px=target_width_px,
                    new_height_px=target_height_px,
                    make_hit_map_bool=True,
                    interpolation_mode="bilinear",
                )

            out_dict["rgbd_sphere"] = rgbd_sphere  # (b=1, q_sphere, h, w)

    # load rgbd_cond
    if load_rgbd_cond:
        cond_view_info = index_dict.get("rgbd_cond_info", None)
        if cond_view_info is not None:
            max_q = cond_view_info["q"]
            assert max_q > 0, f"{max_q=}"
            if num_cond_views is None or num_cond_views < 0:
                qidxs = None
            else:
                # load num_cond_views
                qidxs = np.random.permutation(max_q)[:num_cond_views]
                qidxs = qidxs.tolist()

            rgbd_cond = structures.RGBDImage.load_from_byte_dict(
                byte_dict=byte_dict,
                prefix=cond_view_info["name"],
                qidxs=qidxs,
            )

            _b, _q, _h, _w = rgbd_cond.shape
            if target_height_px is None:
                target_height_px = _h
            if target_width_px is None:
                target_width_px = _w
            if target_height_px != _h or target_width_px != _w:
                rgbd_cond = rgbd_cond.resize(
                    new_width_px=target_width_px,
                    new_height_px=target_height_px,
                    make_hit_map_bool=True,
                    interpolation_mode="bilinear",
                )

            out_dict["rgbd_cond"] = rgbd_cond  # (b=1, q_sphere, h, w)

    ttime = timer() - stime
    out_dict["total_decoding_time"] = ttime  # secs

    return out_dict


def repeat_sample(
    data: T.Iterator,
    num_repeat: int,
):
    for d in data:
        for i in range(num_repeat):
            yield d


def s3url_to_cache_name(url: str):
    parts = url.split(" ")
    for part in parts:
        if part.startswith("s3://"):
            part = part.replace("s3://", "", 1)
            return part.replace("/", "_")
        if part.startswith("conductor://"):
            part = part.replace("conductor://", "", 1)
            return part.replace("/", "_")
    return url


def compute_occupancy_grid(
    xyz_w: torch.Tensor,  # (n, 3xyz_w)
    grid_size: int = 64,
    min_xyz_w: float = -1,
    max_xyz_w: float = 1,
):
    """
    Compute the occupancy grid of each frame from points.
    The grid is edge to edge from -1 to 1, and of resolution 64.

    Args:
        xyz_w:
             (n, 3xyz_w)

    Returns:
        occ_grid:
            (res_z=64, res_y=64, res_x=64) bool
            True if a point is in the cell
    """

    n, _3xyz = xyz_w.shape
    cell_width = (max_xyz_w - min_xyz_w) / grid_size

    # compute ijk
    ijk = torch.floor((xyz_w - min_xyz_w) / cell_width).long()  # (n, 3)
    ijk = torch.clamp(ijk, min=0, max=grid_size - 1)  # (n, 3ijk)
    offset = torch.tensor([1, grid_size, grid_size * grid_size], dtype=torch.long)  # (3ijk,)  x moves fastest

    occ_grid = torch.zeros(grid_size * grid_size * grid_size, dtype=torch.bool, device=xyz_w.device)
    idx = ijk[:, 0] + ijk[:, 1] * offset[1] + ijk[:, 2] * offset[2]  # (n,)
    idx = torch.unique(idx)
    occ_grid[idx] = 1
    occ_grid = occ_grid.reshape(grid_size, grid_size, grid_size)
    return occ_grid


class ObjIterableDataset(IterableDataset):
    """
    This is an iterable dataset implemented based on webdataset.

    My understanding of the webdataset:
    1. When resampled = True, different dataloader workers on different nodes sample different shards randomly, and
    they use different local random seeds inside webdataset (based on worker id, rank, and epoch)
    2. The global random seed for each dataloader worker is controlled by pytorch and still need to be set by us to make
    sure they are different.
    3. Webdataaset reads each tar file sequentially and put the data into the shuffle buffer.
    4. The data from shuffle buffer is read out in random order (different for each worker and rank)
    5. We can use persistent worker = True

    """

    def __init__(
        self,
        shard_blobby_urls: T.List[str],
        batch_size: int,
        mode: str,
        mode_config_dict: T.Dict[str, T.Any] = None,
        nsamples: int = -1,  # 10000 (-1, unlimited),
        num_random_views: int = -1,
        num_sphere_views: int = -1,
        num_cond_views: int = 0,
        target_height_px: int = None,
        target_width_px: int = None,
        min_num_voxels: int = -1,
        max_num_voxels: int = -1,
        printout: bool = False,
        max_num_shards_to_use: int = None,
        wds_buffer_size: int = 64,  # say 1 sample take 50MB, 200 samples take 10GB, per gpu, per worker
        wds_cache_size_tb: int = 10,  # for a100, -1: unlimited
        wds_use_cache: bool = False,
        wds_cache_dir: str = "/mnt/data/wds_cache",
        forbidden_uid_list: T.List[str] = tuple(),
        img_cond_ldm_cam_transformation: str | None = None,
        img_cond_ldm_scene_renormalize_scale_threshold: float | None = None,
        # for mesh-related occupancy processing
        for_mesh_occ_num_points_to_use: int = 65536,
        for_mesh_occ_grid_res: int = 64,
        for_mesh_occ_method: str = "torch",
        for_mesh_overfit: bool = False,
        for_mesh_min_num_occ_to_add: int = 0,
        for_mesh_max_num_occ_to_add: int = 0,
        for_mesh_occ_expand_type: T.Optional[str] = None,
        debug_num_repeat: int = 1,
        debug_byte_dict_repeat: int = 1,
        min_num_points: int = -1,
        wds_sample_life: int = 1,
    ):
        """
        Args:
            shard_blobby_urls:
                list of blobby url to the tar shards
            num_random_views:
                number of rgbd_random images to load. -1: load all available
            num_sphere_views:
                number of rgbd_sphere images to load. -1: load all available
            batch_size:
                It is recommended to batch in dataset when using iterable dataset.
                This is helpful when using streaming datasets. For example, if
                batch inside dataloader, the dataloader may be waiting for
                the dataset to download new shard and idles (causing entire problem
                to wait if it is the turn of the dataloading worker to emit batch).
            wds_use_cache:
                whether to use save shard locally and resue
            wds_cache_dir:
                if not None, will save tar files into the dir. The tar files will only be used
                when fully downloaded
                https://github.com/webdataset/webdataset/blob/main/FAQ.md?plain=1#L1168
            forbidden_uid_list:
                list of str, containing the uid that should not be used (eg, specific test case we
                want to preserve for testing).
            img_cond_ldm_cam_transformation:
                this specifies how we want to rotate the camera pose corresponding to the conditioning images.
                It is important to make the camera pose the same across data
                to ease the job of generative model training.
            img_cond_ldm_scene_renormalize_scale_threshold:
                if not None, this specifies the minimum rescaling factor that we will bear when training for generative
                model after aligning the conditioning view's corresponding camera pose.
            for_mesh_overfit:
                if True, we will always use a fixed set of xyz_w instead of randomly sampling.
        """
        self.shard_blobby_urls = shard_blobby_urls
        self.batch_size = batch_size
        self.nsamples = nsamples
        self.num_random_views = num_random_views
        self.num_sphere_views = num_sphere_views
        self.num_cond_views = num_cond_views
        self.target_height_px = target_height_px
        self.target_width_px = target_width_px
        self.min_num_voxels = min_num_voxels
        self.max_num_voxels = max_num_voxels
        self.printout = printout
        self.wds_buffer_size = wds_buffer_size
        self.wds_cache_size_tb = wds_cache_size_tb
        self.wds_cache_dir = wds_cache_dir
        self.wds_use_cache = wds_use_cache
        self.wds_sample_life = wds_sample_life
        self.mode = mode
        self.mode_config_dict = mode_config_dict if mode_config_dict is not None else dict()
        self.forbidden_uid_list = forbidden_uid_list
        self.img_cond_ldm_cam_transformation = img_cond_ldm_cam_transformation
        self.img_cond_ldm_scene_renormalize_scale_threshold = img_cond_ldm_scene_renormalize_scale_threshold
        self.debug_num_repeat = max(1, debug_num_repeat)
        self.debug_byte_dict_repeat = max(1, debug_byte_dict_repeat)
        self.min_num_points = min_num_points
        if self.mode == "all":
            self.load_point = True
            self.load_rgbd_random = True
            self.load_rgbd_sphere = True
            self.load_rgbd_cond = True
        elif self.mode == "tokenizer":
            self.load_point = True
            self.load_rgbd_random = True
            self.load_rgbd_sphere = False
            self.load_rgbd_cond = False
        elif self.mode == "lito_tokenizer":
            self.load_point = True
            self.load_rgbd_random = True
            self.load_rgbd_sphere = False
            self.load_rgbd_cond = False
        elif self.mode == "flexible_tokenizer":
            # construct points by backprojecting rgbd_sphere to support more points
            self.load_point = False
            self.load_rgbd_random = True
            self.load_rgbd_sphere = True
            self.load_rgbd_cond = False
        elif self.mode == "voxel":
            self.load_point = True
            self.load_rgbd_random = False
            self.load_rgbd_sphere = False
            self.load_rgbd_cond = False
        elif self.mode == "mesh":
            self.load_point = True
            self.load_rgbd_random = True
            self.load_rgbd_sphere = False
            self.load_rgbd_cond = False
        elif self.mode == "img_cond_ldm":
            self.load_point = True
            self.load_rgbd_random = False
            self.load_rgbd_sphere = True
            self.load_rgbd_cond = False
        elif self.mode == "img_cond_ldm_with_rgbd_cond":
            self.load_point = True
            self.load_rgbd_random = False
            self.load_rgbd_sphere = False
            self.load_rgbd_cond = True
        elif self.mode == "img_cond_ldm_with_rgbd_cond_or_sphere":
            self.load_point = True
            self.load_rgbd_random = False
            self.load_rgbd_sphere = True
            self.load_rgbd_cond = True
        elif self.mode == "lito_img_cond_ldm_with_rgbd_cond_or_sphere":
            self.load_point = True
            self.load_rgbd_random = False
            self.load_rgbd_sphere = True
            self.load_rgbd_cond = True
        else:
            raise NotImplementedError

        if max_num_shards_to_use is not None and max_num_shards_to_use > 0:
            self.shard_blobby_urls = self.shard_blobby_urls[:max_num_shards_to_use]

        self.for_mesh_occ_num_points_to_use = for_mesh_occ_num_points_to_use
        self.for_mesh_occ_grid_res = for_mesh_occ_grid_res
        self.for_mesh_occ_method = for_mesh_occ_method
        self.for_mesh_overfit = for_mesh_overfit
        self.for_mesh_min_num_occ_to_add = for_mesh_min_num_occ_to_add
        self.for_mesh_max_num_occ_to_add = for_mesh_max_num_occ_to_add
        self.for_mesh_occ_expand_type = for_mesh_occ_expand_type

        # create webdataset
        self.create_webdataset()

        self.rng = np.random.default_rng()

    def create_webdataset(self):
        """create webdataset (an iterable dataset)."""

        # decode byte to data
        extract_func = partial(
            extract_sample_from_byte_dict,
            # byte_dict=byte_dict,
            load_point=self.load_point,
            load_rgbd_random=self.load_rgbd_random,
            load_rgbd_sphere=self.load_rgbd_sphere,
            load_rgbd_cond=self.load_rgbd_cond,
            num_random_views=self.num_random_views,
            num_sphere_views=self.num_sphere_views,
            num_cond_views=self.num_cond_views,
            min_num_voxels=self.min_num_voxels,
            max_num_voxels=self.max_num_voxels,
            printout=False,
            forbidden_list=self.forbidden_uid_list,
            min_num_points=self.min_num_points,
            target_width_px=self.target_width_px,
            target_height_px=self.target_height_px,
        )

        if self.wds_cache_dir is not None and self.wds_use_cache:
            os.makedirs(self.wds_cache_dir, exist_ok=True)

        urls = []
        for url in self.shard_blobby_urls:
            if url.startswith("s3://"):
                url = url.replace("s3://", "conductor://", 1)
            urls.append(url)

        self.webdataset = wds.WebDataset(
            urls,  # use our customized apple_gopen
            handler=wds.handlers.warn_and_continue,  # wds.handlers.warn_and_continue, # wds.handlers.reraise_exception
            resampled=True,  # create endless stream of samples (different samples for different workers and nodes), see `ResampledShardList`
            shardshuffle=False,  # since we set resampled=True, no need to shuffle shards
            cache_size=int(self.wds_cache_size_tb * (1024**4)),
            cache_dir=self.wds_cache_dir if self.wds_use_cache else None,
            url_to_name=s3url_to_cache_name,  # wdset has bug that at v1.0.2, it does not used the function
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
        )

        # shuffle pool size (should be large to ensure mixing/randomness)
        # we put shuffle before map, so this means we buffer the raw data (which should usually be more compact)
        if self.wds_buffer_size > 0:
            if self.wds_sample_life <= 1:
                self.webdataset = self.webdataset.shuffle(
                    self.wds_buffer_size,
                    initial=0,
                    # the default initial=100 causes problem when using pytorch dataloader + multiworker without shuffle buffer.
                    # See https://a1350286.slack.com/archives/C08EX5Z3KD3/p1747437283863339
                )
            else:
                print(f"using shuffle_with_{self.wds_sample_life}_lives")
                shuffle_with_life_func = partial(
                    wds_utils.shuffle_with_life,
                    # iter
                    max_life=self.wds_sample_life,
                    bufsize=self.wds_buffer_size,
                    initial=0,
                    rng=None,
                    seed=None,
                )
                self.webdataset = self.webdataset.compose(shuffle_with_life_func)

        if self.debug_byte_dict_repeat > 1:
            repeat_func = partial(
                repeat_sample,
                # iter
                num_repeat=self.debug_byte_dict_repeat,
            )
            self.webdataset = self.webdataset.compose(repeat_func)

        # decode byte into data (e.g., tensor)
        self.webdataset = self.webdataset.map(
            extract_func,
            handler=wds.handlers.warn_and_continue,  # wds.handlers.warn_and_continue  # wds.handlers.reraise_exception
        )

        # make sure we have at least `batch_size` number of samples available
        # when yielding one sample
        self.webdataset = self.webdataset.listed(
            batchsize=self.batch_size,
            partial=False,
        )

        if self.nsamples is not None and self.nsamples > 0:
            self.webdataset = self.webdataset.with_epoch(self.nsamples)  # controls how many samples form an epoch

    def reset(self):
        """Should be called before every epoch (before dataloader creates new threads)."""
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is None

    def process_for_all(self, sample_dict: T.Dict[str, T.Any]):
        return sample_dict

    def process_for_voxel(self, sample_dict: T.Dict[str, T.Any]):
        """
        Args:
            sample_dict:
                uid (str):
                    Unique identifier from byte_dict["__key__"]

        Returns:
            # point
            point_xyz_w (torch.Tensor):
                (n, 3), Point coordinates in world space, [-1, 1]
            point_rgb (torch.Tensor, optional):
                (n, 3) or -1, Point RGB colors, [0, 1]
            point_normal_w (torch.Tensor, optional):
                (n, 3) or -1, Point normals in world space
        """
        # compute point
        point_xyz_w = sample_dict["point_xyz_w"]  # (n, 3)
        point_rgb = sample_dict["point_rgb"] * 2 - 1  # (n, 3) [-1, 1]
        point_ray_o = sample_dict["point_ray_origin_w"]  # (n, 3)
        point_ray_d = sample_dict["point_ray_direction_w"]  # (n, 3)
        point_plucker = utils.get_plucker_representation(
            ray_origin=point_ray_o,  # (n, 3)
            ray_direction=point_ray_d,  # (n, 3)
        )  # (n, 6)

        # compile out_dict
        out_dict = dict(
            dset_type="tokenizer",
            uid=sample_dict["uid"],
            shard_url=sample_dict["shard_url"],
            # point
            point_xyz_w=point_xyz_w,  # (n, 3xyz_w)  [-1, 1]
            point_rgb=point_rgb,  # (n, 3rgb)  [0, 1] or -1
            point_normal_w=-1,
            point_ray_o=point_ray_o,  # (n, 3xyz_w)
            point_ray_d=point_ray_d,  # (n, 3xyz_w)
            point_plucker=point_plucker,  # (n, 6)
        )

        return out_dict

    def process_for_tokenizer(self, sample_dict: T.Dict[str, T.Any]):
        """
        Args:
            sample_dict:
                uid (str):
                    Unique identifier from byte_dict["__key__"]
                point_xyz_w (torch.Tensor):
                    (n, 3) Point coordinates in world space, [-1, 1]
                point_rgb (torch.Tensor, optional):
                    (n, 3) Point RGB colors, [0, 1]
                point_normal_w (torch.Tensor, optional):
                    (n, 3) Point normals in world space
                point_ray_origin_w (torch.Tensor, optional):
                    (n, 3)
                point_ray_direction_w (torch.Tensor, optional):
                    (n, 3)
                rgbd_random (structures.RGBDImage, optional):
                    (b=1, q, h, w) Multi-view RGBD data
        Returns:
            # point
            point_xyz_w (torch.Tensor):
                (n, 3), Point coordinates in world space, [-1, 1]
            point_rgb (torch.Tensor, optional):
                (n, 3) or -1, Point RGB colors, [-1, 1]
            point_normal_w (torch.Tensor, optional):
                (num_frames, n, 3) or -1, Point normals in world space
            point_plucker:
                (n, 6)
            point_ray_o:
                (n, xyz_w)
            point_ray_d:
                (n, 3xyz_w)

            # rgbd random
            rgb_ori:
                (q, 3rgb, h, w) or -1, [0, 1] the loaded views
            H_c2w:
                (q, 4, 4) or -1, camera pose of the loaded views
            intrinsic:
                (q, 3, 3) or -1, camera intrinsics of the loaded views
            rgb_mask:
                (q, 1, h, w) float, [0, 1]  alpha map
            hit_mask:
                (q, 1, h, w) bool, hit map
            normal_w:
                (q, 3xyz, h, w) or -1, [-1, 1] the loaded views
            z_c:
                (q, 1, h, w)
            keep_ray_t_z_c_bug:
                always False
        """

        # compute point
        point_xyz_w = sample_dict["point_xyz_w"]  # (n, 3)
        point_rgb = sample_dict["point_rgb"] * 2 - 1  # (n, 3) [-1, 1]
        point_ray_o = sample_dict["point_ray_origin_w"]  # (n, 3)
        point_ray_d = sample_dict["point_ray_direction_w"]  # (n, 3)
        point_plucker = utils.get_plucker_representation(
            ray_origin=point_ray_o,  # (n, 3)
            ray_direction=point_ray_d,  # (n, 3)
        )  # (n, 6)

        # gather rgbd random
        rgbd: structures.RGBDImage = sample_dict["rgbd_random"]  # (b=1, num_views, h, w)
        _, q, h, w = rgbd.shape
        rgbd_dict = dict(
            rgb_ori=rgbd.rgb.squeeze(0).permute(0, 3, 1, 2),  # (num_views, 3rgb, h, w) [0, 1]
            z_c=rgbd.depth.reshape(q, 1, h, w),  # (num_views, 1, h, w)
            normal_w=rgbd.normal_w.squeeze(0).permute(0, 3, 1, 2),  # num_views, 3xyz, h, w)
            hit_mask=rgbd.hit_map.bool().reshape(q, 1, h, w),  #  (num_views, 1, h, w) bool
            rgb_mask=rgbd.other_maps["alpha"].reshape(q, 1, h, w),  #  (num_views, 1, h, w) bool
            H_c2w=rgbd.camera.H_c2w.squeeze(0),  # (num_views, 4, 4)
            intrinsic=rgbd.camera.intrinsic.squeeze(0),  # (num_views, 3, 3)
            keep_ray_t_z_c_bug=False,
        )

        # compile out_dict
        out_dict = dict(
            dset_type="tokenizer",
            uid=sample_dict["uid"],
            shard_url=sample_dict["shard_url"],
            # point
            point_xyz_w=point_xyz_w,  # (n, 3xyz_w)  [-1, 1]
            point_rgb=point_rgb,  # (n, 3rgb)  [0, 1] or -1
            point_normal_w=-1,
            point_ray_o=point_ray_o,  # (n, 3xyz_w)
            point_ray_d=point_ray_d,  # (n, 3xyz_w)
            point_plucker=point_plucker,  # (n, 6)
            point_ray_origin_direction_w=torch.cat([point_ray_o, point_ray_d], dim=-1),  # (n, 6)
            # rgbd random
            **rgbd_dict,
        )

        return out_dict

    def process_for_lito_tokenizer(self, sample_dict: T.Dict[str, T.Any]):
        """
        Args:
            sample_dict:
                uid (str):
                    Unique identifier from byte_dict["__key__"]
                point_xyz_w (torch.Tensor):
                    (n, 3) Point coordinates in world space, [-1, 1]
                point_rgb (torch.Tensor, optional):
                    (n, 3) Point RGB colors, [0, 1]
                point_normal_w (torch.Tensor, optional):
                    (n, 3) Point normals in world space
                point_ray_origin_w (torch.Tensor, optional):
                    (n, 3)
                point_ray_direction_w (torch.Tensor, optional):
                    (n, 3)
                rgbd_random (structures.RGBDImage, optional):
                    (b=1, q, h, w) Multi-view RGBD data
        Returns:
            # point
            point_xyz_w (torch.Tensor):
                (n, 3), Point coordinates in world space, [0, 1]
            point_rgb (torch.Tensor, optional):
                (n, 3) or -1, Point RGB colors, [0, 1]
            point_ray_origin_direction_w (torch.Tensor, optional):
                (n, 6) or -1, first 3 dimension is ray origin, the next 3 dimension is ray direction

            rgbd_dict_random:
                rgb:
                    (q, h, w, 3rgb) [0, 1], straight
                depth:
                    (q, h, w)
                normal_w:
                    (q, h, w, 3xyz_w) in world coordinate
                hit_map:
                    (q, h, w) bool
                alpha:
                    (q, h, w, 1) [0, 1]
                other_maps:
                    dict, each value is (q, h, w, d)

                H_c2w:
                    (1, q, 4, 4)  camera pose
                intrinsic:
                    (1, q, 3, 3)  camera intrinsics
        """

        # compute point
        pdict = dict(
            point_xyz_w=sample_dict["point_xyz_w"],  # (n, 3xyz_w)  [-1, 1]
            point_rgb=sample_dict.get("point_rgb"),  # (n, 3rgb) [0, 1]
            point_normal_w=sample_dict.get("point_normal_w"),  # (n, 3xyz_w)
            point_alpha=sample_dict.get("point_alpha"),  # (n, 1) [0, 1]
            point_ray_origin_direction_w=sample_dict.get("point_ray_origin_direction_w"),  # (n, 6)
        )
        if pdict["point_ray_origin_direction_w"] is None:
            point_ray_o = sample_dict["point_ray_origin_w"]  # (n, 3)
            point_ray_d = sample_dict["point_ray_direction_w"]  # (n, 3)
            assert point_ray_o is not None and point_ray_d is not None
            point_ray_origin_direction_w = torch.cat([point_ray_o, point_ray_d], dim=-1)
            pdict["point_ray_origin_direction_w"] = point_ray_origin_direction_w

        # gather rgbd random
        rgbd: structures.RGBDImage = sample_dict["rgbd_random"]  # (b=1, num_views, h, w)
        rgbd_dict_random = dict(
            rgb=rgbd.rgb,  # (1, num_views, h, w, 3rgb) [0, 1] straight
            depth=rgbd.depth,  # (1, num_views, h, w)
            normal_w=rgbd.normal_w,
            hit_map=rgbd.hit_map,
            alpha=rgbd.other_maps.get("alpha", None),  # (1, num_views, h, w, 1) [0, 1]
            H_c2w=rgbd.camera.H_c2w,  # (1, num_views, 4, 4)
            intrinsic=rgbd.camera.intrinsic,  # (1, num_views, 3, 3)
        )

        for key in pdict:
            if pdict[key] is None:
                pdict[key] = -1

        for key in rgbd_dict_random:
            if rgbd_dict_random[key] is None:
                rgbd_dict_random[key] = -1
            if isinstance(rgbd_dict_random[key], torch.Tensor):
                rgbd_dict_random[key] = rgbd_dict_random[key].squeeze(0)

        # compile out_dict
        out_dict = dict(
            dset_type="lito_tokenizer",
            uid=sample_dict["uid"],
            shard_url=sample_dict["shard_url"],
            # point
            **pdict,
            #
            rgbd_dict_random=rgbd_dict_random,  # (q, h, w)
            total_decoding_time=sample_dict["total_decoding_time"],  # float, secs
        )

        return out_dict

    def process_for_flexible_tokenizer(self, sample_dict: T.Dict[str, T.Any]):
        """
        Args:
            sample_dict:
                uid (str):
                    Unique identifier from byte_dict["__key__"]
                rgbd_random (structures.RGBDImage):
                    (b=1, q, h, w) Multi-view RGBD data
                rgbd_sphere (structures.RGBDImage):
                    (b=1, q, h, w) Multi-view RGBD data
        Returns:
            rgbd_dict_random:
                rgb:
                    (1, q, h, w, 3rgb) [0, 1], straight
                depth:
                    (1, q, h, w)
                normal_w:
                    (1, q, h, w, 3xyz_w) in world coordinate
                hit_map:
                    (1, q, h, w) bool
                alpha:
                    (1, q, h, w, 1) [0, 1]

                H_c2w:
                    (1, q, 4, 4)  camera pose
                intrinsic:
                    (1, q, 3, 3)  camera intrinsics

            rgbd_dict_sphere:
        """
        # gather rgbd random
        rgbd: structures.RGBDImage = sample_dict["rgbd_random"]  # (b=1, num_views, h, w)
        rgbd_dict_random = dict(
            rgb=rgbd.rgb,  # (1, num_views, h, w, 3rgb) [0, 1] straight
            depth=rgbd.depth,  # (1, num_views, h, w)
            normal_w=rgbd.normal_w,
            hit_map=rgbd.hit_map,
            alpha=rgbd.other_maps.get("alpha", None),  # (1, num_views, h, w, 1) [0, 1]
            H_c2w=rgbd.camera.H_c2w,  # (1, num_views, 4, 4)
            intrinsic=rgbd.camera.intrinsic,  # (1, num_views, 3, 3)
            keep_ray_t_z_c_bug=False,
        )
        for key in rgbd_dict_random:
            if rgbd_dict_random[key] is None:
                rgbd_dict_random[key] = -1

        # gather rgbd random
        rgbd: structures.RGBDImage = sample_dict["rgbd_sphere"]  # (b=1, num_views, h, w)
        rgbd_dict_sphere = dict(
            rgb=rgbd.rgb,  # (1, num_views, h, w, 3rgb) [0, 1] straight
            depth=rgbd.depth,  # (1, num_views, h, w)
            normal_w=rgbd.normal_w,
            hit_map=rgbd.hit_map,
            alpha=rgbd.other_maps.get("alpha", None),  # (1, num_views, h, w, 1) [0, 1]
            H_c2w=rgbd.camera.H_c2w,  # (1, num_views, 4, 4)
            intrinsic=rgbd.camera.intrinsic,  # (1, num_views, 3, 3)
            keep_ray_t_z_c_bug=False,
        )
        for key in rgbd_dict_sphere:
            if rgbd_dict_sphere[key] is None:
                rgbd_dict_sphere[key] = -1

        # compile out_dict
        out_dict = dict(
            dset_type="flexible_tokenizer",
            uid=sample_dict["uid"],
            shard_url=sample_dict["shard_url"],
            rgbd_dict_random=rgbd_dict_random,  # (1, q, h, w)
            rgbd_dict_sphere=rgbd_dict_sphere,  # (1, q, h, w)
        )

        return out_dict

    def process_for_mesh(self, sample_dict: T.Dict[str, T.Any]):
        """
        Args:
            sample_dict:
                uid (str):
                    Unique identifier from byte_dict["__key__"]
                point_xyz_w (torch.Tensor):
                    (n, 3) Point coordinates in world space, [-1, 1]
                point_rgb (torch.Tensor, optional):
                    (n, 3) Point RGB colors, [0, 1]
                point_normal_w (torch.Tensor, optional):
                    (n, 3) Point normals in world space
                point_ray_origin_w (torch.Tensor, optional):
                    (n, 3)
                point_ray_direction_w (torch.Tensor, optional):
                    (n, 3)
                rgbd_random (structures.RGBDImage, optional):
                    (b=1, q, h, w) Multi-view RGBD data
        Returns:
            # point
            point_xyz_w (torch.Tensor):
                (n, 3), Point coordinates in world space, [-1, 1]
            point_rgb (torch.Tensor, optional):
                (n, 3) or -1, Point RGB colors, [-1, 1]
            point_normal_w (torch.Tensor, optional):
                (num_frames, n, 3) or -1, Point normals in world space
            point_plucker:
                (n, 6)
            point_ray_o:
                (n, xyz_w)
            point_ray_d:
                (n, 3xyz_w)

            # rgbd random
            rgb_ori:
                (q, 3rgb, h, w) or -1, [0, 1] the loaded views
            H_c2w:
                (q, 4, 4) or -1, camera pose of the loaded views
            intrinsic:
                (q, 3, 3) or -1, camera intrinsics of the loaded views
            rgb_mask:
                (q, 1, h, w) float, [0, 1]  alpha map
            hit_mask:
                (q, 1, h, w) bool, hit map
            normal_w:
                (q, 3xyz, h, w) or -1, [-1, 1] the loaded views
            z_c:
                (q, 1, h, w)
            keep_ray_t_z_c_bug:
                always False
        """

        # compute point
        point_xyz_w = sample_dict["point_xyz_w"]  # (n, 3)
        point_rgb = sample_dict["point_rgb"] * 2 - 1  # (n, 3) [-1, 1]
        point_ray_o = sample_dict["point_ray_origin_w"]  # (n, 3)
        point_ray_d = sample_dict["point_ray_direction_w"]  # (n, 3)
        point_plucker = utils.get_plucker_representation(
            ray_origin=point_ray_o,  # (n, 3)
            ray_direction=point_ray_d,  # (n, 3)
        )  # (n, 6)

        # gather rgbd random
        rgbd: structures.RGBDImage = sample_dict["rgbd_random"]  # (b=1, num_views, h, w)
        _, q, h, w = rgbd.shape

        # tmp = torch.eye(4).repeat(*rgbd.camera.H_c2w.shape[:-2], 1, 1)
        # print(f"\n\n{rgbd.depth.shape=}, {rgbd.camera.intrinsic.shape=}, {tmp.shape=}\n\n")

        unproject_dict = utils.compute_3d_xyz(
            z_map=rgbd.depth,  # (b, q, h, w)
            intrinsic=rgbd.camera.intrinsic,  # (b, q, 3, 3)
            H_c2w=torch.eye(4).repeat(*rgbd.camera.H_c2w.shape[:-2], 1, 1),  # to get results in camera coordinates
            subsample=1,
            other_maps=None,
        )  # (b, q, h, w, 3)  can contain nan
        xyz_c = unproject_dict["xyz_w"]  # (b, q, h, w, 3)

        rgbd_dict = dict(
            rgb_ori=rgbd.rgb.squeeze(0).permute(0, 3, 1, 2),  # (num_views, 3rgb, h, w) [0, 1]
            z_c=rgbd.depth.reshape(q, 1, h, w),  # (num_views, 1, h, w)
            xyz_c=xyz_c[0].permute(0, 3, 1, 2),  # (num_views, 3, h, w)
            normal_w=rgbd.normal_w.squeeze(0).permute(0, 3, 1, 2),  # num_views, 3xyz, h, w)
            hit_mask=rgbd.hit_map.bool().reshape(q, 1, h, w),  #  (num_views, 1, h, w) bool
            rgb_mask=rgbd.other_maps["alpha"].reshape(q, 1, h, w),  #  (num_views, 1, h, w) bool
            H_c2w=rgbd.camera.H_c2w.squeeze(0),  # (num_views, 4, 4)
            intrinsic=rgbd.camera.intrinsic.squeeze(0),  # (num_views, 3, 3)
            keep_ray_t_z_c_bug=False,
        )

        if self.for_mesh_occ_num_points_to_use > 0:
            if self.for_mesh_overfit:
                _xyz_w = point_xyz_w[: self.for_mesh_occ_num_points_to_use]
            else:
                ridx = self.rng.permutation(point_xyz_w.shape[0])[: self.for_mesh_occ_num_points_to_use]
                _xyz_w = point_xyz_w[ridx]  # (n', 3)
        else:
            _xyz_w = point_xyz_w  # (n, 3)

        if self.for_mesh_occ_method == "o3d":
            o3d_pcd = o3d_utils.creat_pcd(points=_xyz_w.cpu().float().numpy())
            occ_grid = o3d_utils.create_dense_voxel_grid_from_o3d_pcd(
                o3d_pcd=o3d_pcd,
                num_voxels=self.for_mesh_occ_grid_res,
                cell_width=2.0 / self.for_mesh_occ_grid_res,
                start_xyz_w=-1.0,
            )  # (res_z, res_y, res_x)  bool
            occ_grid = torch.from_numpy(occ_grid)  # (res_z, res_y, res_x)  bool
        elif self.for_mesh_occ_method == "torch":
            occ_grid = compute_occupancy_grid(
                xyz_w=_xyz_w,
                grid_size=64,
                min_xyz_w=-1,
                max_xyz_w=1,
            )  # (res_z, res_y, res_x)  bool
        else:
            raise NotImplementedError(f"{self.for_mesh_occ_method=}")

        min_num_to_add = self.for_mesh_min_num_occ_to_add
        max_num_to_add = self.for_mesh_max_num_occ_to_add
        num_to_add = np.random.randint(min_num_to_add, max_num_to_add + 1)
        occ_expand_type = self.for_mesh_occ_expand_type
        if num_to_add > 0:
            expanded_occ_grid = expand_occ_grid_for_mesh(
                occ_grid=occ_grid[None, None],
                num_to_add=num_to_add,
                occ_expand_type=occ_expand_type,
                dtype=torch.float,
                device=torch.device("cpu"),
            )[0, 0]  # (res_z, res_y, res_x)
            n_occ_voxels = torch.sum(expanded_occ_grid)
            if self.max_num_voxels > 0:
                assert n_occ_voxels <= self.max_num_voxels, f"{n_occ_voxels=}, {self.max_num_voxels=}"

            occ_grid = expanded_occ_grid

        # print(f"\n\n[dataset] {torch.sum(_xyz_w.abs())=}, {torch.sum(occ_grid.float().abs())=}\n\n")

        # compile out_dict
        out_dict = dict(
            dset_type="tokenizer",
            uid=sample_dict["uid"],
            shard_url=sample_dict["shard_url"],
            # point
            point_xyz_w=point_xyz_w,  # (n, 3xyz_w)  [-1, 1]
            point_rgb=point_rgb,  # (n, 3rgb)  [0, 1] or -1
            point_normal_w=-1,
            point_ray_o=point_ray_o,  # (n, 3xyz_w)
            point_ray_d=point_ray_d,  # (n, 3xyz_w)
            point_plucker=point_plucker,  # (n, 6)
            # for mesh
            occ_grid=occ_grid,
            # rgbd random
            **rgbd_dict,
        )

        return out_dict

    def process_for_img_cond_ldm(self, sample_dict: T.Dict[str, T.Any]):
        """
        Args:
            sample_dict:
                uid (str):
                    Unique identifier from byte_dict["__key__"]
                point_xyz_w (torch.Tensor):
                    (n, 3) Point coordinates in world space, [-1, 1]
                point_rgb (torch.Tensor, optional):
                    (n, 3) Point RGB colors, [0, 1]
                point_normal_w (torch.Tensor, optional):
                    (n, 3) Point normals in world space
                point_ray_origin_w (torch.Tensor, optional):
                    (n, 3)
                point_ray_direction_w (torch.Tensor, optional):
                    (n, 3)
                rgbd_sphere (structures.RGBDImage, optional):
                    (b=1, q, h, w) Multi-view RGBD data
        Returns:
            # point
            point_xyz_w (torch.Tensor):
                (n, 3), Point coordinates in world space, [-1, 1]
            point_rgb (torch.Tensor, optional):
                (n, 3) or -1, Point RGB colors, [-1, 1]
            point_normal_w (torch.Tensor, optional):
                (num_frames, n, 3) or -1, Point normals in world space
            point_plucker:
                (n, 6)
            point_ray_o:
                (n, xyz_w)
            point_ray_d:
                (n, 3xyz_w)

            # rgbd sphere
            rgb_ori:
                (q, 3rgb, h, w) or -1, [0, 1] the loaded views
            H_c2w:
                (q, 4, 4) or -1, camera pose of the loaded views
            intrinsic:
                (q, 3, 3) or -1, camera intrinsics of the loaded views
            rgb_mask:
                (q, 1, h, w) float, [0, 1]  alpha map
            hit_mask:
                (q, 1, h, w) bool, hit map
            normal_w:
                (q, 3xyz, h, w) or -1, [-1, 1] the loaded views
            z_c:
                (q, 1, h, w)
            keep_ray_t_z_c_bug:
                always False
        """

        # compute point
        point_xyz_w = sample_dict["point_xyz_w"]  # (n, 3)
        point_rgb = sample_dict["point_rgb"] * 2 - 1  # (n, 3) [-1, 1]
        point_ray_o = sample_dict["point_ray_origin_w"]  # (n, 3)
        point_ray_d = sample_dict["point_ray_direction_w"]  # (n, 3)

        # gather rgbd random
        if self.mode == "img_cond_ldm":
            rgbd: structures.RGBDImage = sample_dict["rgbd_sphere"]  # (b=1, num_views, h, w)
        elif self.mode == "img_cond_ldm_with_rgbd_cond":
            rgbd: structures.RGBDImage = sample_dict["rgbd_cond"]  # (b=1, num_views, h, w)
        elif self.mode == "img_cond_ldm_with_rgbd_cond_or_sphere":
            if "rgbd_cond" in sample_dict:
                rgbd: structures.RGBDImage = sample_dict["rgbd_cond"]  # (b=1, num_views, h, w)
            elif "rgbd_sphere" in sample_dict:
                rgbd: structures.RGBDImage = sample_dict["rgbd_sphere"]  # (b=1, num_views, h, w)
            else:
                raise ValueError(f"{list(sample_dict.keys())=}")
        else:
            raise ValueError(f"{self.mode=}")

        scene_renormalize_scale = 1.0

        if self.img_cond_ldm_cam_transformation is not None:
            H_c2w_for_cond_img = rgbd.camera.H_c2w[0, 0]  # (4, 4)

            cam_dict = convert_camera_wrt_cond_img_space(
                dtype=rgbd.camera.H_c2w.dtype,
                device=rgbd.camera.H_c2w.device,
                H_c2w=None,
                H_c2w_for_cond_img=H_c2w_for_cond_img,
                img_cond_ldm_cam_transformation=self.img_cond_ldm_cam_transformation,
            )
            R_w2b = cam_dict["R_w2b"]
            H_w2b = cam_dict["H_w2b"]

            # if self.img_cond_ldm_cam_transformation == "y_up":
            #     # rotate the world coordinate so that the first camera's camera pose
            #     # is at diag([1, -1, -1]), ie, the world is y-up
            #     R_w2c = rgbd.camera.H_c2w[0, 0, :3, :3].t()  # (3, 3)
            #     R_c2b = torch.tensor(
            #         [
            #             [1, 0, 0],
            #             [0, -1, 0],
            #             [0, 0, -1],
            #         ],
            #         dtype=R_w2c.dtype,
            #         device=R_w2c.device,
            #     )  # (3, 3)
            #     R_w2b = R_c2b @ R_w2c
            #     H_w2b = torch.eye(4)  # (4, 4)
            #     H_w2b[:3, :3] = R_w2b
            # elif self.img_cond_ldm_cam_transformation == "z_up":
            #     # rotate the world coordinate so that the first camera's camera pose
            #     # is below and the pinhole is at (r, 0, 0), ie, the world is z-up
            #     R_w2c = rgbd.camera.H_c2w[0, 0, :3, :3].t()  # (3, 3)
            #     R_c2b = torch.tensor(
            #         [
            #             [0, 0, -1],
            #             [1, 0, 0],
            #             [0, -1, 0],
            #         ],
            #         dtype=R_w2c.dtype,
            #         device=R_w2c.device,
            #     )  # (3, 3)
            #     R_w2b = R_c2b @ R_w2c
            #     H_w2b = torch.eye(4)  # (4, 4)
            #     H_w2b[:3, :3] = R_w2b
            # elif self.img_cond_ldm_cam_transformation == "identity_rotation":
            #     # rotate the world coordinate so that the first camera's oriention aligns with the world.
            #     # The pinhole is at (0, 0, -r)
            #     R_w2c = rgbd.camera.H_c2w[0, 0, :3, :3].t()  # (3, 3)
            #     R_c2b = torch.eye(3)  # (3, 3)
            #     R_w2b = R_c2b @ R_w2c
            #     H_w2b = torch.eye(4)  # (4, 4)
            #     H_w2b[:3, :3] = R_w2b
            # else:
            #     raise ValueError(f"{self.img_cond_ldm_cam_transformation=}")

            # debug_diff = torch.sum(torch.abs(H_w2b_new - H_w2b))
            # print(f"\n\ndiff: {debug_diff=}, {H_w2b_new.shape=}, {H_w2b.shape=}\n\n")

            point_xyz_w = linalg_utils.matmul(R_w2b, point_xyz_w.unsqueeze(-1)).squeeze(-1)  # (n, 3)
            point_ray_o = linalg_utils.matmul(R_w2b, point_ray_o.unsqueeze(-1)).squeeze(-1)  # (n, 3)
            point_ray_d = linalg_utils.matmul(R_w2b, point_ray_d.unsqueeze(-1)).squeeze(-1)  # (n, 3)

            # print(f"rgbd.camera.H_c2w: {rgbd.camera.H_c2w}")
            rgbd.coordinate_transform(H_w2n=H_w2b.unsqueeze(0))
            # print(f"rgbd.camera.H_c2b: {rgbd.camera.H_c2w}")

            # NOTE: after rotation, xyz_w could be out of bounding box of [-1, 1].
            # If that happens, we need to scale it.
            point_xyz_w_max_abs = torch.abs(point_xyz_w).max()
            if point_xyz_w_max_abs <= 1.0 + 1e-6:
                scene_renormalize_scale = 1.0
                flag_renormalize = False
            else:
                scene_renormalize_scale = (1 - 1e-6) / point_xyz_w_max_abs
                flag_renormalize = True

        if self.img_cond_ldm_scene_renormalize_scale_threshold is not None:
            # NOTE: since this is a IterableDataset and we have `try...except...` in the __iter__ function.
            # So we can safely raise Error here.
            assert scene_renormalize_scale >= self.img_cond_ldm_scene_renormalize_scale_threshold, (
                f"We ignore data that will be shrinked too much as {scene_renormalize_scale=} "
                f"with {self.img_cond_ldm_scene_renormalize_scale_threshold=}."
            )

        _, q, h, w = rgbd.shape

        depth = rgbd.depth.reshape(q, 1, h, w)
        H_c2w = rgbd.camera.H_c2w.squeeze(0)  # (1, num_views, 4, 4) -> (num_views, 4, 4)

        if flag_renormalize:
            point_xyz_w = point_xyz_w * scene_renormalize_scale
            point_ray_o = point_ray_o * scene_renormalize_scale
            # Since the point_ray_d is normalized, we do not need to change it
            depth = depth * scene_renormalize_scale
            H_c2w[:, :3, 3] = H_c2w[:, :3, 3] * scene_renormalize_scale

        # compute point_plucker in the new b coordinate
        point_plucker = utils.get_plucker_representation(
            ray_origin=point_ray_o,  # (n, 3)
            ray_direction=point_ray_d,  # (n, 3)
        )  # (n, 6)

        rgbd_dict = dict(
            rgb_ori=rgbd.rgb.squeeze(0).permute(0, 3, 1, 2),  # (num_views, 3rgb, h, w) [0, 1]
            z_c=depth,  # (num_views, 1, h, w)
            normal_w=rgbd.normal_w.squeeze(0).permute(0, 3, 1, 2),  # num_views, 3xyz, h, w)
            hit_mask=rgbd.hit_map.bool().reshape(q, 1, h, w),  #  (num_views, 1, h, w) bool
            rgb_mask=rgbd.other_maps["alpha"].reshape(q, 1, h, w),  #  (num_views, 1, h, w) float
            H_c2w=H_c2w,  # (num_views, 4, 4)
            intrinsic=rgbd.camera.intrinsic.squeeze(0),  # (num_views, 3, 3)
            keep_ray_t_z_c_bug=False,
        )

        # compile out_dict
        out_dict = dict(
            dset_type="img_cond_ldm",
            uid=sample_dict["uid"],
            shard_url=sample_dict["shard_url"],
            # point
            point_xyz_w=point_xyz_w,  # (n, 3xyz_w)  [-1, 1]
            point_rgb=point_rgb,  # (n, 3rgb)  [0, 1] or -1
            point_normal_w=-1,
            point_ray_o=point_ray_o,  # (n, 3xyz_w)
            point_ray_d=point_ray_d,  # (n, 3xyz_w)
            point_plucker=point_plucker,  # (n, 6)
            # normalization
            scene_renormalize_scale=torch.FloatTensor([scene_renormalize_scale]),
            # rgbd random
            **rgbd_dict,
        )

        return out_dict

    def process_for_lito_img_cond_ldm(self, sample_dict: T.Dict[str, T.Any]):
        """
        Args:
            sample_dict:
                uid (str):
                    Unique identifier from byte_dict["__key__"]
                point_xyz_w (torch.Tensor):
                    (n, 3) Point coordinates in world space, [-1, 1]
                point_rgb (torch.Tensor, optional):
                    (n, 3) Point RGB colors, [0, 1]
                point_normal_w (torch.Tensor, optional):
                    (n, 3) Point normals in world space
                point_ray_origin_w (torch.Tensor, optional):
                    (n, 3)
                point_ray_direction_w (torch.Tensor, optional):
                    (n, 3)
                rgbd_sphere (structures.RGBDImage, optional):
                    (b=1, q, h, w) Multi-view RGBD data
        Returns:
            # point
            point_xyz_w (torch.Tensor):
                (n, 3), Point coordinates in world space, [-1, 1]
            point_rgb (torch.Tensor, optional):
                (n, 3) or -1, Point RGB colors, [-1, 1]
        """

        # compute point
        pdict = dict(
            point_xyz_w=sample_dict["point_xyz_w"],  # (n, 3xyz_w)  [-1, 1]
            point_rgb=sample_dict.get("point_rgb"),  # (n, 3rgb) [0, 1]
            point_normal_w=sample_dict.get("point_normal_w"),  # (n, 3xyz_w)
            point_alpha=sample_dict.get("point_alpha"),  # (n, 1) [0, 1]
            point_ray_origin_direction_w=sample_dict.get("point_ray_origin_direction_w"),  # (n, 6)
        )
        if pdict["point_ray_origin_direction_w"] is None:
            point_ray_o = sample_dict["point_ray_origin_w"]  # (n, 3)
            point_ray_d = sample_dict["point_ray_direction_w"]  # (n, 3)
            assert point_ray_o is not None and point_ray_d is not None
            point_ray_origin_direction_w = torch.cat([point_ray_o, point_ray_d], dim=-1)
            pdict["point_ray_origin_direction_w"] = point_ray_origin_direction_w

        # gather rgbd random
        if self.mode == "lito_img_cond_ldm":
            rgbd: structures.RGBDImage = sample_dict["rgbd_sphere"]  # (b=1, num_views, h, w)
        elif self.mode == "lito_img_cond_ldm_with_rgbd_cond":
            rgbd: structures.RGBDImage = sample_dict["rgbd_cond"]  # (b=1, num_views, h, w)
        elif self.mode == "lito_img_cond_ldm_with_rgbd_cond_or_sphere":
            _rgbds = []
            if "rgbd_cond" in sample_dict:
                _rgbds.append(sample_dict["rgbd_cond"])  # (b=1, num_views, h, w)
                # rgbd: structures.RGBDImage = sample_dict["rgbd_cond"]  # (b=1, num_views, h, w)
            if "rgbd_sphere" in sample_dict:
                _rgbds.append(sample_dict["rgbd_sphere"])  # (b=1, num_views, h, w)
                # rgbd: structures.RGBDImage = sample_dict["rgbd_sphere"]  # (b=1, num_views, h, w)
            assert len(_rgbds) > 0
            idx = np.random.randint(low=0, high=len(_rgbds))
            rgbd = _rgbds[idx]
            del _rgbds
        else:
            raise ValueError(f"{self.mode=}")

        scene_renormalize_scale = 1.0

        if self.img_cond_ldm_cam_transformation is not None:
            H_c2w_for_cond_img = rgbd.camera.H_c2w[0, 0]  # (4, 4)

            cam_dict = convert_camera_wrt_cond_img_space(
                dtype=rgbd.camera.H_c2w.dtype,
                device=rgbd.camera.H_c2w.device,
                H_c2w=None,
                H_c2w_for_cond_img=H_c2w_for_cond_img,
                img_cond_ldm_cam_transformation=self.img_cond_ldm_cam_transformation,
            )
            R_w2b = cam_dict["R_w2b"]
            H_w2b = cam_dict["H_w2b"]

            for key in [
                "point_xyz_w",
                "point_normal_w",
                "point_ray_origin_direction_w",
            ]:
                arr = pdict.get(key, None)
                if arr is not None:
                    if key != "point_ray_origin_direction_w":
                        arr = linalg_utils.matmul(R_w2b, arr.unsqueeze(-1)).squeeze(-1)  # (n, 3)
                    else:
                        arr = torch.cat(
                            [
                                linalg_utils.matmul(R_w2b, arr[..., :3].unsqueeze(-1)).squeeze(-1),  # (n, 3)
                                linalg_utils.matmul(R_w2b, arr[..., 3:6].unsqueeze(-1)).squeeze(-1),  # (n, 3)
                            ],
                            dim=-1,
                        )  # (n, 6)
                    pdict[key] = arr

            rgbd.coordinate_transform(H_w2n=H_w2b.unsqueeze(0))

            # NOTE: after rotation, xyz_w could be out of bounding box of [-1, 1].
            # If that happens, we need to scale it.
            point_xyz_w_max_abs = torch.abs(pdict["point_xyz_w"]).max()
            if point_xyz_w_max_abs <= 1.0 + 1e-6:
                scene_renormalize_scale = 1.0
                flag_renormalize = False
            else:
                scene_renormalize_scale = (1 - 1e-6) / point_xyz_w_max_abs
                flag_renormalize = True

        else:
            flag_renormalize = False

        if self.img_cond_ldm_scene_renormalize_scale_threshold is not None:
            # NOTE: since this is a IterableDataset and we have `try...except...` in the __iter__ function.
            # So we can safely raise Error here.
            assert scene_renormalize_scale >= self.img_cond_ldm_scene_renormalize_scale_threshold, (
                f"We ignore data that will be shrinked too much as {scene_renormalize_scale=} "
                f"with {self.img_cond_ldm_scene_renormalize_scale_threshold=}."
            )

        _, q, h, w = rgbd.shape

        if flag_renormalize:
            for key in [
                "point_xyz_w",
            ]:
                arr = pdict.get(key, None)
                if arr is not None:
                    arr = arr * scene_renormalize_scale
                    pdict[key] = arr

            for key in [
                "point_ray_origin_direction_w",
            ]:
                arr = pdict.get(key, None)
                if arr is not None:
                    arr[..., :3] = arr[..., :3] * scene_renormalize_scale
                    pdict[key] = arr

            # Since the point_ray_d is normalized, we do not need to change it
            rgbd.depth = rgbd.depth * scene_renormalize_scale
            rgbd.camera.H_c2w[:, :3, 3] = rgbd.camera.H_c2w[:, :3, 3] * scene_renormalize_scale

        assert rgbd.rgb.isfinite().all(), (
            f"{sample_dict['uid']}, {sample_dict['shard_url']}, "
            f"rgb, nan: {rgbd.rgb.isnan().any()}"
            f"inf: {rgbd.rgb.isinf().any()}"
        )
        if rgbd.other_maps.get("alpha", None) is not None:
            assert rgbd.other_maps["alpha"].isfinite().all(), (
                f"{sample_dict['uid']}, {sample_dict['shard_url']}, "
                f"alpha, nan: {rgbd.other_maps['alpha'].isnan().any()}"
                f"inf: {rgbd.other_maps['alpha'].isinf().any()}"
            )

        # create rgbd_dict_cond
        rgbd_dict_cond = dict(
            rgb=rgbd.rgb,  # (1, num_views, h, w, 3rgb) [0, 1] straight
            # depth=rgbd.depth,  # (1, num_views, h, w)
            # normal_w=rgbd.normal_w,
            # hit_map=rgbd.hit_map,
            alpha=rgbd.other_maps.get("alpha", None),  # (1, num_views, h, w, 1) [0, 1]
            # albedo=rgbd.other_maps.get("albedo", None),  # (1, num_views, h, w, 3) [0, 1]
            # roughness_metallic=rgbd.other_maps.get("roughness_metallic", None),  # (1, num_views, h, w, 2) [0, 1]
            H_c2w=rgbd.camera.H_c2w,  # (1, num_views, 4, 4)
            intrinsic=rgbd.camera.intrinsic,  # (1, num_views, 3, 3)
        )

        for key in pdict:
            if pdict[key] is None:
                pdict[key] = -1

        for key in rgbd_dict_cond:
            if rgbd_dict_cond[key] is None:
                rgbd_dict_cond[key] = -1
            if isinstance(rgbd_dict_cond[key], torch.Tensor):
                rgbd_dict_cond[key] = rgbd_dict_cond[key].squeeze(0)

        # compile out_dict
        out_dict = dict(
            dset_type="lito_img_cond_ldm",
            uid=sample_dict["uid"],
            shard_url=sample_dict["shard_url"],
            # point
            **pdict,
            # normalization
            scene_renormalize_scale=torch.FloatTensor([scene_renormalize_scale]),
            rgbd_dict_cond=rgbd_dict_cond,  # (q, h, w)
            total_decoding_time=sample_dict["total_decoding_time"],  # float, secs
        )

        return out_dict

    def __iter__(self):
        """
        Returns:

        """
        # add a random sleep so all workers do not crowd the bandwidth
        if torch.utils.data.get_worker_info() is not None:  # within a worker
            r_time = np.random.rand(1).item() * 10
            time.sleep(r_time)

        base_iter = iter(self.webdataset)

        while True:
            try:
                sample_dicts = next(base_iter)  # (b,) list of sample_dict
                # convert sample to the expect format for each mode
                out_dicts = []
                for sample_dict in sample_dicts:
                    if self.mode == "all":
                        out_dict = self.process_for_all(sample_dict)
                    elif self.mode == "voxel":
                        out_dict = self.process_for_voxel(sample_dict)
                    elif self.mode == "mesh":
                        out_dict = self.process_for_mesh(sample_dict)
                    elif self.mode == "tokenizer":
                        out_dict = self.process_for_tokenizer(sample_dict)
                    elif self.mode == "lito_tokenizer":
                        out_dict = self.process_for_lito_tokenizer(sample_dict)
                    elif self.mode == "flexible_tokenizer":
                        out_dict = self.process_for_flexible_tokenizer(sample_dict)
                    elif self.mode in [
                        "img_cond_ldm",
                        "img_cond_ldm_with_rgbd_cond",
                        "img_cond_ldm_with_rgbd_cond_or_sphere",
                    ]:
                        out_dict = self.process_for_img_cond_ldm(sample_dict)
                    elif self.mode in [
                        "lito_img_cond_ldm",
                        "lito_img_cond_ldm_with_rgbd_cond",
                        "lito_img_cond_ldm_with_rgbd_cond_or_sphere",
                    ]:
                        out_dict = self.process_for_lito_img_cond_ldm(sample_dict)
                    else:
                        raise NotImplementedError(self.mode)
                    out_dicts.append(out_dict)

                for r in range(self.debug_num_repeat):
                    yield out_dicts  # list of dict

            except StopIteration:
                break
            except Exception:
                traceback.print_exc()
                continue


class ObjIterableDataModule(base.BaseDataModule):
    """
    Since DObjIterableDataset uses webdataset and is expected to
    be used with `resampled = True` when creating the webdataset,
    we do not need to do any special things besides
    concatenate the urls to each tar files.
    """

    name = "obj_wdset"

    def __init__(
        self,
        split_dict_urls: T.List[str],  # train/valid/test -> list of urls
        valid_id_urls: T.List[str] | None = None,
        overfit_urls: T.List[str] | None = None,
        method: str = "wdset",  # specifically different from the old `webdataset` method
        train_dataset_class: str = "ObjIterableDataset",
        train_dataset_config: T.Optional[T.Dict[str, T.Any]] = None,
        valid_dataset_class: str = "ObjIterableDataset",
        valid_dataset_config: T.Optional[T.Dict[str, T.Any]] = None,
        test_dataset_class: str = "ObjIterableDataset",
        test_dataset_config: T.Optional[T.Dict[str, T.Any]] = None,
        predict_dataset_class: str = "ObjIterableDataset",
        predict_dataset_config: T.Optional[T.Dict[str, T.Any]] = None,
        train_dataloader_config: T.Optional[T.Dict[str, T.Any]] = None,
        valid_dataloader_config: T.Optional[T.Dict[str, T.Any]] = None,
        test_dataloader_config: T.Optional[T.Dict[str, T.Any]] = None,
        predict_dataloader_config: T.Optional[T.Dict[str, T.Any]] = None,
    ):
        super().__init__(
            method=method,
            train_dataset_config=train_dataset_config,
            valid_dataset_config=valid_dataset_config,
            test_dataset_config=test_dataset_config,
            predict_dataset_config=predict_dataset_config,
            train_dataloader_config=train_dataloader_config,
            valid_dataloader_config=valid_dataloader_config,
            test_dataloader_config=test_dataloader_config,
            predict_dataloader_config=predict_dataloader_config,
        )
        self.train_dataset_class = train_dataset_class
        self.valid_dataset_class = valid_dataset_class
        self.test_dataset_class = test_dataset_class
        self.predict_dataset_class = predict_dataset_class
        self.split_dict_urls = split_dict_urls

        if isinstance(self.split_dict_urls, str):
            self.split_dict_urls = [self.split_dict_urls]

        all_split_dict = dict()
        for url in self.split_dict_urls:
            if url.startswith("s3://"):
                url = url.replace("s3://", "conductor://", 1)
            with af.open(url, "r") as f:
                split_dict = json.load(f)

            for key in split_dict:
                if key not in all_split_dict:
                    all_split_dict[key] = split_dict[key]
                else:
                    all_split_dict[key] = all_split_dict[key] + split_dict[key]

        # overfit to specific urls
        if overfit_urls is not None:
            for tmp_k in all_split_dict:
                all_split_dict[tmp_k] = overfit_urls

        for key in all_split_dict:
            all_split_dict[key] = sorted(all_split_dict[key])
            print(f"{key}: total {len(all_split_dict[key])} tars")
        self.all_split_dict = all_split_dict

        if self.train_dataset_config is not None:
            assert len(self.all_split_dict.get("train", [])) > 0
        if self.valid_dataset_config is not None:
            assert len(self.all_split_dict.get("valid", [])) > 0
        if self.test_dataset_config is not None:
            assert len(self.all_split_dict.get("test", [])) > 0
        if self.predict_dataset_config is not None:
            assert len(self.all_split_dict.get("predict", [])) > 0

    def prepare_data(self, world_size: int = None) -> None:
        """
        Only called by processes with local_rank = 0.
        """
        pass

    def setup(self, stage: str) -> None:
        """
        We create dataset here.
        """
        if self.trainer is not None:
            global_rank = self.trainer.global_rank
        else:
            print("warning, self.trainer is None, set global rank = 0")
            global_rank = 0

        # train
        if self.train_dataset_config is None:
            self.train_dataset = None
        else:
            if self.train_dataset_class == "ObjIterableDataset":
                self.train_dataset = ObjIterableDataset(
                    shard_blobby_urls=self.all_split_dict["train"],
                    **self.train_dataset_config,
                )
            else:
                raise NotImplementedError
            print(f"(global rank = {global_rank}) finished getting train dataset")

        # valid
        if self.valid_dataset_config is None:
            self.val_dataset = None
        else:
            if self.valid_dataset_class == "ObjIterableDataset":
                self.val_dataset = ObjIterableDataset(
                    shard_blobby_urls=self.all_split_dict["valid"],
                    **self.valid_dataset_config,
                )
            else:
                raise NotImplementedError
            print(f"(global rank = {global_rank}) finished getting valid dataset")

        # test
        if self.test_dataset_config is None:
            self.test_dataset = None
        else:
            if self.test_dataset_class == "ObjIterableDataset":
                self.test_dataset = ObjIterableDataset(
                    shard_blobby_urls=self.all_split_dict["test"],
                    **self.test_dataset_config,
                )
            else:
                raise NotImplementedError
            print(f"(global rank = {global_rank}) finished getting test dataset")

        # test
        if self.predict_dataset_config is None:
            self.predict_dataset = None
        else:
            if self.predict_dataset_class == "ObjIterableDataset":
                self.predict_dataset = ObjIterableDataset(
                    shard_blobby_urls=self.all_split_dict["predict"],
                    **self.predict_dataset_config,
                )
            else:
                raise NotImplementedError
            print(f"(global rank = {global_rank}) finished getting predict dataset")


if __name__ == "__main__":
    dset = ObjIterableDataset(
        shard_blobby_urls=[
            "/Users/jenhao_chang/Research/shape_tokenization/data/TRELLIS500KObjaverse_tar/57055aad-dc75-42c1-b1d8-290e25a870bf.tar",
        ],
        batch_size=2,
        mode="all",
    )

    dset_iter = iter(dset)
    data = next(dset_iter)
    print(data)
