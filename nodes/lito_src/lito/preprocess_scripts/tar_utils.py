#
# Copyright (C) 2026 Apple Inc. All rights reserved.
#
# The file implements util functions to tar samples for webdataset.


import json
import os
from timeit import default_timer as timer
import typing as T

import numpy as np

import torch

from plibs import structures, utils


def erode2d(
    mask: torch.Tensor,  # (b, q, h, w) bool
    k: int = 3,
    iters: int = 1,
):
    """
    Args:
        mask:
            (b, q, h, w) bool
        k:
            odd kernel size

    Returns:
        (b, q, h, w) bool
    """
    assert k % 2 == 1
    b, q, h, w = mask.shape
    m = mask.float().reshape(b * q, 1, h, w)  # (bq, 1, h, w)
    all_one_kernel = torch.ones((1, 1, k, k), device=m.device, dtype=m.dtype)
    pad = k // 2
    need = float(k * k)

    for _ in range(iters):
        s = torch.nn.functional.conv2d(m, all_one_kernel, padding=pad)  # (bq, 1, h, w)
        m = (s == need).float()  # erosion: all ones in neighborhood

    m = m.reshape(b, q, h, w)
    return m > 0.5  # (b, q, h, w) bool


def save_one_sample_from_rendered_rgbds(
    data_dir: str,
    out_dir: str,
    uid: str,
    new_uid: str,
    num_points: int,
    num_random_views: int,
    num_cond_views: int,
    num_gen_eval_views: int,
    keep_normal: bool = False,
    min_num_points: int = 3_000_000,
    rgbd_save_format: str = "png",
    seed: int = None,
):
    """
    Given a local dir containing rendered rgbd images,
    create a sample and save it to out_dir in the format
    that can be used by webdataset.

    Args:
        data_dir:
            str, the folder containing:
                - rgbd_regular
                - rgbd_random
                - rgbd_cond
                - rgbd_gen_eval
        out_dir:
            str, the folder where to save the sample
        uid:
            str, the original uid of the sample (ie, mesh)
        new_uid:
            str, the new uid of the sample, eg, {uid}_r0000
        num_points:
            int, number of points to sample from rgbd_regular images
        num_random_views / num_cond_views / num_gen_eval_views:
            int, number of views to randomly select from the corresponding rgbd images
        keep_normal:
            bool, whether to keep normal in the tar
        min_num_points:
            int, minimum number of valid points to consider the sample valid.
        rgbd_save_format:
            str, "png", "qoi" (faster but no preview on mac)

    Returns:

    Notes:
        The function raises error if the sample has insufficient number of points.

    """

    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    os.makedirs(out_dir, exist_ok=True)

    # move individual files to out_dir
    info_dict = dict()
    if num_points is not None and num_points > 0:
        # randomly select points for input
        index_filename = os.path.join(data_dir, "rgbd_regular", "index.json")
        rgbd = structures.RGBDImage.load_from(index_filename)

        # there are multiple considerations:
        # 1) if we use hit_map = hit_map && alpha> 0.99, we lose points on transparent surfaces.
        # 2) if we erode, we might lose points on transparent surfaces
        _hit_map = erode2d(rgbd.hit_map, k=3, iters=1)  # (b, q, h, w) bool

        other_maps = [rgbd.rgb, rgbd.other_maps["alpha"]]
        if keep_normal:
            other_maps.append(rgbd.normal_w)

        # check for nan
        _hit_map = torch.isfinite(_hit_map) & torch.isfinite(rgbd.depth)
        for om in other_maps:
            finite_map = torch.all(torch.isfinite(om), dim=-1)  # (b, q, h, w)
            assert len(finite_map.shape) == 4
            _hit_map = _hit_map & finite_map

        pdict = utils.compute_xyz_w_and_select_random_points(
            z_map=rgbd.depth,  # (b, q, h, w)
            hit_map=_hit_map,  # (b, q, h, w)
            intrinsic=rgbd.camera.intrinsic,  # (b, q, 3, 3)
            H_c2w=rgbd.camera.H_c2w,  # (b, q, 4, 4)
            num_points=num_points,
            other_maps=other_maps,
            return_pinhole_w=False,
            return_pinhole_idx=True,
        )
        point_xyz_w = pdict["xyz_w"]  # (b, num_points, 3xyz_w)
        point_rgb = pdict["other_maps"][0]  # (b, num_points, 3rgb) [0, 1]
        point_alpha = pdict["other_maps"][1]  # (b, num_points, 1) [0, 1]
        point_normal_w = pdict["other_maps"][2] if keep_normal else None  # (b, num_points, xyz_w)
        point_pinhole_idx = pdict["pinhole_idx"]  # (b, num_points, 1) long
        point_valid_len = pdict["valid_seq_lens"]  # (b,)

        # if can compress point pinhole idx to uint8, do it
        if point_pinhole_idx.max() <= 255:
            point_pinhole_idx = point_pinhole_idx.to(dtype=torch.uint8)

        # save points
        point_info = dict()
        z_valid_len = point_valid_len[0]  # int

        assert z_valid_len >= min_num_points, f"{uid} has insufficient points ({z_valid_len}). skipping"
        point_dict = dict(
            xyz_w=point_xyz_w[0, :z_valid_len],  # (n, 3)
            rgb=(point_rgb[0, :z_valid_len] * 255).to(dtype=torch.uint8),  # (n, 3) [0, 255] uint8
            alpha=(point_alpha[0, :z_valid_len] * 255).to(dtype=torch.uint8),  # (n, 1) [0, 255] uint8
            pinhole_idx=point_pinhole_idx[0, :z_valid_len],  # (n, 1) uint8
        )
        if keep_normal:
            point_dict["normal_w"] = point_normal_w[0, :z_valid_len]

        # save individual point attributes as npy
        for key in point_dict:
            if point_dict[key] is not None:
                point_info[key] = f"point.{key}.npy"
                filename = os.path.join(out_dir, f"{new_uid}.{point_info[key]}")
                arr = point_dict[key].detach().cpu().numpy()
                assert len(arr.shape) == 2
                np.save(
                    filename,
                    arr=arr,  # (num_vertices, 3xyz_w)
                )
            else:
                point_info[key] = None

        # save pinhole info to output
        pinhole_w = rgbd.camera.H_c2w[0, :, :3, 3].cpu().numpy()  # (q, 3xyz_w)
        np.save(os.path.join(out_dir, f"{new_uid}.pinhole_w_for_points"), pinhole_w)
        point_info["pinhole_w_for_points"] = "pinhole_w_for_points.npy"

        info_dict["point_info"] = point_info

    def load_rgbd(name: str, num_views: int):
        if num_views is None or num_views > 0:  # only save rgbd info
            index_filename = os.path.join(data_dir, name, "index.json")
            rgbd = structures.RGBDImage.load_from(index_filename)
            if not keep_normal:
                rgbd.normal_w = None
            if num_views is not None:
                total_views = rgbd.rgb.shape[1]
                random_qidxs = rng.permutation(total_views)[:num_views]
                random_qidxs = random_qidxs.tolist()
                rgbd = rgbd.index_select(
                    dim=1,
                    index=torch.tensor(random_qidxs, dtype=torch.long),
                )
            multiview_info = rgbd.save_as_flat(
                out_dir=out_dir,
                prefix=f"{new_uid}.{name}",
                overwrite=True,
                mode=rgbd_save_format,
                background_color=0,
                save_attr_names=[
                    "rgb",
                    "depth",
                    "normal_w",
                    "hit_map",
                    "alpha",
                    "obj_id",
                ],
            )
            info_dict[f"{name}_info"] = dict(
                index=f"{name}.index.json",
                name=name,
                b=multiview_info["b"],
                q=multiview_info["q"],
                h=multiview_info["h"],
                w=multiview_info["w"],
            )

    load_rgbd("rgbd_random", num_random_views)
    load_rgbd("rgbd_cond", num_cond_views)
    load_rgbd("rgbd_gen_eval", num_gen_eval_views)

    # save info_dict
    filename = os.path.join(out_dir, f"{new_uid}.sample_index.json")
    with open(filename, "w") as f:
        json.dump(info_dict, f, indent=2)
