#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#


import json
import typing as T

from lpips import LPIPS
import numpy as np
import skimage
import skimage.metrics

import torch

from plibs import json_utils, structures, utils

# def psnr(
#         rgb: torch.Tensor,
#         gts: torch.Tensor,
# ) -> float:
#     """
#     Calculate the PSNR metric. Non-differentiable.
#
#     Args:
#         rgb: (h, w, 3), in the range of [0, 1]
#         gts: (h, w, 3), in the range of [0, 1]
#
#     Returns:
#         psnr value
#     """
#     assert (rgb.shape[-1] == 3)
#     assert (gts.shape[-1] == 3)
#
#     mse = torch.mean((rgb[..., :3] - gts[..., :3]) ** 2).item()
#     return 10 * np.log10(1.0 / mse)


def get_lpips_model(device: torch.device("cpu")) -> torch.nn.Module:
    """
    Return lpips model
    """

    lpips_model = LPIPS(net="vgg").to(device=device)
    return lpips_model


def lpips_function(
    rgb: torch.Tensor,
    gts: torch.Tensor,
    lpips_model: torch.nn.Module = None,
) -> float:
    """
    Convenient function to call lpips library to calculate the LPIPS metric.
    Not differentiable.

    Args:
        rgb: (h, w, 3), in the range of [0, 1]
        gts: (h, w, 3), in the range of [0, 1]

    Returns:
        LPIPS value
    """
    assert rgb.shape[-1] == 3
    assert gts.shape[-1] == 3

    if lpips_model is None:
        lpips_model = LPIPS(net="vgg").to(device=rgb.device)

    return (
        lpips_model(
            (2.0 * rgb[..., :3] - 1.0).permute(2, 0, 1),
            (2.0 * gts[..., :3] - 1.0).permute(2, 0, 1),
        )
        .mean()
        .item()
    )


def ssim_function(
    rgb: torch.Tensor,
    gts: torch.Tensor,
) -> float:
    """
    Convenient function to call skimage's ssim.  Not differentiable.

    Args:
        rgb: (h, w, 3), in the range of [0, 1]
        gts: (h, w, 3), in the range of [0, 1]

    Returns:
        ssim value
    """
    # print(f'rgb.shape: {rgb.shape}, gts.shape: {gts.shape}')

    return skimage.metrics.structural_similarity(
        rgb[..., :3].cpu().numpy(),
        gts[..., :3].cpu().numpy(),
        multichannel=True,
        data_range=1,
        gaussian_weights=True,
        sigma=1.5,
        channel_axis=-1,  # need skimage >= 0.19
    )


def compute_mse(
    arr: torch.Tensor,
    ref: torch.Tensor,
    ndim_b: int = None,
    valid_mask: torch.Tensor = None,
):
    """
    Compute the mean squared error between arr and ref.
    The average is taken over the d_shape.

    Args:
        arr: (*b_shape, *d_shape)
        ref: (*b_shape, *d_shape)
        ndim_b:
            number of dimension of b_shape. If None, = 0.
        valid_mask: (*b_shape, *d_shape)

    Returns:
        mse: (*b_shape,)
    """
    if ndim_b is None:
        ndim_b = 0

    squared_error = (arr - ref) ** 2  # (*b, *d)
    squared_error = squared_error.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
    if valid_mask is None:
        mse = squared_error.mean(dim=-1)  # (*b,)
    else:
        valid_mask = valid_mask.view(*(valid_mask.shape), *([1] * (arr.ndim - valid_mask.ndim))).expand_as(arr)
        valid_mask = valid_mask.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
        mse = (squared_error * valid_mask).sum(dim=-1) / valid_mask.sum(-1)
    return mse


def compute_rmse(
    arr: torch.Tensor,
    ref: torch.Tensor,
    ndim_b: int = None,
    valid_mask: torch.Tensor = None,
):
    """
    Compute the root mean squared error between arr and ref.
    The average is taken over the d_shape.

    Args:
        arr: (*b_shape, *d_shape)
        ref: (*b_shape, *d_shape)
        ndim_b:
            number of dimension of b_shape. If None, = 0.
        valid_mask: (*b_shape, *d_shape)

    Returns:
        mse: (*b_shape,)
    """
    mse = compute_mse(arr=arr, ref=ref, ndim_b=ndim_b, valid_mask=valid_mask)  # (*b,)
    rmse = mse**0.5  # (*b,)
    return rmse


def compute_psnr(
    arr: torch.Tensor,
    ref: torch.Tensor,
    ndim_b: int = None,
    max_val: float = 1.0,
    valid_mask: torch.Tensor = None,
):
    """
    Compute peak signal to noise ratio
    Args:
        arr: (*b_shape, *d_shape)
        ref: (*b_shape, *d_shape)
        ndim_b:
            number of dimension of b_shape. If None, = 0.

    Returns:
        psnr: (*b_shape,)
    """

    mse = compute_mse(arr=arr, ref=ref, ndim_b=ndim_b, valid_mask=valid_mask)  # (*b,)
    psnr = 10 * torch.log10((max_val * max_val) / mse)  # (*b,)
    return psnr


def compute_ssim(
    arr: torch.Tensor,
    ref: torch.Tensor,
    ndim_b: int = None,
    # valid_mask: torch.Tensor = None,
):
    """
    Compute the ssim.
    Args:
        arr: (*b_shape, *d_shape)
        ref: (*b_shape, *d_shape)
        ndim_b:
            number of dimension of b_shape. If None, = 0.
        # valid_mask: (*b_shape, *d_shape)

    Returns:
        ssim_scores: (*b_shape,)

    Note:
        This function is NOT differentiable
    """
    if ndim_b is None:
        ndim_b = 0

    ori_shape = arr.shape
    b_shape = ori_shape[:ndim_b]
    d_shape = ori_shape[ndim_b:]
    arr = arr.reshape(-1, *d_shape)  # (b, *d)
    ref = ref.reshape(-1, *d_shape)  # (b, *d)
    b = arr.size(0)

    # print(f'ori_shape: {ori_shape}, b_shape: {b_shape}, d_shape: {d_shape}')

    assert len(d_shape) == 3
    assert d_shape[-1] >= 3

    ssim_scores = []
    for ib in range(b):
        # if valid_mask is None:
        ssim_score = ssim_function(rgb=arr[ib], gts=ref[ib])  # float
        ssim_scores.append(ssim_score)
    ssim_scores = torch.tensor(ssim_scores, dtype=torch.float, device=arr.device)
    ssim_scores = ssim_scores.reshape(*b_shape)
    return ssim_scores


def compute_lpips(
    arr: torch.Tensor,
    ref: torch.Tensor,
    ndim_b: int = None,
    lpips_model: torch.nn.Module = None,
    device: torch.device = torch.device("cuda"),
):
    """
    Compute the LPIPS metric.
    Args:
        arr: (*b_shape, *d_shape)  Assumes the RGB image is in [0,1]
        ref: (*b_shape, *d_shape)  Assumes the RGB image is in [0,1]
        ndim_b:
            number of dimension of b_shape. If None, = 0.

    Returns:
        lpips_score: (*b_shape,)

    Note:
        This function is NOT differentiable
    """
    if ndim_b is None:
        ndim_b = 0

    ori_shape = arr.shape
    b_shape = ori_shape[:ndim_b]
    d_shape = ori_shape[ndim_b:]
    arr = arr.reshape(-1, *d_shape)  # (b, *d)
    ref = ref.reshape(-1, *d_shape)  # (b, *d)
    b = arr.size(0)

    arr_device = arr.device

    arr = arr.to(device=device)
    ref = ref.to(dtype=arr.dtype, device=device)

    assert len(d_shape) == 3
    assert d_shape[-1] >= 3

    scores = []
    for ib in range(b):
        score = lpips_function(rgb=arr[ib], gts=ref[ib], lpips_model=lpips_model)  # float
        scores.append(score)
    scores = torch.tensor(scores, dtype=torch.float, device=arr_device)
    scores = scores.reshape(*b_shape)
    return scores.to(device=arr_device)


def compute_l1(
    arr: torch.Tensor,
    ref: torch.Tensor,
    ndim_b: int = None,
    valid_mask: torch.Tensor = None,
):
    """
    Compute average l1 distance between arr and ref.

    Args:
        arr: (*b_shape, *d_shape)
        ref: (*b_shape, *d_shape)
        ndim_b:
            number of dimension of b_shape. If None, = 0.
        valid_mask: (*b_shape, *d_shape)

    Returns:
        err: (*b_shape,)
    """
    if ndim_b is None:
        ndim_b = 0

    err = (arr - ref).abs()  # (*b, *d)
    err = err.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
    if valid_mask is None:
        err = err.mean(dim=-1)  # (*b,)
    else:
        valid_mask = valid_mask.view(*(valid_mask.shape), *([1] * (arr.ndim - valid_mask.ndim))).expand_as(arr)
        valid_mask = valid_mask.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
        err = (err * valid_mask).sum(dim=-1) / valid_mask.sum(-1)

    return err


def compute_area(
    arr: torch.Tensor,
    ref: torch.Tensor,
    ndim_b: int = None,
    normalized: bool = True,
    valid_mask: torch.Tensor = None,
):
    """
    Compute the area spanned by the unit vectors in arr and ref.

    Args:
        arr: (*b_shape, *d_shape, 3)
        ref: (*b_shape, *d_shape, 3)
        ndim_b:
            number of dimension of b_shape. If None, = 0.
        normalized:
            whetehr arr and ref are unit vectors
        valid_mask: (*b_shape, *d_shape,)

    Returns:
        err: (*b_shape,)
    """
    if ndim_b is None:
        ndim_b = 0

    if not normalized:
        arr = torch.nn.functional.normalize(arr, p=2, dim=-1)
        ref = torch.nn.functional.normalize(ref, p=2, dim=-1)

    out = torch.linalg.cross(arr, ref, dim=-1)  # (*b, *d, 3)
    area = torch.linalg.vector_norm(out, ord=2, dim=-1)  # (*b, *d,)

    if valid_mask is None:
        area = area.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
        area = area.mean(dim=-1)  # (*b,)
    else:
        valid_mask = valid_mask.view(*(valid_mask.shape), *([1] * (area.ndim - valid_mask.ndim))).expand_as(area)
        valid_mask = valid_mask.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
        area = area.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
        area = (area * valid_mask).sum(dim=-1) / valid_mask.sum(-1)

    return area


def compute_diff_angle(
    arr: torch.Tensor,
    ref: torch.Tensor,
    ndim_b: int = None,
    normalized: bool = True,
    valid_mask: torch.Tensor = None,
):
    """
    Compute the angle spanned by the unit vectors in arr and ref.

    Args:
        arr: (*b_shape, *d_shape, 3)
        ref: (*b_shape, *d_shape, 3)
        ndim_b:
            number of dimension of b_shape. If None, = 0.
        normalized:
            whetehr arr and ref are unit vectors
        valid_mask: (*b_shape, *d_shape,)

    Returns:
        err: (*b_shape,) angle in degree
    """
    if ndim_b is None:
        ndim_b = 0

    if not normalized:
        arr = torch.nn.functional.normalize(arr, p=2, dim=-1)
        ref = torch.nn.functional.normalize(ref, p=2, dim=-1)

    # make sure arr and ref points to the same direction
    out = torch.sum(arr * ref, dim=-1)  # (*b, *d)
    arr = arr * out.sign().unsqueeze(-1)

    # recompute inner product
    out = torch.sum(arr * ref, dim=-1)  # (*b, *d)

    angle = torch.arccos(out.clamp(min=-1 + 1e-9, max=1 - 1e-9)) * (180.0 / torch.pi)  # (*b, *d) in degree
    if valid_mask is None:
        angle = angle.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
        angle = angle.mean(dim=-1)  # (*b,)
    else:
        valid_mask = valid_mask.view(*(valid_mask.shape), *([1] * (angle.ndim - valid_mask.ndim))).expand_as(angle)
        valid_mask = valid_mask.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
        angle = angle.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
        angle = (angle * valid_mask).sum(dim=-1) / valid_mask.sum(-1)
    return angle


def compute_accuracy(
    arr: torch.Tensor,
    ref: torch.Tensor,
    ndim_b: int = None,
    valid_mask: torch.Tensor = None,
):
    """
    Compute the accuracy.

    Args:
        arr: (*b_shape, *d_shape), binary
        ref: (*b_shape, *d_shape), binary
        ndim_b:
            number of dimension of b_shape. If None, = 0.
        valid_mask: (*b_shape, *d_shape,)

    Returns:
        acc: (*b_shape,)
    """
    if ndim_b is None:
        ndim_b = 0

    same = (arr > 0.5) == (ref > 0.5)  # (*b, *d)
    same = same.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
    if valid_mask is None:
        acc = same.float().mean(dim=-1)  # (*b,)
    else:
        valid_mask = valid_mask.view(*(valid_mask.shape), *([1] * (arr.ndim - valid_mask.ndim))).expand_as(arr)
        valid_mask = valid_mask.reshape(*(arr.shape[:ndim_b]), -1)  # (*b, numel_d)
        acc = (same.float() * valid_mask).sum(dim=-1) / valid_mask.sum(-1)
    return acc


def compute_metrics_for_rgbd_images(
    rgbd_images: T.List[structures.RGBDImage],
    ref_rgbd_image: structures.RGBDImage,
    names: T.List[str] = None,
    rgb_metric: T.List[str] = "psnr",
    depth_metric: T.List[str] = "rmse",
    normal_metric: T.List[str] = "avg_angle",
    hit_metric: T.List[str] = "accuracy",
    output_filename: str = None,
) -> T.Dict[str, T.Any]:
    """
    Compare a list of rgbd_images generated by different candidates,
    create gif, compute difference from ref_rgbd_image if given.
    Note we assume the camera used to capture the rgbd images are the same.

    Args:
        rgbd_images:
            (num_candidates, ) list of rgbd_image (b, q, h, w) to compare.
        ref_rgbd_image:
            (b, q, h, w) reference rgbd image to compute the error against
        names:
            (num_candidates, ) name of the rgbd_images. If None, it will become their indexes.
        rgb_metric:
            'psnr', 'ssim', 'lpips'
        depth_metric:
            'rmse',
        normal_metric:
            'avg_angle'
        hit_metric:
            'accuracy'

    Returns:
        rgb_err_dicts:
            name or index in list (str) -> error dict for rgb (metric_name -> val (b,q)).
            err_dict will be None if input/gt not presented.
            Additionally, both mean and std over (b, q) are recorded:
            f'avg_{metric_name}' = err.mean() f'std_{metric_name}' = err.std()
        depth_err_dicts:
            name or index in list (str) -> error dict for deoth (metric_name -> val (b,q)).
            err_dict will be None if input/gt not presented.
            Additionally, both mean and std over (b, q) are recorded:
            f'avg_{metric_name}' = err.mean() f'std_{metric_name}' = err.std()
        normal_err_dicts:
            name or index in list (str) -> error dict for normal (metric_name -> val (b,q)).
            err_dict will be None if input/gt not presented.
            Additionally, both mean and std over (b, q) are recorded:
            f'avg_{metric_name}' = err.mean() f'std_{metric_name}' = err.std()
        hit_err_dicts:
            name or index in list (str) -> error dict for hit (metric_name -> val (b,q)).
            err_dict will be None if input/gt not presented.
            Additionally, both mean and std over (b, q) are recorded:
            f'avg_{metric_name}' = err.mean() f'std_{metric_name}' = err.std()

    Procedure:
        - before adding the name to the image, compute the error to the reference
        - create tmp rgb, depth, normal_w, hit_map if not None.  If one content is None, skip the image
    """
    assert ref_rgbd_image is not None

    if isinstance(rgb_metric, str):
        rgb_metric = [rgb_metric]
    if isinstance(depth_metric, str):
        depth_metric = [depth_metric]
    if isinstance(normal_metric, str):
        normal_metric = [normal_metric]
    if isinstance(hit_metric, str):
        hit_metric = [hit_metric]

    if names is None or len(names) == 0:
        names = [f"{i}" for i in range(len(rgbd_images))]
    assert len(names) == len(rgbd_images)

    # get hit map (to identify valid pixels)
    hit_maps = [rgbd.hit_map for rgbd in rgbd_images]  # (b, q, h, w,)

    # # debug
    # for rgbd in rgbd_images:
    #     print(f'rgbd.rgb..shape = {rgbd.rgb.shape}')
    #     print(f'rgbd.normal_w.shape = {rgbd.normal_w.shape}')
    #     print(f'rgbd.depth.shape = {rgbd.depth.shape}')
    #     print(f'rgbd.hit_map.shape = {rgbd.hit_map.shape}')
    #
    # print(f'ref_rgbd.rgb..shape = {ref_rgbd_image.rgb.shape}')
    # print(f'ref_rgbd.normal_w.shape = {ref_rgbd_image.normal_w.shape}')
    # print(f'ref_rgbd.depth.shape = {ref_rgbd_image.depth.shape}')
    # print(f'ref_rgbd.hit_map.shape = {ref_rgbd_image.hit_map.shape}')
    # # end debug

    # compute errors
    # rgb: (b, q, h, w, 3)
    rgbs = [rgbd.rgb for rgbd in rgbd_images]  # (b, q, h, w, 3)
    gt = ref_rgbd_image.rgb  # (b, q, h, w, 3)
    gt_hit_map = ref_rgbd_image.hit_map  # (b, q, h, w)
    if gt_hit_map is None:
        gt_hit_map = torch.ones_like(gt[..., 0])  # (b, q, h, w)

    rgb_err_dicts = dict()  # a list containing the err_dict for each input rgbd_img
    for i in range(len(rgbs)):
        arr = rgbs[i]
        if arr is None or gt is None:
            rgb_err_dicts[names[i]] = None
            continue
        hit_map = hit_maps[i] if hit_maps[i] is not None else torch.ones(*arr.shape[:-1], device=arr.device)

        valid_mask = torch.logical_and(gt_hit_map, hit_map)
        err_dict = dict()
        if "lpips" in rgb_metric:
            lpips_device = torch.device("cuda")
            lpips_model = get_lpips_model(device=lpips_device)
        else:
            lpips_model = None
            lpips_device = None
        for metric_name in rgb_metric:
            if metric_name == "psnr":
                err = compute_psnr(
                    arr=arr * hit_map.unsqueeze(-1) + (1 - hit_map.float()).unsqueeze(-1).expand_as(arr),
                    ref=gt * gt_hit_map.unsqueeze(-1) + (1 - gt_hit_map.float()).unsqueeze(-1).expand_as(gt),
                    # assume background is white
                    ndim_b=2,
                    max_val=1.0,
                    valid_mask=None,  # calculate the full image to be fair to all methods
                    # valid_mask=valid_mask,  # calculate only on the valid hit region to be fair to all methods
                )  # (b, q)
                err_dict[metric_name] = err  # (b, q)
                err = err[torch.isfinite(err)]
                err_dict[f"avg_{metric_name}"] = err.mean()
                err_dict[f"std_{metric_name}"] = err.std()
            elif metric_name == "ssim":
                err = compute_ssim(
                    arr=arr * hit_map.unsqueeze(-1) + (1 - hit_map.float()).unsqueeze(-1).expand_as(arr),
                    ref=gt * gt_hit_map.unsqueeze(-1) + (1 - gt_hit_map.float()).unsqueeze(-1).expand_as(gt),
                    # assume background is white
                    ndim_b=2,
                )  # (b, q)
                err_dict[metric_name] = err  # (b, q)
                err = err[torch.isfinite(err)]
                err_dict[f"avg_{metric_name}"] = err.mean()
                err_dict[f"std_{metric_name}"] = err.std()

            elif metric_name == "lpips":
                err = compute_lpips(
                    arr=arr * hit_map.unsqueeze(-1) + (1 - hit_map.float()).unsqueeze(-1).expand_as(arr),
                    ref=gt * gt_hit_map.unsqueeze(-1) + (1 - gt_hit_map.float()).unsqueeze(-1).expand_as(gt),
                    # assume background is white
                    ndim_b=2,
                    lpips_model=lpips_model,
                    device=lpips_device,
                )  # (b, q)
                err_dict[metric_name] = err  # (b, q)
                err = err[torch.isfinite(err)]
                err_dict[f"avg_{metric_name}"] = err.mean()
                err_dict[f"std_{metric_name}"] = err.std()
            else:
                raise NotImplementedError

        rgb_err_dicts[names[i]] = err_dict

    # depth: (b, q, h, w)
    device_chamfer = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    depths = [rgbd.depth for rgbd in rgbd_images]  # list of (b, q, h, w,)
    rays = [rgbd.camera.generate_camera_rays() for rgbd in rgbd_images]  # list of (b, q, h, w)
    gt = ref_rgbd_image.depth  # (b, q, h, w)
    depth_err_dicts = dict()  # a list containing the err_dict for each input rgbd_img
    for i in range(len(depths)):
        arr = depths[i]  # (b, q, h, w,)
        ray = rays[i]  # (b, q, h, w)
        if arr is None or gt is None:
            depth_err_dicts[names[i]] = None
            continue
        hit_map = (
            hit_maps[i] if hit_maps[i] is not None else torch.ones(*arr.shape, dtype=torch.bool, device=arr.device)
        )
        # valid_mask = torch.logical_or(gt_hit_map, hit_map)
        valid_mask = torch.logical_and(gt_hit_map, hit_map)
        err_dict = dict()
        for metric_name in depth_metric:
            if metric_name in err_dict:
                continue

            if metric_name == "rmse":
                err = compute_rmse(
                    arr=arr * hit_map,
                    ref=gt * gt_hit_map,
                    ndim_b=2,
                    valid_mask=valid_mask,
                )  # (b, q)
                err_dict[metric_name] = err  # (b, q)
                err = err[torch.isfinite(err)]
                err_dict[f"avg_{metric_name}"] = err.mean()
                err_dict[f"std_{metric_name}"] = err.std()

            elif metric_name in {
                "chamfer_est2gt",
                "chamfer_gt2est",
                "chamfer_symmetric",
                "silhouette_chamfer_est2gt",
                "silhouette_chamfer_gt2est",
                "silhouette_chamfer_symmetric",
                "valid_chamfer_est2gt",
                "valid_chamfer_gt2est",
                "valid_chamfer_symmetric",
            }:
                from chamferdist import ChamferDistance

                chamferDist = ChamferDistance()

                # we compute the distance for each view (q) separately and then merge
                # silhouette uses the gt_hip_map (not the estimated hit_map)
                all_dist_unmasked_est2gts = []
                all_dist_unmasked_gt2ests = []
                all_dist_unmasked_symmetrics = []
                all_dist_masked_est2gts = []
                all_dist_masked_gt2ests = []
                all_dist_masked_symmetrics = []
                all_dist_valid_est2gts = []
                all_dist_valid_gt2ests = []
                all_dist_valid_symmetrics = []
                for ib in range(arr.size(0)):
                    dist_unmasked_est2gts = []
                    dist_unmasked_gt2ests = []
                    dist_unmasked_symmetrics = []
                    dist_masked_est2gts = []
                    dist_masked_gt2ests = []
                    dist_masked_symmetrics = []
                    dist_valid_est2gts = []
                    dist_valid_gt2ests = []
                    dist_valid_symmetrics = []
                    for iq in range(arr.size(1)):
                        arr_q = arr[ib, iq]  # (h, w)
                        gt_q = gt[ib, iq]  # (h, w)

                        point_q_w = ray.origins_w[ib, iq] + arr_q.unsqueeze(-1) * ray.directions_w[ib, iq]  # (h, w, 3)
                        # print(f'point_q_w.shape = {point_q_w.shape}')
                        point_q_w = point_q_w.reshape(-1, 3)  # (hw, 3)
                        gt_q_w = ray.origins_w[ib, iq] + gt_q.unsqueeze(-1) * ray.directions_w[ib, iq]  # (h, w, 3)
                        # print(f'gt_q_w.shape = {gt_q_w.shape}')
                        gt_q_w = gt_q_w.reshape(-1, 3)  # (hw, 3)

                        est_hit_map_q = hit_map[ib, iq].reshape(-1)  # (hw)
                        total_est_hit_map_q = est_hit_map_q.sum().clamp(min=1)
                        # print(f'est_hit_map_q.shape = {est_hit_map_q}')
                        gt_hit_map_q = gt_hit_map[ib, iq].reshape(-1)  # (hw)
                        total_gt_hit_map_q = gt_hit_map_q.sum().clamp(min=1)
                        # print(f'gt_hit_map_q.shape = {gt_hit_map_q}')
                        valid_mask_q = valid_mask[ib, iq].reshape(-1)  # (hw)
                        total_valid_mask_q = valid_mask_q.sum().clamp(min=1)
                        # print(f'valid_mask_q.shape = {valid_mask_q}')

                        # unmasked chamfer: est -> gt
                        dist_unmasked_est2gt = (
                            chamferDist(
                                point_q_w[est_hit_map_q].unsqueeze(0).to(device_chamfer),  # (1, n_est, 3)
                                gt_q_w[gt_hit_map_q].unsqueeze(0).to(device_chamfer),  # (1, n_gt, 3)
                            )
                            .detach()
                            .cpu()
                        )  # (,)
                        dist_unmasked_est2gt = dist_unmasked_est2gt / total_est_hit_map_q
                        # unmasked chamfer: gt -> est
                        dist_unmasked_gt2est = (
                            chamferDist(
                                gt_q_w[gt_hit_map_q].unsqueeze(0).to(device_chamfer),  # (1, n_gt, 3)
                                point_q_w[est_hit_map_q].unsqueeze(0).to(device_chamfer),  # (1, n_est, 3)
                            )
                            .detach()
                            .cpu()
                        )  # (,)
                        dist_unmasked_gt2est = dist_unmasked_gt2est / total_gt_hit_map_q
                        dist_unmasked_symmetric = dist_unmasked_est2gt + dist_unmasked_gt2est

                        # silhouette_masked chamfer: est -> gt
                        dist_masked_est2gt = (
                            chamferDist(
                                point_q_w[gt_hit_map_q].unsqueeze(0).to(device_chamfer),  # (1, n_est, 3)
                                gt_q_w[gt_hit_map_q].unsqueeze(0).to(device_chamfer),  # (1, n_gt, 3)
                            )
                            .detach()
                            .cpu()
                        )  # (,)
                        dist_masked_est2gt = dist_masked_est2gt / total_gt_hit_map_q
                        # silhouette_masked chamfer: gt -> est
                        dist_masked_gt2est = (
                            chamferDist(
                                gt_q_w[gt_hit_map_q].unsqueeze(0).to(device_chamfer),  # (1, n_gt, 3)
                                point_q_w[gt_hit_map_q].unsqueeze(0).to(device_chamfer),  # (1, n_est, 3)
                            )
                            .detach()
                            .cpu()
                        )  # (,)
                        dist_masked_gt2est = dist_masked_gt2est / total_gt_hit_map_q
                        dist_masked_symmetric = dist_masked_est2gt + dist_masked_gt2est

                        # both_masked chamfer: est -> gt
                        dist_valid_est2gt = (
                            chamferDist(
                                point_q_w[valid_mask_q].unsqueeze(0).to(device_chamfer),  # (1, n_est, 3)
                                gt_q_w[valid_mask_q].unsqueeze(0).to(device_chamfer),  # (1, n_gt, 3)
                            )
                            .detach()
                            .cpu()
                        )  # (,)
                        dist_valid_est2gt = dist_valid_est2gt / total_valid_mask_q
                        # both_masked chamfer: gt -> est
                        dist_valid_gt2est = (
                            chamferDist(
                                gt_q_w[valid_mask_q].unsqueeze(0).to(device_chamfer),  # (1, n_gt, 3)
                                point_q_w[valid_mask_q].unsqueeze(0).to(device_chamfer),  # (1, n_est, 3)
                            )
                            .detach()
                            .cpu()
                        )  # (,)
                        dist_valid_gt2est = dist_valid_gt2est / total_valid_mask_q
                        dist_valid_symmetric = dist_valid_est2gt + dist_valid_gt2est

                        dist_unmasked_est2gts.append(dist_unmasked_est2gt)
                        dist_unmasked_gt2ests.append(dist_unmasked_gt2est)
                        dist_unmasked_symmetrics.append(dist_unmasked_symmetric)
                        dist_masked_est2gts.append(dist_masked_est2gt)
                        dist_masked_gt2ests.append(dist_masked_gt2est)
                        dist_masked_symmetrics.append(dist_masked_symmetric)
                        dist_valid_est2gts.append(dist_valid_est2gt)
                        dist_valid_gt2ests.append(dist_valid_gt2est)
                        dist_valid_symmetrics.append(dist_valid_symmetric)

                    dist_unmasked_est2gts = torch.stack(dist_unmasked_est2gts, dim=0)  # (q,)
                    dist_unmasked_gt2ests = torch.stack(dist_unmasked_gt2ests, dim=0)  # (q,)
                    dist_unmasked_symmetrics = torch.stack(dist_unmasked_symmetrics, dim=0)  # (q,)
                    dist_masked_est2gts = torch.stack(dist_masked_est2gts, dim=0)  # (q,)
                    dist_masked_gt2ests = torch.stack(dist_masked_gt2ests, dim=0)  # (q,)
                    dist_masked_symmetrics = torch.stack(dist_masked_symmetrics, dim=0)  # (q,)
                    dist_valid_est2gts = torch.stack(dist_valid_est2gts, dim=0)  # (q,)
                    dist_valid_gt2ests = torch.stack(dist_valid_gt2ests, dim=0)  # (q,)
                    dist_valid_symmetrics = torch.stack(dist_valid_symmetrics, dim=0)  # (q,)

                    all_dist_unmasked_est2gts.append(dist_unmasked_est2gts)
                    all_dist_unmasked_gt2ests.append(dist_unmasked_gt2ests)
                    all_dist_unmasked_symmetrics.append(dist_unmasked_symmetrics)
                    all_dist_masked_est2gts.append(dist_masked_est2gts)
                    all_dist_masked_gt2ests.append(dist_masked_gt2ests)
                    all_dist_masked_symmetrics.append(dist_masked_symmetrics)
                    all_dist_valid_est2gts.append(dist_valid_est2gts)
                    all_dist_valid_gt2ests.append(dist_valid_gt2ests)
                    all_dist_valid_symmetrics.append(dist_valid_symmetrics)

                all_dist_unmasked_est2gts = torch.stack(all_dist_unmasked_est2gts, dim=0)  # (b, q)
                all_dist_unmasked_gt2ests = torch.stack(all_dist_unmasked_gt2ests, dim=0)  # (b, q)
                all_dist_unmasked_symmetrics = torch.stack(all_dist_unmasked_symmetrics, dim=0)  # (b, q)
                all_dist_masked_est2gts = torch.stack(all_dist_masked_est2gts, dim=0)  # (b, q)
                all_dist_masked_gt2ests = torch.stack(all_dist_masked_gt2ests, dim=0)  # (b, q)
                all_dist_masked_symmetrics = torch.stack(all_dist_masked_symmetrics, dim=0)  # (b, q)
                all_dist_valid_est2gts = torch.stack(all_dist_valid_est2gts, dim=0)  # (b, q)
                all_dist_valid_gt2ests = torch.stack(all_dist_valid_gt2ests, dim=0)  # (b, q)
                all_dist_valid_symmetrics = torch.stack(all_dist_valid_symmetrics, dim=0)  # (b, q)

                for mname, err in [
                    ["chamfer_est2gt", all_dist_unmasked_est2gts],
                    ["chamfer_gt2est", all_dist_unmasked_gt2ests],
                    ["chamfer_symmetric", all_dist_unmasked_symmetrics],
                    ["silhouette_chamfer_est2gt", all_dist_masked_est2gts],
                    ["silhouette_chamfer_gt2est", all_dist_masked_gt2ests],
                    ["silhouette_chamfer_symmetric", all_dist_masked_symmetrics],
                    ["valid_chamfer_est2gt", all_dist_valid_est2gts],
                    ["valid_chamfer_gt2est", all_dist_valid_gt2ests],
                    ["valid_chamfer_symmetric", all_dist_valid_symmetrics],
                ]:
                    err_dict[mname] = err
                    tmp_err = err[torch.isfinite(err)]
                    err_dict[f"avg_{mname}"] = tmp_err.mean()
                    err_dict[f"std_{mname}"] = tmp_err.std()
            else:
                raise NotImplementedError

        depth_err_dicts[names[i]] = err_dict

    # normal_w: (b, q, h, w, 3)
    normal_ws = [rgbd.normal_w for rgbd in rgbd_images]  # (b, q, h, w, 3)
    gt = ref_rgbd_image.normal_w
    normal_err_dicts = dict()  # a list containing the err_dict for each input rgbd_img
    for i in range(len(normal_ws)):
        arr = normal_ws[i]
        if arr is None or gt is None:
            normal_err_dicts[names[i]] = None
            continue
        hit_map = hit_maps[i] if hit_maps[i] is not None else torch.ones(*arr.shape[:-1], device=arr.device)
        # valid_mask = torch.logical_or(gt_hit_map, hit_map)
        valid_mask = torch.logical_and(gt_hit_map, hit_map)
        err_dict = dict()
        for metric_name in normal_metric:
            if metric_name == "avg_angle":
                err = compute_diff_angle(
                    arr=arr * hit_map.unsqueeze(-1),
                    ref=gt * gt_hit_map.unsqueeze(-1),
                    ndim_b=2,
                    normalized=False,
                    valid_mask=valid_mask,
                )  # (b, q)
                err_dict[metric_name] = err  # (b, q)
                err = err[torch.isfinite(err)]
                err_dict[f"avg_{metric_name}"] = err.mean()
                err_dict[f"std_{metric_name}"] = err.std()
            else:
                raise NotImplementedError

        normal_err_dicts[names[i]] = err_dict

    # hit_map: (b, q, h, w,)
    # hit_maps = [rgbd.hit_map for rgbd in rgbd_images]  # (b, q, h, w, 3)
    gt = ref_rgbd_image.hit_map
    hit_err_dicts = dict()  # a list containing the err_dict for each input rgbd_img
    for i in range(len(hit_maps)):
        arr = hit_maps[i]
        if arr is None or gt is None:
            hit_err_dicts[names[i]] = None
            continue

        err_dict = dict()
        for metric_name in hit_metric:
            if metric_name == "accuracy":
                err = compute_accuracy(
                    arr=arr,
                    ref=gt,
                    ndim_b=2,
                    valid_mask=None,
                )  # (b, q)
                err_dict[metric_name] = err  # (b, q)
                err = err[torch.isfinite(err)]
                err_dict[f"avg_{metric_name}"] = err.mean()
                err_dict[f"std_{metric_name}"] = err.std()
            else:
                raise NotImplementedError

        hit_err_dicts[names[i]] = err_dict

    out_dict = dict(
        rgb_err_dicts=rgb_err_dicts,
        depth_err_dicts=depth_err_dicts,
        normal_err_dicts=normal_err_dicts,
        hit_err_dicts=hit_err_dicts,
    )

    if output_filename is not None:
        with open(output_filename, "w") as f:
            json.dump(
                utils.to_numpy(out_dict),
                f,
                indent=2,
                cls=json_utils.NumpyJsonEncoder,
            )

    return out_dict


def compute_pointersect_metrics(
    est_rgb: T.Optional[torch.Tensor] = None,
    gt_rgb: T.Optional[torch.Tensor] = None,
    est_ray_t: T.Optional[torch.Tensor] = None,
    gt_ray_t: T.Optional[torch.Tensor] = None,
    est_normal_w: T.Optional[torch.Tensor] = None,
    gt_normal_w: T.Optional[torch.Tensor] = None,
    est_hit: T.Optional[torch.Tensor] = None,
    gt_hit: T.Optional[torch.Tensor] = None,
    valid_mask: T.Optional[torch.Tensor] = None,
):
    """
    Compute the pointersect metrics:
    psnr for rgb, rmse for ray_t, cos(theta) for normal, accuracy for hit

    Args:
        est_rgb:
            (*, 3rgb) [0, 1]
        gt_rgb:
            (*, 3rgb) [0, 1]
        est_ray_t:
            (*,)
        gt_ray_t:
            (*,)
        est_normal_w:
            (*, 3xyz_w)
        gt_normal_w:
            (*, 3xyz_w)
        est_hit:

        gt_hit:
        valid_mask:

    Returns:

    """
