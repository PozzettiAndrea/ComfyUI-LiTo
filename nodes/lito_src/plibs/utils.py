#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from __future__ import annotations  # type annotations stay as strings, never evaluated

import copy
import json
import math
import os
from timeit import default_timer as timer
import typing as T

import numpy as np
import skimage

import torch

# Optional heavy/platform-specific imports — only needed in helper paths the
# ComfyUI inference wrapper does not exercise.
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import open3d as o3d
except ImportError:
    o3d = None
try:
    from pygltflib.utils import glb2gltf
except ImportError:
    glb2gltf = None
try:
    from pytorch3d.renderer.points.pulsar.renderer import Renderer as PulsarRenderer
    from pytorch3d.renderer.points.rasterize_points import _RasterizePoints as RasterizePoints
except ImportError:
    PulsarRenderer = None
    RasterizePoints = None

from plibs import linalg_utils, rigid_motion
from contextlib import nullcontext as _nullcontext
pr_utils = None
render = None
imagesc = None


def to_tensor(
    arr: T.Union[np.ndarray, T.List[np.ndarray], T.Dict[str, T.Any]],
    dtype: torch.dtype = None,
) -> T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]]:
    """
    Convert each element in arr from np.ndarray to torch.Tensor.
    Note that the output share the same memory as arr.
    """
    if isinstance(arr, np.ndarray):
        if arr.dtype == object:
            # scalar object array (e.g. np.array(None, dtype=object))
            if arr.shape == () or arr.ndim == 0:
                return to_tensor(arr.item(), dtype=dtype)
            # element-wise convert
            return [to_tensor(x, dtype=dtype) for x in arr.tolist() if x is not None]
        if arr.size == 0:  # empty numpy array
            return torch.empty(0, dtype=dtype if dtype else torch.float32)
        arr = torch.from_numpy(arr)
        if dtype is not None:
            arr = arr.to(dtype=dtype)
        return arr
    elif isinstance(arr, torch.Tensor) and dtype is not None:
        arr = arr.to(dtype=dtype)
        return arr
    elif isinstance(arr, (list, tuple)):
        return [to_tensor(x, dtype=dtype) for x in arr]
    elif isinstance(arr, dict):
        out_dict = dict()
        for key, val in arr.items():
            out_dict[key] = to_tensor(val, dtype=dtype)
        return out_dict
    else:
        return arr


def to_numpy(
    arr: T.Union[np.ndarray, T.List[np.ndarray], T.Dict[str, T.Any]],
    dtype: np.dtype = None,
) -> T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]]:
    """
    Convert each element in arr from torch.Tensor to numpy ndarray.
    Note that the output share the same memory as arr if on cpu.
    """
    if isinstance(arr, torch.Tensor):
        if arr.dtype != torch.bfloat16:
            arr = arr.detach().cpu().numpy()
        else:
            arr = arr.detach().cpu().float().numpy()
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr
    elif isinstance(arr, np.ndarray) and dtype is not None:
        arr = arr.astype(dtype)
        return arr
    elif isinstance(arr, (list, tuple)):
        return [to_numpy(x, dtype=dtype) for x in arr]
    elif isinstance(arr, dict):
        out_dict = dict()
        for key, val in arr.items():
            out_dict[key] = to_numpy(val, dtype=dtype)
        return out_dict
    else:
        return arr


def detach_dict(
    arr: T.Union[None, torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]],
) -> T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any], None]:
    """
    detach each torch.Tensor in arr to device.
    """
    if arr is None:
        return None
    elif isinstance(arr, torch.Tensor):
        arr = arr.detach()
        return arr
    elif isinstance(arr, (list, tuple)):
        return [detach_dict(x) for x in arr]
    elif isinstance(arr, dict):
        for key, val in arr.items():
            arr[key] = detach_dict(val)
        return arr
    else:
        return arr


def clone_dict(
    arr: T.Union[None, torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]],
) -> T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any], None]:
    """
    clone each torch.Tensor in arr to device.
    """
    if arr is None:
        return None
    elif isinstance(arr, torch.Tensor):
        arr = arr.clone()
        return arr
    elif isinstance(arr, (list, tuple)):
        return [clone_dict(x) for x in arr]
    elif isinstance(arr, dict):
        for key, val in arr.items():
            arr[key] = clone_dict(val)
        return arr
    else:
        return arr


def to_device(
    arr: T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any], None],
    device: torch.device,
) -> T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any], None]:
    """
    Send each torch.Tensor in arr to device.
    """
    if arr is None:
        return None
    elif isinstance(arr, torch.Tensor):
        arr = arr.to(device=device)
        return arr
    elif isinstance(arr, (list, tuple)):
        return [to_device(x, device=device) for x in arr]
    elif isinstance(arr, dict):
        for key, val in arr.items():
            arr[key] = to_device(val, device=device)
        return arr
    else:
        return arr


def to_dtype(
    arr: T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]],
    dtype: torch.dtype,
) -> T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]]:
    """
    Send each torch.Tensor in arr to device.
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.to(dtype=dtype)
        return arr
    elif isinstance(arr, (list, tuple)):
        return [to_dtype(x, dtype=dtype) for x in arr]
    elif isinstance(arr, dict):
        for key, val in arr.items():
            arr[key] = to_dtype(val, dtype=dtype)
        return arr
    else:
        return arr


def get_subsample_idx(
    n: int,
    num_samples: int,
    repeat_if_not_enough: bool,
    device: torch.device = torch.device("cpu"),
    generator: torch.Generator = None,
):
    """
    Given the total size `n` and the number of samples needed,
    randomly choose from the samples

    Args:
        n:
            tensor shape along the direction to sample
        num_samples:
            number of samples to draw
        repeat_if_not_enough:
            if n < num_samples, whether to repeat or raise error

    Returns:
        (num_samples,)
    """

    if n == num_samples:
        return torch.arange(n, device=device)
    elif n > num_samples:
        ii = torch.randperm(n, generator=generator, device=device)[:num_samples]
        return ii
    elif n < num_samples:
        # for n < num_samples, we repeat n
        if not repeat_if_not_enough:
            raise RuntimeError(f"n = {n} < num_samples = {num_samples}")
        # num_repeats = (n + num_samples - 1) // num_samples
        num_repeats = (num_samples + n - 1) // n
        r = torch.arange(n, device=device).repeat(num_repeats)
        ii = torch.randperm(r.size(0), generator=generator, device=device)[:num_samples]
        r = r[ii]
        return r
    else:
        # never would happen
        raise NotImplementedError


def cat_dict(
    # dict_list: T.List[T.Dict[str, torch.Tensor]],
    dict_list: T.List[T.Dict[str, T.Union[torch.Tensor, T.List[torch.Tensor]]]],
    dim_dict: T.Union[int, T.Dict[str, int]],
) -> T.Dict[str, torch.Tensor]:
    """
    Given a list of dict, each of which contains torch.Tensor or a list of torch.Tensor,
    we concat along `dim` and create a dict by concat each of them.
    Args:
        dict_list:
        dim_dict:

    Returns:
        a dict having the same keys
    """
    if len(dict_list) == 0:
        return dict()

    if None in dict_list:
        return None

    if isinstance(dim_dict, int):
        dim = dim_dict
        dim_dict = dict()
        for key in dict_list[0]:
            dim_dict[key] = dim

    out_dict = dict()
    for key in dict_list[0]:
        out_dict[key] = [d[key] for d in dict_list]

    for key in out_dict:
        if out_dict[key][0] is None:
            out_dict[key] = None
        elif isinstance(out_dict[key][0], torch.Tensor):
            out_dict[key] = torch.cat(out_dict[key], dim=dim_dict[key])
        else:  # out_dict[key] is a list of "list of tensor"
            batch_num = len(out_dict[key])
            tensor_num = len(out_dict[key][0])
            feature_list = [""] * tensor_num  # initialize list
            for tensor_id in range(tensor_num):
                feature_list[tensor_id] = torch.cat(
                    [out_dict[key][batch_id][tensor_id] for batch_id in range(batch_num)],
                    dim=dim_dict[key],
                )
            out_dict[key] = feature_list

    return out_dict


def index_select_dict(
    dict_tensor: T.Dict[str, torch.Tensor],
    dim: int,
    index: torch.Tensor,
) -> T.Union[None, T.Dict[str, torch.Tensor]]:
    """
    Given dict of tensors, perform index_select
    on each of the tensors.
    """
    if dict_tensor is None:
        return None
    out_dict = dict()
    for key in dict_tensor:
        arr = dict_tensor[key]
        if arr is None:
            out_dict[key] = None
        else:
            out_dict[key] = torch.index_select(
                input=arr,
                dim=dim,
                index=index,
            )
    return out_dict


def chunk_dict(
    dict_tensor: T.Dict[str, torch.Tensor],
    chunks: int,
    dim: int = 0,
) -> T.Union[None, T.List[T.Dict[str, torch.Tensor]]]:
    """
    Given dict of tensors, chunk the tensors and returns a
    list of dict, which contains one chunk.
    """
    if dict_tensor is None:
        return None

    out_dict = dict()
    for key in dict_tensor:
        arr = dict_tensor[key]
        if arr is None:
            out_dict[key] = [None] * chunks
        else:
            out_dict[key] = torch.chunk(
                input=arr,
                chunks=chunks,
                dim=dim,
            )

    out_list = []
    for i in range(chunks):
        subdict = dict()
        for name in out_list:
            subdict[name] = out_list[name][i]
        out_list.append(subdict)
    return out_list


def reshape(
    arr: T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]],
    start: int = 0,
    end: int = -1,  # included
    shape: T.Union[int, T.List[int], T.Tuple[int]] = None,
) -> T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]]:
    """
    Given a dict or a list containing tensor,
    reshape dimension `start` to `end` to the new shape
    """
    if isinstance(shape, int):
        shape = [shape]
    shape = list(shape)

    if isinstance(arr, torch.Tensor):
        arr_shape = arr.shape
        if end < 0:
            end = end + arr.ndim
        new_shape = list(arr_shape[:start]) + list(shape) + list(arr_shape[end + 1 :])
        arr = arr.reshape(*new_shape)
        return arr
    elif isinstance(arr, (list, tuple)):
        return [reshape(x, start=start, end=end, shape=shape) for x in arr]
    elif isinstance(arr, dict):
        for key, val in arr.items():
            arr[key] = reshape(val, start=start, end=end, shape=shape)
        return arr
    else:
        return arr


def unsqueeze(
    arr: T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]],
    dim: int = 0,
) -> T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]]:
    """
    Given a dict or a list containing tensor, unsqueeze the given dimension.
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.unsqueeze(dim=dim)
        return arr
    elif isinstance(arr, (list, tuple)):
        return [unsqueeze(x, dim=dim) for x in arr]
    elif isinstance(arr, dict):
        for key, val in arr.items():
            arr[key] = unsqueeze(val, dim=dim)
        return arr
    else:
        return arr


def squeeze(
    arr: T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]],
    dim: int = 0,
) -> T.Union[torch.Tensor, T.List[torch.Tensor], T.Dict[str, T.Any]]:
    """
    Given a dict or a list containing tensor, squeeze the given dimension.
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.squeeze(dim=dim)
        return arr
    elif isinstance(arr, (list, tuple)):
        return [squeeze(x, dim=dim) for x in arr]
    elif isinstance(arr, dict):
        for key, val in arr.items():
            arr[key] = squeeze(val, dim=dim)
        return arr
    else:
        return arr


def create_pcd(
    points: T.Union[torch.Tensor, np.ndarray],
    colors: T.Union[torch.Tensor, np.ndarray] = None,
    remove_nan_inf: bool = True,
) -> o3d.geometry.PointCloud:
    """
    Create o3d point cloud from points
    Args:
        points:
            (*, 3)
        colors:
            (*, 3) optional
        remove_nan_inf:

    Returns:
        an o3d point cloud
    """

    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    points = points.reshape(-1, 3)  # (n, 3)
    if colors is not None:
        colors = colors.reshape(-1, 3)  # (n, 3)

    # remove any inf or nan points
    if remove_nan_inf:
        idxs = np.all(np.isfinite(points), axis=-1)  # (n,)
        points = points[idxs]
        if colors is not None:
            colors = colors[idxs]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def create_octree(
    points: T.Union[torch.Tensor, np.ndarray, o3d.geometry.PointCloud],
    max_depth: int = 5,
    remove_nan_inf: bool = True,
) -> o3d.geometry.Octree:
    """
    Store the xys points into a octree.

    Args:
        points:
            (*, 3)
        max_depth:
            max depth of the octree. As the tree depth increases,
            internal (and eventually leaf) nodes represents a smaller partition of 3D space.

    Returns:
        an o3d octree
    """
    if isinstance(points, o3d.geometry.PointCloud):
        pcd = points
    elif isinstance(points, (torch.Tensor, np.ndarray)):
        pcd = create_pcd(
            points=points,
            colors=None,
            remove_nan_inf=remove_nan_inf,
        )
    else:
        raise NotImplementedError

    octree = o3d.geometry.Octree(max_depth=max_depth)
    octree.convert_from_point_cloud(
        point_cloud=pcd,
    )
    return octree


@linalg_utils.disable_tf32_and_autocast()
def ray_aabb_intersection(
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
    bbox_min_bounds: torch.Tensor,
    bbox_max_bounds: torch.Tensor,
    bbox_scaling_ratio: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0e10,
) -> T.Dict[str, T.Any]:
    """
    Check whether a ray intersect with an axis-aligned bounding box.

    Args:
        ray_origin:
            (*, 3) a point on the ray (with t = 0)
        ray_direction:
            (*, 3) ray direction
        bbox_min_bounds:
            (3,) or (*, 3) top left corner
        bbox_max_bounds:
            (3,) or (*, 3) bottom right corner.  The bbox encloses bbox_min_bounds to bbox_max_bounds.
        bbox_scaling_ratio:
            scalar, (3,) or (*, 3) a scalar where we will scale the bbox wrt to its center.
        t_min:
            min t to consider
        t_max:
            max t to consider

    Returns:
        is_intersected: True if intersect.
        t_near: first intersection t
        t_far: second intersection t
    """

    # scale bbox
    bbox_center = 0.5 * (bbox_min_bounds + bbox_max_bounds)  # (3,) or (*, 3)
    bbox_min_bounds = bbox_center + (bbox_min_bounds - bbox_center) * bbox_scaling_ratio  # (3,) or (*, 3)
    bbox_max_bounds = bbox_center + (bbox_max_bounds - bbox_center) * bbox_scaling_ratio  # (3,) or (*, 3)

    inv_ray_direction = 1.0 / ray_direction  # (*, 3)
    _t_nears = (bbox_min_bounds - ray_origin) * inv_ray_direction  # (*, 3)
    _t_fars = (bbox_max_bounds - ray_origin) * inv_ray_direction  # (*, 3)

    # make sure t_near < t_far
    t_nears = torch.where(_t_nears > _t_fars, _t_fars, _t_nears)  # (*, 3)
    t_fars = torch.where(_t_nears > _t_fars, _t_nears, _t_fars)  # (*, 3)

    t_nears[torch.isnan(t_nears)] = -torch.inf
    t_fars[torch.isnan(t_fars)] = torch.inf

    t_near, _ = torch.max(t_nears, dim=-1)  # (*,), use fmin to ignore nan, no need for max
    t_far, _ = torch.min(t_fars, dim=-1)  # (*,), use fmax to ignore nan, no need for min

    t_near = torch.max(t_near, torch.ones_like(t_near) * t_min)  # (*,)
    t_far = torch.min(t_far, torch.ones_like(t_far) * t_max)  # (*,)

    is_intersect = t_near <= t_far  # (*,)
    return dict(
        is_intersected=is_intersect,  # (*,)
        t_near=t_near,  # (*,)
        t_far=t_far,  # (*,)
    )


@linalg_utils.disable_tf32_and_autocast()
def ray_aabb_intersection_2(
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
    bbox_min_bounds: torch.Tensor,
    bbox_max_bounds: torch.Tensor,
    bbox_scaling_ratio: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0e8,
) -> T.Dict[str, T.Any]:
    """
    Check whether a ray intersect with an axis-aligned bounding box.

    Args:
        ray_origin:
            (3,) a point on the ray (with t = 0)
        ray_direction:
            (3,) ray direction
        bbox_min_bounds:
            (3,)  top left corner
        bbox_max_bounds:
            (3,)  bottom right corner.  The bbox encloses bbox_min_bounds to bbox_max_bounds.
        bbox_scaling_ratio:
            a scalar where we will scale the bbox wrt to its center.
        t_min:
            min t to consider
        t_max:
            max t to consider

    Returns:
        is_intersected: True if intersect.
        t0: first intersection t
        t1: second intersection t
    """

    # scale bbox
    bbox_center = 0.5 * (bbox_min_bounds + bbox_max_bounds)
    bbox_min_bounds = bbox_center + (bbox_min_bounds - bbox_center) * bbox_scaling_ratio
    bbox_max_bounds = bbox_center + (bbox_max_bounds - bbox_center) * bbox_scaling_ratio

    inv_ray_direction = 1.0 / ray_direction  # (3,)
    _t_nears = (bbox_min_bounds - ray_origin) * inv_ray_direction  # (3,)
    _t_fars = (bbox_max_bounds - ray_origin) * inv_ray_direction  # (3,)

    t_near = t_min
    t_far = t_max
    for i in range(3):
        if _t_nears[i] > _t_fars[i]:
            _t_near = _t_fars[i]
            _t_far = _t_nears[i]
        else:
            _t_near = _t_nears[i]
            _t_far = _t_fars[i]
        if t_near < _t_near:
            t_near = _t_near
        if t_far > _t_far:
            t_far = _t_far
        if t_near > t_far:
            return dict(
                is_intersected=False,
                t_near=t_near,
                t_far=t_far,
            )

    return dict(
        is_intersected=True,
        t_near=t_near,
        t_far=t_far,
    )


@linalg_utils.disable_tf32_and_autocast()
def compute_point_ray_distance_in_chunks(
    points: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    max_chunk_size: int = int(1e8),
) -> T.Dict[str, T.Any]:
    """
    Compute the distance between each point to each ray
    Args:
        points:
            (*, n, 3)
        ray_origins:
            (*, m, 3)
        ray_directions:
            (*, m, 3)
        max_chunk_size:
            max number of (*, m, n) to avoid using all memory.
            If more than `max_chunk_size`, we will chunk it and use
            for loop for each.   -1: ignored

    Returns:
        dists: (*, m, n) distance between each point to each ray
        projections:  (*, m, n, 3) the projected points on the ray
        ts: (*, m, n) length on ray (can be negative)
    """

    *b_size, n, _ = points.shape
    m = ray_origins.size(-2)

    points = points.reshape(-1, n, 3)  # (b, n, 3)
    ray_origins = ray_origins.reshape(-1, m, 3)  # (b, m, 3)
    ray_directions = ray_directions.reshape(-1, m, 3)  # (b, m, 3)
    b = points.size(0)

    if max_chunk_size < 0:
        max_chunk_size = np.inf
    # check if m*n > or <= max_chunk_size:
    # if mn > max_chunk_size: we chunk along m and for loop on b
    # if mn <= max_chunk_size: we use as large b as possible (chunk along b)
    mn = m * n
    if mn > max_chunk_size:  # chunk along m
        max_m = max(1, int(max_chunk_size / n))
        num_chunks = math.ceil(m / max_m)
        chunk_dim = 1
        ray_origins_chunks = torch.chunk(ray_origins, chunks=num_chunks, dim=chunk_dim)
        ray_directions_chunks = torch.chunk(ray_directions, chunks=num_chunks, dim=chunk_dim)
        points_chunks = [points] * len(ray_origins_chunks)  # reference
    else:
        # chunk along b
        max_b = max(1, int(max_chunk_size / mn))
        num_chunks = math.ceil(b / max_b)
        chunk_dim = 0
        ray_origins_chunks = torch.chunk(ray_origins, chunks=num_chunks, dim=chunk_dim)
        ray_directions_chunks = torch.chunk(ray_directions, chunks=num_chunks, dim=chunk_dim)
        points_chunks = torch.chunk(points, chunks=num_chunks, dim=chunk_dim)

    out_dicts = []
    for i in range(len(ray_origins_chunks)):
        out_dict = compute_point_ray_distance(
            points=points_chunks[i],
            ray_origins=ray_origins_chunks[i],
            ray_directions=ray_directions_chunks[i],
        )
        out_dicts.append(out_dict)

    # concatenate along chunk dimension
    out_dict = cat_dict(
        dict_list=out_dicts,
        dim_dict=chunk_dim,
    )

    # reshape b -> b_shape
    for key in out_dict:
        shape = list(out_dict[key].shape)
        out_dict[key] = torch.reshape(out_dict[key], list(b_size) + shape[1:])

    return out_dict


@linalg_utils.disable_tf32_and_autocast()
def compute_point_ray_distance(
    points: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
) -> T.Dict[str, T.Any]:
    """
    Compute the distance between each point to each ray
    Args:
        points:
            (*, n, 3)
        ray_origins:
            (*, m, 3)
        ray_directions:
            (*, m, 3)

    Returns:
        dists: (*, m, n) distance between each point to each ray
        projections:  (*, m, n, 3) the projected points on the ray
        ts: (*, m, n) length on ray (can be negative)
    """

    points = points.unsqueeze(-3)  # (*, 1, n, 3)
    ray_origins = ray_origins.unsqueeze(-2)  # (*, m, 1, 3)
    ray_directions = ray_directions.unsqueeze(-2)  # (*, m, 1, 3)
    dv = points - ray_origins  # (*, m, n, 3)
    ray_directions_norm = torch.linalg.norm(ray_directions, ord=2, dim=-1, keepdim=True)  # (*, m, 1, 1)

    # assert torch.all(ray_directions_norm > 0)
    # ray_directions = ray_directions / ray_directions_norm  # (*, m, 1, 3)
    # assert torch.allclose(ray_directions_norm, torch.ones_like(ray_directions_norm), rtol=1e-3)

    ts = (dv * ray_directions).sum(dim=-1, keepdim=True)  # (*, m, n, 1)  projected length on ray (can be negative)
    projections = ray_origins + ts * ray_directions  # (*, m, n, 3)
    dists = torch.linalg.norm(points - projections, ord=2, dim=-1)  # (*, m, n)

    return dict(
        dists=dists,  # (*, m, n)
        projections=projections,  # (*, m, n, 3)
        ts=ts.squeeze(-1),  # (*, m, n)
    )


@linalg_utils.disable_tf32_and_autocast()
def get_k_neighbor_points_in_chunks(
    points: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    k: int,
    t_min: float = 0.0,
    t_max: float = 1.0e10,
    t_init: torch.Tensor = None,
    max_chunk_size: int = int(1e8),
    pr_params: T.Dict[str, T.Any] = None,
    printout: bool = False,
    cached_info: T.Union[T.Dict[str, torch.Tensor], None] = None,
    valid_mask: torch.Tensor = None,  # (b, n, 1)
    mode: str = "nearest",
) -> T.Dict[str, T.Any]:
    """
    Given n points (xyz) and m rays, return the neighboring points to each ray.

    Args:
        points:
            (*, n, 3)
        ray_origins:
            (*, m, 3)
        ray_directions:
            (*, m, 3)
        k:
            k nearest neighbors
        t_min:
            min t to consider
        t_max:
            max t to consider
        max_chunk_size:
            max number of (*, m, n) to avoid using all memory.
            If more than `max_chunk_size`, we will chunk it and use
            for loop for each.   -1: ignored
        cached_info:
            a dictionary containing the grid cell to point index so pr does not
            need to construct it again.
        valid_mask:
            (b, n, 1) bool, whether the point should be considered in neighbor search
        mode:
            'nearest': return k nearest points within ray radius
            'random': return k random points with ray radius

    Returns:
        sorted_dists:
            (*, m, min(k, n))  the distance of the k nearest points to each ray (inf if not within t range)
        sorted_idxs:
            (*, m, min(k, n)) the index of points of the k nearest points
        # dist_dict:
        #     output of :py:compute_point_ray_distance

        cached_info:
            a dict containing the cached info for pr
    """

    *b_size, n, _ = points.shape
    m = ray_origins.size(-2)
    device = points.device

    points = points.reshape(-1, n, 3)  # (b, n, 3)
    ray_origins = ray_origins.reshape(-1, m, 3)  # (b, m, 3)
    ray_directions = ray_directions.reshape(-1, m, 3)  # (b, m, 3)
    b = points.size(0)

    # if valid_mask is None:
    #     valid_mask = torch.ones(b, n, 1, dtype=torch.bool, device=device)
    #
    if valid_mask is not None and valid_mask.ndim == 2:
        valid_mask = valid_mask.unsqueeze(-1)

    if max_chunk_size is None or max_chunk_size < 0:
        max_chunk_size = int(1e13)  # 1e6
    # check if m*n > or <= max_chunk_size:
    # if mn > max_chunk_size: we chunk along m and for loop on b
    # if mn <= max_chunk_size: we use as large b as possible (chunk along b)
    mn = m * n
    # if mn > max_chunk_size:  # chunk along m
    #     max_m = max(1, int(max_chunk_size / n))

    # since we use pr_cuda v3, we support large number of points
    # so we only need to concern about m being too large
    if m > max_chunk_size:  # chunk along m
        max_m = max_chunk_size
        if printout:
            print(f"max_m = {max_m}")
        num_chunks = math.ceil(m / max_m)
        chunk_dim = 1
        ray_origins_chunks = torch.chunk(ray_origins, chunks=num_chunks, dim=chunk_dim)
        ray_directions_chunks = torch.chunk(ray_directions, chunks=num_chunks, dim=chunk_dim)
        points_chunks = [points] * len(ray_origins_chunks)  # reference
        t_init_chunks = torch.chunk(t_init, chunks=num_chunks, dim=chunk_dim) if t_init is not None else None
        valid_mask_chunks = [valid_mask] * len(ray_origins_chunks)
        reuse_cache = True
    else:
        # chunk along b
        # max_b = max(1, int(max_chunk_size / mn))
        max_b = max(1, int(max_chunk_size / m))

        if printout:
            print(f"max_b = {max_b}")
        num_chunks = math.ceil(b / max_b)
        chunk_dim = 0
        ray_origins_chunks = torch.chunk(ray_origins, chunks=num_chunks, dim=chunk_dim)
        ray_directions_chunks = torch.chunk(ray_directions, chunks=num_chunks, dim=chunk_dim)
        points_chunks = torch.chunk(points, chunks=num_chunks, dim=chunk_dim)
        t_init_chunks = torch.chunk(t_init, chunks=num_chunks, dim=chunk_dim) if t_init is not None else None
        valid_mask_chunks = (
            torch.chunk(valid_mask, chunks=num_chunks, dim=chunk_dim) if valid_mask is not None else None
        )
        reuse_cache = False if num_chunks > 1 else True

    out_dicts = []
    for i in range(len(ray_origins_chunks)):
        if printout:
            if torch.cuda.is_available():
                # torch.cuda.synchronize(device)
                print(
                    f"before {i}/{len(ray_origins_chunks)}-th get_k_neighbor_points: {torch.cuda.memory_allocated(device=device) / 2**30} GB"
                )
                # torch.cuda.synchronize(device)

        stime = timer()
        if pr_params is not None:
            # use pr data structure
            out_dict = get_k_neighbor_within_ray(
                points=points_chunks[i],
                ray_origins=ray_origins_chunks[i],
                ray_directions=ray_directions_chunks[i],
                k=k,
                t_min=t_min,
                t_max=t_max,
                t_init=t_init_chunks[i] if t_init_chunks is not None else None,
                printout=printout,
                cached_info=cached_info,
                valid_mask=valid_mask_chunks[i] if valid_mask_chunks is not None else None,
                mode=mode,
                **pr_params,
            )
            find_neighbor_time = timer() - stime
            if printout:
                print(f"find_neighbor_time pr={find_neighbor_time:.4f} secs", flush=True)

        else:
            # use brute force method
            out_dict = get_k_neighbor_points(
                points=points_chunks[i],
                ray_origins=ray_origins_chunks[i],
                ray_directions=ray_directions_chunks[i],
                k=k,
                t_min=t_min,
                t_max=t_max,
                t_init=t_init_chunks[i] if t_init_chunks is not None else None,
                printout=printout,
            )
            find_neighbor_time = timer() - stime
            if printout:
                print(
                    f"find_neighbor_time normal={find_neighbor_time:.4f} secs",
                    flush=True,
                )

        if not reuse_cache:
            cached_info = None
        else:
            cached_info = out_dict.get("cached_info", None)

        if "cached_info" in out_dict:
            del out_dict["cached_info"]
        out_dicts.append(out_dict)
        if printout:
            if torch.cuda.is_available():
                # torch.cuda.synchronize(device)
                print(
                    f"after {i}/{len(ray_origins_chunks)}-th get_k_neighbor_points: {torch.cuda.memory_allocated(device=device) / 2**30} GB"
                )
            for key in out_dict:
                print(f"  {key}: {out_dict[key].shape} {int(np.prod(out_dict[key].shape)) * 4 / 2**30:.2f} GB")
            # torch.cuda.synchronize(device)

    # concatenate along chunk dimension
    out_dict = cat_dict(
        dict_list=out_dicts,
        dim_dict=chunk_dim,
    )

    # reshape b -> b_shape
    for key in out_dict:
        shape = list(out_dict[key].shape)
        out_dict[key] = torch.reshape(out_dict[key], list(b_size) + shape[1:])

    out_dict["cached_info"] = cached_info
    return out_dict


@linalg_utils.disable_tf32_and_autocast()
def get_k_neighbor_within_ray(
    points: torch.Tensor,  # ( b, n, 3)
    ray_origins: torch.Tensor,  # ( b, m, 3)
    ray_directions: torch.Tensor,  # ( b, m, 3)
    # nohit_token: torch.Tensor,
    k: int,
    ray_radius: float = -1.0,
    grid_size: int = 100,
    grid_center: float = 0,
    grid_width: float = 2,
    t_min: float = 0.0,
    t_max: float = 1.0e10,
    t_init: torch.Tensor = None,
    # max_chunk_size: int = int(1e8),
    printout: bool = False,
    cached_info: T.Union[T.Dict[str, torch.Tensor], None] = None,
    valid_mask: torch.Tensor = None,  # (b, n, 1)
    mode: str = "nearest",
):
    """

    Args:
        points:
        ray_origins:
        ray_directions:
        k:
        ray_radius:
        grid_size:
        grid_center:
        grid_width:
        t_min:
        t_max:
        t_init:
        printout:
        cached_info:
            a dictionary containing the grid cell to point index so pr does not
            need to construct it again.
        valid_mask:
            (b, n, 1) whether to include a point in the neighbor search.
        mode:
            'nearest': return k nearest points within ray radius
            'random': return k random points with ray radius
    Returns:

    """
    raise RuntimeError("pr not loaded")
    if mode == "nearest":
        mode_idx = 0
    elif mode == "random":
        mode_idx = 1
    else:
        raise NotImplementedError

    device = points.device
    batch_size, n_rays, _ = ray_origins.shape

    # no_hit_token = torch.ones(batch_size, 1, 3, device = device) * torch.inf

    stime = timer()
    direct_process = True

    # if ray_radius is None or ray_radius < 0:
    #     ray_radius = grid_width / grid_size * 2
    #     if isinstance(ray_radius, torch.Tensor):
    #         ray_radius = ray_radius.max()

    if cached_info is None:
        gidx2pidx_bank = None
        gidx_start_idx = None
        refresh_cache = True
    else:
        gidx2pidx_bank = cached_info.get("gidx2pidx_bank", None)
        gidx_start_idx = cached_info.get("gidx_start_idx", None)
        refresh_cache = False

    if direct_process:
        with torch.no_grad():
            if t_init is not None:
                k = k * 2  # select 2k first

            out_dict = pr_utils.find_k_neighbor_points_of_rays(
                points=points.contiguous(),
                k=k,
                ray_origins=ray_origins.contiguous(),
                ray_directions=ray_directions.contiguous(),
                ray_radius=ray_radius,
                grid_size=grid_size,
                grid_center=grid_center,
                grid_width=grid_width,
                gidx2pidx_bank=gidx2pidx_bank,
                gidx_start_idx=gidx_start_idx,
                refresh_cache=refresh_cache,
                valid_mask=valid_mask,
                mode=mode_idx,
            )
            all_idxs = out_dict["ray2pidx_heap"]  # (b, m, k)
            neighbor_num = out_dict["ray_neighbor_num"]  # (b, m)
            # ray2dist_heap = out_dict['ray2dist_heap']  # (b, m, k)  we not only need dist, we also need t
            gidx2pidx_bank = out_dict["gidx2pidx_bank"]  # (b, n)
            gidx_start_idx = out_dict["gidx_start_idx"]  # (b, max_n_cell+1)
            cached_info = dict(
                gidx2pidx_bank=gidx2pidx_bank,
                gidx_start_idx=gidx_start_idx,
            )

            # construct invalid mask
            neighbor_idx = (
                torch.arange(
                    k,
                    device=neighbor_num.device,
                )
                .unsqueeze(0)
                .expand(batch_size, n_rays, k)
            )  # (b, m, k)
            invalid_mask = neighbor_idx >= neighbor_num.unsqueeze(-1)  # (b, m, k)

            # we assume index 0 is a point at inf
            all_idxs = all_idxs.masked_fill(invalid_mask, 0)  # (b, m, k)

            neighbor_points = torch.gather(
                input=points,  # (b, n, 3)
                dim=-2,
                index=all_idxs.reshape(batch_size, n_rays * k, 1).expand(-1, -1, 3),  # (b, m*k, 3)
            )  # (b, m*k, 3)

            dist_dict = compute_point_ray_distance(
                points=neighbor_points.reshape(batch_size * n_rays, k, 3),  # ( b*m, k, 3)
                ray_origins=ray_origins.reshape(-1, 1, 3),  # ( b*m, 1, 3)
                ray_directions=ray_directions.reshape(-1, 1, 3),  # ( b*m, 1, 3)
            )

            all_dists = dist_dict["dists"].squeeze(-2).reshape(batch_size, n_rays, k)  # (b, m, k)
            all_ts = dist_dict["ts"].squeeze(-2).reshape(batch_size, n_rays, k)  # (b, m, k)

            # set invalid neighbor points (point at inf) to have negative t and dist=inf
            all_dists = all_dists.masked_fill(invalid_mask, 1e12)
            all_ts = all_ts.masked_fill(invalid_mask, t_min - 1)  # will be ignored in rectify

            if t_init is not None:
                # select k points closer to t_init from the 2k points
                k = k // 2
                assert k > 0
                sorted_point_dist = torch.square(all_ts - t_init) + torch.square(all_dists)
                _, sorted_ts_idxs = torch.sort(sorted_point_dist, dim=-1)
                sorted_ts_idxs = sorted_ts_idxs[..., :k].clone()

                all_dists = torch.gather(
                    input=all_dists,  # (*, m, 2*k)
                    dim=-1,
                    index=sorted_ts_idxs,  # (*, m, k)
                )

                all_idxs = torch.gather(
                    input=all_idxs,  # (*, m, 2*k)
                    dim=-1,
                    index=sorted_ts_idxs,  # (*, m, k)
                )

                all_ts = torch.gather(
                    input=all_ts,  # (*, m, 2*k)
                    dim=-1,
                    index=sorted_ts_idxs,  # (*, m, k)
                )

    else:
        # the implementation below has been deprecated
        # but may be useful for debug
        # note that below randomly select k points if more are found, not k closest ones
        # initialize return dict

        all_dists = torch.ones(batch_size, n_rays, k, device=device) * 1e12
        all_idxs = torch.zeros(batch_size, n_rays, k, device=device, dtype=torch.long)
        all_ts = torch.ones(batch_size, n_rays, k, device=device) * -1
        neighbor_num = torch.ones(batch_size, n_rays, device=device)

        with torch.no_grad():
            all_ray2pidxs = pr_utils.find_neighbor_points_of_rays(
                points=points,
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                ray_radius=ray_radius,
                grid_size=grid_size,
                grid_center=grid_center,
                grid_width=grid_width,
            )  # (b, m, undef), list of list of tensor

        if printout:
            process_time = timer() - stime
            if printout:
                print(f"neighbor_time={process_time:.4f} secs", flush=True)
            stime = timer()

        for b in range(batch_size):
            # points[b, 0] = points[b, 0] + 1e12  # temp make the first point as missing token

            ray2pidxs = all_ray2pidxs[b]
            for m, pidxs in enumerate(ray2pidxs):
                # number of neighbor point
                n = len(pidxs)
                neighbor_num[b, m] = min(n, k)
                if n != 0:
                    # fill with empty
                    if n <= k:
                        # all_dists[ b, m, :n] = dists
                        all_idxs[b, m, :n] = pidxs
                        # all_ts[ b, m, :n] = ts
                    # random select k
                    elif n > k:
                        # all_dists[ b, m, :n] = dists[torch.randperm(n)[:k]]
                        all_idxs[b, m, :n] = pidxs[torch.randperm(n)[:k]]
                        # all_ts[ b, m, :n] = ts[torch.randperm(n)[:k]]

            dist_dict = compute_point_ray_distance(
                points=points[b, all_idxs[b]],  # (m, k, 3)
                ray_origins=ray_origins[b].reshape(-1, 1, 3),  # (m, 1, 3)
                ray_directions=ray_directions[b].reshape(-1, 1, 3),  # (m, 1, 3)
            )
            all_dists[b] = dist_dict["dists"].squeeze(-2)  # ( m , k)
            all_ts[b] = dist_dict["ts"].squeeze(-2)  # ( m , k)

        if printout:
            select_time = timer() - stime
            if printout:
                print(f"select_time={select_time:.4f} secs", flush=True)

        cached_info = None

    invalid_mask = torch.logical_or(all_ts < t_min, all_ts > t_max)  # (*, m, n)
    # notes: two kinds of invalid points:
    # (1) the background point, the position itself is now set to 1e12, see render.rasterize
    # (2) points lies in the opposite direction of the ray
    all_dists[invalid_mask] = torch.inf

    # note that all of these are not really "sorted"
    # the name are just to match the usage of get_k_neighbor_points
    return dict(
        sorted_dists=all_dists,  # (*, m, k)
        sorted_idxs=all_idxs,  # (*, m, k)
        sorted_ts=all_ts,  # (*, m, k) length on ray (can be negative)
        neighbor_num=neighbor_num,  # (*, m) number of neighbors of each ray
        cached_info=cached_info,
    )


@linalg_utils.disable_tf32_and_autocast()
def get_k_neighbor_points(
    points: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    k: int,
    t_min: float = 0.0,
    t_max: float = 1.0e10,
    t_init: torch.Tensor = None,
    printout: bool = False,
) -> T.Dict[str, T.Any]:
    """
    Given n points (xyz) and m rays, return the neighboring points to each ray.

    Args:
        points:
            (*, n, 3)
        ray_origins:
            (*, m, 3)
        ray_directions:
            (*, m, 3)
        k:
            k nearest neighbors
        t_min:
            min t to consider
        t_max:
            max t to consider

    Returns:
        sorted_dists:
            (*, m, min(k, n))  the distance of the k nearest points to each ray (inf if not within t range)
        sorted_idxs:
            (*, m, min(k,n)) the index of points of the k nearest points
        sorted_ts:
            (*, m, min(k, n)) projection length of each point on ray from the ray_origins (can be negative)
        # dist_dict:
        #     output of :py:compute_point_ray_distance
    """
    device = points.device

    if printout:
        if torch.cuda.is_available():
            # torch.cuda.synchronize(device)
            print(f"  before compute ts: {torch.cuda.memory_allocated(device=device) / 2**30} GB")
        #    torch.cuda.synchronize(device)

    dist_dict = compute_point_ray_distance(
        points=points,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
    )
    dists = dist_dict["dists"]  # (*, m, n)  (ray, point)
    ts = dist_dict["ts"]  # (*, m, n)

    if printout:
        if torch.cuda.is_available():
            # torch.cuda.synchronize(device)
            print(f"  after compute ts: {torch.cuda.memory_allocated(device=device) / 2**30} GB")
            for key in dist_dict:
                print(f"    {key}: {dist_dict[key].shape} {int(np.prod(dist_dict[key].shape)) * 4 / 2**30:.2f} GB")
            # torch.cuda.synchronize(device)

    # map invalid ts's dist to inf
    invalid_mask = torch.logical_or(ts < t_min, ts > t_max)  # (*, m, n)
    # notes: two kinds of invalid points:
    # (1) the background point, the position itself is now set to 1e12, see render.rasterize
    # (2) points lies in the opposite direction of the ray
    dists[invalid_mask] = torch.inf

    # sort dists of the points for each ray
    # we can imagine sort() as doing two operations:
    # indices = argsort(x,dim)
    # y = gather(x,indices,dim)
    # and the argsort part is not differentiable
    # should give an option to do torch no_grad to make it clear
    # but not done yet
    # https://discuss.pytorch.org/t/differentiable-sorting-and-indices/89304
    sorted_dists, sorted_idxs = torch.sort(dists, dim=-1)  # (*, m, n), (*, m, n)

    if printout:
        if torch.cuda.is_available():
            # torch.cuda.synchronize(device)
            print(f"  before gather ts: {torch.cuda.memory_allocated(device=device) / 2**30} GB")
            # torch.cuda.synchronize(device)

    # multiple passes
    # find 2k amount of neighbors first and keep only k nearest
    if t_init is not None:
        # keep only k nearest neighbors
        sorted_dists = sorted_dists[
            ..., : 2 * k
        ].clone()  # (*, m, 2*k) the distance of the k nearest points to each ray
        sorted_idxs = sorted_idxs[..., : 2 * k].clone()  # (*, m, 2*k) the index of k nearest points
        sorted_ts = torch.gather(
            input=ts,
            dim=-1,
            index=sorted_idxs,  # (*, m, n)  # (*, m, 2*k)
        )  # (b, m, k)  neighbot_ts[b, m, i] = ts[b, m, neighbor_xyz_w_idxs[b, m, i]]

        # ray norm = 1, t difference = distance projected on ray
        sorted_point_dist = torch.square(sorted_ts - t_init) + torch.square(sorted_dists)
        _, sorted_ts_idxs = torch.sort(sorted_point_dist, dim=-1)
        sorted_ts_idxs = sorted_ts_idxs[..., :k].clone()

        sorted_dists = torch.gather(
            input=sorted_dists,
            dim=-1,
            index=sorted_ts_idxs,  # (*, m, 2*k)  # (*, m, k)
        )

        sorted_idxs = torch.gather(
            input=sorted_idxs,
            dim=-1,
            index=sorted_ts_idxs,  # (*, m, 2*k)  # (*, m, k)
        )

        sorted_ts = torch.gather(
            input=sorted_ts,
            dim=-1,
            index=sorted_ts_idxs,  # (*, m, 2*k)  # (*, m, k)
        )
    else:
        # keep only k nearest neighbors
        sorted_dists = sorted_dists[..., :k].clone()  # (*, m, k) the distance of the k nearest points to each ray
        sorted_idxs = sorted_idxs[..., :k].clone()  # (*, m, k) the index of k nearest points

        sorted_ts = torch.gather(
            input=ts,
            dim=-1,
            index=sorted_idxs,  # (*, m, n)  # (*, m, k)
        )  # (b, m, k)  neighbot_ts[b, m, i] = ts[b, m, neighbor_xyz_w_idxs[b, m, i]]

    # # clone so that sorted_ts is not a view of the original ts (b, m, n), which then can be deleted
    # sorted_ts = sorted_ts.clone()
    # del dist_dict
    # del invalid_mask
    # del dists
    # del ts

    if printout:
        if torch.cuda.is_available():
            # torch.cuda.synchronize(device)
            print(f"  after gather ts: {torch.cuda.memory_allocated(device=device) / 2**30} GB")
            # torch.cuda.synchronize(device)

    return dict(
        sorted_dists=sorted_dists,  # (*, m, k)
        sorted_idxs=sorted_idxs,  # (*, m, k)
        sorted_ts=sorted_ts,  # (*, m, k) length on ray (can be negative)
        # dist_dict=dist_dict,  # output of compute_point_ray_distance
    )


def rectify_points(
    points: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    translate: bool = False,
    randomize_translate: bool = False,
    ts: torch.Tensor = None,
    t_min: float = 0.0,
    t_max: float = 1e6,
    # normalize_t_std: bool = True,
):
    """
    Given n points associated with each of the m rays (each row in points),
    rotate and translate the coordinate so that
    - ray direction becomes (0,0,1)
    - ray origin becomes (0,0,0)
    - if translate is True, the coordinate origin is chosen so that the t to the closest projection is 1

    Args:
        points:
            (*, m, n, 3)  xyz
        ray_origins:
            (*, m, 3)
        ray_directions:
            (*, m, 3)
        translate:
            whether to translate the coord so that the closest point's projection on the
            ray has t = 0 for all t > 0.
        randomize_translate:
            only used when translate is true.
            If randomize_translate = true, tt will be one random distance between 0 and closest point
        ts:
            (*, m, n)  the projection length each points on the ray
            should be given only if translate is True.
        # normalize_t_std:
        #     whether to scale the coordinate system so that the std of ts = 1 for each ray

    Returns:
        points_n:
            (*, m, n, 3)  the transformed points
        Rs_w2n:
            (*, m, 3, 3) the rotation matrix that transform the world coord to the rectified coord
        translation_w2n:
            (*, m, 3, 1) the translation vector that transform the world coord to the rectified coord
        tt:
            (*, m) the t that we subtract from the input ts
    """
    timing_info = dict()
    stime_total = timer()
    stime_frame = timer()
    y = torch.zeros_like(ray_directions)  # (*, m, 3)
    y[..., 1] = 1.0  # try to use the current y-axis as the y-axis
    Rs_n2w = rigid_motion.construct_coord_frame(
        z=ray_directions,
        y=y,
    )  # (*, m, 3, 3)  the column in the 3*3 matrix is the axis coord with unit norm
    # Rs_n2w = torch.randn(*(ray_directions.shape[:-1]), 3, 3, device=ray_directions.device)
    # it can be thought of as the transform matrix from the new coord to the world coord
    timing_info["create_R_n2w_in_rectify"] = timer() - stime_frame

    # if normalize_t_std:
    #     assert ts is not None
    #     assert ts.isfinite().all()
    #     t_stds = torch.std(ts, dim=-1, unbiased=False)  # (*, m)

    stime_ts = timer()
    if translate:
        assert ts is not None
        ts = ts.clone()
        ts[torch.logical_or(ts < t_min, ts > t_max)] = (
            torch.inf
        )  # we do not care about negative ts (potential problem: all ts could be neg)
        if ts.size(-1) > 0:
            tt, _ = ts.min(dim=-1, keepdim=True)  # (*, m, 1)
        else:
            tt = ts.new(*ts.shape[:-1], 1)  # (*, m, 1)
        tt[~torch.isfinite(tt)] = 0  # if all ts are neg, do not move the origin
        # in pr, invalid point is represented by background far-away point
        # do not shift if all points are background point as well
        # note that training process could occassionaly have a large hit error (should be OK since gradient is clipped)
        # maybe it is because far_thres 1e6 too large to thres out some points...?

        tt[tt > t_max] = 0
        if randomize_translate:
            # tt = tt * torch.rand(tt.shape, device=tt.device)
            tt = tt - torch.rand(tt.shape, device=tt.device) * 0.1
        else:
            pass
            # tt = tt - 0.05

        origins_w = ray_origins + tt * ray_directions  # (*, m, 3)
    else:
        tt_shape = list(ray_origins.shape)
        tt_shape[-1] = 1
        tt = torch.zeros(*tt_shape, device=ray_origins.device)  # (*, m, 1)
        origins_w = ray_origins  # (*, m, 3)
    timing_info["create_ts_in_rectify"] = timer() - stime_ts

    # create H_w2n (note the inversion)
    stime_H = timer()
    Rs_w2n = Rs_n2w.transpose(-1, -2)  # (*, m, 3, 3)
    translation_w2n = -1.0 * linalg_utils.matmul(Rs_w2n, origins_w.unsqueeze(-1))  # (*, m, 3, 1)
    timing_info["create_H_in_rectify"] = timer() - stime_H
    # Hs_w2n = torch.zeros(y.size(0), 4, 4, device=y.device)  # (m, 4, 4)
    # Hs_w2n[:, :3, :3] = Rs_w2n
    # Hs_w2n[:, :3, 3:4] = -1.0 * (Rs_w2n @ origins_w.unsqueeze(-1))  # (m, 3, 1)
    # Hs_w2n[:, 3, 3] = 1

    # transform the points (m, n, 3):  (*, m, 1, 3, 3) @ (*, m, n, 3, 1) + (*, m, 1, 3, 1)
    stime_transform = timer()
    points_n = linalg_utils.matmul(Rs_w2n.unsqueeze(-3), points.unsqueeze(-1)) + translation_w2n.unsqueeze(
        -3
    )  # (*, m, n, 3, 1)
    timing_info["transform_in_rectify"] = timer() - stime_transform
    timing_info["total_in_rectify"] = timer() - stime_total

    return dict(
        points_n=points_n.squeeze(-1),  # (*, m, n, 3)
        Rs_w2n=Rs_w2n,  # (*, m, 3, 3)
        translation_w2n=translation_w2n,  # (*, m, 3, 1)
        tt=tt.squeeze(-1),  # (*, m) the t that we subtract from the input ts
        timing_info=timing_info,
    )


@linalg_utils.disable_tf32_and_autocast()
def compute_3d_xyz(
    z_map: torch.Tensor,
    intrinsic: torch.Tensor,
    H_c2w: torch.Tensor,
    subsample: int = 1,
    other_maps: T.List[torch.Tensor] = None,
):
    """
    Compute the xyz in the world coordinate using z_map and camera pose.

    Important note:
        The function uses a image coordinate system: x to right, y to "down", z to far.
        If the world coordinate is a different one (say x to right, y to "up", z to us),
        H_c2w need to include the image coordinate to world (ie. flip y and z),
        ex: H_actual * H_i2l

    Args:
        z_map:
            (*, h, w) the z coordinate of the point in the camera coordinate on the sensor,
            not along the corresponding camera ray.
        intrinsic:
            (*, 3, 3) camera intrinsic matrix
        H_c2w:
            (*, 4, 4) homegeneous matrix that convert camera coord to world coord.
            Note that the y axis should be inverted in the cam_poses.
        subsample: int
            index stride
        other_maps:
            a list of (*, h, w, d) to associated with each point

    Returns:
        points:
            (*, h//subsample, w//subsample, 3) xyz in world coordinates

    Notes:
        The function assumes the image coordinate origin is at the upper-left. So H_c2w should include
        the y-inverted transformation (e.g., H_c2w = H_c2w_y_flipped * H_flip_y).

    Notes2:
        If an element in z_map == torch.inf or torch.nan,
        the output xyz_w of the corresponding point will be torch.nan.
        Other points will be normal.

    """
    if other_maps is None:
        other_maps = []

    dtype = z_map.dtype
    device = z_map.device
    h, w = z_map.size(-2), z_map.size(-1)

    # generate u v w on the sensor coord
    u, v = torch.meshgrid(
        torch.arange(0, w, subsample, device=device),
        torch.arange(0, h, subsample, device=device),
        indexing="xy",
    )  # u: (h', w') for x,  v: (h', w') for y in the sensor coord
    # uv_shape = list(z_map.shape)
    # uv_shape[-2] = u.size(0)
    # uv_shape[-1] = u.size(1)
    # u = u.expand(uv_shape)  # (*, h', w')
    # v = v.expand(uv_shape)  # (*, h', w')

    z_map = z_map[..., v, u]  # (*, h', w')
    uvw = torch.stack(((u + 0.5) * z_map, (v + 0.5) * z_map, z_map), dim=-1).unsqueeze(-1).to(dtype)  # (*, h, w, 3, 1)
    inv_intrinsic = torch.linalg.inv(intrinsic).unsqueeze(-3).unsqueeze(-3)  # (*, 1, 1, 3, 3)
    xyz_c = linalg_utils.matmul(inv_intrinsic, uvw)  # (*, h', w', 3, 1)  xyz in cam coord
    xyz_c_shape = list(xyz_c.shape)
    xyz_c_shape[-2] = 1
    xyz_c_shape[-1] = 1
    xyz1_c = torch.cat(
        (xyz_c, torch.ones(*xyz_c_shape, dtype=dtype, device=device)),
        dim=-2,
    )  # (*, h', w', 4, 1)
    H_c2w = H_c2w.unsqueeze(-3).unsqueeze(-3)  # (*, 1, 1, 4, 4)
    xyz1_w = linalg_utils.matmul(H_c2w, xyz1_c)  # (*, h', w', 4, 1) xyz in world coord
    xyz_w = xyz1_w[..., :3, 0]  # (*, h', w', 3)

    # other_maps
    all_features = []
    for o_map in other_maps:
        if o_map is not None:
            out = o_map[..., v, u, :]  # (*, h', w', d)
        else:
            out = None
        all_features.append(out)

    return dict(
        xyz_w=xyz_w,
        other_maps=all_features,
    )


@linalg_utils.disable_tf32_and_autocast()
def compute_xyz_w_from_uv(
    uv_c: torch.Tensor,
    z_c: torch.Tensor,
    intrinsic: torch.Tensor,
    H_c2w: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the xyz_w in the world coordinate using the image coordinate and its z_c.

    Important note:
        The function assumes an image coordinate system: x to right, y to "down", z to far.
        If the world coordinate is a different one (say x to right, y to "up", z to us),
        H_c2w need to include the image coordinate to world (ie. flip y and z),
        ex: H_actual * H_i2l

    Args:
        uv_c:
            (*b_shape, *n_shape, 2) the image coordinate (in pixel), u to the right, v to down,
            (0, 0) is top left, (w, h) is bottom right
            In other words, u is column index, v is row index.
        z_c:
            (*b_shape, *n_shape, )  the z coordinate of the point in the camera coordinate on the sensor,
            not along the corresponding camera ray.
        intrinsic:
            (*b_shape, 3, 3) camera intrinsic matrix
        H_c2w:
            (*b_shape, 4, 4) homegeneous matrix that convert camera coord to world coord.
            Note that the y axis should be inverted in the cam_poses.

    Returns:
        (*b_shape, *n_shape, 3) xyz in world coordinates


    Notes:
        The function assumes the image coordinate origin is at the upper-left. So H_c2w should include
        the y-inverted transformation (e.g., H_c2w = H_c2w_y_flipped * H_flip_y).

    """

    *b_shape, _, _ = intrinsic.shape
    n_shape = z_c.shape[len(b_shape) :]

    z_c = z_c.unsqueeze(-1)  # (*b_shape, *n_shape, 1)
    uvw = torch.cat([uv_c * z_c, z_c], dim=-1)  # (*b_shape, *n_shape, 3)

    uvw = uvw.reshape(*b_shape, *n_shape, 3, 1)  # (*b, *n, 3, 1)

    inv_intrinsic = torch.linalg.inv(intrinsic)  # (*b, 3, 3)
    inv_intrinsic = inv_intrinsic.reshape(*b_shape, *([1] * len(n_shape)), 3, 3)  # (*b, *n, 3, 3)

    xyz_c = linalg_utils.matmul(inv_intrinsic, uvw)  # (*b, *n, 3, 1)  xyz in cam coord
    xyz_c_shape = list(xyz_c.shape)
    xyz_c_shape[-2] = 1
    xyz1_c = torch.cat(
        [
            xyz_c,  # (*b, *n, 3, 1)
            torch.ones(*xyz_c_shape, device=xyz_c.device, dtype=xyz_c.dtype),  # (*b, *n, 1, 1)
        ],
        dim=-2,
    )  # (*b, *n, 4, 1)

    H_c2w = H_c2w.reshape(*b_shape, *([1] * len(n_shape)), 4, 4)  # (*b, *n, 4, 4)
    xyz1_w = linalg_utils.matmul(H_c2w, xyz1_c)  # (*b, *n, 4, 1) xyz in world coord
    xyz_w = xyz1_w[..., :3, 0]  # (*b, *n, 3)

    return xyz_w


@linalg_utils.disable_tf32_and_autocast()
def pinhole_projection(
    xyz_w: torch.Tensor,
    intrinsics: torch.Tensor,
    H_c2w: torch.Tensor,
    dim_b: int = 0,
    return_depth_map: bool = False,
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the image coordinates of the 3D points in the world.

    Args:
        xyz_w:
            (*b_shape, *n_shape, 3) the points in world coordinate
        intrinsics:
            (*b_shape, *m_shape, 3, 3) the camera intrinsics
        H_c2w:
            (*b_shape, *m_shape, 4, 4) homegeneous matrix (camera -> world)
        dim_b:
            length of b_shape
    Returns:
        uv_c:
            (*b_shape, *m_shape, *n_shape, 2) (col, row) on the images,
            can be outside of the image boundary
        xyz_c:
            (*b_shape, *m_shape, *n_shape, 3) xyz in the camera coordinate
    """
    *bn_shape, _ = xyz_w.shape
    *bm_shape, _, _ = intrinsics.shape
    n_shape = bn_shape[dim_b:]
    m_shape = bm_shape[dim_b:]
    assert bn_shape[:dim_b] == bm_shape[:dim_b]
    b_shape = bn_shape[:dim_b]

    H_w2c = rigid_motion.inv_homogeneous_tensors(H_c2w)  # (*b, *m, 4, 4)
    H_w2c = H_w2c.reshape(*b_shape, *m_shape, *([1] * len(n_shape)), 4, 4)  # (*b, *m, *n, 4, 4)
    intrinsics = intrinsics.reshape(*b_shape, *m_shape, *([1] * len(n_shape)), 3, 3)  # (*b, *m, *n, 3, 3)
    xyz_w = torch.cat(
        [
            xyz_w,  # (*b, *n, 3)
            torch.ones(*b_shape, *n_shape, 1, device=xyz_w.device, dtype=xyz_w.dtype),
        ],
        dim=-1,
    )  # (*b, *n, 4)
    xyz_w = xyz_w.reshape(*b_shape, *([1] * len(m_shape)), *n_shape, 4, 1)  # (*b, *m, *n, 4, 1)
    xyz_c = linalg_utils.matmul(H_w2c, xyz_w)  # (*b, *m, *n, 4, 1)
    uvw_c = linalg_utils.matmul(intrinsics, xyz_c[..., :3, :])  # (*b, *m, *n, 3, 1)

    # causing nan if training with uv
    # uv_c = uvw_c[..., :2, 0] / uvw_c[..., 2:3, 0]  # (*b, *m, *n, 2)

    _w = torch.clamp(uvw_c[..., 2:3, 0].abs(), min=1e-9) * ((uvw_c[..., 2:3, 0] >= 0).float() - 0.5) * 2
    # print(f'min_w = {_w.abs().min()} {_w.min()} {_w.max()}')
    uv_c = uvw_c[..., :2, 0] / _w  # (*b, *m, *n, 2)

    if not return_depth_map:
        return uv_c, xyz_c[..., :3, 0]
    else:
        return uv_c, xyz_c[..., :3, 0], _w


@linalg_utils.disable_tf32_and_autocast()
def find_corresponding_uv(
    uv_c: torch.Tensor,
    z_map: torch.Tensor,
    intrinsics_from: torch.Tensor,
    H_c2w_from: torch.Tensor,
    intrinsics_to: torch.Tensor,
    H_c2w_to: torch.Tensor,
    dim_b: int = 0,
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the correspoding points in image coordinates of the pixels in a source image.

    Args:
        uv_c:
            (*b_shape, *n_shape, 2) the source pixels, within image boundary [0, w], [0, h].
            u is along the column axis, v is along the row axis.
            Note that the pixel center is at [x.5, y.5].
        z_map:
            (*b_shape, h, w) the z coordinate of the point in the source camera coordinate on the sensor,
            not along the corresponding camera ray.
        intrinsics_from:
            (*b_shape, 3, 3) the source camera intrinsics
        H_c2w_from:
            (*b_shape, 4, 4) the homegeneous matrix (source camera -> world)
        intrinsics_to:
            (*b_shape, *m_shape, 3, 3) the target camera intrinsics
        H_c2w_to:
            (*b_shape, *m_shape, 4, 4) the homegeneous matrix (target camera -> world)
        dim_b:
            number of dimensions in b.  <= 1

    Returns:
        uv:
            (*b_shape, *m_shape, *n_shape, 2), uv on the images, can be outside of the image boundary.
            Note that the pixel center is at x.5, y.5.
            The projected uv from each of the *n_shape to each of the *m_shape
        xyz_c:
            (*b_shape, *m_shape, *n_shape, 3) xyz in the camera coordinate
    """
    assert dim_b <= 1, "only support dim_b = 0 or dim_b = 1"
    *bn_shape, _ = uv_c.shape
    b_shape = bn_shape[:dim_b]
    n_shape = bn_shape[dim_b:]
    h, w = z_map.size(-2), z_map.size(-1)
    b = int(np.prod(b_shape))
    n = int(np.prod(n_shape))

    # get z_c
    # if dim_b == 0:
    #     z_c = z_map[uv_c[..., 1], uv_c[..., 0]]  # (*b_shape, *n_shape,)
    # elif dim_b == 1:
    #     z_c = []
    #     for b in range(b_shape[0]):
    #         z = z_map[b, uv_c[b][..., 1], uv_c[b][..., 0]]  # (*n_shape,)
    #         z_c.append(z)
    #     z_c = torch.stack(z_c, dim=0)  # (*b_shape, *n_shape)
    # else:
    #     raise NotImplementedError

    z_c = uv_sampling(
        uv=uv_c.reshape(b, n, 2),
        feature_map=z_map.reshape(b, h, w, 1),  # (b, h, w, 1)
        uv_normalized=False,
    )  # (b, n, dim=1)
    z_c = z_c.reshape(*b_shape, *n_shape)

    # project to world coord
    xyz_w = compute_xyz_w_from_uv(
        uv_c=uv_c,  # (*b_shape, *n_shape, 2)
        z_c=z_c,  # (*b_shape, *n_shape)
        intrinsic=intrinsics_from,  # (*b_shape, 3, 3)
        H_c2w=H_c2w_from,  # (*b_shape, 4, 4)
    )  # (*b_shape, *n_shape, 3)

    # map xyz_w to each camera
    uv_cs, xyz_cs = pinhole_projection(
        xyz_w=xyz_w,  # (*b_shape, *n_shape, 3)
        intrinsics=intrinsics_to,  # (*b_shape, *m_shape, 3, 3)
        H_c2w=H_c2w_to,  # (*b_shape, *m_shape, 4, 4)
        dim_b=dim_b,
    )  # (*b_shape, *m_shape, *n_shape, 2), (*b_shape, *m_shape, *n_shape, 3)

    return uv_cs, xyz_cs


def uv_sampling(
    uv: torch.Tensor,  # (b, *p, 2)
    feature_map: torch.Tensor,  # (b, h, w, dim)
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    uv_normalized: bool = True,
):
    """
    Sample the feature map at the uv values.

    Args:
        uv:
            (b, *p, 2)  values between [0, 1] or not normalized [0, w] [0, h].
            note that pixel center at *.5.
        feature_map:
            (b, h, w, dim), boundary corresponded to u=0, u=1, v=0, v=1.  (0,0) at top left, u to right, v to down
        pad_one_outsize:
            If true, values outside feature_map will be set as 1, else 0
        mode:
            mode used by grid_sample
        padding_mode:
            padding mode used by grid_sample. "zeros", "border", "reflection"
        uv_normalized:
            whether uv is normalized to [0, 1]. if None, uv is in the range of [0, w] [0, h]

    Returns:
        resampled_feature:
            (b, *p, dim)
    """

    if not uv_normalized:
        b, h, w, dim = feature_map.shape
        uv = uv.clone()
        uv[..., 0] = uv[..., 0] / w
        uv[..., 1] = uv[..., 1] / h

    # [0, 1] -> [-1, 1] used by grid_sampling
    uv = 2 * uv - 1  # (b, *p, 2)

    b, *p_shape, _2 = uv.shape
    assert _2 == 2
    uv = uv.reshape(b, 1, -1, 2)  # (b, 1, p, 2)

    # (b, h, w, dim) -> (b, dim, h, w)
    feature_map = feature_map.permute(0, 3, 1, 2)  # (b, dim, h, w)

    resampled_feature = torch.nn.functional.grid_sample(
        input=feature_map,  # (b, dim, h, w)
        grid=uv,  # (b, 1, p, 2)
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )
    # (b, dim, 1, p) -> (b, 1, p, dim)
    resampled_feature = resampled_feature.permute(0, 2, 3, 1)  # (b, 1, p, dim)
    resampled_feature = resampled_feature.reshape(b, *p_shape, resampled_feature.size(-1))  # (b, *p, dim)

    return resampled_feature  # (b, *p, dim)


def sparse_uv_sampling(
    uv: torch.Tensor,  # (b, r, *p, 2)
    qidx: torch.Tensor,  # (b, r)  long
    feature_map: torch.Tensor,  # (b, q, h, w, dim)
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    uv_normalized: bool = True,
):
    """
    Perform UV sampling on selected feature_maps only.
    Specifically, uv[b, r] performs 2d uv_sampling on feature_map[b, qidx[b, r]].

    Args:
        uv:
            (b, r *p, 2) float. values between [0, 1] or not normalized ([0, w] [0, h]).
        qidx:
            (b, r) the index of feature_map at the "q" dimension
        feature_map:
            (b, q, h, w, dim), boundary corresponded to u=0, u=1, v=0, v=1.
            (0,0) at top left, u to right, v to down.
        mode:
            mode used by grid_sample
        padding_mode:
            padding mode used by grid_sample. "zeros", "border", "reflection"
        uv_normalized:
            whether uv is normalized to [0, 1]. if False, uv is in the range of [0, w] [0, h]

    Returns:
        resampled_feature:
            (b, r, *p, dim)

    Notes:
        The function used 5D grid_sample to fake sparse index selection.
        To select the qidx, the w (depth) index is set to the corresponding
        pixel center.
    """

    b, q, h, w, dim = feature_map.shape

    if not uv_normalized:
        uv = uv.clone()
        uv[..., 0] = uv[..., 0] / w
        uv[..., 1] = uv[..., 1] / h

    # [0, 1] -> [-1, 1] used by grid_sampling
    uv = 2 * uv - 1  # (b, r *p, 2)
    b, r, *p_shape, _2 = uv.shape
    p = int(np.prod(p_shape))
    assert _2 == 2
    assert qidx.shape == (b, r)

    # create qq (depth) from qidx
    qi = qidx.float() + 0.5  # shift to pixel center 0.5, 1.5,  (b, r)
    qi = qi * (2 / q) - 1  # [0, q] -> [0, 1] -> [-1, 1]   (b, r)
    uvw = torch.cat(
        [
            uv,  # (b, r, *p, 2)
            qi.reshape(b, r, *([1] * len(p_shape)), 1).expand(b, r, *p_shape, 1),
        ],
        dim=-1,
    )  # (b, r *p, 3)
    uvw = uvw.reshape(b, r, 1, p, 3)  # (b, r, 1, p, 3)

    # (b, q, h, w, dim) -> (b, dim, q, h, w)
    feature_map = feature_map.permute(0, 4, 1, 2, 3)  # (b, dim, q, h, w)

    resampled_feature = torch.nn.functional.grid_sample(
        input=feature_map,  # (b, dim, q, h, w)
        grid=uvw.to(dtype=feature_map.dtype),  # (b, r, 1, p, 3)
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )
    # (b, dim, r, 1, p) -> (b, r, 1, p, dim)
    resampled_feature = resampled_feature.permute(0, 2, 3, 4, 1)  # (b, r, 1, p, dim)
    resampled_feature = resampled_feature.reshape(b, r, *p_shape, dim)  # (b, r, *p, dim)

    return resampled_feature  # (b, r, *p, dim)


@linalg_utils.disable_tf32_and_autocast()
def compute_3d_zdir_and_dps(
    z_map: torch.Tensor,
    intrinsic: torch.Tensor,
    H_c2w: torch.Tensor,
    subsample: int = 1,
):
    """
    Compute the zdir and dps in the world coordinate using z_map and camera pose.

    Args:
        z_map:
            (*, h, w) the z coordinate of the point in the camera coordinate on the sensor,
            not along the corresponding camera ray.
        intrinsic:
            (*, 3, 3) camera intrinsic matrix
        H_c2w:
            (*, 4, 4) homegeneous matrix that convert camera coord to world coord.
            Note that the y axis should be inverted in the cam_poses.
        subsample: int
            index stride
        other_maps:
            a list of (*, h, w, d) to associated with each point

    Returns:
        zdir_w:
            (*, h//subsample, w//subsample, 3) camera z direction in world coordinates
        dps_w:
            (*, h//subsample, w//subsample, 1) distance per sample in world coordinates
        dps_uw:
            (*, h//subsample, w//subsample, 3) distance per sample in u direction in world coordinates
        dps_vw:
            (*, h//subsample, w//subsample, 3) distance per sample in v direction in world coordinates
    """

    dtype = z_map.dtype
    device = z_map.device
    h, w = z_map.size(-2), z_map.size(-1)

    # generate u v w on the sensor coord
    u, v = torch.meshgrid(
        torch.arange(0, w, subsample, device=device),
        torch.arange(0, h, subsample, device=device),
        indexing="xy",
    )  # u: (h', w') for x,  v: (h', w') for y in the sensor coord

    z_map = z_map[..., v, u]  # (*, h', w')
    valid_z = z_map < 1e11  # default background z is set to 1e12
    inv_intrinsic = torch.linalg.inv(intrinsic).unsqueeze(-3).unsqueeze(-3)  # (*, 1, 1, 3, 3)

    # get distance per sample in u/v direction;
    dps_u = torch.stack((subsample * z_map, 0 * z_map), dim=-1)  # (*, h, w, 2) # distance per sample in u direction
    dps_v = torch.stack((0 * z_map, subsample * z_map), dim=-1)  # (*, h, w, 2) # distance per sample in v direction
    dps_uv = torch.stack((dps_u, dps_v), dim=-1)

    dps_uvc = linalg_utils.matmul(inv_intrinsic[..., :2, :2], dps_uv)  # (*, h', w', 2, 2) distance in cam cord

    dps_uvc_shape = list(dps_uvc.shape)
    dps_uvc_shape[-2] = 1
    # dps_uvc_shape[-1] = 2

    dps0_uvc = torch.cat(
        (dps_uvc, torch.zeros(*dps_uvc_shape, dtype=dtype, device=device)),
        dim=-2,
    )  # (*, h', w', 3, 2)
    dps_uvw = linalg_utils.matmul(
        H_c2w[..., :3, :3].unsqueeze(-3).unsqueeze(-3),
        dps0_uvc,
    )  # (*, h', w', 3, 2)
    dps_uw = dps_uvw[..., 0]  # (*, h', w', 3)
    dps_vw = dps_uvw[..., 1]  # (*, h', w', 3)
    dps_w = torch.norm(dps_uw, dim=-1, keepdim=True)  # (*, h', w',1) # assume the same for u and v

    # if not hit, dps is set to 0
    dps_uw = dps_uw * valid_z.unsqueeze(-1)
    dps_vw = dps_vw * valid_z.unsqueeze(-1)
    dps_w = dps_w * valid_z.unsqueeze(-1)  # norm

    # get z direction for the camera
    one_shape = dps_uvc_shape
    one_shape[-1] = 1
    zero_shape = copy.deepcopy(one_shape)
    zero_shape[-2] = 2

    zdir_c = torch.cat(
        (
            torch.zeros(*zero_shape, dtype=dtype, device=device),
            torch.ones(*one_shape, dtype=dtype, device=device),
        ),
        dim=-2,
    )  # (*, h', w', 3, 1)
    zdir_w = linalg_utils.matmul(
        H_c2w[..., :3, :3].unsqueeze(-3).unsqueeze(-3),
        zdir_c,
    )  # (*, h', w', 3, 1)
    zdir_w = zdir_w[..., 0]  # (*, h', w', 3)

    return dict(
        zdir_w=zdir_w,
        dps_w=dps_w,
        dps_uw=dps_uw,
        dps_vw=dps_vw,
    )


def plot_points_and_rays(
    points: torch.Tensor,  # (n, 3)
    ray_origins: torch.Tensor,  # (m, 3)
    ray_directions: torch.Tensor,  # (m, 3)
    ray_lengths: T.Union[torch.Tensor, float] = 10.0,  # (m, 3) or float
    special_points: torch.Tensor = None,  # (p, 3)
    fig=None,
    point_size: float = 0.1,
    point_alpha: float = 0.1,
    special_point_size: float = 0.5,
    ray_color: T.List[float] = "b",
    ray_linewidth: float = 0.1,
):
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        xs=points[:, 0],
        ys=points[:, 1],
        zs=points[:, 2],
        s=point_size,
        alpha=point_alpha,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # plot a ray
    t = ray_lengths
    xs_from, xs_to = (
        ray_origins[..., 0],
        ray_origins[..., 0] + t * ray_directions[..., 0],
    )  # (n,)
    ys_from, ys_to = (
        ray_origins[..., 1],
        ray_origins[..., 1] + t * ray_directions[..., 1],
    )  # (n,)
    zs_from, zs_to = (
        ray_origins[..., 2],
        ray_origins[..., 2] + t * ray_directions[..., 2],
    )  # (n,)

    for i in range(len(xs_from)):
        ax.plot(
            xs=[xs_from[i], xs_to[i]],
            ys=[ys_from[i], ys_to[i]],
            zs=[zs_from[i], zs_to[i]],
            color=ray_color,
            linewidth=ray_linewidth,
        )
    # plot origin
    ax.scatter(
        xs=ray_origins[..., 0],
        ys=ray_origins[..., 1],
        zs=ray_origins[..., 2],
        s=point_size,
    )
    # plot intersection
    if special_points is not None:
        ax.scatter(
            xs=special_points[..., 0],
            ys=special_points[..., 1],
            zs=special_points[..., 2],
            s=special_point_size,
            c="r",
            marker="x",
        )

    ax.axes.set_xlim3d(left=-1, right=1)
    ax.axes.set_ylim3d(bottom=-1, top=1)
    ax.axes.set_zlim3d(bottom=-1, top=1)

    return fig, ax


@linalg_utils.disable_tf32_and_autocast()
def generate_camera_rays(
    cam_poses: torch.Tensor,  # (m, 4, 4) target camera pose, cam to world
    intrinsics: torch.Tensor,  # (m, 3, 3)  intrinsic matrix of the camrea
    width_px: int,
    height_px: int,
    subsample: int = 1,  # only trace 1 ray every subsample sensor pixels
    offsets: T.Union[float, str, torch.Tensor] = "center",
    use_quick_inv_intrinsic: bool = False,
    normalize_ray_direction: bool = True,
    device: torch.device = None,
):
    """
    Generate camera rays (origin and direction) in the world coordinate.
    The function reproduces `o3d.t.geometry.RaycastingScene.create_rays_pinhole`
    when offset = ''center.

    Args:
        cam_poses:
            (m, 4, 4) homegeneous matrix that transforms the camera coord to world coord
            Note that to use the rays to render an image, you can have the y axis inverted in the cam_poses.
        intrinsics:
            (m, 3, 3) intrinsic matrix for each camera pose
        width_px:
            number of pixels on the sensor (before subsample)
        height_px:
            number of pixels on the sensor (before subsample)
        subsample:
            subsample the sensor (camera ray)
        offsets:
            float or (m, h, w) that will be added to the pixel location on the sensor
            If 0 or 'center', ray will be coming from the center of a pixel
            'rand': offset = [-0.5, 0.5)
        normalize_ray_direction:
            if True, we return normalized ray direction
        device:

    Returns:
        ray_origins_w:  (m, h', w', 3)
        ray_directions_w:  (m, h', w', 3)  normalized to unit norm

    Note:
        The function does not flip the y axis (which should be handled by the image coordinate)
        To use the rays to render an image, you can have the y axis inverted in the cam_poses.
    """

    if device is None:
        device = cam_poses.device

    m = cam_poses.size(0)

    # generate u v w on the sensor coord
    u, v = torch.meshgrid(
        torch.arange(0, width_px, subsample, dtype=cam_poses.dtype, device=device),
        torch.arange(0, height_px, subsample, dtype=cam_poses.dtype, device=device),
        indexing="xy",
    )  # u: (h', w') for x,  v: (h', w') for y in the sensor coord
    u = u + 0.5  # (h', w')
    v = v + 0.5  # (h', w')

    # uv1 = torch.stack((u, v, torch.ones_like(u)), dim=-1).to(dtype=cam_poses.dtype)  # (h', w', 3)
    # uv1_shape = uv1.shape  # (h', w', 3)
    # uv1 = uv1.expand(m, *uv1_shape).unsqueeze(-1)  # (m, h', w', 3, 1)
    # if isinstance(offsets, str):
    #     if offsets == 'center':
    #         pass
    #     elif offsets == 'rand':
    #         offsets = torch.rand_like(uv1) - 0.5
    #         offsets[..., 2, 0] = 0
    #         uv1 = uv1 + offsets
    #     else:
    #         raise NotImplementedError
    # elif isinstance(offsets, torch.Tensor):
    #     # given the 2d uv offset (m, 2)
    #     offsets = torch.cat([offsets, torch.zeros([m, 1], dtype=offsets.dtype, device=device)], dim=1)
    #     uv1 = uv1 + offsets.unsqueeze(1).unsqueeze(2).unsqueeze(-1) #m(m, 1, 1, 3 ,1)
    # else:
    #     raise NotImplementedError
    #
    # # compute the inverse of the intrinsic matrices
    # inv_intrinsics = torch.linalg.inv(intrinsics)  # (m, 3, 3)
    # inv_intrinsics = inv_intrinsics.reshape(m, 1, 1, 3, 3)  # (m, 1, 1, 3, 3)
    # ray_directions_c = inv_intrinsics @ uv1  # (m, h', w', 3, 1)
    #
    # # cam coord -> world coord
    # cam_poses = cam_poses.reshape(m, 1, 1, 4, 4)  # (m, 1, 1, 4, 4)
    # ray_directions_w = cam_poses[..., :3, :3] @ ray_directions_c  # (m, h', w', 3, 1), not normalized
    # ray_origins_w = cam_poses[..., :3, 3].expand(m, *uv1_shape)  # (m, h', w', 3)
    #
    # # normalize direction
    # ray_directions_w = ray_directions_w.squeeze(-1)  # (m, h', w', 3)
    # ray_directions_w = ray_directions_w / torch.linalg.vector_norm(ray_directions_w, dim=-1, keepdims=True)  # (m, h', w', 3)
    #
    # return ray_origins_w, ray_directions_w

    uv = torch.stack((u, v), dim=-1).to(dtype=cam_poses.dtype)  # (h', w', 2)
    uv_shape = uv.shape  # (h', w', 2)
    uv = uv.expand(m, *uv_shape)  # (m, h', w', 2)
    if isinstance(offsets, str):
        if offsets == "center":
            pass
        elif offsets == "rand":
            offsets = torch.rand_like(uv) - 0.5  # (m, h', w', 2)
            uv = uv + offsets
        else:
            raise NotImplementedError
    elif isinstance(offsets, torch.Tensor):
        # given the 2d uv offset (m, 2)
        uv = uv + offsets.to(dtype=uv.dtype, device=device).unsqueeze(1).unsqueeze(2)  # (m, 1, 1, 2)
    else:
        raise NotImplementedError

    return generate_camera_rays_from_uv(
        cam_poses=cam_poses,
        intrinsics=intrinsics,
        uv=uv,  # (m, h', w', 2)
        use_quick_inv_intrinsic=use_quick_inv_intrinsic,
        normalize_ray_direction=normalize_ray_direction,
        device=device,
    )


@linalg_utils.disable_tf32_and_autocast()
def generate_camera_rays_from_uv(
    cam_poses: torch.Tensor,  # (m, 4, 4) target camera pose, cam to world
    intrinsics: torch.Tensor,  # (m, 3, 3)  intrinsic matrix of the camrea
    uv: torch.Tensor,  # (m, *p, 2)
    use_quick_inv_intrinsic: bool = False,
    normalize_ray_direction: bool = True,
    device=torch.device("cpu"),
) -> T.Union[torch.Tensor, torch.Tensor]:
    """
    Generate camera rays (origin and direction) in the world coordinate given
    uv coordinate on the image. The uv coordinate on image is origin at top left,
    u to right, v to bottom.

    Args:
        cam_poses:
            (m, 4, 4) homegeneous matrix that transforms the camera coord to world coord
            Note that to use the rays to render an image, you can have the y axis inverted in the cam_poses.
        intrinsics:
            (m, 3, 3) intrinsic matrix for each camera pose
        width_px:
            number of pixels on the sensor (before subsample)
        height_px:
            number of pixels on the sensor (before subsample)
        uv:
            uv coordinate.  uv[..., 0]: u coordinate [0, w], uv[..., 1]: v coordinate [0, h],
        device:

    Returns:
        ray_origins_w:  (m, *p, 3)
        ray_directions_w:  (m, *p, 3)  normalized to unit norm

    Note:
        The function does not flip the y axis (which should be handled by the image coordinate)
        To use the rays to render an image, you can have the y axis inverted in the cam_poses.
    """

    m = cam_poses.size(0)
    _m, *p_shape, _2 = uv.shape
    assert m == _m
    assert _2 == 2

    uv1 = torch.cat((uv, torch.ones(_m, *p_shape, 1, dtype=uv.dtype, device=uv.device)), dim=-1).to(
        dtype=cam_poses.dtype, device=device
    )  # (m, *p, 3)

    # compute the inverse of the intrinsic matrices
    if use_quick_inv_intrinsic:
        inv_intrinsics = rigid_motion.inv_intrinsic_tensors(intrinsics.to(device=device))  # (m, 3, 3)
    else:
        inv_intrinsics = torch.linalg.inv(intrinsics.to(device=device))  # (m, 3, 3)
    inv_intrinsics = inv_intrinsics.reshape(m, *([1] * len(p_shape)), 3, 3)  # (m, 1, 1, 3, 3)
    ray_directions_c = linalg_utils.matmul(inv_intrinsics, uv1.unsqueeze(-1))  # (m, *p, 3, 1)

    # cam coord -> world coord
    cam_poses = cam_poses.reshape(m, *([1] * len(p_shape)), 4, 4).to(device=device)  # (m, *p, 4, 4)
    ray_directions_w = linalg_utils.matmul(cam_poses[..., :3, :3], ray_directions_c)  # (m, *p, 3, 1), not normalized
    ray_origins_w = cam_poses[..., :3, 3].clone().expand(m, *p_shape, 3)  # (m, *p, 3)
    ray_directions_w = ray_directions_w.squeeze(-1)  # (m, *p, 3)

    if normalize_ray_direction:
        # normalize direction
        ray_directions_w = ray_directions_w / torch.linalg.vector_norm(
            ray_directions_w, dim=-1, keepdims=True
        )  # (m, *p, 3)

    return ray_origins_w, ray_directions_w


def sample_regions_camera_rays_and_features(
    cam_poses: torch.Tensor,  # (m, 4, 4) target camera pose, cam to world
    intrinsics: torch.Tensor,  # (m, 3, 3)  intrinsic matrix of the camera
    sample_center: torch.Tensor,
    region_width_px: int,
    region_height_px: int,
    features: T.Dict[str, torch.Tensor] = None,
    device=torch.device("cpu"),
):
    """
     used in PCD version of inverse rendering

    Args:
        cam_poses:
        intrinsics:
        sample_center:
        region_width_px:
        region_height_px:
        features:
        device:

    Returns:

    """
    uv_c, _ = pinhole_projection(
        xyz_w=sample_center,
        intrinsics=intrinsics,
        H_c2w=cam_poses,
    )
    uv_c = torch.squeeze(uv_c)  # (m, 2)

    uv_offset = uv_c - torch.Tensor([region_width_px / 2, region_height_px / 2]).to(device=device).unsqueeze(0)

    ray_origins_w, ray_directions_w = generate_camera_rays(
        cam_poses=cam_poses,
        intrinsics=intrinsics,
        width_px=region_width_px,
        height_px=region_height_px,
        subsample=1,
        offsets=uv_offset,
        device=device,
    )  # (b*m, h, w, 3), (b*m, h, w, 3) normalized

    u, v = torch.meshgrid(
        torch.arange(0, region_width_px, 1, device=device),
        torch.arange(0, region_height_px, 1, device=device),
        indexing="xy",
    )  # u: (h', w') for x,  v: (h', w') for y in the sensor coord
    u = u + 0.5
    v = v + 0.5

    if features is not None:
        resampled_features = dict()
        for key in features.keys():
            feature = features[key]
            assert len(feature.shape) >= 3, "dimension of feature should contain at least n, h, w "
            if len(feature.shape) == 3:
                feature = feature.unsqueeze(-1)

            # the part below should be replaced by another function resample_uv

            b, width_px, height_px, _ = feature.shape
            feature = feature.transpose(1, 3)

            grid = torch.stack([u, v], dim=2).unsqueeze(0) + uv_offset.unsqueeze(1).unsqueeze(2)  # (m, h, w, 2)
            grid = grid / torch.Tensor([width_px / 2, height_px / 2]).to(device=device) - 1  # normalize to [-1,1]

            # grid sample: in default is a bilinear interpolator
            # see https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            resampled_feature = torch.nn.functional.grid_sample(feature, grid)
            resampled_feature = resampled_feature.transpose(1, 3)
            resampled_features[key] = resampled_feature

    out_dict = dict(
        ray_origins=ray_origins_w,
        ray_directions=ray_directions_w,
    )

    if features is not None:
        out_dict["resampled_features"] = resampled_features

    return out_dict


def plot_multiple_images(
    imgs: T.Union[torch.Tensor, np.ndarray],
    dpi=150,
    mode="tile",  # 'horizontal', 'vertical', 'tile'
    fig=None,
    ax=None,
    colorbar=True,
    valrange=None,  # (min, max)
    ncols: int = 6,
    background_color: float = 0.0,
    vmin: float = None,
    vmax: float = None,
):
    """Plot multiple images by concatenate them in space."""
    # imgs: (b, h, w, *)

    if isinstance(imgs, np.ndarray):
        imgs = torch.from_numpy(imgs)

    if isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().cpu()

    mask = torch.logical_not(imgs.isfinite())
    imgs = imgs.masked_fill(mask, 0)

    imgs_list = torch.chunk(imgs, chunks=imgs.shape[0], dim=0)
    imgs_list = [img[0] for img in imgs_list]  # list of (h, w, *)

    # print(f'imgs_list shape:')
    # for i in range(len(imgs_list)):
    #     print(f'{imgs_list[i].shape}')

    if mode == "horizontal":
        imgs = torch.cat(imgs_list, dim=1)
    elif mode == "vertical":
        imgs = torch.cat(imgs_list, dim=0)
    elif mode == "tile":
        assert len(imgs_list) > 0
        if imgs_list[0].ndim == 2:
            squeeze_last_dim = True
            imgs_list = [img.unsqueeze(-1) for img in imgs_list]  # list of (h, w, c)
        else:
            squeeze_last_dim = False
        imgs = render.tile_images(
            images=imgs_list,
            ncols=ncols,
            background_color=background_color,
        )
        if squeeze_last_dim:
            imgs = imgs.squeeze(-1)
    else:
        raise NotImplementedError

    if valrange is not None:
        imgs = (imgs - valrange[0]) / (valrange[1] - valrange[0])

    imgs = imgs.detach().cpu().numpy()

    # plot
    fig, axes = imagesc(
        arr=imgs,
        fig=fig,
        axes=ax,
        dpi=dpi,
        colorbar=colorbar,
        vmin=vmin,
        vmax=vmax,
    )
    plt.axis("scaled")
    return fig, axes


def fit_hyperplane(
    points: torch.Tensor,  # (*, n, d)
    centers: torch.Tensor = None,  # (*, d)
    weights: torch.Tensor = None,  # (*, n)
    th_eig_val: float = 1.0e-3,
) -> T.Dict[str, T.Any]:
    r"""
    Fit a hyperplane for the n points such that the plane minimizes the point-to-plane distances.

    .. math::

        \min_{n,c}  \sum_i w_i (n^T * (p_i - c))^2,

    where :math:`n` is the normal of the hyperplane, :math:`c` is the anchor of the plane.

    Args:
        points:
            (*, n, d) the points in d-dimensinoal space
        centers:
            (*, d) the given anchors of the hyperplanes
        weights:
            (*, n) the weight to each points.  If None, all points have unit weights.
        # ray_origins:
        #     (*, d) the ray origins.  If given, we will also compute the intersection of the ray on the plane.
        # ray_directions:
        #     (*, d) the ray directions. If given, we will also compute the intersection of the ray on the plane.

    Returns:
        plane_normals:
            (*, d)  Note that the plane normal can points to one of the two directions
        centers:
            (*, d)
        ts:
            (*,) or None.  ts on the ray to the plane.

    """
    *p_shape, d = points.shape
    if weights is None:
        weights = torch.ones(*p_shape, 1, dtype=points.dtype, device=points.device)  # (*, n, 1)
    else:
        weights = weights.unsqueeze(-1)  # (*, n, 1)

    if centers is None:
        centers = (points * weights).sum(dim=-2) / weights.sum(dim=-2)  # (*, d)

    # print(f'centers.shape = {centers.shape}')
    # print(f'points.shape = {points.shape}')
    # print(f'weights.shape = {weights.shape}')

    points_centered = points - centers.unsqueeze(-2)  # (*, n, d)
    PTP = linalg_utils.matmul(
        (weights * points_centered).transpose(-1, -2),
        points_centered,
    )  # (*, d, d)

    # we will make sure PTP is at least rank 2 (at least two points not on the same line)
    with torch.no_grad():
        eig_vals, eig_vecs = torch.linalg.eigh(PTP)  # eig_vecs: (*, d, d),  eig_vals: (*, d) small to large
        valid_mask = eig_vals[..., -2] > th_eig_val  # (*,)
        valid_mask = torch.logical_and(
            valid_mask,
            (eig_vals[..., 1] - eig_vals[..., 0]) > th_eig_val,
        )  # (*,)

    # # fill invalid points_centered with random points to avoid degenerative
    # points_centered = points_centered.masked_fill(
    #     ~valid_mask.unsqueeze(-1).unsqueeze(-1),
    #     0,
    # ) + torch.rand_like(points_centered).masked_fill(
    #     valid_mask.unsqueeze(-1).unsqueeze(-1),
    #     0,
    # )  # (*, n, d)
    # PTP = (weights * points_centered).transpose(-1, -2) @ points_centered  # (*, d, d)

    # fill PTP with full rank matrix if invalid
    PTP = PTP.masked_fill(~valid_mask.unsqueeze(-1).unsqueeze(-1), 0) + torch.randn_like(PTP).masked_fill(
        valid_mask.unsqueeze(-1).unsqueeze(-1), 0
    )

    # surface normal is the smallest eigen-vector of PTP
    eig_vals, eig_vecs = torch.linalg.eigh(PTP)  # eig_vecs: (*, d, d),  eig_vals: (*, d)
    plane_normals = eig_vecs[..., 0]  # (*, d)

    # # check whether the normal is unique
    # valid_mask = torch.logical_and(
    #     (eig_vals[..., 1] - eig_vals[..., 0]) > th_eig_val,
    #     eig_vals[..., 0] > 0,
    # )  # (*,)

    return dict(
        plane_normals=plane_normals,  # (*, d)
        centers=centers,  # (*, d)
        # eig_vals=eig_vals,  # (*, d) in ascending order
        valid_mask=valid_mask,  # (*,) bool.  True: valid to use
    )


@linalg_utils.disable_tf32_and_autocast()
def plane_ray_intersection(
    plane_centers: torch.Tensor,  # (*, d)
    plane_normals: torch.Tensor,  # (*, d)
    ray_origins: torch.Tensor,  # (*, d)
    ray_directions: torch.Tensor,  # (*, d)
) -> torch.Tensor:
    """
    Compute the intersection between plane_i and ray_i.

    Returns:
        ts: (*,)
    """

    nt_d = (ray_directions * plane_normals).sum(dim=-1)  # (*,)
    co = plane_centers - ray_origins  # (*, d)
    nt_co = (plane_normals * co).sum(dim=-1)  # (*,)
    mask = nt_d.abs() < 1.0e-8  # (*,)
    ts = nt_co / nt_d  # (if nt_d == 0 -> t = inf)
    ts = ts.masked_fill(mask, torch.inf)
    return ts  # (*,)


def baseline_pcd_ray_intersection(
    points_w: np.ndarray,  # (n, 3)
    cam_poses: np.ndarray,  # (m, 4, 4) target camera pose, cam to world
    intrinsics: np.ndarray,  # (m, 3, 3)  intrinsic matrix of the camrea
    width_px: int,
    height_px: int,
    k: int,
    method: str = "alpha",  # 'poisson', 'alpha', 'ball'
    poisson_depth: int = 9,
    alpha: float = 0.01,
    ball_radii: T.List[float] = (0.005, 0.01, 0.02, 0.04),
    points_rgb: np.ndarray = None,  # (n, 3)
):
    """
    Baseline point cloud-ray intersection.

    Algorithm:
    1. poisson surface reconstruction from point cloud to create a mesh
    2. ray tracing using the mesh to get surface normal

    Returns:
        mesh:
        est_ts_w:
            (m, h, w) distance on the ray direction to the intersection point
        est_surface_normals_w:
            (m, h, w, 3) surface normal at the intersection point, in the world coordinate
        est_hits:
            (m, h, w)  whether the ray intersect with a surface
    """

    # create pcd
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_w)
    # if points_rgb is not None:
    #    pcd.rgbs = o3d.utility.Vector3dVector(points_rgb)

    # estimate normal at each vertex
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(k=k)

    # create mesh from pcd
    if method == "poisson":
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)
    elif method == "alpha":
        tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha, tetra_mesh, pt_map)
    elif method == "ball":
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(ball_radii)
        )
    else:
        raise NotImplementedError

    # create ray tracing scene
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    extrinsic_matrices = [
        rigid_motion.RigidMotion.invert_homogeneous_matrix(cam_poses[i]) for i in range(cam_poses.shape[0])
    ]

    # ray trace to get surface normal (using ray tracing)
    all_rays = []  # (h, w, 6)  [ox, oy, oz, dx, dy, dz]
    all_surface_normals = []  # (h, w, 3)  [dx, dy, dz]
    all_ts = []  # (h, w)  inf if not hit
    for i in range(cam_poses.shape[0]):
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=intrinsics[i],
            extrinsic_matrix=extrinsic_matrices[i],
            width_px=width_px,
            height_px=height_px,
        )  # (height_px, width_px, 6)  [ox, oy, oz, dx, dy, dz]  origin is the pinhole

        # cast the rays, get the intersections
        raycast_results = scene.cast_rays(rays)
        t_hits = raycast_results["t_hit"]  # (height_px, width_px), inf if not hit the mesh
        hit_map = 1 - np.isinf(raycast_results["t_hit"].numpy())  # (h, w)  1 if hit a surface, 0 otherwise

        # note that primitive_normals is the normal of the triangle face
        # we can use uv map to interpolate vertex normal
        # interpolate surface normal using uv map
        if mesh.has_vertex_normals():
            surface_normals = render.interp_surface_normal_from_ray_tracing_results(
                mesh=mesh,
                raycast_results=raycast_results,
            )  # (height_px, width_px, 3)
        else:
            surface_normals = raycast_results["primitive_normals"].numpy()  # (height_px, width_px, 3)

        # if not hit a surface, set surface normal to (0, 0, 0)
        surface_normals = surface_normals * np.expand_dims(hit_map, axis=-1)  # (h, w, 3)

        # make sure surface normal points in the opposite direction of the ray
        rays = rays.numpy()
        est_normal_sign = np.sign(np.sum(surface_normals * rays[..., 3:], axis=-1, keepdims=True))  # (h, w, 1)
        surface_normals = surface_normals * (-1 * est_normal_sign)

        all_rays.append(rays)
        all_surface_normals.append(surface_normals)
        all_ts.append(t_hits.numpy())

    all_ts = np.stack(all_ts, axis=0)  # (n_target_img, h, w)
    all_surface_normals = np.stack(all_surface_normals, axis=0)  # (n_target_img, h, w, 3)
    all_ray_hits = np.isfinite(all_ts)  # (n_target_img, h, w)

    return dict(
        pcd=pcd,
        mesh=mesh,
        est_ts_w=all_ts,  # (m, h, w)
        est_surface_normals_w=all_surface_normals,  # (m, h, w ,3)
        est_hits=all_ray_hits,  # (m, h, w)
    )


@linalg_utils.disable_tf32_and_autocast()
def generate_camera_circle_path(
    num_poses: int,
    d_to_origin: float,
    r_circle: float,
    center_angles: T.Union[torch.Tensor, np.ndarray, T.List[float]],
    invert_yz: bool = True,
    alt_yaxis: bool = False,
) -> T.Union[torch.Tensor, np.ndarray]:
    """
    Generate a camera path that looks at the world origin
    Args:
        num_poses:
            number of camera poses sampled on the circle
        d_to_origin:
            distance to the origin
        r_circle:
            radius of the circle
        center_direction:
            (2,) theta (angle between x-axis), phi (angle between xy plane),
            the viewing direction of the center of the circle.  All in degree.
            The angles are given in the final coordinate (after yz is inverted)
        invert_yz:
            whether to invert the direction of y axis and z axis (since images y coord is flipped)
            This is to account for the difference in the image coordinate (x to right, y to down, z to far)
            and the world/opengl coordinate (x to right, y to up, z to us)
        alt_yaxis:
            an option to use an alternative definition of yaxis and makes a more stable circular path
    Returns:
        (num_poses, 4, 4) camera poses (that converts camera coord to world coords)
    """

    if isinstance(center_angles, np.ndarray):
        center_angles = torch.from_numpy(center_angles).float()
    elif isinstance(center_angles, (list, tuple)):
        center_angles = torch.tensor(center_angles).float()

    center_angles = center_angles.float()

    if invert_yz:
        # the coordinate is currently pre-yz-inverted
        # but center_angles are given after yz-inverted
        center_angles = -1 * center_angles

    # generate a circle on the xy plane (i.e., on the plane z = d_to_origin)
    thetas = torch.linspace(0, torch.pi * 2, num_poses) + torch.pi  # (n,)
    cam_positions_c = torch.stack(
        [
            torch.cos(thetas) * float(r_circle),
            torch.sin(thetas) * float(r_circle),
            torch.ones(num_poses) * float(d_to_origin),
        ],
        dim=1,
    )  # (n, 3)

    # print(f'cam_positions_c.shape = {cam_positions_c.shape}')

    # rotate the camera positions
    v1 = torch.tensor([0, 0, 1], dtype=torch.float)
    v2 = torch.stack(
        [
            torch.cos(center_angles[1] * torch.pi / 180.0) * torch.cos(center_angles[0] * torch.pi / 180.0),
            torch.cos(center_angles[1] * torch.pi / 180.0) * torch.sin(center_angles[0] * torch.pi / 180.0),
            torch.sin(center_angles[1] * torch.pi / 180.0),
        ],
        dim=0,
    )

    # print(f'v1.shape = {v1.shape}')
    # print(f'v2.shape = {v2.shape}')

    R = rigid_motion.get_min_R(
        v1=v1,
        v2=v2,
    )  # (3,3)   v2 = R @ v1

    # print(f'R = {R}')

    cam_positions_w = linalg_utils.matmul(
        R.unsqueeze(0),
        cam_positions_c.unsqueeze(-1),
    ).squeeze(-1)  # (n, 3)

    # create camera coordinate
    if not alt_yaxis:
        # all cameras look at the origin of the world -> -cam_positions_w
        ys = torch.zeros_like(cam_positions_w)
        ys[..., 1] = 1
    else:
        # the above implementation will make y-axis flip in a larger transform
        ys = torch.zeros_like(cam_positions_w)
        ys[..., 2] = 1
        ys = linalg_utils.matmul(
            R.unsqueeze(0),
            ys.unsqueeze(-1),
        ).squeeze(-1)  # (n, 3)

    Rs_c2w = rigid_motion.construct_coord_frame(
        z=-1 * cam_positions_w,  # (n, 3)
        y=ys,  # (n, 3, 3)
    )

    *b_shape, a, b = Rs_c2w.shape
    Hs_c2w = torch.zeros(*b_shape, 4, 4)
    Hs_c2w[..., :3, :3] = Rs_c2w
    Hs_c2w[..., :3, 3] = cam_positions_w
    Hs_c2w[..., 3, 3] = 1

    if invert_yz:
        H = torch.eye(4)
        H[1, 1] = -1.0
        H[2, 2] = -1.0
        Hs_c2w = linalg_utils.matmul(H.unsqueeze(0), Hs_c2w)

    return Hs_c2w  # (n, 4, 4)


@linalg_utils.disable_tf32_and_autocast()
def generate_camera_rect_path(
    num_poses: int,
    d_to_origin: float,
    x_length: float,
    y_length: float,
    center_angles: T.Union[torch.Tensor, np.ndarray, T.List[float]],
    x_center: float = 0,
    y_center: float = 0,
    invert_yz: bool = True,
    alt_yaxis: bool = False,
) -> T.Union[torch.Tensor, np.ndarray]:
    """
    Generate a rect camera path that looks at x_center, y_center
    Args:
        num_poses:
            number of camera poses sampled on the rect
        d_to_origin:
            distance to the origin
        x_length:
            length of x side
        y_length:
            length of y side
        x_center:
            center of x side
        y_center:
            center of y side
        center_direction:
            (2,) theta (angle between x-axis), phi (angle between xy plane),
            the viewing direction of the center of the circle.  All in degree.
            The angles are given in the final coordinate (after yz is inverted)
        invert_yz:
            whether to invert the direction of y axis and z axis (since images y coord is flipped)
            This is to account for the difference in the image coordinate (x to right, y to down, z to far)
            and the world/opengl coordinate (x to right, y to up, z to us)
        alt_yaxis:
            an option to use an alternative definition of yaxis and makes a more stable circular path
    Returns:
        (num_poses, 4, 4) camera poses (that converts camera coord to world coords)
    """

    if isinstance(center_angles, np.ndarray):
        center_angles = torch.from_numpy(center_angles).float()
    elif isinstance(center_angles, (list, tuple)):
        center_angles = torch.tensor(center_angles).float()

    center_angles = center_angles.float()

    if invert_yz:
        # the coordinate is currently pre-yz-inverted
        # but center_angles are given after yz-inverted
        center_angles = -1 * center_angles

    # generate a circle on the xy plane (i.e., on the plane z = d_to_origin)
    thetas = torch.linspace(0, torch.pi * 2, num_poses) + torch.pi  # (n,)

    step_size = 2 * (x_length + y_length) / num_poses

    # intitial poses: start from (0, y_length/2), keep increasing
    # bending the poses four times

    all_poses = np.stack([np.arange(num_poses) * step_size, np.ones(num_poses) * y_length / 2], axis=-1)
    bend_ids = np.where(all_poses[:, 0] > (x_length / 2))[0]
    all_poses[bend_ids] = np.stack(
        [
            np.ones(len(bend_ids)) * x_length / 2,
            y_length / 2 + (-all_poses[bend_ids, 0] + x_length / 2),
        ],
        axis=-1,
    )
    bend_ids = np.where(all_poses[:, 1] < (-y_length / 2))[0]
    all_poses[bend_ids] = np.stack(
        [
            x_length / 2 + (all_poses[bend_ids, 1] + y_length / 2),
            -np.ones(len(bend_ids)) * y_length / 2,
        ],
        axis=-1,
    )
    bend_ids = np.where(all_poses[:, 0] < -(x_length / 2))[0]
    all_poses[bend_ids] = np.stack(
        [
            -np.ones(len(bend_ids)) * x_length / 2,
            -y_length / 2 + (-all_poses[bend_ids, 0] - x_length / 2),
        ],
        axis=-1,
    )
    bend_ids = np.where(all_poses[:, 1] > (y_length / 2))[0]
    all_poses[bend_ids] = np.stack(
        [
            -x_length / 2 + (all_poses[bend_ids, 1] - y_length / 2),
            np.ones(len(bend_ids)) * y_length / 2,
        ],
        axis=-1,
    )

    cam_positions_c = torch.stack(
        [
            torch.from_numpy(all_poses[:, 0] + x_center).to(dtype=torch.float),
            torch.from_numpy(all_poses[:, 1] + y_center).to(dtype=torch.float),
            torch.ones(num_poses) * float(d_to_origin),
        ],
        dim=1,
    )  # (n, 3)

    cam_direction_c = torch.stack(
        [
            -torch.from_numpy(all_poses[:, 0]).to(dtype=torch.float),
            -torch.from_numpy(all_poses[:, 1]).to(dtype=torch.float),
            -torch.ones(num_poses) * float(d_to_origin),
        ],
        dim=1,
    )  # (n, 3)

    # cam_direction_c = torch.stack([
    #     torch.from_numpy( (all_poses[:,0] == -x_length/2).astype(np.float) -
    #                       (all_poses[:,0] == x_length/2).astype(np.float)  ),
    #     torch.from_numpy( (all_poses[:,1] == -y_length/2).astype(np.float) -
    #                       (all_poses[:,1] == y_length/2).astype(np.float) ),
    #     torch.zeros(num_poses,dtype=torch.float),
    # ], dim=1)  # (n, 3)
    #
    # cam_direction_c = torch.nn.functional.normalize(cam_direction_c - torch.nn.functional.normalize(cam_positions_c))
    #
    # cam_direction_c = cam_direction_c.to(dtype=torch.float)

    # print(f'cam_positions_c.shape = {cam_positions_c.shape}')

    # rotate the camera positions
    v1 = torch.tensor([0, 0, 1], dtype=torch.float)
    v2 = torch.stack(
        [
            torch.cos(center_angles[1] * torch.pi / 180.0) * torch.cos(center_angles[0] * torch.pi / 180.0),
            torch.cos(center_angles[1] * torch.pi / 180.0) * torch.sin(center_angles[0] * torch.pi / 180.0),
            torch.sin(center_angles[1] * torch.pi / 180.0),
        ],
        dim=0,
    )

    # print(f'v1.shape = {v1.shape}')
    # print(f'v2.shape = {v2.shape}')

    R = rigid_motion.get_min_R(
        v1=v1,
        v2=v2,
    )  # (3,3)   v2 = R @ v1

    # print(f'R = {R}')

    cam_positions_w = linalg_utils.matmul(
        R.unsqueeze(0),
        cam_positions_c.unsqueeze(-1),
    ).squeeze(-1)  # (n, 3)
    cam_direction_w = linalg_utils.matmul(
        R.unsqueeze(0),
        cam_direction_c.unsqueeze(-1),
    ).squeeze(-1)  # (n, 3)

    # create camera coordinate
    if not alt_yaxis:
        # all cameras look at the origin of the world -> -cam_positions_w
        ys = torch.zeros_like(cam_positions_w)
        ys[..., 1] = 1
    else:
        # the above implementation will make y-axis flip in a larger transform
        ys = torch.zeros_like(cam_positions_w)
        ys[..., 2] = 1
        ys = linalg_utils.matmul(R.unsqueeze(0), ys.unsqueeze(-1)).squeeze(-1)  # (n, 3)

    Rs_c2w = rigid_motion.construct_coord_frame(
        # z=-1 * cam_positions_w,  # (n, 3)
        z=cam_direction_w,  # -1 * cam_positions_w,  # (n, 3)
        y=ys,  # (n, 3, 3)
    )

    *b_shape, a, b = Rs_c2w.shape
    Hs_c2w = torch.zeros(*b_shape, 4, 4)
    Hs_c2w[..., :3, :3] = Rs_c2w
    Hs_c2w[..., :3, 3] = cam_positions_w
    Hs_c2w[..., 3, 3] = 1

    if invert_yz:
        H = torch.eye(4)
        H[1, 1] = -1.0
        H[2, 2] = -1.0
        Hs_c2w = linalg_utils.matmul(H.unsqueeze(0), Hs_c2w)

    return Hs_c2w  # (n, 4, 4)


@linalg_utils.disable_tf32_and_autocast()
def generate_camera_spiral_path(
    num_poses: int,
    num_circle: int,
    init_phi: float,
    center_angles: T.Union[torch.Tensor, np.ndarray, T.List[float]],
    r_max: float = 1,
    r_min: float = 1,
    r_freq: float = 1,
    invert_yz: bool = True,
) -> T.Union[torch.Tensor, np.ndarray]:
    """
    Generate a spiral camera path that looks at the world origin
    Args:
        num_poses:
            number of camera poses sampled on the spiral
        num_circle:
            number of circle the spiral made in xy plane
        init_phi:
            initial phi (angle between xy plane) of the path, the path will go from phi to -phi
        r_circle:
            radius of the spiral
        center_direction:
            (2,) theta (angle between x-axis), phi (angle between xy plane),
            the viewing direction of the center of the circle.  All in degree.
            The angles are given in the final coordinate (after yz is inverted)
        invert_yz:
            whether to invert the direction of y axis and z axis (since images y coord is flipped)
            This is to account for the difference in the image coordinate (x to right, y to down, z to far)
            and the world/opengl coordinate (x to right, y to up, z to us)

    Returns:
        (num_poses, 4, 4) camera poses (that converts camera coord to world coords)
    """

    if isinstance(center_angles, np.ndarray):
        center_angles = torch.from_numpy(center_angles).float()
    elif isinstance(center_angles, (list, tuple)):
        center_angles = torch.tensor(center_angles).float()

    center_angles = center_angles.float()

    if num_poses % 2 != 0:
        print("Warning: automatically change num_poses to be even")
        num_poses = num_poses + 1

    if invert_yz:
        # the coordinate is currently pre-yz-inverted
        # but center_angles are given after yz-inverted
        center_angles = -1 * center_angles

    # generate a circle on the xy plane (i.e., on the plane z = d_to_origin)
    thetas = torch.linspace(0, torch.pi * 2 * num_circle, num_poses) + torch.pi  # (n,)

    # uniformly sample along phi by cosine weighted sample
    # https://alexanderameye.github.io/notes/sampling-the-hemisphere/

    # calculate complementary phi: angle between z axis and camara position
    init_z = torch.cos(torch.pi / 2 - torch.tensor(init_phi))
    half_num_poses = int(num_poses / 2)
    comp_phi = torch.acos(torch.linspace(init_z, -init_z, half_num_poses))
    comp_phi = torch.concat([comp_phi, comp_phi[range(half_num_poses - 1, -1, -1)]], dim=0)
    phi = torch.pi / 2 - comp_phi

    r = (r_max - r_min) / 2 * torch.cos(thetas * r_freq) + (r_max + r_min) / 2

    cam_positions_c = torch.stack(
        [
            torch.cos(thetas) * torch.cos(phi) * r,
            torch.sin(thetas) * torch.cos(phi) * r,
            torch.sin(phi) * r,
        ],
        dim=1,
    )  # (n, 3)

    # print(f'cam_positions_c.shape = {cam_positions_c.shape}')

    # rotate the camera positions
    v1 = torch.tensor([0, 0, 1], dtype=torch.float)
    v2 = torch.stack(
        [
            torch.cos(center_angles[1] * torch.pi / 180.0) * torch.cos(center_angles[0] * torch.pi / 180.0),
            torch.cos(center_angles[1] * torch.pi / 180.0) * torch.sin(center_angles[0] * torch.pi / 180.0),
            torch.sin(center_angles[1] * torch.pi / 180.0),
        ],
        dim=0,
    )

    # print(f'v1.shape = {v1.shape}')
    # print(f'v2.shape = {v2.shape}')

    R = rigid_motion.get_min_R(
        v1=v1,
        v2=v2,
    )  # (3,3)   v2 = R @ v1

    # print(f'R = {R}')

    cam_positions_w = linalg_utils.matmul(R.unsqueeze(0), cam_positions_c.unsqueeze(-1)).squeeze(-1)  # (n, 3)

    # create camera coordinate
    # all cameras look at the origin of the world -> -cam_positions_w
    # ys = torch.zeros_like(cam_positions_w)
    ys = torch.stack(
        [
            -torch.cos(thetas) * torch.sin(phi),
            -torch.sin(thetas) * torch.sin(phi),
            torch.cos(phi),
        ],
        dim=1,
    )

    # ys[..., 2] = 1
    ys = linalg_utils.matmul(R.unsqueeze(0), ys.unsqueeze(-1)).squeeze(-1)  # (n, 3)
    Rs_c2w = rigid_motion.construct_coord_frame(
        z=-1 * cam_positions_w,  # (n, 3)
        y=ys,  # (n, 3, 3)
    )

    *b_shape, a, b = Rs_c2w.shape
    Hs_c2w = torch.zeros(*b_shape, 4, 4)
    Hs_c2w[..., :3, :3] = Rs_c2w
    Hs_c2w[..., :3, 3] = cam_positions_w
    Hs_c2w[..., 3, 3] = 1

    if invert_yz:
        H = torch.eye(4)
        H[1, 1] = -1.0
        H[2, 2] = -1.0
        Hs_c2w = linalg_utils.matmul(H.unsqueeze(0), Hs_c2w)

    return Hs_c2w  # (n, 4, 4)


@linalg_utils.disable_tf32_and_autocast()
def generate_camera_grids(
    num_x: int,
    num_y: int,
    cam_position_center,  # (1, 3)
    delta: float = 0.5,
    # invert_yz: bool = True,
) -> T.Union[torch.Tensor, np.ndarray]:
    if isinstance(cam_position_center, np.ndarray):
        cam_position_center = torch.from_numpy(cam_position_center).float()
    elif isinstance(cam_position_center, (list, tuple)):
        cam_position_center = torch.tensor(cam_position_center).float()

    cam_position_center = cam_position_center.float()

    ys = torch.zeros_like(cam_position_center)
    ys[..., 2] = 1
    Rs_c2w_grid = rigid_motion.construct_coord_frame(
        z=-1 * cam_position_center,  # (n, 3)
        y=ys,  # (n, 3, 3)
    )

    x_sample = torch.arange(num_x) - (num_x - 1) / 2
    y_sample = torch.arange(num_y) - (num_y - 1) / 2
    grid_x, grid_y = torch.meshgrid(x_sample, y_sample)

    grid_id = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (num_x*num_y,2)
    cam_positions_w = linalg_utils.matmul(
        grid_id,
        Rs_c2w_grid[..., 0:2].t() * delta,
    ) + cam_position_center.unsqueeze(0)

    ys = torch.zeros_like(cam_positions_w)
    ys[..., 2] = 1
    Rs_c2w = rigid_motion.construct_coord_frame(
        z=-1 * cam_positions_w,  # (n, 3)
        y=ys,  # (n, 3, 3)
    )

    *b_shape, a, b = Rs_c2w.shape
    Hs_c2w = torch.zeros(*b_shape, 4, 4)
    Hs_c2w[..., :3, :3] = Rs_c2w
    Hs_c2w[..., :3, 3] = cam_positions_w
    Hs_c2w[..., 3, 3] = 1

    # if invert_yz:
    #     H = torch.eye(4)
    #     H[1, 1] = -1.
    #     H[2, 2] = -1.
    #     Hs_c2w = H.unsqueeze(0) @ Hs_c2w

    return Hs_c2w  # (n, 4, 4)


def generate_camera_polar_grids(
    num_theta: int,
    num_phi: int,
    r: int,
):
    """
    Sample grid on theta and phi
    phi here: angle between z-axis and camera position

    Args:
        num_theta:
        num_phi: including two polar (0, pi)
        r:

    Returns:

    """

    theta_sample = torch.arange(num_theta + 1) / num_theta * torch.pi * 2  # [0,..., 2*pi]
    theta_sample = theta_sample[:-1]  # remove 2pi

    #  phi here: angle between z-axis and camera position
    phi_sample = torch.arange(num_phi) / (num_phi - 1) * torch.pi  # [0,..., pi]
    phi_sample = phi_sample[1:-1]  # remove polar

    theta, phi = torch.meshgrid(theta_sample, phi_sample)
    theta = theta.reshape(-1)
    phi = phi.reshape(-1)

    # position and angle of camera in grids
    grid_cam_positions = torch.stack(
        [
            torch.cos(theta) * torch.sin(phi) * r,
            torch.sin(theta) * torch.sin(phi) * r,
            torch.cos(phi) * r,
        ],
        dim=1,
    )  # (n, 3)
    ys = torch.zeros_like(grid_cam_positions)
    ys[..., 2] = 1
    grid_cam_frames = rigid_motion.construct_coord_frame(
        z=-1 * grid_cam_positions,  # (n, 3)
        y=ys,  # (n, 3, 3)
    )

    # position and angle of camera in polars
    polar_cam_positions = torch.cat([torch.zeros(2, 2), torch.tensor([r, -r]).unsqueeze(-1)], dim=1)
    ys = torch.zeros_like(polar_cam_positions)
    ys[..., 1] = 1
    polar_cam_frames = rigid_motion.construct_coord_frame(
        z=-1 * polar_cam_positions,  # (n, 3)
        y=ys,  # (n, 3, 3)
    )

    # concat
    Rs_c2w = torch.cat([grid_cam_frames, polar_cam_frames], dim=0)
    cam_positions_w = torch.cat([grid_cam_positions, polar_cam_positions], dim=0)

    *b_shape, a, b = Rs_c2w.shape
    Hs_c2w = torch.zeros(*b_shape, 4, 4)
    Hs_c2w[..., :3, :3] = Rs_c2w
    Hs_c2w[..., :3, 3] = cam_positions_w
    Hs_c2w[..., 3, 3] = 1

    # find neighbor camera positions
    grid_num = num_theta * (num_phi - 2)
    id_grid = np.concatenate(
        [
            np.ones([num_theta, 1]) * grid_num,
            np.arange(num_theta * (num_phi - 2)).reshape([num_theta, num_phi - 2]),
            np.ones([num_theta, 1]) * (grid_num + 1),
        ],
        axis=1,
    )
    id_grid = np.concatenate([id_grid, id_grid[[0], :]], axis=0)
    id_grid = id_grid.astype("int")

    # list of neighbor set
    neighbor_ids = [None] * (grid_num + 2)
    for self_id in range(len(neighbor_ids)):
        neighbor_ids[self_id] = set()

    for i in range(num_theta + 1):
        for j in range(num_phi):
            id_ij = id_grid[i, j]
            neighbor_ids[id_ij].add(id_grid[max(i - 1, 0), j])
            neighbor_ids[id_ij].add(id_grid[min(i + 1, num_theta), j])
            neighbor_ids[id_ij].add(id_grid[i, max(j - 1, 0)])
            neighbor_ids[id_ij].add(id_grid[i, min(j + 1, num_phi - 1)])

    # remove self from neighbors
    for self_id in range(len(neighbor_ids)):
        if self_id in neighbor_ids[self_id]:
            neighbor_ids[self_id].remove(self_id)

    return Hs_c2w, neighbor_ids  # (n, 4, 4)


def get_o3d_camera_frame(
    H_c2w: T.Union[torch.Tensor, np.ndarray],
    frame_size: float = 1.0,
) -> o3d.geometry.TriangleMesh:
    """create a camera coordinate frame (as a mesh) in the world coordinate."""

    if isinstance(H_c2w, torch.Tensor):
        H_c2w = H_c2w.detach().cpu().numpy()
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    cam_frame.transform(H_c2w)

    return cam_frame


def draw_geometries(
    geometry_list: T.List[o3d.geometry.Geometry],
    window_name: str = "",
    width: int = 1920,
    height: int = 1080,
    left: int = 50,
    top: int = 50,
    light_on: bool = True,
) -> T.Union["o3d.visualization.Visualizer", None]:
    """
    Mimic the behavior of :py:`o3d.visualization.draw_geometries` but free objects properly.

    Args:
        geometry_list:
            List of geometries to be visualized.
        window_name:
            The displayed title of the visualization window
        width:
            The width of the visualization window.
        height:
            The height of the visualization window.
        left:
            The left margin of the visualization window.
        top:
            The top margin of the visualization window.
        light_on:
            Whether to turn off lighting

    Returns:
        vis or None
    """

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=width,
        height=height,
        left=left,
        top=top,
        window_name=window_name,
        visible=True,
    )
    # show back face to make sure ray-casting and rendering results are the same
    vis.get_render_option().mesh_show_back_face = True

    # lighting
    vis.get_render_option().light_on = light_on

    for mesh in geometry_list:
        vis.add_geometry(mesh)
    vis.run()
    vis = destroy_vis(vis)


def destroy_vis(vis: T.Union[o3d.visualization.Visualizer, None]):
    if vis is not None:
        vis.clear_geometries()
        del vis
        vis = None
    return vis


def visualize_mesh_sequence(
    meshes: T.List[o3d.geometry.TriangleMesh],
    static_meshes: T.List[o3d.geometry.TriangleMesh] = None,
    window_name: str = "",
    width: int = 1920,
    height: int = 1080,
    left: int = 50,
    top: int = 50,
    light_on: bool = True,
):
    """
    Visualize a sequence of Open3D meshes.
    Press the space bar to switch to the next mesh.

    Args:
        meshes (list): List of open3d.geometry.TriangleMesh objects
    """
    if not meshes:
        print("No meshes to visualize.")
        return

    if static_meshes is None:
        static_meshes = []

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        width=width,
        height=height,
        left=left,
        top=top,
        window_name=window_name,
        visible=True,
    )
    # show back face to make sure ray-casting and rendering results are the same
    vis.get_render_option().mesh_show_back_face = True

    # lighting
    vis.get_render_option().light_on = light_on

    # add static
    for o3d_mesh in static_meshes:
        vis.add_geometry(o3d_mesh)

    # Add first mesh
    mesh_idx = [0]  # mutable container so it can be modified in callback
    vis.add_geometry(meshes[mesh_idx[0]])

    def next_mesh(vis):
        nonlocal meshes, mesh_idx
        ctr = vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()

        # Remove current mesh
        vis.remove_geometry(meshes[mesh_idx[0]], reset_bounding_box=False)

        # Update index
        mesh_idx[0] = (mesh_idx[0] + 1) % len(meshes)

        # Add next mesh
        vis.add_geometry(meshes[mesh_idx[0]], reset_bounding_box=False)

        # Restore camera parameters
        ctr.convert_from_pinhole_camera_parameters(params)
        return False  # don’t exit

    # Register spacebar (32 = space)
    vis.register_key_callback(32, next_mesh)

    vis.run()
    vis = destroy_vis(vis)


def clean_up_glb_write_gltf(
    filename_glb: str,
    overwrite_gltf: bool,
    exists_ok: bool,
    image_ext: str = ".png",
) -> str:
    """
    Clean up glb such that open3d can load the texture maps.
    Note that the function creates a gtlf file (with the same name)
    from the glb file. If the .gltf file exists and `overwrite_gltf`
    is False, it throws a runtime error.

    Args:
        filename_glb:
            the filename of the glb file.
        overwrite_gltf:
            whether to overwrite the gltf file in the same folder
            if existed.

    Returns:
        filename_gltf: str
    """

    name, ext = os.path.splitext(filename_glb)
    filename_gltf = f"{name}.gltf"

    if os.path.exists(filename_gltf):
        if overwrite_gltf:
            os.remove(filename_gltf)
        elif exists_ok:
            pass
        else:
            raise RuntimeError(f"glft file {filename_gltf} already exists")

    # convert glb to gltf
    if ext == ".glb" and not os.path.exists(filename_gltf):
        glb2gltf(filename_glb)

    assert os.path.exists(filename_gltf)

    # read gltf
    with open(filename_gltf, "r") as f:
        mesh_dict = json.load(f)

    # add extension to images
    imgs = mesh_dict["images"]
    # pprint(imgs)
    for i in range(len(imgs)):
        if "uri" in imgs[i] and len(os.path.splitext(imgs[i]["uri"])[1]) == 0:
            # fn = imgs[i]['name']
            imgs[i]["uri"] = f"{imgs[i]['uri']}{image_ext}"
        elif "name" in imgs[i] and "uri" not in imgs[i]:
            fn = imgs[i]["name"]
            imgs[i]["uri"] = f"{fn}{image_ext}"

    # modify pbrMetallicRoughness in materials to use baseColorTexture
    try:
        materials = mesh_dict["materials"]
        for i in range(len(materials)):
            emit_dict = materials[i]["emissiveTexture"]
            materials[i]["pbrMetallicRoughness"]["baseColorTexture"] = emit_dict
    except:
        pass

    with open(filename_gltf, "w") as f:
        json.dump(mesh_dict, f, indent=2)

    return filename_gltf


def get_img_max_val(img: T.Union[np.ndarray, torch.Tensor]):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    dtype = img.dtype
    # print(f'dtype = {dtype}')
    if dtype in {np.float32, np.float64, float}:
        max_val = 1.0
    else:
        max_val = np.iinfo(dtype).max
    return max_val


def sample_patch(
    arr: torch.Tensor,  # (*b, c, h_in, w_in)
    patch_center: torch.Tensor,  # (*b, 2)   h, w
    patch_width_px: int,
    patch_width_pitch_scale: T.Union[float, torch.Tensor] = 1.0,  # (*b,)
    patch_height_px: int = None,  # (*b,)
    patch_height_pitch_scale: T.Union[float, torch.Tensor] = None,  # (*b,)
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    format: str = "chw",  # 'hwc'
):
    """
    Sample from `arr` a patch centered at `center` with a different pixel pitch and number of pixels.

    Args:
        arr:
            (*b, c, h_in, w_in) or (*b, h_in, w_in, c), see `format`. the array to be sampled from.
        patch_center:
            (*b, 2) the center of each patch on arr. u: first dim [0, w_in], v: second dim [0, h_in]
        patch_width_px:
            number of pixels in the patch in width
        patch_width_pitch_scale:
            (*b,) the pitch of the patch (new_pitch / old_pitch)
        patch_height_px:
            if None, the same as `patch_width_px`
        patch_height_pitch_scale:
            if None, the same as `patch_width_pitch_scale`
        format:
            'chw':  arr is (b, c, h, w)
            'hwc':  arr is (b, h, w, c)

    Returns:
        (*b, c, patch_height_px, patch_width_px) or (*b, patch_height_px, patch_width_px, c)

    Note:
        coordinate system:
            The origin of the coordinate is at the top-left corner of `arr`.
            Each pixel in `arr` is 1 unit in width and height.
            The first dimension (u) is toward right and second dimension (v) is toward down.
            The first pixel center is `arr` is at (0.5, 0.5).
        This function should be compared with `uv_sampling`, which uses a different coordinate system.
    """

    if format == "chw":
        *b_shape, c, h, w = arr.shape
        arr = arr.reshape(-1, c, h, w)  # (b, c, h, w)
    elif format == "hwc":
        *b_shape, h, w, c = arr.shape
        arr = arr.reshape(-1, h, w, c).permute(0, 3, 1, 2)  # (b, c, h, w)
    else:
        raise NotImplementedError

    b = int(np.prod(b_shape))
    device = arr.device

    uv = generate_patch_uv(
        patch_center=patch_center,  # (*b, 2)
        patch_width_px=patch_width_px,
        patch_width_pitch_scale=patch_width_pitch_scale,
        patch_height_px=patch_height_px,
        patch_height_pitch_scale=patch_height_pitch_scale,
        device=device,
    )  # (*b, hp, wp, 2)
    uv = uv.reshape(b, uv.size(-3), uv.size(-2), uv.size(-1))  # (b, hp, wp, 2)

    # [0, w] -> [0, 2] -> [-1, 1]
    u = uv[..., 0] * (2 / w) - 1  # (b, hp, wp)
    v = uv[..., 1] * (2 / h) - 1  # (b, hp, wp)
    uv = torch.stack([u, v], dim=-1)  # (b, hp, wp, 2)  [0, w] [0, h]

    # grid_sample
    sampled_patch = torch.nn.functional.grid_sample(
        input=arr,  # (b, c, h, w)
        grid=uv,  # (b, hp, wp, 2)
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False,
    )  # (b, c, hp, wp)

    if format == "chw":
        pass
    elif format == "hwc":
        sampled_patch = sampled_patch.permute(0, 2, 3, 1)
    else:
        raise NotImplementedError

    sampled_patch = sampled_patch.reshape(*b_shape, sampled_patch.size(1), sampled_patch.size(2), sampled_patch.size(3))
    return sampled_patch


def generate_patch_uv(
    patch_center: torch.Tensor,  # (*b, 2)   h, w
    patch_width_px: int,
    patch_width_pitch_scale: T.Union[float, torch.Tensor] = 1.0,  # (*b,)
    patch_height_px: int = None,  # (*b,)
    patch_height_pitch_scale: T.Union[float, torch.Tensor] = None,  # (*b,)
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate uv coordinates ([0, w), [0, h)) of the patches centered at patch_center.

    Args:
        patch_center:
            (*b, 2) the center of each patch on arr. u: first dim [0, w_in], v: second dim [0, h_in]
        patch_width_px:
            number of pixels in the patch in width
        patch_width_pitch_scale:
            (*b,) the pitch of the patch (new_pitch / old_pitch)
        patch_height_px:
            if None, the same as `patch_width_px`
        patch_height_pitch_scale:
            if None, the same as `patch_width_pitch_scale`
        int_only:
            whether the center is always at an integer index

    Returns:
        (*b, patch_height_px, patch_width_px, 2) the u (first dimension) and v (second dimension)
        Note the returned uv can go out of bound.
    """

    *b_shape, _2 = patch_center.shape
    b = int(np.prod(b_shape))
    patch_center = patch_center.reshape(b, 2)  # (b, 2)
    if isinstance(patch_width_pitch_scale, (int, float)):
        patch_width_pitch_scale = torch.ones(b, dtype=torch.float, device=device) * patch_width_pitch_scale
    if isinstance(patch_width_pitch_scale, torch.Tensor):
        patch_width_pitch_scale = patch_width_pitch_scale.reshape(b).to(device=device)  # (b,)

    if patch_height_px is None:
        patch_height_px = patch_width_px

    if patch_height_pitch_scale is None:
        patch_height_pitch_scale = patch_width_pitch_scale
    if isinstance(patch_height_pitch_scale, (int, float)):
        patch_height_pitch_scale = torch.ones(b, dtype=torch.float, device=device) * patch_height_pitch_scale
    if isinstance(patch_height_pitch_scale, torch.Tensor):
        patch_height_pitch_scale = patch_height_pitch_scale.reshape(b).to(device=device)  # (b,)

    # generate the canonical grid for the patch
    patch_half_width_px = patch_width_px / 2
    patch_half_height_px = patch_height_px / 2

    u, v = torch.meshgrid(
        torch.arange(patch_width_px, dtype=torch.float, device=device),
        torch.arange(patch_height_px, dtype=torch.float, device=device),
        indexing="xy",
    )  # u: (hp, wp), [0, w-1],  v: (hp, wp) [0, h-1]  top-left (0,0)
    u = u + (0.5 - patch_half_width_px)
    v = v + (0.5 - patch_half_height_px)
    # u: (hp, wp), [-0.5, 0, 0.5],  v: (hp, wp)  [-0.5, 0, 0.5]

    # scale and recenter the canonical grid
    u = u.unsqueeze(0).expand(b, -1, -1) * patch_width_pitch_scale.reshape(b, 1, 1) + patch_center[:, 0].reshape(
        b, 1, 1
    )  # [0, w]
    v = v.unsqueeze(0).expand(b, -1, -1) * patch_height_pitch_scale.reshape(b, 1, 1) + patch_center[:, 1].reshape(
        b, 1, 1
    )  # [0, h]

    uv = torch.stack([u, v], dim=-1)  # (b, hp, wp, 2)  [0, w] [0, h]
    uv = uv.reshape(*b_shape, *(uv.shape[1:]))
    return uv


def sample_random_patch_uv(
    b_shape: T.Union[int, T.List[int]],
    width_px: int,
    height_px: int,
    patch_width_px: int,
    patch_width_pitch_scale: T.Union[float, torch.Tensor] = 1.0,  # (*b,)
    patch_height_px: int = None,  # (*b,)
    patch_height_pitch_scale: T.Union[float, torch.Tensor] = None,  # (*b,)
    int_only: bool = True,
    inbound_only: bool = True,
    prob_density: torch.Tensor = None,  # (*bq, height_px, width_px)
    prob_density_bias: float = 5.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Samples random uv coordinates ([0, w), [0, h)) to create random patches.

    Args:
        b_shape:
            (*b,) determines the number of patches to sample
        width_px:
            width (in pixel) of the image to sample from, determines the range of u [0,w)
        height_px:
            height (in pixel) of the images to sample from, determines the range of v [0,h)
        patch_width_px:
            number of pixels in the patch in width
        patch_width_pitch_scale:
            (*b,) the pitch of the patch (new_pitch / old_pitch)
        patch_height_px:
            if None, the same as `patch_width_px`
        patch_height_pitch_scale:
            if None, the same as `patch_width_pitch_scale`
        int_only:
            whether the center is always at an integer index
        inbound_only:
            whether to make sure the sample patch will be entirely within valid image range
        prob_density:
            (*bq, h, w) the probability to select each pixel.
            If given, the function will try to sample based on the probability
            summed in a patch.
            It only support int_only = True
        prob_density_bias:
            if higher, it will more likely sample patches with higher prob_density

    Returns:
        (*b, patch_height_px, patch_width_px, 2) the u (first dimension) and v (second dimension)

        Note that the patch can go out of bound
    """
    if isinstance(b_shape, int):
        b_shape = [b_shape]

    if patch_height_px is None:
        patch_height_px = patch_width_px
    if patch_height_pitch_scale is None:
        patch_height_pitch_scale = patch_width_pitch_scale

    if prob_density is not None:
        *bq_shape, height_px, width_px = prob_density.shape
        bq = math.prod(bq_shape)
        assert len(bq_shape) <= len(b_shape)
        assert bq_shape == b_shape[: len(bq_shape)]
        num_samples = math.prod(b_shape[len(bq_shape) :])

        # Goal is to sample patch center (*b, 2uv) based on prob_density
        # we first accumulate the total density of each patch location
        # by summing the density for each pixel
        cumsum_density = torch.cumsum(
            torch.cumsum(prob_density, dim=-1),
            dim=-2,
        )  # (*bq, h, w)
        hidx, widx = torch.meshgrid(
            torch.arange(height_px, dtype=torch.float, device=device),
            torch.arange(width_px, dtype=torch.float, device=device),
            indexing="ij",
        )  # (h, w) [0, h-1] [0, w-1]
        # uv = torch.stack([widx, hidx], dim=-1)  # (h, w, 2uv)
        du = patch_width_px * patch_width_pitch_scale * 0.5
        dv = patch_height_px * patch_height_pitch_scale * 0.5

        def get_cumsum_density(_widx, _hidx):
            # widx: (h, w) not normalized,  hidx: (h, w) not normalized
            _widx = torch.clamp(_widx, min=0, max=width_px - 1)
            _hidx = torch.clamp(_hidx, min=0, max=height_px - 1)
            _uv = torch.stack([_widx, _hidx], dim=-1)  # (h, w, 2uv)
            _uv = _uv + 0.5

            out = uv_sampling(
                uv=_uv.expand(*bq_shape, height_px, width_px, 2).reshape(bq, height_px, width_px, 2),  # (bq, h, w, 2uv)
                feature_map=cumsum_density.reshape(bq, height_px, width_px, 1),  # (bq, h, w, 1)
                mode="bilinear",  # to support patch width that is not odd.  'nearest'
                padding_mode="border",
                uv_normalized=False,
            ).reshape(*bq_shape, height_px, width_px)  # (*bq, h, w)
            return out  # (*bq, h, w)

        # compute total probability in a patch (and put it at the patch center)
        bottom_right = get_cumsum_density(
            _widx=widx + du,
            _hidx=hidx + dv,  # (h, w, 2uv)
        )  # (*bq, h, w)
        bottom_left = get_cumsum_density(
            _widx=widx - du,
            _hidx=hidx + dv,  # (h, w, 2uv)
        )  # (*bq, h, w)
        top_right = get_cumsum_density(
            _widx=widx + du,
            _hidx=hidx - dv,  # (h, w, 2uv)
        )  # (*bq, h, w)
        top_left = get_cumsum_density(
            _widx=widx - du,
            _hidx=hidx - dv,  # (h, w, 2uv)
        )  # (*bq, h, w)
        patch_prob = bottom_right - top_right - bottom_left + top_left  # (*bq, h, w)
        patch_prob = torch.clamp(patch_prob, min=0)  # numerical error may cause 0

        # print(f'patch_prob.min = {patch_prob.min()}')

        if inbound_only:
            patch_prob[..., : math.ceil(dv), :] = 0
            patch_prob[..., -math.ceil(dv) :, :] = 0
            patch_prob[..., :, : math.ceil(du)] = 0
            patch_prob[..., :, -math.ceil(du) :] = 0

        # make sure sum of patch_prob > 0
        patch_prob_sum = patch_prob.reshape(*bq_shape, height_px * width_px).sum(-1)  # (*bq,)
        patch_prob = patch_prob.masked_fill(
            (patch_prob_sum < 1e-6).reshape(*bq_shape, 1, 1),
            -1000.0,
        )  # (*bq, h, w)

        # # debug
        # msg = 'top 5 patch_prob from each image: \n'
        # _patch_prob = patch_prob.reshape(bq, height_px * width_px)
        # for ii in range(bq):
        #     pp, hwidx = torch.sort(_patch_prob[ii], descending=True)
        #     pp = pp[:5]
        #     hwidx = hwidx[:5]
        #     msg += f'  {ii}:'
        #     for jj in range(len(pp)):
        #         msg += f' hw={hwidx[jj].item()} ({pp[jj].item():.1f})'
        #     msg += '\n'
        # print(msg)
        # # end debug

        # bias toward patch that has density
        patch_prob = torch.nn.functional.softmax(
            input=patch_prob.reshape(bq, height_px * width_px) * prob_density_bias,
            dim=-1,
        )  # (bq, hw)

        # sample based on patch_prob
        hwidxs = torch.multinomial(
            input=patch_prob.reshape(bq, height_px * width_px),  # (bq, hw)
            num_samples=num_samples,
            replacement=True,
        )  # (bq, num_samples)  long
        widxs = hwidxs % width_px  # (bq, num_samples)
        hidxs = torch.floor(hwidxs.float() / width_px).long()  # (bq, num_samples)

        # # debug
        # msg = 'selected samples from each image: \n'
        # for ii in range(bq):
        #     pp = hwidxs[ii]
        #     ww = widxs[ii]
        #     hh = hidxs[ii]
        #     msg += f'  {ii}:'
        #     for jj in range(len(pp)):
        #         msg += f' hw={pp[jj].item()} ({ww[jj]}, {hh[jj]})'
        #     msg += '\n'
        # print(msg)
        # # end debug

        patch_center = (
            torch.stack(
                [
                    widxs.reshape(*b_shape),
                    hidxs.reshape(*b_shape),
                ],
                dim=-1,
            )
            + 0.5
        )  # (*b, 2)

    else:
        # randomly sample patch center
        patch_center = torch.rand(*b_shape, 2, device=device)  # (*b, 2)  [0,1)

        if not inbound_only:
            patch_center[..., 0] = patch_center[..., 0] * width_px  # (*b, 2)  [0,w) [0,h]
            patch_center[..., 1] = patch_center[..., 1] * height_px
        else:
            dw = patch_width_px * patch_width_pitch_scale
            if patch_height_px is None or patch_height_pitch_scale is None:
                dh = dw
            else:
                dh = patch_height_px * patch_height_pitch_scale
            patch_center[..., 0] = patch_center[..., 0] * (width_px - dw) + 0.5 * dw
            patch_center[..., 1] = patch_center[..., 1] * (height_px - dh) + 0.5 * dh

    if int_only:
        # we need to snap to 0.5, 1.5, 2.5, which are the actual pixel center
        patch_center = torch.floor(patch_center) + 0.5

    uv = generate_patch_uv(
        patch_center=patch_center,  # (*b, 2)
        patch_width_px=patch_width_px,
        patch_width_pitch_scale=patch_width_pitch_scale,
        patch_height_px=patch_height_px,
        patch_height_pitch_scale=patch_height_pitch_scale,
        device=device,
    )
    return uv  # (*b, hp, wp, 2)


def compute_fov_overlapping_with_depth_map(
    z_map: torch.Tensor,
    intrinsic: torch.Tensor,
    H_c2w: torch.Tensor,
    num_points: int = 100,
    th_diopter: float = 0.25,
) -> torch.Tensor:
    """
    Compute the field-of-view overlapping between images using
    the help of depth maps

    Important note:
        The function uses a image coordinate system: x to right, y to "down", z to far.
        If the world coordinate is a different one (say x to right, y to "up", z to us),
        H_c2w need to include the image coordinate to world (ie. flip y and z),
        ex: H_actual * H_i2l

    Args:
        z_map:
            (n, h, w) the z coordinate of the point in the camera coordinate on the sensor,
            not along the corresponding camera ray.
        intrinsic:
            (n, 3, 3) camera intrinsic matrix
        H_c2w:
            (n, 4, 4) homegeneous matrix that convert camera coord to world coord.
            Note that the y axis should be already inverted (so the yaxis is toward down).
        num_points: int
            number of points to select
        th_diopter:
            threshold in diopter to consider a match

    Returns:
        overlapping:
            (n, n) [0, 1] overlapping between i-th and j-th images

    Algorithm:
        For each image:
        1. randomly select `num_points` pixels and create 3d points using depth map
        2. project the 3d points and calculate the z in the camera
            coordinate of the rest of the images
        3. use the z_cam to consider occlusion
        4. compute the percentage of seen points
    """

    assert z_map.ndim == 3
    assert intrinsic.ndim == 3
    assert H_c2w.ndim == 3

    n, h, w = z_map.shape
    assert intrinsic.size(0) == n
    assert H_c2w.size(0) == n
    hw = h * w
    assert hw >= num_points

    device = z_map.device

    # # randomly select pixels
    # u_all, v_all = torch.meshgrid(
    #     torch.arange(0, w, device=device),
    #     torch.arange(0, h, device=device),
    #     indexing='xy',
    # )  # u: (h, w) for x,  v: (h, w) for y in the sensor coord
    # u_all = u_all.reshape(-1)  # (hw,)
    # v_all = v_all.reshape(-1)  # (hw,)
    # uv_all = torch.stack([u_all, v_all], dim=-1)  # (hw, 2)
    # probs = torch.ones(hw, device=device) / hw
    # idxs = torch.multinomial(probs, num_samples=num_points * n, replacement=True) # (n * num_points,)
    # uv = uv_all[idxs]  # (n * num_points, 2)
    # uv = uv.reshape(n, num_points, 2)  # (n, num_points, 2)
    #
    # find_corresponding_uv(
    #     uv_c=uv + 0.5,  # (b=n, num_points, 2)
    #     z_map=z_map,  # (b=n, h, w)
    #     intrinsics_from=intrinsic,   # (b=n, 3, 3)
    #     H_c2w_from=H_c2w,  # (b=n, 4, 4)
    #     intrinsics_to=intrinsic,  # (b=n, 3, 3)
    #     H_c2w_to=H_c2w, # (b=n, 4, 4)
    # )

    overlapping = torch.eye(n, device=device)  # (n, n)

    # generate u v w on the sensor coord (integer)
    u_all, v_all = torch.meshgrid(
        torch.arange(0, w, device=device),
        torch.arange(0, h, device=device),
        indexing="xy",
    )  # u: (h, w) for x,  v: (h, w) for y in the sensor coord
    u_all = u_all.reshape(-1)  # (hw,)
    v_all = v_all.reshape(-1)  # (hw,)
    uv_all = torch.stack([u_all, v_all], dim=-1)  # (hw, 2)
    probs = torch.ones(u_all.numel(), device=device) / u_all.numel()

    for i in range(n):
        # randomly select pixels
        idxs = torch.multinomial(probs, num_samples=num_points)  # (num_points,)
        uv = uv_all[idxs]  # (num_points, 2)

        # create 3d points, project 3d points to all images, calculate corresponding uv and xyz_c
        uv_to, xyz_to = find_corresponding_uv(
            uv_c=uv + 0.5,  # (num_points, 2)
            z_map=z_map[i],  # (h, w)
            intrinsics_from=intrinsic[i],  # (3, 3)
            H_c2w_from=H_c2w[i],  # (4, 4)
            intrinsics_to=intrinsic,  # (n, 3, 3)
            H_c2w_to=H_c2w,  # (n, 4, 4)
            dim_b=0,
        )
        #  uv_to: (n, num_points, 2)  [0, w] [0, h] can be out of bound
        #  xyz_to: (n, num_points, 3)
        z_to = xyz_to[..., 2]  # (n, num_points)

        # get actual zc of uv_c
        z_c_gt = uv_sampling(
            uv=uv_to,  # (n, num_points, 2)
            feature_map=z_map.reshape(n, h, w, 1),  # (n, h, w, 1)
            uv_normalized=False,
        ).squeeze(-1)  # (n, num_points)

        # use uv_to and z_to to determine inlier
        valid_z = (torch.clamp(1 / z_c_gt, min=0, max=1e3) - torch.clamp(1 / z_to, min=0, max=1e3)).abs() < th_diopter
        valid_u = torch.logical_and(uv_to[..., 0] <= w, uv_to[..., 0] >= 0)
        valid_v = torch.logical_and(uv_to[..., 1] <= h, uv_to[..., 1] >= 0)
        valid = torch.logical_and(valid_z, torch.logical_and(valid_u, valid_v))  # (n, num_points)

        # overlapping is calculating as the percentage of valid points
        overlapping[i] = valid.sum(dim=-1) / num_points  # (n,)

    # make sure overlapping is symmetric
    overlapping = (overlapping + overlapping.t()) / 2
    return overlapping  # (n, n)


def compute_dsd(
    o3d_scene: o3d.t.geometry.RaycastingScene,
    origin_w: torch.Tensor,
    direction_w: torch.Tensor,
    o3d_mesh: T.Optional[o3d.geometry.TriangleMesh],
    th_on_surface: float = 1.0e-3,
    t_inf: float = 1e4,
    version: int = 3,
    num_mesh_intersect_offset: int = 0,
):
    """
    Compute the directed distance using o3d raycasting scene.

    Args:
        o3d_scene:
            the o3d raycasting scene
        origin_w:
            (*, 3xyz_w) the ray origin in the world coordinate
        direction_w:
            (*, 3xyz_w) the ray direction in the world coordinate
        o3d_mesh:
            o3d mesh containing the texture, normal
        th_on_surface:
            if |ud| < th_on_surface, true
        t_inf:
            if not None, dsd will be clamp to [-t_inf, t_inf]
        version:
            1: find both direction_w and -direction_w.  If -direction_w is closer to surface, use its distance with negative sign.
            2: if origin_w is inside a closed surface, use -direction_w's distance with negative sign.
            3: just sdf (but only along the ray):  if inside an object, negative; outside, positive.  The absolute value is distance to the closest surface.

    Returns:

        dsd:
            (*,) ray_t to the nearest surface on the ray.
            negative if the nearest surface is in the opposite direction of direction_w

        normal_w:
            (*, 3_xyz_w) in the opposite direction of direction_w, regardless of pos/neg is closer

        rgb:
            (*, 3rgb)

        hit_map:
            (*,) bool, whether the ray will hit a surface

        hit_at_the_point:
            (*,) bool, whether the point is on a surface

        is_inside:
            (*,) bool, whether origin_w is inside a closed surface
            we estimate this by checking the number of surface intersection from
            origin_w through direction_w
    """
    if version == 1:
        return compute_dsd_v1(
            o3d_scene=o3d_scene,
            origin_w=origin_w,
            direction_w=direction_w,
            o3d_mesh=o3d_mesh,
            th_on_surface=th_on_surface,
            t_inf=t_inf,
            num_mesh_intersect_offset=num_mesh_intersect_offset,
        )
    elif version == 2:
        return compute_dsd_v2(
            o3d_scene=o3d_scene,
            origin_w=origin_w,
            direction_w=direction_w,
            o3d_mesh=o3d_mesh,
            th_on_surface=th_on_surface,
            t_inf=t_inf,
            num_mesh_intersect_offset=num_mesh_intersect_offset,
        )
    elif version == 3:
        return compute_dsd_v3(
            o3d_scene=o3d_scene,
            origin_w=origin_w,
            direction_w=direction_w,
            o3d_mesh=o3d_mesh,
            th_on_surface=th_on_surface,
            t_inf=t_inf,
            num_mesh_intersect_offset=num_mesh_intersect_offset,
        )
    else:
        raise NotImplementedError


def compute_dsd_v1(
    o3d_scene: o3d.t.geometry.RaycastingScene,
    origin_w: torch.Tensor,
    direction_w: torch.Tensor,
    o3d_mesh: o3d.geometry.TriangleMesh,
    th_on_surface: float = 1.0e-3,
    t_inf: float = 1e4,
    num_mesh_intersect_offset: int = 0,
):
    """
    Compute the directed signed distance field.
    v1:  always find the closest surface in both forward and backward direction

    Args:
        o3d_scene:
            the o3d raycasting scene
        origin_w:
            (*, 3xyz_w) the ray origin in the world coordinate
        direction_w:
            (*, 3xyz_w) the ray direction in the world coordinate
        o3d_mesh:
            o3d mesh containing the texture, normal
        t_inf:
            if not None, dsd will be clamp to [-t_inf, t_inf]

    Returns:

        dsd:
            ray_t to the nearest surface on the ray.
            negative if the nearest surface is in the opposite direction of direction_w

        normal_w:
            (*, 3_xyz_w) in the opposite direction of direction_w, regardless of pos/neg is closer

        rgb:
            (*, 3rgb)

        hit_map:
            (*,) bool, whether the ray hit a surface

        is_inside:
            (*,) bool, whether origin_w is inside a closed surface
            we estimate this by checking the number of surface intersection from
            origin_w through direction_w
    """

    torch_dtype = origin_w.dtype
    device = origin_w.device
    *b_shape, _3xyz = origin_w.shape
    b = math.prod(b_shape)

    origin_w = origin_w.reshape(b, 3)  # (b, 3)
    direction_w = direction_w.reshape(b, 3)  # (b, 3)

    # unsigned distance
    ud = o3d_scene.compute_distance(
        origin_w.detach().cpu().float().numpy()  # (b, 3)  float32
    ).numpy()  # (b,)
    ud = torch.from_numpy(ud).to(dtype=torch_dtype, device=device)  # (b,)
    hit_at_the_point = ud <= th_on_surface

    # cast the rays toward + direction, get the intersections
    rays = torch.cat(
        (
            origin_w,
            direction_w,
        ),
        dim=-1,
    )  # (b, 6)
    rays = rays.detach().cpu().float().numpy()  # (b, 6) float32
    raycast_results_pos = o3d_scene.cast_rays(rays)

    # check origin_w is inside or outside a closed surface
    intersection_counts = torch.from_numpy(o3d_scene.count_intersections(rays).numpy())  # (b,)
    intersection_counts = intersection_counts - num_mesh_intersect_offset
    assert (intersection_counts >= 0).all()
    is_inside = (intersection_counts % 2 == 1).to(dtype=torch.bool, device=device)  # (b,) bool

    # cast the rays backward (- direction), get the intersections
    rays = torch.cat(
        (
            origin_w,
            -1 * direction_w,
        ),
        dim=-1,
    )  # (b, 6)
    rays = rays.detach().cpu().float().numpy()  # (b, 6) float32
    raycast_results_neg = o3d_scene.cast_rays(rays)

    # select the nearest surface
    pos_closer = torch.from_numpy(raycast_results_pos["t_hit"].numpy()) <= torch.from_numpy(
        raycast_results_neg["t_hit"].numpy()
    )  # (b,) bool

    # merge raycast_results
    raycast_results = dict()
    for name in [
        "t_hit",  # (b,)
        "geometry_ids",  # (b,)
        "primitive_ids",  # (b,)
        "primitive_uvs",  # (b, 2)
        "primitive_normals",  # (b, 3)
    ]:
        pos = torch.from_numpy(raycast_results_pos[name].numpy())
        neg = torch.from_numpy(raycast_results_neg[name].numpy())
        if pos.ndim == 1:
            out = torch.where(pos_closer, pos, neg)
        else:
            out = torch.where(pos_closer.unsqueeze(-1), pos, neg)
        raycast_results[name] = o3d.core.Tensor(out.numpy())
    del raycast_results_pos
    del raycast_results_neg

    t_hits = torch.from_numpy(raycast_results["t_hit"].numpy()).to(
        dtype=torch_dtype, device=device
    )  # (b,), inf if not hit the mesh
    hit_map = torch.logical_not(torch.isinf(t_hits))  # (b,) bool, true if hit a surface, 0 otherwise
    dsd = t_hits * ((pos_closer.to(dtype=torch_dtype, device=device) - 0.5) * 2)  # (b,)
    if t_inf is not None:
        dsd = dsd.clamp(min=-t_inf, max=t_inf)

    # record the actual intersection point's ray_t (in terms of ray_direction_w)
    ray_t = t_hits * ((pos_closer.to(dtype=torch_dtype, device=device) - 0.5) * (2))  # (b,)

    # render rgb of the ray
    if o3d_mesh is not None and o3d_mesh.has_textures():
        ray_rgbs = render.interp_texture_map_from_ray_tracing_results(
            mesh=o3d_mesh,
            raycast_results=raycast_results,
            texture_maps=[skimage.img_as_float(np.array(img)).astype(np.float32) for img in o3d_mesh.textures],
            merge_textures=True,  # combine results from multiple textures.
        )[0]  # (b, 3)
    else:
        ray_rgbs = np.ones((b, 3), dtype=np.float32)
    ray_rgbs = torch.from_numpy(ray_rgbs).to(dtype=torch_dtype, device=device)  # (b, 3)

    # note that primitive_normals is the normal of the triangle face
    # we can use uv map to interpolate vertex normal
    # interpolate surface normal using uv map to get better normal estimation
    if o3d_mesh is not None and o3d_mesh.has_vertex_normals():
        surface_normals = render.interp_surface_normal_from_ray_tracing_results(
            mesh=o3d_mesh,
            raycast_results=raycast_results,
        )  # (b, 3)
    else:
        surface_normals = raycast_results["primitive_normals"].numpy()  # (b, 3)
    surface_normals = torch.from_numpy(surface_normals).to(dtype=torch_dtype, device=device)  # (b, 3)
    surface_normals = torch.nn.functional.normalize(
        surface_normals,
        dim=-1,
        eps=1e-9 if torch_dtype == torch.float32 else 1e-4,
    )
    # make sure surface_normals points at the negative direction of direction_w
    same_dir = ((surface_normals * direction_w).sum(dim=-1) > 0).to(dtype=torch_dtype)  # (b,)  {1, 0}
    surface_normals = surface_normals * ((same_dir - 0.5) * (-2)).unsqueeze(-1)  # (b, 3)

    # if not hit a surface, set surface normal to (0, 0, 0)
    surface_normals = surface_normals.masked_fill(hit_map.unsqueeze(-1), 0)  # (b, 3)

    out_dict = dict(
        rgb=ray_rgbs,
        dsd=dsd,
        normal_w=surface_normals,
        hit_map=hit_map,
        is_inside=is_inside,
        ud=ud,
        hit_at_the_point=hit_at_the_point,
        ray_t=ray_t,
    )

    # reshape to (*b, d)
    out_dict = reshape(
        out_dict,
        start=0,
        end=0,
        shape=b_shape,
    )
    return out_dict


def compute_dsd_v2(
    o3d_scene: o3d.t.geometry.RaycastingScene,
    origin_w: torch.Tensor,
    direction_w: torch.Tensor,
    o3d_mesh: o3d.geometry.TriangleMesh,
    th_on_surface: float = 1.0e-3,
    t_inf: float = 1e4,
    num_mesh_intersect_offset: int = 0,
):
    """
    Compute the directed signed distance field.
    v2:  If origin_w is inside a closed surface, we will return the distance to the
    surface in -1 * direction_w direction (with negative sign).

    Args:
        o3d_scene:
            the o3d raycasting scene
        origin_w:
            (*, 3xyz_w) the ray origin in the world coordinate
        direction_w:
            (*, 3xyz_w) the ray direction in the world coordinate
        o3d_mesh:
            o3d mesh containing the texture, normal
        t_inf:
            if not None, dsd will be clamp to [-t_inf, t_inf]

    Returns:

        dsd:
            ray_t to the nearest surface on the ray.
            negative if the nearest surface is in the opposite direction of direction_w

        normal_w:
            (*, 3_xyz_w) in the opposite direction of direction_w, regardless of pos/neg is closer

        rgb:
            (*, 3rgb)

        hit_map:
            (*,) bool, whether the ray hit a surface

        is_inside:
            (*,) bool, whether origin_w is inside a closed surface
            we estimate this by checking the number of surface intersection from
            origin_w through direction_w
    """

    torch_dtype = origin_w.dtype
    device = origin_w.device
    *b_shape, _3xyz = origin_w.shape
    b = math.prod(b_shape)

    origin_w = origin_w.reshape(b, 3)  # (b, 3)
    direction_w = direction_w.reshape(b, 3)  # (b, 3)

    # unsigned distance
    ud = o3d_scene.compute_distance(
        origin_w.detach().cpu().float().numpy()  # (b, 3)  float32
    ).numpy()  # (b,)
    ud = torch.from_numpy(ud).to(dtype=torch_dtype, device=device)  # (b,)
    hit_at_the_point = ud <= th_on_surface

    # cast the rays toward + direction, get the intersections
    rays = torch.cat(
        (
            origin_w,
            direction_w,
        ),
        dim=-1,
    )  # (b, 6)
    rays = rays.detach().cpu().float().numpy()  # (b, 6) float32
    raycast_results_pos = o3d_scene.cast_rays(rays)

    # check origin_w is inside or outside a closed surface
    intersection_counts = torch.from_numpy(o3d_scene.count_intersections(rays).numpy())  # (b,)
    intersection_counts = intersection_counts - num_mesh_intersect_offset
    assert (intersection_counts >= 0).all()
    is_inside = (intersection_counts % 2 == 1).to(dtype=torch.bool, device=device)  # (b,) bool

    # cast the rays backward (- direction), get the intersections
    if is_inside.any():
        rays = torch.cat(
            (
                origin_w,
                -1 * direction_w,
            ),
            dim=-1,
        )  # (b, 6)
        rays = rays.detach().cpu().float().numpy()  # (b, 6) float32
        raycast_results_neg = o3d_scene.cast_rays(rays)

        # merge raycast_results
        raycast_results = dict()
        for name in [
            "t_hit",  # (b,)
            "geometry_ids",  # (b,)
            "primitive_ids",  # (b,)
            "primitive_uvs",  # (b, 2)
            "primitive_normals",  # (b, 3)
        ]:
            pos = torch.from_numpy(raycast_results_pos[name].numpy())
            neg = torch.from_numpy(raycast_results_neg[name].numpy())
            if pos.ndim == 1:
                out = torch.where(is_inside, neg, pos)
            else:
                out = torch.where(is_inside.unsqueeze(-1), neg, pos)
            raycast_results[name] = o3d.core.Tensor(out.numpy())
        del raycast_results_neg
    del raycast_results_pos

    t_hits = torch.from_numpy(raycast_results["t_hit"].numpy()).to(
        dtype=torch_dtype, device=device
    )  # (b,), inf if not hit the mesh
    hit_map = torch.logical_not(torch.isinf(t_hits))  # (b,) bool, true if hit a surface, 0 otherwise
    # make sure if is_inside = True, dsd is negative
    dsd = t_hits * ((is_inside.to(dtype=torch_dtype, device=device) - 0.5) * (-2))  # (b,)
    if t_inf is not None:
        dsd = dsd.clamp(min=-t_inf, max=t_inf)

    # record the actual intersection point's ray_t (in terms of ray_direction_w)
    ray_t = t_hits * ((is_inside.to(dtype=torch_dtype, device=device) - 0.5) * (-2))  # (b,)

    # render rgb of the ray
    if o3d_mesh is not None and o3d_mesh.has_textures():
        ray_rgbs = render.interp_texture_map_from_ray_tracing_results(
            mesh=o3d_mesh,
            raycast_results=raycast_results,
            texture_maps=[skimage.img_as_float(np.array(img)).astype(np.float32) for img in o3d_mesh.textures],
            merge_textures=True,  # combine results from multiple textures.
        )[0]  # (b, 3)
    else:
        ray_rgbs = np.ones((b, 3), dtype=np.float32)
    ray_rgbs = torch.from_numpy(ray_rgbs).to(dtype=torch_dtype, device=device)  # (b, 3)

    # note that primitive_normals is the normal of the triangle face
    # we can use uv map to interpolate vertex normal
    # interpolate surface normal using uv map to get better normal estimation
    if o3d_mesh is not None and o3d_mesh.has_vertex_normals():
        surface_normals = render.interp_surface_normal_from_ray_tracing_results(
            mesh=o3d_mesh,
            raycast_results=raycast_results,
        )  # (b, 3)
    else:
        surface_normals = raycast_results["primitive_normals"].numpy()  # (b, 3)
    surface_normals = torch.from_numpy(surface_normals).to(dtype=torch_dtype, device=device)  # (b, 3)
    surface_normals = torch.nn.functional.normalize(
        surface_normals,
        dim=-1,
        eps=1e-9 if torch_dtype == torch.float32 else 1e-4,
    )
    # make sure surface_normals points at the negative direction of direction_w
    same_dir = ((surface_normals * direction_w).sum(dim=-1) > 0).to(dtype=torch_dtype)  # (b,)  {1, 0}
    surface_normals = surface_normals * ((same_dir - 0.5) * (-2)).unsqueeze(-1)  # (b, 3)

    # if not hit a surface, set surface normal to (0, 0, 0)
    surface_normals = surface_normals.masked_fill(hit_map.unsqueeze(-1), 0)  # (b, 3)

    out_dict = dict(
        rgb=ray_rgbs,
        dsd=dsd,
        normal_w=surface_normals,
        hit_map=hit_map,
        is_inside=is_inside,
        ud=ud,
        hit_at_the_point=hit_at_the_point,
        ray_t=ray_t,
    )

    # reshape to (*b, d)
    out_dict = reshape(
        out_dict,
        start=0,
        end=0,
        shape=b_shape,
    )
    return out_dict


def compute_dsd_v3(
    o3d_scene: o3d.t.geometry.RaycastingScene,
    origin_w: torch.Tensor,
    direction_w: torch.Tensor,
    o3d_mesh: o3d.geometry.TriangleMesh,
    th_on_surface: float = 1.0e-3,
    t_inf: float = 1e4,
    num_mesh_intersect_offset: int = 0,
    num_ray_samples: int = 3,
):
    """
    Compute the directed signed distance field.
    v3:  It is similar to signed distance function, but it only considers
    the surface points on the ray.

    The absolute value of the returned value is the absolute distance to the closest
    surface point on the ray.

    The sign of the returned value: positive if outside of any object; negative if inside.

    Args:
        o3d_scene:
            the o3d raycasting scene
        origin_w:
            (*, 3xyz_w) the ray origin in the world coordinate
        direction_w:
            (*, 3xyz_w) the ray direction in the world coordinate
        o3d_mesh:
            o3d mesh containing the texture, normal
        t_inf:
            if not None, dsd will be clamp to [-t_inf, t_inf]
        num_ray_samples:
            number of rays used to determine the point is inside or outside a mesh
            should be an odd number

    Returns:

        dsd:
            ray_t to the nearest surface on the ray.
            negative if the nearest surface is in the opposite direction of direction_w

        normal_w:
            (*, 3_xyz_w) in the opposite direction of direction_w, regardless of pos/neg is closer

        rgb:
            (*, 3rgb)

        hit_map:
            (*,) bool, whether the ray hit a surface

        is_inside:
            (*,) bool, whether origin_w is inside a closed surface
            we estimate this by checking the number of surface intersection from
            origin_w through direction_w
    """

    torch_dtype = origin_w.dtype
    device = origin_w.device
    *b_shape, _3xyz = origin_w.shape
    b = math.prod(b_shape)

    origin_w = origin_w.reshape(b, 3)  # (b, 3)
    direction_w = direction_w.reshape(b, 3)  # (b, 3)

    # unsigned distance
    ud = o3d_scene.compute_distance(
        origin_w.detach().cpu().float().numpy()  # (b, 3)  float32
    ).numpy()  # (b,)
    ud = torch.from_numpy(ud).to(dtype=torch_dtype, device=device)  # (b,)
    hit_at_the_point = ud <= th_on_surface

    # cast the rays toward + direction, get the intersections
    rays = torch.cat(
        (
            origin_w,
            direction_w,
        ),
        dim=-1,
    )  # (b, 6)
    rays = rays.detach().cpu().float().numpy()  # (b, 6) float32
    raycast_results_pos = o3d_scene.cast_rays(rays)

    # check origin_w is inside or outside a closed surface
    if num_ray_samples > 1:
        assert num_ray_samples % 2 == 1
        _origin_w = origin_w.expand(num_ray_samples, b, 3)
        _direction_w = direction_w.reshape(1, b, 3) + torch.randn(
            num_ray_samples, b, 3, dtype=direction_w.dtype, device=direction_w.device
        )
        _direction_w = torch.nn.functional.normalize(_direction_w, dim=-1)
        rays = torch.cat(
            (
                _origin_w,  # (num_samples, b, 3)
                _direction_w,  # (num_samples, b, 3)
            ),
            dim=-1,
        )  # (num_samples, b, 6)
        rays = rays.detach().cpu().float().numpy()

    intersection_counts_pos = torch.from_numpy(o3d_scene.count_intersections(rays).numpy())
    intersection_counts_pos = intersection_counts_pos - num_mesh_intersect_offset
    is_inside = (intersection_counts_pos % 2 == 1).to(dtype=torch.bool, device=device)  # (b,) bool
    if is_inside.ndim == 2:  # (num_sample, b) bool
        # vote
        is_inside = is_inside.float().mean(dim=0) >= 0.5  # (b,) bool

    # cast the rays backward (- direction), get the intersections
    rays = torch.cat(
        (
            origin_w,
            -1 * direction_w,
        ),
        dim=-1,
    )  # (b, 6)
    rays = rays.detach().cpu().float().numpy()  # (b, 6) float32
    raycast_results_neg = o3d_scene.cast_rays(rays)

    # select the nearest surface
    pos_closer = torch.from_numpy(raycast_results_pos["t_hit"].numpy()) <= torch.from_numpy(
        raycast_results_neg["t_hit"].numpy()
    )  # (b,) bool

    # merge raycast_results
    raycast_results = dict()
    pos_closer_np = pos_closer.detach().cpu().numpy()  # (b,) bool
    for name in [
        "t_hit",  # (b,)
        "geometry_ids",  # (b,)
        "primitive_ids",  # (b,)
        "primitive_uvs",  # (b, 2)
        "primitive_normals",  # (b, 3)
    ]:
        pos = raycast_results_pos[name].numpy()
        neg = raycast_results_neg[name].numpy()
        if len(pos.shape) == 1:
            out = np.where(np.reshape(pos_closer_np, (b,)), pos, neg)
        else:
            out = np.where(np.reshape(pos_closer_np, (b, 1)), pos, neg)
        raycast_results[name] = o3d.core.Tensor(out)
    del raycast_results_pos
    del raycast_results_neg

    t_hits = torch.from_numpy(raycast_results["t_hit"].numpy()).to(
        dtype=torch_dtype, device=device
    )  # (b,), inf if not hit the mesh
    hit_map = torch.logical_not(torch.isinf(t_hits))  # (b,) bool, true if hit a surface, 0 otherwise

    # if inside, set the sign to be negative
    dsd = t_hits * ((is_inside.to(dtype=torch_dtype, device=device) - 0.5) * (-2))  # (b,)
    if t_inf is not None:
        dsd = dsd.clamp(min=-t_inf, max=t_inf)

    # record the actual intersection point's ray_t (in terms of ray_direction_w)
    ray_t = t_hits * ((pos_closer.to(dtype=torch_dtype, device=device) - 0.5) * 2)  # (b,)

    # render rgb of the ray
    if o3d_mesh is not None and o3d_mesh.has_textures():
        ray_rgbs = render.interp_texture_map_from_ray_tracing_results(
            mesh=o3d_mesh,
            raycast_results=raycast_results,
            texture_maps=[skimage.img_as_float(np.array(img)).astype(np.float32) for img in o3d_mesh.textures],
            merge_textures=True,  # combine results from multiple textures.
        )[0]  # (b, 3)
    else:
        ray_rgbs = np.ones((b, 3), dtype=np.float32)
    ray_rgbs = torch.from_numpy(ray_rgbs).to(dtype=torch_dtype, device=device)  # (b, 3)

    # note that primitive_normals is the normal of the triangle face
    # we can use uv map to interpolate vertex normal
    # interpolate surface normal using uv map to get better normal estimation
    if o3d_mesh is not None and o3d_mesh.has_vertex_normals():
        surface_normals = render.interp_surface_normal_from_ray_tracing_results(
            mesh=o3d_mesh,
            raycast_results=raycast_results,
        )  # (b, 3)
    else:
        surface_normals = raycast_results["primitive_normals"].numpy()  # (b, 3)

    # debug
    # print(raycast_results['primitive_normals'].numpy())
    # end debug

    surface_normals = torch.from_numpy(surface_normals).to(dtype=torch_dtype, device=device)  # (b, 3)
    surface_normals = torch.nn.functional.normalize(
        surface_normals,
        dim=-1,
        eps=1e-9 if torch_dtype == torch.float32 else 1e-4,
    )
    # make sure surface_normals points at the negative direction of direction_w
    same_dir = ((surface_normals * direction_w).sum(dim=-1) > 0).to(dtype=torch_dtype)  # (b,)  {1, 0}
    surface_normals = surface_normals * ((same_dir - 0.5) * (-2)).unsqueeze(-1)  # (b, 3)

    # if not hit a surface, set surface normal to (0, 0, 0)
    surface_normals = surface_normals.masked_fill(~hit_map.unsqueeze(-1), 0)  # (b, 3)

    out_dict = dict(
        rgb=ray_rgbs,
        dsd=dsd,
        normal_w=surface_normals,
        hit_map=hit_map,
        is_inside=is_inside,
        ud=ud,
        hit_at_the_point=hit_at_the_point,
        ray_t=ray_t,
    )

    # reshape to (*b, d)
    out_dict = reshape(
        out_dict,
        start=0,
        end=0,
        shape=b_shape,
    )
    return out_dict


@linalg_utils.disable_tf32_and_autocast()
def get_neighbor_points_with_rasterization(
    xyz_w: torch.Tensor,  # (b, n, 3)
    H_c2w: torch.Tensor,  # (b, q, 4, 4)
    intrinsic: torch.Tensor,  # (b, q, 3, 3)
    width_px: int,
    height_px: int,
    k: int,
    method: str,  # 'cone', 'cylinder'
    radius: float,  # px, meter
    t_min: float = 0.0,
    t_max: float = 1.0e10,
    valid_mask: torch.Tensor = None,  # (b, n)
):
    """
    Gather the first k neighbor points by rasterizing points
    onto image plane.

    Args:
        xyz_w:
            (b, n, 3)
        H_c2w:
            (b, q, 4, 4)
        intrinsic:
            (b, q, 3, 3)
        k:
            max number of neighbor points to gather
        method:
            'cone': find the points in a cone surrounding a camera ray
            'cylinder': find the points in a cylindar surrounding a camera ray
        radius:
            if `method` == `cone`: number of pixels in radius on the image plane
            if `method` == `cylinder`: cylinder radius in meter
        t_min:
        t_max:
        valid_mask:
            (b, n)

    Returns:
        nidxs:
            (b, q, height_px, width_px, k), the index of `n` for each camera ray, padded with 0
        valid_mask:
            (b, q, height_px, width_px, k), whether nidx is valid
    """

    b, q, _41, _42 = H_c2w.shape
    _b, n, _3xyz = xyz_w.shape
    assert b == _b
    device = xyz_w.device
    dtype = xyz_w.dtype

    # convert point to the NDC used by pytorch3d
    K = torch.zeros(b, q, 4, 4, dtype=dtype, device=device)  # (b, q, 4, 4)
    K[..., 0, 0] = intrinsic[..., 0, 0]
    K[..., 1, 1] = intrinsic[..., 1, 1]
    K[..., 0, 2] = intrinsic[..., 0, 2]
    K[..., 1, 2] = intrinsic[..., 1, 2]
    K[..., 2, 2] = t_min + t_max
    K[..., 2, 3] = -(t_min * t_max)
    K[..., 3, 2] = 1

    N = torch.zeros(b, q, 4, 4, device=device)  # (b, q, 4, 4)
    N[..., 0, 0] = 2 / width_px
    N[..., 1, 1] = 2 / height_px
    N[..., 2, 2] = 2 / (t_max - t_min)
    N[..., 0, 3] = -1
    N[..., 1, 3] = -1
    N[..., 2, 3] = -(t_max + t_min) / (t_max - t_min)
    N[..., 3, 3] = 1

    proj_mtx = linalg_utils.matmul(N, K)  # (b, q, 4, 4)
    H_w2c = rigid_motion.inv_homogeneous_tensors(H_c2w)  # (b, q, 4, 4)

    # pytorch3d ndc is x to left, y to up, z to far
    H_c2o = (
        torch.tensor(
            [
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .to(dtype=proj_mtx.dtype, device=proj_mtx.device)
        .reshape(1, 1, 4, 4)
    )
    mvp_mtx = linalg_utils.matmul(H_c2o, proj_mtx)

    # convert xyz_w to xyz_ndc
    xyzw = torch.cat(
        [
            xyz_w,
            torch.ones(b, n, 1, dtype=xyz_w.dtype, device=xyz_w.device),
        ],
        dim=-1,
    )  # (b, n, 4)
    xyzc = linalg_utils.matmul(
        H_w2c.reshape(b, q, 1, 4, 4),
        xyzw.reshape(b, 1, n, 4, 1),
    ).squeeze(-1)  # (b, q, n, 4)

    # mark those before t_min and t_max as invalid
    # checking (abs(xyz_ndc[2]) <= 1 does not work if tmin = 0)
    valid_mask_ndc = torch.logical_and(
        xyzc[..., 2] >= t_min,
        xyzc[..., 2] < t_max,
    )  # (b, q, n)

    xyzw_ndc = linalg_utils.matmul(
        mvp_mtx.reshape(b, q, 1, 4, 4),
        xyzc.reshape(b, q, n, 4, 1),
    ).squeeze(-1)  # (b, q, n, 4)
    xyz_ndc = xyzw_ndc[..., :3] / xyzw_ndc[..., -1:]  # (b, q, n, 3)

    if valid_mask is None:
        valid_mask = valid_mask_ndc  # (b, q, n)
    else:
        valid_mask = torch.logical_and(
            valid_mask.unsqueeze(2),  # (b, 1, n)
            valid_mask_ndc,  # (b, q, n)
        )  # (b, q, n)

    # compute cone radius in ndc unit (indep. to z)
    if method == "cone_":
        tan_theta = np.tan(radius * (np.pi / 180.0))
        cx = intrinsic[..., 0, 2]  # (b, q)
        fx = intrinsic[..., 0, 0]  # (b, q)
        r_ndc = (2 / width_px) * cx - 1 + (2 / width_px * tan_theta) * fx  # (b, q)
        r_ndc = r_ndc.abs()
        r_ndc = r_ndc.reshape(b, q, 1).expand(b, q, n)  # (b, q, n)
    elif method == "cone":
        cx = intrinsic[..., 0, 2]  # (b, q)
        r_ndc = (2 * cx - width_px + 2 * radius) / width_px  # (b, q)
        r_ndc = r_ndc.abs()
        r_ndc = r_ndc.reshape(b, q, 1).expand(b, q, n)  # (b, q, n)
    elif method == "cylinder":
        _xyzc = xyzc.clone()  # (b, q, n, 4)
        # we use fx to determine the radius
        _xyzc[..., 0] = radius
        r_ndc = linalg_utils.matmul(
            mvp_mtx.reshape(b, q, 1, 4, 4),
            _xyzc.reshape(b, q, n, 4, 1),
        ).squeeze(-1)  # (b, q, n, 4)
        r_ndc = r_ndc[..., 0] / r_ndc[..., -1]  # (b, q, n)
        r_ndc = r_ndc.abs()
    else:
        raise NotImplementedError

    # convert r_ndc to packed format
    packed_dict = pr_utils.pack(
        val_matrix=r_ndc.reshape(b * q, n),  # (bq, n)
        valid_mask=valid_mask.reshape(b * q, n),  # (bq, n)
    )
    r_ndc_packed = packed_dict["val_arr"]  # (n_total,)

    # convert xyz_ndc to packed format
    packed_dict = pr_utils.pack(
        val_matrix=xyz_ndc.reshape(b * q, n, 3),  # (bq, n, 3)
        valid_mask=valid_mask.reshape(b * q, n),  # (bq, n)
    )
    xyz_ndc_packed = packed_dict["val_arr"]  # (n_total, 3xyz)
    xyz_ndc_start_idx = packed_dict["start_idxs"]  # (bq,)
    xyz_ndc_count = packed_dict["counts"]  # (bq,)

    pidx, zbuf_ndc, dists2_ndc = rasterize_points_packed(
        xyz_ndc_packed=xyz_ndc_packed,
        xyz_ndc_start_idx=xyz_ndc_start_idx,
        xyz_ndc_count=xyz_ndc_count,
        image_size=(height_px, width_px),
        radius_ndc_packed=r_ndc_packed,
        points_per_pixel=k,
    )
    # pidx: (bq, h, w, k) index of xyz_ndc_packed, padded with -1
    # zbuf_ndc: (bq, h, w, k) z_ndc of the point
    # dist2_ndc: (bq, h, w, k) xy dist (in ndc)
    valid_pidx = pidx >= 0  # (bq, h, w, k)
    pidx[~valid_pidx] = 0

    packed_dict = pr_utils.pack(
        val_matrix=torch.arange(n, device=device).expand(b * q, n),  # (bq, n)
        valid_mask=valid_mask.reshape(b * q, n),  # (bq, n)
    )
    nidx_packed = packed_dict["val_arr"]  # (n_total,)
    if nidx_packed.numel() > 0:
        nidx = nidx_packed[pidx.long()]  # (bq, h, w, k)  padded with 0
    else:
        nidx = torch.zeros_like(valid_pidx)  # (bq, h, w, k)

    return dict(
        nidxs=nidx.reshape(b, q, height_px, width_px, k),
        valid_mask=valid_pidx.reshape(b, q, height_px, width_px, k),
    )


@linalg_utils.disable_tf32_and_autocast()
def rasterize_points_packed(
    xyz_ndc_packed: torch.Tensor,  # (n_total, 3xyz_w)
    xyz_ndc_start_idx: torch.Tensor,  # (b,)
    xyz_ndc_count: torch.Tensor,  # (b,)
    image_size: T.Union[int, T.List[int], T.Tuple[int, int]] = 256,
    radius_ndc_packed: T.Union[float, torch.Tensor] = 0.01,  # (n_total,)
    points_per_pixel: int = 8,
    bin_size: T.Optional[int] = None,
    max_points_per_bin: T.Optional[int] = None,
):
    """(modified from pytorch3d's implementation to take packed directly)
    Each pointcloud is rasterized onto a separate image of shape
    (H, W) if `image_size` is a tuple or (image_size, image_size) if it
    is an int.

    If the desired image size is non square (i.e. a tuple of (H, W) where H != W)
    the aspect ratio needs special consideration. There are two aspect ratios
    to be aware of:
        - the aspect ratio of each pixel
        - the aspect ratio of the output image
    The camera can be used to set the pixel aspect ratio. In the rasterizer,
    we assume square pixels, but variable image aspect ratio (i.e rectangle images).

    In most cases you will want to set the camera aspect ratio to
    1.0 (i.e. square pixels) and only vary the
    `image_size` (i.e. the output image dimensions in pix

    Args:
        xyz_ndc_packed:
            (n_total, 3xyz)
            Pointclouds representing a batch of point clouds to be
            rasterized. This is a batch of N pointclouds in a packed format,
            where each point cloud can have a different number of points;
            the coordinates of each point are (x, y, z). The coordinates are expected to
            be in normalized device coordinates (NDC): [-1, 1]^3 with the camera at
            (0, 0, 0); In the camera coordinate frame the x-axis goes from right-to-left,
            the y-axis goes from bottom-to-top, and the z-axis goes from back-to-front.
        xyz_ndc_start_idx:
            (b,) where the i-th point cloud in `xyz_w_packed` starts
        xyz_ndc_count:
            (b,) number of points in the i-th point cloud
        image_size: Size in pixels of the output image to be rasterized.
            Can optionally be a tuple of (H, W) in the case of non square images.
        radius_ndc_packed (Optional): The radius (in NDC units) of the disk to
            be rasterized. This can either be a float in which case the same radius is used
            for each point, or a torch.Tensor of shape (n_total,) giving a radius per point
            in the batch.
        points_per_pixel (Optional): We will keep track of this many points per
            pixel, returning the nearest points_per_pixel points along the z-axis
        bin_size: Size of bins to use for coarse-to-fine rasterization. Setting
            bin_size=0 uses naive rasterization; setting bin_size=None attempts to
            set it heuristically based on the shape of the input. This should not
            affect the output, but can affect the speed of the forward pass.
        max_points_per_bin: Only applicable when using coarse-to-fine rasterization
            (bin_size > 0); this is the maximum number of points allowed within each
            bin. This should not affect the output values, but can affect
            the memory usage in the forward pass.

    Returns:
        3-element tuple containing

        - **idx**: int32 Tensor of shape (N, image_size, image_size, points_per_pixel)
          giving the indices of the nearest points at each pixel, in ascending
          z-order. Concretely `idx[n, y, x, k] = p` means that `points[p]` is the kth
          closest point (along the z-direction) to pixel (y, x) - note that points
          represents the packed points of shape (P, 3).
          Pixels that are hit by fewer than points_per_pixel are padded with -1.
        - **zbuf**: Tensor of shape (N, image_size, image_size, points_per_pixel)
          giving the z-coordinates of the nearest points at each pixel, sorted in
          z-order. Concretely, if `idx[n, y, x, k] = p` then
          `zbuf[n, y, x, k] = points[p, 2]`. Pixels hit by fewer than
          points_per_pixel are padded with -1
        - **dists2**: Tensor of shape (N, image_size, image_size, points_per_pixel)
          giving the squared Euclidean distance (in NDC units) in the x/y plane
          for each point closest to the pixel. Concretely if `idx[n, y, x, k] = p`
          then `dists[n, y, x, k]` is the squared distance between the pixel (y, x)
          and the point `(points[p, 0], points[p, 1])`. Pixels hit with fewer
          than points_per_pixel are padded with -1.

        In the case that image_size is a tuple of (H, W) then the outputs
        will be of shape `(N, H, W, ...)`.
    """
    points_packed = xyz_ndc_packed  # (n_total, 3)
    cloud_to_packed_first_idx = xyz_ndc_start_idx  # (b,)
    num_points_per_cloud = xyz_ndc_count  # (b,)

    radius = radius_ndc_packed
    if isinstance(radius, (float, int)):
        radius = torch.ones_like(points_packed) * radius
    assert radius.shape == (points_packed.size(0),)

    # In the case that H != W use the max image size to set the bin_size
    # to accommodate the num bins constraint in the coarse rasterizer.
    # If the ratio of H:W is large this might cause issues as the smaller
    # dimension will have fewer bins.
    # TODO: consider a better way of setting the bin size.
    if isinstance(image_size, int):
        im_size = (image_size, image_size)
    else:
        im_size = image_size
    assert len(im_size) == 2
    max_image_size = max(*im_size)

    if bin_size is None:
        if not points_packed.is_cuda:
            # Binned CPU rasterization not fully implemented
            bin_size = 0
        else:
            bin_size = int(2 ** max(np.ceil(np.log2(max_image_size)) - 4, 4))

    if bin_size != 0:
        kMaxPointsPerBin = 22
        # There is a limit on the number of points per bin in the cuda kernel.
        points_per_bin = 1 + (max_image_size - 1) // bin_size
        if points_per_bin >= kMaxPointsPerBin:
            raise ValueError(
                "bin_size too small, number of points per bin must be less than %d; got %d"
                % (kMaxPointsPerBin, points_per_bin)
            )

    if max_points_per_bin is None:
        max_points_per_bin = int(max(10000, xyz_ndc_count.max().item() / 5))

    # Function.apply cannot take keyword args, so we handle defaults in this
    # wrapper and call apply with positional args only

    return RasterizePoints.apply(
        points_packed,
        cloud_to_packed_first_idx,
        num_points_per_cloud,
        im_size,
        radius,
        points_per_pixel,
        bin_size,
        max_points_per_bin,
    )


def get_neighbor_points_with_pulsar_chunk(
    xyz_w: torch.Tensor,  # (b, n, 3)
    H_c2w: torch.Tensor,  # (b, q, 4, 4)
    intrinsic: torch.Tensor,  # (b, q, 3, 3)
    width_px: int,
    height_px: int,
    k: int,
    radius: T.Union[float, torch.Tensor],  # meter, (b, n)
    t_min: float = 0.0,
    t_max: float = 1.0e10,
    pulsar_renderer: PulsarRenderer = None,
    max_q_chunk_size: int = -1,
) -> T.Dict[str, torch.Tensor]:
    b, q, _41, _42 = H_c2w.shape
    if max_q_chunk_size <= 0 or q <= max_q_chunk_size:
        return get_neighbor_points_with_pulsar(
            xyz_w=xyz_w,  # (b, n, 3)
            H_c2w=H_c2w,  # (b, q, 4, 4)
            intrinsic=intrinsic,  # (b, q, 3, 3)
            width_px=width_px,
            height_px=height_px,
            k=k,
            radius=radius,  # meter
            t_min=t_min,
            t_max=t_max,
            pulsar_renderer=pulsar_renderer,
        )

    num_chunks = (q + max_q_chunk_size - 1) // max_q_chunk_size
    H_c2ws = H_c2w.chunk(num_chunks, dim=1)
    intrinsics = intrinsic.chunk(num_chunks, dim=1)

    all_nidxs = []
    all_valid_masks = []
    for i in range(len(H_c2ws)):
        out_dict = get_neighbor_points_with_pulsar(
            xyz_w=xyz_w,  # (b, n, 3)
            H_c2w=H_c2ws[i],  # (b, q', 4, 4)
            intrinsic=intrinsics[i],  # (b, q', 3, 3)
            width_px=width_px,
            height_px=height_px,
            k=k,
            radius=radius,  # meter
            t_min=t_min,
            t_max=t_max,
            pulsar_renderer=pulsar_renderer,
        )
        nidxs = out_dict["nidxs"]  # (b, q', h, w, k)
        valid_mask = out_dict["valid_mask"]  # (b, q', h, w, k)
        pulsar_renderer = out_dict["pulsar_renderer"]
        all_nidxs.append(nidxs)
        all_valid_masks.append(valid_mask)
    nidxs = torch.cat(all_nidxs, dim=1)  # (b, q, h, w, k)
    valid_mask = torch.cat(all_valid_masks, dim=1)  # (b, q, h, w, k)
    assert nidxs.size(1) == q
    assert valid_mask.size(1) == q

    return dict(
        nidxs=nidxs.reshape(b, q, height_px, width_px, k),
        valid_mask=valid_mask.reshape(b, q, height_px, width_px, k),
        pulsar_renderer=pulsar_renderer,
    )


def get_neighbor_points_with_pulsar(
    xyz_w: torch.Tensor,  # (b, n, 3)
    H_c2w: torch.Tensor,  # (b, q, 4, 4)
    intrinsic: torch.Tensor,  # (b, q, 3, 3)
    width_px: int,
    height_px: int,
    k: int,
    radius: T.Union[float, torch.Tensor],  # meter, (b, n)
    t_min: float = 0.0,
    t_max: float = 1.0e10,
    pulsar_renderer: PulsarRenderer = None,
) -> T.Dict[str, torch.Tensor]:
    """
    Gather the first k neighbor points (in depth) by rasterizing points
    onto image plane with pulsar's implementation. Pulsar gathers all the
    spheres with `radius` that intersect with the frustrum of a pixel (ie. the cone).

    Args:
        xyz_w:
            (b, n, 3)
        H_c2w:
            (b, q, 4, 4)
        intrinsic:
            (b, q, 3, 3)
        k:
            max number of neighbor points to gather. If more points than k,
            pulsar prioritizes those closer to the pinhole.
        radius:
            float or (b, n) or (b, n, 1) the radius of the sphere in meter
        t_min:
        t_max:
        pulsar_renderer:
            existing pulsar renderer that is already initialized with the
            correct image resolution, etc.

    Returns:
        nidxs:
            (b, q, height_px, width_px, k), the index of `n` for each camera ray, padded with 0
        valid_mask:
            (b, q, height_px, width_px, k), whether nidx is valid
        pulsar_renderer:

    """
    b, n, _3xyz = xyz_w.shape
    _b, q, _41, _42 = H_c2w.shape
    bq = b * q
    assert b == _b
    c = 1
    device = xyz_w.device
    dtype = xyz_w.dtype

    if pulsar_renderer is None:
        max_num_balls = int(1e6)
        while max_num_balls < n:
            max_num_balls *= 2

        pulsar_renderer = PulsarRenderer(
            width=width_px,
            height=height_px,
            max_num_balls=max_num_balls,
            orthogonal_projection=False,
            right_handed_system=True,
            background_normalized_depth=1e-9,
            n_channels=c,  # since we are not using color
            n_track=k,
        ).to(device=device)

    H_c2w = H_c2w.reshape(bq, 4, 4)  # (bq, 4, 4)
    pinhole_pos = H_c2w[:, :3, 3]  # (bq, 3)
    R_c2w = H_c2w[:, :3, :3].clone()  # (bq, 3, 3)
    # since pulsar takes image coordinate (which is x to right, y to down, z to front)
    # whereas our coordinate is x to right, y to up, z to us
    R_c2w[:, :, 1] *= -1
    R_c2w[:, :, 2] *= -1
    R_c2w_two_rows = R_c2w[:, :2].reshape(bq, 6)  # (bq, 6)

    intrinsic = intrinsic.reshape(bq, 3, 3)  # (bq, 3, 3)
    fx, fy = intrinsic[:, 0, 0], intrinsic[:, 1, 1]  # (bq,)
    assert torch.allclose(fx, fy)
    f = fx  # focal length in px
    focal_length_px = f / width_px
    znear = 0.1  # meter
    focal_length = torch.tensor([znear - 1e-5], dtype=intrinsic.dtype, device=intrinsic.device)
    sensor_width = focal_length / focal_length_px
    f_meter = f * sensor_width / width_px  # (bq,)  focal length in meter
    cx = intrinsic[:, 0, 2]  # (bq,)
    cy = intrinsic[:, 1, 2]  # (bq,)
    # Transfer principal point offset into centered offset.
    d_cx = -(cx - width_px / 2)  # (bq,)
    d_cy = -(cy - height_px / 2)  # (bq,)

    cam_params = torch.cat(
        [
            pinhole_pos,  # (bq, 3)
            R_c2w_two_rows,  # (bq, 6)
            f_meter.unsqueeze(-1),  # (bq, 1)  focal length in meter
            sensor_width.unsqueeze(-1),  # (bq, 1)  in meter
            d_cx.unsqueeze(-1),  # (bq, 1)
            d_cy.unsqueeze(-1),  # (bq, 1)
        ],
        dim=-1,
    )  # (bq, 13)

    if t_min is None or t_min < 1e-6:
        min_depth = 0
    else:
        min_depth = max(f_meter.max().item() + 1e-6, t_min)

    with _nullcontext():
        if isinstance(radius, float):
            vert_rad = torch.ones(bq, n, dtype=torch.float, device=xyz_w.device) * radius  # (bq, n)
        else:
            vert_rad = radius.reshape(b, 1, n).expand(b, q, n).reshape(bq, n)  # (bq, n)

        img, render_info = pulsar_renderer(
            vert_pos=xyz_w.reshape(b, 1, n, 3).expand(b, q, n, 3).reshape(bq, n, 3).float(),  # (bq, n, 3)
            vert_col=torch.zeros(bq, n, c, dtype=torch.float, device=xyz_w.device),  # (bq, n, c)
            vert_rad=vert_rad,
            cam_params=cam_params,  # (bq, 13)
            gamma=0.5,
            max_depth=t_max,
            min_depth=min_depth,
            percent_allowed_difference=0,
            mode=0,
            return_forward_info=True,  # mode needs to be 0
        )
        point_ids = pulsar_renderer.sphere_ids_from_result_info_nograd(render_info).long()  # (bq, h, w, k)

        if point_ids.size(-1) > k:
            point_ids = point_ids[..., :k]

    valid_mask = point_ids >= 0  # (bq, h, w, k)
    point_ids[~valid_mask] = 0

    return dict(
        nidxs=point_ids.reshape(b, q, height_px, width_px, k),
        valid_mask=valid_mask.reshape(b, q, height_px, width_px, k),
        pulsar_renderer=pulsar_renderer,
    )


def create_pcd_with_scene_craving(
    rgb: torch.Tensor,  # (q, h, w, 3)
    valid_mask: torch.Tensor,  # (q, h, w)
    H_c2w: torch.Tensor,  # (q, 4, 4)
    intrinsic: torch.Tensor,  # (q, 3, 3)
    world_size: torch.Tensor,  # (3, 2)
    num_points: int,
    remove_hidden_points: bool = True,
    device: torch.device = torch.device("cpu"),
) -> T.Dict[str, torch.Tensor]:
    """
    Given rgb images and their valid_mask (foreground mask),
    use scene carving to create a point cloud.

    Args:
        rgb:
            rgb images. (q, h, w, 3)
        valid_mask:
            valid_masks, where >= 0.5 is considered foreground. (q, h, w)
        H_c2w:
            (q, 4, 4) that maps a position in camera coordinate to the
            world coordinate
        intrinsic:
            (q, 3, 3) camera intrinsics
        world_size:
            (3, 2) [xmin xmax; ymin ymax; zmin zmax]
        num_points:
            number of points per dimension to crave from
        remove_hidden_points:
            whether to remove hidden points.
            Note that since we use rasterization to remove hidden points,
            in this case the image resolution used in the rasterization
            essentially defines the max number of points.

    Returns:
        xyz_w:
            (n', 3xyz_w)
        rgb:
            (n', 3rgb)
    """

    q, h, w, _3 = rgb.shape
    n = num_points

    intrinsic = intrinsic.to(device=device)
    rgb = rgb.to(device=device)
    valid_mask = valid_mask.to(device=device)
    H_c2w = H_c2w.to(device=device)
    intrinsic = intrinsic.to(device=device)
    world_size = world_size.to(device=device)

    # create coordinate
    xs = torch.linspace(world_size[0, 0], world_size[0, 1], n, device=device)  # (n,)
    ys = torch.linspace(world_size[1, 0], world_size[1, 1], n, device=device)  # (n,)
    zs = torch.linspace(world_size[2, 0], world_size[2, 1], n, device=device)  # (n,)

    # we will crave one z-plane at a time
    all_xyz_ws = []
    all_rgbs = []
    for z_idx in range(n):
        x, y = torch.meshgrid(xs, ys, indexing="ij")  # (n, n)
        z = torch.ones_like(x) * zs[z_idx]  # (n, n)
        xyz_w = torch.stack([x, y, z], dim=-1)  # (n, n, 3)

        # project xyz_w onto each images
        uv_c, xyz_c = pinhole_projection(
            xyz_w=xyz_w,  # (n, n, 3)
            intrinsics=intrinsic,  # (q, 3, 3)
            H_c2w=H_c2w,  # (q, 3, 3)
            dim_b=0,
        )  # (q, n, n, 2uv) [0, w] [0, h], (q, n, n, 3xyz_c)

        p_rgb = uv_sampling(
            uv=uv_c,  # (q, n, n, 2)
            feature_map=rgb,  # (q, h, w, 3)
            mode="bilinear",
            padding_mode="zeros",
            uv_normalized=False,
        )  # (q, n, n, 3rgb)

        p_valid = uv_sampling(
            uv=uv_c,  # (q, n, n, 2)
            feature_map=valid_mask.float().unsqueeze(-1),  # (q, h, w, 1)
            mode="bilinear",
            padding_mode="zeros",
            uv_normalized=False,
        ).squeeze(-1)  # (q, n, n)

        # ignore out of bound
        oob_mask_u = torch.logical_or(
            uv_c[..., 0] < 0,  # (q, n, n)
            uv_c[..., 0] > w,  # (q, n, n)
        )  # (q, n, n)
        oob_mask_v = torch.logical_or(
            uv_c[..., 1] < 0,  # (q, n, n)
            uv_c[..., 1] > h,  # (q, n, n)
        )  # (q, n, n)
        oob_mask = torch.logical_or(oob_mask_u, oob_mask_v)  # (q, n, n)
        p_valid = torch.logical_or(
            p_valid,
            oob_mask,
        )  # (q, n, n)

        # if a point is not valid in any image, it is filtered out
        vmask = (p_valid > 0.5).sum(dim=0) > (q - 1)  # (n, n), bool
        vrgb = p_rgb.mean(dim=0)  # (n, n, 3rgb)

        all_xyz_ws.append(xyz_w[vmask])  # (n', 3)
        all_rgbs.append(vrgb[vmask])  # (n', 3)

    all_xyz_ws = torch.cat(all_xyz_ws, dim=0)  # (n', 3)
    all_rgbs = torch.cat(all_rgbs, dim=0)  # (n', 3)

    # remove hidden points
    if remove_hidden_points:
        radius = ((world_size[:, 1] - world_size[:, 0]) / n).max().item()
        k = 5
        resolution_scale = 2
        new_h, new_w = int(h * resolution_scale), int(w * resolution_scale)
        resolution_scale_w = new_w / w
        resolution_scale_h = new_h / h
        # pulsar needs fx = fy
        assert np.isclose(resolution_scale_w, resolution_scale_h)
        new_intrinsic = intrinsic.clone()
        new_intrinsic[..., 0, 0] *= resolution_scale_w
        new_intrinsic[..., 1, 1] *= resolution_scale_w
        new_intrinsic[..., 0, 2] *= resolution_scale_w
        new_intrinsic[..., 1, 2] *= resolution_scale_w

        with torch.no_grad():
            out_dict = get_neighbor_points_with_pulsar_chunk(
                xyz_w=all_xyz_ws.unsqueeze(0),  # (1, n', 3)
                H_c2w=H_c2w.unsqueeze(0),  # (1, q, 4, 4)
                intrinsic=new_intrinsic.unsqueeze(0),  # (1, q, 3, 3)
                width_px=new_w,
                height_px=new_h,
                k=k,
                radius=radius,
                t_min=0,
                t_max=world_size.abs().max() * 10 + 100,
                max_q_chunk_size=1,
            )
            nidxs = out_dict["nidxs"]  # (b=1, q, h, w, k)
            valid_mask = out_dict["valid_mask"]  # (b=1, q, h, w, k)

        # gather all_xyz_ws
        nidxs = nidxs[valid_mask].unique()  # (m,)
        all_xyz_ws = all_xyz_ws[nidxs]  # (m, 3)
        all_rgbs = all_rgbs[nidxs]  # (m, 3)

    return dict(
        xyz_w=all_xyz_ws,  # (n', 3xyz_w)
        rgb=all_rgbs,  # (n', 3rgb)
    )


@linalg_utils.disable_tf32_and_autocast()
def calculate_patch_intrinsic(
    img_intrinsic: torch.Tensor,  # (b, 3, 3)
    img_width_px: int,
    img_height_px: int,
    patch_topleft_ij: torch.Tensor,  # (b, p, 2ij)  p patches
    patch_width_px: int,
    patch_height_px: int,
):
    """
    Calculate the intrinsics that corresponds to rendering a patch.

    Args:
        img_intrinsic:
            (b, 3, 3) the original intrinsic of the full image
        img_width_px:
            original image width in pixel
        img_height_px:
            original image height in pixel
        patch_topleft_ij:
            the corresponding ij in the original image of the patch's top left corner.
            For example, if we want to get the top left patch, set patch_topleft_ij to be
            (0, 0), which corresponding to the top left pixel.
            i is along the x axis, and j is along the y axis
        patch_width_px:
        patch_height_px:

    Returns:
        new_intrinsics:
            (b, p, 3, 3) that will sample the patch.
    """

    b, p, _2ij = patch_topleft_ij.shape

    # subsample
    patch_intrinsic = img_intrinsic.unsqueeze(1).expand(b, p, 3, 3).clone()  # (b, p, 3, 3)

    # since the camera's physical focal length and pixel pitch
    # do not change, no need to change the focal length.
    # Instead, we need to adjust cx and cy.
    patch_intrinsic[..., 0, 2] -= (img_width_px - patch_width_px) / 2
    patch_intrinsic[..., 1, 2] -= (img_height_px - patch_height_px) / 2

    # we need uv to align with corner (integer)
    current_top_left_corner_u = img_width_px / 2 - patch_width_px / 2
    current_top_left_corner_v = img_height_px / 2 - patch_height_px / 2

    # then apply shift
    new_top_left_corner_u = patch_topleft_ij[..., 0]  # (b, p)
    new_top_left_corner_v = patch_topleft_ij[..., 1]  # (b, p)

    # align corner
    patch_intrinsic[..., 0, 2] -= new_top_left_corner_u - current_top_left_corner_u
    patch_intrinsic[..., 1, 2] -= new_top_left_corner_v - current_top_left_corner_v

    return patch_intrinsic  # (b, p, 3, 3)


@linalg_utils.disable_tf32_and_autocast()
def get_ray_sphere_intersection(
    ray_origin: torch.Tensor,  # (*r, 3xyz)
    ray_direction: torch.Tensor,  # (*r, 3xyz)
    sphere_center: torch.Tensor,  # (*r, *s, 3xyz)
    sphere_radius: T.Union[torch.Tensor, float],  # (*r, *s)
    ignore_inside: bool,
) -> T.Dict[str, T.Any]:
    """
    Compute the two intersection points between rays and spheres.

    Args:
        ray_origin:
            (*r, 3)
        ray_direction:
            (*r, 3)
        sphere_center:
            (*r, *s, 3) or (3,)
        sphere_radius:
            (*r, *s) or float
        ignore_inside:
            if True, we set intersect_mask to be False if ray_origin is inside the sphere

    Returns:
        ta:
            (*r_shape, *s_shape), t_near. If not intersected with sphere (intersect_sphere is False), set to 0
        tb:
            (*r_shape, *s_shape), t_far.  If not intersected with sphere (intersect_sphere is False), set to 0
        inside:
            (*r_shape, *s_shape)  whether ray origin is inside the sphere
        intersected:
            (*r_shape, *s_shape)  whether ray intersect with sphere (sphere could be behind the ray)
        intersect_mask:
            (*r_shape, *s_shape) bool, True if ray intersect with sphere

    Note:
        if ray_origin is in sphere, we set it to invalid
    """

    *r_shape, _3 = ray_origin.shape
    if sphere_center.ndim > 1:
        *s_shape, _3 = sphere_center.shape[len(r_shape) :]
    else:
        sphere_center = sphere_center.reshape(*([1] * len(r_shape)), 3).expand(*r_shape, 3)  # (*r, 3)
        s_shape = []  # ss = 1
    rr = math.prod(r_shape)
    ss = math.prod(s_shape)

    ray_origin = ray_origin.reshape(rr, 1, 3)  # (rr, 1, 3)
    ray_direction = ray_direction.reshape(rr, 1, 3)  # (rr, 1, 3)
    sphere_center = sphere_center.reshape(rr, ss, 3)  # (rr, ss, 3)

    if isinstance(sphere_radius, torch.Tensor) and sphere_radius.numel() > 1:
        sphere_radius = sphere_radius.reshape(rr, ss)  # (rr, ss)

    # compute ta and tb
    p_ro = sphere_center - ray_origin  # (rr, ss, 3)
    sphere_radius2 = sphere_radius**2  # (rr, ss)
    with torch.profiler.record_function("matmul: p_ro @ rd"):
        t_proj = linalg_utils.matmul(
            p_ro.reshape(rr, ss, 1, 3),
            ray_direction.reshape(rr, 1, 3, 1),
        ).reshape(rr, ss)  # (rr, ss)
    with torch.profiler.record_function("p_ro_2"):
        p_ro2 = (p_ro**2).sum(dim=-1)  # (rr, ss)

    with torch.profiler.record_function("dr2"):
        dr2 = p_ro2 - (t_proj**2)  # (rr, ss)

    inside_sphere = p_ro2 < sphere_radius2  # (rr, ss)
    intersected = dr2 < sphere_radius2  # (rr, ss)

    # # intersect_mask = dr2 <= (sphere_radius ** 2)  # (rr, ss) bool
    # # we do not count just intersect as intersect
    # intersect_mask = torch.logical_and(
    #     dr2 < (sphere_radius2 - 1e-6),  # (rr, ss) bool
    #     t_proj > 0.001,   # (rr, ss),  we ignore all points inside the sphere
    # )

    with torch.profiler.record_function("dt"):
        dt = (sphere_radius2 - dr2).clamp(min=1e-10).sqrt()  # (rr, ss)
        ta = t_proj - dt  # (rr, ss)  # ta is always smaller than tb (for negative, it is further away)
        tb = t_proj + dt  # (rr, ss)

    in_front = torch.logical_and(
        ta >= 0,
        tb >= 0,
    )  # (rr, ss)

    # we consider intersect if it actually intersect and is inside or infront
    if ignore_inside:
        intersect_mask = torch.logical_and(
            intersected,
            in_front,
        )  # (rr, ss)
    else:
        intersect_mask = torch.logical_and(
            intersected,
            torch.logical_or(in_front, inside_sphere),
        )  # (rr, ss)

    # set nonintersected ta, tb to 0
    ta = ta.masked_fill(~intersected, 0)
    tb = tb.masked_fill(~intersected, 0)

    if len(r_shape) > 0 or len(s_shape) > 0:
        ta = ta.reshape(*r_shape, *s_shape)  # (*r, *s)
        tb = tb.reshape(*r_shape, *s_shape)  # (*r, *s)
        intersect_mask = intersect_mask.reshape(*r_shape, *s_shape)  # (*r, *s)
        inside_sphere = inside_sphere.reshape(*r_shape, *s_shape)  # (*r, *s)
        intersected = intersected.reshape(*r_shape, *s_shape)  # (*r, *s)
    else:
        ta = ta.reshape(-1)  # (,)
        tb = tb.reshape(-1)  # (,)
        intersect_mask = intersect_mask.reshape(-1)  # (,)
        inside_sphere = inside_sphere.reshape(-1)  # (,)
        intersected = intersected.reshape(-1)  # (,)

    return dict(
        ta=ta,  # (*r, *s)
        tb=tb,  # (*r, *s)
        intersect_mask=intersect_mask,  # (*r, *s)
        inside=inside_sphere,  # (*r, *s)
        intersected=intersected,  # (*r, *s)
    )


@linalg_utils.disable_tf32_and_autocast()
def get_ray_ellipsoid_intersection_ori(
    ray_origin: torch.Tensor,  # (*r, 3xyz)
    ray_direction: torch.Tensor,  # (*r, 3xyz)
    ellipsoid_center: torch.Tensor,  # (*r, *s, 3xyz)
    ellipsoid_radius: torch.Tensor,  # (*r, *s, 3xyz)
    ellipsoid_R_e2w: torch.Tensor,  # (*r, *s, 3, 3)
    ignore_inside: bool,
) -> T.Dict[str, T.Any]:
    """
    Compute the two intersection points between rays and spheres.

    Args:
        ray_origin:
            (*r, 3)
        ray_direction:
            (*r, 3)
        ellipsoid_center:
            (*r, *s, 3)
        ellipsoid_radius:
            (*r, *s, 3) radius before rotation (ie, in the e coordinate)
        ellipsoid_R_e2w:
            (*r, *s, 3, 3) R_e2w (columns are 3 axes of the ellipsoid in the world coordinate)
        ignore_inside:
            if True, we set intersect_mask to be False if ray_origin is inside the ellipsoid

    Returns:
        ta:
            (*r_shape, *s_shape), t_near. If not intersected, pad with 0
        tb:
            (*r_shape, *s_shape), t_far.  If not intersected, pad with 0
        intersect_mask:
            (*r_shape, *s_shape) bool, True if ray intersect with sphere

    Note:
        if ray_origin is in sphere, we set it to invalid
    """

    *r_shape, _3 = ray_origin.shape
    *s_shape, _3 = ellipsoid_center.shape[len(r_shape) :]
    rr = math.prod(r_shape)
    ss = math.prod(s_shape)

    ray_origin = ray_origin.reshape(rr, 1, 3)  # (rr, 1, 3)
    ray_direction = ray_direction.reshape(rr, 1, 3)  # (rr, 1, 3)
    ellipsoid_center = ellipsoid_center.reshape(rr, ss, 3)  # (rr, ss, 3)
    ellipsoid_radius = ellipsoid_radius.reshape(rr, ss, 3)  # (rr, ss, 3)
    ellipsoid_R_e2w = ellipsoid_R_e2w.reshape(rr, ss, 3, 3)  # (rr, ss, 3, 3)

    # our strategy is to use ray_unit_sphere intersection
    # step 1: construct H_e2w = [R_e2w, e_center]
    # step 2: compute H_w2e
    # step 3: transform ro and rd by H_w2e
    # step 4: scale ro and rd by inv(scale)
    # step 5: calculate ray unit sphere intersection => xyz1, xyz2
    # step 6: scale xyz1 and xyz2 by scale
    # step 7: compute ta and tb

    # transform ro and rd to the ellipsoid coordinate
    ro_e = ray_origin - ellipsoid_center  # (rr, ss, 3)
    ellipsoid_R_w2e = ellipsoid_R_e2w.transpose(-1, -2)  # (rr, ss, 3, 3)
    ro_e = linalg_utils.matmul(
        ellipsoid_R_w2e,  # (rr, ss, 3, 3)
        ro_e.unsqueeze(-1),
    ).squeeze(-1)  # (rr, ss, 3xyz_e)
    rd_e = linalg_utils.matmul(
        ellipsoid_R_w2e,  # (rr, ss, 3, 3)
        ray_direction.unsqueeze(-1),
    ).squeeze(-1)  # (rr, ss, 3xyz_e)

    # create a fake rt (to compute rd in the scaled space)
    rt_e = ro_e + rd_e  # (rr, ss, 3xyz_e)

    # apply inv scale
    ro_e_s = ro_e / ellipsoid_radius  # (rr, ss, 3xyz_es)
    rt_e_s = rt_e / ellipsoid_radius  # (rr, ss, 3xyz_es)
    rd_e_s = torch.nn.functional.normalize(rt_e_s - ro_e_s, dim=-1)  # (rr, ss, 3xyz_es)

    # ray unit-sphere intersection
    out_dict = get_ray_sphere_intersection(
        ray_origin=ro_e_s,  # (rr, ss, 3xyz_es)
        ray_direction=rd_e_s,  # (rr, ss, 3xyz_es)
        sphere_center=torch.zeros(rr, ss, 3, dtype=ro_e_s.dtype, device=ro_e_s.device),  # (rr, ss, 3xyz_es)
        sphere_radius=torch.ones(rr, ss, dtype=ro_e_s.dtype, device=ro_e_s.device),  # (rr, ss)
    )
    ta_e_s = out_dict["ta"]  # (rr, ss)
    tb_e_s = out_dict["tb"]  # (rr, ss)
    intersect_mask = out_dict["intersect_mask"]  # (rr, ss) bool
    inside = out_dict["inside"]  # (rr, ss) bool
    intersected = out_dict["intersected"]  # (rr, ss) bool

    xyz_e_s_a = ro_e_s + ta_e_s.unsqueeze(-1) * rd_e_s  # (rr, ss, 3xyz_es)
    xyz_e_s_b = ro_e_s + tb_e_s.unsqueeze(-1) * rd_e_s  # (rr, ss, 3xyz_es)

    xyz_e_a = xyz_e_s_a * ellipsoid_radius  # (rr, ss, 3xyz_e)
    xyz_e_b = xyz_e_s_b * ellipsoid_radius  # (rr, ss, 3xyz_e)

    ta = ((xyz_e_a - ro_e) * rd_e).sum(dim=-1)  # (rr, ss)
    tb = ((xyz_e_b - ro_e) * rd_e).sum(dim=-1)  # (rr, ss)

    if len(r_shape) > 0 or len(s_shape) > 0:
        ta = ta.reshape(*r_shape, *s_shape)  # (*r, *s)
        tb = tb.reshape(*r_shape, *s_shape)  # (*r, *s)
        intersect_mask = intersect_mask.reshape(*r_shape, *s_shape)  # (*r, *s)
        inside = inside.reshape(*r_shape, *s_shape)  # (*r, *s)
        intersected = intersected.reshape(*r_shape, *s_shape)  # (*r, *s)
    else:
        ta = ta.reshape(-1)  # (,)
        tb = tb.reshape(-1)  # (,)
        intersect_mask = intersect_mask.reshape(-1)  # (,)
        inside = inside.reshape(-1)  # (,)
        intersected = intersected.reshape(-1)  # (,)

    return dict(
        ta=ta,  # (*r, *s)
        tb=tb,  # (*r, *s)
        intersect_mask=intersect_mask,  # (*r, *s)
        inside=inside,  # (*r, *s)
        intersected=intersected,  # (*r, *s)
    )


@linalg_utils.disable_tf32_and_autocast()
def get_ray_ellipsoid_intersection(
    ray_origin: torch.Tensor,  # (*r, 3xyz)
    ray_direction: torch.Tensor,  # (*r, 3xyz)
    ellipsoid_center: torch.Tensor,  # (*r, *s, 3xyz)
    ellipsoid_radius: torch.Tensor,  # (*r, *s, 3xyz)
    ellipsoid_R_e2w: torch.Tensor,  # (*r, *s, 3, 3)
    only_positive_t: bool = True,
    check_finite: bool = False,
) -> T.Dict[str, T.Any]:
    """
    Compute the two intersection points between rays and spheres.

    Args:
        ray_origin:
            (*r, 3)
        ray_direction:
            (*r, 3)
        ellipsoid_center:
            (*r, *s, 3)
        ellipsoid_radius:
            (*r, *s, 3) radius before rotation (ie, in the e coordinate)
        ellipsoid_R_e2w:
            (*r, *s, 3, 3) R_e2w (columns are 3 axes of the ellipsoid in the world coordinate)
        only_positive_t:
            only consider if both ta and tb are positive

    Returns:
        ta:
            (*r_shape, *s_shape), t_near. If not intersected, pad with 0
        tb:
            (*r_shape, *s_shape), t_far.  If not intersected, pad with 0
        intersect_mask:
            (*r_shape, *s_shape) bool, True if ray intersect with sphere

    Note:
        if ray_origin is in sphere, we set it to invalid
    """

    *r_shape, _3 = ray_origin.shape
    *s_shape, _3 = ellipsoid_center.shape[len(r_shape) :]
    rr = math.prod(r_shape)
    ss = math.prod(s_shape)

    ray_origin = ray_origin.reshape(rr, 1, 3)  # (rr, 1, 3)
    ray_direction = ray_direction.reshape(rr, 1, 3)  # (rr, 1, 3)
    ellipsoid_center = ellipsoid_center.reshape(rr, ss, 3)  # (rr, ss, 3)
    ellipsoid_radius = ellipsoid_radius.reshape(rr, ss, 3)  # (rr, ss, 3)
    ellipsoid_R_e2w = ellipsoid_R_e2w.reshape(rr, ss, 3, 3)  # (rr, ss, 3, 3)

    # our strategy is to solve the quatratic equation analytically
    # we first need to transform ro and rd to the ellipsoid coordinate
    # we do not need to handle the scale, which will be handle by the
    # quadratic equation
    ro_e = ray_origin - ellipsoid_center  # (rr, ss, 3)
    ellipsoid_R_w2e = ellipsoid_R_e2w.transpose(-1, -2)  # (rr, ss, 3, 3)
    ro_e = linalg_utils.matmul(
        ellipsoid_R_w2e,  # (rr, ss, 3, 3)
        ro_e.unsqueeze(-1),
    ).squeeze(-1)  # (rr, ss, 3xyz_e)
    rd_e = linalg_utils.matmul(
        ellipsoid_R_w2e,  # (rr, ss, 3, 3)
        ray_direction.unsqueeze(-1),
    ).squeeze(-1)  # (rr, ss, 3xyz_e)

    # solve (rox + t rdx)^2/a^2 + (roy + t rdy)^2/b^2 + (roz + t rdz)^2/c^2 = 1
    # find the min of a, b, c for numerical stable
    with torch.no_grad():
        q = ellipsoid_radius.min(dim=-1, keepdim=True).values  # (rr, ss, 1)
    scaled_ellipsoid_radius = ellipsoid_radius / q  # (rr, ss, 3)

    # rdx/(a/q)
    alpha = ((rd_e / scaled_ellipsoid_radius) ** 2).sum(dim=-1, keepdim=True)  # (rr, ss, 1)
    beta = (rd_e * ro_e / (scaled_ellipsoid_radius**2)).sum(dim=-1, keepdim=True)  # (rr, ss, 1)
    gamma = ((ro_e / scaled_ellipsoid_radius) ** 2).sum(dim=-1, keepdim=True) - q**2  # (rr, ss, 1)

    if check_finite:
        assert alpha.isfinite().all(), f"nan: {alpha.isnan().any()}  inf: {alpha.isinf().any()}"
        assert beta.isfinite().all(), f"nan: {beta.isnan().any()}  inf: {beta.isinf().any()}"
        assert gamma.isfinite().all(), f"nan: {gamma.isnan().any()}  inf: {gamma.isinf().any()}"

    sqrt_in = beta**2 - alpha * gamma  # (rr, ss, 1)

    # # debug
    # print(f'alpha = {alpha}, beta = {beta}, gamma = {gamma}')
    # print(f'sqrt_in = {sqrt_in}')
    # # end deub

    intersect_mask = sqrt_in > 1e-6  # (rr, ss, 1) bool
    sqrt_in[~intersect_mask] = 0

    dt = torch.sqrt(sqrt_in)  # (rr, ss, 1)
    ta = (-beta - dt) / alpha  # (rr, ss, 1)
    tb = (-beta + dt) / alpha  # (rr, ss, 1)

    if check_finite:
        assert sqrt_in.isfinite().all(), f"nan: {sqrt_in.isnan().any()}  inf: {sqrt_in.isinf().any()}"
        assert dt.isfinite().all(), f"{sqrt_in.min()} {sqrt_in.max()}  nan: {dt.isnan().any()}  inf: {dt.isinf().any()}"
        assert ta.isfinite().all(), f"nan: {ta.isnan().any()}  inf: {ta.isinf().any()}"
        assert tb.isfinite().all(), f"nan: {tb.isnan().any()}  inf: {tb.isinf().any()}"

    # we ignore if ta, tb < 0
    if only_positive_t:
        front = torch.logical_and(
            ta > 0,
            tb > 0,
        )
        intersect_mask = torch.logical_and(
            intersect_mask,
            front,
        )  # (rr, ss, 1) bool

    if len(r_shape) > 0 or len(s_shape) > 0:
        ta = ta.reshape(*r_shape, *s_shape)  # (*r, *s)
        tb = tb.reshape(*r_shape, *s_shape)  # (*r, *s)
        intersect_mask = intersect_mask.reshape(*r_shape, *s_shape)  # (*r, *s)
    else:
        ta = ta.reshape(-1)  # (,)
        tb = tb.reshape(-1)  # (,)
        intersect_mask = intersect_mask.reshape(-1)  # (,)

    return dict(
        ta=ta,  # (*r, *s)
        tb=tb,  # (*r, *s)
        intersect_mask=intersect_mask,  # (*r, *s)
    )


def create_ref_normal_patch(
    patch_size: int,
    R_c2w: torch.Tensor = None,
):
    """
    Create a patch_size x patch_size image containing the normal sphere for reference.

    Args:
        patch_size:
        R_c2w:
            (3, 3)

    Returns:
        normal:
            (ph, pw, 3)  normal_c or normal_w if R_c2w is given
        intersect_mask:
            (ph, pw)  bool
    """

    if R_c2w is not None:
        dtype = R_c2w.dtype
        device = R_c2w.device
    else:
        dtype = torch.float
        device = torch.device("cpu")

    ph, pw = patch_size, patch_size
    # create orthographic camera rays  (rd = 0,0,1,  ro=x,y,0)
    xs = torch.linspace(-1, 1, patch_size, dtype=dtype, device=device)
    ys = torch.linspace(1, -1, patch_size, dtype=dtype, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")  # (ph, pw)
    Z = torch.zeros_like(X)  # (ph, pw)
    ro = torch.stack([X, Y, Z], dim=-1)  # (ph, pw, 3_xyzc)
    rd = torch.zeros_like(ro)  # (ph, pw, 3_xyzc)
    rd[..., 2] = 1

    # create sphere (center at 0,0,1,  radius=1)
    # compute ta and tb
    sphere_center_c = torch.zeros_like(ro)  # (ph, pw, 3_xyzc)
    sphere_center_c[..., 2] = 1
    sphere_radius2 = 1  # (rr, ss)

    p_ro = sphere_center_c - ro
    t_proj = (p_ro * rd).sum(dim=-1)  # (ph, pw)
    dr2 = (p_ro**2).sum(dim=-1) - (t_proj**2)  # (ph, pw)
    intersect_mask = dr2 < (sphere_radius2 - 1e-6)  # (ph, pw) bool

    dt = (sphere_radius2 - dr2).clamp(min=1e-10).sqrt()  # (ph, pw)
    ta = t_proj - dt  # (ph, pw)

    # calculate the intersection point
    xyz_c = ro + ta.unsqueeze(-1) * rd  # (ph, pw, 3)
    # calculate normal
    normal_c = torch.nn.functional.normalize(xyz_c - sphere_center_c, dim=-1)  # (ph, pw, 3)
    # no need to check the sign, since all normal should be toward the camera
    normal_c[~intersect_mask] = 0

    if R_c2w is not None:
        normal_c = linalg_utils.matmul(
            R_c2w.reshape(1, 1, 3, 3),
            normal_c.reshape(ph, pw, 3, 1),
        ).squeeze(-1)  # (ph, pw, 3)

    rgb = (normal_c + 1) / 2
    rgb[~intersect_mask] = 0.5

    return dict(
        normal=normal_c,  # (ph, pw, 3)
        intersect_mask=intersect_mask,  # (ph, pw)
        rgb=rgb,
    )


@linalg_utils.disable_tf32_and_autocast()
def get_plucker_representation(
    ray_origin: torch.Tensor,
    ray_direction: torch.Tensor,
):
    """
    Compute the plucker representation of the rays.

    Args:
        ray_origin:
            (*, 3)
        ray_direction:
            (*, 3) normalized

    Returns:
        (*, 6)  (ro x rd, rd)
    """
    roxrd = torch.cross(ray_origin, ray_direction, dim=-1)  # (*, 3)
    plucker = torch.cat([roxrd, ray_direction], dim=-1)  # (*, 6)
    return plucker


# def get_img_feat_with_ray_embedding(
#     model: torch.nn.Module,
#     rgb: torch.Tensor,  # (b, c, h, w)
#     H_c2w: T.Optional[torch.Tensor],  # (b, 4, 4)
#     intrinsic: T.Optional[torch.Tensor],  # (b, 3, 3)
#     resize_scale: float = 1,
#     append_ro_w: bool = False,
#     debug: bool = False,
# ):
#     """
#     Compute patch feature and the plucker ray embedding
#     at the center of the patch.
#
#     Args:
#         model:
#             dino model
#         rgb:
#             (b, c, h, w) already normalized
#         H_c2w:
#             (b, 4, 4) or None.  If both H_c2w and intrinsic are given, plucker ray will be computed.
#         intrinsic:
#             (b, 3, 3) or None
#         resize_scale:
#             to resize the image
#         append_ro_w:
#             whether to append ro_w in ray embedding
#
#     Returns:
#         feature:
#             (b, d, ph, pw)
#         ray_embedding
#             (b, 6, ph, pw) or (b, 9, ph, pw) or None
#     """
#
#     if debug:
#         assert rgb.isfinite().all(), f"{rgb.shape}, nan {rgb.isnan().any()}, inf {rgb.isinf().any()}"
#         assert H_c2w.isfinite().all(), f"{H_c2w.shape}, nan {H_c2w.isnan().any()}, inf {H_c2w.isinf().any()}"
#         assert intrinsic.isfinite().all(), (
#             f"{intrinsic.shape}, nan {intrinsic.isnan().any()}, inf {intrinsic.isinf().any()}"
#         )
#
#     b, c, h, w = rgb.shape
#     patch_size = model.patch_size
#
#     # note that if h and w are not the same or the scaling factor of h and w
#     # are not the same, the image is warpped
#     new_h = math.ceil(resize_scale * h / patch_size) * patch_size
#     new_w = math.ceil(resize_scale * w / patch_size) * patch_size
#
#     # make sure rgb is a multiple of patch_size used by dino
#     if new_h != h or new_w != w:
#         rgb = torch.nn.functional.interpolate(
#             rgb,
#             size=(new_h, new_w),
#             mode="bilinear",
#             align_corners=False,
#         )  # (b, c, new_h, new_w)
#         if intrinsic is not None:
#             intrinsic = intrinsic.clone()  # (b, 3, 3)
#             intrinsic[:, 0, :] = intrinsic[:, 0, :] * (new_w / w)
#             intrinsic[:, 1, :] = intrinsic[:, 1, :] * (new_h / h)
#
#         if debug:
#             assert rgb.isfinite().all(), f"{rgb.shape}, nan {rgb.isnan().any()}, inf {rgb.isinf().any()}"
#             if intrinsic is not None:
#                 assert intrinsic.isfinite().all(), (
#                     f"{intrinsic.shape}, nan {intrinsic.isnan().any()}, inf {intrinsic.isinf().any()}"
#                 )
#
#     # run dino (get patch feature)
#     feature = model(rgb)  # (b, d, hp, wp) or (b, d, seq_len)
#     _b, d, hp, wp = feature.shape
#
#     if debug:
#         assert feature.isfinite().all(), f"{feature.shape}, nan {feature.isnan().any()}, inf {feature.isinf().any()}"
#
#     # get the uv of the patch centers
#     us = torch.arange(wp, dtype=feature.dtype, device=feature.device)  # (wp,)
#     vs = torch.arange(hp, dtype=feature.dtype, device=feature.device)  # (hp,)
#     us = us * patch_size + 0.5 * patch_size  # this accounts for pixel center at x.5
#     vs = vs * patch_size + 0.5 * patch_size  # this accounts for pixel center at x.5
#     U, V = torch.meshgrid(us, vs, indexing="xy")  # (hp, wp)
#     uv = torch.stack([U, V], dim=-1)  # (hp, wp, 2uv)
#
#     if debug:
#         assert uv.isfinite().all(), f"{uv.shape}, nan {uv.isnan().any()}, inf {uv.isinf().any()}"
#
#     # compute ray direction and ray origin of the patch centers
#     if H_c2w is not None and intrinsic is not None:
#         ro_w, rd_w = generate_camera_rays_from_uv(
#             cam_poses=H_c2w,  # (b, 4, 4)
#             intrinsics=intrinsic,  # (b, 3, 3)
#             uv=uv.expand(b, hp, wp, 2),  # (b, hp, wp, 2uv)
#             use_quick_inv_intrinsic=True,
#             device=feature.device,
#         )  # (b, hp, wp, 3xyz)
#         if debug:
#             assert ro_w.isfinite().all(), f"{ro_w.shape}, nan {ro_w.isnan().any()}, inf {ro_w.isinf().any()}"
#             assert rd_w.isfinite().all(), f"{rd_w.shape}, nan {rd_w.isnan().any()}, inf {rd_w.isinf().any()}"
#
#         # compute ray embedding
#         plucker = get_plucker_representation(
#             ray_origin=ro_w,  # (b, hp, wp, 3xyz)
#             ray_direction=rd_w,  # (b, hp, wp, 3xyz)
#         )  # (b, hp, wp, 6)
#
#         if debug:
#             assert plucker.isfinite().all(), (
#                 f"{plucker.shape}, nan {plucker.isnan().any()}, inf {plucker.isinf().any()}"
#             )
#
#         if append_ro_w:
#             plucker = torch.cat([plucker, ro_w], dim=-1)  # (b, hp, wp, 9)
#
#         ray_embedding = plucker.permute(0, 3, 1, 2)  # (b, 6/9, hp, wp)
#         ro_w = ro_w.permute(0, 3, 1, 2)  # (b, 3xyz, hp, wp)
#         rd_w = rd_w.permute(0, 3, 1, 2)  # (b, 3xyz, hp, wp)
#     else:
#         ro_w = None
#         rd_w = None
#         ray_embedding = None
#
#     return dict(
#         feature=feature,  # (b, d, hp, wp)
#         ray_embedding=ray_embedding,  # (b, 6/9, hp, wp) or None
#         ro_w=ro_w,  # (b, 3xyz, hp, wp) or None
#         rd_w=rd_w,  # (b, 3xyz, hp, wp) or None
#         uv=uv,  # (hp, wp, 2uv)  [0, w] [0, h]
#     )


def get_img_feat_with_ray_embedding(
    model: torch.nn.Module,
    rgb: torch.Tensor,  # (b, c, h, w)
    H_c2w: T.Optional[torch.Tensor],  # (b, 4, 4)
    intrinsic: T.Optional[torch.Tensor],  # (b, 3, 3)
    alpha: T.Optional[torch.Tensor] = None,
    resize_scale: float = 1,
    append_ro_w: bool = False,
    debug: bool = False,
):
    """
    Compute patch feature and the plucker ray embedding
    at the center of the patch.

    Args:
        model:
            dino model
        rgb:
            (b, c, h, w) already normalized
        H_c2w:
            (b, 4, 4) or None.  If both H_c2w and intrinsic are given, plucker ray will be computed.
        intrinsic:
            (b, 3, 3) or None
        resize_scale:
            to resize the image
        append_ro_w:
            whether to append ro_w in ray embedding

    Returns:
        feature:
            (b, d, ph, pw)
        ray_embedding
            (b, 6, ph, pw) or (b, 9, ph, pw) or None
    """

    if debug:
        assert rgb.isfinite().all(), f"{rgb.shape}, nan {rgb.isnan().any()}, inf {rgb.isinf().any()}"
        assert H_c2w.isfinite().all(), f"{H_c2w.shape}, nan {H_c2w.isnan().any()}, inf {H_c2w.isinf().any()}"
        assert intrinsic.isfinite().all(), (
            f"{intrinsic.shape}, nan {intrinsic.isnan().any()}, inf {intrinsic.isinf().any()}"
        )

    b, c, h, w = rgb.shape
    patch_size = model.patch_size

    # note that if h and w are not the same or the scaling factor of h and w
    # are not the same, the image is warpped
    new_h = math.ceil(resize_scale * h / patch_size) * patch_size
    new_w = math.ceil(resize_scale * w / patch_size) * patch_size

    # make sure rgb is a multiple of patch_size used by dino
    if new_h != h or new_w != w:
        rgb = torch.nn.functional.interpolate(
            rgb,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )  # (b, c, new_h, new_w)
        if intrinsic is not None:
            intrinsic = intrinsic.clone()  # (b, 3, 3)
            intrinsic[:, 0, :] = intrinsic[:, 0, :] * (new_w / w)
            intrinsic[:, 1, :] = intrinsic[:, 1, :] * (new_h / h)

        if debug:
            assert rgb.isfinite().all(), f"{rgb.shape}, nan {rgb.isnan().any()}, inf {rgb.isinf().any()}"
            if intrinsic is not None:
                assert intrinsic.isfinite().all(), (
                    f"{intrinsic.shape}, nan {intrinsic.isnan().any()}, inf {intrinsic.isinf().any()}"
                )

    # run dino (get patch feature)
    with _nullcontext():
        feature = model(x=rgb, hit=alpha)  # (b, d, hp, wp)
    if debug:
        assert feature.isfinite().all(), f"{feature.shape}, nan {feature.isnan().any()}, inf {feature.isinf().any()}"

    if getattr(model, "output_flattened", False):
        # the feature has already been flattened
        feature_flattened = True

        ro_w = None
        rd_w = None
        ray_embedding = None

        uv = None
    else:
        feature_flattened = False

        _b, d, hp, wp = feature.shape

        # get the uv of the patch centers
        us = torch.arange(wp, dtype=feature.dtype, device=feature.device)  # (wp,)
        vs = torch.arange(hp, dtype=feature.dtype, device=feature.device)  # (hp,)
        us = us * patch_size + 0.5 * patch_size  # this accounts for pixel center at x.5
        vs = vs * patch_size + 0.5 * patch_size  # this accounts for pixel center at x.5
        U, V = torch.meshgrid(us, vs, indexing="xy")  # (hp, wp)
        uv = torch.stack([U, V], dim=-1)  # (hp, wp, 2uv)

        if debug:
            assert uv.isfinite().all(), f"{uv.shape}, nan {uv.isnan().any()}, inf {uv.isinf().any()}"

        # compute ray direction and ray origin of the patch centers
        if H_c2w is not None and intrinsic is not None:
            ro_w, rd_w = generate_camera_rays_from_uv(
                cam_poses=H_c2w,  # (b, 4, 4)
                intrinsics=intrinsic,  # (b, 3, 3)
                uv=uv.expand(b, hp, wp, 2),  # (b, hp, wp, 2uv)
                use_quick_inv_intrinsic=True,
                device=feature.device,
            )  # (b, hp, wp, 3xyz)
            if debug:
                assert ro_w.isfinite().all(), f"{ro_w.shape}, nan {ro_w.isnan().any()}, inf {ro_w.isinf().any()}"
                assert rd_w.isfinite().all(), f"{rd_w.shape}, nan {rd_w.isnan().any()}, inf {rd_w.isinf().any()}"

            # compute ray embedding
            plucker = get_plucker_representation(
                ray_origin=ro_w,  # (b, hp, wp, 3xyz)
                ray_direction=rd_w,  # (b, hp, wp, 3xyz)
            )  # (b, hp, wp, 6)

            if debug:
                assert plucker.isfinite().all(), (
                    f"{plucker.shape}, nan {plucker.isnan().any()}, inf {plucker.isinf().any()}"
                )

            if append_ro_w:
                plucker = torch.cat([plucker, ro_w], dim=-1)  # (b, hp, wp, 9)

            ray_embedding = plucker.permute(0, 3, 1, 2)  # (b, 6/9, hp, wp)
            ro_w = ro_w.permute(0, 3, 1, 2)  # (b, 3xyz, hp, wp)
            rd_w = rd_w.permute(0, 3, 1, 2)  # (b, 3xyz, hp, wp)
        else:
            ro_w = None
            rd_w = None
            ray_embedding = None

    return dict(
        feature_flattened=feature_flattened,  # bool
        feature=feature,  # (b, d, hp, wp) or (b, d, seq_len)
        ray_embedding=ray_embedding,  # (b, 6/9, hp, wp) or None
        ro_w=ro_w,  # (b, 3xyz, hp, wp) or None
        rd_w=rd_w,  # (b, 3xyz, hp, wp) or None
        uv=uv,  # (hp, wp, 2uv)  [0, w] [0, h]
    )


def get_img_feat_plus_with_ray_embedding(
    model: torch.nn.Module,
    rgb: torch.Tensor,  # (b, c, h, w)
    H_c2w: torch.Tensor,  # (b, 4, 4)
    intrinsic: torch.Tensor,  # (b, 3, 3)
    input_types: T.List[str],
    z_c: torch.Tensor,  # (b, h, w)
    hit_map: T.Optional[torch.Tensor] = None,  # (b, h, w) bool
    append_ro_w: bool = False,
    debug: bool = False,
):
    """
    Compute patch feature and the plucker ray embedding
    at the center of the patch.
    The difference between the function and `get_img_feat_plus_with_ray_embedding`
    is that the model is assumed to take dense map of xyz_w and/or plucker
    as input as well.

    Args:
        model:
            dino model
        rgb:
            (b, c, h, w) already normalized
        H_c2w:
            (b, 4, 4)
        intrinsic:
            (b, 3, 3)
        input_types:
            list of str, 'rgb', 'xyz_w', 'plucker', 'hit'
        z_c:
            (b, h, w)
        hit_map
            (b, h, w) bool
        resize_scale:
            to resize the image
        append_ro_w:
            whether to append ro_w in ray embedding

    Returns:
        feature:
            (b, d, ph, pw)
        ray_embedding
            (b, 6, ph, pw) or (b, 9, ph, pw) or None
    """

    if debug:
        assert rgb.isfinite().all(), f"{rgb.shape}, nan {rgb.isnan().any()}, inf {rgb.isinf().any()}"
        assert H_c2w.isfinite().all(), f"{H_c2w.shape}, nan {H_c2w.isnan().any()}, inf {H_c2w.isinf().any()}"
        assert intrinsic.isfinite().all(), (
            f"{intrinsic.shape}, nan {intrinsic.isnan().any()}, inf {intrinsic.isinf().any()}"
        )

    b, c, h, w = rgb.shape
    patch_size = model.patch_size

    # compute dense geometry maps (xyz_w, plucker)
    xyz_w = compute_3d_xyz(
        z_map=z_c,  # (b, h, w)
        intrinsic=intrinsic,  # (b, 3, 3)
        H_c2w=H_c2w,
    )["xyz_w"]  # (b, h, w, 3xyz_w)

    # set xyz_w to 0 if not hit
    if hit_map is not None:
        xyz_w[~hit_map] = 0

    # get the uv of the patch centers
    us = torch.arange(w, dtype=xyz_w.dtype, device=xyz_w.device)  # (w,)
    vs = torch.arange(h, dtype=xyz_w.dtype, device=xyz_w.device)  # (h,)
    us = us + 0.5  # this accounts for pixel center at x.5
    vs = vs + 0.5  # this accounts for pixel center at x.5
    U, V = torch.meshgrid(us, vs, indexing="xy")  # (h, w)
    uv = torch.stack([U, V], dim=-1)  # (h, w, 2uv)

    if debug:
        assert uv.isfinite().all(), f"{uv.shape}, nan {uv.isnan().any()}, inf {uv.isinf().any()}"

    # compute ray direction and ray origin of the patch centers
    ro_w, rd_w = generate_camera_rays_from_uv(
        cam_poses=H_c2w,  # (b, 4, 4)
        intrinsics=intrinsic,  # (b, 3, 3)
        uv=uv.expand(b, h, w, 2),  # (b, h, w, 2uv)
        use_quick_inv_intrinsic=True,
        device=xyz_w.device,
    )  # (b, h, w, 3xyz)
    if debug:
        assert ro_w.isfinite().all(), f"{ro_w.shape}, nan {ro_w.isnan().any()}, inf {ro_w.isinf().any()}"
        assert rd_w.isfinite().all(), f"{rd_w.shape}, nan {rd_w.isnan().any()}, inf {rd_w.isinf().any()}"

    # compute ray embedding
    plucker = get_plucker_representation(
        ray_origin=ro_w,  # (b, h, w, 3xyz)
        ray_direction=rd_w,  # (b, h, w, 3xyz)
    )  # (b, h, w, 6)

    if debug:
        assert plucker.isfinite().all(), f"{plucker.shape}, nan {plucker.isnan().any()}, inf {plucker.isinf().any()}"

    # run dino (get patch feature)
    feature = model(
        x=rgb if "rgb" in input_types else None,  # (b, c, h, w)
        xyz_w=xyz_w.permute(0, 3, 1, 2) if "xyz_w" in input_types else None,  # (b, 3, h, w)
        plucker=plucker.permute(0, 3, 1, 2) if "plucker" in input_types else None,  # (b, 6, h, w)
        hit=hit_map.unsqueeze(1).to(dtype=rgb.dtype)
        if hit_map is not None and "hit" in input_types
        else None,  # (b, 1, h, w)
    )  # (b, d, hp, wp)
    _b, d, hp, wp = feature.shape

    if debug:
        assert feature.isfinite().all(), f"{feature.shape}, nan {feature.isnan().any()}, inf {feature.isinf().any()}"

    # get plucker embedding for patch center
    us = torch.arange(wp, dtype=feature.dtype, device=feature.device)  # (wp,)
    vs = torch.arange(hp, dtype=feature.dtype, device=feature.device)  # (hp,)
    us = us * patch_size + 0.5 * patch_size  # this accounts for pixel center at x.5
    vs = vs * patch_size + 0.5 * patch_size  # this accounts for pixel center at x.5
    U, V = torch.meshgrid(us, vs, indexing="xy")  # (hp, wp)
    uv = torch.stack([U, V], dim=-1)  # (hp, wp, 2uv)

    if debug:
        assert uv.isfinite().all(), f"{uv.shape}, nan {uv.isnan().any()}, inf {uv.isinf().any()}"

    # compute ray direction and ray origin of the patch centers
    ro_w, rd_w = generate_camera_rays_from_uv(
        cam_poses=H_c2w,  # (b, 4, 4)
        intrinsics=intrinsic,  # (b, 3, 3)
        uv=uv.expand(b, hp, wp, 2),  # (b, hp, wp, 2uv)
        use_quick_inv_intrinsic=True,
        device=feature.device,
    )  # (b, hp, wp, 3xyz)
    if debug:
        assert ro_w.isfinite().all(), f"{ro_w.shape}, nan {ro_w.isnan().any()}, inf {ro_w.isinf().any()}"
        assert rd_w.isfinite().all(), f"{rd_w.shape}, nan {rd_w.isnan().any()}, inf {rd_w.isinf().any()}"

    # compute ray embedding
    plucker = get_plucker_representation(
        ray_origin=ro_w,  # (b, hp, wp, 3xyz)
        ray_direction=rd_w,  # (b, hp, wp, 3xyz)
    )  # (b, hp, wp, 6)

    if debug:
        assert plucker.isfinite().all(), f"{plucker.shape}, nan {plucker.isnan().any()}, inf {plucker.isinf().any()}"

    if append_ro_w:
        plucker = torch.cat([plucker, ro_w], dim=-1)  # (b, hp, wp, 9)

    ray_embedding = plucker.permute(0, 3, 1, 2)  # (b, 6/9, hp, wp)
    ro_w = ro_w.permute(0, 3, 1, 2)  # (b, 3xyz, hp, wp)
    rd_w = rd_w.permute(0, 3, 1, 2)  # (b, 3xyz, hp, wp)

    return dict(
        feature=feature,  # (b, d, hp, wp)
        ray_embedding=ray_embedding,  # (b, 6/9, hp, wp) or None
        ro_w=ro_w,  # (b, 3xyz, hp, wp) or None
        rd_w=rd_w,  # (b, 3xyz, hp, wp) or None
        uv=uv,  # (hp, wp, 2uv)  [0, w] [0, h]
    )


def align_normal_with_ref_point(
    xyz_w: torch.Tensor,
    normal_w: torch.Tensor,
    ref_xyz_w: torch.Tensor,
    second_ref_xyz_w: T.Optional[torch.Tensor] = None,
    opposite: bool = False,
):
    """
    Align the vertex normal to point towards the reference point.

    Args:
        xyz_w:
            (*, 3)
        normal_w:
            (*, 3)
        ref_xyz_w:
            (3,) or (*, 3)
        second_ref_xyz_w:
            (3,) or (*, 3)  After aligning toward the first, align toward the second point.
            This is to align the normal_w that are orthogonal to the first reference point.
        opposite:
            point in the opposite direction of ref_xyz_w - xyz_w

    Returns:
        aligned_normal_w:
            (*, 3)
    """

    *b_shape, _3xyz = xyz_w.shape
    normal_w = normal_w.clone()

    # align using the first reference point
    n_dot_d = (normal_w * torch.nn.functional.normalize(ref_xyz_w - xyz_w, dim=-1)).sum(dim=-1, keepdim=True)  # (*b, 1)
    if not opposite:
        mask = n_dot_d < -1.0e-4
    else:
        mask = n_dot_d > 1.0e-4
    normal_w[mask.expand_as(normal_w)] *= -1

    if second_ref_xyz_w is not None:
        # align using the first reference point
        n_dot_d = (normal_w * torch.nn.functional.normalize(second_ref_xyz_w - xyz_w, dim=-1)).sum(
            dim=-1, keepdim=True
        )  # (*b, 1)
        if not opposite:
            mask = n_dot_d < -1.0e-4
        else:
            mask = n_dot_d > 1.0e-4
        normal_w[mask.expand_as(normal_w)] *= -1

    return normal_w


def list_of_dicts_to_dict_of_lists(lst):
    """
    x = [
        {'foo': 3, 'bar': 1},
        {'foo': 4, 'bar': 2},
        {'foo': 5, 'bar': 3},
    ]
    ppp.list_of_dicts__to__dict_of_lists(x)
    # Output:
    # {'foo': [3, 4, 5], 'bar': [1, 2, 3]}
    """
    assert isinstance(lst, (list, tuple)), type(lst)
    if len(lst) == 0:
        return {}
    keys = lst[0].keys()
    output_dict = dict()
    for d in lst:
        assert set(d.keys()) == set(keys), (d.keys(), keys)
        for k in keys:
            if k not in output_dict:
                output_dict[k] = []
            output_dict[k].append(d[k])
    return output_dict


def gather_patch_feature(
    xyz_w: torch.Tensor,  # (b, n, 3xyz_w)
    feature_map: torch.Tensor,  # (b, q, h/s, w/s, d)
    H_c2w: torch.Tensor,  # (b, q, 4, 4)
    intrinsic: torch.Tensor,  # (b, q, 3, 3)
    width_px: int,
    height_px: int,
    patch_width_px: int = 14,
    patch_height_px: int = 14,
    neighbor_size: int = 1,
    mode: str = "bilinear",
    uv_only: bool = False,
) -> T.Dict[str, torch.Tensor]:
    """
    Project each point to image, calculate the patch idxs near the projected points,
    and gather the patch features.

    Note that the function uses a lot of memory.
    For example, when b = 8; n = 16384; d = 512; q = 100; ph = 32; pw = 32; mh = 1; mw = 1,
    the output itself uses 27 GB, making it difficult to learn the aggregation.

    Args:
        H_c2w:
            (b, q, 4, 4) camera pose of the images (camera coordinate to world coordinate).
        intrinsic:
            (b, q, 3, 3) camera intrinsic of the images
        width_px:
            int, number of pixel in width of the image
        height_px:
            int, number of pixel in height of the image
        patch_width_px:
            int, number of pixel in width of the patch.
            The patch starts from pixel (0, 0)
        patch_height_px:
            int, number of pixel in height of the patch.
        neighbor_size:
            int, number of patches per side to gather.
        feature_map:
            (b, q, ph=h/patch_height_px, pw=w/patch_width_px, d)  patch feature map
        mode:
            'nearest' or 'bilinear'
        uv_only:
            whether to compute uv_patch without doing the interpolation.

    Returns:
        uv_patch:
            (b, q, n, mh, mw, 2uv_patch) [0, w/ph] [0, h/ph]
            note that patch center are at x.5
        gathered_feat:
            (b, q, n, mh, mw, d)
            out of bound or any that touches the boundary would be all zero
        valid_mask:
            (b, q, n, mh, mw) bool
    """
    b, q, _41, _42 = H_c2w.shape
    _b, _q, ph, pw, d = feature_map.shape
    mw, mh = neighbor_size, neighbor_size
    n = xyz_w.size(1)

    uv_img = pinhole_projection(
        xyz_w=xyz_w,  # (b, n, 3xyz_w)
        intrinsics=intrinsic,  # (b, q, 3, 3)
        H_c2w=H_c2w,  # (b, q, 4, 4)
        dim_b=1,
    )[0]  # uv_img: (b, q, n, 2uv_img) [0, w/h] can oob,  xyz_c: (b, q, n, 3xyz_c)

    # we assume the image and feature map's origin is at top left corner.
    uv_patch_center = uv_img / torch.tensor(
        [
            patch_width_px,
            patch_height_px,
        ],
        dtype=uv_img.dtype,
        device=uv_img.device,
    )  # (b, q, n, 2uv_patch) [0, w//pw]

    local_neighbor = torch.arange(neighbor_size, dtype=uv_patch_center.dtype, device=uv_patch_center.device) - (
        (neighbor_size - 1) / 2
    )  # (mh,)
    local_neighbor_u, local_neighbor_v = torch.meshgrid(
        local_neighbor, local_neighbor, indexing="xy"
    )  # (mh, mw),  # (mh, mw)
    local_neighbor = torch.stack([local_neighbor_u, local_neighbor_v], dim=-1)  # (mh, mw, 2uv_patch)

    uv_patch = uv_patch_center.unsqueeze(-2).unsqueeze(-2) + local_neighbor  # (b, q, n, mh, mw, 2uv_patch)

    # we are being reserved (we consider any that will interpolate with boundary as oob)
    valid_mask = torch.logical_and(
        torch.logical_and(
            uv_patch[..., 0] >= 0.5,
            uv_patch[..., 0] < pw - 0.5,
        ),
        torch.logical_and(
            uv_patch[..., 1] >= 0.5,
            uv_patch[..., 1] < ph - 0.5,
        ),
    )  # (b, q, n, mh, mw)  bool

    if not uv_only:
        gathered_feat = uv_sampling(
            uv=uv_patch.reshape(b * q, n, mh, mw, 2),  # (bq, n, mh, mw, 2uv_patch)
            feature_map=feature_map.reshape(b * q, ph, pw, d),  # (bq, h/ph, w/pw, d)
            mode=mode,
            padding_mode="zeros",
            uv_normalized=False,
        ).reshape(b, q, n, mh, mw, d)  # (b, q, n, mh, mw, d)
        gathered_feat[~valid_mask] = 0
    else:
        gathered_feat = None

    return dict(
        uv_patch=uv_patch,  # (b, q, n, mh, mw, 2uv_patch)
        valid_mask=valid_mask,  # (b, q, n, mh, mw)  bool
        gathered_feat=gathered_feat,  # (b, q, n, mh, mw, d) or None
    )


def forward_occ_filtered_bilinear_avg_v2(
    xyz_w: torch.Tensor,  # (b, n, 3xyz_w)
    feature_map: torch.Tensor,  # (b, q, h/patch_height_px, w/patch_width_px, d)
    z_c: torch.Tensor,  # (b, q, h, w)
    H_c2w: torch.Tensor,  # (b, q, 4, 4)
    intrinsic: torch.Tensor,  # (b, q, 3, 3)
    hit_map: T.Optional[torch.Tensor] = None,  # (b, q, h, w)
    uv_only: bool = False,
    th_z_c: float = 0,
    downsample_hit_map=False,
    patch_width_px: int = 14,
    patch_height_px: int = 14,
    return_without_averaging=True,
) -> T.Dict[str, torch.Tensor]:
    """
    bilinear interpolate the feature map and average across views that sees the point
    (within some distance threshold).

    Note that depending on the hyper-parameters (b, q, region_size), it may use A LOT OF
    memory just to store the output of the grid_sampling.

    Args:
        xyz_w:
            (b, n, 3xyz_w)  the points
        feature_map:
            (b, q, h', w', d)
        z_c:
            (b, q, h, w) the depth map (z in the camera coordinate)
        hit_map:
            (b, q, h, w) bool, whether a pixel sees anything
        H_c2w:
            (b, q, 4, 4) camera poses of the original image
        intrinsic:
            (b, q, 3, 3) camera intrinsic of the original image (not the feature map)
        width_px:
            number of pixels in width of the original image
        height_px:
            number of pixels in height of the original image

    Returns:
        feature:
            (b, n, d)
        uv_patch
            (b, n, q, 2uv_patch)  [0, pw] [0, ph]
    """
    assert z_c is not None
    b, n, _3 = xyz_w.shape
    _b, q, h, w = z_c.shape
    _b, _q, ph, pw, d = feature_map.shape

    # project each point onto images
    uv_img, xyz_c = pinhole_projection(
        xyz_w=xyz_w,  # (b, n, 3xyz_w)
        intrinsics=intrinsic,  # (b, q, 3, 3)
        H_c2w=H_c2w,  # (b, q, 4, 4)
        dim_b=1,
    )  # uv_img: (b, q, n, 2uv_img) [0, w/h] can oob,  xyz_c: (b, q, n, 3xyz_c)

    # interpolate to get the img_xyz_c at the projected point
    project_z_c = uv_sampling(
        uv=uv_img.reshape(b * q, n, 2),  # (bq, n, 2uv_img)
        feature_map=z_c.reshape(b * q, h, w, 1),  # (bq, h, w, 1)
        mode="bilinear",
        padding_mode="zeros",
        uv_normalized=False,
    ).reshape(b, q, n)  # (b, q, n)

    valid_mask = project_z_c >= (xyz_c[..., 2] - th_z_c)  # (b, q, n) bool

    if hit_map is not None:
        if downsample_hit_map:
            hit_map = torch.nn.functional.interpolate(
                input=hit_map.reshape(b * q, 1, h, w).float(),
                size=(ph, pw),
                mode="nearest-exact",
            )  # (bq, 1, ph, pw)
            hit_map = (
                torch.nn.functional.interpolate(
                    input=hit_map.float(),  # (bq, 1, ph, pw)
                    size=(h, w),
                    mode="nearest-exact",
                )
                > 0.5
            )  # (bq, 1, h, w)
            hit_map = hit_map.reshape(b, q, h, w)

        project_hit = uv_sampling(
            uv=uv_img.reshape(b * q, n, 2),  # (bq, n, 2uv_img)
            feature_map=hit_map.to(dtype=uv_img.dtype).reshape(b * q, h, w, 1),  # (bq, h, w, 1)
            mode="bilinear",  # using nearest intentionally (since we are to threshold below, equivalent?)
            padding_mode="zeros",
            uv_normalized=False,
        ).reshape(b, q, n)  # (b, q, n)

        valid_mask = torch.logical_and(
            valid_mask,
            project_hit > 0.5,
        )  # (b, q, n)

    # we assume the image and feature map's origin is at top left corner.
    uv_patch = uv_img / torch.tensor(
        [
            patch_width_px,
            patch_height_px,
        ],
        dtype=uv_img.dtype,
        device=uv_img.device,
    )  # (b, q, n, 2uv_patch) [0, w//pw]

    valid_mask = torch.logical_and(
        valid_mask,  # (b, q, n)
        torch.logical_and(
            uv_patch[..., 0] >= 0.5,
            uv_patch[..., 0] < pw - 0.5,
        ),  # (b, q, n)
    )  # (b, q, n)

    valid_mask = torch.logical_and(
        valid_mask,  # (b, q, n)
        torch.logical_and(
            uv_patch[..., 1] >= 0.5,
            uv_patch[..., 1] < ph - 0.5,
        ),
    )  # (b, q, n)  bool

    if not uv_only:
        # potentially we can improve this by using sparse_uv_sampling.
        # however, the number valid images may not be the same for each pixel
        gathered_feat = uv_sampling(
            uv=uv_patch.reshape(b * q, n, 2),  # (bq, n, 2uv_patch)
            feature_map=feature_map.reshape(b * q, ph, pw, d),
            mode="bilinear",
            padding_mode="zeros",
            uv_normalized=False,
        ).reshape(b, q, n, d)  # (b, q, n, d)
        if return_without_averaging:
            gathered_feat_nonaveraged = gathered_feat
        gathered_feat[~valid_mask] = 0

        # aggregate
        gathered_feat = gathered_feat.sum(dim=1) / torch.clamp(valid_mask.sum(dim=1).unsqueeze(-1), min=1)  # (b, n, d)
    else:
        gathered_feat = None

    return dict(
        feature=gathered_feat.reshape(b, n, d) if gathered_feat is not None else None,  # (b, n, d)
        uv_patch=uv_patch.permute(0, 2, 1, 3),  # (b, n, q, 2)
        valid_mask=valid_mask.permute(0, 2, 1),  # (b, n, q)
        point_valid_mask=valid_mask.any(dim=1),  # (b, n)
        gathered_feat_nonaveraged=gathered_feat_nonaveraged if return_without_averaging else None,
    )


def select_topk_views(
    k: int,
    xyz_w: torch.Tensor,  # (b, n, 3xyz_w)
    z_c_map: torch.Tensor,  # (b, q, h, w)
    H_c2w: torch.Tensor,  # (b, q, 4, 4)
    intrinsic: torch.Tensor,  # (b, q, 3, 3)
    hit_map: torch.Tensor,  # (b, q, h, w)
    xyz_w_map: T.Optional[torch.Tensor] = None,  # (b, q, h, w, 3xyz_w)
    normal_w_map: T.Optional[torch.Tensor] = None,  # (b, q, h, w)
    normal_w: T.Optional[torch.Tensor] = None,  # (b, n, 3xyz_w)
    th_z_c: float = 1e-3,  # controls valid_mask, set to None to turn off
    th_xyz_norm: float = 1e-3,  # controls valid_mask, set to None to turn off
    th_normal: float = None,  # controls valid_mask, set to None to turn off
    th_point_normal: float = 0.25,  # controls valid_mask, set to None to turn off
    std_xyz_norm: float = 0.1,  # controls prob
    std_dist: float = 0.5,  # controls prob
    cos_softmax_scale: float = 5.0,  # controls prob
    point_cos_softmax_scale: float = 5.0,  # controls prob
    prob_use_dist: bool = True,
    prob_use_hit: bool = True,
    prob_use_z_c: bool = True,
    prob_use_xyz_norm: bool = True,
    prob_use_normal: bool = False,
    prob_use_point_normal: bool = False,
    deterministic: bool = False,
) -> T.Dict[str, torch.Tensor]:
    """
    Choose from one of the views that sees the point using
    distance between pinhole and the point, visibility,
    normal between ray and surface.

    Args:
        k:
            number of views to select
        xyz_w:
            (b, n, 3xyz_w)  the points
        xyz_w_map:
            (b, q, h, w, 3xyz_w)
        z_c_map:
            (b, q, h, w) the depth map (z in the camera coordinate)
        hit_map:
            (b, q, h, w) bool, whether a pixel sees anything
        normal_w_map:
            (b, q, h, w, 3xyz_w) rendered normal maps
        normal_w:
            (b, n, 3) point normal
        H_c2w:
            (b, q, 4, 4) camera poses of the original image
        intrinsic:
            (b, q, 3, 3) camera intrinsic of the original image (not the feature map)
        deterministic:
            whether to directly select the top-k prob_q

    Returns:
        qidxs:
            (b, n, k) long
        projected_uv_img:
            (b, n, k, 2uv_img) [0, w] [0, h]
        valid_qidxs:
            (b, n, k) bool.  It can be a point does not have any valid view

        prob_dist:
            (b, q, n) or 1 (float)
        valid_mask:
            (b, q, n) bool
        prob_xyz_w:
            (b, q, n) or 1 (float)
        prob_normal_w:
            (b, q, n) or 1 (float)
        prob_point_normal_w:
            (b, q, n) or 1 (float)
        prob_q:
            (b, q, n)
    """
    assert z_c_map is not None
    b, n, _3 = xyz_w.shape
    _b, q, h, w = z_c_map.shape

    # make sure we dont do this in half or bfloat16
    xyz_w = xyz_w.float()
    if z_c_map is not None:
        z_c_map = z_c_map.float()
    if H_c2w is not None:
        H_c2w = H_c2w.float()
    if intrinsic is not None:
        intrinsic = intrinsic.float()
    if xyz_w_map is not None:
        xyz_w_map = xyz_w_map.float()
    if normal_w_map is not None:
        normal_w_map = normal_w_map.float()
    if normal_w is not None:
        normal_w = normal_w.float()

    # we prefer closer images
    delta_xyz_w = xyz_w.unsqueeze(1) - H_c2w[:, :, :3, 3].unsqueeze(2)  # (b, q, n, 3)
    dist = torch.linalg.vector_norm(delta_xyz_w, dim=-1)  # (b, q, n)
    logit_dist = -dist / (2 * std_dist**2)  # (b, q, n)
    prob_dist = torch.nn.functional.softmax(logit_dist, dim=1)  # (b, q, n)

    # project each point onto images
    uv_img, xyz_c = pinhole_projection(
        xyz_w=xyz_w,  # (b, n, 3xyz_w)
        intrinsics=intrinsic,  # (b, q, 3, 3)
        H_c2w=H_c2w,  # (b, q, 4, 4)
        dim_b=1,
    )  # uv_img: (b, q, n, 2uv_img) [0, w/h] can oob,  xyz_c: (b, q, n, 3xyz_c)

    uv_img_normalized = uv_img.clone()  # (b, q, n, 2uv_img) [0, 1]
    uv_img_normalized[..., 0] = uv_img_normalized[..., 0] / w
    uv_img_normalized[..., 1] = uv_img_normalized[..., 1] / h

    # we first use depth (z_c) to filter out images that actually sees the point
    # interpolate to get the z_c at the projected point
    # when using bilinear interp, we want to mask nonhit (so it does not go to inf)
    assert hit_map is not None
    z_c_map = z_c_map * hit_map.to(dtype=z_c_map.dtype)  # (b, q, h, w)
    project_z_c = uv_sampling(
        uv=uv_img_normalized.reshape(b * q, n, 2),  # (bq, n, 2uv_img)
        feature_map=z_c_map.reshape(b * q, h, w, 1),  # (bq, h, w, 1)
        mode="bilinear",
        padding_mode="zeros",
        uv_normalized=True,
    ).reshape(b, q, n)  # (b, q, n)

    # valid if the pixel on image sees further than the point
    # no need to check ray_t as it is same pinhole and pixel so z_c is enough.
    # this does not filter out background (whose prject_z_c is at inf)
    if th_z_c is not None:
        valid_mask_z_c = project_z_c >= (xyz_c[..., 2] - th_z_c)  # (b, q, n) bool
        valid_mask = valid_mask_z_c  # (b, q, n)
        # when doing bilinear interp, 1e-3 seems to be a good value (for bunny)
    else:
        valid_mask_z_c = None
        valid_mask = torch.ones(b, q, n, dtype=torch.bool, device=xyz_w.device)  # (b, q, n)

    # hit
    project_hit = uv_sampling(
        uv=uv_img_normalized.reshape(b * q, n, 2),  # (bq, n, 2uv_img)
        feature_map=hit_map.to(dtype=uv_img.dtype).reshape(b * q, h, w, 1),  # (bq, h, w, 1)
        mode="bilinear",  # using nearest intentionally (to match above)
        padding_mode="zeros",
        uv_normalized=True,
    ).reshape(b, q, n)  # (b, q, n)
    valid_mask_hit = project_hit >= 0.5
    valid_mask = torch.logical_and(valid_mask, valid_mask_hit)  # (b, q, n)

    if th_xyz_norm is not None or prob_use_xyz_norm:
        # when using bilinear interp, we want to mask xyz_w_map (so it does not go to inf)
        assert xyz_w_map is not None
        assert hit_map is not None
        xyz_w_map = xyz_w_map * hit_map.unsqueeze(-1).to(dtype=xyz_w_map.dtype)  # (b, q, h, w, 3)
        project_xyz_w = uv_sampling(
            uv=uv_img_normalized.reshape(b * q, n, 2),  # (bq, n, 2uv_img)
            feature_map=xyz_w_map.to(dtype=uv_img.dtype).reshape(b * q, h, w, 3),  # (bq, h, w, 3)
            mode="bilinear",  # using nearest intentionally (to match above)
            padding_mode="zeros",
            uv_normalized=True,
        ).reshape(b, q, n, 3)  # (b, q, n, 3)

        project_xyz_w_error = torch.linalg.vector_norm(project_xyz_w - xyz_w.unsqueeze(1), dim=-1)  # (b, q, n)
        logit = -project_xyz_w_error / (2 * std_xyz_norm**2)  # (b, q, n)

        if th_xyz_norm is not None:
            valid_mask_xyz_w = project_xyz_w_error < th_xyz_norm  # (b, q, n)
            valid_mask = torch.logical_and(valid_mask, valid_mask_xyz_w)  # (b, q, n)
            logit = logit.masked_fill(~valid_mask_xyz_w, 0)
        else:
            valid_mask_xyz_w = None

        prob_xyz_w = torch.nn.functional.softmax(logit, dim=1)  # (b, q, n)

    else:
        valid_mask_xyz_w = None
        prob_xyz_w = None

    if prob_use_normal or th_normal is not None or cos_softmax_scale is not None:
        assert normal_w_map is not None
        project_normal_w = uv_sampling(
            uv=uv_img_normalized.reshape(b * q, n, 2),  # (bq, n, 2uv_img)
            feature_map=normal_w_map.to(dtype=uv_img.dtype).reshape(b * q, h, w, 3),  # (bq, h, w, 3)
            mode="bilinear",
            padding_mode="zeros",
            uv_normalized=True,
        ).reshape(b, q, n, 3)  # (b, q, n, 3xyz_w)
        project_normal_w = torch.nn.functional.normalize(project_normal_w, dim=-1)

        ray_dir_w = torch.nn.functional.normalize(
            delta_xyz_w,  # (b, q, n, 3)
            dim=-1,
        )  # (b, q, n, 3xyz_w)

        # we want normal to be similar direction as ray_dir (high cos)
        # we do not care about sign for now
        abs_cos_vals = (project_normal_w * ray_dir_w).sum(dim=-1).abs()  # (b, q, n)
        # abs_cos_vals = -1 * (project_normal_w * ray_dir_w).sum(dim=-1)  # (b, q, n)

        if th_normal is not None:
            valid_mask_normal = abs_cos_vals >= th_normal  # (b, q, n)
            valid_mask = torch.logical_and(valid_mask, valid_mask_normal)  # (b, q, n)
            abs_cos_vals = abs_cos_vals.masked_fill(~valid_mask_normal, 0)
        else:
            valid_mask_normal = None

        # use softmax as prob
        prob_normal = torch.nn.functional.softmax(abs_cos_vals * cos_softmax_scale, dim=1)  # (b, q, n)
    else:
        valid_mask_normal = None
        prob_normal = None

    if prob_use_point_normal or th_point_normal is not None or point_cos_softmax_scale is not None:
        assert normal_w is not None
        ray_dir_w = torch.nn.functional.normalize(
            delta_xyz_w,  # (b, q, n, 3)
            dim=-1,
        )  # (b, q, n, 3xyz_w)

        # we want ray_dir parallel to the opposite direction of the normal
        neg_cos_vals = -1 * (normal_w.unsqueeze(1) * ray_dir_w).sum(dim=-1)  # (b, q, n)

        if th_point_normal is not None:
            valid_mask_point_normal = neg_cos_vals >= th_point_normal  # (b, q, n)
            valid_mask = torch.logical_and(valid_mask, valid_mask_point_normal)  # (b, q, n)
            neg_cos_vals = neg_cos_vals.masked_fill(~valid_mask_point_normal, 0)
        else:
            valid_mask_point_normal = None

        # use softmax as prob
        prob_point_normal = torch.nn.functional.softmax(neg_cos_vals * point_cos_softmax_scale, dim=1)  # (b, q, n)
    else:
        valid_mask_point_normal = None
        prob_point_normal = None

    # make sure only choose from valid
    prob_q = torch.ones(b, q, n, dtype=torch.float, device=xyz_w.device)  # (b, q, n)
    if prob_use_dist:
        assert prob_dist is not None
        prob_q = prob_q * prob_dist
    if prob_use_xyz_norm:
        assert prob_xyz_w is not None
        prob_q = prob_q * prob_xyz_w
    if prob_use_z_c:
        assert valid_mask_z_c is not None
        prob_q = prob_q * valid_mask_z_c.float()
    if prob_use_hit:
        assert valid_mask_hit is not None
        prob_q = prob_q * valid_mask_hit.float()
    if prob_use_normal:
        assert prob_normal is not None
        prob_q = prob_q * prob_normal  # (b, q, n)
    if prob_use_point_normal:
        assert prob_point_normal is not None
        prob_q = prob_q * prob_point_normal  # (b, q, n)

    # make sure prob_q is 0 if invalid
    prob_q = prob_q.masked_fill(~valid_mask, 0)  # (b, q, n)

    if deterministic:
        qidxs = torch.argsort(prob_q, dim=1, descending=True)  # (b, q, n)
        qidxs = qidxs[:, :k]  # (b, min(k, q), n)
        qidxs = qidxs.permute(0, 2, 1)  # (b, n, min(k, q))
    else:
        # select qidx based on prob_q
        qidxs = torch.multinomial(
            input=prob_q.permute(0, 2, 1).reshape(b * n, q) + 1e-8,  # (bn, q)
            num_samples=min(k, q),
            replacement=False,
        ).reshape(b, n, k)  # (b, n, k)

    projected_uv_img = torch.gather(
        uv_img,  # (b, q, n, 2uv_img) [0, h] [0, w]
        dim=1,
        index=qidxs.permute(0, 2, 1).reshape(b, k, n, 1).expand(b, k, n, 2),  # (b, q=1, n, 2)
    )  # (b, k, n, 2uv)  [0, 1]

    valid_qidxs = torch.gather(
        input=valid_mask.permute(0, 2, 1),  # (b, n, q)
        dim=2,
        index=qidxs,  # (b, n, k)
    )  # (b, n, k)

    xyz_c = torch.gather(
        input=xyz_c.permute(0, 2, 1, 3),  # (b, n, q, 3xyz_c)
        dim=2,
        index=qidxs.reshape(b, n, k, 1).expand(b, n, k, 3),
    )  # (b, n, k, 3xyz_c)

    return dict(
        qidxs=qidxs,  # (b, n, k)
        projected_uv_img=projected_uv_img.permute(0, 2, 1, 3),  # (b, n, k, 2uv) [0, h] [0, w]
        valid_qidxs=valid_qidxs,  # (b, n, k)  bool
        prob_dist=prob_dist,  # (b, q, n) or 1(float)
        valid_mask=valid_mask,  # (b, q, n)
        valid_mask_z_c=valid_mask_z_c,  # (b, q, n) or None
        valid_mask_hit=valid_mask_hit,  # (b, q, n) or None
        valid_mask_normal=valid_mask_normal,  # (b, q, n) or None
        valid_mask_point_normal=valid_mask_point_normal,  # (b, q, n) or None
        valid_mask_xyz_w=valid_mask_xyz_w,
        prob_xyz_w=prob_xyz_w,  # (b, q, n) or 1(float)
        prob_normal_w=prob_normal,  # (b, q, n) or 1(float)
        prob_point_normal_w=prob_point_normal,  # (b, q, n) or 1(float)
        prob_q=prob_q,  # (b, q, n)
        xyz_c=xyz_c,  # (b, n, k, 3xyz_c)
    )


def forward_topk_occ_xyz_hit(
    k: int,
    xyz_w: torch.Tensor,  # (b, n, 3xyz_w)
    feature_map: torch.Tensor,  # (b, q, h/patch_height_px, w/patch_width_px, d)
    H_c2w: torch.Tensor,  # (b, q, 4, 4)
    intrinsic: torch.Tensor,  # (b, q, 3, 3)
    z_c: T.Optional[torch.Tensor],  # (b, q, h, w)
    hit_map: T.Optional[torch.Tensor],  # (b, q, h, w) bool
    interp_method: str,  # "bilinear", "nearest"
    th_z_c: float = None,
    th_xyz_norm: float = 1e-3,
    patch_width_px: int = 14,
    patch_height_px: int = 14,
):
    b, n, _3 = xyz_w.shape
    # compute xyz_w_maps
    xyz_w_map = compute_3d_xyz(
        z_map=z_c.float(),  # (b, q, h, w)
        intrinsic=intrinsic.float(),  # (b, q, 3, 3)
        H_c2w=H_c2w.float(),  # (b, q, 4, 4)
    )["xyz_w"]  # (b, q, h, w, 3xyz_w)

    sdict = select_topk_views(
        k=k,
        xyz_w=xyz_w,  # (b, n, 3xyz_w)
        z_c_map=z_c,  # (b, q, h, w)
        H_c2w=H_c2w,  # (b, q, 4, 4)
        intrinsic=intrinsic,  # (b, q, 3, 3)
        hit_map=hit_map,  # (b, q, h, w)
        xyz_w_map=xyz_w_map,  # (b, q, h, w, 3xyz_w)
        normal_w_map=None,  # (b, q, h, w)
        th_z_c=th_z_c,
        th_xyz_norm=th_xyz_norm,
        th_normal=None,
        th_point_normal=None,
        std_xyz_norm=0.1,
        std_dist=0.5,
        cos_softmax_scale=None,
        prob_use_dist=True,
        prob_use_hit=True,
        prob_use_z_c=True,
        prob_use_xyz_norm=True,
        prob_use_normal=False,
        deterministic=True,
        point_cos_softmax_scale=None,
    )
    qidxs = sdict["qidxs"]  # (b, n, k)
    valid_qidxs = sdict["valid_qidxs"]  # (b, n, k)
    uv_img = sdict["projected_uv_img"]  # (b, n, k, 2uv_img) [0, w] [0, h]
    valid_mask = sdict["valid_mask"].permute(0, 2, 1)  # (b, n, q)

    # we assume the image and feature map's origin is at top left corner.
    uv_patch = uv_img / torch.tensor(
        [
            patch_width_px,
            patch_height_px,
        ],
        dtype=uv_img.dtype,
        device=uv_img.device,
    )  # (b, n, k, 2uv_patch) [0, w//pw]

    d = feature_map.size(-1)
    patch_feature = sparse_uv_sampling(
        uv=uv_patch.reshape(b, n * k, 2),  # (b, nk, 2uv_patch)  [0, w//pw]
        qidx=qidxs.reshape(b, n * k),  # (b, nk)
        feature_map=feature_map,  # (b, q, ph, pw, d)
        mode=interp_method,  # "bilinear",
        padding_mode="zeros",
        uv_normalized=False,
    ).reshape(b, n, k, d)  # (b, n, k, d)

    # aggregate
    patch_feature = patch_feature.masked_fill(~valid_qidxs.unsqueeze(-1), 0)  # (b, n, k, d)
    patch_feature = patch_feature.sum(dim=2) / torch.clamp(valid_qidxs.sum(dim=2).unsqueeze(-1), min=1)  # (b, n, d)

    point_valid_mask = valid_qidxs.any(dim=2)  # (b, n)

    return dict(
        feature=patch_feature.reshape(b, n, d),  # (b, n, d)
        point_valid_mask=point_valid_mask,  # (b, n)
        uv_patch=uv_patch.unsqueeze(-2),  # (b, n, 1, 2)
        valid_mask=valid_mask,  # (b, n, q)
    )


def compute_visibility_mask_for_pcd_with_depth(
    *,
    xyz_w: torch.Tensor,  # (b, n, 3xyz_w)
    z_c: torch.Tensor,  # (b, q, h, w)
    H_c2w: torch.Tensor,  # (b, q, 4, 4)
    intrinsic: torch.Tensor,  # (b, q, 3, 3)
    hit_map: T.Optional[torch.Tensor] = None,  # (b, q, h, w)
    err_rtol_list: T.List[float] = [0.0],
) -> T.Dict[float, torch.Tensor]:
    """
    bilinear interpolate the feature map and average across views that sees the point
    (within some distance threshold).

    Note that depending on the hyper-parameters (b, q, region_size), it may use A LOT OF
    memory just to store the output of the grid_sampling.

    Args:
        xyz_w:
            (b, n, 3xyz_w)  the points
        z_c:
            (b, q, h, w) the depth map (z in the camera coordinate)
        H_c2w:
            (b, q, 4, 4) camera poses of the original image
        intrinsic:
            (b, q, 3, 3) camera intrinsic of the original image (not the feature map)
        hit_map:
            (b, q, h, w) bool, whether a pixel sees anything
        err_rtol_list:
            a list of float, each of which indicates a level of depth error tolerance.
            For example, a value of err_rtol = 0.01 means that you allow the depth value to have 1% relative errors,
            which will mark more points to be visible.

    Returns:
        visibility_mask:
            (b, #point, #image), bool
    """
    assert (xyz_w.ndim == 3) and (xyz_w.shape[2] == 3), f"{xyz_w.shape=}"
    b, n, _ = xyz_w.shape
    assert (z_c.ndim == 4) and (z_c.shape[0] == b), f"{z_c.shape=}"
    b_z_c, q, h, w = z_c.shape

    # project each point onto images
    uv_img, xyz_c = pinhole_projection(
        xyz_w=xyz_w,  # (b, n, 3xyz_w)
        intrinsics=intrinsic,  # (b, q, 3, 3)
        H_c2w=H_c2w,  # (b, q, 4, 4)
        dim_b=1,
    )  # uv_img: (b, q, n, 2uv_img) [0, w/h] can oob,  xyz_c: (b, q, n, 3xyz_c)

    # interpolate to get the img_xyz_c at the projected point
    z_c_interpolated = uv_sampling(
        uv=uv_img.reshape(b * q, n, 2),  # (bq, n, 2uv_img)
        feature_map=z_c.reshape(b * q, h, w, 1),  # (bq, h, w, 1)
        mode="nearest",
        padding_mode="zeros",
        uv_normalized=False,
    )  # (bq, n, 1)
    z_c_interpolated = z_c_interpolated.reshape(b, q, n)  # (b, q, n)

    visibility_mask_dict = {}
    for err_rtol in err_rtol_list:
        visibility_mask_dict[err_rtol] = z_c_interpolated >= (xyz_c[..., 2] * (1 - err_rtol))  # (b, q, n), bool

    if hit_map is not None:
        project_hit = uv_sampling(
            uv=uv_img.reshape(b * q, n, 2),  # (bq, n, 2uv_img)
            feature_map=hit_map.to(dtype=uv_img.dtype).reshape(b * q, h, w, 1),  # (bq, h, w, 1)
            mode="nearest",
            padding_mode="zeros",
            uv_normalized=False,
        ).reshape(b, q, n)  # (b, q, n)

        for err_rtol in visibility_mask_dict:
            visibility_mask_dict[err_rtol] = torch.logical_and(
                visibility_mask_dict[err_rtol],
                project_hit > 0.5,
            )  # (b, q, n)

    return visibility_mask_dict


def get_np_rng(rng: int | np.random.Generator | None) -> np.random.Generator:
    """
    Return a new random number generator.

    Args:
        rng:
            int, random seed
            None: new random number generator (independent to np.random)
    """
    if rng is None:
        rng = np.random.default_rng()
    else:
        if isinstance(rng, int):
            rng = np.random.Generator(np.random.Philox(seed=rng, counter=0))
        elif isinstance(rng, np.random.Generator):
            pass
        else:
            raise ValueError(f"{type(rng)=}")
    return rng


@linalg_utils.disable_tf32_and_autocast()
def compute_xyz_w_and_select_random_points(
    z_map: torch.Tensor,
    hit_map: torch.Tensor,
    intrinsic: torch.Tensor,
    H_c2w: torch.Tensor,
    num_points: int,
    other_maps: T.List[torch.Tensor] = None,
    return_pinhole_w: bool = False,
    return_pinhole_idx: bool = False,
    repeat_if_not_enough: bool = False,
) -> T.Dict[str, T.Any]:
    """
    First randomly select valid pixels from hit_map.
    Compute the xyz in the world coordinate using z_map and camera pose for the selected pixels.

    Important note:
        The function uses an image coordinate system: x to right, y to "down", z to far.

    Args:
        z_map:
            (b, q, h, w) the z coordinate of the point in the camera coordinate on the sensor,
            not along the corresponding camera ray.
        hit_map:
            (b, q, h, w) bool. The valid pixels to be backprojected.
        intrinsic:
            (b, q, 3, 3) camera intrinsic matrix
        H_c2w:
            (b, q, 4, 4) homegeneous matrix that convert camera coord to world coord.
            Note that the y axis should be inverted in the cam_poses.
        num_points:
            int, total number of point to be selected from images with the same bidx.
        other_maps:
            a list of (b, q, h, w, d) to associated with each point

    Returns:
        xyz_w:
            (b, num_points, 3xyz_w) xyz in world coordinates
        other_maps:
            list, same length as other_maps, each is (b, num_points, d)

    Notes:
        This function can be memory intensive. If memory is a concern, we can trade compute
        as well by first backprojecting all pixels.
    """
    with _nullcontext():
        b, q, h, w = hit_map.shape
        assert hit_map.dtype == torch.bool, f"{hit_map.dtype=}"
        hit_map = torch.logical_and(
            hit_map,
            z_map.isfinite(),
        )  # (b, q, h, w)

        # randomly select pixels
        hit_map_flat = hit_map.view(b, q * h * w)  # (b, q*h*w)
        total_valid_pixels = hit_map_flat.sum(dim=-1)  # (b,)
        mask_enough = total_valid_pixels >= num_points  # (b,) bool
        total_enough = mask_enough.sum().item()
        total_notenough = b - total_enough

        selected_linear_qhw = torch.zeros(
            b,
            num_points,
            dtype=torch.long,
            device=z_map.device,
        )  # (b, num_points) long, each is the lienar index of qhw (w moves fastest)

        valid_seq_lens = torch.ones(b, dtype=torch.long, device=z_map.device) * num_points  # (b,)
        if total_enough > 0:
            # # multinomial supports only 2^24 categories
            # idx_qhw = torch.multinomial(
            #     hit_map_flat[mask_enough].float(),
            #     num_points,
            #     replacement=False,
            # )  # (b, num_points)
            # selected_linear_qhw[mask_enough] = idx_qhw

            idx_qhw = linalg_utils.gumbel_multinomial(
                input=hit_map_flat[mask_enough].float(),  # (b, qhw)
                num_samples=num_points,
                replacement=False,
            )  # (b, num_points)
            selected_linear_qhw[mask_enough] = idx_qhw

        if total_notenough > 0:
            if repeat_if_not_enough:
                idx_qhw = linalg_utils.gumbel_multinomial(
                    input=hit_map_flat[~mask_enough].float(),  # (b, qhw)
                    num_samples=num_points,
                    replacement=True,
                )  # (b, num_points)
                selected_linear_qhw[~mask_enough] = idx_qhw
            else:
                # let's keep it simple
                for ib in range(b):
                    if not mask_enough[ib]:
                        available_idx_qhw = torch.arange(q * h * w)[hit_map_flat[ib]]
                        assert available_idx_qhw.shape[0] == total_valid_pixels[ib]

                        # shuffle it
                        rand_idx = torch.randperm(available_idx_qhw.shape[0])
                        selected_linear_qhw[ib, : total_valid_pixels[ib]] = available_idx_qhw[rand_idx]
                        valid_seq_lens[ib] = total_valid_pixels[ib]

        # linear index -> (q, h, w)
        hw = h * w
        q_idx = selected_linear_qhw // hw  # (b, n)
        rem = selected_linear_qhw % hw  # (b, n)
        h_idx = rem // w  # (b, n)  # v
        w_idx = rem % w  # (b, n)  # u

        # get information for backprojection
        z_c = torch.gather(
            input=z_map.reshape(b, q * h * w),  # (b, qhw)
            dim=-1,
            index=selected_linear_qhw,  # (b, num_points)
        )  # (b, num_points)
        uvw = torch.stack(
            [
                w_idx * z_c,  # (b, n) [0, w]
                h_idx * z_c,  # (b, n) [0, h]
                z_c,  # (b, n)
            ],
            dim=-1,
        )  # (b, n, 3)

        inv_intrinsic = rigid_motion.inv_intrinsic_tensors(intrinsic)  # (b, q, 3, 3)
        # this can be memory intensive. If memory is a concern, we can trade compute
        # as well by first backprojecting all pixels.
        inv_tensor = torch.gather(
            input=inv_intrinsic,  # (b, q, 3, 3)
            dim=1,
            index=q_idx.unsqueeze(-1).unsqueeze(-1).expand(b, num_points, 3, 3),
        )  # (b, n, 3, 3)
        del inv_intrinsic

        # get xyz in camera coordinate
        xyz = linalg_utils.matmul(
            inv_tensor.float(),  # (b, n, 3, 3)
            uvw.float().unsqueeze(-1),  # (b, n, 3, 1)
        )  # (b, n, 3, 1)  xyz in cam coord

        R_c2w = torch.gather(
            input=H_c2w[:, :, :3, :3],  # (b, q, 3, 3)
            dim=1,
            index=q_idx.unsqueeze(-1).unsqueeze(-1).expand(b, num_points, 3, 3),
            out=inv_tensor,  # directly save to inv_tensor (prevent fragmentation)
        )  # (b, n, 3, 3)

        # get xyz in world coordinate
        xyz = linalg_utils.matmul(R_c2w.float(), xyz.float())  # (b, n, 3, 1) xyz in world coord (not yet translated)
        del inv_tensor

        pinhole_w = torch.gather(
            input=H_c2w[:, :, :3, 3],  # (b, q, 3)
            dim=1,
            index=q_idx.unsqueeze(-1).expand(b, num_points, 3),
        )  # (b, n, 3xyz_w)
        xyz = xyz.float().squeeze(-1) + pinhole_w.float()  # (b, n, 3xyz_w)

    # get other maps
    if other_maps is not None:
        all_features = []
        for o_map in other_maps:
            if o_map is not None:
                assert o_map.ndim == 5, f"{o_map.shape=}"
                d = o_map.size(-1)
                out = torch.gather(
                    input=o_map.reshape(b, q * h * w, d),  # (b, qhw, d)
                    dim=-2,
                    index=selected_linear_qhw.unsqueeze(-1).expand(b, num_points, d),  # (b, num_points, d)
                )  # (b, num_points, d)
            else:
                out = None
            all_features.append(out)
    else:
        all_features = None

    return dict(
        xyz_w=xyz,  # (b, num_points, 3xyz_w)
        other_maps=all_features,  # list of (b, num_points, d)
        pinhole_w=pinhole_w if return_pinhole_w else None,  # (b, num_points, 3xyz_w)
        pinhole_idx=q_idx.unsqueeze(-1) if return_pinhole_idx else None,  # (b, num_points, 1) long
        valid_seq_lens=valid_seq_lens,  # (b,)
    )


def get_cubemap_camera(
    center_xyz_w: T.Union[torch.Tensor, T.List[float]],  # (*b, 3xyz_w)
    width_px: int,
    height_px: int,
):
    """
    Generate cubemap cameras (H_c2w, intrinsic)

    Args:
        center_xyz_w:
            (*b, 3xyz_w), the center of the cube
        width_px:
            int, resolution of each image
        height_px:
            int, resolution of each image

    Returns:
        H_c2w:
            (*b, 6, 4, 4)
        intrinsic:
            (*b, 6, 3, 3)
        width_px:
            int
        height_px:
            int

    Notes:
        we assume z-up.
    """

    if isinstance(center_xyz_w, (list, tuple)):
        center_xyz_w = torch.tensor(center_xyz_w, dtype=torch.float)

    *b_shape, _3xyz_w = center_xyz_w.shape

    # view directions
    view_dirs_xyz_w = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # +x
            [0.0, 1.0, 0.0],  # +y
            [-1.0, 0.0, 0.0],  # -x
            [0.0, -1.0, 0.0],  # -y
            [0.0, 0.0, 1.0],  # +z
            [0.0, 0.0, -1.0],  # -z
        ],
        dtype=torch.float,
        device=center_xyz_w.device,
    )  # (6, 3xyz_w)

    up_dirs_xyz_w = torch.tensor(
        [
            [0.0, 0.0, 1.0],  # +x
            [0.0, 0.0, 1.0],  # +y
            [0.0, 0.0, 1.0],  # -x
            [0.0, 0.0, 1.0],  # -y
            [1.0, 0.0, 0.0],  # +z
            [-1.0, 0.0, 0.0],  # -z
        ],
        dtype=torch.float,
        device=center_xyz_w.device,
    )  # (6, 3xyz_w)

    H_c2w = rigid_motion.get_H_c2w_lookat(
        pinhole_location_w=center_xyz_w.unsqueeze(-2).expand(*b_shape, 6, 3),  # (*b, 6, 3xyz_w)
        look_at_w=center_xyz_w.unsqueeze(-2) + view_dirs_xyz_w,  # (*b, 6, 3xyz_w)
        up_w=up_dirs_xyz_w.expand(*b_shape, 6, 3),  # (*b, 6, 3xyz_w)
        invert_y=True,
    )  # (*b, 6, 4, 4)

    intrinsic = torch.from_numpy(
        render.derive_camera_intrinsics(
            width_px=width_px,
            height_px=height_px,
            fov=90,
        )
    ).to(dtype=torch.float, device=center_xyz_w.device)  # (3, 3)
    intrinsic = intrinsic.expand(*b_shape, 6, 3, 3)  # (*b, 6, 3, 3)

    return dict(
        H_c2w=H_c2w,  # (*b, 6, 4, 4)
        intrinsic=intrinsic,  # (*b, 6, 3, 3)
        width_px=width_px,
        height_px=height_px,
    )
