#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements util functions of using pointnet.

from timeit import default_timer as timer
import typing as T

import pytorch3d
import pytorch3d.ops
import torch
from contextlib import nullcontext as _nullcontext
_OPS = None
def _ops():
    global _OPS
    if _OPS is None:
        import comfy.ops
        _OPS = comfy.ops.manual_cast
    return _OPS



def voxel_downsampling(
    xyz_w: torch.Tensor,  # (n, 3)
    bidx: torch.Tensor,  # (n,)
    cell_width: float,
    sigma: float = 0.5,
    features: T.List[torch.Tensor] = None,  # list of (n, d)
    point_aggregation_method: str = "avg",
    feature_aggregation_method: str = "gaussian_avg",
) -> T.Dict[str, T.Any]:
    """
    Use a voxel grid to downsample the input point cloud, the output will
    be a point cloud, one point for each occupied voxel.

    Procedure:
    - Points are discretized into voxels.
    - Each occupied voxel generates exactly one point by averaging all points inside
      or select one point location inside the voxel

    Args:
        xyz_w:
            (n, 3) packed array of point coodinates
        bidx:
            (n,) batch index for individual points (for the packed array)
        cell_width:
            the width of each grid cell.
        sigma:
            the sigma (with respect to 1 cell_width) used in computing the gaussian weight.
        features:
            list of (n, d), point features
        point_aggregation_method:
            'mean': average
            'random_point': randomly select an input point in each voxel
            'random_xyz': randomly select a location in each voxel
        feature_aggregation_method:
            'mean': average of all feature
            'gaussian_avg': weighted average by gaussian
            'exact': if using `random_point`, use the exact feature at the point
            'amax': maxpool of all feature

    Returns:
        new_xyz_w:
            (num_occupied_cells, 3) new point locations
        new_bidx:
            (num_occupied_cells, ) the corresponding batch index
        new_features:
            list of (num_occupied_cells, d) or None, new point features
        vidxs:
            (n,) the voxel index used in the downsampling
        selected_xyz_w_idx:
            (num_occupied_cells, ) if using 'random_point' or None.
    """
    assert cell_width > 0
    n, _3xyz = xyz_w.shape
    assert bidx.shape == (n,)
    sigma = sigma * cell_width

    # discretize the points into voxel ids
    # this means that we voxelize with [cell_width*i, cell_width*(i+1))^3
    voxel_ijk = (xyz_w / cell_width).floor().long()  # (n, 3ijk)
    voxel_bijk = torch.cat([bidx.reshape(n, 1), voxel_ijk], dim=-1)  # (n, 4bijk)
    del voxel_ijk

    # give each voxel (across batch) a unique index
    unique_voxel_bijk, vidxs, counts = torch.unique(voxel_bijk, dim=0, return_inverse=True, return_counts=True)
    # unique_voxel_bijk: (num_nonempty_voxels, 4bijk)
    # vidxs: (n,) long, start from 0
    # counts: (num_nonempty_voxels,)
    num_occupied_cells = counts.size(0)
    new_bidx = unique_voxel_bijk[:, 0]  # (num_nonempty_voxels,)

    # point aggregation
    selected_xyz_w_idx = None
    if point_aggregation_method == "mean":
        # average xyz to get new xyz (so implicitly weighted by sample density)
        new_xyz_w = torch.zeros(
            num_occupied_cells, _3xyz, dtype=xyz_w.dtype, device=xyz_w.device
        )  # (num_occupied_cells, 3xyz)
        new_xyz_w.scatter_reduce_(
            dim=0,
            index=vidxs.unsqueeze(-1).expand(-1, _3xyz),  # (n, 3)
            src=xyz_w,  # (n, 3)
            reduce="mean",
            include_self=False,
        )  # (num_occupied_cells, 3xyz)

    elif point_aggregation_method == "random_point":
        ii = torch.randperm(n, device=vidxs.device)  # (n,)
        shuffled_vidxs = vidxs[ii]  # (n,)  # shuffled_vidxs[j] is from ii[j]
        sorted_idxs, jj = torch.sort(shuffled_vidxs, dim=0, stable=True)

        # select the first
        _, _, kk = torch.unique_consecutive(sorted_idxs, return_inverse=True, return_counts=True)
        kk = torch.cat([kk.new(1).fill_(0), kk[:-1].cumsum(dim=0)], dim=0)  # (num_occupied_cells,)
        # sorted_idx[kk] is selected, which means shuffled_vidxs[jj[kk]] is selected,
        # which means ii[jj[kk]] is selected

        selected_xyz_w_idx = ii[jj[kk]]  # (num_occupied_cells, )
        new_xyz_w = xyz_w[selected_xyz_w_idx]  # (num_occupied_cells, 3)

    elif point_aggregation_method == "random_xyz":
        unique_voxel_xyz_from = unique_voxel_bijk * cell_width  # (num_occupied_cells, 3xyz)
        new_xyz_w = (
            torch.rand(num_occupied_cells, 3, dtype=unique_voxel_xyz_from.dtype, device=unique_voxel_xyz_from.device)
            * cell_width
            + unique_voxel_xyz_from
        )  # (num_occupied_cells, 3xyz)
    else:
        raise NotImplementedError

    # compute feature
    if features is not None and len(features) > 0:
        new_features = []
        if feature_aggregation_method == "gaussian_avg":
            new_xyz_w_expanded = new_xyz_w[vidxs]  # (n, 3)
            squared_dists = torch.sum((xyz_w - new_xyz_w_expanded) ** 2, dim=-1)  # (n,)
            weights = torch.exp(-1 * squared_dists / (2 * sigma**2))  # (n,)
            weights_sum = torch.zeros(
                num_occupied_cells, dtype=weights.dtype, device=weights.device
            )  # (num_occupied_cells,)
            weights_sum.scatter_reduce_(
                dim=0,
                index=vidxs,  # (n,)
                src=weights,  # (n,)
                reduce="sum",
                include_self=False,
            )  # (num_occupied_cells,)

            normalized_weights = weights / weights_sum[vidxs]  # (n,)
            normalized_weights = normalized_weights.unsqueeze(-1)  # (n, 1)

            # weighted average each feature
            for f in features:
                if f is None:
                    new_features.append(None)
                    continue
                _arr = f * normalized_weights  # (n, dim)
                arr = torch.zeros(
                    num_occupied_cells, _arr.size(-1), dtype=_arr.dtype, device=_arr.device
                )  # (num_occupied_cells, dim)
                arr.scatter_reduce_(
                    dim=0,
                    index=vidxs.unsqueeze(-1).expand(-1, _arr.size(-1)),  # (n, dim)
                    src=_arr,  # (n,)
                    reduce="sum",
                    include_self=False,
                )  # (num_occupied_cells,)
                new_features.append(arr)
        elif feature_aggregation_method in ["mean", "amax"]:
            for f in features:
                if f is None:
                    new_features.append(None)
                    continue
                arr = torch.zeros(
                    num_occupied_cells, f.size(-1), dtype=f.dtype, device=f.device
                )  # (num_occupied_cells, dim)
                arr.scatter_reduce_(
                    dim=0,
                    index=vidxs.unsqueeze(-1).expand(-1, f.size(-1)),  # (n, dim)
                    src=f,  # (n,)
                    reduce=feature_aggregation_method,
                    include_self=False,
                )  # (num_occupied_cells,)
                new_features.append(arr)
        elif feature_aggregation_method == "exact":
            assert selected_xyz_w_idx is not None
            for f in features:
                if f is None:
                    new_features.append(None)
                    continue
                arr = f[selected_xyz_w_idx]  # (num_occupied_cells, d)
                new_features.append(arr)
        else:
            raise NotImplementedError
    else:
        new_features = None

    return dict(
        new_xyz_w=new_xyz_w,  # (num_occupied_cells, 3)
        new_bidx=new_bidx,  # (num_occupied_cells,)
        new_features=new_features,  # list of (num_occupied_cells, d)
        selected_xyz_w_idx=selected_xyz_w_idx,  # (num_occupied_cells,)  long
        vidxs=vidxs,  # (n,)
    )


def voxel_discretization(
    xyz: torch.Tensor,  # (n, 3)
    bidx: torch.Tensor,  # (n,)
    cell_width: float,
) -> T.Dict[str, T.Any]:
    """
    Determine voxel index by discretizing xyz

    Procedure:
    - Points are discretized into voxels.
    - Each occupied voxel generates exactly one point by averaging all points inside
      or select one point location inside the voxel

    Args:
        xyz:
            (n, 3) packed array of point coodinates
        bidx:
            (n,) batch index for individual points (for the packed array)
        cell_width:
            the width of each grid cell.

    Returns:
        vidxs:
            (n,) the voxel index used in the downsampling
        voxel_bidx:
            (num_occupied_cells,)
    """
    assert cell_width > 0
    n, _3xyz = xyz.shape
    assert bidx.shape == (n,)

    # discretize the points into voxel ids
    # this means that we voxelize with [cell_width*i, cell_width*(i+1))^3
    voxel_ijk = (xyz / cell_width).floor().long()  # (n, 3ijk)
    voxel_bijk = torch.cat([bidx.reshape(n, 1), voxel_ijk], dim=-1)  # (n, 4bijk)
    del voxel_ijk

    # give each voxel (across batch) a unique index
    # print(f'voxel_bijk.shape = {voxel_bijk.shape}')
    unique_voxel_bijk, vidx, counts = torch.unique(voxel_bijk, dim=0, return_inverse=True, return_counts=True)
    # unique_voxel_bijk: (num_nonempty_voxels, 4bijk)
    # vidxs: (n,) long, start from 0
    # counts: (num_nonempty_voxels,)
    num_occupied_cells = counts.size(0)
    new_bidx = unique_voxel_bijk[:, 0]  # (num_nonempty_voxels,)

    return dict(
        voxel_bidx=new_bidx,  # (num_occupied_cells,)
        vidx=vidx,  # (n,)
        counts=counts,  # (num_occupied_cells,)
    )


def voxel_aggregation(
    num_occupied_voxels: int,
    vidx: torch.Tensor,  # (n,)
    arr: torch.Tensor,  # (n, d)
    aggregation_method: str,
    voxel_xyz: T.Optional[torch.Tensor] = None,  # (m, 3)
    xyz: T.Optional[torch.Tensor] = None,  # (n, 3)
    sigma: T.Optional[float] = None,
    selected_idx: T.Optional[torch.Tensor] = None,  # (m,)
):
    """
    Aggregate features lying in the same voxel.

    Args:
        num_occupied_voxels:
            int, max number of voxels
        vidx:
            (n,) voxel index
        arr:
            (n, d) feature array
        aggregation_method:
            'gaussian_avg'
            'mean',
            'amax',
            'random_select'
            'exact'
        voxel_xyz:
            (m, 3) voxel xyz, needed if using 'gaussian_avg'
        xyz:
            (n, 3) xyz of arr
        sigma:
            sigma in the world coordinate, used by 'gaussian_avg'
        selected_idx:
            (m,) the index of arr each voxel will use
    Returns:
        out:
            (m, d) the aggregated voxel feature
        selected_idx:
            (m,) or None, if using 'random_select' or 'exact'
    """

    m = num_occupied_voxels
    n, d = arr.shape
    assert vidx.shape == (n,)

    _selected_idx = None
    if aggregation_method == "gaussian_avg":
        new_xyz_expanded = voxel_xyz[vidx]  # (n, 3)
        squared_dists = torch.sum((xyz - new_xyz_expanded) ** 2, dim=-1)  # (n,)
        weights = torch.exp(-1 * squared_dists / (2 * sigma**2))  # (n,)
        weights_sum = torch.zeros(m, dtype=weights.dtype, device=weights.device)  # (m,)
        weights_sum.scatter_reduce_(
            dim=0,
            index=vidx,  # (n,)
            src=weights,  # (n,)
            reduce="sum",
            include_self=False,
        )  # (m,)

        # note that since weights > 0, weights_sum[vidx] should be > 0 for every element
        normalized_weights = weights / weights_sum[vidx]  # (n,)
        normalized_weights = normalized_weights.unsqueeze(-1)  # (n, 1)

        # weighted average each feature
        _arr = arr * normalized_weights  # (n, d)
        out = torch.zeros(m, d, dtype=_arr.dtype, device=_arr.device)  # (m, d)
        out.scatter_reduce_(
            dim=0,
            index=vidx.unsqueeze(-1).expand(-1, d),  # (n, d)
            src=_arr,  # (n, d)
            reduce="sum",
            include_self=False,
        )  # (m, d)

    elif aggregation_method in ["mean", "amax"]:
        out = torch.zeros(m, d, dtype=arr.dtype, device=arr.device)  # (m, d)
        out.scatter_reduce_(
            dim=0,
            index=vidx.unsqueeze(-1).expand(-1, d),  # (n, d)
            src=arr,  # (n,)
            reduce=aggregation_method,
            include_self=False,
        )  # (m, d)

    elif aggregation_method == "random_select":
        ii = torch.randperm(n, device=vidx.device)  # (n,)
        shuffled_vidxs = vidx[ii]  # (n,)  # shuffled_vidxs[j] is from ii[j]
        sorted_idxs, jj = torch.sort(shuffled_vidxs, dim=0, stable=True)

        # select the first
        _, kk = torch.unique_consecutive(sorted_idxs, return_inverse=False, return_counts=True)
        kk = torch.cat([kk.new(1).fill_(0), kk[:-1].cumsum(dim=0)], dim=0)  # (m,)
        # sorted_idx[kk] is selected, which means shuffled_vidxs[jj[kk]] is selected,
        # which means ii[jj[kk]] is selected
        _selected_idx = ii[jj[kk]]  # (m, )
        out = arr[_selected_idx]  # (m, d)

    elif aggregation_method == "exact":
        assert selected_idx is not None
        out = arr[selected_idx]  # (m, d)
        _selected_idx = selected_idx

    else:
        raise NotImplementedError

    return dict(
        out=out,  # (m, d)
        selected_idx=_selected_idx,  # (m,)
    )


def compute_knn(
    query: torch.Tensor,  # (b, m, 3)
    ref: torch.Tensor,  # (b, n, 3)
    k: int,
):
    """
    Find the k-nearest neighbors in ref for each point in query.

    Args:
        query:
            (b, m, 3)
        ref:
            (b, n, 3)
        k:
            `n` needs to be >= k

    Returns:
        (b, m, k) long, index in ref for each point in query
    """

    assert ref.size(1) >= k, f"{ref.shape}"
    if query.is_cuda:
        with torch.profiler.record_function("knn with pytorch3d"):
            with _nullcontext():
                out = pytorch3d.ops.knn_points(
                    p1=query.float(),  # (b, m, 3)
                    p2=ref.float(),  # (b, n, 3),
                    K=k,
                    return_nn=False,
                    return_sorted=True,  # important if using pcf
                )
        idx = out.idx  # (b, m, k)
        return idx
    else:
        with torch.profiler.record_function("knn with pytorch3d"):
            out = pytorch3d.ops.knn_points(
                p1=query,  # (b, m, 3)
                p2=ref,  # (b, n, 3),
                K=k,
                return_nn=False,
                return_sorted=True,  # important if using pcf
            )
        idx = out.idx  # (b, m, k)
        return idx


def compute_ball_query(
    query: torch.Tensor,  # (b, m, 3)
    ref: torch.Tensor,  # (b, n, 3)
    k: int,
    radius: float,
):
    """
    Find the max of k neighbors in ref within the radius ball of each point of query

    Args:
        query:
            (b, m, 3)
        ref:
            (b, n, 3)
        k:
            `n` needs to be >= k
        radius:
            radius of the ball

    Returns:
        (b, m, k) long, index in ref for each point in query
    """

    assert ref.size(1) >= k, f"{ref.shape}"
    if query.is_cuda:
        with torch.profiler.record_function("ball query with pytorch3d"):
            with _nullcontext():
                out = pytorch3d.ops.ball_query(
                    p1=query.float(),  # (b, m, 3)
                    p2=ref.float(),  # (b, n, 3),
                    K=k,
                    radius=radius,
                    return_nn=False,
                )
    else:
        with torch.profiler.record_function("ball query with pytorch3d"):
            out = pytorch3d.ops.ball_query(
                p1=query,  # (b, m, 3)
                p2=ref,  # (b, n, 3),
                K=k,
                radius=radius,
                return_nn=False,
            )

    idx = out.idx  # (b, m, k)

    # handle padding (-1 in idx) by replacing with the first element
    # this is a design choice since we use max pooling to mix neighbor
    first = idx[:, :, 0:1].repeat(1, 1, idx.size(-1))  # (b, s, k)
    assert (first >= 0).all()
    invalid_mask = idx < 0  # (b, s, k)
    idx[invalid_mask] = first[invalid_mask]

    return idx


def gather_arr(arr: torch.Tensor, idx: torch.Tensor):
    """
    Gather value from arr.

    Args:
        arr:
            (b, n, d) input data
        idx:
            (b, *s) index, each element is the index of n

    Returns:
        (b, *s, d) gathered data
    """
    device = arr.device
    b = arr.shape[0]
    view_shape = list(idx.shape)  # (b, *s)
    view_shape[1:] = [1] * (len(view_shape) - 1)  # (b, *1s)
    repeat_shape = list(idx.shape)  # (b, *s)
    repeat_shape[0] = 1  # (1, *s)
    batch_indices = (
        torch.arange(b, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)  # (b, *s)
    )
    new_points = arr[batch_indices, idx, :]  # (b, *s, d)
    return new_points


def random_sample_and_knn_group(
    npoint: int,
    num_neighbors: int,
    xyz: torch.Tensor,
    feature: T.Optional[torch.Tensor],
    radius: float = None,
    fps_deterministic: bool = False,
    th_fps: int = 50_000,
    printout: bool = False,
):
    """
    Select next-level's point xyz, gather neighbor points, and construct feature.

    Input:
        npoint:
            number of samples to return in the farthest point sampling.
            ie, number of samples to retain after the processing
        num_neighbors:
            number of neighbor samples
        xyz:
            (b, n, 3xyz), input points position data
        feature:
            (b, n, d), input points data
        radius:
            float, radius of ball query. If None: use knn.
        fps_deterministic:
            whether to use deterministic farthest point sampling
        th_fps:
            max number of points to use farthest point sampling

    Return:
        new_xyz:
            (b, npoint, 3xyz), sampled points position data
        neighbor_feature:
            (b, npoint, num_neighbors, 3xyz + d), selected neighbor point's feature

    Notes:
        We use random sampling when the number of points (ie, n) is larger than
        the th_fps.
    """
    b, n, _3xyz = xyz.shape
    m = npoint
    assert n >= num_neighbors

    # determine next-level centroids' xyz
    if printout:
        print(f"start sampling points..", flush=True)
    stime = timer()
    if n > m:
        if n <= th_fps:
            # farthest point sampling
            with torch.profiler.record_function("farthest point sampling"):
                new_xyz, fps_idx = pytorch3d.ops.sample_farthest_points(
                    points=xyz,  # (b, n, 3)
                    K=m,
                    random_start_point=not fps_deterministic,
                )  # (b, m, 3),  (b, m)
        else:
            # random selection
            with torch.profiler.record_function("random point sampling"):
                fps_idx = torch.randperm(n, device=xyz.device)[:m]  # (m,)
                new_xyz = xyz[:, fps_idx]  # (b, m, 3)
    elif m == n:
        new_xyz = xyz  # (b, m=n, 3)
    else:
        # m > n
        # use all current points and randomly select a few
        fps_idx = torch.randperm(n, device=xyz.device)[: (m - n)]  # (m-n,)
        new_xyz = torch.cat(
            [
                xyz,  # (b, n, 3)
                xyz[:, fps_idx],  # (b, m-s, 3)
            ],
            dim=1,
        )  # (b, m, 3)

    ttime = timer() - stime
    if printout:
        print(f"  finished sampling points, used {ttime * 1000:.3f} ms", flush=True)

    # gather neighboring points
    if radius is None or radius <= 0:
        if printout:
            print(f"start knn..", flush=True)
        stime = timer()
        with torch.profiler.record_function("compute_knn"):
            with torch.no_grad():
                neighbor_idx = compute_knn(
                    query=new_xyz,  # (b, m, 3)
                    ref=xyz,  # (b, n, 3),
                    k=num_neighbors,
                )  # (b, m, num_neighbors)
        ttime = timer() - stime
        if printout:
            print(f"  finished knn, used {ttime * 1000:.3f} ms", flush=True)
    else:
        # ball query
        if printout:
            print(f"start ball query..", flush=True)
        stime = timer()
        with torch.profiler.record_function("ball_query"):
            with torch.no_grad():
                neighbor_idx = compute_ball_query(
                    query=new_xyz,  # (b, m, 3)
                    ref=xyz,  # (b, n, 3),
                    k=num_neighbors,
                    radius=radius,
                )  # (b, m, num_neighbors)
        ttime = timer() - stime
        if printout:
            print(f"  finished ball query, used {ttime * 1000:.3f} ms", flush=True)

    neighbor_xyz = gather_arr(
        arr=xyz,  # (b, n, 3)
        idx=neighbor_idx,  # (b, m, num_neighbors)
    )  # (b, m, num_neighbors, 3xyz)

    # use local coodindate of the centroid
    neighbor_xyz = neighbor_xyz - new_xyz.view(b, m, 1, _3xyz)  # (b, m, num_neighbors, 3xyz)

    if feature is not None:
        neighbor_feature = gather_arr(
            arr=feature,  # (b, n, d)
            idx=neighbor_idx,  # (b, m, num_neighbors)
        )  # (b, m, num_neighbors, d)
        neighbor_feature = torch.cat([neighbor_xyz, neighbor_feature], dim=-1)  # (b, m, num_neighbors, 3+d)
    else:
        neighbor_feature = neighbor_xyz

    return new_xyz, neighbor_feature


def batch_to_packed(
    arr: T.Union[torch.Tensor, T.List[torch.Tensor]],
):
    """
    Convert batch format (b, n, *d) to (bn, *d)

    Args:
        arr:
            (b, n, *d) or list of (b, n, *d)

    Returns:
        arr:
            (bn, *d) or list of (bn, *d)
        bidx:
            (bn,)  long
    """
    if isinstance(arr, torch.Tensor):
        arr = [arr]
        is_tensor = True
    else:
        is_tensor = False

    b, n = arr[0].shape[:2]
    bn = b * n
    bidx = torch.arange(b, device=arr[0].device).unsqueeze(-1).expand(-1, n).reshape(bn)  # (bn,)
    out_arr = []
    for i in range(len(arr)):
        _b, _n, *d = arr[i].shape
        assert b == _b and n == _n
        _arr = arr[i].reshape(bn, *d)  # (bn, *d)
        out_arr.append(_arr)

    if is_tensor:
        out_arr = out_arr[0]

    return dict(
        arr=out_arr,  # (bn, *d) or list of (bn, *d)
        bidx=bidx,  # (bn,)
    )


def packed_to_batch(
    packed_arr: T.Union[torch.Tensor, T.List[torch.Tensor]],
    bidx: torch.Tensor,
    b: int = None,
    m: int = None,
    random_select: bool = False,
):
    """
    Convert packed format (bn, *d) to batch format (b, n, *d)

    Args:
        packed_arr:
            (bn, *d) or a list of (bn, *d)
        bidx:
            (bn,)
        m:
            max number elements to unpack.  None: keep all
        random_select:
            whether to first shuffle (so the kept elements are randomly selected)

    Returns:
        arr:
            (b, m, *d)
        count:
            (b,) valid number of element along n for each sample in the batch
    """

    if isinstance(packed_arr, torch.Tensor):
        packed_arr = [packed_arr]
        is_tensor = True
    else:
        is_tensor = False

    bn = packed_arr[0].size(0)
    for i in range(len(packed_arr)):
        assert packed_arr[i].size(0) == bn

    if random_select:
        ii = torch.randperm(bn, device=packed_arr[0].device)
        bidx = bidx[ii]
        for i in range(len(packed_arr)):
            packed_arr[i] = packed_arr[i][ii]
        del ii

    bidx, ii = torch.sort(bidx, descending=False, stable=True)  # (m,),  (m,)
    _, counts = torch.unique_consecutive(bidx, return_counts=True)  # (b,)
    first_idxs = torch.cat([counts.new(1).fill_(0), counts.cumsum(dim=0)[:-1]], dim=0)  # (b,)

    if m is None:
        m = counts.max().item()

    # make sure all samples have the elements
    if b is not None:
        assert counts.size(0) == b

    out_arr = []
    for i in range(len(packed_arr)):
        # sort by bidx (small to large)
        arr = packed_arr[i][ii]  # (m, *d)

        arr = pytorch3d.ops.packed_to_padded(
            inputs=arr,  # (m, *d)
            first_idxs=first_idxs,
            max_size=m,
        )  # (b, m, *d)

        out_arr.append(arr)

    if is_tensor:
        out_arr = out_arr[0]

    return dict(
        arr=out_arr,  # (b, m, *d) or list of (b, m, *d)
        counts=counts,  # (b,)
        first_idxs=first_idxs,  # (b,)
    )


class PointNetLayer(torch.nn.Module):
    def __init__(
        self,
        in_channel: int,
        mlp: T.List[int],
        num_neighbors: int,
        th_fps: int = 50_000,
        norm_type: str = "batchnorm",
    ):
        """
        Args:
            in_channel:
                dimension of input feature (excluding 3xyz)
            mlp:
                (num_layer,) the feature dimension of each layer of the mlp.
                Its last element will be the output dimension of the layer.
            num_neighbors:
                number of neighbors
            th_fps:
                max number of points to use farthest point sampling
            norm_type:
                'layernorm'
                'batchnorm'
                'none'
        """
        super().__init__()
        self.in_channel = in_channel + 3  # with xyz
        self.num_neighbors = num_neighbors
        self.th_fps = th_fps
        self.norm_type = norm_type

        # linear layer implemented by 1x1 conv2d
        self.mlp_convs = torch.nn.ModuleList()
        # batchnorm
        self.mlp_bns = torch.nn.ModuleList()
        last_channel = self.in_channel
        for out_channel in mlp:
            self.mlp_convs.append(_ops().Conv2d(in_channels=last_channel, out_channels=out_channel, kernel_size=1))

            if self.norm_type == "layernorm":
                norm = _ops().LayerNorm(normalized_shape=out_channel)
            elif self.norm_type == "batchnorm":
                norm = torch.nn.BatchNorm2d(num_features=out_channel)
            elif self.norm_type == "none":
                norm = torch.nn.Identity()
            else:
                raise NotImplementedError
            self.mlp_bns.append(norm)
            last_channel = out_channel
        self.dim_out = last_channel

    def forward(
        self,
        xyz: torch.Tensor,
        feature: torch.Tensor,
        npoint: int,
        radius: float = None,
        printout: bool = False,
    ):
        """
        Args:
            xyz:
                (b, n, 3xyz_w), input coordinate
            feature:
                (b, n, d), input feature
            radius:
                float, radius of ball query. If None: use knn.
        Returns:
            new_xyz:
                (b, npoint, 3xyz_w), output centroid coordinate (ie, next level's coordinate)
            new_feature:
                (b, npoint, dim_out), feature associated with the output centroid
        """

        # compute centroid for output and their neighbors
        new_xyz, neighbor_feature = random_sample_and_knn_group(
            npoint=npoint,
            num_neighbors=self.num_neighbors,
            xyz=xyz,  # (b, n, 3xyz)
            feature=feature,  # (b, n, d)
            radius=radius,
            fps_deterministic=not self.training,
            th_fps=self.th_fps,
            printout=printout,
        )  # (b, npoint, 3), (b, npoint, num_neighbors, 3+d)
        assert neighbor_feature.size(-1) == self.in_channel, f"{neighbor_feature.shape}, {self.in_channel}"

        neighbor_feature = neighbor_feature.permute(0, 3, 2, 1)  # (b, 3+d, num_neighbors, npoint)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            # linear -> batch_norm (along feature) -> relu
            neighbor_feature = conv(neighbor_feature)  # (b, 3+d, num_neighbors, npoint)

            if self.norm_type in ["batchnorm", "none"]:
                neighbor_feature = bn(neighbor_feature)  # (b, 3+d, num_neighbors, npoint)
            elif self.norm_type in ["layernorm"]:
                neighbor_feature = neighbor_feature.permute(0, 2, 3, 1)  # (b, num_neighbors, npoint, 3+d)
                neighbor_feature = bn(neighbor_feature).permute(0, 3, 1, 2)  # (b, 3+d, num_neighbors, npoint)
            else:
                raise NotImplementedError
            # relu
            neighbor_feature = torch.nn.functional.relu(neighbor_feature, inplace=True)

        # max pool along the neighbor dimension
        new_feature = torch.max(neighbor_feature, dim=2)[0]  # (b, 3+d, npoint)
        new_feature = new_feature.permute(0, 2, 1)
        return new_xyz, new_feature


class PointNet(torch.nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        num_layers: int,
        npoint_ratios: T.Union[float, T.List[float]] = 1 / 3.0,
        num_neighbors: T.Union[int, float, T.List[T.Union[int, float]]] = 16,
        mlps_base: T.List[int] = (64, 64, 128),
        width_mult: int = 1,
        layer_mult: T.Union[T.Union[int, float], T.List[T.Union[int, float]]] = 2,
        th_fps: int = 50_000,
        th_radius: float = None,
        norm_type: str = "batchnorm",
    ):
        """
        Pointnet

        Args:
            in_channel:
                input feature dimension (excluding 3xyz)
            out_channel:
                output feature dimension (excluding 3xyz)
            num_layers:
                number of pointnet layers
            npoint_ratios:
                (num_lyaers,) or float,  the downsampling ratio used at each layer. (0, 1]. Use 1 if no downsampling
                wanted for that layer.
            num_neighbors:
                (num_lyaers,) or int, number of neighbors used in each layer
            mlps_base:
                (num_mlp_layers,) base mlp dimension, one for each linear layer.
            width_mult:
                int, multiply upon mlps_base
            th_fps:
                max number of points to use farthest point sampling
            th_radius:
                if radius < th_radius, use ball query instead of knn.
                None: always use knn
            norm_type:
                'layernorm'
                'batchnorm'
                'none'
        """
        super().__init__()
        self.width_mult = width_mult
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.th_fps = th_fps
        self.th_radius = th_radius
        self.num_layers = num_layers
        self.npoint_ratios = npoint_ratios
        if isinstance(npoint_ratios, (int, float)):
            self.npoint_ratios = [self.npoint_ratios] * self.num_layers
        assert len(self.npoint_ratios) == self.num_layers

        if isinstance(layer_mult, (int, float)):
            layer_mult = [layer_mult] * self.num_layers
        assert len(layer_mult) == self.num_layers

        # construct mlp feature dim for each layer
        self.mlps = []
        for i in range(self.num_layers):
            mlp = [int(d * width_mult * layer_mult[i]) for d in mlps_base]
            self.mlps.append(mlp)

        # construct num_neighbors
        self.num_neighbors = num_neighbors
        if isinstance(self.num_neighbors, int):
            self.num_neighbors = [self.num_neighbors] * self.num_layers
        assert len(self.num_neighbors) == self.num_layers

        # construct main layers
        current_dim = self.in_channel
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            layer = PointNetLayer(
                in_channel=current_dim,
                mlp=self.mlps[i],
                num_neighbors=self.num_neighbors[i],
                th_fps=self.th_fps,
                norm_type=norm_type,
            )
            current_dim = layer.dim_out
            self.layers.append(layer)

        # final linear layer
        self.final_linear = _ops().Linear(current_dim, self.out_channel)

    def forward(
        self,
        xyz: torch.Tensor,
        feature: torch.Tensor,
        radius: float = None,
        printout: bool = False,
    ):
        """
        Args:
            xyz:
                (b, n, 3xyz_w) input coordinates
            feature:
                (b, n, d) input feature associated with each point
            radius:
                float, radius of ball query. If None: use knn.
        Returns:
            new_xyz:
                (b, npoint_out, 3xyz_w), output centroid coordinate
            new_feature:
                (b, npoint_out, dim_out), feature associated with the output centroid
        """

        assert xyz.shape[:2] == feature.shape[:2]

        if self.th_radius is None:
            radius = None

        for i in range(len(self.layers)):
            # compute next level's npoint
            npoint = max(1, int(xyz.size(1) * self.npoint_ratios[i]))

            layer = self.layers[i]
            xyz, feature = layer(
                xyz=xyz,
                feature=feature,
                npoint=npoint,
                radius=radius if radius is not None and radius < self.th_radius else None,
                printout=printout,
            )  # (b, npoint, 3xyz_w),  (b, npoint, d)

            if radius is not None:
                radius *= 2

        # final layer
        feature = self.final_linear(feature)
        return xyz, feature


class VNetLayer(torch.nn.Module):
    r"""
    The network utilizes voxel downsampling (ie, scatter_reduce) as the aggregation
    method to downsample a featured point cloud.
    """

    def __init__(
        self,
        in_channel: int,
        mlp: T.List[int],
        cell_width: float,
        point_aggregation_method: str,
        feature_aggregation_method: str,
        norm_type: str = "batchnorm",
    ):
        """
        Args:
            in_channel:
                dimension of input feature (excluding 3xyz)
            mlp:
                (num_layer,) the feature dimension of each layer of the mlp.
                Its last element will be the output dimension of the layer.
            cell_width:
                voxel width used in voxel downsampling
            point_aggregation_method:
                'mean': average of all feature
                'random_select':
            feature_aggregation_method:
                'mean': average of all feature
                'gaussian_avg': weighted average by gaussian
                'exact': if using `random_point`, use the exact feature at the point
                'amax': maxpool of all feature
            norm_type:
                'layernorm'
                'batchnorm'
                'none'
        """
        super().__init__()
        self.in_channel = in_channel + 3  # with xyz
        self.point_aggregation_method = point_aggregation_method
        self.feature_aggregation_method = feature_aggregation_method
        self.norm_type = norm_type
        self.cell_width = cell_width

        # linear layer implemented by a linear layer
        self.mlp_linears = torch.nn.ModuleList()
        # batchnorm
        self.mlp_bns = torch.nn.ModuleList()
        last_channel = self.in_channel
        for out_channel in mlp:
            self.mlp_linears.append(_ops().Linear(in_features=last_channel, out_features=out_channel))

            if self.norm_type == "layernorm":
                norm = _ops().LayerNorm(normalized_shape=out_channel)
            elif self.norm_type == "batchnorm":
                # note that it is different -- original pointnet uses batchnorm2d,
                # which compute statistics among selected neighbor-only (b, m, k).
                # Here since all points will be used, we directly compute batchnorm
                # on the entire points (b, n)
                norm = torch.nn.BatchNorm1d(num_features=out_channel)
            elif self.norm_type == "none":
                norm = torch.nn.Identity()
            else:
                raise NotImplementedError
            self.mlp_bns.append(norm)
            last_channel = out_channel
        self.dim_out = last_channel

    def forward(
        self,
        bidx: torch.Tensor,
        xyz: torch.Tensor,
        feature: torch.Tensor,
        printout: bool = False,
    ):
        """
        Args:
            bidx:
                (n,) batch index for the packed format
            xyz:
                (n, 3xyz_w), input coordinate, in packed format
            feature:
                (n, d), input feature, in packed format
        Returns:
            new_bidx:
                (m,), output batch index, for the packed format
            new_xyz:
                (m, 3xyz_w), output centroid coordinate (ie, next level's coordinate), packed
            new_feature:
                (m, dim_out), feature associated with the output centroid, packed
        """

        # discretize into voxel
        with torch.profiler.record_function(f"voxel discretization (n={xyz.size(0)}, cw={self.cell_width})"):
            vdict = voxel_discretization(
                xyz=xyz,  # (n, 3)
                bidx=bidx,  # (n,)
                cell_width=self.cell_width,
            )
        voxel_bidx = vdict["voxel_bidx"]  # (m=num_occupied_cells,)
        vidx = vdict["vidx"]  # (n,)
        m = voxel_bidx.size(0)

        # print(f'num_occupied_cells = {m}')

        # determine centroid of occupied voxels
        with torch.profiler.record_function(f"voxel aggregation for xyz (m={m})"):
            odict = voxel_aggregation(
                num_occupied_voxels=m,
                vidx=vidx,  # (n,)
                arr=xyz,  # (n, 3)
                aggregation_method=self.point_aggregation_method,
            )
        voxel_xyz = odict["out"]  # (m, 3)
        selected_idx = odict["selected_idx"]  # (m,) or None

        # compute relative xyz for each point
        relative_xyz = xyz - voxel_xyz[vidx]  # (n, 3xyz_r)
        feature = torch.cat([relative_xyz, feature], dim=-1)  # (n, d+3)

        # compute features for each point
        with torch.profiler.record_function(f"feature forward"):
            for i, linear_layer in enumerate(self.mlp_linears):
                # linear -> batch_norm (along feature) -> relu
                bn = self.mlp_bns[i]

                # linear
                feature = linear_layer(feature)  # (n, d)

                # batchnorm
                if self.norm_type in ["batchnorm", "none"]:
                    feature = bn(feature)  # (n, d)  statistics along n
                elif self.norm_type in ["layernorm"]:
                    feature = bn(feature)  # (n, d)  statistics along d
                else:
                    raise NotImplementedError

                # relu
                feature = torch.nn.functional.relu(feature, inplace=True)  # (n, d)

        # aggregate feature
        with torch.profiler.record_function(f"voxel aggregation for feature (m={m})"):
            odict = voxel_aggregation(
                num_occupied_voxels=m,
                vidx=vidx,  # (n,)
                arr=feature,  # (n, d)
                aggregation_method=self.feature_aggregation_method,
                voxel_xyz=voxel_xyz,  # (m, 3)
                xyz=xyz,  # (n, 3)
                sigma=0.5 * self.cell_width,
                selected_idx=selected_idx,  # (m,) or None
            )
        voxel_feature = odict["out"]  # (m, d)

        return dict(
            new_bidx=voxel_bidx,  # (m,)
            new_xyz=voxel_xyz,  # (m, 3)
            new_feature=voxel_feature,  # (m, d)
        )


class VNet(torch.nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        cell_widths: T.List[float],
        mlps_base: T.List[int] = (8, 8, 16),
        width_mult: int = 1,
        layer_mult: T.Union[T.Union[int, float], T.List[T.Union[int, float]]] = 2,
        norm_type: str = "batchnorm",
        point_aggregation_method: str = "mean",
        feature_aggregation_method: str = "amax",
    ):
        """
        Pointnet-style but only uses voxel downsampling

        Args:
            in_channel:
                input feature dimension (excluding 3xyz)
            out_channel:
                output feature dimension (excluding 3xyz)
            cell_widths:
                (num_layers,) each layer's voxel cell width
            mlps_base:
                (num_mlp_layers,) base mlp dimension, one for each linear layer.
            width_mult:
                int, multiply upon mlps_base
            norm_type:
                'layernorm'
                'batchnorm'
                'none'
            point_aggregation_method:
                'mean': average of all feature
                'random_select':
            feature_aggregation_method:
                'mean': average of all feature
                'gaussian_avg': weighted average by gaussian
                'exact': if using `random_point`, use the exact feature at the point
                'amax': maxpool of all feature
        """
        super().__init__()
        self.width_mult = width_mult
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.cell_widths = cell_widths
        self.num_layers = len(self.cell_widths)
        self.point_aggregation_method = point_aggregation_method
        self.feature_aggregation_method = feature_aggregation_method

        if isinstance(layer_mult, (int, float)):
            layer_mult = [layer_mult] * self.num_layers
        assert len(layer_mult) == self.num_layers

        print(f"vnet:\n  cell_widths: {self.cell_widths}\n")

        # construct mlp feature dim for each layer
        self.mlps = []
        for i in range(self.num_layers):
            mlp = [int(d * width_mult * layer_mult[i]) for d in mlps_base]
            self.mlps.append(mlp)

        # construct main layers
        current_dim = self.in_channel
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            layer = VNetLayer(
                in_channel=current_dim,
                mlp=self.mlps[i],
                cell_width=self.cell_widths[i],
                point_aggregation_method=self.point_aggregation_method,
                feature_aggregation_method=self.feature_aggregation_method,
                norm_type=norm_type,
            )
            current_dim = layer.dim_out
            self.layers.append(layer)

        # final linear layer
        self.final_linear = _ops().Linear(current_dim, self.out_channel)

    def forward(
        self,
        xyz: torch.Tensor,
        feature: torch.Tensor,
        max_m: T.Optional[int],
        bidx: torch.Tensor = None,
        b: int = None,
        input_format: str = "batch",
        output_format: str = "batch",
        printout: bool = False,
    ):
        """
        Args:
            xyz:
                (b, n, 3xyz_w) batch or (bn, 3xyz_w) packed, input coordinates
            feature:
                (b, n, d) batch or (bn, d) packed, input feature associated with each point
            max_m:
                int or None. If None, keep all element for each batch
            bidx:
                (bn,) needed if packed
            input_format:
                'packed'
                'batch'
            output_format:
                'packed'
                'batch'
        Returns:
            new_xyz:
                (b, npoint_out, 3xyz_w), output centroid coordinate
            new_feature:
                (b, npoint_out, dim_out), feature associated with the output centroid
            valid_len:
                (b,)
        """

        # convert to packed format
        if input_format == "batch":
            assert xyz.shape[:2] == feature.shape[:2]
            b, n, _3xyz = xyz.shape

            # convert to packed format
            odict = batch_to_packed(
                arr=[xyz, feature],
            )
            xyz = odict["arr"][0]  # (bn, 3)
            feature = odict["arr"][1]  # (bn, d)
            bidx = odict["bidx"]  # (bn,)

        elif input_format == "packed":
            assert bidx is not None
            assert b is not None
            assert xyz.size(0) == bidx.size(0)
            assert feature.size(0) == bidx.size(0)

        else:
            raise NotImplementedError

        for i in range(len(self.layers)):
            layer = self.layers[i]

            # print(f'layer {i}: cw = {layer.cell_width}')

            out_dict = layer(
                bidx=bidx,
                xyz=xyz,
                feature=feature,
            )
            bidx = out_dict["new_bidx"]  # (m,)
            xyz = out_dict["new_xyz"]  # (m, 3)
            feature = out_dict["new_feature"]  # (m, dout)

            # print(f'vnet layer {i}: xyz.shape = {xyz.shape}')

        # final layer
        feature = self.final_linear(feature)  # (m, dout)

        # convert packed format back to batch format
        if output_format == "batch":
            odict = packed_to_batch(
                packed_arr=[xyz, feature],
                bidx=bidx,
                b=b,
                m=max_m,
                random_select=True,
            )
            xyz, feature = odict["arr"]  # (b, n, 3),  (b, n, d)
            counts = odict["counts"]  # (b,)
            first_idxs = odict["first_idxs"]  # (b,)
            return dict(
                xyz=xyz,  # (b, max_m, 3)
                feature=feature,  # (b, max_m, d)
                valid_len=counts,  # (b,)
                first_idxs=first_idxs,  # (b,)
            )

        elif output_format == "packed":
            if max_m is not None:
                # each b should still have at most max_m
                assert NotImplementedError

            return dict(
                xyz=xyz,  # (m, 3)
                feature=feature,  # (m, d)
                bidx=bidx,  # (m,)
                b=b,
            )
        else:
            raise NotImplementedError
