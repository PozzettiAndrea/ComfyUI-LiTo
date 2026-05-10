#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements evaluation metrics for point clouds.
import json
import os.path
import tempfile
import typing as T

import numpy as np

import torch

try:
    from third_party.LION.utils import evaluation_metrics_fast
except:
    evaluation_metrics_fast = None


def normalize_point_clouds_to_aabb(
    xyz_w: torch.Tensor,
) -> T.Dict[str, torch.Tensor]:
    """
    Normalize point cloud to fit in [-1, 1] aabb centered at the origin.

    Args:
        xyz_w:
            (*b, n, 3xyz_w)

    Returns:
        xyz_c:
            (*b, n, 3xyz_c) after normalization
        center_w:
            (*b, 1, 3xyz_w) center of each point cloud
        radius_w:
            (*b, 1, 1) scale of each point cloud
    """

    # get bounding box
    grid_to = xyz_w.max(dim=-2, keepdim=True)[0]  # (*b, 1, 3xyz_w)
    grid_from = xyz_w.min(dim=-2, keepdim=True)[0]  # (*b, 1, 3xyz_w)
    grid_width = grid_to - grid_from  # (*b, 1, 3)

    # scale_w is the max of the three side * 0.5
    radius_w = grid_width.max(dim=-1, keepdim=True)[0] * 0.5  # (*b, 1, 1)
    center_w = (grid_from + grid_to) * 0.5  # (*b, 1, 3xyz_w)

    # shift then scale to [-1, 1]
    xyz_c = (xyz_w - center_w) / radius_w  # (*b, n, 3xyz_c)

    return dict(
        xyz_c=xyz_c,  # (*b, n, 3xyz_c)
        center_w=center_w,  # (*b, 1, 3xyz_w)
        radius_w=radius_w,  # (*b, 1, 1)
    )


@torch.no_grad()
def compute_unpaired_metrics_1nna_cov_mmd_jsd(
    xyz_w: torch.Tensor,
    ref_xyz_w: torch.Tensor,
    batch_size: int = 256,
    norm_box: bool = True,
    accelerated_cd: bool = True,
) -> T.Dict[str, float]:
    """
    Compute 1-nna, coverage, minimum match distance, and jensen shannon divergence
    between unpaired point clouds.

    This is following LION's protocal that makes sure the number of tested point cloud
    is the same as the reference point cloud. Specifically, it randomly selects point
    clouds from xyz_w (along b dimension and n dimension) to match the shape of
    ref_xyz_w's shape.

    Args:
        xyz_w:
            (b1, n1, 3xyz_w)
        ref_xyz_w:
            (b2, n2, 3xyz_w)
        batch_size:
            batch size used to compute metrics (not affecting result)
        norm_box:
            whether to normalize point clouds to fit [-1, 1] aabb
        accelerated_cd:
            whether to use cuda to help compute chamfer distance

    Returns:
        lgan_mmd-CD:
            (,) mmd from ref_pcs to sample_pcs (standard mmd) using chamfer distance
        lgan_mmd_smp-CD:
            (,) mmd from sample_pcs to ref_pcs (reversed mmd) using chamfer distance
        lgan_cov-CD:
            (,) coverage using chamfer distance
        1-NN-CD-acc:
            (,) 1-nn using chamfer distance
        lgan_mmd-EMD:
            (,) mmd from ref_pcs to sample_pcs (standard mmd) using chamfer distance
        lgan_mmd_smp-EMD:
            (,) mmd from sample_pcs to ref_pcs (reversed mmd) using chamfer distance
        lgan_cov-EMD:
            (,) coverage using chamfer distance
        1-NN-EMD-acc:
            (,) 1-nn using chamfer distance
        jsd:
            (,) jensen shannon divergence proposed in https://github.com/optas/latent_3d_points
    """

    b1, n1, _3xyz1 = xyz_w.shape
    b2, n2, _3xyz2 = ref_xyz_w.shape
    assert _3xyz1 == 3 and _3xyz2 == 3
    assert b1 >= b2
    assert n1 >= n2

    # randomly sample points from the generated point cloud to match
    # the number of points in the reference point cloud
    if n1 > n2:
        xperm = torch.randperm(n1)[:n2]
        xyz_w = xyz_w[:, xperm]

    if b1 > b2:
        xyz_w = xyz_w[:b2]

    assert xyz_w.shape == ref_xyz_w.shape
    b, n, _3xyz = xyz_w.shape

    # normalize to fit a aabb [-0.5, 0.5] box centered at the origin
    if norm_box:
        ref_xyz_w = 0.5 * normalize_point_clouds_to_aabb(ref_xyz_w)["xyz_c"]
        xyz_w = 0.5 * normalize_point_clouds_to_aabb(xyz_w)["xyz_c"]

    # compute cov, mmd, 1-nna between generated and synthetic point clouds
    device = xyz_w.device
    results: T.Dict[str, float] = evaluation_metrics_fast.compute_all_metrics(
        sample_pcs=xyz_w.to(device).float(),  # (b, n, 3)
        ref_pcs=ref_xyz_w.to(device).float(),  # (b, n, 3)
        batch_size=batch_size,
        accelerated_cd=accelerated_cd,
        metric1="CD",
        metric2="EMD",
    )

    # compute metrics between generated and synthetic point clouds
    jsd = evaluation_metrics_fast.jsd_between_point_cloud_sets(
        sample_pcs=xyz_w.detach().cpu().numpy(),
        ref_pcs=ref_xyz_w.detach().cpu().numpy(),
    )
    results["jsd"] = jsd
    msg = evaluation_metrics_fast.print_results(results)

    return results


@torch.no_grad()
def compute_unpaired_metrics_pfid(
    xyz_w: T.Union[torch.Tensor, str],
    ref_xyz_w: T.Union[torch.Tensor, str],
) -> float:
    """
    Compute the p-fid from point-e between
    two sets of point clouds.

    Args:
        xyz_w:
            (b1, n, 3xyz)
        ref_xyz_w:
            (b2, m, 3xyz)

    Notes:
        Both xyz_w and ref_xyz_w can be the filenames of an npz file,
        where there is an arr_0 key of shape [N x K x 3],
        where K is the number of points in each point cloud
        and N is the number of clouds.

    Returns:
        the p-fid value as float
    """
    delete_xyz_w_filename = False
    delete_ref_xyz_w_filename = False

    if isinstance(xyz_w, str):
        assert xyz_w.endswith(".npz")
        filename_xyz_w = xyz_w
    elif isinstance(xyz_w, torch.Tensor):
        # save to npz file
        xyz_w = xyz_w.detach().cpu().numpy()
        _, filename_xyz_w = tempfile.mkstemp(dir=".", prefix="xyz_w", suffix=".npz")
        np.savez(filename_xyz_w, arr_0=xyz_w)
        delete_xyz_w_filename = True
    else:
        raise NotImplementedError

    if isinstance(ref_xyz_w, str):
        assert ref_xyz_w.endswith(".npz")
        filename_ref_xyz_w = ref_xyz_w
    elif isinstance(xyz_w, torch.Tensor):
        # save to npz file
        ref_xyz_w = ref_xyz_w.detach().cpu().numpy()
        _, filename_ref_xyz_w = tempfile.mkstemp(dir=".", prefix="ref_xyz_w", suffix=".npz")
        np.savez(filename_ref_xyz_w, arr_0=ref_xyz_w)
        delete_ref_xyz_w_filename = True
    else:
        raise NotImplementedError

    # call the pfid script from point-e
    repo_root = os.path.abspath(
        os.path.normpath(
            os.path.join(
                __file__,
                "../..",
            )
        )
    )
    point_e_dir = os.path.join(repo_root, "third_party/point_e")
    script_filename = "point_e/evals/scripts/evaluate_pfid.py"
    _, out_filename = tempfile.mkstemp(dir=".", prefix="pfid_out", suffix=".json")
    cmd = (
        f"cd {point_e_dir} && python {script_filename} "
        f"{filename_ref_xyz_w} {filename_xyz_w} "
        f"--out_filename {out_filename} "
    )
    os.system(cmd)

    # read result
    assert os.path.exists(out_filename)
    with open(out_filename, "r") as f:
        out_dict = json.load(f)
        pfid = out_dict["p_fid"]  # float

    # remove tmp files
    if delete_xyz_w_filename and os.path.exists(filename_xyz_w):
        os.remove(filename_xyz_w)
    if delete_ref_xyz_w_filename and os.path.exists(filename_ref_xyz_w):
        os.remove(filename_ref_xyz_w)
    if os.path.exists(out_filename):
        os.remove(out_filename)

    return pfid


@torch.no_grad()
def compute_unparied_metrics_pis(
    xyz_w: T.Union[torch.Tensor, str],
) -> float:
    """
    Compute the p-is from point-e of a set of point cloud.

    Args:
        xyz_w:
            (b, n, 3xyz)

    Notes:
        xyz_w can be the filenames of an npz file,
        where there is an arr_0 key of shape [N x K x 3],
        where K is the number of points in each point cloud
        and N is the number of clouds.

    Returns:
        the p-is value as float
    """
    delete_xyz_w_filename = False

    if isinstance(xyz_w, str):
        assert xyz_w.endswith(".npz")
        filename_xyz_w = xyz_w
    elif isinstance(xyz_w, torch.Tensor):
        # save to npz file
        xyz_w = xyz_w.detach().cpu().numpy()
        _, filename_xyz_w = tempfile.mkstemp(dir=".", prefix="xyz_w", suffix=".npz")
        np.savez(filename_xyz_w, arr_0=xyz_w)
        delete_xyz_w_filename = True
    else:
        raise NotImplementedError

    # call the pfid script from point-e
    repo_root = os.path.abspath(
        os.path.normpath(
            os.path.join(
                __file__,
                "../..",
            )
        )
    )
    point_e_dir = os.path.join(repo_root, "third_party/point_e")
    script_filename = "point_e/evals/scripts/evaluate_pis.py"
    _, out_filename = tempfile.mkstemp(dir=".", prefix="pis_out", suffix=".json")
    cmd = f"cd {point_e_dir} && python {script_filename} {filename_xyz_w} --out_filename {out_filename} "
    os.system(cmd)

    # read result
    assert os.path.exists(out_filename)
    with open(out_filename, "r") as f:
        out_dict = json.load(f)
        pis = out_dict["p_is"]  # float

    # remove tmp files
    if delete_xyz_w_filename and os.path.exists(filename_xyz_w):
        os.remove(filename_xyz_w)
    if os.path.exists(out_filename):
        os.remove(out_filename)

    return pis


@torch.no_grad()
def compute_paired_metrics_cd_emd(
    xyz_w: torch.Tensor,
    ref_xyz_w: torch.Tensor,
    chunk_size: int = 256,
    norm_box: bool = True,
    accelerated_cd: bool = True,
    reduced: bool = False,
    compute_emd: bool = False,
) -> T.Dict[str, torch.Tensor]:
    """
    Compute symmetric chamfer distance, earth mover's distance
    between paired point clouds.

    Args:
        xyz_w:
            (b, n1, 3xyz_w)
        ref_xyz_w:
            (b, n2, 3xyz_w)
        chunk_size:
            chunk size (along b) used to compute metrics (not affecting result)
        norm_box:
            whether to normalize point clouds to fit [-0.5, 0.5] aabb
        accelerated_cd:
            whether to use cuda to help compute chamfer distance
        reduced:
            whether to average across b (True) or not
        compute_emd:
            whether to compute emd

    Returns:
        CD:
            (b,) or (,)
        EMD:
            (b,) or (,) or None if not computed
    """

    b1, n1, _3xyz1 = xyz_w.shape
    b2, n2, _3xyz2 = ref_xyz_w.shape
    assert _3xyz1 == 3 and _3xyz2 == 3
    assert b1 == b2

    # normalize to fit a aabb [-0.5, 0.5] box centered at the origin
    if norm_box:
        ref_xyz_w = 0.5 * normalize_point_clouds_to_aabb(ref_xyz_w)["xyz_c"]
        xyz_w = 0.5 * normalize_point_clouds_to_aabb(xyz_w)["xyz_c"]

    out_dict = evaluation_metrics_fast.EMD_CD(
        sample_pcs=xyz_w,
        ref_pcs=ref_xyz_w,
        batch_size=chunk_size,
        accelerated_cd=accelerated_cd,
        reduced=reduced,
        require_grad=False,
        compute_emd=compute_emd,
    )
    return out_dict
