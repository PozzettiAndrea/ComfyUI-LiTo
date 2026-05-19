#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
# The file implements the basic data containers.
from __future__ import annotations  # type annotations stay as strings, never evaluated

import copy
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from timeit import default_timer as timer
import typing as T
import warnings
import zipfile

import numpy as np
import scipy as sp
import scipy.signal
from scipy.spatial.transform import Rotation
import skimage

import torch

# Optional heavy/platform-specific imports — only used in mesh-rendering / EXR
# paths that the ComfyUI inference wrapper does not call.
try:
    import imageio
except ImportError:
    imageio = None
try:
    import open3d as o3d
except ImportError:
    o3d = None
try:
    import pyexr
except ImportError:
    pyexr = None
try:
    import torchvision
except ImportError:
    torchvision = None
try:
    import pytorch3d.ops
    import pytorch3d.renderer
    import pytorch3d.structures
except ImportError:
    pytorch3d = None

from plibs import gs_utils, linalg_utils, rigid_motion, utils
from contextlib import nullcontext as _nullcontext
byte_dict_utils = None
exr_utils = None
img_utils = None
mesh_utils = None
render = None
sample_utils = None

try:
    import nvdiffrast.torch as dr

    NVDIFFRAST_LOADED = True
except:
    NVDIFFRAST_LOADED = False

INF = 1e12


class PointCloud:
    def __init__(
        self,
        xyz_w: torch.Tensor,  # (b, n, 3)
        rgb: T.Optional[torch.Tensor] = None,  # (b, n, 3)
        normal_w: T.Optional[torch.Tensor] = None,  # (b, n, 3)  vertex normal in world coordinate
        captured_z_direction_w: T.Optional[torch.Tensor] = None,
        # (b, n, 3)  # (z axis of the camera in world coordinate)
        captured_dps: T.Optional[torch.Tensor] = None,  # (b, n, 1)  distance per sample
        captured_dps_u_w: T.Optional[torch.Tensor] = None,
        # (b, n, 3)  distance per sample in camera x direction in the word coord
        captured_dps_v_w: T.Optional[torch.Tensor] = None,
        # (b, n, 3)  distance per sample in camera y direction in the word coord
        captured_pinhole_w: T.Optional[torch.Tensor] = None,  # (b, n, 3) pinhole location of the captured camera
        captured_view_direction_w: T.Optional[torch.Tensor] = None,
        # (b, n, 3)  # unit vector pointing from capturing camera pinhole to the point
        valid_mask: T.Optional[torch.Tensor] = None,  # (b, n, 1)
        included_point_at_inf: bool = False,  # whether n==0 represents point at inf
        feature: T.Optional[torch.Tensor] = None,  # (b, n, f)
        img_idxs: T.Optional[torch.Tensor] = None,  # (b, n, 1)  linear index of qhw in the original rgbd image
        timestamp: T.Optional[torch.Tensor] = None,  # (b, n, 1)  float, timestamp of each point
        **other_attrs,
    ):
        self.xyz_w = xyz_w
        self.rgb = rgb
        self.normal_w = normal_w
        self.captured_z_direction_w = captured_z_direction_w
        self.captured_view_direction_w = captured_view_direction_w
        self.captured_pinhole_w = captured_pinhole_w
        self.captured_dps = captured_dps
        self.captured_dps_u_w = captured_dps_u_w
        self.captured_dps_v_w = captured_dps_v_w
        self.valid_mask = valid_mask
        self.feature = feature
        self.img_idxs = img_idxs
        self.timestamp = timestamp

        self.attr_names = [
            "xyz_w",
            "rgb",
            "captured_z_direction_w",
            "captured_pinhole_w",
            "captured_view_direction_w",
            "captured_dps",
            "captured_dps_u_w",
            "captured_dps_v_w",
            "normal_w",
            "valid_mask",
            "feature",
            "img_idxs",
            "timestamp",
        ]

        # NOTE: this is a hacky way to allow calling PointCloud(**dict).
        for tmp_k, tmp_v in other_attrs.items():
            if tmp_v is not None:
                assert (tmp_v.ndim == 3) and (tmp_v.shape[:2] == self.xyz_w.shape[:2]), (
                    f"{tmp_k=}, {tmp_v.shape=}, {self.xyz_w.shape=}"
                )
                setattr(self, tmp_k, tmp_v)

        other_attrs_names = list(other_attrs.keys())
        assert len(set(self.attr_names).intersection(set(other_attrs_names))) == 0, (
            f"{self.attr_names=}, {other_attrs_names=}"
        )
        self.attr_names += other_attrs_names

        self.included_point_at_inf = included_point_at_inf
        self.check_dim()

    @staticmethod
    def from_o3d_pcd(
        o3d_pcd: o3d.geometry.PointCloud,
    ) -> "PointCloud":
        """
        Return a point cloud object from o3d_pcd
        Args:
            o3d_pcd:
                (n,)
        Returns:
            (b=1, n)
        """
        xyz_w = torch.from_numpy(np.array(o3d_pcd.points)).float().unsqueeze(0)  # (1, n, 3)
        if o3d_pcd.has_normals():
            normals = torch.from_numpy(np.array(o3d_pcd.normals)).float().unsqueeze(0)  # (1, n, 3)
        else:
            normals = None
        if o3d_pcd.has_colors():
            colors = torch.from_numpy(np.array(o3d_pcd.colors)).float().unsqueeze(0)  # (1, n, 3)
            assert colors.ndim == 3
            assert colors.size(-1) == 3
        else:
            colors = None
        return PointCloud(
            xyz_w=xyz_w,  # (1, n, 3)
            rgb=colors,
            normal_w=normals,
        )

    def get_num_points(self) -> int:
        """
        get number of points (excluding point at inf)
        but including the invalid points.
        """
        if self.included_point_at_inf:
            return self.xyz_w.size(1) - 1
        else:
            return self.xyz_w.size(1)

    def get_num_valid_points(self, bidx: int) -> int:
        """
        get number of valid points (excluding point at inf)
        but including the invalid points.
        """
        if self.valid_mask is None:
            if self.included_point_at_inf:
                return self.xyz_w.size(1) - 1
            else:
                return self.xyz_w.size(1)
        else:
            return self.valid_mask[bidx, :].sum().detach().cpu().item()

    def to(self, device: torch.device) -> "PointCloud":
        for attr_name in self.attr_names:
            arr = getattr(self, attr_name, None)
            if arr is not None:
                setattr(self, attr_name, arr.to(device=device))
        return self

    def detach(self) -> "PointCloud":
        for attr_name in self.attr_names:
            arr = getattr(self, attr_name, None)
            if arr is not None:
                setattr(self, attr_name, arr.detach())
        return self

    def clone(self) -> "PointCloud":
        data_dict = dict()
        for attr_name in self.attr_names:
            arr = getattr(self, attr_name, None)
            if arr is not None:
                data_dict[attr_name] = arr.clone()
            else:
                data_dict[attr_name] = None
        data_dict["included_point_at_inf"] = self.included_point_at_inf
        return PointCloud(**data_dict)

    def drop_features(self, drop_normal: bool = True):
        """Drop features related to camera pose used to capture the point"""
        for attr_name in [
            "captured_z_direction_w",
            "captured_view_direction_w",
            "captured_pinhole_w",
            "captured_dps",
            "captured_dps_u_w",
            "captured_dps_v_w",
        ]:
            setattr(self, attr_name, None)

        if drop_normal:
            setattr(self, "normal_w", None)

    def check_dim(self):
        """Check all attributes have the same number of points."""
        n = self.xyz_w.size(1)
        for name in self.attr_names:
            arr = getattr(self, name, None)
            if arr is None:
                continue
            else:
                assert arr.size(1) == n, f"{name}.shape = {arr.shape}, num_points = {n}"

    def insert_point_at_inf(self):
        """
        insert a point representing inf at n=0
        """
        if self.included_point_at_inf:
            return

        b = self.xyz_w.size(0)
        # when using pr to find k points within a fixed distance of ray
        # use a far-away point (1e12) to represent that the point is not found
        # this far-away point will be replaced by a learned token in the model later on

        for name in self.attr_names:
            arr = getattr(self, name, None)
            if arr is None:
                continue
            ndim = arr.shape[2:]
            if name == "xyz_w":
                val = INF
            elif name == "valid_mask":
                val = 0
            else:
                val = 0

            arr_requires_grad = arr.requires_grad

            arr = torch.cat(
                (
                    (torch.ones(b, 1, *ndim, dtype=arr.dtype, device=arr.device) * val).to(dtype=arr.dtype),
                    arr,
                ),
                dim=1,
            )
            arr.requires_grad = arr_requires_grad
            setattr(self, name, arr)

        self.included_point_at_inf = True
        self.check_dim()

    def reset_point_at_inf(self):
        """
        reset the point representing inf at n=0
        """
        if not self.included_point_at_inf:
            return

        with torch.no_grad():
            for name in self.attr_names:
                arr = getattr(self, name, None)
                if arr is None:
                    continue

                if name == "xyz_w":
                    val = INF
                elif name == "valid_mask":
                    val = 0
                else:
                    val = 0

                arr[:, 0, :] = val
                setattr(self, name, arr)
        self.check_dim()

    def remove_point_at_inf(self):
        """
        remove the point representing inf at n=0
        """
        if not self.included_point_at_inf:
            return

        b = self.xyz_w.size(0)
        for name in self.attr_names:
            arr = getattr(self, name, None)
            if arr is None:
                continue
            arr_requires_grad = arr.requires_grad
            arr = arr[:, 1:]
            arr.requires_grad = arr_requires_grad
            setattr(self, name, arr)
        self.included_point_at_inf = False
        self.check_dim()

    def state_dict(self) -> T.Dict[str, T.Any]:
        """Returns a dictionary that can be saved or load."""
        to_save = dict()
        for name in self.attr_names:
            to_save[name] = getattr(self, name, None)
        to_save["included_point_at_inf"] = self.included_point_at_inf
        return to_save

    def load_state_dict(
        self,
        state_dict: T.Dict[str, T.Any],
    ):
        """Load the state dictionary."""
        for name in self.attr_names:
            setattr(self, name, state_dict.get(name, None))
        setattr(self, "included_point_at_inf", state_dict.get("included_point_at_inf", False))

    def extract_valid_attr(self, arr: torch.Tensor, bidx: int) -> T.Union[torch.Tensor, None]:
        """
        Args:
            arr:
                (b, n, *)
            bidx:
                batch index
        Returns:
            (n, dim)
        """

        if arr is None:
            return None

        if self.valid_mask is None:
            if not self.included_point_at_inf:
                return arr[bidx]  # (n, dim)
            else:
                return arr[bidx, 1:]  # (n, dim)
        else:
            if self.valid_mask.dtype == torch.bool:
                valid_mask = self.valid_mask
            else:
                valid_mask = self.valid_mask > 0.5
            if self.included_point_at_inf:
                assert (valid_mask[bidx, 0] < 0.5).all()
            arr = arr[bidx, valid_mask[bidx, :, 0]]  # (n, dim)
            return arr

    def extract_valid_point_cloud(self, bidx: int) -> "PointCloud":
        """
        Return a new PointCloud `(1, n)` that contains only the valid points.
        Args:
            bidx:

        Returns:
            new_point_cloud: (b=1, n)
        """
        data_dict = dict()
        for attr_name in self.attr_names:
            arr = getattr(self, attr_name, None)
            if arr is None:
                data_dict[attr_name] = None
            else:
                arr = self.extract_valid_attr(arr=arr, bidx=bidx).unsqueeze(0)
                data_dict[attr_name] = arr.clone()
        new_point_cloud = PointCloud(**data_dict)  # include_inf = False
        assert not new_point_cloud.included_point_at_inf
        return new_point_cloud

    def get_o3d_pcds(
        self,
        use_normal_for_color: bool = False,
        estimate_normal_if_not_exist: bool = False,
    ) -> T.List[o3d.geometry.PointCloud]:
        """Return a list of b o3d pcds containing xyz and rgb."""

        # if not self.included_point_at_inf:
        #     xyz_w = self.xyz_w.detach().cpu().numpy()
        #     rgb = self.rgb.detach().cpu().numpy() if self.rgb is not None else None
        #     normal_w = self.normal_w.detach().cpu().numpy() if self.normal_w is not None else None
        # else:
        #     xyz_w = self.xyz_w[:, 1:].detach().cpu().numpy()
        #     rgb = self.rgb[:, 1:].detach().cpu().numpy() if self.rgb is not None else None
        #     normal_w = self.normal_w[:, 1:].detach().cpu().numpy() if self.normal_w is not None else None

        o3d_pcds = []
        for i in range(self.xyz_w.size(0)):
            xyz_w = self.extract_valid_attr(
                arr=self.xyz_w,
                bidx=i,
            )  # (n, 3)
            if xyz_w is not None:
                xyz_w = xyz_w.detach().cpu().numpy()  # (n, 3)

            rgb = self.extract_valid_attr(
                arr=self.rgb,
                bidx=i,
            )  # (n, 3)
            if rgb is not None:
                rgb = rgb.detach().cpu().numpy()  # (n, 3)
            normal_w = self.extract_valid_attr(
                arr=self.normal_w,
                bidx=i,
            )  # (n, 3)
            if normal_w is not None:
                normal_w = normal_w.detach().cpu().numpy()  # (n, 3)

            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(xyz_w)  # (n, 3)

            # determine color
            if use_normal_for_color:
                if normal_w is not None:
                    colors = normal_w
                else:
                    colors = np.ones_like(xyz_w) * 0.5  # (n, 3)
            else:
                if rgb is not None:
                    colors = rgb
                elif normal_w is not None:
                    colors = (normal_w + 1) * 0.5
                else:
                    colors = np.ones_like(xyz_w) * 0.5  # (n, 3)

            o3d_pcd.colors = o3d.utility.Vector3dVector(colors)  # (n, 3)

            if normal_w is not None:
                o3d_pcd.normals = o3d.utility.Vector3dVector(normal_w)  # (n, 3)

            if estimate_normal_if_not_exist and normal_w is None:
                o3d_pcd.estimate_normals()  # default parameter (number of neighbor points = 30), random direction

            o3d_pcds.append(o3d_pcd)
        return o3d_pcds

    def get_mesh(
        self,
        bidx: int,
        method: str,
        recompute_normal: bool,
        alpha: float = 0.03,
        ball_radii: T.List[float] = (0.005, 0.01, 0.02, 0.04),
        poisson_depth: int = 8,
        poisson_density_quantile_th: float = None,
    ) -> "Mesh":
        """
        Reconstruct a mesh from the point cloud.

        Args:
            bidx:
                the batch index.
            method:
                'poisson': use Poisson surface reconstruction
                'alpha': use Alpha shapes
                'ball': use Ball pivoting
            recompute_normal:
                whether to recompute the vertex normal (assuming no normal at the points)
            alpha:
                alpha value used by alpha shapes
            ball_radii:
                radii used by ball pivot.
                radii of the individual balls that are pivoted on the point cloud.
                When the ball touches three points, a triangle is created.
            poisson_depth:
                depth used by poisson reconstruction.
                A higher depth value means a mesh with more details.
            poisson_density_quantile_th:
                [0, 1] the min density for a vertices to be considered as valid
                when performing poisson reconstruction.  If not None, we
                will remove low density vertices from mesh.
                None: no vertices will be removed.
        """

        o3d_pcd: o3d.geometry.PointCloud = self.get_o3d_pcds(
            estimate_normal_if_not_exist=True,
        )[bidx]

        if recompute_normal:
            o3d_pcd.estimate_normals()

        if method == "alpha":
            o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(o3d_pcd, alpha)
        elif method == "ball":
            o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                o3d_pcd,
                o3d.utility.DoubleVector(ball_radii),
            )
        elif method == "poisson":
            o3d_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                o3d_pcd, depth=poisson_depth
            )
            if poisson_density_quantile_th is not None and poisson_density_quantile_th > 0:
                vertices_to_remove = densities < np.quantile(densities, poisson_density_quantile_th)
                o3d_mesh.remove_vertices_by_mask(vertices_to_remove)
        else:
            raise NotImplementedError

        return Mesh(
            mesh=o3d_mesh,
            scale=None,
            center_w=None,
            preprocess_mesh=False,
        )

    @staticmethod
    def cat(point_clouds: T.List["PointCloud"], dim: int) -> "PointCloud":
        """concatenate at `dim`."""

        if dim == 0:
            # check all point clouds have the same included_point_at_inf and n
            for i in range(1, len(point_clouds)):
                assert point_clouds[i].included_point_at_inf == point_clouds[0].included_point_at_inf
                assert point_clouds[i].xyz_w.size(1) == point_clouds[0].xyz_w.size(1)

            out_dict = dict()
            for name in point_clouds[0].attr_names:
                arr = [getattr(p, name, None) for p in point_clouds]
                if None in arr:
                    out_dict[name] = None
                else:
                    out_dict[name] = torch.cat(arr, dim=dim)

            if len(point_clouds) > 0:
                out_dict["included_point_at_inf"] = point_clouds[0].included_point_at_inf
            else:
                out_dict["included_point_at_inf"] = False
        elif dim == 1:
            out_dict = dict()
            for name in point_clouds[0].attr_names:
                arrs = []
                for i in range(len(point_clouds)):
                    arr = getattr(point_clouds[i], name, None)
                    if arr is not None:
                        if point_clouds[i].included_point_at_inf and i > 0:
                            start_idx = 1  # remove point at inf for i >= 1
                        else:
                            start_idx = 0
                        arr = arr[:, start_idx:]
                    arrs.append(arr)

                if None in arrs:
                    out_dict[name] = None
                else:
                    out_dict[name] = torch.cat(arrs, dim=dim)

            if len(point_clouds) > 0:
                out_dict["included_point_at_inf"] = point_clouds[0].included_point_at_inf
            else:
                out_dict["included_point_at_inf"] = False
        else:
            raise NotImplementedError
        return PointCloud(**out_dict)

    def random_sample(self, *, n: int, rng: int | np.random.Generator | None = None):
        """This function randomly samples the existing point cloud and return a new pcd."""
        rng = utils.get_np_rng(rng)

        assert (self.xyz_w.ndim == 3) and (self.xyz_w.shape[2] == 3), f"{self.xyz_w.shape=}"
        select_mask = torch.zeros(self.xyz_w.shape[:2], dtype=bool, device=self.xyz_w.device)

        bs, n_all, _ = self.xyz_w.shape
        assert n <= n_all, f"{n=}, {n_all=}"

        for tmp_i in range(self.xyz_w.shape[0]):
            tmp_select_idxs = rng.choice(n_all, n, replace=False)
            select_mask[tmp_i, tmp_select_idxs] = True

        new_pcd_dict = {}
        for attr_name in self.attr_names:
            arr = getattr(self, attr_name, None)
            if arr is not None:
                arr = arr[select_mask, :].reshape((bs, n, -1))
            new_pcd_dict[attr_name] = arr

        new_pcd = PointCloud(included_point_at_inf=self.included_point_at_inf, **new_pcd_dict)
        return new_pcd

    def get_voxel_downsampling_cell_width_from_resolution(
        self,
        *,
        resolution: int,
        bidx: int = 0,
    ):
        """Compute voxel's cell width based on the point cloud size and the target resolution.

        Procedure:
        - compute point cloud size on each axis
        - compute the cell width on each axis with given resolution
        - choose the minimum cell width from all 3 axises
        """
        xyz_w = self.extract_valid_attr(
            arr=self.xyz_w,
            bidx=bidx,
        ).unsqueeze(0)  # (b=1, n, 3)

        grid_to = xyz_w.max(dim=-2, keepdim=True)[0] + 1.0e-3  # (b, 1, 3)
        grid_from = xyz_w.min(dim=-2, keepdim=True)[0] - 1.0e-3  # (b, 1, 3)
        grid_width = grid_to - grid_from  # (b, 1, 3)
        grid_cell_width = grid_width / resolution  # (b, 1, 3)
        cell_width = float(torch.min(grid_cell_width).cpu())
        return cell_width

    @linalg_utils.disable_tf32_and_autocast()
    def voxel_downsampling(
        self,
        cell_width: float,
        sigma: float = 0.5,
        drop_features: bool = True,
        bidx: int = 0,
        mode: str = "avg",
        discretize_method: str = "min_max",
        min_point_count: int = 0,
        ref_val: torch.Tensor = None,
    ) -> "PointCloud":
        """
        Voxel downsampling uses a voxel grid to uniformly downsample the input point cloud.

        Procedure:
        - Points are discretized into voxels.
        - Each occupied voxel generates exactly one point by
        (avg)averaging all points inside or (random) randomly selecting a point

        Args:
            cell_width:
                the width of each grid cell.
                If <0, return self (do nothing)
            sigma:
                the sigma used in computing the gaussian weight
            mode:
                'avg': average color, normal in each cell
                'random': randomly select a point in each cell
            discretize_method:
                'min_max': use min xyz_w and max_xyz_w to determine cell idx
                'origin': use origin as reference point
            min_point_count:
                only consider a cell if it contains more than `min_point_count` number of points

        Returns:

        Notes:
            the function is currently implemented timeless.
            It only do voxel downsampling within xyz cube, not xyzt cube.
        """
        assert self.timestamp is None, "time has not be implemented yet"

        if mode == "avg":
            return self.voxel_downsampling_avg(
                cell_width=cell_width,
                sigma=sigma,
                drop_features=drop_features,
                bidx=bidx,
                discretize_method=discretize_method,
                min_point_count=min_point_count,
            )
        elif mode == "random":
            return self.voxel_downsampling_random(
                cell_width=cell_width,
                drop_features=drop_features,
                bidx=bidx,
            )
        elif mode == "min":
            return self.voxel_downsampling_min(
                cell_width=cell_width,
                ref_val=ref_val,
                drop_features=drop_features,
                bidx=bidx,
            )
        else:
            raise NotImplementedError

    @linalg_utils.disable_tf32_and_autocast()
    def voxel_downsampling_avg(
        self,
        cell_width: float,
        sigma: float = 0.5,
        drop_features: bool = True,
        bidx: int = 0,
        discretize_method: str = "min_max",
        min_point_count: int = 0,
    ) -> "PointCloud":
        """
        Voxel downsampling uses a voxel grid to uniformly downsample the input point cloud.
        Aggregation is performed by averaging.

        Procedure:
        - Points are discretized into voxels.
        - Each occupied voxel generates exactly one point by
          averaging all points inside

        Args:
            cell_width:
                the width of each grid cell.
                If <0, return self (do nothing)
            sigma:
                the sigma used in computing the gaussian weight
            min_point_count:
                only consider a cell if it contains more than `min_point_count` number of points

        Returns:
        """
        assert self.timestamp is None, "time has not be implemented yet"

        if cell_width < 0:
            return self

        print(f"voxel downsampling started, original num points = {self.xyz_w.size(1)}")

        sigma = sigma * cell_width

        xyz_w = self.extract_valid_attr(
            arr=self.xyz_w,
            bidx=bidx,
        ).unsqueeze(0)  # (b=1, n, 3)

        assert xyz_w.size(0) == 1

        # if self.included_point_at_inf:
        #     start_idx = 1
        # else:
        #     start_idx = 0

        # construct grid
        if discretize_method == "min_max":
            grid_to = xyz_w.max(dim=-2, keepdim=True)[0] + 1.0e-3  # (b, 1, 3)
            grid_from = xyz_w.min(dim=-2, keepdim=True)[0] - 1.0e-3  # (b, 1, 3)
            grid_width = grid_to - grid_from  # (b, 1, 3)
            grid_size = torch.ceil(grid_width / cell_width).long()  # (b, 1, 3)
            cell_width = grid_width / grid_size.float()  # (b, 1, 3)

            # discretize to cell idx
            subidxs = torch.floor((xyz_w - grid_from) / cell_width).long()  # (b, n, 3)
            inds = (
                subidxs[..., 2]
                + subidxs[..., 1] * grid_size[..., 2]
                + subidxs[..., 0] * (grid_size[..., 1] * grid_size[..., 2])
            )  # (b=1, n)
            # _, idxs, counts = torch.unique(inds.squeeze(0), return_inverse=True, return_counts=True, dim=0)  # (n,)
            _, idxs, counts = torch.unique(inds.squeeze(0), return_inverse=True, return_counts=True)  # (n,)
            # idxs: (n,)
            # counts: (num_occupied_cells,)

        elif discretize_method == "origin":
            subidxs = torch.floor(xyz_w / cell_width).long()  # (b=1, n, 3)
            # use unique to index
            _, idxs, counts = torch.unique(subidxs.squeeze(0), return_inverse=True, return_counts=True, dim=0)  # (n,)
            # idxs: (n,)
            # counts: (num_occupied_cells,)
        else:
            raise NotImplementedError

        # remap ind to unique index (remove unused grid_cells)
        # xyz_w = self.xyz_w[b, start_idx:]
        # _, idxs, counts = torch.unique(inds[b], return_inverse=True, return_counts=True)
        # idxs: (n,)
        # counts: (num_occupied_cells,)
        num_occupied_cells = counts.size(0)

        # average xyz to get new xyz (so implicitly weighted by sample density)
        xyz_w_mean = torch.zeros(
            num_occupied_cells, 3, dtype=xyz_w.dtype, device=xyz_w.device
        )  # (num_occupied_cells, 3)
        xyz_w_mean.scatter_reduce_(
            dim=0,
            index=idxs.unsqueeze(-1).expand(-1, 3),  # (n, 3)
            src=xyz_w.squeeze(0),  # (n, 3)
            reduce="mean",
            include_self=False,  # important, do not want to include 0 and the count
        )
        # xyz_w_mean = torch_scatter.scatter_mean(
        #     xyz_w.squeeze(0),  # (n, 3)
        #     index=idxs.unsqueeze(-1),  # (n, 1)
        #     dim=-2,
        # )  # (num_occupied_cells, 3)

        xyz_w_mean_expanded = xyz_w_mean[idxs]  # (n, 3)
        squared_dists = torch.sum((xyz_w.squeeze(0) - xyz_w_mean_expanded) ** 2, dim=-1)  # (n,)
        weights = torch.exp(-1 * squared_dists / (2 * sigma**2))  # (n,)

        weights_sum = torch.zeros(
            num_occupied_cells, dtype=weights.dtype, device=weights.device
        )  # (num_occupied_cells,)
        weights_sum.scatter_reduce_(
            dim=0,
            index=idxs,  # (n,)
            src=weights,  # (n,)
            reduce="sum",
            include_self=False,
        )
        # weights_sum = torch_scatter.scatter_sum(
        #     weights,  # (n,)
        #     index=idxs,  # (n,)
        #     dim=0,
        # )  # (num_occupied_cells,)
        normalized_weights = weights / weights_sum[idxs]  # (n,)
        normalized_weights = normalized_weights.unsqueeze(-1)  # (n, 1)

        # weighted average each feature
        out_dict = dict()
        out_dict["xyz_w"] = xyz_w_mean.unsqueeze(0)  # (1, num_occupied_cells, 3)

        for attr_name in self.attr_names:
            if attr_name == "xyz_w":
                continue

            if attr_name == "img_idxs":
                out_dict[attr_name] = None
                continue

            if not drop_features or attr_name in {"rgb", "normal_w", "feature"}:
                arr = getattr(self, attr_name, None)
                if arr is None:
                    out_dict[attr_name] = None
                    continue

                arr = (
                    self.extract_valid_attr(
                        arr=arr,
                        bidx=bidx,
                    )
                    * normalized_weights
                )  # (n, dim)

                _arr = torch.zeros(
                    num_occupied_cells, arr.size(-1), dtype=arr.dtype, device=arr.device
                )  # (num_occupied_cells,)
                _arr.scatter_reduce_(
                    dim=0,
                    index=idxs.unsqueeze(-1).expand(-1, arr.size(-1)),  # (n, dim)
                    src=arr,  # (n, dim)
                    reduce="sum",
                    include_self=False,
                )  # (num_occupied_cells, dim)
                arr = _arr
                # arr = torch_scatter.scatter_sum(
                #     arr,  # (n, dim)
                #     index=idxs.unsqueeze(-1),  # (n, 1)
                #     dim=0,
                # )  # (num_occupied_cells, dim)

                # make sure direction are unit norm
                if attr_name in {"normal_w", "captured_z_direction_w", "captured_view_direction_w"}:
                    arr = torch.nn.functional.normalize(arr, p=2.0, dim=-1)  # (num_occupied_cells, dim)

                out_dict[attr_name] = arr.unsqueeze(0)  # (1, num_occupied_cells, dim)
            else:
                out_dict[attr_name] = None

        if min_point_count > 0:
            mask = counts >= min_point_count  # (num_occupied_cells,)
            for key in out_dict:
                if out_dict[key] is None:
                    continue
                out_dict[key] = out_dict[key][:, mask]

        point_cloud = PointCloud(**out_dict)  # included_point_at_inf = False
        # point_cloud = PointCloud.cat(all_point_clouds, dim=0)
        print(
            f"voxel downsampling finished, num points = {point_cloud.xyz_w.size(1)} "
            f"({point_cloud.xyz_w.size(1) / self.xyz_w.size(1) * 100.0:.2f}%)"
        )
        return point_cloud

    @linalg_utils.disable_tf32_and_autocast()
    def voxel_downsampling_random(
        self,
        cell_width: float,
        drop_features: bool = True,
        discretize_method: str = "min_max",
        bidx: int = 0,
        printout: bool = False,
        min_point_count: int = 0,
    ) -> "PointCloud":
        """
        Voxel downsampling uses a voxel grid to uniformly downsample the input point cloud.
        Aggregation is performed by randomly selecting a point in each cell.

        Procedure:
        - Points are discretized into voxels.
        - Each occupied voxel generates exactly one point by
          randomly selecting a point in each cell.

        Args:
            cell_width:
                the width of each grid cell.
                If <0, return self (do nothing)
            discretize_method:
                'min_max': use min xyz_w and max_xyz_w to determine cell idx
                'origin': use origin as reference point
            min_point_count:
                only consider a cell if it contains more than `min_point_count` number of points

        Returns:
            point_cloud:
                (b=1, m)
        """
        assert self.timestamp is None, "time has not be implemented yet"

        if cell_width < 0:
            return self

        if printout:
            print(f"voxel downsampling started, original num points = {self.xyz_w.size(1)}")

        xyz_w = self.extract_valid_attr(
            arr=self.xyz_w,
            bidx=bidx,
        ).unsqueeze(0)  # (b=1, n, 3)
        n = xyz_w.size(1)

        # construct grid
        if discretize_method == "min_max":
            grid_to = xyz_w.max(dim=-2, keepdim=True)[0] + 1.0e-3  # (b=1, 1, 3)
            grid_from = xyz_w.min(dim=-2, keepdim=True)[0] - 1.0e-3  # (b=1, 1, 3)
            grid_width = grid_to - grid_from  # (b=1, 1, 3)
            grid_size = torch.ceil(grid_width / cell_width).long()  # (b=1, 1, 3)
            cell_width = grid_width / grid_size.float()  # (b=1, 1, 3)

            # discretize to cell idx
            subidxs = torch.floor((xyz_w - grid_from) / cell_width).long()  # (b=1, n, 3)
            cell_inds = (
                subidxs[..., 2]
                + subidxs[..., 1] * grid_size[..., 2]
                + subidxs[..., 0] * (grid_size[..., 1] * grid_size[..., 2])
            )  # (b=1, n)
            cell_inds = cell_inds.squeeze(0)  # (n,)
        elif discretize_method == "origin":
            subidxs = torch.floor(xyz_w / cell_width).long()  # (b=1, n, 3)
            # use unique to index
            _, cell_inds = torch.unique(subidxs.squeeze(0), return_inverse=True, dim=0)  # (n,)
        else:
            raise NotImplementedError

        # random shuffle (to randomize selection probability of each point in a cell)
        ridxs = torch.randperm(n, device=cell_inds.device)  # (n,)
        cell_inds = cell_inds[ridxs]  # (n,)
        pidxs = torch.arange(n, device=cell_inds.device)  # (n,) index in xyz_w
        pidxs = pidxs[ridxs]  # (n,)

        # stable sort to gather points in the same cell
        cell_inds, ii = torch.sort(cell_inds, stable=True)  # (n,)
        pidxs = pidxs[ii]  # (n,)

        # for each cell, get the first point
        _, cc = torch.unique_consecutive(
            input=cell_inds,  # (n,)
            return_inverse=False,
            return_counts=True,
        )  # (num_occupied_cells,)

        ii = torch.cat(
            [
                torch.zeros(1, dtype=cc.dtype, device=cc.device),
                torch.cumsum(cc[:-1], dim=0),
            ],
            dim=0,
        )  # (num_occupied_cells,)
        pidxs = pidxs[ii]  # (num_occupied_cells,)

        out_dict = dict()
        out_dict["xyz_w"] = xyz_w[:, pidxs]  # (1, num_occupied_cells, 3)

        for attr_name in self.attr_names:
            if attr_name == "xyz_w":
                continue

            if not drop_features or attr_name in {"rgb", "normal_w", "feature"}:
                arr = getattr(self, attr_name, None)
                if arr is None:
                    out_dict[attr_name] = None
                    continue

                arr = self.extract_valid_attr(
                    arr=arr,
                    bidx=bidx,
                )  # (n, dim)
                arr = arr[pidxs]  # (num_occupied_cells, dim)
                out_dict[attr_name] = arr.unsqueeze(0)  # (1, num_occupied_cells, dim)
            else:
                out_dict[attr_name] = None

        if min_point_count > 0:
            mask = cc >= min_point_count  # (num_occupied_cells,)
            for key in out_dict:
                if out_dict[key] is None:
                    continue
                out_dict[key] = out_dict[key][:, mask]

        point_cloud = PointCloud(**out_dict)  # included_point_at_inf = False

        if printout:
            print(
                f"voxel downsampling finished, num points = {point_cloud.xyz_w.size(1)} "
                f"({point_cloud.xyz_w.size(1) / self.xyz_w.size(1) * 100.0:.2f}%)"
            )
        return point_cloud

    @linalg_utils.disable_tf32_and_autocast()
    def voxel_downsampling_min(
        self,
        cell_width: float,
        ref_val: torch.Tensor,
        drop_features: bool = True,
        discretize_method: str = "min_max",
        bidx: int = 0,
        printout: bool = False,
        min_point_count: int = 0,
    ) -> "PointCloud":
        """
        Voxel downsampling uses a voxel grid to uniformly downsample the input point cloud.
        Aggregation is performed by selecting the point with min ref_val in each cell.

        Args:
            cell_width:
                the width of each grid cell.
                If <0, return self (do nothing)
            ref_val:
                (b, n)
            discretize_method:
                'min_max': use min xyz_w and max_xyz_w to determine cell idx
                'origin': use origin as reference point
            min_point_count:
                only consider a cell if it contains more than `min_point_count` number of points

        Returns:
            point_cloud:
                (b=1, m)
        """
        assert self.timestamp is None, "time has not be implemented yet"

        if cell_width < 0:
            return self

        if printout:
            print(f"voxel downsampling started, original num points = {self.xyz_w.size(1)}")

        assert ref_val is not None

        xyz_w = self.extract_valid_attr(
            arr=self.xyz_w,
            bidx=bidx,
        ).unsqueeze(0)  # (b=1, n, 3)
        n = xyz_w.size(1)

        ref_val = self.extract_valid_attr(
            arr=ref_val,
            bidx=bidx,
        ).unsqueeze(0)  # (b=1, n)

        assert ref_val.size(1) == n

        # construct grid
        if discretize_method == "min_max":
            grid_to = xyz_w.max(dim=-2, keepdim=True)[0] + 1.0e-3  # (b=1, 1, 3)
            grid_from = xyz_w.min(dim=-2, keepdim=True)[0] - 1.0e-3  # (b=1, 1, 3)
            grid_width = grid_to - grid_from  # (b=1, 1, 3)
            grid_size = torch.ceil(grid_width / cell_width).long()  # (b=1, 1, 3)
            cell_width = grid_width / grid_size.float()  # (b=1, 1, 3)

            # discretize to cell idx
            subidxs = torch.floor((xyz_w - grid_from) / cell_width).long()  # (b=1, n, 3)
            cell_inds = (
                subidxs[..., 2]
                + subidxs[..., 1] * grid_size[..., 2]
                + subidxs[..., 0] * (grid_size[..., 1] * grid_size[..., 2])
            )  # (b=1, n)
            cell_inds = cell_inds.squeeze(0)  # (n,)
        elif discretize_method == "origin":
            subidxs = torch.floor(xyz_w / cell_width).long()  # (b=1, n, 3)
            # use unique to index
            _, cell_inds = torch.unique(subidxs.squeeze(0), return_inverse=True, dim=0)  # (n,)
        else:
            raise NotImplementedError

        # first we sort based on ref_val
        _, ridxs = torch.sort(ref_val[0])  # (n,)
        cell_inds = cell_inds[ridxs]  # (n,)
        pidxs = torch.arange(n, device=cell_inds.device)  # (n,) index in xyz_w
        pidxs = pidxs[ridxs]  # (n,)

        # then, stable sort to gather points in the same cell
        cell_inds, ii = torch.sort(cell_inds, stable=True)  # (n,)
        pidxs = pidxs[ii]  # (n,)

        # for each cell, get the first point
        _, cc = torch.unique_consecutive(
            input=cell_inds,  # (n,)
            return_inverse=False,
            return_counts=True,
        )  # (num_occupied_cells,)

        ii = torch.cat(
            [
                torch.zeros(1, dtype=cc.dtype, device=cc.device),
                torch.cumsum(cc[:-1], dim=0),
            ],
            dim=0,
        )  # (num_occupied_cells,)
        pidxs = pidxs[ii]  # (num_occupied_cells,)

        out_dict = dict()
        out_dict["xyz_w"] = xyz_w[:, pidxs]  # (1, num_occupied_cells, 3)

        for attr_name in self.attr_names:
            if attr_name == "xyz_w":
                continue

            if not drop_features or attr_name in {"rgb", "normal_w", "feature"}:
                arr = getattr(self, attr_name, None)
                if arr is None:
                    out_dict[attr_name] = None
                    continue

                arr = self.extract_valid_attr(
                    arr=arr,
                    bidx=bidx,
                )  # (n, dim)
                arr = arr[pidxs]  # (num_occupied_cells, dim)
                out_dict[attr_name] = arr.unsqueeze(0)  # (1, num_occupied_cells, dim)
            else:
                out_dict[attr_name] = None

        if min_point_count > 0:
            mask = cc >= min_point_count  # (num_occupied_cells,)
            for key in out_dict:
                if out_dict[key] is None:
                    continue
                out_dict[key] = out_dict[key][:, mask]

        point_cloud = PointCloud(**out_dict)  # included_point_at_inf = False

        if printout:
            print(
                f"voxel downsampling finished, num points = {point_cloud.xyz_w.size(1)} "
                f"({point_cloud.xyz_w.size(1) / self.xyz_w.size(1) * 100.0:.2f}%)"
            )
        return point_cloud

    def remove_outlier(
        self,
        radius: float,
        min_num_points_in_radius: int,
        printout: bool = False,
    ) -> "PointCloud":
        """
        Remove the outlier points in the point cloud:
        1. removes points that have few neighbors in a given sphere around them.

        Args:
            radius:
                the radius of the sphere
            min_num_points_in_radius:
                minimum number of points within the sphere to consider a point as inlier

        Returns:
            self, with the valid_mask modified
        """
        assert self.timestamp is None, "time has not be implemented yet"

        if radius is None or radius < 0 or min_num_points_in_radius is None or min_num_points_in_radius < 0:
            return self

        # make sure valid mask is not None
        self.realize_valid_mask()

        if printout:
            print("Removing outlier points:", flush=True)
        # we use the raw xyz_w instead of the o3d_pcd, so the index mapping is easy
        for ib in range(self.xyz_w.size(0)):
            xyz_w = self.xyz_w[ib]  # (n, 3)
            if self.included_point_at_inf:
                xyz_w = xyz_w[1:]  # (n, 3)
                idx_offset = 1
            else:
                idx_offset = 0

            xyz_w = xyz_w.detach().cpu().numpy()
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(xyz_w)  # (n, 3)

            cleaned_o3d_pcd, valid_idxs = o3d_pcd.remove_radius_outlier(
                nb_points=min_num_points_in_radius, radius=radius
            )  # valid_idxs contains the list of valid index in o3d_pcd

            invalid_mask = torch.ones(self.xyz_w.size(1), dtype=torch.bool)
            for idx in valid_idxs:
                invalid_mask[idx + idx_offset] = 0
            invalid_mask = invalid_mask.to(device=self.valid_mask.device)

            # mark the valid_mask
            self.valid_mask[ib, invalid_mask, :] = 0

            if printout:
                print(
                    f"  ({ib}): removed {xyz_w.shape[0] - len(valid_idxs)} / {xyz_w.shape[0]}"
                    f" = {(xyz_w.shape[0] - len(valid_idxs)) / xyz_w.shape[0] * 100.0:.2f}% points",
                    flush=True,
                )

        return self

    def save(
        self,
        output_dir: str,
        overwrite: bool = False,
        save_ply: bool = True,
        save_pt: bool = True,
        remove_outlier: bool = False,
        nb_neighbors: int = 3,
        std_ratio: float = 2.0,
    ):
        """Save the point cloud as a ply file and a npz."""
        if os.path.exists(output_dir) and not overwrite:
            raise RuntimeError
        os.makedirs(output_dir, exist_ok=True)

        # o3d point cloud
        if save_ply:
            o3d_pcds = self.get_o3d_pcds()
            for i in range(self.xyz_w.size(0)):
                if remove_outlier:
                    o3d_pcd, _ = o3d_pcds[i].remove_statistical_outlier(
                        nb_neighbors=nb_neighbors,
                        std_ratio=std_ratio,
                    )
                else:
                    o3d_pcd = o3d_pcds[i]
                filename = os.path.join(output_dir, f"pcd_{i}.ply")
                o3d.io.write_point_cloud(
                    filename=filename,
                    pointcloud=o3d_pcd,
                )

        # pt
        if save_pt:
            filename = os.path.join(output_dir, f"state_dict.pt")
            state_dict = self.state_dict()
            torch.save(state_dict, filename)

    def save_as_gaussians(
        self,
        out_dir: str,
        point_radius: T.Union[float, torch.Tensor],
        opacity: float,
        use_2d_gaussian: bool,
        color_mode: str = "rgb",
    ):
        """
        Save individual point cloud as 2d/3d gaussians.

        Args:
            out_dir:
                the folder to save the ply files
            point_radius:
                float or (b,) or (b, n).  The std of the gaussians
            use_2d_gaussian:
                whether to use 2d gaussians.  need normal_w.
            color_mode:
                'rgb': use the rgb in the point cloud
                (r, g, b):  uniform color

        Returns:
             (b,) filenames of the saved gaussian ply
        """

        b, n, _3 = self.xyz_w.shape

        if color_mode == "rgb":
            rgb = self.rgb if self.rgb is not None else None
        elif isinstance(color_mode, (tuple, list, np.ndarray)) and len(color_mode) == 3:
            rgb = torch.zeros_like(self.xyz_w)  # (b, n, 3)
            rgb[..., 0] = color_mode[0]
            rgb[..., 1] = color_mode[1]
            rgb[..., 2] = color_mode[2]
        else:
            raise NotImplementedError

        os.makedirs(out_dir, exist_ok=True)

        filenames = []
        for i in range(b):
            gaussians = gs_utils.construct_gaussians_from_point_cloud(
                point_radius=point_radius,
                xyz_w=self.xyz_w[i],
                rgb=rgb[i],
                normal_w=self.normal_w[i] if self.normal_w is not None else None,
                opacity=opacity,
                use_2d_gaussian=use_2d_gaussian,
            )
            filename = os.path.join(out_dir, f"{i:05d}.ply")
            gaussians.save_ply(filename)
            filenames.append(filename)

        return filenames

    def save_as_npbgpp(
        self,
        filenames: T.List[str],
        overwrite: bool = False,
    ):
        """Save the ib-th point cloud as a ply file."""
        assert len(filenames) == self.xyz_w.size(0)
        for filename in filenames:
            if os.path.exists(filename) and not overwrite:
                raise RuntimeError

        # o3d point cloud
        o3d_pcds = self.get_o3d_pcds()
        for i in range(self.xyz_w.size(0)):
            o3d.io.write_point_cloud(
                filename=filenames[i],
                pointcloud=o3d_pcds[i],
            )

    def realize_valid_mask(self):
        if self.valid_mask is not None:
            return
        b, n, _dim = self.xyz_w.shape
        self.valid_mask = torch.ones(b, n, 1, dtype=torch.bool, device=self.xyz_w.device)
        if self.included_point_at_inf:
            self.valid_mask[:, 0] = 0

    def reset_valid_mask(self):
        """set all points as valid"""
        if self.valid_mask is None:
            return
        b, n, _dim = self.xyz_w.shape
        self.valid_mask = torch.ones(b, n, 1, dtype=torch.bool, device=self.xyz_w.device)
        if self.included_point_at_inf:
            self.valid_mask[:, 0] = 0

    @linalg_utils.disable_tf32_and_autocast()
    def rasterize_surfel(
        self,
        camera: "Camera",
        point_size: float = 1.0,
        default_rgb: T.List[float] = (0.5, 0.5, 0.5),
        render_normal_map: bool = True,
        rgb_shading_mode: str = "raw",
        light_direction_w: T.Optional[T.List[float]] = None,
        directional_light_weight: float = 0.5,
        diffuse_light_weight: float = 0.5,
        # o3d_normal_radius: float = 0.1,
        # o3d_normal_max_nn: int = 30,
    ) -> "RGBDImage":
        """
        Render the point cloud using surfel rasterization.

        Args:
            camera:
                camera (b, q)
            point_size:
                size of the points
            default_rgb:
                the color of the points when `self.rgb` is None
            render_normal_map:
                whether to rasterize normal_w (as rgb color of points).
                If `self.normal_w` is None, use o3d to estimate with default
                parameters (max_nn = 30).
            rgb_shading_mode:
                'raw': use the rgb values, this is the same as uniform lighting
                'directional': assume directional light comes from the camera center
                'half': 0.5 uniform + 0.5 directional
                'given': the light direction is given by `light_direction`
            light_direction_w:
                (3,) the light direction
            light_direction_ratio:
                directional light ratio
            # o3d_normal_radius:
            #     search radius (in meter) to use in o3d vertex normal estimation
            # o3d_normal_max_nn:
            #     max number of neighboring points to use in o3d vertex normal estimation

        Returns:
        """

        if isinstance(default_rgb, (int, float)):
            default_rgb = [float(default_rgb)] * 3
        assert len(default_rgb) == 3

        b = camera.H_c2w.size(0)
        q = camera.H_c2w.size(1)
        assert b == self.xyz_w.size(0)

        # get o3d pcds
        estimate_normal = render_normal_map or rgb_shading_mode in {"directional", "half"}
        o3d_pcds = self.get_o3d_pcds(
            estimate_normal_if_not_exist=estimate_normal,
        )

        intrinsic = camera.intrinsic.detach().cpu().numpy()  # (b, q, 3, 3)
        H_w2c = camera.get_H_w2c().detach().cpu().numpy()  # (b, q, 4, 4)
        H_c2w = camera.H_c2w.detach().cpu().numpy()  # (b, q, 4, 4)

        # render each b
        all_imgs = []
        all_depths = []
        all_hit_maps = []
        all_normal_ws = []
        for ib in range(b):
            # rgb
            if o3d_pcds[ib].colors is None or np.asarray(o3d_pcds[ib].colors).shape[0] == 0:
                n = np.asarray(o3d_pcds[ib].points).shape[0]
                rgb = np.ones((n, 3))
                for c in range(3):
                    rgb[:, c] = default_rgb[c]
                o3d_pcds[ib].colors = o3d.utility.Vector3dVector(rgb)  # (n, 3)
            # rasterize rgb
            if rgb_shading_mode in {"raw", "uniform"}:
                # make sure o3d_pcd has no normal
                tmp_normal = np.asarray(o3d_pcds[ib].normals)
                o3d_pcds[ib].normals = o3d.utility.Vector3dVector(np.zeros((0, 3)))

                # debug
                # o3d_pcds[ib].colors = o3d.utility.Vector3dVector(
                #     (np.array(o3d_pcds[ib].colors) * 255).astype(np.uint8))
                # print(f'o3d_pcds[{ib}].colors: min: {np.min(np.asarray(o3d_pcds[ib].colors))} max: {np.max(np.asarray(o3d_pcds[ib].colors))}')
                # end debug

                out_dict = render.rasterize(
                    meshes=o3d_pcds[ib],
                    intrinsic_matrix=intrinsic[ib],  # (q, 3, 3)
                    extrinsic_matrices=H_w2c[ib],
                    width_px=camera.width_px,
                    height_px=camera.height_px,
                    get_point_cloud=False,
                    point_size=point_size,
                    light_on=False,
                )  # imgs: list of q images (h, w, 3), z_maps list of q depth map (h, w)

                # debug
                # print(
                #     f'out_dict["imgs"][0]: min: {np.min(out_dict["imgs"][0])} max: {np.max(out_dict["imgs"][0])}')
                # end debug

                # recover the original normal
                o3d_pcds[ib].normals = o3d.utility.Vector3dVector(tmp_normal)
            elif rgb_shading_mode in {"directional", "half", "given"}:
                assert o3d_pcds[ib].normals is not None or np.asarray(o3d_pcds[ib].normals).shape[0] > 0
                assert o3d_pcds[ib].colors is not None or np.asarray(o3d_pcds[ib].colors).shape[0] > 0

                if rgb_shading_mode == "given":
                    assert light_direction_w is not None
                    if isinstance(light_direction_w, (tuple, list)):
                        light_direction_w = np.array(light_direction_w)  # (3,)
                    light_direction_w = light_direction_w / (
                        np.linalg.norm(light_direction_w, ord=2, axis=-1, keepdims=True) + 1e-9
                    )

                imgs = []
                z_maps = []
                hit_maps = []
                rgb = np.copy(np.asarray(o3d_pcds[ib].colors))  # (n, 3)
                normal_w = np.array(o3d_pcds[ib].normals)  # (n, 3)
                normal_w = normal_w / (np.linalg.norm(normal_w, ord=2, axis=-1, keepdims=True) + 1e-9)

                for iq in range(q):
                    # shading (by treating rgb as albedo)
                    if rgb_shading_mode in {"directional", "half"}:
                        ld = camera.H_c2w[ib, iq, :3, 3].detach().cpu().numpy()
                    elif rgb_shading_mode == "given":
                        ld = light_direction_w  # (3,)
                    else:
                        raise NotImplementedError

                    ld_norm = (ld**2).sum()
                    if ld_norm > 1e-6:
                        ld = ld / ld_norm
                    else:
                        ld = np.zeros((3,))
                        ld[2] = 1
                    if rgb_shading_mode == "directional":
                        n_dot_l = np.abs(np.sum(normal_w * ld, axis=-1, keepdims=True))  # (n, 1)
                        colors = rgb * n_dot_l  # (n, 3)
                        o3d_pcds[ib].colors = o3d.utility.Vector3dVector(colors)  # (n, 3)
                    elif rgb_shading_mode in ["half", "given"]:
                        n_dot_l = np.abs(np.sum(normal_w * ld, axis=-1, keepdims=True))  # (n, 1)
                        colors = rgb * (directional_light_weight * n_dot_l + diffuse_light_weight)  # (n, 3)
                        o3d_pcds[ib].colors = o3d.utility.Vector3dVector(colors)  # (n, 3)
                    else:
                        raise NotImplementedError

                    # make sure o3d_pcd has no normal
                    tmp_normal = np.asarray(o3d_pcds[ib].normals)
                    o3d_pcds[ib].normals = o3d.utility.Vector3dVector(np.zeros((0, 3)))

                    out_dict = render.rasterize(
                        meshes=o3d_pcds[ib],
                        intrinsic_matrix=intrinsic[ib, iq],  # (3, 3)
                        extrinsic_matrices=H_w2c[ib, iq],
                        width_px=camera.width_px,
                        height_px=camera.height_px,
                        get_point_cloud=False,
                        point_size=point_size,
                    )  # imgs: list of q images (h, w, 3), z_maps list of q depth map (h, w)

                    # recover the original normal
                    o3d_pcds[ib].normals = o3d.utility.Vector3dVector(tmp_normal)

                    imgs.append(out_dict["imgs"][0])  # (h, w, 3)
                    z_maps.append(out_dict["z_maps"][0])  # (h, w)
                    hit_maps.append(out_dict["hit_maps"][0])  # (h, w)

                imgs = np.stack(imgs, axis=0)  # (q, h, w, 3)
                z_maps = np.stack(z_maps, axis=0)  # (q, h, w)
                hit_maps = np.stack(hit_maps, axis=0)  # (q, h, w)
                out_dict = dict(
                    imgs=imgs,
                    z_maps=z_maps,
                    hit_maps=hit_maps,
                )
            else:
                raise NotImplementedError

            imgs = np.stack(out_dict["imgs"], axis=0)  # (q, h, w, 3)
            depths = np.stack(out_dict["z_maps"], axis=0)  # (q, h, w)
            hit_maps = np.stack(out_dict["hit_maps"], axis=0)  # (q, h, w)
            all_imgs.append(imgs)
            all_depths.append(depths)
            all_hit_maps.append(hit_maps)

            # normal
            # note that since each normal vector can be flipped randomly,
            # we need to rasterize one image at a time to orient normal to
            # the camera center
            if render_normal_map:
                assert o3d_pcds[ib].normals is not None

                # # debug
                # tmp_dir = '/task_runtime/23234'
                # os.makedirs(tmp_dir, exist_ok=True)
                # filename = os.path.join(tmp_dir, f'{ib}.ply')
                # o3d.io.write_point_cloud(filename, o3d_pcds[ib])
                # # end debug

                out_imgs = []
                for iq in range(q):
                    # o3d_pcds[ib].orient_normals_to_align_with_direction(
                    #     H_c2w[ib, iq, :3, 3]
                    # )

                    normal_w = np.array(o3d_pcds[ib].normals)  # (n, 3)
                    assert normal_w.shape[0] > 0
                    normal_w = normal_w / (np.linalg.norm(normal_w, ord=2, axis=-1, keepdims=True) + 1e-9)

                    # align normal to point the opposite direction of the camera ray
                    ray_direction_w = np.array(o3d_pcds[ib].points) - np.reshape(H_c2w[ib, iq, :3, 3], (1, 3))  # (n, 3)
                    normal_w = normal_w * (-1 * np.sign(np.sum(normal_w * ray_direction_w, axis=-1, keepdims=True)))

                    # map normals to rgb ([-1, 1] -> [0, 1])
                    normal_w = (normal_w + 1) * 0.5  # (n, 3)
                    o3d_pcds[ib].colors = o3d.utility.Vector3dVector(normal_w)  # (n, 3)

                    # make sure o3d_pcd has no normal
                    tmp_normal = np.asarray(o3d_pcds[ib].normals)
                    o3d_pcds[ib].normals = o3d.utility.Vector3dVector(np.zeros((0, 3)))

                    out_dict = render.rasterize(
                        meshes=o3d_pcds[ib],
                        intrinsic_matrix=intrinsic[ib, iq],  # (3, 3)
                        extrinsic_matrices=H_w2c[ib, iq],  # (4, 4)
                        width_px=camera.width_px,
                        height_px=camera.height_px,
                        get_point_cloud=False,
                        point_size=point_size,
                    )  # imgs: list of q images (h, w, 3), z_maps list of q depth map (h, w)
                    out_imgs.append(out_dict["imgs"][0])

                    # recover the original normal
                    o3d_pcds[ib].normals = o3d.utility.Vector3dVector(tmp_normal)

                all_normal_ws.append(
                    np.stack(out_imgs, axis=0)  # (q, h, w, 3)
                )

        all_imgs = np.stack(all_imgs, axis=0)  # (b, q, h, w, 3)
        all_depths = np.stack(all_depths, axis=0)  # (b, q, h, w)
        all_hit_maps = np.stack(all_hit_maps, axis=0)  # (b, q, h, w)
        if len(all_normal_ws) > 0:
            all_normal_ws = np.stack(all_normal_ws, axis=0)  # (b, q, h, w, 3)
            # [0, 1] -> [-1, 1]
            all_normal_ws = (all_normal_ws - 0.5) * 2.0
            all_normal_ws = all_normal_ws / np.linalg.norm(all_normal_ws, ord=2, axis=-1, keepdims=True)
        else:
            all_normal_ws = None

        return RGBDImage(
            rgb=torch.from_numpy(all_imgs),
            depth=torch.from_numpy(all_depths),
            hit_map=torch.from_numpy(all_hit_maps),
            camera=camera,  # we choose not to deepcopy, watch out
            normal_w=torch.from_numpy(all_normal_ws) if all_normal_ws is not None else None,
        )

    @linalg_utils.disable_tf32_and_autocast()
    def silhouette_carving(
        self,
        hit_map: torch.Tensor,
        camera: "Camera",
        dilate: bool = True,
    ) -> "PointCloud":
        """
        Use the hit_map and the camera in the rgbd_image to invalid points.
        It marks the valid_mask to be False for point that should be carved
        without actually remove them.

        Args:
            hit_map:
                hit_map: (b, q, h, w)
            camera:
                (b, q), h, w
            clone:
                whether to create a new point cloud (new memory)


        Returns:
            a new point cloud  (b, n')
        """

        ori_include_inf = self.included_point_at_inf
        self.remove_point_at_inf()
        self.realize_valid_mask()

        b, n, _3 = self.xyz_w.shape
        _b, q, h, w = hit_map.shape
        bq = b * q
        assert b == 1

        # project onto the sensor
        uv_cs, _ = utils.pinhole_projection(
            xyz_w=self.xyz_w,  # (b, n)
            intrinsics=camera.intrinsic,  # (b, q)
            H_c2w=camera.H_c2w,  # (b, q)
            dim_b=1,
        )  # (b, q, n, 2), [0, w] [0, h]

        # convert to [0, 1]
        uv_cs[..., 0] = uv_cs[..., 0] / camera.width_px
        uv_cs[..., 1] = uv_cs[..., 1] / camera.height_px

        uv_cs = uv_cs.reshape(bq, n, 2)
        hit_map = hit_map.reshape(bq, h, w, 1)  # (bq, h, w, 1)

        # dilation the hit map so the boundary is more robust
        if dilate:
            import kornia

            hit_map = kornia.morphology.dilation(
                tensor=hit_map.permute(0, 3, 1, 2).float(),  # (bq, 1, h, w)
                kernel=torch.ones(3, 3, dtype=torch.float, device=hit_map.device),
            ).permute(0, 2, 3, 1)  # (bq, h, w, 1)
        else:
            hit_map = hit_map.float()

        miss_map = 1 - hit_map  # (bq, h, w, 1)
        # uv-sample to get the miss_map (we use miss map in case a pixel is not visible in some images)
        miss = utils.uv_sampling(
            uv=uv_cs,  # (bq, n, 2)
            feature_map=miss_map,  # (bq, h, w, 1)
            mode="bilinear",
            padding_mode="border",
        )  # (bq, n, 1)
        hit = miss <= 0.5  # (bq, n, 1)
        hit = hit.reshape(b, q, n)

        # if a point is not hit in any image, it is invalid
        valid_mask = hit.all(dim=1)  # (b, n)

        # combine valid_mask with the original valid_mask
        self.valid_mask = torch.logical_and(
            self.valid_mask,  # (b, n, 1)
            valid_mask.unsqueeze(-1),  # (b, n, 1)
        )

        # valid_mask = valid_mask[0]  # (n,)
        # data_dict = dict()
        # for attr_name in self.attr_names:
        #     arr = getattr(self, attr_name, None)
        #     if arr is None:
        #         data_dict[attr_name] = None
        #         continue
        #
        #     arr = arr[:, valid_mask].clone()
        #     data_dict[attr_name] = arr
        #
        # point_cloud = PointCloud(**data_dict)

        if ori_include_inf:
            self.insert_point_at_inf()

        return self


class Ray:
    def __init__(
        self,
        origins_w: torch.Tensor,  # (b, *m_shape, 3)
        directions_w: torch.Tensor,  # (b, *m_shape, 3)
    ):
        self.origins_w = origins_w
        self.directions_w = directions_w

    def to(self, device: torch.device) -> "Ray":
        for attr_name in ["origins_w", "directions_w"]:
            arr = getattr(self, attr_name, None)
            if arr is not None:
                setattr(self, attr_name, arr.to(device=device))
        return self

    def clone(self) -> "Ray":
        return Ray(
            origins_w=self.origins_w.clone(),
            directions_w=self.directions_w.clone(),
        )

    def reshape(self, *shape: T.List[int]):
        self.origins_w = self.origins_w.reshape(*shape)
        self.directions_w = self.directions_w.reshape(*shape)
        return self

    def chunk(self, chunks: int, dim: int = 0) -> T.List["Ray"]:
        """Return a"""
        origins_w_list = self.origins_w.chunk(chunks, dim)
        directions_w_list = self.directions_w.chunk(chunks, dim)
        rays = []
        for origins_w, directions_w in zip(origins_w_list, directions_w_list):
            ray = Ray(
                origins_w=origins_w,
                directions_w=directions_w,
            )
            rays.append(ray)
        return rays

    def random_perturb_direction(
        self,
        shift: T.Optional[float],
        angle: T.Optional[float],
    ):
        """
        Perturb the rays by randomly shifting the ray origin by [-shift, shift],
        and randomly rotating the ray direction with [-angle, angle] in degree.

        Args:
            shift:
                [-shift, shift]
            angle:
                [-angle, angle] in degrees
            rng:
                the random generator
        """

        if shift is not None and math.fabs(shift) > 1e-6:
            r_shifts = (torch.rand_like(self.origins_w) - 0.5) * 2.0 * shift  # [-shift, shift]
            self.origins_w = self.origins_w + r_shifts  # (b, *m_shape, 3)
        if angle is not None and math.fabs(angle) > 1e-3:
            r_angles = (
                (torch.rand_like(self.directions_w) - 0.5) * 2.0 * angle
            )  # (b, *m_shape, 3) [-angle, angle] in degree
            r_angles = r_angles.view(-1, 3)  # (b*m_shape, 3)
            Rs = torch.from_numpy(
                Rotation.from_euler("xyz", r_angles, degrees=True).as_matrix()
            )  # (bm, 3, 3) rotation matrix
            Rs = Rs.reshape(*(self.directions_w.shape[:-1]), 3, 3)  # (b, *m_shape, 3, 3)
            self.directions_w = linalg_utils.matmul(
                Rs,
                self.directions_w.unsqueeze(-1),
            ).squeeze(-1)  # (b, *m, 3)

    def uniform_sample_points(
        self,
        num_samples: int,
        t_min: T.Union[torch.Tensor, float],
        t_max: T.Union[torch.Tensor, float],
        stratified: bool,
        must_have_ts: torch.Tensor = None,
        add_mid_points: bool = False,
    ) -> torch.Tensor:
        """
        Uniformly sample points on the rays between min_t and max_t.

        Args:
            num_samples:
                number of samples to sample on a ray
            t_min:
                (b, *m) or float.  The minimum ray_t to start.
            t_max:
                (b, *m) or float.  The maximum ray_t to start.
            stratified:
                whether we should perturb a bit the sample within the bin
            must_have_ts:
                (b, *m, k), if not None, will insert these ts
            add_mid_points:
                whether to add mid points to sampled points, such that
                x[1] = (x[0] + x[2]) / 2 and so on.

        Returns:
            (b, *m, num_samples)  the sampled ray_t sorted from small to large
        """
        assert num_samples > 0
        b, *m_shape, _3 = self.origins_w.shape
        device = self.origins_w.device

        if isinstance(t_min, (float, int)):
            t_min = torch.ones(b, *m_shape, dtype=self.origins_w.dtype, device=device) * t_min

        if isinstance(t_max, (float, int)):
            t_max = torch.ones(b, *m_shape, dtype=self.origins_w.dtype, device=device) * t_max

        assert t_min.shape == (b, *m_shape)
        assert t_max.shape == (b, *m_shape)

        # create standard num_samples bins (then scale later)
        bin_edges = torch.linspace(0, 1, num_samples + 1, device=device)  # (num_samples+1,)
        bin_starts = bin_edges[..., :-1]  # (num_samples,)
        bin_ends = bin_edges[..., 1:]  # (num_samples,)
        bin_centers = (bin_starts + bin_ends) * 0.5  # (num_samples,)

        if stratified:
            u = torch.rand(b, *m_shape, num_samples, device=device) * (1 / num_samples)  # (b, *m, num_samples)
            sampled_ts = bin_starts.expand(b, *m_shape, num_samples) + u  # (b, *m, num_samples)
        else:
            sampled_ts = bin_centers.expand(b, *m_shape, num_samples)  # (b, *m, num_samples)

        # scale sampled_ts to t_min and t_max
        sampled_ts = sampled_ts * (t_max - t_min).unsqueeze(-1) + t_min.unsqueeze(-1)  # (b, *m, num_samples)

        # must haves
        if must_have_ts is not None:
            sampled_ts = torch.cat([sampled_ts, must_have_ts], dim=-1)  # (b, *m, num_samples)
            # need to sort to ensure small to large
            sampled_ts, _ = sampled_ts.sort(dim=-1)  # (b, *m, num_samples)

        if add_mid_points:
            mid_points = (sampled_ts[..., :-1] + sampled_ts[..., 1:]) * 0.5  # (b, *m, num_samples-1)
            sampled_ts = torch.cat([sampled_ts, mid_points], dim=-1)
            # need to sort to ensure small to large
            sampled_ts, _ = sampled_ts.sort(dim=-1)  # (b, *m, num_samples)

        return sampled_ts

    def weighted_sample_points(
        self,
        num_samples: int,
        weights: torch.Tensor,
        weight_ts: torch.Tensor,
        weight_ts_sorted: bool,
        t_min: T.Union[torch.Tensor, float],
        t_max: T.Union[torch.Tensor, float],
        eps: float = 1e-4,
        must_have_ts: torch.Tensor = None,
    ) -> torch.Tensor:
        """
         Sample points on the rays between min_t and max_t based on weights.
         The higher the weights, the more samples we will assign to the region.

        Args:
            weights:
                (b, *m, l) weight to be l-th point
            weight_ts:
                (b, *m, l) ray_ts (in world coordinate) where weights is on
            weight_ts_sorted:
                whether weight_ts is already sorted from small to large
            num_samples:
                number of samples to sample on a ray
            t_min:
                (b, *m) or float.  The minimum ray_t to start.
            t_max:
                (b, *m) or float.  The maximum ray_t to start.
            must_have_ts:
                (b, *m, k), if not None, will insert these ts

        Returns:
            (b, *m, num_samples)  the sampled ray_t
        """
        assert num_samples > 0
        b, *m_shape, _3 = self.origins_w.shape
        num_weights = weights.size(-1)
        device = self.origins_w.device

        if isinstance(t_min, (float, int)):
            t_min = torch.ones(b, *m_shape, dtype=self.origins_w.dtype, device=device) * t_min

        if isinstance(t_max, (float, int)):
            t_max = torch.ones(b, *m_shape, dtype=self.origins_w.dtype, device=device) * t_max

        assert t_min.shape == (b, *m_shape)
        assert t_max.shape == (b, *m_shape)
        assert weights.shape == (b, *m_shape, num_weights)
        assert weight_ts.shape == (b, *m_shape, num_weights)

        if not weight_ts_sorted:
            weight_ts, ii = weight_ts.sort(dim=-1)  # (b, *m, num_weights)
            weights = torch.gather(weights, dim=-1, index=ii)  # (b, *m, num_weights)

        # make sure the weights is summed > 0
        weights_sum = torch.sum(weights, dim=-1, keepdim=True)  # (b, *m, 1)
        weight_base = torch.clamp(eps - weights_sum, min=0) / num_weights  # (b, *m, 1)
        weights = weights + weight_base  # (b, *m, num_weights)

        # treat each num_sample as a bin with prob = weights
        sample_bin_idxs = torch.multinomial(
            weights.reshape(-1, num_weights),  # (bm, num_weights)
            num_samples=num_samples,
            replacement=True,
        ).reshape(b, *m_shape, num_samples)  # (b, *m, num_samples)

        # construct bin_edges
        bin_edges = torch.cat(
            [
                t_min.unsqueeze(-1),  # (b, *m, 1)
                (weight_ts[..., :-1] + weight_ts[..., 1:]) * 0.5,  # (b, *m, num_weights-1)
                t_max.unsqueeze(-1),  # (b, *m, 1)
            ],
            dim=-1,
        )  # (b, *m, num_weights+1)

        bin_starts = bin_edges[..., :-1]  # (b, *m, num_weights)
        bin_ends = bin_edges[..., 1:]  # (b, *m, num_weights)
        bin_widths = bin_ends - bin_starts  # (b, *m, num_weights)

        # gather bin width each sample lies in
        sample_bin_widths = torch.gather(
            bin_widths,
            dim=-1,
            index=sample_bin_idxs,
        )  # (b, *m, num_samples)
        sample_bin_starts = torch.gather(
            bin_starts,
            dim=-1,
            index=sample_bin_idxs,
        )  # (b, *m, num_samples)

        # sample point inside the bin
        sampled_ts = (
            torch.rand(b, *m_shape, num_samples, device=device) * sample_bin_widths + sample_bin_starts
        )  # (b, *m, num_samples)

        # must haves
        if must_have_ts is not None:
            sampled_ts = torch.cat([sampled_ts, must_have_ts], dim=-1)  # (b, *m, num_samples)

        # sort ts from small to large
        sampled_ts, _ = sampled_ts.sort(dim=-1)  # (b, *m, num_samples)

        return sampled_ts

    @staticmethod
    def cat(rays: T.List["Ray"], dim: int) -> "Ray":
        out_dict = dict()
        for name in ["origins_w", "directions_w"]:
            arr = [getattr(p, name, None) for p in rays]
            if None in arr:
                out_dict[name] = None
            else:
                out_dict[name] = torch.cat(arr, dim=dim)
        return Ray(**out_dict)

    def masked_fill(self, mask: torch.Tensor, ray_src: "Ray"):
        self.origins_w[mask] = ray_src.origins_w[mask]
        self.directions_w[mask] = ray_src.directions_w[mask]

    @property
    def dtype(self):
        return self.origins_w.dtype

    @property
    def shape(self):
        return self.origins_w.shape

    @property
    def size(self):
        return self.origins_w.size

    @property
    def device(self):
        return self.origins_w.device

    def state_dict(self) -> T.Dict[str, T.Any]:
        """Returns a dictionary that can be saved or load."""
        to_save = dict()
        for name in ["origins_w", "directions_w"]:
            to_save[name] = getattr(self, name, None)
        return to_save

    def load_state_dict(
        self,
        state_dict: T.Dict[str, T.Any],
    ):
        """Load the state dictionary."""
        for name in ["origins_w", "directions_w"]:
            setattr(self, name, state_dict.get(name, None))

    def save(
        self,
        output_dir: str,
        overwrite: bool = False,
        save_ply: bool = True,
        save_pt: bool = True,
        cylinder_radius: float = 0.1,
        cone_radius: float = None,
        max_cone_height: float = None,
        end_xyz_w: torch.Tensor = None,  # same shape as origins_w
    ):
        if self.origins_w is None or self.directions_w is None:
            return

        if cone_radius is None:
            cone_radius = cylinder_radius * 1.5

        if max_cone_height is None:
            max_cone_height = cylinder_radius * 2.0

        if os.path.exists(output_dir) and not overwrite:
            raise RuntimeError(f"output dir {output_dir} exists")
        os.makedirs(output_dir, exist_ok=True)

        if save_pt:
            filename = os.path.join(output_dir, "state_dict.pt")
            torch.save(self.state_dict(), filename)

        if end_xyz_w is None:
            ray_ts = torch.ones(*self.origins_w.shape[:-1], device=self.origins_w.device)
            add_end_ball = False
        else:
            ray_ts = torch.linalg.vector_norm(end_xyz_w - self.origins_w, ord=2, dim=-1)
            add_end_ball = True

        b, *m_shape, _3 = self.origins_w.shape
        origins_w = self.origins_w.reshape(b, -1, 3)
        directions_w = self.directions_w.reshape(b, -1, 3)
        ray_ts = ray_ts.reshape(b, -1)
        if save_ply:
            # we are going to save individual camera frames
            for ib in range(origins_w.size(0)):
                sub_dir = os.path.join(output_dir, f"batch_{ib}")
                os.makedirs(sub_dir, exist_ok=True)

                for im in range(origins_w.size(1)):
                    R = rigid_motion.get_min_R(
                        v1=np.array([0, 0, 1.0], dtype=np.float32),
                        v2=directions_w[ib, im].detach().cpu().numpy(),
                    )
                    H_c2w = np.eye(4)
                    H_c2w[:3, :3] = R
                    H_c2w[:3, 3] = origins_w[ib, im].detach().cpu().float().numpy()

                    # ray origin
                    mesh = o3d.geometry.TriangleMesh.create_sphere(
                        radius=cylinder_radius * 1.1,
                        # width=cylinder_radius * 2.2,
                        # height=cylinder_radius * 2.2,
                        # depth=cylinder_radius * 2.2,
                    )
                    mesh.transform(H_c2w)
                    o3d.io.write_triangle_mesh(
                        filename=os.path.join(sub_dir, f"{im}_from.ply"),
                        mesh=mesh,
                    )

                    # ray direction
                    t = ray_ts[ib, im].detach().cpu().numpy()
                    if t > 0:
                        cone_height = min(t * 0.05, max_cone_height)
                        cylinder_height = t - cone_height
                        mesh = o3d.geometry.TriangleMesh.create_arrow(
                            cylinder_radius=cylinder_radius,
                            cone_radius=cone_radius,
                            cylinder_height=cylinder_height,
                            cone_height=cone_height,
                        )
                        mesh.transform(H_c2w)
                        o3d.io.write_triangle_mesh(
                            filename=os.path.join(sub_dir, f"{im}_arrow.ply"),
                            mesh=mesh,
                        )

                    # end point
                    if add_end_ball:
                        H_c2w2 = np.eye(4)
                        H_c2w2[:3, :3] = R
                        H_c2w2[:3, 3] = (
                            origins_w[ib, im].detach().cpu().float().numpy()
                            + t * directions_w[ib, im].detach().cpu().float().numpy()
                        )

                        mesh = o3d.geometry.TriangleMesh.create_sphere(
                            radius=cylinder_radius * 1.1,
                        )
                        mesh.transform(H_c2w2)
                        o3d.io.write_triangle_mesh(
                            filename=os.path.join(sub_dir, f"{im}_to.ply"),
                            mesh=mesh,
                        )


class PointersectRecord:
    def __init__(
        self,
        intersection_xyz_w: torch.Tensor,  # (b, *m_shape, 3)
        intersection_surface_normal_w: torch.Tensor,  # (b, *m_shape, 3)
        intersection_rgb: torch.Tensor,  # (b, *m_shape, 3)
        blending_weights: torch.Tensor,  # (b, *m_shape, k)  k: # neighbor points
        neighbor_point_idxs: torch.Tensor,  # long (b, *m_shape, k)
        neighbor_point_valid_len: torch.Tensor,  # long (b, *m_shape)
        ray_t: torch.Tensor,  # (b, *m_shape)
        ray_hit: torch.Tensor,  # (b, *m_shape)  bool
        ray_hit_logit: torch.Tensor,  # (b, *m_shape)
        model_attn_weights: torch.Tensor,  # (b, *m_shape, k+1, n_layers)
        refined_ray_hit: T.Optional[torch.Tensor] = None,  # (b, *m_shape)  bool
        model_info: T.Optional[T.Dict[str, T.Any]] = None,
        intersection_plane_normals_w: torch.Tensor = None,  # (b, *m_shape, 3)
        geometry_weights: torch.Tensor = None,  # (b, *m_shape, k)
        valid_neighbor_idx_mask: torch.Tensor = None,  # (b, *m_shape, k)  whether the neighbor_point_idxs is valid
        valid_plane_normal_mask: torch.Tensor = None,  # (b, *m_shape)
        total_time: float = None,
    ):
        self.intersection_xyz_w = intersection_xyz_w
        self.intersection_surface_normal_w = intersection_surface_normal_w
        self.intersection_rgb = intersection_rgb
        self.blending_weights = blending_weights
        self.neighbor_point_idxs = neighbor_point_idxs
        self.neighbor_point_valid_len = neighbor_point_valid_len
        self.ray_t = ray_t
        self.ray_hit = ray_hit
        self.ray_hit_logit = ray_hit_logit
        self.model_attn_weights = model_attn_weights
        self.refined_ray_hit = refined_ray_hit
        self.intersection_plane_normals_w = intersection_plane_normals_w
        self.geometry_weights = geometry_weights
        self.valid_neighbor_idx_mask = valid_neighbor_idx_mask
        self.valid_plane_normal_mask = valid_plane_normal_mask
        self.total_time = total_time
        self.model_info = model_info

        # self.cached_info = cached_info  # not saved nor concat nor reshaped

        self.attr_names = [
            "intersection_xyz_w",
            "intersection_surface_normal_w",
            "intersection_rgb",
            "blending_weights",
            "neighbor_point_idxs",
            "neighbor_point_valid_len",
            "ray_t",
            "ray_hit",
            "ray_hit_logit",
            "model_attn_weights",
            "refined_ray_hit",
            "model_info",
            "intersection_plane_normals_w",
            "geometry_weights",
            "valid_neighbor_idx_mask",
            "valid_plane_normal_mask",
        ]

    def to(self, device: torch.device) -> "PointersectRecord":
        for attr_name in self.attr_names:
            if attr_name == "model_info":
                continue
            arr = getattr(self, attr_name, None)
            if arr is not None:
                setattr(self, attr_name, arr.to(device=device))
        return self

    def state_dict(self) -> T.Dict[str, T.Any]:
        """Returns a dictionary that can be saved or load."""
        to_save = dict()
        for name in self.attr_names:
            to_save[name] = getattr(self, name, None)
        return to_save

    def load_state_dict(
        self,
        state_dict: T.Dict[str, T.Any],
    ):
        """Load the state dictionary."""
        for name in self.attr_names:
            setattr(self, name, state_dict.get(name, None))

    @staticmethod
    def cat(records: T.List["PointersectRecord"], dim: int) -> "PointersectRecord":
        """
        Concatenate a list of PointersectRecord at the given dimension.
        It is useful to split m_shape.
        """
        out = dict()
        for name in [
            "intersection_xyz_w",
            "intersection_surface_normal_w",
            "intersection_rgb",
            "blending_weights",
            "neighbor_point_idxs",
            "neighbor_point_valid_len",
            "ray_t",
            "ray_hit",
            "ray_hit_logit",
            "model_attn_weights",
            "refined_ray_hit",
            "intersection_plane_normals_w",
            "geometry_weights",
            "valid_neighbor_idx_mask",
            "valid_plane_normal_mask",
        ]:
            arr = [getattr(r, name, None) for r in records]
            if None in arr:
                out[name] = None
            else:
                out[name] = torch.cat(arr, dim=dim)

        if len(records) > 0:
            out["model_info"] = records[0].model_info

        return PointersectRecord(**out)

    def chunk(self, chunks: int, dim: int) -> T.List["PointersectRecord"]:
        """
        Chunk the PointersectRecord at the given dimension.
        As pytorch, the resulted tensors are views to the original ones.
        """

        attr_names = [
            "intersection_xyz_w",
            "intersection_surface_normal_w",
            "intersection_rgb",
            "blending_weights",
            "neighbor_point_idxs",
            "neighbor_point_valid_len",
            "ray_t",
            "ray_hit",
            "ray_hit_logit",
            "model_attn_weights",
            "refined_ray_hit",
            "intersection_plane_normals_w",
            "geometry_weights",
            "valid_neighbor_idx_mask",
            "valid_plane_normal_mask",
        ]

        actual_chunks = None
        out = dict()
        for name in attr_names:
            arr = getattr(self, name, None)
            if arr is None:
                out[name] = None
            else:
                out[name] = arr.chunk(chunks=chunks, dim=dim)
                if actual_chunks is None:
                    actual_chunks = len(out[name])
                else:
                    assert len(out[name]) == actual_chunks

        results = []
        for i in range(actual_chunks):
            tmp_dict = dict()
            for name in attr_names:
                arr_list = out[name]
                if arr_list is None:
                    tmp_dict[name] = None
                else:
                    tmp_dict[name] = arr_list[i]
            tmp_dict["model_info"] = self.model_info
            p = PointersectRecord(**tmp_dict)
            results.append(p)

        return results

    @staticmethod
    def aggregate(records: T.List["PointersectRecord"]) -> "PointersectRecord":
        """
        Aggregate a list of PointersectRecord of the same shape.
        Note that it will set many attributes to None.
        """
        out = dict()
        # simple average
        for name in [
            "intersection_xyz_w",
            "intersection_rgb",
            "ray_t",
            "ray_hit",
            "ray_hit_logit",
            "model_attn_weights",
            "refined_ray_hit",
        ]:
            arr = [getattr(r, name, None) for r in records]
            arr = [a for a in arr if a is not None]
            if len(arr) == 0:
                out[name] = None
            else:
                out[name] = sum(arr) / len(arr)

        # set to be the first
        for name in [
            "blending_weights",
            "neighbor_point_idxs",
            "neighbor_point_valid_len",
            "model_attn_weights",
            "geometry_weights",
            "valid_neighbor_idx_mask",
        ]:
            arr = [getattr(r, name, None) for r in records]
            if len(arr) == 0:
                out[name] = None
            else:
                out[name] = arr[0]

        # sum -> normalize to unit norm
        for name in [
            "intersection_surface_normal_w",
            "intersection_plane_normals_w",
        ]:
            arr = [getattr(r, name, None) for r in records]
            arr = [a for a in arr if a is not None]
            if len(arr) == 0:
                out[name] = None
            else:
                out[name] = sum(arr)
                out[name] = torch.nn.functional.normalize(out[name], p=2, dim=-1)

        # set to be and
        for name in ["valid_plane_normal_mask"]:
            arr = [getattr(r, name, None) for r in records]
            arr = [a for a in arr if a is not None]
            if len(arr) == 0:
                out[name] = None
            else:
                out[name] = arr[0]
                for i in range(1, len(arr)):
                    out[name] = torch.logical_and(out[name], arr[i])

        if len(records) > 0:
            out["model_info"] = records[0].model_info

        return PointersectRecord(**out)

    def reshape(self, new_b: int, new_m_shape: T.List[int]):
        """Reshape each attributes."""
        if isinstance(new_b, int):
            new_b = [new_b]
        if isinstance(new_m_shape, int):
            new_m_shape = [new_m_shape]

        b, *m_shape, _ = self.intersection_xyz_w.shape

        # (b, *m_shape, d)
        for name in [
            "intersection_xyz_w",
            "intersection_surface_normal_w",
            "intersection_rgb",
            "blending_weights",
            "neighbor_point_idxs",
            "intersection_plane_normals_w",
            "geometry_weights",
            "valid_neighbor_idx_mask",
        ]:
            arr = getattr(self, name, None)
            if arr is None:
                continue
            arr = arr.reshape(*new_b, *new_m_shape, arr.size(-1))
            setattr(self, name, arr)

        # (b, *m_shape)
        for name in [
            "neighbor_point_valid_len",
            "ray_t",
            "ray_hit",
            "ray_hit_logit",
            "refined_ray_hit",
            "valid_plane_normal_mask",
        ]:
            arr = getattr(self, name, None)
            if arr is None:
                continue
            arr = arr.reshape(*new_b, *new_m_shape)
            setattr(self, name, arr)

        # # (b, *m_shape, k+1, n_layers)
        for name in [
            "model_attn_weights",
        ]:
            arr = getattr(self, name, None)
            if arr is None:
                continue
            arr = arr.reshape(*new_b, *new_m_shape, arr.size(-2), arr.size(-1))
            setattr(self, name, arr)

    def save(
        self,
        output_dir: str,
        overwrite: bool = False,
    ):
        if os.path.exists(output_dir) and not overwrite:
            raise RuntimeError
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, "state_dict.pt")
        torch.save(self.state_dict(), filename)

    def get_rgbd_image(
        self,
        camera: "Camera",
        th_hit_prob: float = None,
        th_dot_product: float = None,  # (less than 60 degree)
        use_plane_normal: bool = False,
    ) -> "RGBDImage":
        """
        Return rgbd_image (with surface normal) to compare with other methods.
        Camera should be the orignal camera used to cast the camera rays.
        Only support shape = (b, q, h, w).
        """
        b, *m_shape, _ = self.intersection_xyz_w.shape  # (b, q, h, w, 3)
        assert len(m_shape) == 3  # (q, h, w)
        q, h, w = m_shape

        if th_hit_prob is None:
            th_hit_prob = 0
        if th_dot_product is None:
            th_dot_product = 0

        # xyz_w -> xyz_c
        H_w2c = camera.get_H_w2c()  # (b, q, 4, 4)
        H_w2c = H_w2c.reshape(b, q, 1, 1, 4, 4)  # (b, q, 1, 1, 4, 4)
        xyz_w = self.intersection_xyz_w.unsqueeze(-1)  # (b, q, h, w, 3, 1)
        xyz_c = (
            linalg_utils.matmul(
                H_w2c[..., :3, :3],
                xyz_w,
            )
            + H_w2c[..., :3, 3:4]
        )  # (b, q, h, w, 3, 1)
        # depth is the z in camera coordinate
        depth = xyz_c[..., 2, 0]  # (b, q, h, w)
        assert depth.shape == torch.Size([b, *m_shape])

        # determine hit map to use
        if self.refined_ray_hit is not None:
            hit_map = self.refined_ray_hit  # (b, q, h, w)
        else:
            hit_map = self.ray_hit  # (b, q, h, w)

        if th_dot_product > 1e-6 or th_hit_prob > 1e-6:
            ray = camera.generate_camera_rays(device=camera.H_c2w.device)
            confidence = self.compute_confidence(
                ray_direction_w=ray.directions_w,  # (b, q, h, w, 3)
                th_hit_prob=th_hit_prob,
                th_dot_product=th_dot_product,
            )  # (b, q, h, w)
            hit_map = torch.logical_and(hit_map, confidence)

        if use_plane_normal and self.intersection_plane_normals_w is not None:
            normal_w = self.intersection_plane_normals_w * hit_map.unsqueeze(-1)  # (b, q, h, w, 3)
        else:
            normal_w = self.intersection_surface_normal_w * hit_map.unsqueeze(-1)  # (b, q, h, w, 3)

        return RGBDImage(
            rgb=self.intersection_rgb * hit_map.unsqueeze(-1),  # (b, q, h, w, 3)
            depth=depth * hit_map,  # (b, q, h, w)
            normal_w=normal_w,  # (b, q, h, w, 3)
            hit_map=hit_map,
            camera=camera,
        )

    def compute_confidence(
        self,
        ray_direction_w: T.Optional[torch.Tensor] = None,
        th_hit_prob: float = 0.85,
        th_dot_product: float = 0.5,  # (less than 60 degree)
    ):
        """
        Compute the binary confidence of the ray based on hit
        and the angle between the ray direction and the surface normal
        is small enough.

        Args:
            ray_direction_w:
                (b, *m_shape, 3) ray direction used to query the points
            th_hit_prob:
                confident if hit_prob is large enough
            th_dot_product:
                confident if the angle between ray and surface normal
                is small enough.

        Returns:
            confidence (b, *m_shape,), {0, 1: confident}
        """
        hit_probs = self.ray_hit_logit.sigmoid()
        hit = hit_probs >= min(1 - 1e-4, th_hit_prob)  # (b, *m_shape,) bool
        miss = hit_probs <= max(1e-4, (1 - th_hit_prob))  # (b, *m_shape,) bool
        hit_miss = hit + miss

        if ray_direction_w is not None:
            dot_prod = (ray_direction_w * self.intersection_surface_normal_w).sum(dim=-1)  # (b, *m)
            angle = dot_prod.abs() >= th_dot_product
            return torch.logical_and(hit_miss, angle)
        else:
            return hit_miss


class Camera:
    def __init__(
        self,
        H_c2w: torch.Tensor,  # (b, q, 4, 4)  camera pose in the world coord
        intrinsic: torch.Tensor,  # (b, q, 3, 3)  camera intrinsics
        width_px: int,
        height_px: int,
        timestamp: torch.Tensor = None,  # float (b, q)
    ):
        self.H_c2w = H_c2w
        self.intrinsic = intrinsic
        self.width_px = width_px
        self.height_px = height_px
        self.timestamp = timestamp

        self.attr_names = ["H_c2w", "intrinsic", "width_px", "height_px", "timestamp"]

    @staticmethod
    def allclose(
        camera1: "Camera",
        camera2: "Camera",
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
        raise_error: bool = False,
    ) -> bool:
        """
        Check if two camera objects are the same.
        """
        if camera1.width_px != camera2.width_px:
            if raise_error:
                raise ValueError(f"camera1.width_px {camera1.width_px} != camera2.width_px {camera2.width_px}")
            return False
        if camera1.height_px != camera2.height_px:
            if raise_error:
                raise ValueError(f"camera1.height_px {camera1.height_px} != camera2.height_px {camera2.height_px}")
            return False

        _result = render.allclose_intrinsic(
            intrinsic1=camera1.intrinsic,
            intrinsic2=camera2.intrinsic,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )
        if not _result:
            if raise_error:
                raise ValueError(f"camera1.intrinsic {camera1.intrinsic} != camera2.intrinsic {camera2.intrinsic}")
            return False

        _result = rigid_motion.allclose_H_c2w(
            H_c2w1=camera1.H_c2w,
            H_c2w2=camera2.H_c2w,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )
        if not _result:
            if raise_error:
                raise ValueError(f"camera1.H_c2w {camera1.H_c2w} != camera2.H_c2w {camera2.H_c2w}")
            return False

        # timestamp
        if camera1.timestamp is not None or camera2.timestamp is not None:
            assert camera1.timestamp is not None
            assert camera2.timestamp is not None
            _result = torch.allclose(
                camera1.timestamp,
                camera2.timestamp,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            )
            if not _result:
                if raise_error:
                    raise ValueError(f"camera1.timestamp {camera1.timestamp} != camera2.timestamp {camera2.timestamp}")
                return False

        return True

    def contiguous(self):
        for key in [
            "H_c2w",
            "intrinsic",
            "timestamp",
        ]:
            arr = getattr(self, key, None)
            if arr is not None:
                setattr(self, key, arr.contiguous())
        return self

    @linalg_utils.disable_tf32_and_autocast()
    def resize(self, new_height_px: int, new_width_px: int) -> "Camera":
        """
        Change the resolution without changing the field of view.

        Args:
            new_height_px:
                new number of pixels in height
            new_width_px:
                new number of pixels in width

        Returns:
            new camera
        """
        scale_w = new_width_px / self.width_px
        scale_h = new_height_px / self.height_px
        new_camera = self.clone()
        new_camera.height_px = new_height_px
        new_camera.width_px = new_width_px
        new_camera.intrinsic[:, :, 0, :] = new_camera.intrinsic[:, :, 0, :] * scale_w
        new_camera.intrinsic[:, :, 1, :] = new_camera.intrinsic[:, :, 1, :] * scale_h
        return new_camera

    @linalg_utils.disable_tf32_and_autocast()
    def coordinate_transform(self, H_w2n: torch.Tensor):
        """
        Transform the coodinate system to "new", ie, multiply everything with H_w2n.
        The transformation is performed inplace.

        Args:
            H_w2n:
                (b, 4, 4)  convert the current world coordinate to the new coordinate
        """
        b, _41, _42 = H_w2n.shape

        # (b, 4, 4) (b, q, h, w, 3)
        self.H_c2w = linalg_utils.matmul(
            H_w2n.reshape(b, 1, 4, 4).to(dtype=self.H_c2w.dtype),  # (b, 1q, 4, 4)
            self.H_c2w,  # (b, q, 4, 4)
        )  # (b, q, 4, 4)

        return self

    def index_select(self, dim: int, index: torch.Tensor) -> "Camera":
        camera = self.clone()
        for attr_name in ["H_c2w", "intrinsic", "timestamp"]:
            arr = getattr(camera, attr_name, None)
            if arr is not None:
                setattr(camera, attr_name, torch.index_select(arr, dim=dim, index=index))
        return camera

    def chunk(self, chunks: int, dim: int = 0) -> T.List["Camera"]:
        out_dict = dict()
        total = None
        for attr_name in ["H_c2w", "intrinsic", "timestamp"]:
            arr = getattr(self, attr_name, None)
            if arr is not None:
                chunked_arr = arr.chunk(chunks=chunks, dim=dim)
                out_dict[attr_name] = chunked_arr
                if total is None:
                    total = len(chunked_arr)
                else:
                    assert len(chunked_arr) == total
        cameras = []
        for i in range(total):
            d = dict()
            for attr_name in out_dict:
                d[attr_name] = out_dict[attr_name][i]
            camera = Camera(**d, width_px=self.width_px, height_px=self.height_px)
            cameras.append(camera)
        return cameras

    def to(self, device: torch.device, dtype: torch.dtype = None) -> "Camera":
        self.H_c2w = self.H_c2w.to(device=device, dtype=dtype)
        self.intrinsic = self.intrinsic.to(device=device, dtype=dtype)
        if self.timestamp is not None:
            self.timestamp = self.timestamp.to(device=device, dtype=dtype)
        return self

    def detach(self) -> "Camera":
        self.H_c2w = self.H_c2w.detach()
        self.intrinsic = self.intrinsic.detach()
        if self.timestamp is not None:
            self.timestamp = self.timestamp.detach()
        return self

    def clone(self) -> "Camera":
        return Camera(
            H_c2w=self.H_c2w.clone() if self.H_c2w is not None else None,
            intrinsic=self.intrinsic.clone() if self.intrinsic is not None else None,
            width_px=self.width_px,
            height_px=self.height_px,
            timestamp=self.timestamp.clone() if self.timestamp is not None else None,
        )

    @staticmethod
    def cat(cameras: T.List["Camera"], dim: int) -> "Camera":
        out = dict()
        for name in ["H_c2w", "intrinsic", "timestamp"]:
            arr = [getattr(r, name, None) for r in cameras]
            if None in arr:
                out[name] = None
            else:
                out[name] = torch.cat(arr, dim=dim)
        width_pxs = [getattr(r, "width_px", None) for r in cameras]
        height_pxs = [getattr(r, "height_px", None) for r in cameras]
        assert len(np.unique(width_pxs)) == 1, f"{len(np.unique(width_pxs))=}, {np.unique(width_pxs)=}"
        assert len(np.unique(height_pxs)) == 1, f"{len(np.unique(height_pxs))=}, {np.unique(height_pxs)=}"
        out["width_px"] = width_pxs[0]
        out["height_px"] = height_pxs[0]

        return Camera(**out)

    def expand(self, n: int, dim: int) -> "Camera":
        for name in ["H_c2w", "intrinsic", "timestamp"]:
            arr = getattr(self, name, None)
            if arr is not None:
                assert arr.shape[dim] == 1, f"{name} shape: {arr.shape}, {dim}"
                setattr(self, name, arr.expand(*arr.shape[:dim], n, *arr.shape[dim + 1 :]))
        return self

    def contiguous(self):
        for name in ["H_c2w", "intrinsic", "timestamp"]:
            arr = getattr(self, name, None)
            if arr is not None:
                setattr(self, name, arr.contiguous())
        return self

    def __getitem__(self, ib) -> "Camera":
        """slice the camera in the b dimension. Always retain (b, q, 4, 4)
        even when ib is int."""
        if isinstance(ib, (int, torch.Size)):
            ib = slice(int(ib), int(ib) + 1)

        camera = Camera(
            H_c2w=self.H_c2w[ib],
            intrinsic=self.intrinsic[ib],
            width_px=self.width_px,
            height_px=self.height_px,
            timestamp=self.timestamp[ib] if self.timestamp is not None else None,
        )
        assert camera.H_c2w.ndim == 4
        assert camera.intrinsic.ndim == 4
        assert camera.timestamp is None or camera.timestamp.ndim == 2
        return camera

    def state_dict(self) -> T.Dict[str, T.Any]:
        """Returns a dictionary that can be saved or load."""
        to_save = dict()
        for name in self.attr_names:
            to_save[name] = getattr(self, name, None)
        return to_save

    def load_state_dict(
        self,
        state_dict: T.Dict[str, T.Any],
    ):
        """Load the state dictionary."""
        for name in self.attr_names:
            setattr(self, name, state_dict.get(name, None))

    def load_state_dict_numpy(self, state_dict: dict):
        """Load the state dictionary."""
        for name in self.attr_names:
            val = state_dict.get(name, None)
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            setattr(self, name, val)

    def get_H_w2c(self) -> torch.Tensor:
        """
        Returns extrinsic matrices (inverse of H_c2w), shape: (b, q, 4, 4).
        """
        return rigid_motion.inv_homogeneous_tensors(self.H_c2w)

    @linalg_utils.disable_tf32_and_autocast()
    def generate_camera_rays(
        self,
        subsample: int = 1,
        offsets: str = "center",
        device: torch.device = torch.device("cpu"),
    ) -> Ray:
        """
        Generate camera rays: ray_origin is at pinhole and
        ray directions outward from a pixel location (somewhere withing
        a pixel pitch) to pinhole.

        Args:
            offsets:
                'center' or 0, ray will be coming from the center of a pixel
                'rand': random offset = [-0.5, 0.5)

        Returns:
            camera ray: (b, q, h, w)
        """

        *b_shape, _, _ = self.H_c2w.shape  # (b, q, 4, 4)

        ray_origins_w, ray_directions_w = utils.generate_camera_rays(
            cam_poses=self.H_c2w.reshape(-1, 4, 4),
            intrinsics=self.intrinsic.reshape(-1, 3, 3),
            width_px=self.width_px,
            height_px=self.height_px,
            subsample=subsample,
            offsets=offsets,
            device=device,
        )  # (bq, h, w, 3), (bq, h, w, 3)

        bq, h, w, _ = ray_origins_w.shape

        return Ray(
            origins_w=ray_origins_w.reshape(*b_shape, h, w, 3),  # (b, q, h, w, 3)
            directions_w=ray_directions_w.reshape(*b_shape, h, w, 3),  # (b, q, h, w, 3)
        )

    @linalg_utils.disable_tf32_and_autocast()
    def generate_random_patch_rays(
        self,
        num_patches_per_q: int,
        patch_width_px: int,
        patch_width_pitch_scale: T.Union[float, torch.Tensor] = 1.0,  # (*b,)
        patch_height_px: int = None,  # (*b,)
        patch_height_pitch_scale: T.Union[float, torch.Tensor] = None,  # (*b,)
        prob_density: torch.Tensor = None,  # (b, q, h, w)
        prob_density_bias: float = 5.0,
        int_only: bool = True,
        inbound_only: bool = True,
    ) -> T.Dict[str, T.Any]:
        """
        Generate rays to form patches on the corresponding images.

        Args:
            num_patches_per_q:
                number of patches from each q
            patch_width_px:
                number of pixels in the patch in width
            patch_width_pitch_scale:
                (*b,) the pitch of the patch (new_pitch / old_pitch)
            patch_height_px:
                if None, the same as `patch_width_px`
            patch_height_pitch_scale:
                if None, the same as `patch_width_pitch_scale`
            prob_density:
                (b, q, h, w) to scale the probability to sample each pixel
            prob_density_bias:
                if higher, more likely to sample patches with higher prob_density
            int_only:
                whether the center is always at an integer index

        Returns:
            ray:
                (b, q, num_patches_per_q, hp, wp)
            uv:
                (b, q, num_patches_per_q, hp, wp, 2)
        """
        b, q, _41, _42 = self.H_c2w.shape
        bq = b * q

        # sample uv
        uv = utils.sample_random_patch_uv(
            b_shape=[b, q, num_patches_per_q],
            width_px=self.width_px,
            height_px=self.height_px,
            patch_width_px=patch_width_px,
            patch_width_pitch_scale=patch_width_pitch_scale,
            patch_height_px=patch_height_px,
            patch_height_pitch_scale=patch_height_pitch_scale,
            prob_density=prob_density,
            prob_density_bias=prob_density_bias,
            int_only=int_only,
            inbound_only=inbound_only,
            device=self.H_c2w.device,
        )  # (b, q, num_patches_per_q, hp, wp, 2), [0, w], [0, h]

        # use the uv to create rays
        ray_origins_w, ray_directions_w = utils.generate_camera_rays_from_uv(
            cam_poses=self.H_c2w.reshape(bq, 4, 4),  # (bq, 4, 4)
            intrinsics=self.intrinsic.reshape(bq, 3, 3),  # (bq, 3, 3)
            uv=uv.flatten(start_dim=0, end_dim=1),  # (bq, num_patches_per_q, hp, wp, 2)
            device=self.H_c2w.device,
        )  # (bq, num_patches_per_q, hp, wp, 3)

        _bq, *m_shape, _3 = ray_origins_w.shape

        ray = Ray(
            origins_w=ray_origins_w.reshape(b, q, *m_shape, 3),  # (b, q, np, hp, wp, 3)
            directions_w=ray_directions_w.reshape(b, q, *m_shape, 3),  # (b, q, np, hp, wp, 3)
        )

        return dict(
            ray=ray,  # (b, q, np, hp, wp, 3)
            uv=uv,  # (b, q, np, hp, wp, 2)
        )

    def split(self, chunk_size: int) -> T.List["Camera"]:
        """
        Split camera (b, q) into a list of cameras (b', q'),
        such that b' * q' * h * w < chunk_size.

        Note that we only chunk q or chunk b

        Returns:
            list of cameras
        """
        if chunk_size < 0:
            return [self]

        hw = self.width_px * self.height_px
        N = max(1, int(chunk_size / hw))  # max bq for each chunk
        q = self.H_c2w.size(1)
        b = self.H_c2w.size(0)

        if N >= b * q:
            return [self]
        elif N > q:
            # chunk b
            chunk_dim = 0
            chunks = math.ceil(b / int(N / q))

            H_c2w_list = torch.chunk(self.H_c2w, chunks=chunks, dim=chunk_dim)
            intrinsic_list = torch.chunk(self.intrinsic, chunks=chunks, dim=chunk_dim)
            if self.timestamp is not None:
                timestamp_list = torch.chunk(self.timestamp, chunks=chunks, dim=chunk_dim)
            else:
                timestamp_list = [None] * len(H_c2w_list)

            cameras = []
            for H, ins, ts in zip(H_c2w_list, intrinsic_list, timestamp_list):
                cameras.append(
                    Camera(
                        H_c2w=H,
                        intrinsic=ins,
                        width_px=self.width_px,
                        height_px=self.height_px,
                        timestamp=ts,
                    )
                )
            return cameras
        else:
            # chunk b and q
            cameras = []
            for ib in range(b):
                chunk_dim = 1
                chunks = math.ceil(q / N)
                H_c2w_list = torch.chunk(self.H_c2w[ib : ib + 1], chunks=chunks, dim=chunk_dim)
                intrinsic_list = torch.chunk(self.intrinsic[ib : ib + 1], chunks=chunks, dim=chunk_dim)
                if self.timestamp is not None:
                    timestamp_list = torch.chunk(self.timestamp[ib : ib + 1], chunks=chunks, dim=chunk_dim)
                else:
                    timestamp_list = [None] * len(H_c2w_list)
                for H, ins, ts in zip(H_c2w_list, intrinsic_list, timestamp_list):
                    cameras.append(
                        Camera(
                            H_c2w=H,
                            intrinsic=ins,
                            width_px=self.width_px,
                            height_px=self.height_px,
                            timestamp=ts,
                        )
                    )
            return cameras

    @torch.no_grad()
    def uniformly_sample(self, num_samples: int) -> "Camera":
        """
        Uniformly sample more cameras from the current ones.
        Currently we do not support gradient (though nothing stops it theoretically).

        Note:
            The function currently does not care about timestamps. It simply samples new
            camera poses between two q_idxs.

        Returns:
            new camera: (b, num_samples)
        """
        length = self.H_c2w.size(1)

        idxs = np.linspace(0, 1 - 1e-8, num_samples) * (length - 1)
        self_H_c2w = self.H_c2w.detach().cpu().numpy()  # (b, q, 4, 4)
        self_intrinsic = self.intrinsic.detach().cpu().numpy()  # (b, q, 3, 3)
        self_timestamp = self.timestamp.detach().cpu().numpy() if self.timestamp is not None else None  # (b, q)

        all_H_c2ws = []
        all_intrinsics = []
        all_timestamps = []
        for b in range(self.H_c2w.size(0)):
            H_c2ws = []
            intrinsics = []
            timestamps = []

            for i in range(len(idxs)):
                idx = idxs[i]
                idx_from = math.floor(idx)
                idx_to = idx_from + 1
                t = idx - idx_from
                H_c2w = rigid_motion.interp_homegeneous_matrices(
                    t=t,
                    H0=self_H_c2w[b, idx_from],
                    H1=self_H_c2w[b, idx_to],
                )
                H_c2w = torch.from_numpy(H_c2w)
                if (
                    torch.norm(H_c2w, p=2, dim=-2).any() <= 1e-6
                    or torch.logical_not(torch.norm(H_c2w, p=2, dim=-2).isfinite()).any()
                ):
                    print("oh no!")
                H_c2ws.append(H_c2w)

                intrinsic = (1 - t) * self_intrinsic[b, idx_from] + t * self_intrinsic[b, idx_to]
                intrinsics.append(torch.from_numpy(intrinsic))

                if self_timestamp is not None:
                    timestamp = (1 - t) * self_timestamp[b, idx_from] + t * self_timestamp[b, idx_to]
                    timestamps.append(torch.from_numpy(timestamp))
                else:
                    timestamps.append(None)

            H_c2ws = torch.stack(H_c2ws, dim=0)
            intrinsics = torch.stack(intrinsics, dim=0)
            timestamps = torch.stack(timestamps, dim=0) if self.timestamp is not None else None
            all_H_c2ws.append(H_c2ws)
            all_intrinsics.append(intrinsics)
            all_timestamps.append(timestamps)

        all_H_c2ws = torch.stack(all_H_c2ws, dim=0)
        all_intrinsics = torch.stack(all_intrinsics, dim=0)
        all_timestamps = torch.stack(all_timestamps, dim=0) if self_timestamp is not None else None

        return Camera(
            H_c2w=all_H_c2ws.to(device=self.H_c2w.device, dtype=self.H_c2w.dtype),
            intrinsic=all_intrinsics.to(device=self.intrinsic.device, dtype=self.intrinsic.dtype),
            width_px=self.width_px,
            height_px=self.height_px,
            timestamp=all_timestamps.to(device=self.H_c2w.device, dtype=self.H_c2w.dtype)
            if self.timestamp is not None
            else None,
        )

    def get_camera_frames(
        self,
        camera_frame_size: float = 0.1,
    ) -> T.List[T.List[o3d.geometry.TriangleMesh]]:
        """
        Create o3d meshes of camera frames
        """
        all_camera_frames = []
        for ib in range(self.H_c2w.size(0)):
            cam_frames = []
            for iq in range(self.H_c2w.size(1)):
                cam_frame = utils.get_o3d_camera_frame(
                    self.H_c2w[ib, iq],
                    frame_size=camera_frame_size,
                )
                cam_frames.append(cam_frame)
            all_camera_frames.append(cam_frames)
        return all_camera_frames

    def get_camera_trajectory_arrows(
        self,
        radius=0.1,
    ) -> T.List[T.List[o3d.geometry.TriangleMesh]]:
        """
        Create arrows pointing from one camera position to the next one.

        Returns:
            list of list of arrows.  (b, q-1)
        """
        all_camera_arrows = []
        for ib in range(self.H_c2w.size(0)):
            cam_arrows = []
            for iq in range(self.H_c2w.size(1) - 1):
                current_xyz_w = self.H_c2w[ib, iq, :3, 3]  # (3,)
                next_xyz_w = self.H_c2w[ib, iq + 1, :3, 3]  # (3,)
                v = next_xyz_w - current_xyz_w
                v_length = torch.linalg.norm(v, ord=2)  # (,)
                v_direction = v / (v_length + 1e-12)

                cone_height = radius * 1.5
                cylinder_height = max(0.1, v_length - cone_height)

                cam_arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=radius,
                    cone_radius=radius * 1.5,
                    cylinder_height=cylinder_height,
                    cone_height=cone_height,
                )

                # translate and rotate the arrow
                R = rigid_motion.construct_coord_frame(
                    z=v_direction,
                )
                t = current_xyz_w
                H = torch.zeros(4, 4)
                H[:3, :3] = R
                H[:3, 3] = t
                H[3, 3] = 1
                cam_arrow.transform(H)
                cam_arrows.append(cam_arrow)
            all_camera_arrows.append(cam_arrows)
        return all_camera_arrows

    def save(
        self,
        output_dir: str,
        overwrite: bool = False,
        save_ply: bool = True,
        save_individual_ply: bool = True,
        save_pt: bool = True,
        world_frame_size: float = 1.0,
        camera_frame_size: float = 0.1,
        scene_meshes: T.Optional[T.List[o3d.geometry.TriangleMesh]] = None,
    ):
        if os.path.exists(output_dir) and not overwrite:
            raise RuntimeError(f"output dir {output_dir} exists")
        os.makedirs(output_dir, exist_ok=True)

        if scene_meshes is not None and not isinstance(scene_meshes, (list, tuple)):
            scene_meshes = [scene_meshes] * self.H_c2w.size(0)

        if save_pt:
            filename = os.path.join(output_dir, "state_dict.pt")
            torch.save(self.state_dict(), filename)

        if save_individual_ply:
            # save a ply file containing world and camera coordinates
            all_camera_frames = self.get_camera_frames(camera_frame_size=camera_frame_size)

            # we are going to save individual camera frames
            for ib in range(self.H_c2w.size(0)):
                sub_dir = os.path.join(output_dir, f"batch_{ib}")
                os.makedirs(sub_dir, exist_ok=True)

                for iq in range(self.H_c2w.size(1)):
                    filename = os.path.join(sub_dir, f"{iq}.ply")
                    o3d.io.write_triangle_mesh(
                        filename=filename,
                        mesh=all_camera_frames[ib][iq],
                    )

                # save world coord
                world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=world_frame_size)
                filename = os.path.join(sub_dir, f"world.ply")
                o3d.io.write_triangle_mesh(
                    filename=filename,
                    mesh=world_frame,
                )

                # scene
                if scene_meshes is not None and scene_meshes[ib] is not None:
                    filename = os.path.join(sub_dir, f"scene.obj")
                    try:
                        o3d.io.write_triangle_mesh(
                            filename=filename,
                            mesh=scene_meshes[ib],
                        )
                    except:
                        pass

        if save_ply:
            # save a ply file containing world and camera coordinates
            all_camera_frames = self.get_camera_frames(camera_frame_size=camera_frame_size)

            # typical ply (no color)
            for ib in range(self.H_c2w.size(0)):
                cam_frames = all_camera_frames[ib]
                # world coord
                world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=world_frame_size)

                # combine all meshes
                mesh = world_frame

                # scene box
                if scene_meshes is not None:
                    try:
                        mesh = mesh + scene_meshes[ib]
                    except:
                        pass

                for cam_frame in cam_frames:
                    mesh = mesh + cam_frame

                filename = os.path.join(output_dir, f"batch_{ib}.obj")
                o3d.io.write_triangle_mesh(
                    filename=filename,
                    mesh=mesh,
                )

    @linalg_utils.disable_tf32_and_autocast()
    def get_perspective_projection_mtx(
        self,
        t_min: float = 1.0e-4,
        t_max: float = None,
        invert_z: bool = False,
    ) -> torch.Tensor:
        """
        Return the opengl perspective projection matrix, which includes the
        effect of intrinsic matrix and mapping to NDC.
        See: http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/

        Args:
            t_min:
                near point of ray
            t_max:
                far point of ray, if None or INF, inf

        Returns:
            P:
                (b, q, 4, 4) the perspective projection matrix

        Note that the perspective matrix assumes the camera is looking at -z
        (ie, it maps -znear to -1 and -zfar to 1).
        To use the perspective project as the intrinsic matrix and get the same
        rendering result as ours, set invert_z = True.
        """
        assert t_min >= 1e-6
        b, q, _31, _32 = self.intrinsic.shape
        P = torch.zeros(b, q, 4, 4, dtype=self.intrinsic.dtype, device=self.intrinsic.device)  # (b, q, 4, 4)

        fx = self.intrinsic[..., 0, 0]  # (b, q)
        fy = self.intrinsic[..., 1, 1]  # (b, q)
        cx = self.intrinsic[..., 0, 2]  # (b, q)
        cy = self.intrinsic[..., 1, 2]  # (b, q)

        P[..., 0, 0] = fx * (2 / self.width_px)  # (b, q)
        P[..., 1, 1] = fy * (2 / self.height_px)  # (b, q)
        P[..., 0, 2] = 1.0 - cx * (2.0 / self.width_px)  # (b, q)
        P[..., 1, 2] = cy * (2.0 / self.height_px) - 1  # (b, q)
        P[..., 3, 2] = -1

        if t_max is None or t_max >= INF:
            P[..., 2, 2] = -1.0
            P[..., 2, 3] = -2.0 * t_min
        else:
            P[..., 2, 2] = (t_max + t_min) / (t_min - t_max)
            P[..., 2, 3] = (2 * t_max * t_min) / (t_min - t_max)

        if invert_z:
            P[..., :, 2] = P[..., :, 2] * -1

        return P  # (b, q, 4, 4)

    @linalg_utils.disable_tf32_and_autocast()
    def get_inv_perspective_projection_mtx(
        self,
        t_min: float = 1.0e-4,
        t_max: float = None,
        invert_z: bool = False,
    ) -> torch.Tensor:
        """
        Return the opengl perspective projection matrix, which includes the
        effect of intrinsic matrix and mapping to NDC.
        See: http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/

        Args:
            t_min:
                near point of ray
            t_max:
                far point of ray, if None or INF, inf

        Returns:
            P:
                (b, q, 4, 4) the perspective projection matrix
        """
        assert t_min >= 1e-6
        b, q, _31, _32 = self.intrinsic.shape
        invP = torch.zeros(b, q, 4, 4, dtype=self.intrinsic.dtype, device=self.intrinsic.device)  # (b, q, 4, 4)

        fx = self.intrinsic[..., 0, 0]  # (b, q)
        fy = self.intrinsic[..., 1, 1]  # (b, q)
        cx = self.intrinsic[..., 0, 2]  # (b, q)
        cy = self.intrinsic[..., 1, 2]  # (b, q)

        invP[..., 0, 0] = (0.5 * self.width_px) / fx  # (b, q)
        invP[..., 1, 1] = (0.5 * self.height_px) / fy  # (b, q)
        invP[..., 0, 3] = -(2 * cx - self.width_px) / (2 * fx)  # (b, q)
        invP[..., 1, 3] = (2 * cy - self.height_px) / (2 * fy)  # (b, q)
        if not invert_z:
            invP[..., 2, 3] = -1
        else:
            invP[..., 2, 3] = 1

        if t_max is None or t_max >= INF:
            invP[..., 3, 2] = -1 / (2 * t_min)
            invP[..., 3, 3] = 1 / (2 * t_min)
        else:
            invP[..., 3, 2] = (t_min - t_max) / (2 * t_min * t_max)
            invP[..., 3, 3] = (t_min + t_max) / (2 * t_min * t_max)

        return invP


class RGBDImage:
    def __init__(
        self,
        rgb: T.Optional[torch.Tensor],  # (b, q, h, w, 3)  b: different scene, q: multiple imgs of same scene
        depth: T.Optional[torch.Tensor],  # (b, q, h, w)  z_c
        camera: Camera,  # (b, q)
        normal_w: T.Optional[torch.Tensor] = None,  # (b, q, h, w, 3)  surface normal in world coord
        hit_map: T.Optional[torch.Tensor] = None,  # (b, q, h, w)  bool, 1: valid
        feature: T.Optional[torch.Tensor] = None,  # (b, q, h, w, f)  feature
        other_maps: T.Optional[T.Dict[str, torch.Tensor]] = None,  # (b, q, h, w, d)
    ):
        self.rgb = rgb
        self.depth = depth
        self.camera = camera
        self.normal_w = normal_w
        self.hit_map = hit_map
        self.feature = feature
        self.pyramid = None
        self.other_maps = other_maps

        if self.hit_map is not None:
            assert self.hit_map.dtype == torch.bool, f"{self.hit_map.dtype=}"

        for tmp_name, tmp_arr in [("depth", self.depth), ("hit_map", self.hit_map)]:
            if tmp_arr is not None:
                assert tmp_arr.shape == self.shape, f"{tmp_name=}, {tmp_arr.shape=}"

        other_maps_list = [(k, v) for k, v in other_maps.items()] if other_maps is not None else []
        for tmp_name, tmp_arr in [("normal_w", self.normal_w), ("feature", self.feature)] + other_maps_list:
            if tmp_arr is not None:
                assert (tmp_arr.ndim in [4, 5]) and (tmp_arr.shape[:4] == self.shape), (
                    f"{tmp_name=}, {tmp_arr.ndim=}, {tmp_arr.shape=}, {self.shape=}"
                )

    @property
    def shape(self):
        if self.rgb is not None:
            b, q, h, w, _3 = self.rgb.shape
        elif self.normal_w is not None:
            b, q, h, w, _3 = self.normal_w.shape
        elif self.depth is not None:
            b, q, h, w = self.depth.shape
        elif self.hit_map is not None:
            b, q, h, w = self.hit_map.shape
        else:
            raise RuntimeError

        return b, q, h, w

    @property
    def device(self) -> torch.device:
        if self.rgb is not None:
            return self.rgb.device
        elif self.normal_w is not None:
            return self.normal_w.device
        elif self.depth is not None:
            return self.depth.device
        elif self.hit_map is not None:
            return self.hit_map.device
        else:
            raise RuntimeError

    @staticmethod
    def allclose(
        rgbd1: "RGBDImage",
        rgbd2: "RGBDImage",
        rtol: float = 1e-05,
        atol: float = 1e-06,
        rtol_depth: float = 1e-04,
        atol_depth: float = 1e-04,
        rtol_normal_angle: float = 3,  # degree
        equal_nan: bool = False,
        raise_error: bool = False,
    ) -> bool:
        """
        Check if all attributes (including cameras and other_maps) are the same in the two rgbd
        """

        # check camera
        is_camera_same = Camera.allclose(
            rgbd1.camera,
            rgbd2.camera,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            raise_error=raise_error,
        )
        if not is_camera_same:
            if raise_error:
                raise RuntimeError("camera not the same")
            return False

        for key in ["rgb", "hit_map", "feature"]:
            arr1 = getattr(rgbd1, key, None)
            arr2 = getattr(rgbd2, key, None)
            if arr1 is None or arr2 is None:
                if arr1 is not None or arr2 is not None:
                    if raise_error:
                        raise RuntimeError(f"{key}:\n{arr1}\n{arr2}\n")
                    return False
                else:
                    continue
            _result = torch.allclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=equal_nan)
            if not _result:
                if raise_error:
                    raise RuntimeError(f"{key}:\n{arr1}\n{arr2}\n")
                return False

        # for depth, we only care about the valid region
        for key in ["depth"]:
            arr1 = getattr(rgbd1, key, None)
            arr2 = getattr(rgbd2, key, None)
            if arr1 is None or arr2 is None:
                if arr1 is not None or arr2 is not None:
                    if raise_error:
                        raise RuntimeError(f"{key}:\n{arr1}\n{arr2}\n")
                    return False
                else:
                    continue

            if rgbd1.hit_map is not None:
                if key == "depth":
                    arr1 = arr1 * rgbd1.hit_map.to(dtype=arr1.dtype)
                else:
                    raise NotImplementedError

            if rgbd2.hit_map is not None:
                if key == "depth":
                    arr2 = arr2 * rgbd2.hit_map.to(dtype=arr2.dtype)
                else:
                    raise NotImplementedError

            _result = torch.allclose(
                arr1,
                arr2,
                rtol=rtol_depth,
                atol=atol_depth,
                equal_nan=equal_nan,
            )
            if not _result:
                if raise_error:
                    raise RuntimeError(f"{key}:\n{arr1}\n{arr2}\n")
                return False

        # for normal, we check the angle
        for key in ["normal_w"]:
            arr1 = getattr(rgbd1, key, None)
            arr2 = getattr(rgbd2, key, None)
            if arr1 is None or arr2 is None:
                if arr1 is not None or arr2 is not None:
                    if raise_error:
                        raise RuntimeError(f"{key}:\n{arr1}\n{arr2}\n")
                    return False
                else:
                    continue

            if rgbd1.hit_map is not None:
                if key == "normal_w":
                    arr1 = arr1 * rgbd1.hit_map.to(dtype=arr1.dtype).unsqueeze(-1) + (
                        1 - rgbd1.hit_map.to(dtype=arr1.dtype).unsqueeze(-1)
                    ) * (3**0.5)
                else:
                    raise NotImplementedError

            if rgbd2.hit_map is not None:
                if key == "normal_w":
                    arr2 = arr2 * rgbd2.hit_map.to(dtype=arr2.dtype).unsqueeze(-1) + (
                        1 - rgbd2.hit_map.to(dtype=arr2.dtype).unsqueeze(-1)
                    ) * (3**0.5)
                else:
                    raise NotImplementedError

            _result = (arr1 * arr2).sum(dim=-1).abs()  # (b, q, h, w)
            if not (_result > np.cos(rtol_normal_angle * np.pi / 180).item()).all():
                if raise_error:
                    raise RuntimeError(f"{key}:\n{arr1}\n{arr2}\n{_result}")
                return False

        other_map_keys1 = set(rgbd1.other_maps.keys()) if rgbd1.other_maps is not None else set()
        other_map_keys2 = set(rgbd2.other_maps.keys()) if rgbd2.other_maps is not None else set()
        all_other_map_keys = other_map_keys1.union(other_map_keys2)

        # check other_maps
        for key in all_other_map_keys:
            if rgbd1.other_maps is None:
                arr1 = None
            else:
                arr1 = getattr(rgbd1.other_maps, key, None)
            if rgbd2.other_maps is None:
                arr2 = None
            else:
                arr2 = getattr(rgbd2.other_maps, key, None)

            if arr1 is None or arr2 is None:
                if arr1 is not None or arr2 is not None:
                    if raise_error:
                        raise RuntimeError(f"{key}:\n{arr1}\n{arr2}\n")
                    return False
                else:
                    continue

            _result = torch.allclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=equal_nan)
            if not _result:
                if raise_error:
                    raise RuntimeError(f"{key}:\n{arr1}\n{arr2}\n")
                return False

        return True

    def contiguous(self):
        for key in [
            "rgb",
            "depth",
            "normal_w",
            "hit_map",
            "feature",
        ]:
            arr = getattr(self, key, None)
            if arr is not None:
                setattr(self, key, arr.contiguous())

        if self.camera is not None:
            self.camera = self.camera.contiguous()

        if self.other_maps is not None:
            for key, arr in self.other_maps.items():
                if arr is not None:
                    self.other_maps[key] = self.other_maps[key].contiguous()

        return self

    def remove_invalid(
        self,
        min_depth: float = None,
        max_depth: float = None,
        background_color: float = 0,
    ):
        """
        check nan and inf in rgbd, set the pixels with nan / inf to 0
        and hit_map to 0.
        """
        b, q, h, w = self.shape

        if self.hit_map is not None:
            valid_mask = torch.logical_and(
                self.hit_map >= 1e-6,
                self.hit_map.isfinite(),
            )  # (b, q, h, w)
        else:
            valid_mask = torch.ones(b, q, h, w, dtype=torch.bool, device=self.device)  # (b, q, h, w)

        arrs = [
            self.rgb,
            self.normal_w,
            self.depth,
            self.feature,
        ]

        if self.other_maps is not None:
            arrs += list(self.other_maps.values())

        for arr in arrs:
            if arr is None:
                continue

            if arr.ndim == 5:
                valid = arr.isfinite().all(dim=-1)  # (b, q, h, w)
            else:
                assert arr.ndim == 4
                valid = arr.isfinite()  # (b, q, h, w)

            valid_mask = torch.logical_and(
                valid_mask,
                valid,
            )  # (b, q, h, w)

        if self.depth is not None:
            if min_depth is not None:
                valid = self.depth >= min_depth
                valid_mask = torch.logical_and(
                    valid_mask,
                    valid,
                )  # (b, q, h, w)
            if max_depth is not None:
                valid = self.depth < max_depth
                valid_mask = torch.logical_and(
                    valid_mask,
                    valid,
                )  # (b, q, h, w)

        invalid_mask = ~valid_mask  # (b, q, h, w)

        # Note: since we set valid_depth with a hard threshold, we may still see
        # aliasing (sawtooth) pattern in the hit_map
        if self.hit_map is not None:
            self.hit_map = self.hit_map.masked_fill(invalid_mask, 0)

        # replace non hit
        if self.rgb is not None:
            self.rgb = self.rgb.masked_fill(invalid_mask.unsqueeze(-1), background_color)

        if self.normal_w is not None:
            self.normal_w = self.normal_w.masked_fill(invalid_mask.unsqueeze(-1), 1)
            self.normal_w = torch.nn.functional.normalize(self.normal_w, dim=-1)

        if self.depth is not None:
            self.depth = self.depth.masked_fill(invalid_mask, INF)

        if self.other_maps is not None:
            for key in self.other_maps:
                if self.other_maps[key] is not None:
                    self.other_maps[key] = self.other_maps[key].masked_fill(invalid_mask.unsqueeze(-1), 0)

        return self

    @linalg_utils.disable_tf32_and_autocast()
    def coordinate_transform(self, H_w2n: torch.Tensor):
        """
        Transform the coodinate system to "new", ie, multiply everything with H_w2n.
        The transformation is performed inplace.

        Args:
            H_w2n:
                (b, 4, 4)  convert the current world coordinate to the new coordinate

        Notes:
            We do not transform anything in other_maps nor feature.

            self.depth stores z_c, so we do not need to change
        """
        b, _41, _42 = H_w2n.shape
        self.camera.coordinate_transform(H_w2n=H_w2n)

        # (b, 4, 4) (b, q, h, w, 3)
        if self.normal_w is not None:
            self.normal_w = torch.nn.functional.normalize(
                linalg_utils.matmul(
                    H_w2n[..., :3, :3].reshape(b, 1, 1, 1, 3, 3).to(dtype=self.normal_w.dtype),  # (b, 1q, 1h, 1w, 3, 3)
                    self.normal_w.unsqueeze(-1),  # (b, q, h, w, 3, 1)
                ).squeeze(-1),
                dim=-1,
            )  # (b, q, h, w, 3xyz_n)

        return self

    def resize(
        self,
        new_width_px: int,
        new_height_px: int,
        make_hit_map_bool: bool = True,
        interpolation_mode: str = "bilinear",  # pytorch interpolate methods.
    ) -> "RGBDImage":
        """
        Resize the attributes and adjust the camera intrinsics.
        """
        b, q, h, w, _3 = self.rgb.shape
        scale_h = new_height_px / h
        scale_w = new_width_px / w

        new_dict = dict()
        # (b, q, h, w, c)
        for key in ["rgb", "normal_w", "feature"]:
            arr = getattr(self, key, None)
            if arr is None:
                new_dict[key] = None
                continue

            d = arr.size(-1)
            arr = arr.reshape(b * q, h, w, d).permute(0, 3, 1, 2)  # (bq, d, h, w)
            orig_dtype = arr.dtype
            with _nullcontext():
                arr = torch.nn.functional.interpolate(
                    arr.float(),  # (bq, d, h, w)
                    size=(new_height_px, new_width_px),
                    mode=interpolation_mode,
                    align_corners=False
                    if interpolation_mode in ["linear", "bilinear", "bicubic", "trilinear"]
                    else None,
                ).to(dtype=orig_dtype)  # (bq, d, h, w)
            new_dict[key] = arr.permute(0, 2, 3, 1).reshape(b, q, new_height_px, new_width_px, d)  # (b, q, h, w, d)

        # (b, q, h, w)
        for key in ["depth", "hit_map"]:
            arr = getattr(self, key, None)
            if arr is None:
                new_dict[key] = None
                continue

            arr = arr.reshape(b * q, 1, h, w)  # (b, d, h, w)
            orig_dtype = arr.dtype
            with _nullcontext():
                arr = torch.nn.functional.interpolate(
                    arr.float(),  # (b, d, h, w)
                    size=(new_height_px, new_width_px),
                    mode=interpolation_mode,
                    align_corners=False
                    if interpolation_mode in ["linear", "bilinear", "bicubic", "trilinear"]
                    else None,
                ).to(dtype=orig_dtype)  # (bq, d, h, w)
            new_dict[key] = arr.reshape(b, q, new_height_px, new_width_px)  # (b, q, h, w)

        if make_hit_map_bool and new_dict.get("hit_map", None) is not None:
            new_dict["hit_map"] = new_dict["hit_map"] >= 0.5

        # other maps
        new_other_maps = dict()
        if self.other_maps is not None:
            for key in self.other_maps:
                arr = self.other_maps[key]
                if arr is None:
                    new_other_maps[key] = None
                    continue

                d = arr.size(-1)
                arr = arr.reshape(b * q, h, w, d).permute(0, 3, 1, 2)  # (bq, d, h, w)
                orig_dtype = arr.dtype
                with _nullcontext():
                    arr = torch.nn.functional.interpolate(
                        arr.float(),  # (bq, d, h, w)
                        size=(new_height_px, new_width_px),
                        mode=interpolation_mode,
                        align_corners=False
                        if interpolation_mode in ["linear", "bilinear", "bicubic", "trilinear"]
                        else None,
                    ).to(orig_dtype)  # (bq, d, h, w)
                new_other_maps[key] = arr.permute(0, 2, 3, 1).reshape(
                    b, q, new_height_px, new_width_px, d
                )  # (b, q, h, w, d)
            new_dict["other_maps"] = new_other_maps
        else:
            new_dict["other_maps"] = None

        # camera
        new_dict["camera"] = self.camera.resize(
            new_height_px=new_height_px,
            new_width_px=new_width_px,
        )
        # new_camera = self.camera.clone()
        # new_camera.height_px = new_height_px
        # new_camera.width_px = new_width_px
        # new_camera.intrinsic[:, :, 0, :] = new_camera.intrinsic[:, :, 0, :] * scale_w
        # new_camera.intrinsic[:, :, 1, :] = new_camera.intrinsic[:, :, 1, :] * scale_h
        # new_dict["camera"] = new_camera

        new_rgbd = RGBDImage(**new_dict)
        return new_rgbd

    def set_other_map(self, name: str, arr: torch.Tensor):
        if self.other_maps is None:
            self.other_maps = dict()
        self.other_maps[name] = arr

    def get_other_map(self, name: str):
        if self.other_maps is None:
            return None
        else:
            return self.other_maps.get(name, None)

    @linalg_utils.disable_tf32_and_autocast()
    def compute_ray_normal_dot_product(self) -> T.Union[torch.Tensor, None]:
        """
        Compute the dot product between the normal_w and the camera ray

        Returns:
            (b, q, h, w) the dot product (cos(theta)) between normal_w and camera ray
        """
        if self.camera is None or self.normal_w is None:
            return None
        ray: Ray = self.camera.generate_camera_rays(device=self.normal_w.device)  # (b, q, h, w)
        ray_direction_w = ray.directions_w  # (b, q, h, w, 3)
        dot_prod = (ray_direction_w * self.normal_w).sum(dim=-1)  # (b, q, h, w)
        return dot_prod  # can be negative

    def index_select(self, dim: int, index: torch.Tensor) -> "RGBDImage":
        rgbd_image = self.clone()
        for attr_name in ["rgb", "depth", "normal_w", "hit_map", "camera", "feature"]:
            arr = getattr(rgbd_image, attr_name, None)
            if arr is not None:
                setattr(rgbd_image, attr_name, arr.index_select(dim=dim, index=index))

        rgbd_image.other_maps = utils.index_select_dict(
            dict_tensor=rgbd_image.other_maps,
            dim=dim,
            index=index,
        )
        return rgbd_image

    def chunk(self, chunks: int, dim: int = 0) -> T.List["RGBDImage"]:
        out_dict = dict()
        total = None
        for attr_name in ["rgb", "depth", "normal_w", "hit_map", "camera", "feature"]:
            arr = getattr(self, attr_name, None)
            if arr is not None:
                chunked_arr = arr.chunk(chunks=chunks, dim=dim)
                out_dict[attr_name] = chunked_arr
                if total is None:
                    total = len(chunked_arr)
                else:
                    assert len(chunked_arr) == total

        other_maps = utils.chunk_dict(
            dict_tensor=self.other_maps,
            chunks=chunks,
            dim=dim,
        )
        if other_maps is not None:
            assert len(other_maps) == total
        rgbd_images = []
        for i in range(total):
            d = dict(
                rgb=None,
                depth=None,
                camera=None,
            )
            for attr_name in out_dict:
                d[attr_name] = out_dict[attr_name][i]
            if other_maps is not None:
                d["other_maps"] = other_maps[i]
            rgbd_image = RGBDImage(**d)
            rgbd_images.append(rgbd_image)
        return rgbd_images

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> "RGBDImage":
        for attr_name in ["rgb", "depth", "normal_w", "hit_map", "camera", "feature"]:
            arr = getattr(self, attr_name, None)
            if arr is not None and attr_name != "hit_map":
                setattr(self, attr_name, arr.to(device=device, dtype=dtype))
            elif attr_name == "hit_map":
                setattr(self, attr_name, arr.to(device=device))
        if device is not None:
            self.other_maps = utils.to_device(self.other_maps, device=device)
        if dtype is not None:
            self.other_maps = utils.to_numpy(self.other_maps, dtype=dtype)
        return self

    def detach(self) -> "RGBDImage":
        for attr_name in ["rgb", "depth", "normal_w", "hit_map", "camera", "feature"]:
            arr = getattr(self, attr_name, None)
            if arr is not None:
                setattr(self, attr_name, arr.detach())
        self.other_maps = utils.detach_dict(self.other_maps)
        return self

    def clone(self) -> "RGBDImage":
        return RGBDImage(
            rgb=self.rgb.clone() if self.rgb is not None else None,
            depth=self.depth.clone() if self.depth is not None else None,
            camera=self.camera.clone() if self.camera is not None else None,
            normal_w=self.normal_w.clone() if self.normal_w is not None else None,
            hit_map=self.hit_map.clone() if self.hit_map is not None else None,
            feature=self.feature.clone() if self.feature is not None else None,
            other_maps=utils.clone_dict(self.other_maps),
        )

    @staticmethod
    def cat(rgbd_images: T.List["RGBDImage"], dim: int) -> "RGBDImage":
        out = dict()
        for name in ["rgb", "depth", "normal_w", "hit_map", "feature"]:
            arr = [getattr(r, name, None) for r in rgbd_images]
            if None in arr:
                out[name] = None
            else:
                out[name] = torch.cat(arr, dim=dim)

        out["other_maps"] = utils.cat_dict(
            dict_list=[rgbd.other_maps for rgbd in rgbd_images],
            dim_dict=dim,
        )

        # concat camera
        cameras = [getattr(r, "camera", None) for r in rgbd_images]
        assert None not in cameras
        out["camera"] = Camera.cat(cameras, dim=dim)
        return RGBDImage(**out)

    def expand(self, n: int, dim: int) -> "RGBDImage":
        for name in ["rgb", "depth", "normal_w", "hit_map", "feature"]:
            arr = getattr(self, name, None)
            if arr is not None:
                assert arr.size(dim) == 1, f"{arr.shape}"
                setattr(self, name, arr.expand(*arr.shape[:dim], n, *arr.shape[dim + 1 :]))

        if self.other_maps is not None:
            for key, arr in self.other_maps.items():
                if arr is not None:
                    assert arr.size(dim) == 1, f"{arr.shape}"
                    self.other_maps[key] = arr.expand(*arr.shape[:dim], n, *arr.shape[dim + 1 :])

        # camera
        self.camera = self.camera.expand(n=n, dim=dim)
        return self

    def contiguous(self) -> "RGBDImage":
        for name in ["rgb", "depth", "normal_w", "hit_map", "feature"]:
            arr = getattr(self, name, None)
            if arr is not None:
                setattr(self, name, arr.contiguous())

        if self.other_maps is not None:
            for key, arr in self.other_maps.items():
                if arr is not None:
                    self.other_maps[key] = arr.contiguous()

        # camera
        self.camera = self.camera.contiguous()
        return self

    def __getitem__(self, index):
        """
        Given index returns that batch as an RGBDImage class insance.
        Input: index: int
        Output: RGBDImage
        """
        out = dict()
        for name in ["rgb", "depth", "normal_w", "hit_map", "feature"]:
            arr = getattr(self, name, None)
            if arr is None:
                out[name] = None
            else:
                out[name] = arr[index][None]

        # out["other_maps"] = utils.cat_dict(
        #     dict_list=[rgbd.other_maps for rgbd in rgbd_images],
        #     dim_dict=dim,
        # )

        # concat camera
        cameras = getattr(self, "camera", None)
        assert cameras is not None
        out["camera"] = cameras[index]
        return RGBDImage(**out)

    @staticmethod
    def optimize_exposure_whitebalance(
        rgbd_images: T.List["RGBDImage"],
        ref_idxs: T.Optional[T.Union[int, T.List[int]]],
        correction_type: str = "wrgb",
        img_order: str = "temporal",
        num_ref_to_use: int = 5,
        num_random_ref_to_use: int = 5,
        min_iter: int = 10,
        max_iter: int = 100,
        num_pixels: int = 100,
        blur_sigma: float = 10,
        loss_type: str = "huber",
        loss_outlier_percentage: float = 0.2,  # disgard 20% highest loss
        lr: float = 1e-3,
        device: torch.device = None,
        th_loss: float = 1e-6,
        timestamps: T.List[T.Union[str, float, int]] = None,
        ref_rgbd_images: T.List["RGBDImage"] = None,  # if given, will ignore ref_idxs
        ref_timestamps: T.List[T.Union[str, float, int]] = None,
        ref_blur_sigma: float = 10.0,
        print_every_iter: int = 1,
    ) -> T.Dict[str, T.Any]:
        r"""
        Given a list of rgbd_images, each of which might be captured using different
        exposure and white balancing settings, find the correction such that after
        applied the correction the rgbd_images look similar in terms of exposure and
        white balancing.

        Args:
            rgbd_images:
                list of rgbd_images (b=1, q_i=1, h_i, w_i)  i=1...m
            ref_idxs:
                the indexes of rgbd_images to be used as the reference.
            correction_type:
                'wrgb': the correction is 3 scalars \in [0, 1] that multiply to RGB channels separately
            img_order:
                'temporal': assume the input rgbd_images is a video, choose one image at a time
                'overlapping': calculate the overlapping of field of view using depth map to consider occlusion
            max_iter:
                number of iterations to optimize an image
            blur_sigma:
                gaussian kernel sigma to blur the kernel before matching.
                -1: no blur

        Returns:
            rgbd_images:
                list of rgbd_image, the corrected new rgbd_images
            correction:
                the correction parameter used
        """

        num_images = len(rgbd_images)
        if num_images == 0:
            return dict()
        ori_device = rgbd_images[0].rgb.device

        if isinstance(ref_idxs, int):
            ref_idxs = [ref_idxs]

        # currently support b=1, q=1
        for i in range(num_images):
            assert rgbd_images[i].rgb.size(0) == 1  # b == 1
            assert rgbd_images[i].rgb.size(1) == 1  # q == 1
            assert rgbd_images[i].rgb.device == ori_device

        if device is None:
            device = ori_device

        # create a copy to directly work on rgbd_images
        rgbd_images = [rgbd.clone().to(device=device) for rgbd in rgbd_images]
        total_rgbd = len(rgbd_images)

        if ref_rgbd_images is not None:
            ref_rgbd_images = [rgbd.clone().to(device=device) for rgbd in ref_rgbd_images]
            total_ref_rgbd = len(ref_rgbd_images)
        else:
            total_ref_rgbd = 0

        # compute overlapping
        if img_order == "overlapping":
            assert False
        else:
            overlapping = None

        # blur images
        if blur_sigma > 0:
            # generate gaussian kernel
            blur_kernel_size = max(1, int(blur_sigma * 4))
            blur_kernel_size = (blur_kernel_size // 2) * 2 + 1  # make sure kernel size is odd
            blur_kernal = sp.signal.windows.gaussian(M=blur_kernel_size, std=blur_sigma, sym=True)  # (k, )
            blur_kernal = torch.from_numpy(blur_kernal).float().to(device=device)
            blur_kernal = torch.outer(blur_kernal, blur_kernal)  # (k, k)
            # normalize sum to 1
            blur_kernal = blur_kernal / blur_kernal.sum()  # (k, k)

            for i in range(num_images):
                rgb = rgbd_images[i].rgb  # (b, q, h, w, c)
                b, q, h, w, c = rgb.shape
                bq = b * q
                rgb = rgb.reshape(bq, h, w, c).permute(0, 3, 1, 2)  # (bq, c, h, w)
                rgb = torch.nn.functional.conv2d(
                    rgb,
                    blur_kernal.expand(c, 1, -1, -1),
                    groups=c,
                    padding="same",
                )
                rgbd_images[i].rgb = rgb.permute(0, 2, 3, 1).reshape(b, q, h, w, c)

        else:
            blur_kernel_size = 0

        if ref_blur_sigma > 0:
            # generate gaussian kernel
            ref_blur_kernel_size = max(1, int(ref_blur_sigma * 4))
            ref_blur_kernel_size = (ref_blur_kernel_size // 2) * 2 + 1  # make sure kernel size is odd
            blur_kernal = sp.signal.windows.gaussian(M=ref_blur_kernel_size, std=ref_blur_sigma, sym=True)  # (k, )
            blur_kernal = torch.from_numpy(blur_kernal).float().to(device=device)
            blur_kernal = torch.outer(blur_kernal, blur_kernal)  # (k, k)
            # normalize sum to 1
            blur_kernal = blur_kernal / blur_kernal.sum()  # (k, k)

            if ref_rgbd_images is not None:
                for i in range(len(ref_rgbd_images)):
                    rgb = ref_rgbd_images[i].rgb  # (b, q, h, w, c)
                    b, q, h, w, c = rgb.shape
                    bq = b * q
                    rgb = rgb.reshape(bq, h, w, c).permute(0, 3, 1, 2)  # (bq, c, h, w)
                    rgb = torch.nn.functional.conv2d(
                        rgb,
                        blur_kernal.expand(c, 1, -1, -1),
                        groups=c,
                        padding="same",
                    )
                    ref_rgbd_images[i].rgb = rgb.permute(0, 2, 3, 1).reshape(b, q, h, w, c)
        else:
            ref_blur_kernel_size = 0

        # # create corerction modules
        correctors = []  # (num_images)
        for i in range(num_images):
            if ref_idxs is not None and i in ref_idxs:
                corrector = ColorCorrector(correction_type="identify")
            else:
                corrector = ColorCorrector(correction_type=correction_type)
            correctors.append(corrector.to(device=device))

        # loss
        if loss_type == "l2":
            loss_fn = torch.nn.MSELoss(reduction="none")
        elif loss_type == "l1":
            loss_fn = torch.nn.L1Loss(reduction="none")
        elif loss_type == "huber":
            loss_fn = torch.nn.HuberLoss(reduction="none")
        else:
            raise NotImplementedError

        # our strategy:
        # we iteratively correct one image at a time in the list,
        # once it is done, we add it into the reference set
        ref_idxs = set(ref_idxs) if ref_idxs is not None else None
        finished_idxs = []
        if ref_idxs is not None:
            for i in ref_idxs:
                finished_idxs.append(i)
        all_idxs = set(range(num_images))

        if ref_rgbd_images is None:
            ref_blur_kernel_size = blur_kernel_size

        while len(finished_idxs) < num_images:
            # find the image with the highest coverage of the reference point cloud
            if ref_rgbd_images is None:
                assert ref_idxs is not None
                if img_order == "temporal":
                    candidate_idx = min(all_idxs.difference(ref_idxs))
                    # candidate_idx = max(ref_idxs) + 1
                    if timestamps is None:
                        ridxs = np.array(list(ref_idxs))
                        iis = np.argsort(-1 * ridxs)
                        niis = iis[:num_ref_to_use]  # take those with largest indexes
                        ref_idx = [ridxs[i] for i in niis]
                    else:
                        # find the ref_idx that is closest to candidate idx temporally
                        candidate_timestamp = float(timestamps[candidate_idx])  # (,)
                        ridxs = list(ref_idxs)
                        ref_timestamps = np.array([float(timestamps[idx]) for idx in ridxs])  # (n,)
                        diff_times = np.abs(ref_timestamps - candidate_timestamp)  # (n,)
                        iis = np.argsort(diff_times)  # (n,) small to large
                        niis = iis[:num_ref_to_use]
                        ref_idx = [ridxs[i] for i in niis]

                    # add random ref
                    if num_random_ref_to_use > 0:
                        ll = set([iis[i] for i in range(min(500, len(iis)))])  # take the closest 500 frames
                        ridxs = list(ll.difference(set(ref_idx)))
                        np.random.shuffle(ridxs)
                        ridxs = ridxs[:num_random_ref_to_use]
                        for idx in ridxs:
                            ref_idx.append(idx)
                else:
                    raise NotImplementedError

                ref_rgbd = [rgbd_images[idx] for idx in ref_idx]  # (b, qr, h, w, c)
                ref_rgbd = RGBDImage.cat(ref_rgbd, dim=1)  # (b, qr, h, w, c)
            else:
                assert ref_timestamps is not None
                candidate_idx = min(all_idxs.difference(set(finished_idxs)))
                # find the ref_idx that is closest to candidate idx temporally
                candidate_timestamp = float(timestamps[candidate_idx])  # (,)
                ridxs = list(range(len(ref_rgbd_images)))
                r_timestamps = np.array([float(ref_timestamps[idx]) for idx in ridxs])  # (n,)
                diff_times = np.abs(r_timestamps - candidate_timestamp)  # (n,)
                iis = np.argsort(diff_times)  # (n,) small to large
                niis = iis[:num_ref_to_use]
                ref_idx = [ridxs[i] for i in niis]

                # add random ref
                if num_random_ref_to_use > 0:
                    ll = set([iis[i] for i in range(min(500, len(iis)))])  # take the closest 500 frames
                    ridxs = list(ll.difference(set(ref_idx)))
                    np.random.shuffle(ridxs)
                    ridxs = ridxs[:num_random_ref_to_use]
                    for idx in ridxs:
                        ref_idx.append(idx)

                ref_rgbd = [ref_rgbd_images[idx] for idx in ref_idx]  # (b, qr, h, w, c)
                ref_rgbd = RGBDImage.cat(ref_rgbd, dim=1)  # (b, qr, h, w, c)

            ref_rgb = ref_rgbd.rgb  # (b, qr, hr, wr, c), color corrected
            # ref_depth = ref_rgbd.depth  # (b, qr, hr, wr)
            ref_intrinsics = ref_rgbd.camera.intrinsic  # (b, qr, 3, 3)
            ref_H_c2w = ref_rgbd.camera.H_c2w  # (b, qr, 4, 4)
            b, qr, hr, wr, c = ref_rgb.shape
            candidate_rgb = rgbd_images[candidate_idx].rgb  # (b, qc, hc, wc, c)
            candidate_depth = rgbd_images[candidate_idx].depth  # (b, qc, hc, wc)
            candidate_hit = rgbd_images[candidate_idx].hit_map  # (b, qc, hc, wc) bool
            candidate_intrinsics = rgbd_images[candidate_idx].camera.intrinsic  # (b, qc, 3, 3)
            candidate_H_c2w = rgbd_images[candidate_idx].camera.H_c2w  # (b, qc, 4, 4)
            b, qc, hc, wc, c = candidate_rgb.shape
            assert hr >= 2 * ref_blur_kernel_size
            assert wr >= 2 * ref_blur_kernel_size
            assert hc >= 2 * blur_kernel_size
            assert wc >= 2 * blur_kernel_size

            # create an optimizer
            corrector = correctors[candidate_idx]
            optimizer = torch.optim.Adam(
                corrector.parameters(),
                lr=lr,
                betas=(0.9, 0.98),
                eps=1e-9,
            )

            loss = 0
            for iter in range(max_iter):
                # randomly select a few pixels in the candidate image
                # we use only pixels from [blur_kernel_size, h-blur_kernel_size] to avoid
                # the boundary issue of gaussian blur
                u = (
                    torch.randint(
                        low=blur_kernel_size,
                        high=wc - blur_kernel_size,
                        size=(num_pixels,),
                        device=device,
                    ).float()
                    + 0.5
                )  # (qr, num_pixels,)
                v = (
                    torch.randint(
                        low=blur_kernel_size,
                        high=hc - blur_kernel_size,
                        size=(num_pixels,),
                        device=device,
                    ).float()
                    + 0.5
                )  # (num_pixels,)
                uv_c = torch.stack([u, v], dim=-1).expand(b, num_pixels, 2)  # (b=1, num_pixels, 2)

                # get the rgb (before correction) on the candidate image
                assert candidate_rgb.size(1) == 1  # qc == 1
                ori_rgb = utils.uv_sampling(
                    uv=uv_c,  # (b=1, num_pixel, 2)
                    feature_map=candidate_rgb.squeeze(1),  # (b=1, hc, wc, c)
                    uv_normalized=False,
                ).unsqueeze(1)  # (b=1, qc=1, num_pixel, c)
                est_rgb = corrector(ori_rgb)  # (b=1, qc=1, num_pixel, c)
                candidate_valid_mask = (
                    utils.uv_sampling(
                        uv=uv_c,  # (b=1, num_pixel, 2)
                        feature_map=candidate_hit.float().squeeze(1).unsqueeze(-1),  # (b=1, hc, wc, 1)
                        uv_normalized=False,
                    )
                    .unsqueeze(1)
                    .squeeze(-1)
                )  # (b=1, qc=1, num_pixel)
                candidate_valid_mask = candidate_valid_mask > 0.5  # (b=1, qc=1, num_pixel)

                # find correponding uv in the reference images
                uv_r, _ = utils.find_corresponding_uv(
                    uv_c=uv_c,  # (b, num_pixels, 2)
                    z_map=candidate_depth.squeeze(1),  # (b, hc, wc)
                    intrinsics_from=candidate_intrinsics.squeeze(1),  # (b, 3, 3)
                    H_c2w_from=candidate_H_c2w.squeeze(1),  # (b, 4, 4)
                    intrinsics_to=ref_intrinsics,  # (b, qr, 3, 3)
                    H_c2w_to=ref_H_c2w,  # (b, qr, 4, 4)
                    dim_b=1,
                )  # (b, qr, num_pixel, 2)  [0, wr]  [0, hr]

                # calculate the valid_mask
                gt_valid_mask = torch.logical_and(
                    uv_r[..., 0] >= ref_blur_kernel_size,
                    uv_r[..., 0] < wr - ref_blur_kernel_size,
                )  # (b, qr, num_pixel)
                gt_valid_mask = torch.logical_and(
                    gt_valid_mask,
                    uv_r[..., 1] >= ref_blur_kernel_size,
                )
                gt_valid_mask = torch.logical_and(
                    gt_valid_mask,
                    uv_r[..., 1] < hr - ref_blur_kernel_size,
                )  # (b, qr, num_pixel)
                gt_valid_mask = torch.logical_and(
                    gt_valid_mask,  # (b, qr, num_pixel)
                    candidate_valid_mask,  # (b, qc=1, num_pixel)
                )
                gt_total_valid = torch.clamp(gt_valid_mask.sum() * c, min=1)  # considered qr and c

                # get color on the reference images
                assert uv_r.size(0) == 1  # b == 1
                gt_rgb = utils.uv_sampling(
                    uv=uv_r.squeeze(0),  # (qr, num_pixel, 2)
                    feature_map=ref_rgb.squeeze(0),  # (qr, hr, wr, c)
                    uv_normalized=False,
                ).unsqueeze(0)  # (b=1, qr, num_pixel, c)

                # calculate the rgb loss
                loss = loss_fn(est_rgb.expand(b, qr, num_pixels, c), gt_rgb)  # (b, qr, num_pixel, c)

                if loss_outlier_percentage is None or loss_outlier_percentage < 1.0e-4:
                    # loss = loss.masked_fill(
                    #     torch.logical_not(gt_valid_mask.unsqueeze(-1)),
                    #     0,
                    # ).sum() / gt_total_valid
                    loss = (loss * gt_valid_mask.unsqueeze(-1)).sum() / gt_total_valid
                    total_valid = gt_total_valid  # considered qr and c
                else:
                    # disgard top `loss_outlier_percentage` outlier loss

                    # bug!!  need to handle masked values
                    valid_ratio = gt_total_valid / loss.numel()
                    adjusted_loss_outlier_percentage = valid_ratio * loss_outlier_percentage

                    loss = loss * gt_valid_mask.unsqueeze(-1)  # (b, qr, num_pixel, c)
                    th = torch.quantile(loss, q=(1 - adjusted_loss_outlier_percentage))
                    loss_valid_mask = loss < th  # (b, qr, num_pixel, c)
                    loss = loss * loss_valid_mask
                    total_valid = torch.logical_and(loss_valid_mask, gt_valid_mask.unsqueeze(-1)).sum().clamp(min=1)
                    loss = loss.sum() / total_valid

                if loss.detach().cpu() < th_loss:
                    if (print_every_iter > 0 and iter % print_every_iter == 0) or (iter >= min_iter):
                        print(
                            f"{len(finished_idxs)} / {num_images}, {iter}: "
                            f"{loss.detach().cpu().numpy():.4e} < th = {th_loss:.4e}, break"
                        )
                    if iter >= min_iter:
                        break
                else:
                    # update
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if print_every_iter > 0 and iter % print_every_iter == 0:
                        print(
                            f"{len(finished_idxs)} / {num_images}, {iter}: "
                            f"{loss.detach().cpu().numpy():.4e} "
                            f"total_valid = {total_valid.detach().cpu().item()} ({total_valid / num_pixels / c / qr * 100:.2f}%)"
                        )

            if isinstance(loss, torch.Tensor):
                loss = loss.detach().cpu().numpy()
            print(f"{len(finished_idxs)} / {num_images}: {loss:.4e}")

            # correct the entire image and add to the ref
            with torch.no_grad():
                rgbd_images[candidate_idx].rgb = corrector(candidate_rgb).detach()  # (b, qc, hc, wc, c)

            # add candidate_idx to ref_idxs
            if ref_idxs is not None:
                ref_idxs.add(candidate_idx)
            finished_idxs.append(candidate_idx)

        # send to the original device
        rgbd_images = [rgbd.to(device=ori_device) for rgbd in rgbd_images]
        correctors = [corrector.to(device=ori_device) for corrector in correctors]

        return dict(
            rgbd_images=rgbd_images,  # blurred
            correctors=correctors,
        )

    @staticmethod
    def optimize_exposure_whitebalance_v2(
        rgbd_images: T.List["RGBDImage"],
        ref_idxs: T.Optional[T.Union[int, T.List[int]]],
        correction_type: str = "wrgb",
        img_order: str = "overlapping",
        num_ref_to_use: int = 5,
        min_iter: int = 500,
        max_iter: int = 5000,
        num_pixels: int = 1000,
        blur_sigma: float = 1,
        loss_type: str = "huber",
        loss_outlier_percentage: float = 0.1,  # disgard 20% highest loss
        lr: float = 1e-4,
        device: torch.device = None,
        th_loss: float = 1e-5,
        th_diopter: float = 0.25,
        max_loss_for_ref: float = 1e-4,  # max loss to be valid to be added into future ref
        ref_rgbd_images: T.List["RGBDImage"] = None,  # if given, will ignore ref_idxs
        ref_blur_sigma: float = None,
        print_every_iter: int = 500,
        max_retry: int = 1,
        min_overlapping: float = 0.25,  # min overlapping to be considered a valid ref
    ) -> T.Dict[str, T.Any]:
        r"""
        Given a list of rgbd_images, each of which might be captured using different
        exposure and white balancing settings, find the correction such that after
        applied the correction the rgbd_images look similar in terms of exposure and
        white balancing.

        Args:
            rgbd_images:
                list of rgbd_images (b=1, q_i=1, h_i, w_i)  i=1...m
            ref_idxs:
                the indexes of rgbd_images to be used as the reference.
            correction_type:
                'wrgb': the correction is 3 scalars \in [0, 1] that multiply to RGB channels separately
            img_order:
                'overlapping': calculate the overlapping of field of view using depth map to consider occlusion
            max_iter:
                number of iterations to optimize an image
            blur_sigma:
                gaussian kernel sigma to blur the kernel before matching.
                -1: no blur

        Returns:
            rgbd_images:
                list of rgbd_image, the corrected new rgbd_images
            correction:
                the correction parameter used
            loss:
        """

        if ref_blur_sigma is None:
            ref_blur_sigma = blur_sigma

        num_images = len(rgbd_images)
        if num_images == 0:
            return dict()
        ori_device = rgbd_images[0].rgb.device

        if ref_idxs is None:
            ref_idxs = []
        elif isinstance(ref_idxs, int):
            ref_idxs = [ref_idxs]

        # currently support b=1, q=1
        for i in range(num_images):
            assert rgbd_images[i].rgb.size(0) == 1  # b == 1
            assert rgbd_images[i].rgb.size(1) == 1  # q == 1
            assert rgbd_images[i].rgb.device == ori_device

        if device is None:
            device = ori_device

        # create a copy to directly work on rgbd_images
        rgbd_images = [rgbd.clone().to(device=device) for rgbd in rgbd_images]
        n = len(rgbd_images)

        if ref_rgbd_images is not None:
            ref_rgbd_images = [rgbd.clone().to(device=device) for rgbd in ref_rgbd_images]
            nr = len(ref_rgbd_images)
        else:
            nr = 0

        # compute overlapping
        if img_order == "overlapping":
            z_map = []
            intrinsic = []
            H_c2w = []
            if ref_rgbd_images is not None:
                z_map += [rgbd_image.depth for rgbd_image in ref_rgbd_images]
                intrinsic += [rgbd_image.camera.intrinsic for rgbd_image in ref_rgbd_images]
                H_c2w += [rgbd_image.camera.H_c2w for rgbd_image in ref_rgbd_images]

            z_map += [rgbd_image.depth for rgbd_image in rgbd_images]  # list of (b=1, q=1, h, w)
            intrinsic += [rgbd_image.camera.intrinsic for rgbd_image in rgbd_images]  # list of (b=1, q=1, 3, 3)
            H_c2w += [rgbd_image.camera.H_c2w for rgbd_image in rgbd_images]  # list of (b=1, q=1, 4, 4)

            z_map = torch.cat(z_map, dim=1).squeeze(0)  # (nr+n, h, w)
            intrinsic = torch.cat(intrinsic, dim=1).squeeze(0)  # (nr+n, 3, 3)
            H_c2w = torch.cat(H_c2w, dim=1).squeeze(0)  # (nr+n, 4, 4)

            overlapping = utils.compute_fov_overlapping_with_depth_map(
                z_map=z_map,
                intrinsic=intrinsic,
                H_c2w=H_c2w,
                num_points=num_pixels,
                th_diopter=0.1,
            )  # (nr+n, nr+n),  ref at the beginning
        else:
            overlapping = None

        # blur images
        if blur_sigma > 0:
            # generate gaussian kernel
            blur_kernel_size = max(1, int(blur_sigma * 4))
            blur_kernel_size = (blur_kernel_size // 2) * 2 + 1  # make sure kernel size is odd
            blur_kernal = sp.signal.windows.gaussian(M=blur_kernel_size, std=blur_sigma, sym=True)  # (k, )
            blur_kernal = torch.from_numpy(blur_kernal).float().to(device=device)
            blur_kernal = torch.outer(blur_kernal, blur_kernal)  # (k, k)
            # normalize sum to 1
            blur_kernal = blur_kernal / blur_kernal.sum()  # (k, k)

            for i in range(len(rgbd_images)):
                rgb = rgbd_images[i].rgb  # (b, q, h, w, c)
                b, q, h, w, c = rgb.shape
                bq = b * q
                rgb = rgb.reshape(bq, h, w, c).permute(0, 3, 1, 2)  # (bq, c, h, w)
                rgb = torch.nn.functional.conv2d(
                    rgb,
                    blur_kernal.expand(c, 1, -1, -1),
                    groups=c,
                    padding="same",
                )
                rgbd_images[i].rgb = rgb.permute(0, 2, 3, 1).reshape(b, q, h, w, c)

        else:
            blur_kernel_size = 0

        if ref_blur_sigma > 0:
            # generate gaussian kernel
            ref_blur_kernel_size = max(1, int(ref_blur_sigma * 4))
            ref_blur_kernel_size = (ref_blur_kernel_size // 2) * 2 + 1  # make sure kernel size is odd
            blur_kernal = sp.signal.windows.gaussian(M=ref_blur_kernel_size, std=ref_blur_sigma, sym=True)  # (k, )
            blur_kernal = torch.from_numpy(blur_kernal).float().to(device=device)
            blur_kernal = torch.outer(blur_kernal, blur_kernal)  # (k, k)
            # normalize sum to 1
            blur_kernal = blur_kernal / blur_kernal.sum()  # (k, k)

            if ref_rgbd_images is not None:
                for i in range(len(ref_rgbd_images)):
                    rgb = ref_rgbd_images[i].rgb  # (b, q, h, w, c)
                    b, q, h, w, c = rgb.shape
                    bq = b * q
                    rgb = rgb.reshape(bq, h, w, c).permute(0, 3, 1, 2)  # (bq, c, h, w)
                    rgb = torch.nn.functional.conv2d(
                        rgb,
                        blur_kernal.expand(c, 1, -1, -1),
                        groups=c,
                        padding="same",
                    )
                    ref_rgbd_images[i].rgb = rgb.permute(0, 2, 3, 1).reshape(b, q, h, w, c)
        else:
            ref_blur_kernel_size = 0

        # combine ref rgbd and rgbd
        rgbds = ref_rgbd_images + rgbd_images if ref_rgbd_images else rgbd_images
        ref_idxs = [idx + nr for idx in ref_idxs]
        ref_idxs += list(range(nr))

        # loss
        if loss_type == "l2":
            loss_fn = torch.nn.MSELoss(reduction="none")
        elif loss_type == "l1":
            loss_fn = torch.nn.L1Loss(reduction="none")
        elif loss_type == "huber":
            loss_fn = torch.nn.HuberLoss(reduction="none")
        else:
            raise NotImplementedError

        # our strategy:
        # we iteratively correct one image at a time in the list,
        # once it is done, we add it into the reference set
        assert overlapping is not None

        # if no ref_idx is given, use the image with the highest overlapping
        if len(ref_idxs) == 0:
            idx = torch.argmax(overlapping.sum(dim=1)).item()
            ref_idxs.append(idx)

        # create corerction modules
        correctors = []  # (nr+n)
        for i in range(nr + n):
            if i in ref_idxs:
                corrector = ColorCorrector(correction_type="identify")
            else:
                corrector = ColorCorrector(correction_type=correction_type)
            correctors.append(corrector.to(device=device))

        retry = 0
        max_retry = max_retry if max_retry > 0 else 1
        final_losses = [-1] * (n + nr)
        while len(ref_idxs) < (n + nr) and retry < max_retry:
            seen_idxs = copy.deepcopy(ref_idxs)
            lr = lr * 0.5
            max_iter = int(max_iter * 1.25)
            while len(seen_idxs) < (n + nr):
                # find the image with the highest coverage of the reference point cloud
                # print(f'seen_idxs = {seen_idxs}')
                # print(f'ref_idxs = {ref_idxs}')

                potential_candidate_mask = torch.ones(n + nr, dtype=torch.bool, device=device)  # (n+nr,)
                potential_candidate_mask[seen_idxs] = 0
                potential_candidate_idxs = torch.arange(n + nr, device=device)
                potential_candidate_idxs = potential_candidate_idxs[potential_candidate_mask]  # (num_candidate,)

                # print(f'potential_candidate_idxs = {potential_candidate_idxs}')
                # for i in potential_candidate_idxs:
                #     assert i not in seen_idxs

                # print(f'overlapping[potential_candidate_mask, ref_idxs].shape = {overlapping[potential_candidate_mask, ref_idxs].shape}')
                tmp_overlapping = overlapping[:, ref_idxs]  # (n+nr, num_ref)
                tmp_overlapping = tmp_overlapping[potential_candidate_mask]  # (num_candidates, num_ref)
                # print(f'tmp_overlapping1.shape = {tmp_overlapping.shape}')
                # assert tmp_overlapping.size(1) == len(ref_idxs)

                # sort the overlapping from high to low
                tmp_overlapping, tmp_ref_idxs = torch.sort(
                    tmp_overlapping, dim=-1, descending=True
                )  # (num_candidates, num_ref), (num_candidates, num_ref)
                # print(f'tmp_overlapping2.shape = {tmp_overlapping.shape}')
                # print(f'tmp_ref_idxs.shape = {tmp_ref_idxs.shape}')
                # assert tmp_ref_idxs.size(1) == len(ref_idxs)

                # use only num_ref
                tmp_overlapping = tmp_overlapping[:, :num_ref_to_use]  # (num_candidates, num_ref)

                # use the sum of the top num_ref overlapping
                max_idx = torch.argmax(tmp_overlapping.sum(dim=1)).item()
                candidate_idx = potential_candidate_idxs[max_idx].item()
                # print(f'candidate_idx = {candidate_idx}')

                tmp_ref_idxs = tmp_ref_idxs[max_idx, :num_ref_to_use]
                # print(f'tmp_ref_idxs2.shape = {tmp_ref_idxs.shape}')
                tmp_ref_idxs = [ref_idxs[idx] for idx in tmp_ref_idxs]
                # print(f'tmp_ref_idxs2 = {tmp_ref_idxs}')
                # print(f'overlapping = {tmp_overlapping[max_idx, :num_ref_to_use]}')

                # remove small overlapping from ref
                _tmp_ref_idxs = [tmp_ref_idxs[0]]
                for i in range(1, len(tmp_ref_idxs)):
                    ov = tmp_overlapping[max_idx, i]
                    if ov >= min_overlapping:
                        _tmp_ref_idxs.append(tmp_ref_idxs[i])
                    else:
                        break
                tmp_ref_idxs = _tmp_ref_idxs

                ref_rgbd = [rgbds[idx] for idx in tmp_ref_idxs]  # (b=1, qr, h, w, c)
                ref_rgbd = RGBDImage.cat(ref_rgbd, dim=1)  # (b=1, qr, h, w, c)
                ref_rgb = ref_rgbd.rgb  # (b, qr, hr, wr, c), color corrected
                ref_depth = ref_rgbd.depth  # (b, qr, hr, wr)
                ref_intrinsics = ref_rgbd.camera.intrinsic  # (b, qr, 3, 3)
                ref_H_c2w = ref_rgbd.camera.H_c2w  # (b, qr, 4, 4)
                b, qr, hr, wr, c = ref_rgb.shape

                candidate_rgb = rgbds[candidate_idx].rgb  # (b=1, qc=1, hc, wc, c)
                candidate_depth = rgbds[candidate_idx].depth  # (b=1, qc=1, hc, wc)
                candidate_hit = rgbds[candidate_idx].hit_map  # (b, qc, hc, wc) bool
                candidate_intrinsics = rgbds[candidate_idx].camera.intrinsic  # (b, qc, 3, 3)
                candidate_H_c2w = rgbds[candidate_idx].camera.H_c2w  # (b, qc, 4, 4)
                b, qc, hc, wc, c = candidate_rgb.shape
                assert b == 1
                assert qc == 1

                assert hr >= 2 * ref_blur_kernel_size
                assert wr >= 2 * ref_blur_kernel_size
                assert hc >= 2 * blur_kernel_size
                assert wc >= 2 * blur_kernel_size

                # create an optimizer
                corrector = correctors[candidate_idx]
                optimizer = torch.optim.Adam(
                    corrector.parameters(),
                    lr=lr,
                    betas=(0.9, 0.98),
                    eps=1e-9,
                )

                loss = 0
                for iter in range(max_iter):
                    # randomly select a few pixels in the candidate image
                    # we use only pixels from [blur_kernel_size, h-blur_kernel_size] to avoid
                    # the boundary issue of gaussian blur
                    u = (
                        torch.randint(
                            low=blur_kernel_size,
                            high=wc - blur_kernel_size,
                            size=(num_pixels,),
                            device=device,
                        ).float()
                        + 0.5
                    )  # (qr, num_pixels,)
                    v = (
                        torch.randint(
                            low=blur_kernel_size,
                            high=hc - blur_kernel_size,
                            size=(num_pixels,),
                            device=device,
                        ).float()
                        + 0.5
                    )  # (num_pixels,)
                    uv_c = torch.stack([u, v], dim=-1).expand(b, num_pixels, 2)  # (b=1, num_pixels, 2)

                    # get the rgb (before correction) on the candidate image
                    assert candidate_rgb.size(1) == 1  # qc == 1
                    ori_rgb = utils.uv_sampling(
                        uv=uv_c,  # (b=1, num_pixel, 2)
                        feature_map=candidate_rgb.squeeze(1),  # (b=1, hc, wc, c)
                        uv_normalized=False,
                    ).unsqueeze(1)  # (b=1, qc=1, num_pixel, c)
                    est_rgb = corrector(ori_rgb)  # (b=1, qc=1, num_pixel, c)
                    candidate_valid_mask = (
                        utils.uv_sampling(
                            uv=uv_c,  # (b=1, num_pixel, 2)
                            feature_map=candidate_hit.float().squeeze(1).unsqueeze(-1),  # (b=1, hc, wc, 1)
                            uv_normalized=False,
                        )
                        .unsqueeze(1)
                        .squeeze(-1)
                    )  # (b=1, qc=1, num_pixel)
                    candidate_valid_mask = candidate_valid_mask > 0.5  # (b=1, qc=1, num_pixel)
                    ratio_cand = candidate_valid_mask.sum() / candidate_valid_mask.numel()

                    # find correponding uv in the reference images
                    uv_r, xyz_r = utils.find_corresponding_uv(
                        uv_c=uv_c,  # (b, num_pixels, 2)
                        z_map=candidate_depth.squeeze(1),  # (b, hc, wc)
                        intrinsics_from=candidate_intrinsics.squeeze(1),  # (b, 3, 3)
                        H_c2w_from=candidate_H_c2w.squeeze(1),  # (b, 4, 4)
                        intrinsics_to=ref_intrinsics,  # (b, qr, 3, 3)
                        H_c2w_to=ref_H_c2w,  # (b, qr, 4, 4)
                        dim_b=1,
                    )  # (b, qr, num_pixel, 2)  [0, wr]  [0, hr],   # (b, qr, num_pixel, 3)

                    # calculate the valid_mask
                    gt_valid_mask = torch.logical_and(
                        uv_r[..., 0] >= ref_blur_kernel_size,
                        uv_r[..., 0] < wr - ref_blur_kernel_size,
                    )  # (b, qr, num_pixel)
                    gt_valid_mask = torch.logical_and(
                        gt_valid_mask,
                        uv_r[..., 1] >= ref_blur_kernel_size,
                    )
                    gt_valid_mask = torch.logical_and(
                        gt_valid_mask,
                        uv_r[..., 1] < hr - ref_blur_kernel_size,
                    )  # (b, qr, num_pixel)

                    ratio_loc = gt_valid_mask.sum() / gt_valid_mask.numel()

                    # use ref_depth to filter out occluded points as well
                    z_r_gt = (
                        utils.uv_sampling(
                            uv=uv_r.squeeze(0),  # (qr, num_pixel, 2)
                            feature_map=ref_depth.squeeze(0).unsqueeze(-1),  # (qr, hr, wr, 1)
                            uv_normalized=False,
                        )
                        .squeeze(-1)
                        .unsqueeze(0)
                    )  # (b=1, qr, num_pixel)
                    valid_z = (
                        torch.clamp(1 / xyz_r[..., 2], min=0, max=1e3)  # (b, qr, num_pixel)
                        - torch.clamp(1 / z_r_gt, min=0, max=1e3)
                    ).abs() < th_diopter  # (b, qr, num_pixel)

                    before_valid = gt_valid_mask.sum()

                    gt_valid_mask = torch.logical_and(
                        gt_valid_mask,  # (b, qr, num_pixel)
                        valid_z,  # (b, qr, num_pixel)
                    )

                    after_valid = gt_valid_mask.sum()
                    ratio_valid_z = after_valid / before_valid

                    before_valid = gt_valid_mask.sum()
                    gt_valid_mask = torch.logical_and(
                        gt_valid_mask,  # (b, qr, num_pixel)
                        candidate_valid_mask,  # (b, qc=1, num_pixel)
                    )
                    after_valid = gt_valid_mask.sum()
                    ratio_candidate_valid = after_valid / before_valid

                    gt_total_valid = torch.clamp(gt_valid_mask.sum() * c, min=1)  # considered qr and c

                    # get color on the reference images
                    assert uv_r.size(0) == 1  # b == 1
                    gt_rgb = utils.uv_sampling(
                        uv=uv_r.squeeze(0),  # (qr, num_pixel, 2)
                        feature_map=ref_rgb.squeeze(0),  # (qr, hr, wr, c)
                        uv_normalized=False,
                    ).unsqueeze(0)  # (b=1, qr, num_pixel, c)

                    # calculate the rgb loss
                    loss = loss_fn(est_rgb.expand(b, qr, num_pixels, c), gt_rgb)  # (b, qr, num_pixel, c)

                    if loss_outlier_percentage is None or loss_outlier_percentage < 1.0e-4:
                        loss = (
                            loss.masked_fill(
                                torch.logical_not(gt_valid_mask.unsqueeze(-1)),
                                0,
                            ).sum()
                            / gt_total_valid
                        )
                        # loss = (loss * gt_valid_mask.unsqueeze(-1)).sum() / gt_total_valid
                        total_valid = gt_total_valid  # considered qr and c
                    else:
                        # disgard top `loss_outlier_percentage` outlier loss

                        # calculate the new outlier percentage over only the valid part
                        valid_ratio = gt_total_valid / loss.numel()
                        adjusted_loss_outlier_percentage = valid_ratio * loss_outlier_percentage

                        loss = loss * gt_valid_mask.unsqueeze(-1)  # (b, qr, num_pixel, c)
                        th = torch.quantile(loss, q=(1 - adjusted_loss_outlier_percentage))
                        loss_valid_mask = loss < th  # (b, qr, num_pixel, c)
                        loss = loss * loss_valid_mask
                        total_valid = torch.logical_and(loss_valid_mask, gt_valid_mask.unsqueeze(-1)).sum().clamp(min=1)
                        loss = loss.sum() / total_valid

                    if loss.detach().cpu() < th_loss:
                        if (print_every_iter > 0 and iter % print_every_iter == 0) or (iter >= min_iter):
                            print(
                                f"{len(ref_idxs) - nr} / {n}, {iter}: "
                                f"{loss.detach().cpu().numpy():.4e} < th = {th_loss:.4e}, break"
                            )
                        if iter >= min_iter:
                            break
                    else:
                        # update
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        if print_every_iter > 0 and iter % print_every_iter == 0:
                            print(
                                f"{len(ref_idxs) - nr} / {n}, cidx {candidate_idx}, retry {retry}, {iter}: "
                                f"{loss.detach().cpu().numpy():.4e} "
                                f"total_valid = {total_valid.detach().cpu().item()} ({total_valid / num_pixels / c / qr * 100:.2f}%)\n"
                                f"     ratio_candi = {ratio_cand * 100:.2f}% "
                                f"ratio_loc = {ratio_loc * 100:.2f}% "
                                f"ratio_z = {ratio_valid_z * 100:.2f}% "
                                f"ratio_candi2 = {ratio_candidate_valid * 100:.2f}%"
                            )

                if isinstance(loss, torch.Tensor):
                    loss = loss.detach().cpu().numpy().item()
                print(f"{len(ref_idxs) - nr} / {n}: {loss:.4e}")
                final_losses[candidate_idx] = loss

                # correct the entire image and add to the ref
                with torch.no_grad():
                    rgbds[candidate_idx].rgb = corrector(candidate_rgb).detach()  # (b, qc, hc, wc, c)

                # add candidate_idx to ref_idxs
                if loss <= max_loss_for_ref:
                    ref_idxs.append(candidate_idx)
                seen_idxs.append(candidate_idx)
            retry += 1

        # send to the original device
        rgbd_images = [rgbds[i].detach().to(device=ori_device) for i in range(nr, nr + n)]
        correctors = [correctors[i].to(device=ori_device) for i in range(nr, nr + n)]
        final_losses = [final_losses[i] for i in range(nr, nr + n)]

        reliable_idxs = [i - nr for i in ref_idxs if i >= nr]
        reliable = torch.zeros(n, dtype=torch.bool, device=ori_device)
        reliable[reliable_idxs] = 1
        reliable = reliable.detach().cpu().tolist()

        return dict(
            rgbd_images=rgbd_images,  # blurred
            correctors=correctors,
            reliable=reliable,  # (n,)  true if reliable
            final_losses=final_losses,  # (n,)
        )

    def get_pcd(
        self,
        subsample: int = 1,
        remove_background: bool = True,
        keep_img_idxs: bool = False,
        compute_ray_feature: bool = False,
    ) -> PointCloud:
        """
        Reproject RGBD pixels to world coordinate.

        Args:
            subsample:
                use 1 out of every `subsample` pixels
            remove_background:
                whether to remove background pixels (depth > 1e6).
                only has effect if b == 1.
            keep_img_idxs:
                whether to record the original image index in the point cloud
            compute_ray_feature:
                whether to compute feature derived (eg, ray direction, etc)
        """
        assert self.depth is not None
        # only has effect if b == 1
        valid_mask = self.hit_map  # None or (b, q, h, w)
        if remove_background:
            if valid_mask is None:
                valid_mask = self.depth < 1.0e6  # (b, q, h, w)
            else:
                valid_mask = torch.logical_and(valid_mask, self.depth < 1.0e6)  # (b, q, h, w)
        depth = self.depth.masked_fill(
            mask=torch.logical_not(valid_mask),
            value=INF,
        )

        if keep_img_idxs:
            b, q, h, w = self.depth.shape
            qhw = q * h * w
            img_idxs = torch.arange(qhw, device=self.depth.device).expand(b, -1)  # (b, qhw)
            img_idxs = img_idxs.reshape(b, q, h, w, 1)  # (b, q, h, w, 1)  it saves the linear index of qhw
        else:
            img_idxs = None

        other_maps_base = [
            self.rgb,
            self.normal_w,
            valid_mask.unsqueeze(-1),
            self.feature,
            img_idxs,
        ]

        # make sure we can retrieve corresponding name
        if self.other_maps is not None:
            names_for_points_other_attrs = list(self.other_maps.keys())
        else:
            names_for_points_other_attrs = []

        other_maps = other_maps_base + [self.other_maps[_] for _ in names_for_points_other_attrs]
        xyz_dict = utils.compute_3d_xyz(
            z_map=depth,
            intrinsic=self.camera.intrinsic,
            H_c2w=self.camera.H_c2w,
            subsample=subsample,
            other_maps=other_maps,
        )  # (b, q, h', w', 3)  can contain nan

        points_w = xyz_dict["xyz_w"]  # (b, q, h', w', 3)

        if self.camera.timestamp is not None:
            b, q, _h, _w, _3 = points_w.shape
            point_timestamp = self.camera.timestamp.reshape(b, q, 1, 1, 1).expand(b, q, _h, _w, 1)
            point_timestamp = point_timestamp.reshape(b, -1, 1)  # (b, n, 1)
        else:
            point_timestamp = None

        b, *qhw, c = points_w.shape
        points_w = points_w.reshape(b, -1, 3)  # (b, n, 3)

        if xyz_dict["other_maps"] is not None and len(xyz_dict["other_maps"]) > 0:
            points_rgb = xyz_dict["other_maps"][0]  # (b, q, h', w', 3)
            if points_rgb is not None:
                points_rgb = points_rgb.reshape(b, -1, 3)  # (b, n, 3)
            points_normal_w = xyz_dict["other_maps"][1]
            if points_normal_w is not None:
                points_normal_w = points_normal_w.reshape(b, -1, 3)  # (b, n, 3)
            valid_mask = xyz_dict["other_maps"][2].reshape(b, -1, 1)  # (b, n, 1)

            points_feature = xyz_dict["other_maps"][3]
            if points_feature is not None:
                dim_feature = points_feature.size(-1)
                points_feature = points_feature.reshape(b, -1, dim_feature)  # (b, n, f)

            points_img_idxs = xyz_dict["other_maps"][4]
            if points_img_idxs is not None:
                points_img_idxs = points_img_idxs.reshape(b, -1, 1)  # (b, n, 1)
        else:
            points_rgb = None
            points_normal_w = None
            points_feature = None
            points_img_idxs = None
            valid_mask = None

        n_pts = points_w.shape[1]
        points_other_attrs = {
            names_for_points_other_attrs[tmp_i]: xyz_dict["other_maps"][len(other_maps_base) + tmp_i].reshape(
                b, n_pts, -1
            )
            for tmp_i in range(len(names_for_points_other_attrs))
        }

        # compute ray-based features
        if compute_ray_feature:
            feature_dict = utils.compute_3d_zdir_and_dps(
                z_map=self.depth,
                intrinsic=self.camera.intrinsic,
                H_c2w=self.camera.H_c2w,
                subsample=subsample,
            )
            for key in feature_dict:
                feature_dim = feature_dict[key].size(-1)
                feature_dict[key] = feature_dict[key].reshape(b, -1, feature_dim)

            # compute view direction
            camera_pinhole_w = self.camera.H_c2w[..., :3, 3]  # (b, q, 3)
            *bq_shape, _ = camera_pinhole_w.shape
            camera_pinhole_w = (
                camera_pinhole_w.view(*bq_shape, 1, 1, 3).expand(b, *qhw, 3).reshape(b, -1, 3)
            )  # (b, n, 3)
            # if valid_mask is not None:
            #     camera_pinhole_w = camera_pinhole_w[:, valid_mask]  # (b, n, 3)
            vdir_w = points_w - camera_pinhole_w  # (b, n, 3)
            vdir_w = torch.nn.functional.normalize(
                input=vdir_w,
                p=2,
                dim=-1,
            )  # (b, n, 3)
        else:
            feature_dict = dict()
            vdir_w = None
            camera_pinhole_w = None

        point_cloud = PointCloud(
            xyz_w=points_w,  # (b, n, 3)
            rgb=points_rgb,  # (b, n, 3)
            normal_w=points_normal_w,  # (b, n, 3)
            captured_z_direction_w=feature_dict.get("zdir_w", None),  # (b, n, 3)
            captured_dps=feature_dict.get("dps_w", None),  # (b, n, 1)
            captured_dps_u_w=feature_dict.get("dps_uw", None),  # (b, n, 3)
            captured_dps_v_w=feature_dict.get("dps_vw", None),  # (b, n, 3)
            captured_pinhole_w=camera_pinhole_w,  # (b, n, 3)
            captured_view_direction_w=vdir_w,  # (b, n, 3)
            valid_mask=valid_mask,  # (b, n, 1)
            included_point_at_inf=False,
            feature=points_feature,  # (b, n, f)
            img_idxs=points_img_idxs,  # (b, n, 1)
            timestamp=point_timestamp,  # (b, n, 1) or None
            **points_other_attrs,
        )

        if b == 1 and remove_background:
            point_cloud = point_cloud.extract_valid_point_cloud(bidx=0)

        return point_cloud

    def sample_random_patches(
        self,
        num_patches_per_q: int,
        patch_width_px: int,
        patch_width_pitch_scale: T.Union[float, torch.Tensor] = 1.0,  # (*b,)
        patch_height_px: int = None,  # (*b,)
        patch_height_pitch_scale: T.Union[float, torch.Tensor] = None,  # (*b,)
        prob_density: torch.Tensor = None,  # (b, q, h, w)
        prob_density_bias: float = 5.0,
        int_only: bool = True,
        inbound_only: bool = True,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        required_attributes: T.List[str] = None,
    ) -> T.Dict[str, T.Any]:
        """
        Generate rays to form patches on the corresponding images.

        Args:
            num_patches_per_q:
                number of patches from each q
            patch_width_px:
                number of pixels in the patch in width
            patch_width_pitch_scale:
                (*b,) the pitch of the patch (new_pitch / old_pitch)
            patch_height_px:
                if None, the same as `patch_width_px`
            patch_height_pitch_scale:
                if None, the same as `patch_width_pitch_scale`
            prob_density:
                (b, q, h, w) to scale the probability to sample each pixel
            prob_density_bias:
                if higher, more likely to sample patches with higher prob_density
            int_only:
                whether the center is always at an integer index
            mode:
                mode used by grid_sample
            padding_mode:
                padding mode used by grid_sample. "zeros", "border", "reflection"
            required_attributes:
                list of str containing the field wanted. If None: all possible fields

        Returns:
            ray:
                (b, q, num_patches_per_q, hp, wp)
            uv:
                (b, q, num_patches_per_q, hp, wp, 2)  [0, w], [0, h]
            rgb:
                (b, q, num_patches_per_q, hp, wp, 3)
            depth:
                (b, q, num_patches_per_q, hp, wp) or None, same coord as the original depth
            normal_w:
                (b, q, num_patches_per_q, hp, wp, 3) or None, same coord as the original surface normal
            hit_map:
                (b, q, num_patches_per_q, hp, wp) or None,  float, [0, 1],  0: not valid, 1: valid
            feature:
                (b, q, num_patches_per_q, hp, wp, f) or None
            ray_t:
                (b, q, num_patches_per_q, hp, wp) or None, ray traveling distance
            other_maps:
                dict of (b, q, num_patches_per_q, hp, wp, d) or None

        Note:
             part of the patches may go out of bound.  The padding_mode determines the behavior
        """
        b, q, h, w, _3 = self.rgb.shape

        # sample random patch and get uv and rays
        ray_uv_dict = self.camera.generate_random_patch_rays(
            num_patches_per_q=num_patches_per_q,
            patch_width_px=patch_width_px,
            patch_width_pitch_scale=patch_width_pitch_scale,
            patch_height_px=patch_height_px,
            patch_height_pitch_scale=patch_height_pitch_scale,
            prob_density=prob_density,
            prob_density_bias=prob_density_bias,
            int_only=int_only,
            inbound_only=inbound_only,
        )  # uv:  (b, q, num_patches_per_q, hp, wp, 2) [0,w] [0,h]
        uv = ray_uv_dict["uv"]

        # [0, w] -> [0, 1]
        u = uv[..., 0] / w
        v = uv[..., 1] / h
        uv = torch.stack([u, v], dim=-1)  # (b, q, num_patches_per_q, hp, wp, 2)  [0, 1]
        _b, _q, *p_shape, _2 = uv.shape

        # interpolate individual attributes
        for key in ["rgb", "depth", "normal_w", "hit_map", "feature"]:
            if required_attributes is not None and key not in required_attributes:
                continue

            arr = getattr(self, key, None)
            if arr is None:
                ray_uv_dict[key] = None
                continue

            if key in ["depth", "hit_map"]:
                squeeze = True
                arr = arr.unsqueeze(-1)
            else:
                squeeze = False

            dim = arr.size(-1)
            out = utils.uv_sampling(
                uv=uv.flatten(0, 1),  # (bq, num_patches_per_q, hp, wp, 2)  [0, 1]
                feature_map=arr.flatten(0, 1).float(),  # (bq, h, w, dim)
                mode=mode,
                padding_mode=padding_mode,
                uv_normalized=True,  # uv in [0, 1]
            )  # (bq, num_patches_per_q, hp, wp, dim)
            out = out.reshape(b, q, *p_shape, dim)

            if squeeze:
                out = out.squeeze(-1)

            ray_uv_dict[key] = out  # (b, q, num_patches_per_q, hp, wp, dim)

        if required_attributes is None or "other_maps" in required_attributes and self.other_maps is not None:
            out_other_maps = dict()
            for key in self.other_maps:
                arr = self.other_maps[key]
                if arr is None:
                    out_other_maps[key] = None
                    continue

                dim = arr.size(-1)
                out = utils.uv_sampling(
                    uv=uv.flatten(0, 1),  # (bq, num_patches_per_q, hp, wp, 2)  [0, 1]
                    feature_map=arr.flatten(0, 1).float(),  # (bq, h, w, dim)
                    mode=mode,
                    padding_mode=padding_mode,
                    uv_normalized=True,  # uv in [0, 1]
                )  # (bq, num_patches_per_q, hp, wp, dim)
                out = out.reshape(b, q, *p_shape, dim)
                out_other_maps[key] = out
            ray_uv_dict["other_maps"] = out_other_maps

        return ray_uv_dict

    def get_patch(
        self,
        patch_idxs: torch.Tensor,
        patch_width_px: int,
        patch_width_pitch_scale: T.Union[float, torch.Tensor] = 1.0,
        patch_height_px: int = None,  # (*b,)
        patch_height_pitch_scale: T.Union[float, torch.Tensor] = None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        required_attributes: T.List[str] = None,
    ) -> T.Dict[str, T.Any]:
        """
        Generate rays to form patches on the corresponding images.

        Args:
            patch_idxs:
                (b, p, 3qhw) top left pixel center [0, q] [0, h (*.5)] [0, w (*.5)]
                If want pixel center, remember to put hw to *.5
            patch_width_px:
                number of pixels in the patch in width
            patch_width_pitch_scale:
                (*b,) the pitch of the patch (new_pitch / old_pitch)
            patch_height_px:
                if None, the same as `patch_width_px`
            patch_height_pitch_scale:
                if None, the same as `patch_width_pitch_scale`
            mode:
                mode used by grid_sample
            padding_mode:
                padding mode used by grid_sample. "zeros", "border", "reflection"
            required_attributes:
                list of str containing the field wanted. If None: all possible fields

        Returns:
            ray:
                (b, p, hp, wp)
            uv:
                (b, p, hp, wp, 2)  [0, w], [0, h]
            rgb:
                (b, p, hp, wp, 3)
            depth:
                (b, p, hp, wp) or None, same coord as the original depth
            normal_w:
                (b, p, hp, wp, 3) or None, same coord as the original surface normal
            hit_map:
                (b, p, hp, wp) or None,  float, [0, 1],  0: not valid, 1: valid
            feature:
                (b, p, hp, wp, f) or None
            other_maps:
                dict of (b, p, hp, wp, d) or None

        Note:
             part of the patches may go out of bound.  The padding_mode determines the behavior
        """
        b, q, h, w, _3 = self.rgb.shape
        _b, p, _3qhw = patch_idxs.shape
        assert b == _b
        device = self.rgb.device
        patch_idxs = patch_idxs.to(device=device)

        if patch_height_px is None:
            patch_height_px = patch_width_px

        # create uv template  (top left = (0, 0))
        uv = utils.generate_patch_uv(
            patch_center=torch.zeros(b, 2, device=device),  # (b, 2)
            patch_width_px=patch_width_px,
            patch_width_pitch_scale=patch_width_pitch_scale,
            patch_height_px=patch_height_px,
            patch_height_pitch_scale=patch_height_pitch_scale,
            device=device,
        )  # (b, patch_height_px, patch_width_px, 2)
        uv = uv - uv[:, 0:1, 0:1, :]  # (b, patch_height_px, patch_width_px, 2)

        # convert patch_idxs (b, p, 3qhw) to uv: (b, p, hp, wp, 2) and qidxs: (b, p)
        qidxs = patch_idxs[..., 0].long()  # (b, p)  long
        v_start = patch_idxs[..., 1]  # (b, p)
        u_start = patch_idxs[..., 2]  # (b, p)
        uv = uv.reshape(b, 1, patch_height_px, patch_width_px, 2)
        uv = torch.stack(
            [
                uv[..., 0] + u_start.reshape(b, p, 1, 1),  # (b, p, hp, wp)
                uv[..., 1] + v_start.reshape(b, p, 1, 1),  # (b, p, hp, wp)
            ],
            dim=-1,
        )  # (b, p, hp, wp, 2)  [0, w] [0, h]

        # use the uv to create rays
        H_c2w = torch.gather(
            input=self.camera.H_c2w,  # (b, q, 4, 4)
            dim=1,
            index=qidxs.reshape(b, p, 1, 1).expand(b, p, 4, 4),  # (b, p, 4, 4)
        )  # (b, p, 4, 4)
        intrinsic = torch.gather(
            input=self.camera.intrinsic,  # (b, q, 3, 3)
            dim=1,
            index=qidxs.reshape(b, p, 1, 1).expand(b, p, 3, 3),  # (b, p, 3, 3)
        )  # (b, p, 3, 3)

        bp = b * p
        ray_origins_w, ray_directions_w = utils.generate_camera_rays_from_uv(
            cam_poses=H_c2w.reshape(bp, 4, 4),  # (bp, 4, 4)
            intrinsics=intrinsic.reshape(bp, 3, 3),  # (bp, 3, 3)
            uv=uv.reshape(bp, patch_height_px, patch_width_px, 2),  # (bp, hp, wp, 2)
            device=device,
        )  # (bp, hp, wp, 3)
        out_dict = dict()
        ray = Ray(
            origins_w=ray_origins_w.reshape(b, p, patch_height_px, patch_width_px, 3),
            directions_w=ray_directions_w.reshape(b, p, patch_height_px, patch_width_px, 3),
        )  # (b, p, hp, wp, 3)
        out_dict["ray"] = ray
        out_dict["uv"] = uv.clone()  # (b, p, hp, wp, 2)  [0, w] [0, h]

        # [0, w] -> [0, 1]
        u = uv[..., 0] / w
        v = uv[..., 1] / h
        uv = torch.stack([u, v], dim=-1)  # (b, p, hp, wp, 2)  [0, 1]

        # interpolate individual attributes
        for key in ["rgb", "depth", "normal_w", "hit_map", "feature"]:
            if required_attributes is not None and key not in required_attributes:
                continue

            arr = getattr(self, key, None)
            if arr is None:
                out_dict[key] = None
                continue

            if key in ["depth", "hit_map"]:
                squeeze = True
                arr = arr.unsqueeze(-1)
            else:
                squeeze = False

            dim = arr.size(-1)
            out = utils.sparse_uv_sampling(
                uv=uv,  # (b, p, hp, wp, 2)
                qidx=qidxs,  # (b, p)
                feature_map=arr,  # (b, q, h, w, d)
                mode=mode,
                padding_mode=padding_mode,
                uv_normalized=True,
            )  # (b, p, hp, wp, d)
            out = out.reshape(b, p, patch_height_px, patch_width_px, dim)

            if squeeze:
                out = out.squeeze(-1)

            out_dict[key] = out  # (b, p, hp, wp, d)

        if required_attributes is None or "other_maps" in required_attributes and self.other_maps is not None:
            out_other_maps = dict()
            for key in self.other_maps:
                arr = self.other_maps[key]
                if arr is None:
                    out_other_maps[key] = None
                    continue

                dim = arr.size(-1)
                out = utils.sparse_uv_sampling(
                    uv=uv,  # (b, p, hp, wp, 2)
                    qidx=qidxs,  # (b, p)
                    feature_map=arr,  # (b, q, h, w, d)
                    mode=mode,
                    padding_mode=padding_mode,
                    uv_normalized=True,
                )  # (b, p, hp, wp, d)
                out = out.reshape(b, p, patch_height_px, patch_width_px, dim)
                out_other_maps[key] = out
            out_dict["other_maps"] = out_other_maps

        return out_dict

    def get_gaussian_pyramid(
        self,
        total_level: int,
        base_sigma: float = 0.865,
        recompute: bool = False,
    ) -> T.List["RGBDImage"]:
        """
        Construct a gaussian pyramid by progressively blurring all content
        with a gaussian kernel.

        Note that unlike typical pyramid which downsamples the image,
        we retain the image resolution and only blur the images
        by sigma, 2 sigma, 4 sigma, 8 sigma, etc.
        All the images share the same camera.

        Args:
            total_level:
                total level of the pyramid, where the 0th level is the
                unblurred content. 1st blur is blurred one time, and so on.
            base_sigma:
                the standard deviation of the gaussian kernel used at the
                first blurring operation.

        Returns:
            a list of RGBDImage:  level (index) -> RGBDImage.  Note that due to the blur,
            hit_map will become floating point.
        """

        # if self.rgb is not None:
        #     device = self.rgb.device
        #     dtype = self.rgb.dtype
        #     b, q, h, w, _3 = self.rgb.shape
        # elif self.depth is not None:
        #     device = self.depth.device
        #     dtype = self.depth.dtype
        #     b, q, h, w = self.depth.shape
        # else:
        #     raise NotImplementedError

        # blur_kernel = torch.tensor(
        #     [
        #         [
        #             [1.0,  4.0,  6.0,  4.0, 1.0],
        #             [4.0, 16.0, 24.0, 16.0, 4.0],
        #             [6.0, 24.0, 36.0, 24.0, 6.0],
        #             [4.0, 16.0, 24.0, 16.0, 4.0],
        #             [1.0,  4.0,  6.0,  4.0, 1.0],
        #         ]
        #     ],
        #     dtype=dtype,
        #     device=device,
        # ) / 256.0  # (1, 5, 5) sum to 1

        # construct pyramid
        assert total_level >= 0

        if not recompute and self.pyramid is not None and len(self.pyramid) == total_level:
            return self.pyramid

        pyramid: T.List[RGBDImage] = [self]

        for i in range(1, total_level):
            ref_rgbd = pyramid[i - 1]
            sigma = (2 ** (i - 1)) * base_sigma
            kernel_size = round((sigma - 0.8) / 0.3 + 1) * 4 + 1
            out_dict = dict()
            for attr_name in ["rgb", "depth", "normal_w", "hit_map", "feature"]:
                arr = getattr(ref_rgbd, attr_name, None)
                if arr is None:
                    out_dict[attr_name] = None
                    continue

                if attr_name in ["depth", "hit_map"]:
                    arr = arr.unsqueeze(-1)  # (b, q, h, w, c)
                arr = arr.permute(0, 1, 4, 2, 3)  # (b, q, c, h, w)

                b, q, c, h, w = arr.shape

                arr = torchvision.transforms.functional.gaussian_blur(
                    arr.reshape(b * q * c, h, w).float(),  # (bqc, h, w)
                    kernel_size=kernel_size,
                    sigma=sigma,
                ).reshape(b, q, c, h, w)
                arr = arr.permute(0, 1, 3, 4, 2)  # (b, q, h, w, c)
                if attr_name in ["depth", "hit_map"]:
                    arr = arr.squeeze(-1)  # (b, q, h, w)

                if attr_name == "normal_w":
                    # normalize
                    arr = torch.nn.functional.normalize(arr, p=2, dim=-1)

                out_dict[attr_name] = arr

            pyramid.append(RGBDImage(camera=self.camera, **out_dict))

        self.pyramid = pyramid
        return self.pyramid

    def state_dict(self) -> T.Dict[str, T.Any]:
        """Returns a dictionary that can be saved or load."""
        to_save = dict()
        for name in ["rgb", "depth", "normal_w", "hit_map", "feature", "other_maps"]:
            to_save[name] = getattr(self, name, None)
        to_save["camera"] = self.camera.state_dict()
        return to_save

    def load_state_dict(
        self,
        state_dict: T.Dict[str, T.Any],
    ):
        """Load the state dictionary."""
        for name in ["rgb", "depth", "normal_w", "hit_map", "feature", "other_maps"]:
            setattr(self, name, state_dict.get(name, None))
        self.camera.load_state_dict(state_dict.get("camera", None))

    def save(
        self,
        output_dir: str,
        overwrite: bool = False,
        save_png: bool = True,
        save_pt: bool = True,
        save_gif: bool = True,
        save_video: bool = True,
        gif_fps: float = 10,
        background_color: T.Union[float, T.List[float]] = 1.0,
        global_min_depth: float = None,
        global_max_depth: float = None,
        hit_only: bool = True,
    ):
        if isinstance(background_color, (int, float)):
            background_color = [background_color] * 3

        if os.path.exists(output_dir) and not overwrite:
            raise RuntimeError
        os.makedirs(output_dir, exist_ok=True)

        if save_pt:
            filename = os.path.join(output_dir, "state_dict.pt")
            torch.save(self.state_dict(), filename)

        # deal with hit_map
        b, q, h, w = self.shape
        if hit_only and self.hit_map is not None:
            hit_map = (self.hit_map > 0.5).float()  # (b, q, h, w)
        else:
            hit_map = torch.ones(b, q, h, w, dtype=self.rgb.dtype, device=self.rgb.device)  # (b, q, h, w)

        background_img = torch.ones_like(self.rgb)  # (b, q, h, w, 3)
        for c in range(3):
            background_img[..., c] = background_color[c]

        # deal with depth
        if self.depth is not None:
            masked_depth = self.depth * hit_map
        else:
            masked_depth = None

        # masked_rgb = self.rgb * hit_map.unsqueeze(-1)
        # masked_rgb = masked_rgb + (1 - hit_map).unsqueeze(-1) * background_img

        # if self.other_maps is None or self.other_maps.get('alpha', None) is None:
        #     rgb_mask = hit_map  # (b, q, h, w)
        # else:
        #     rgb_mask = self.other_maps.get('alpha').squeeze(-1)  # (b, q, h, w)
        # masked_rgb = self.rgb * rgb_mask.unsqueeze(-1)
        # masked_rgb = masked_rgb + (1 - rgb_mask).unsqueeze(-1) * background_img

        # we save rgb as rgba to avoid double matting
        if self.other_maps is not None and self.other_maps.get("alpha", None) is not None:
            rgb_mask = self.other_maps["alpha"]  # (b, q, h, w, 1)
            if len(rgb_mask.shape) == 4:
                rgb_mask = rgb_mask.unsqueeze(-1)  # ensure the mask has channel dimension
        elif self.hit_map is not None:
            rgb_mask = self.hit_map.float().unsqueeze(-1)  # (b, q, h, w, 1)
        else:
            rgb_mask = torch.ones_like(self.rgb[..., :1])  # (b, q, h, w, 1)

        masked_rgb = torch.cat(
            [
                self.rgb.float(),  # (b, q, h, w, 3)
                rgb_mask.float(),  # (b, q, h, w, 1)
            ],
            dim=-1,
        )  # (b, q, h, w, 4rgba)

        if self.normal_w is not None:
            masked_normal_w = self.normal_w * hit_map.unsqueeze(-1)
            masked_normal_w = masked_normal_w + (1 - hit_map).unsqueeze(-1) * background_img
        else:
            masked_normal_w = None

        if save_png:
            # normal: [-1, 1] -> [0, 1]
            if self.normal_w is not None:
                normal_w = (masked_normal_w + 1) / 2.0
            else:
                normal_w = None

            for ib in range(b):
                sub_dir = os.path.join(output_dir, f"batch_{ib}")
                os.makedirs(sub_dir, exist_ok=True)
                if masked_depth is not None:
                    if global_min_depth is None:
                        min_depth = masked_depth[ib].min()
                    else:
                        min_depth = global_min_depth
                else:
                    min_depth = None
                if masked_depth is not None:
                    if global_max_depth is None:
                        max_depth = masked_depth[ib].max()
                    else:
                        max_depth = global_max_depth
                else:
                    max_depth = None

                for iq in range(q):
                    # rgb
                    filename = os.path.join(sub_dir, f"rgb_{iq}.png")
                    imageio.imwrite(
                        filename,
                        (masked_rgb[ib, iq] * 255.0)
                        .detach()
                        .cpu()
                        .float()
                        .clamp(min=0, max=255)
                        .numpy()
                        .astype(np.uint8),
                    )

                    # normalized depth
                    if masked_depth is not None:
                        filename = os.path.join(sub_dir, f"depth_{iq}.png")
                        dd = (masked_depth[ib, iq] - min_depth) / (max_depth - min_depth)
                        imageio.imwrite(
                            filename,
                            (
                                (dd * hit_map[ib, iq] + (1 - hit_map[ib, iq].float()) * background_img[ib, iq, ..., 0])
                                * 255.0
                            )
                            .unsqueeze(-1)
                            .expand(-1, -1, 3)
                            .detach()
                            .cpu()
                            .float()
                            .numpy()
                            .astype(np.uint8),
                        )

                    # normal
                    if normal_w is not None:
                        filename = os.path.join(sub_dir, f"normal_w_{iq}.png")
                        imageio.imwrite(
                            filename,
                            (
                                (
                                    normal_w[ib, iq] * hit_map[ib, iq].unsqueeze(-1)
                                    + (1 - hit_map[ib, iq].unsqueeze(-1).float()) * background_img[ib, iq]
                                )
                                * 255.0
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.uint8),
                        )

                    # hit map
                    if self.hit_map is not None:
                        filename = os.path.join(sub_dir, f"hit_map_{iq}.png")
                        imageio.imwrite(
                            filename, (self.hit_map[ib, iq] * 255.0).detach().cpu().numpy().astype(np.uint8)
                        )

        if save_gif:
            # normal: [-1, 1] -> [0, 1]
            if self.normal_w is not None:
                normal_w = (masked_normal_w + 1) / 2.0
            else:
                normal_w = None

            for ib in range(self.rgb.size(0)):
                # rgb
                sub_dir = os.path.join(output_dir, "rgb")
                os.makedirs(sub_dir, exist_ok=True)
                filename = os.path.join(sub_dir, f"batch_{ib}.gif")

                _masked_rgb = masked_rgb[..., :3] * masked_rgb[..., 3:4] + (1 - masked_rgb[..., 3:4]) * background_img
                render.create_gif(
                    images=_masked_rgb[ib].float().clamp(min=0, max=1.0),
                    filename=filename,
                    fps=gif_fps,
                )

                # depth
                if masked_depth is not None:
                    if global_min_depth is None:
                        min_depth = masked_depth[ib].min()
                    else:
                        min_depth = global_min_depth

                    if global_max_depth is None:
                        max_depth = masked_depth[ib].max()
                    else:
                        max_depth = global_max_depth

                    dd = (masked_depth[ib] - min_depth) / (max_depth - min_depth)
                    sub_dir = os.path.join(output_dir, "depth")
                    os.makedirs(sub_dir, exist_ok=True)
                    filename = os.path.join(sub_dir, f"batch_{ib}.gif")
                    render.create_gif(
                        images=(dd * hit_map[ib] + (1 - hit_map[ib].float()) * background_img[ib, ..., 0])
                        .unsqueeze(-1)
                        .expand(-1, -1, -1, 3),
                        filename=filename,
                        fps=gif_fps,
                    )

                # normal_w
                if normal_w is not None:
                    sub_dir = os.path.join(output_dir, "normal_w")
                    os.makedirs(sub_dir, exist_ok=True)
                    filename = os.path.join(sub_dir, f"batch_{ib}.gif")
                    render.create_gif(
                        images=normal_w[ib] * hit_map[ib].unsqueeze(-1)
                        + (1 - hit_map[ib].float().unsqueeze(-1)) * background_img[ib],
                        filename=filename,
                        fps=gif_fps,
                    )

                # hit map
                if self.hit_map is not None:
                    sub_dir = os.path.join(output_dir, "hit_map")
                    os.makedirs(sub_dir, exist_ok=True)
                    filename = os.path.join(sub_dir, f"batch_{ib}.gif")
                    render.create_gif(
                        images=self.hit_map[ib].unsqueeze(-1).expand(-1, -1, -1, 3).float(),
                        filename=filename,
                        fps=gif_fps,
                    )

        if save_video:
            # normal: [-1, 1] -> [0, 1]
            if self.normal_w is not None:
                normal_w = (masked_normal_w + 1) / 2.0
            else:
                normal_w = None

            for ib in range(self.rgb.size(0)):
                # rgb
                sub_dir = os.path.join(output_dir, "rgb")
                os.makedirs(sub_dir, exist_ok=True)
                filename = os.path.join(sub_dir, f"batch_{ib}.mp4")
                _masked_rgb = masked_rgb[..., :3] * masked_rgb[..., 3:4] + (1 - masked_rgb[..., 3:4]) * background_img

                render.create_video(
                    images=_masked_rgb[ib].float().clamp(min=0, max=1.0),
                    filename=filename,
                    fps=gif_fps,
                    color_format="rgb",
                )

                # depth
                if masked_depth is not None:
                    if global_min_depth is None:
                        min_depth = masked_depth[ib].min()
                    else:
                        min_depth = global_min_depth

                    if global_max_depth is None:
                        max_depth = masked_depth[ib].max()
                    else:
                        max_depth = global_max_depth

                    dd = (masked_depth[ib] - min_depth) / (max_depth - min_depth)
                    sub_dir = os.path.join(output_dir, "depth")
                    os.makedirs(sub_dir, exist_ok=True)
                    filename = os.path.join(sub_dir, f"batch_{ib}.mp4")
                    render.create_video(
                        images=(dd * hit_map[ib] + (1 - hit_map[ib].float()) * background_img[ib, ..., 0])
                        .unsqueeze(-1)
                        .expand(-1, -1, -1, 3),
                        filename=filename,
                        fps=gif_fps,
                        color_format="rgb",
                        val_range="01",
                    )

                # normal_w
                if normal_w is not None:
                    sub_dir = os.path.join(output_dir, "normal_w")
                    os.makedirs(sub_dir, exist_ok=True)
                    filename = os.path.join(sub_dir, f"batch_{ib}.mp4")
                    render.create_video(
                        images=normal_w[ib] * hit_map[ib].unsqueeze(-1)
                        + (1 - hit_map[ib].float().unsqueeze(-1)) * background_img[ib],
                        filename=filename,
                        fps=gif_fps,
                        color_format="rgb",
                        val_range="01",
                    )

                # hit map
                if self.hit_map is not None:
                    sub_dir = os.path.join(output_dir, "hit_map")
                    os.makedirs(sub_dir, exist_ok=True)
                    filename = os.path.join(sub_dir, f"batch_{ib}.video")
                    render.create_video(
                        images=self.hit_map[ib].unsqueeze(-1).expand(-1, -1, -1, 3).float(),
                        filename=filename,
                        fps=gif_fps,
                        color_format="rgb",
                        val_range="01",
                    )

    def save_as(
        self,
        out_dir: str,
        overwrite: bool,
        mode: str,  # 'npy', 'exr', 'png', 'qoi'
        background_color: float = 1.0,
        save_attr_names: T.List[str] = None,
        flag_save_space: bool = False,
        ib_filename_offset: int = None,
        concatenate_along_b: bool = False,
    ):
        """
        Save individual files to allow subsample b and q.
        rgb as png, depth, normal, hitmap as npy.

        saved structure:

        out_dir:
        -- index.json
        -- cameras.npz
        -- {ib}
          -- rgb_xxxx.png (rgb)
          -- depth_xxxx.exr  (bw)  or depth_xxxx.npy
          -- normal_w_xxxx.exr (rgb)  or normal_w_xxxx.npy
          -- hit_map_xxxx.png
          -- raw_rgb_xxxx.exr (rgb)  if in other map
          -- obj_id_xxxx.exr (rgb)  if in other map
        """
        import cv2

        if os.path.exists(out_dir) and not overwrite:
            raise RuntimeError
        os.makedirs(out_dir, exist_ok=True)

        if save_attr_names is None:
            save_attr_names = ["rgb", "normal_w", "depth", "hit_map"]
            if self.other_maps is not None:
                save_attr_names += list(self.other_maps.keys())

        b, q, h, w, _3 = self.rgb.shape

        index_dict = dict(
            b=b,
            q=q,
            h=h,
            w=w,
        )

        alpha_saved = False
        if self.other_maps is not None and self.other_maps.get("alpha", None) is not None:
            rgb_mask = self.other_maps["alpha"]  # (b, q, h, w, 1)
            alpha_saved = True
        else:
            rgb_mask = self.hit_map.float().unsqueeze(-1)  # (b, q, h, w, 1)

        # masked_rgb = self.rgb * rgb_mask + (1 - rgb_mask) * background_color  # (b, q, h, w, 3)

        # we save rgb as rgba to avoid double matting
        masked_rgb = torch.cat(
            [
                self.rgb.float(),  # (b, q, h, w, 3)
                rgb_mask.float(),  # (b, q, h, w, 1)
            ],
            dim=-1,
        )  # (b, q, h, w, 4rgba)

        camera_filename = os.path.join(out_dir, f"cameras.npz")

        if concatenate_along_b and os.path.exists(camera_filename):
            previous_camera = Camera(H_c2w=None, intrinsic=None, timestamp=None, width_px=None, height_px=None)
            with np.load(camera_filename, allow_pickle=True) as _camera_state_dict:
                camera_state_dict = dict(_camera_state_dict)
            previous_camera.load_state_dict_numpy(camera_state_dict)

            # overwrite `ib_filename_offset`
            if ib_filename_offset is not None:
                assert ib_filename_offset == previous_camera.H_c2w.shape[0]
            else:
                ib_filename_offset = previous_camera.H_c2w.shape[0]

            camera = Camera.cat([previous_camera, self.camera], dim=0)
        else:
            camera = self.camera

        camera_state_dict = camera.state_dict()
        camera_state_dict = utils.to_numpy(camera_state_dict)
        np.savez(camera_filename, **camera_state_dict)  # (b, q)
        index_dict["camera"] = os.path.relpath(camera_filename, start=out_dir)

        if ib_filename_offset is None:
            ib_filename_offset = 0

        sub_index_dicts = []
        for ib in range(b):
            sub_dir = os.path.join(out_dir, f"{ib + ib_filename_offset:06d}")
            os.makedirs(sub_dir, exist_ok=True)

            rgb_filenames = []
            normal_filenames = []
            depth_filenames = []
            hit_map_filenames = []
            other_map_filenames = dict()
            for iq in range(q):
                # save rgb
                if "rgb" in save_attr_names and self.rgb is not None:
                    if mode == "qoi":
                        import qoi

                        filename = os.path.join(sub_dir, f"rgb_{iq:06d}.qoi")
                        qoi.write(
                            filename,
                            (masked_rgb[ib, iq] * 255.0)
                            .detach()
                            .cpu()
                            .float()
                            .clamp(min=0, max=255)
                            .numpy()
                            .astype(np.uint8),  # (h, w, 4rgba)
                        )

                    elif mode in ["npy", "exr", "png"]:
                        filename = os.path.join(sub_dir, f"rgb_{iq:06d}.png")
                        imageio.imwrite(
                            filename,
                            (masked_rgb[ib, iq] * 255.0)
                            .detach()
                            .cpu()
                            .float()
                            .clamp(min=0, max=255)
                            .numpy()
                            .astype(np.uint8),
                        )
                    else:
                        raise NotImplementedError
                    rgb_filenames.append(os.path.relpath(filename, start=out_dir))

                # save depth
                if "depth" in save_attr_names and self.depth is not None:
                    if mode in ["exr"]:
                        filename = os.path.join(sub_dir, f"depth_{iq:06d}.exr")
                        exr_utils.write_exr(filename, self.depth[ib, iq])  # (h, w)
                    elif mode in ["png", "qoi"]:
                        # save as 16 bit png along with min and max
                        filename = os.path.join(sub_dir, f"depth_{iq:06d}.png")
                        _arr = self.depth[ib, iq]  # (h, w)

                        filename2 = os.path.join(sub_dir, f"depth_{iq:06d}.pnginfo")
                        RGBDImage.save_depth_png_format(
                            depth=_arr, hit_map=self.hit_map[ib, iq], save_f=filename, save_f_pnginfo=filename2
                        )
                    elif mode == "npy":
                        filename = os.path.join(sub_dir, f"depth_{iq:06d}.npy")
                        np.save(filename, self.depth[ib, iq])  # (h, w)
                    else:
                        raise NotImplementedError
                    depth_filenames.append(os.path.relpath(filename, start=out_dir))

                # save normal_w
                hit_map_saved = False
                if "normal_w" in save_attr_names and self.normal_w is not None:
                    if mode in ["exr", "npy", "png"]:
                        if self.hit_map is not None:
                            _normal_w = (
                                self.normal_w * self.hit_map.unsqueeze(-1).to(dtype=self.normal_w.dtype)
                                + (1 - self.hit_map.unsqueeze(-1).to(dtype=self.normal_w.dtype)) * background_color
                            )  # (b, q, h, w, 3xyz)
                        else:
                            _normal_w = self.normal_w  # (b, q, h, w, 3xyz)
                    elif mode == "qoi":
                        # store normal and hit map together
                        if self.hit_map is not None:
                            _normal_w = torch.cat(
                                [
                                    self.normal_w,
                                    self.hit_map.unsqueeze(-1).to(dtype=self.normal_w.dtype) * 2 - 1,  # {-1, 1}
                                ],
                                dim=-1,
                            )  # (b, q, h, w, 4xyzh)
                            hit_map_saved = True
                        else:
                            _normal_w = self.normal_w  # (b, q, h, w, 3xyz)
                    else:
                        raise NotImplementedError

                    if mode == "exr":
                        filename = os.path.join(sub_dir, f"normal_w_{iq:06d}.exr")
                        exr_utils.write_exr(filename, _normal_w[ib, iq])  # (h, w, 3)
                    elif mode == "npy":
                        filename = os.path.join(sub_dir, f"normal_w_{iq:06d}.npy")
                        np.save(filename, _normal_w[ib, iq])  # (h, w, 3)
                    elif mode == "png":
                        # since we know the max range of normal [-1, 1] we can save as uint16
                        # with limited loss
                        filename = os.path.join(sub_dir, f"normal_w_{iq:06d}.png")
                        _arr = (
                            (((_normal_w[ib, iq] + 1) * 0.5) * 65535)
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
                        assert rs is True, f"{filename=}"
                    elif mode == "qoi":
                        # save as 8bit with hitmap in the alpha channel (xyzh).
                        # This is a tradeoff between storage size and quality
                        import qoi

                        filename = os.path.join(sub_dir, f"normal_w_{iq:06d}.qoi")
                        qoi.write(
                            filename,
                            (((_normal_w[ib, iq] + 1) * 0.5) * 255)
                            .detach()
                            .cpu()
                            .clamp(min=0, max=255)
                            .numpy()
                            .astype(np.uint8),  # (h, w, 4xyzh) or (h, w, 3xyz)
                        )
                    else:
                        raise NotImplementedError
                    normal_filenames.append(os.path.relpath(filename, start=out_dir))

                # save hit_map
                if "hit_map" in save_attr_names and self.hit_map is not None and (not hit_map_saved):
                    # We always save a png file (qoi does not support 1 channel images)
                    filename = os.path.join(sub_dir, f"hit_map_{iq:06d}.png")
                    imageio.imwrite(
                        filename,
                        (self.hit_map[ib, iq].float() * 255.0)
                        .detach()
                        .cpu()
                        .float()
                        .clamp(min=0, max=255)
                        .numpy()
                        .astype(np.uint8),
                    )
                    if (self.hit_map.dtype != torch.bool) and (not flag_save_space):
                        filename = os.path.join(sub_dir, f"hit_map_{iq:06d}.exr")
                        exr_utils.write_exr(filename, self.hit_map[ib, iq])  # (h, w)
                    hit_map_filenames.append(os.path.relpath(filename, start=out_dir))

                # save feature
                if "feature" in save_attr_names and self.feature is not None:
                    if mode in ["exr", "png"]:
                        filename = os.path.join(sub_dir, f"feature_{iq:06d}.exr")
                        exr_utils.write_exr(filename, self.feature[ib, iq])  # (h, w, d)
                    elif mode == "npy":
                        filename = os.path.join(sub_dir, f"feature_{iq:06d}.npy")
                        np.save(filename, self.feature[ib, iq])  # (h, w, d)
                    else:
                        raise NotImplementedError
                    normal_filenames.append(os.path.relpath(filename, start=out_dir))

                # other maps
                if self.other_maps is not None:
                    for key in self.other_maps:
                        arr = self.other_maps.get(key, None)  # (h, w, d)
                        if key in save_attr_names and arr is not None:
                            if mode in ["exr", "png", "qoi"] and key == "obj_id":
                                # we assume max number of objects is 65535
                                filename = os.path.join(sub_dir, f"{key}_{iq:06d}.png")
                                assert arr.size(-1) == 1, f"{arr.shape=}, {filename=}"
                                _arr = (
                                    arr[ib, iq].round().detach().cpu().clamp(min=0, max=65535).numpy().astype(np.uint16)
                                )  # (h, w, 1)
                                rs = cv2.imwrite(filename, _arr[..., 0])
                                assert rs is True
                            elif mode in ["exr", "png", "qoi"] and key == "alpha":
                                if alpha_saved:
                                    continue

                                assert arr[ib, iq].shape == (h, w, 1), f"{arr.shape=}, {(h, w)=}"
                                if mode in ["exr", "png"]:
                                    filename = os.path.join(sub_dir, f"{key}_{iq:06d}.png")
                                    imageio.imwrite(
                                        filename,
                                        (arr[ib, iq, ..., 0] * 255)
                                        .detach()
                                        .cpu()
                                        .float()
                                        .clamp(min=0, max=255)
                                        .numpy()
                                        .astype(np.uint8),
                                    )
                                elif mode == "qoi":
                                    # should not be called if rgb is presented
                                    import qoi

                                    filename = os.path.join(sub_dir, f"{key}_{iq:06d}.qoi")
                                    qoi.write(
                                        filename,
                                        (arr[ib, iq, ..., 0] * 255)
                                        .detach()
                                        .cpu()
                                        .float()
                                        .clamp(min=0, max=255)
                                        .numpy()
                                        .astype(np.uint8),
                                    )
                                else:
                                    raise NotImplementedError

                            elif mode in ["exr", "png", "qoi"] and key == "roughness_metallic":
                                assert arr[ib, iq].shape == (h, w, 2), f"{arr.shape=}, {(h, w)=}"
                                _arr = torch.zeros(h, w, 3)
                                _arr[:, :, :2] = arr[ib, iq]
                                _arr[:, :, 2] = arr[ib, iq, :, :, -1]  # better for compression (easier estimation)
                                if mode == "exr":
                                    filename = os.path.join(sub_dir, f"{key}_{iq:06d}.exr")
                                    exr_utils.write_exr(filename, _arr)
                                elif mode == "png":
                                    filename = os.path.join(sub_dir, f"{key}_{iq:06d}.png")
                                    imageio.imwrite(
                                        filename,
                                        (_arr * 255)
                                        .detach()
                                        .cpu()
                                        .float()
                                        .clamp(min=0, max=255)
                                        .numpy()
                                        .astype(np.uint8),
                                    )
                                elif mode == "qoi":
                                    import qoi

                                    filename = os.path.join(sub_dir, f"{key}_{iq:06d}.qoi")
                                    qoi.write(
                                        filename,
                                        (_arr * 255)
                                        .detach()
                                        .cpu()
                                        .float()
                                        .clamp(min=0, max=255)
                                        .numpy()
                                        .astype(np.uint8),
                                    )
                                else:
                                    raise NotImplementedError

                            elif mode in ["exr", "png", "qoi"] and key in ["albedo"]:
                                assert arr.shape[-1] in [1, 3, 4], f"arr.shape={arr.shape}"

                                if mode == "exr":
                                    filename = os.path.join(sub_dir, f"{key}_{iq:06d}.exr")
                                    exr_utils.write_exr(filename, arr[ib, iq])
                                elif mode == "png":
                                    filename = os.path.join(sub_dir, f"{key}_{iq:06d}.png")
                                    imageio.imwrite(
                                        filename,
                                        (arr[ib, iq] * 255)
                                        .detach()
                                        .cpu()
                                        .float()
                                        .clamp(min=0, max=255)
                                        .numpy()
                                        .astype(np.uint8),
                                    )
                                elif mode == "qoi":
                                    import qoi

                                    filename = os.path.join(sub_dir, f"{key}_{iq:06d}.qoi")
                                    qoi.write(
                                        filename,
                                        (arr[ib, iq] * 255)
                                        .detach()
                                        .cpu()
                                        .float()
                                        .clamp(min=0, max=255)
                                        .numpy()
                                        .astype(np.uint8),
                                    )
                                else:
                                    raise NotImplementedError

                            elif mode in ["exr", "png", "qoi"]:
                                filename = os.path.join(sub_dir, f"{key}_{iq:06d}.exr")
                                exr_utils.write_exr(filename, arr[ib, iq])  # (h, w, d)
                            elif mode == "npy":
                                filename = os.path.join(sub_dir, f"{key}_{iq:06d}.npy")
                                np.save(filename, arr[ib, iq])  # (h, w, d)
                            else:
                                raise NotImplementedError
                            if key not in other_map_filenames:
                                other_map_filenames[key] = []
                            other_map_filenames[key].append(os.path.relpath(filename, start=out_dir))

            sub_index_dict = dict()
            sub_index_dict["rgb"] = rgb_filenames
            sub_index_dict["depth"] = depth_filenames
            sub_index_dict["normal_w"] = normal_filenames
            sub_index_dict["hit_map"] = hit_map_filenames
            sub_index_dict.update(other_map_filenames)
            sub_index_dicts.append(sub_index_dict)

        index_dict["sub_index_dicts"] = sub_index_dicts

        json_filename = os.path.join(out_dir, "index.json")
        if os.path.exists(json_filename) and concatenate_along_b:
            with open(json_filename, "r") as f:
                current_json = json.load(f)
            assert current_json["q"] == index_dict["q"]
            assert current_json["h"] == index_dict["h"]
            assert current_json["w"] == index_dict["w"]
            current_json["b"] += index_dict["b"]
            current_json["sub_index_dicts"] += index_dict["sub_index_dicts"]
            index_dict = current_json

        # save json
        with open(json_filename, "w") as f:
            json.dump(index_dict, f, indent=2)

        return index_dict, json_filename

    def save_as_flat(
        self,
        out_dir: str,
        prefix: str,
        overwrite: bool,
        background_color: float = 1.0,
        save_attr_names: T.List[str] = None,
        ib_filename_offset: int = None,
        concatenate_along_b: bool = False,
        mode: str = "png",  # "png", "qoi"
    ):
        """
        Save the rgbd image as flat tar, expected to be used by webdataset.
        Specifically, the files will be

        if mode == "png":
            {out_dir}/{prefix}.index.json
            {out_dir}/{prefix}.cameras.npz
            {out_dir}/{prefix}.{ib:06d}-depth_{iq:06d}.png
            {out_dir}/{prefix}.{ib:06d}-depth_{iq:06d}.pnginfo
            {out_dir}/{prefix}.{ib:06d}-hit_map_{iq:06d}.png
            {out_dir}/{prefix}.{ib:06d}-rgb_{iq:06d}.png
            {out_dir}/{prefix}.{ib:06d}-normal_w_{iq:06d}.png

        if mode == "qoi:
            {out_dir}/{prefix}.index.json
            {out_dir}/{prefix}.cameras.npz
            {out_dir}/{prefix}.{ib:06d}-depth_{iq:06d}.png
            {out_dir}/{prefix}.{ib:06d}-depth_{iq:06d}.pnginfo
            {out_dir}/{prefix}.{ib:06d}-rgb_{iq:06d}.qoi  # (rgba)
            {out_dir}/{prefix}.{ib:06d}-normal_w_{iq:06d}.qoi  # (xyzh)
            {out_dir}/{prefix}.{ib:06d}-hit_map_{iq:06d}.png  # only if normal is not saved

        Args:
            out_dir:
            prefix:
                e.g., "{uid}.{rgbd_random}"
            overwrite:
            background_color:
            save_attr_names:
            ib_filename_offset:
            concatenate_along_b:

        Returns:
            index_dict:
                index dict will not contain prefix
        """
        assert mode in ["png", "qoi"], f"{mode}"

        with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
            # save as the original structure
            index_dict, json_filename = self.save_as(
                out_dir=tmp_dir,
                overwrite=True,
                mode=mode,
                background_color=background_color,
                save_attr_names=save_attr_names,
                flag_save_space=True,
                ib_filename_offset=ib_filename_offset,
                concatenate_along_b=concatenate_along_b,
            )

            # move individual files to flat structure
            os.makedirs(out_dir, exist_ok=True)
            source_root = Path(tmp_dir)
            output_root = Path(out_dir)

            for path in source_root.rglob("*"):  # find all files recursively in tmp_dir
                if path.is_file():
                    relative_path = path.relative_to(tmp_dir)

                    # create the new filename: {uid}.{flat_name}
                    flat_name = "-".join(relative_path.parts)  # eg: 000000/depth_000000.png -> 000000-depth_000000.png
                    new_filename = f"{prefix}.{flat_name}"
                    destination = output_root / new_filename
                    if overwrite and os.path.exists(destination):
                        os.remove(destination)
                    shutil.move(path, destination)

            # 1. convert all / into '-' in index_dict
            # 2. move file
            new_index_dict = copy.deepcopy(index_dict)
            new_index_dict["prefix"] = prefix

            # camera
            filename = new_index_dict["camera"]  # "cameras.npz"
            flat_name = byte_dict_utils.convert_to_flat_structure_filename(
                filename=filename,
            )  # eg: 000000/depth_000000.png -> 000000-depth_000000.png
            new_filename = f"{prefix}.{flat_name}"
            destination = output_root / new_filename
            # if overwrite and os.path.exists(destination):
            #     os.remove(destination)
            # shutil.move( source_root / Path(filename), destination)
            assert os.path.exists(destination), f"{destination} not exists"
            new_index_dict["camera"] = flat_name  # new_filename.split('.', 1)[1]  # remove {uid}.

            sub_index_dicts = new_index_dict["sub_index_dicts"]
            for ib in range(len(sub_index_dicts)):
                for key in sub_index_dicts[ib]:
                    ori_filenames = sub_index_dicts[ib][key]
                    new_filenames = []
                    for filename in ori_filenames:
                        flat_name = byte_dict_utils.convert_to_flat_structure_filename(
                            filename=filename,
                        )  # eg: 000000/depth_000000.png -> 000000-depth_000000.png
                        new_filename = f"{prefix}.{flat_name}"
                        destination = output_root / new_filename
                        # if overwrite and os.path.exists(destination):
                        #     os.remove(destination)
                        # shutil.move(source_root/ Path(filename), destination)
                        assert os.path.exists(destination), f"{destination} not exists"
                        # new_filenames.append(new_filename.split('.', 1)[1])  # remove {uid}.
                        new_filenames.append(flat_name)  # remove {uid}.
                    sub_index_dicts[ib][key] = new_filenames

            # save the index
            new_filename = output_root / f"{prefix}.index.json"
            with open(new_filename, "w") as f:
                json.dump(new_index_dict, f, indent=2)

            return new_index_dict

    @staticmethod
    def load_from_flat(
        index_filename: str,
        bidxs: T.List[int] = None,
        qidxs: T.Union[T.List[int], T.List[T.List[int]], T.List[np.ndarray]] = None,
        attr_names: T.List[str] = None,
        printout: bool = False,
    ) -> "RGBDImage":
        """
        Load from the byte dict returned by webdataset.

        Args:
            index_filename:
                e.g., {root_dir}/{prefix}.index.json
                It contains
                    cameras.npz
                    {ib:06d}-depth_{iq:06d}.png
                    {ib:06d}-depth_{iq:06d}.pnginfo
                    {ib:06d}-hit_map_{iq:06d}.png
                    {ib:06d}-rgb_{iq:06d}.png/qoi
                    {ib:06d}-normal_w_{iq:06d}.png/qoi
            bidxs:
                (num_bidx_to_load,)
            qidxs:
                list of (num_bidx_to_load,), each is a list of (num_qidx_to_load,) for each bidxs.

        Returns:
            rgbd_image
        """

        time_dict = dict()

        # get index_dict
        assert os.path.exists(index_filename), f"{index_filename}"
        root_dir = os.path.dirname(index_filename)

        with open(index_filename, "r") as f:
            index_dict = json.load(f)

        prefix = os.path.basename(index_filename).rsplit(".index.json")[0]

        b, q, h, w = index_dict["b"], index_dict["q"], index_dict["h"], index_dict["w"]
        # note that q is the max number of q supported

        if bidxs is None:
            bidxs = list(range(b))  # (num_b_to_load)
        assert isinstance(bidxs, (list, tuple, np.ndarray)) and len(bidxs) <= b

        if qidxs is None:
            qidxs = [list(range(q))] * len(bidxs)
        elif len(qidxs) == 0:
            qidxs = [[]] * len(bidxs)
        elif isinstance(qidxs[0], int):
            # same qidxs for all b
            qidxs = [qidxs] * len(bidxs)  # (num_b_to_load, num_q_to_load)

        assert isinstance(qidxs, (list, tuple, np.ndarray)) and len(qidxs) == len(bidxs)
        assert isinstance(qidxs[0], (list, tuple, np.ndarray))
        for qidx in qidxs:
            assert len(qidx) == len(qidxs[0])

        # change q to be the final q (not the max q)
        b = len(bidxs)
        q = len(qidxs[0])

        if attr_names is None:
            if len(index_dict["sub_index_dicts"]) > 0:
                attr_names = []
                for attr_name, filenames in index_dict["sub_index_dicts"][0].items():
                    if len(filenames) > 0:
                        attr_names.append(attr_name)
            else:
                attr_names = ["rgb", "depth", "normal_w", "hit_map"]

        # load camera
        stime = timer()
        filename = os.path.join(root_dir, f"{prefix}.{index_dict['camera']}")
        with np.load(filename, allow_pickle=True) as _camera_state_dict:
            camera_state_dict = dict(_camera_state_dict)
        camera_state_dict = utils.to_tensor(camera_state_dict, dtype=torch.float)

        H_c2w = torch.stack(
            [camera_state_dict["H_c2w"][ib][qidxs[idx]] for idx, ib in enumerate(bidxs)],
            dim=0,
        )  # (b', q', 4, 4)
        intrinsic = torch.stack(
            [camera_state_dict["intrinsic"][ib][qidxs[idx]] for idx, ib in enumerate(bidxs)],
            dim=0,
        )  # (b', q', 4, 4)
        if camera_state_dict.get("timestamp", None) is not None and (
            isinstance(camera_state_dict["timestamp"], torch.Tensor) or None not in camera_state_dict["timestamp"]
        ):
            timestamp = torch.stack(
                [camera_state_dict["timestamp"][ib][qidxs[idx]] for idx, ib in enumerate(bidxs)],
                dim=0,
            )  # (b', q')
        else:
            timestamp = None

        camera = Camera(
            H_c2w=H_c2w,  # (b, q, 4, 4)
            intrinsic=intrinsic,  # (b, q, 3, 3)
            width_px=camera_state_dict["width_px"],
            height_px=camera_state_dict["height_px"],
            timestamp=timestamp,  # (b, q) or None
        )  # (b, q)
        time_dict["read_camera"] = timer() - stime

        # load attribute
        sub_index_dicts = index_dict["sub_index_dicts"]  # list of (total_b), attr_name -> (q,)

        # load each attribute
        rgbd = dict()  # attr_name -> tensor (b, q, h, w)
        for attr_name in attr_names:
            stime = timer()
            rgbd[attr_name] = []
            for ib, bidx in enumerate(bidxs):
                tmp = []
                for iq, qidx in enumerate(qidxs[ib]):
                    filename = os.path.join(root_dir, f"{prefix}.{sub_index_dicts[bidx][attr_name][qidx]}")
                    arr = RGBDImage.load_single_file(filename, attr_name)  # (h, w) or (h, w, 3) or (h, w, 4) tensor
                    tmp.append(arr)
                tmp = torch.stack(tmp, dim=0) if len(tmp) > 1 else tmp[0].unsqueeze(0)  # (q, h, w, d)
                rgbd[attr_name].append(tmp)

            time_dict[f"read_{attr_name}"] = timer() - stime

        for attr_name in rgbd:
            stime = timer()
            rgbd[attr_name] = (
                torch.stack(rgbd[attr_name], dim=0) if len(rgbd[attr_name]) > 1 else rgbd[attr_name][0].unsqueeze(0)
            )  # (b, q, h, w, *)
            time_dict[f"stack_{attr_name}"] = timer() - stime

        other_maps = dict()
        # handle addon in rgb and normal_w
        if rgbd.get("rgb", None) is not None and rgbd["rgb"].size(-1) == 4:
            other_maps["alpha"] = rgbd["rgb"][..., 3:4]  # (b, q, h, w, 1)
            rgbd["rgb"] = rgbd["rgb"][..., :3]  # (b, q, h, w, 3rgb)
        if rgbd.get("normal_w", None) is not None and rgbd["normal_w"].size(-1) == 4:
            if rgbd.get("hit_map", None) is None:
                rgbd["hit_map"] = rgbd["normal_w"][..., 3] > 0.5  # (b, q, h, w)
            rgbd["normal_w"] = rgbd["normal_w"][..., :3]  # (b, q, h, w, 3xyz)

        # move additional maps to other_maps
        for attr_name, arr in rgbd.items():
            if attr_name not in {
                "rgb",
                "depth",
                "camera",
                "normal_w",
                "hit_map",
                "feature",
            }:
                if arr.ndim == 4:  # (b, q, h, w)
                    arr = arr.unsqueeze(-1)  # (b, q, h, w, 1)
                other_maps[attr_name] = arr

        if len(other_maps) == 0:
            other_maps = None
        else:
            for attr_name in other_maps:
                if attr_name in rgbd:
                    del rgbd[attr_name]

        if ("depth" in rgbd) and (rgbd["depth"] is not None):
            rgbd["depth"] = rgbd["depth"].reshape(b, q, h, w)
        if ("hit_map" in rgbd) and (rgbd["hit_map"] is not None):
            rgbd["hit_map"] = rgbd["hit_map"].reshape(b, q, h, w)
        if ("rgb" in rgbd) and (rgbd["rgb"] is not None):
            rgbd["rgb"] = rgbd["rgb"][..., :3]
        if ("normal_w" in rgbd) and (rgbd["normal_w"] is not None):
            rgbd["normal_w"] = rgbd["normal_w"][..., :3]

        rgbd = RGBDImage(**rgbd, other_maps=other_maps, camera=camera)  # (b, q, h, w)
        return rgbd

    @staticmethod
    def load_from_byte_dict(
        byte_dict: T.Dict[str, T.Any],
        prefix: T.Optional[str],
        bidxs: T.List[int] = None,
        qidxs: T.Union[T.List[int], T.List[T.List[int]], T.List[np.ndarray]] = None,
        attr_names: T.List[str] = None,
        printout: bool = False,
    ) -> "RGBDImage":
        """
        Load from the byte dict returned by webdataset.

        Args:
            byte_dict:
                key:
                    The key part of the filename ({uid}.{key}) after uid
                value:
                    the byte content of the file

                For example, the byte_dict contains

                    {prefix}.index.json
                    {prefix}.cameras.npz
                    {prefix}.{ib:06d}-depth_{iq:06d}.png
                    {prefix}.{ib:06d}-depth_{iq:06d}.pnginfo
                    {prefix}.{ib:06d}-hit_map_{iq:06d}.png
                    {prefix}.{ib:06d}-rgb_{iq:06d}.png/qoi
                    {prefix}.{ib:06d}-normal_w_{iq:06d}.png/qoi

            prefix:
                we use the key in the byte_dict pointing to the index file.
                In the example above, {prefix}.index.json.
                Note, prefix should not contain {uid}.

                prefix can also be None. In this case, we assume the byte_dict contains

                    index.json
                    cameras.npz
                    {ib:06d}-depth_{iq:06d}.png
                    {ib:06d}-depth_{iq:06d}.pnginfo
                    {ib:06d}-hit_map_{iq:06d}.png
                    {ib:06d}-rgb_{iq:06d}.png/qoi
                    {ib:06d}-normal_w_{iq:06d}.png/qoi

            bidxs:
                (num_bidx_to_load,)
            qidxs:
                list of (num_bidx_to_load,), each is a list of (num_qidx_to_load,) for each bidxs.

        Returns:
            rgbd_image
        """

        time_dict = dict()

        # get index_dict
        index_dict = byte_dict_utils.load_file_from_byte_dict(
            byte_dict=byte_dict,
            filename=f"{prefix}.index.json" if prefix is not None else "index.json",
        )

        b, q, h, w = index_dict["b"], index_dict["q"], index_dict["h"], index_dict["w"]
        # note that q is the max number of q supported

        if bidxs is None:
            bidxs = list(range(b))  # (num_b_to_load)
        assert isinstance(bidxs, (list, tuple, np.ndarray)) and len(bidxs) <= b

        if qidxs is None:
            qidxs = [list(range(q))] * len(bidxs)
        elif len(qidxs) == 0:
            qidxs = [[]] * len(bidxs)
        elif isinstance(qidxs[0], int):
            # same qidxs for all b
            qidxs = [qidxs] * len(bidxs)  # (num_b_to_load, num_q_to_load)

        assert isinstance(qidxs, (list, tuple, np.ndarray)) and len(qidxs) == len(bidxs)
        assert isinstance(qidxs[0], (list, tuple, np.ndarray))
        for qidx in qidxs:
            assert len(qidx) == len(qidxs[0])

        # change q to be the final q (not the max q)
        b = len(bidxs)
        q = len(qidxs[0])

        if attr_names is None:
            if len(index_dict["sub_index_dicts"]) > 0:
                attr_names = []
                for attr_name, filenames in index_dict["sub_index_dicts"][0].items():
                    if len(filenames) > 0:
                        attr_names.append(attr_name)
            else:
                attr_names = ["rgb", "depth", "normal_w", "hit_map"]

        # load camera
        stime = timer()
        camera_state_dict = byte_dict_utils.load_file_from_byte_dict(
            byte_dict=byte_dict,
            filename=f"{prefix}.{index_dict['camera']}" if prefix is not None else f"{index_dict['camera']}",
        )
        camera_state_dict = utils.to_tensor(camera_state_dict, dtype=torch.float)

        H_c2w = torch.stack(
            [camera_state_dict["H_c2w"][ib][qidxs[idx]] for idx, ib in enumerate(bidxs)],
            dim=0,
        )  # (b', q', 4, 4)
        intrinsic = torch.stack(
            [camera_state_dict["intrinsic"][ib][qidxs[idx]] for idx, ib in enumerate(bidxs)],
            dim=0,
        )  # (b', q', 4, 4)
        if camera_state_dict.get("timestamp", None) is not None and (
            isinstance(camera_state_dict["timestamp"], torch.Tensor) or None not in camera_state_dict["timestamp"]
        ):
            timestamp = torch.stack(
                [camera_state_dict["timestamp"][ib][qidxs[idx]] for idx, ib in enumerate(bidxs)],
                dim=0,
            )  # (b', q')
        else:
            timestamp = None

        camera = Camera(
            H_c2w=H_c2w,  # (b, q, 4, 4)
            intrinsic=intrinsic,  # (b, q, 3, 3)
            width_px=camera_state_dict["width_px"],
            height_px=camera_state_dict["height_px"],
            timestamp=timestamp,  # (b, q) or None
        )  # (b, q)
        time_dict["read_camera"] = timer() - stime

        # load attribute
        sub_index_dicts = index_dict["sub_index_dicts"]  # list of (total_b), attr_name -> (q,)

        # load each attribute
        rgbd = dict()  # attr_name -> tensor (b, q, h, w)
        for attr_name in attr_names:
            stime = timer()
            rgbd[attr_name] = []
            for ib, bidx in enumerate(bidxs):
                tmp = []
                for iq, qidx in enumerate(qidxs[ib]):
                    arr = byte_dict_utils.load_single_rgbd_file_from_byte_dict(
                        byte_dict=byte_dict,
                        filename=f"{prefix}.{sub_index_dicts[bidx][attr_name][qidx]}"
                        if prefix is not None
                        else f"{sub_index_dicts[bidx][attr_name][qidx]}",
                        attr_name=attr_name,
                    )  # (h, w) or (h, w, d) torch.Tensor
                    if attr_name == "alpha":
                        if arr.ndim == 2:
                            arr = arr.unsqueeze(-1)  # (h, w, 1)
                    elif attr_name == "roughness_metallic":
                        arr = arr[..., :2]  # (h, w, 2roughness_metallic)

                    tmp.append(arr)

                tmp = torch.stack(tmp, dim=0) if len(tmp) > 1 else tmp[0].unsqueeze(0)  # (q, h, w, d)
                rgbd[attr_name].append(tmp)

            time_dict[f"read_{attr_name}"] = timer() - stime

        for attr_name in rgbd:
            stime = timer()
            rgbd[attr_name] = (
                torch.stack(
                    rgbd[attr_name],
                    dim=0,
                )
                if len(rgbd[attr_name]) > 1
                else rgbd[attr_name][0].unsqueeze(0)
            )  # (b, q, h, w, *)
            time_dict[f"stack_{attr_name}"] = timer() - stime

        other_maps = dict()
        # handle addon in rgb and normal_w
        if rgbd.get("rgb", None) is not None and rgbd["rgb"].size(-1) == 4:
            other_maps["alpha"] = rgbd["rgb"][..., 3:4]  # (b, q, h, w, 1)
            rgbd["rgb"] = rgbd["rgb"][..., :3]  # (b, q, h, w, 3rgb)
        if rgbd.get("normal_w", None) is not None and rgbd["normal_w"].size(-1) == 4:
            if rgbd.get("hit_map", None) is None:
                rgbd["hit_map"] = rgbd["normal_w"][..., 3] > 0.5  # (b, q, h, w)
            rgbd["normal_w"] = rgbd["normal_w"][..., :3]  # (b, q, h, w, 3xyz)

        # move additional maps to other_maps
        for attr_name, arr in rgbd.items():
            if attr_name not in {
                "rgb",
                "depth",
                "camera",
                "normal_w",
                "hit_map",
                "feature",
            }:
                if arr.ndim == 4:  # (b, q, h, w)
                    arr = arr.unsqueeze(-1)  # (b, q, h, w, 1)
                other_maps[attr_name] = arr

        if len(other_maps) == 0:
            other_maps = None
        else:
            for attr_name in other_maps:
                if attr_name in rgbd:
                    del rgbd[attr_name]

        if ("depth" in rgbd) and (rgbd["depth"] is not None):
            rgbd["depth"] = rgbd["depth"].reshape(b, q, h, w)
        if ("hit_map" in rgbd) and (rgbd["hit_map"] is not None):
            rgbd["hit_map"] = rgbd["hit_map"].reshape(b, q, h, w)
        if ("rgb" in rgbd) and (rgbd["rgb"] is not None):
            # make sure we get the straight rgb (without alpha)
            rgbd["rgb"] = rgbd["rgb"][..., :3]
        if ("normal_w" in rgbd) and (rgbd["normal_w"] is not None):
            rgbd["normal_w"] = rgbd["normal_w"][..., :3]

        if "depth" not in rgbd:
            rgbd["depth"] = None

        rgbd = RGBDImage(**rgbd, other_maps=other_maps, camera=camera)  # (b, q, h, w)
        return rgbd

    @staticmethod
    def save_index_file_and_camera(
        out_dir: str,
        camera: "Camera",
        save_attr_names: T.List[str] = None,
        remove_dir_if_exists: bool = True,
        mode: str = "png",  # "png", "qoi"
    ):
        """
        Save the index file to mimic save_as.
        Only support png mode with save_file_size on.
        This function is for debugging.

        saved structure:

        out_dir:
        -- index.json
        -- cameras.npz
        -- {ib}
          -- rgb_xxxx.png/qoi (rgb)
          -- depth_xxxx.exr  (bw)  or depth_xxxx.npy
          -- normal_w_xxxx.exr (rgb)  or normal_w_xxxx.npy or normal_w_xxxx.png/qoi
          -- hit_map_xxxx.png
          -- raw_rgb_xxxx.exr (rgb)  if in other map
          -- obj_id_xxxx.exr (rgb)  if in other map
        """

        if remove_dir_if_exists and os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        os.makedirs(out_dir, exist_ok=True)

        if save_attr_names is None:
            save_attr_names = ["rgb", "normal_w", "depth", "hit_map"]

        b, q, _4, _4 = camera.H_c2w.shape
        h, w = camera.height_px, camera.width_px

        index_dict = dict(
            b=b,
            q=q,
            h=h,
            w=w,
        )

        # save camera
        camera_state_dict = camera.state_dict()
        camera_state_dict = utils.to_numpy(camera_state_dict)
        camera_filename = os.path.join(out_dir, f"cameras.npz")
        np.savez(camera_filename, **camera_state_dict)  # (b, q)
        index_dict["camera"] = os.path.relpath(camera_filename, start=out_dir)

        # create the filenames for each file
        sub_index_dicts = []
        for ib in range(b):
            sub_dir = os.path.join(out_dir, f"{ib:06d}")
            os.makedirs(sub_dir, exist_ok=True)

            rgb_filenames = []
            normal_filenames = []
            depth_filenames = []
            hit_map_filenames = []
            other_map_filenames = dict()
            for iq in range(q):
                # save rgb
                if "rgb" in save_attr_names:
                    if mode == "png":
                        filename = os.path.join(sub_dir, f"rgb_{iq:06d}.png")
                    elif mode == "qoi":
                        filename = os.path.join(sub_dir, f"rgb_{iq:06d}.qoi")
                    else:
                        raise NotImplementedError(mode)
                    rgb_filenames.append(os.path.relpath(filename, start=out_dir))

                # save depth
                if "depth" in save_attr_names:
                    # save as 16 bit png along with min and max
                    filename = os.path.join(sub_dir, f"depth_{iq:06d}.png")
                    depth_filenames.append(os.path.relpath(filename, start=out_dir))

                # save normal_w
                if "normal_w" in save_attr_names:
                    if mode == "png":
                        # since we know the max range of normal [-1, 1] we can save as uint16
                        # with limited loss
                        filename = os.path.join(sub_dir, f"normal_w_{iq:06d}.png")
                    elif mode == "qoi":
                        filename = os.path.join(sub_dir, f"normal_w_{iq:06d}.qoi")
                    else:
                        raise NotImplementedError(mode)
                    normal_filenames.append(os.path.relpath(filename, start=out_dir))

                # save hit_map
                if "hit_map" in save_attr_names:
                    if mode == "png":
                        # We always save a png file
                        filename = os.path.join(sub_dir, f"hit_map_{iq:06d}.png")
                        hit_map_filenames.append(os.path.relpath(filename, start=out_dir))
                    elif mode == "qoi":
                        # hit map will be saved with normal map
                        pass
                    else:
                        raise NotImplementedError(mode)

            sub_index_dict = dict()
            sub_index_dict["rgb"] = rgb_filenames  # (q,)
            sub_index_dict["depth"] = depth_filenames  # (q,)
            sub_index_dict["normal_w"] = normal_filenames  # (q,)
            sub_index_dict["hit_map"] = hit_map_filenames  # (q,)
            sub_index_dict.update(other_map_filenames)  # (q,)
            sub_index_dicts.append(sub_index_dict)

        index_dict["sub_index_dicts"] = sub_index_dicts  # (b,)

        # save json
        json_filename = os.path.join(out_dir, "index.json")
        with open(json_filename, "w") as f:
            json.dump(index_dict, f, indent=2)

        return index_dict, json_filename

    @staticmethod
    def save_depth_png_format(*, depth: torch.Tensor, hit_map: torch.Tensor, save_f: str, save_f_pnginfo: str):
        import cv2

        hit_map = hit_map.bool()
        if hit_map.sum() > 0:
            min_arr = depth[hit_map].min()
            max_arr = depth[hit_map].max()
        else:
            # no hit
            min_arr = torch.zeros(1, dtype=depth.dtype, device=depth.device)
            max_arr = torch.ones(1, dtype=depth.dtype, device=depth.device)
        depth = (depth - min_arr) * (65535.0 / torch.clamp(max_arr - min_arr, min=1e-9))
        # to convert back: _arr = arr * (max_arr - min_arr) + min_arr (so the clamp is ok)
        assert depth.isfinite().all(), f"max_arr: {max_arr}, min_arr: {min_arr}"
        depth = depth.detach().cpu().float().clamp(min=0, max=65535).numpy().astype(np.uint16)  # (h, w)
        # save as grayscale
        rs = cv2.imwrite(str(save_f), depth)
        assert rs is True
        with open(str(save_f_pnginfo), "w") as f:
            f.write(f"{min_arr.item()}\n")
            f.write(f"{max_arr.item()}\n")

    @staticmethod
    def save_single_file(
        out_dir: str,
        index_dict: T.Dict[str, T.Any],
        arr: torch.Tensor,
        attr_name: str,
        bidx: int,
        qidx: int,
        hit_map: torch.Tensor = None,  # (h, w) bool or alpha (h, w, 1)
        background_color: float = 1,
        mode: str = "png",  # "png", "qoi"
    ):
        """
        Save individual file given the filename.

        Args:
            index_dict:
                b:
                q:
                h:
                w:
                sub_index_dicts:
                    (b,) of dict
                    "rgb": (q,) filenames
                    "depth": (q,) filenames
                    "normal_w": (q,) filenames
                    "hit_map": (q,) filenames
            arr:
                rgb: (h, w, 3)  [0, 1]
                hit_map: (h, w)
                depth: (h, w)
                normal_w: (h, w, 3)


        saved structure:

        out_dir:
        -- index.json
        -- cameras.npz
        -- {ib}
          -- rgb_xxxx.png (rgb)
          -- depth_xxxx.exr  (bw)  or depth_xxxx.npy
          -- normal_w_xxxx.exr (rgb)  or normal_w_xxxx.npy
          -- hit_map_xxxx.png
          -- raw_rgb_xxxx.exr (rgb)  if in other map
          -- obj_id_xxxx.exr (rgb)  if in other map
        """
        import cv2

        os.makedirs(out_dir, exist_ok=True)

        h, w, *_ = arr.shape
        assert bidx < index_dict["b"]
        assert qidx < index_dict["q"]
        assert h == index_dict["h"]
        assert w == index_dict["w"]

        sub_dir = os.path.join(out_dir, f"{bidx:06d}")
        os.makedirs(sub_dir, exist_ok=True)

        filename = os.path.join(
            out_dir,
            index_dict["sub_index_dicts"][bidx][attr_name][qidx],
        )
        if attr_name == "rgb":
            if hit_map is not None:
                arr = torch.cat(
                    [
                        arr.float(),  # (h, w, 3)
                        hit_map.float().unsqueeze(-1) if hit_map.ndim == 2 else hit_map.float(),
                    ],
                    dim=-1,
                )  # (h, w, 4rgba)

            if mode == "png":
                assert filename.lower().endswith(".png"), f"{filename}"
                imageio.imwrite(
                    filename,
                    np.clip(arr.detach().cpu().float().numpy() * 255.0, a_min=0, a_max=255).astype(np.uint8),
                )
            elif mode == "qoi":
                assert filename.lower().endswith(".qoi"), f"{filename}"
                import qoi

                qoi.write(
                    filename,
                    np.clip(arr.detach().cpu().float().numpy() * 255.0, a_min=0, a_max=255).astype(np.uint8),
                )
            else:
                raise NotImplementedError

        elif attr_name == "depth":
            # save as 16 bit png along with min and max
            assert filename.endswith(".png")
            assert hit_map is not None
            filename2 = f"{os.path.splitext(filename)[0]}.pnginfo"
            RGBDImage.save_depth_png_format(depth=arr, hit_map=hit_map, save_f=filename, save_f_pnginfo=filename2)

        elif attr_name == "normal_w":
            if mode == "png":
                if hit_map is not None:
                    arr = (
                        arr.float() * hit_map.unsqueeze(-1).float()
                        + (1 - hit_map.unsqueeze(-1).float()) * background_color
                    )
                arr = torch.nn.functional.normalize(arr, dim=-1)
                # since we know the max range of normal [-1, 1] we can save as uint16
                # with limited loss
                arr = (
                    (((arr + 1) * 0.5) * 65535).detach().cpu().clamp(min=0, max=65535).numpy().astype(np.uint16)
                )  # (h, w, 3xyz)
                # since we use opencv (which takes images as bgr)
                # arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # (h, w, 3zyx)
                arr_bgr = arr[..., ::-1]  # (h, w, 3zyx)
                rs = cv2.imwrite(filename, arr_bgr)
                assert rs == True
            elif mode == "qoi":
                arr = torch.nn.functional.normalize(arr, dim=-1)
                if hit_map is not None:
                    _hit_map = hit_map if len(hit_map.shape) == 2 else hit_map.unsqueeze(-1)
                    arr = torch.cat(
                        [
                            arr.float(),  # [-1, 1]
                            hit_map.float() * 2 - 1,  # [-1, 1]
                        ],
                        dim=-1,
                    )  # (h, w, 4)

                arr = (
                    (((arr + 1) * 0.5) * 255).detach().cpu().clamp(min=0, max=255).numpy().astype(np.uint8)
                )  # (h, w, 3xyz)
                import qoi

                qoi.write(filename, arr)
            else:
                raise NotImplementedError
        elif attr_name == "hit_map":
            # always save as png (qoi does not support 1 channel images)
            imageio.imwrite(
                filename,
                (arr.float() * 255.0).detach().cpu().float().clamp(min=0, max=255).numpy().astype(np.uint8),
            )
            # if mode == "png":
            #     imageio.imwrite(
            #         filename,
            #         (arr.float() * 255.0).detach().cpu().float().clamp(min=0, max=255).numpy().astype(np.uint8),
            #     )
            # elif mode == "qoi":
            #     if len(arr.shape) == 2:
            #         arr = arr.unsqueeze(-1)
            #     if arr.size(-1) == 1:
            #         arr = arr.expand(-1, -1, 3)
            #     import qoi
            #     qoi.write(
            #         filename,
            #         (arr.float() * 255.0).detach().cpu().float().clamp(min=0, max=255).numpy().astype(np.uint8),
            #     )
            # else:
            #     raise NotImplementedError
        else:
            raise NotImplementedError(f"{attr_name}")

        return filename

    @staticmethod
    def load_single_file(
        filename: str,
        attr_name: str,
        zipfile_obj: zipfile.ZipFile | None = None,
    ) -> torch.Tensor:
        if zipfile_obj is None:
            ret_info = RGBDImage.load_single_file_from_disk(
                filename=filename,
                attr_name=attr_name,
            )
        else:
            ret_info = RGBDImage.load_single_file_from_zip(
                zipfile_obj=zipfile_obj,
                filename=filename,
                attr_name=attr_name,
            )
        return ret_info

    @staticmethod
    def load_single_file_from_disk(filename: str, attr_name: str) -> torch.Tensor:
        r"""
        Read a single file (rgb, depth, hit_map, or normal map) of
        an rgbd_image.

        Args:
            filename:
                filename of the file to be read
            attr_name:
                what kind of file is it:
                'rgb', 'depth', 'normal_w', 'hit_map'

        Returns:
            (h, w) or (h, w, d) torch.Tensor
        """
        assert os.path.exists(filename), f"{filename=}"
        if filename.lower().endswith(".npy"):
            # print(f'reading {filename} as npy')
            arr = np.load(filename)  # (h, w) or (h, w, 3)
            arr = torch.from_numpy(arr)
        elif filename.lower().endswith(".exr"):
            # print(f'reading {filename} as exr')
            arr = exr_utils.read_exr(filename)  # (h, w, c)
            arr = torch.from_numpy(arr)
        elif filename.lower().endswith(".png") or filename.lower().endswith(".qoi"):
            arr = img_utils.imread(
                filename=filename,
                mode="scaled",
            )  # (c, h, w) float32 [0, 1]
            if arr.size(0) == 1:
                arr = arr.squeeze(0)  # (h, w)
            else:
                arr = arr.permute(1, 2, 0)  # (h, w, c)

            if attr_name == "normal_w":
                assert len(arr.shape) == 3, f"{len(arr.shape)=}"
                assert arr.shape[2] >= 3, f"{arr.shape=}"
                # normalization should consider only the normal part (not the potential hitmap)
                arr[..., :3] = arr[..., :3] * 2 - 1
                arr[..., :3] = torch.nn.functional.normalize(arr[..., :3], dim=-1)
            elif attr_name == "depth":
                filename2 = f"{os.path.splitext(filename)[0]}.pnginfo"
                assert os.path.exists(filename2), f"{filename2}"
                with open(filename2, "r") as file:
                    lines = file.readlines()
                assert len(lines) == 2, f"{len(lines)=}"
                min_arr = float(lines[0].strip())  # Convert the first line to an integer
                max_arr = float(lines[1].strip())  # Convert the second line to an integer
                arr = arr * (max_arr - min_arr) + min_arr
            elif attr_name == "hit_map":
                assert arr.dtype == torch.float32, f"{arr.dtype=}"
                arr = arr > 1e-6
            elif attr_name == "alpha":
                # We do not enforce alpha to be bool as we need its non-binary values to do alpha blending
                assert arr.dtype == torch.float32, f"{arr.dtype=}"  # [h, w]
                assert arr.ndim == 2, f"{arr.shape=}"
                arr = arr[..., None]  # we specifically add the channel
            elif attr_name == "roughness_metallic":
                # We do not enforce alpha to be bool as we need its non-binary values to do alpha blending
                assert arr.dtype == torch.float32, f"{arr.dtype=}"  # [h, w]
                arr = arr[..., :2]  # only take the first two (roughness, metallic)
        else:
            raise NotImplementedError
        return arr

    @staticmethod
    def load_single_file_from_zip(zipfile_obj: zipfile.ZipFile, filename: str, attr_name: str) -> torch.Tensor:
        r"""
        Read a single file (rgb, depth, hit_map, or normal map) of
        an rgbd_image from a zipfile.

        Args:
            zipfile_obj:
                zipfile.ZipFile object to read from
            filename:
                filename of the file to be read (path within zip)
            attr_name:
                what kind of file is it:
                'rgb', 'depth', 'normal_w', 'hit_map'

        Returns:
            (h, w) or (h, w, d) torch.Tensor
        """
        try:
            zipfile_obj.getinfo(filename)
        except KeyError:
            raise AssertionError(f"{filename} not found in zip")

        if filename.lower().endswith(".npy"):
            # Read npy from zip
            with zipfile_obj.open(filename) as f:
                arr = np.load(f)  # (h, w) or (h, w, 3)
                arr = torch.from_numpy(arr)
        elif filename.lower().endswith(".exr"):
            # Read exr from zip
            arr = exr_utils.read_exr(filename, zipfile_obj=zipfile_obj)  # (h, w, c)
            arr = torch.from_numpy(arr)
        elif filename.lower().endswith(".png") or filename.lower().endswith(".qoi"):
            # Read png from zip
            with zipfile_obj.open(filename) as f:
                data = f.read()
                arr = img_utils.imread(
                    filename=io.BytesIO(data),
                    mode="scaled",
                )  # (c, h, w) float32 [0, 1]
                if arr.size(0) == 1:
                    arr = arr.squeeze(0)  # (h, w)
                else:
                    arr = arr.permute(1, 2, 0)  # (h, w, c)

            if attr_name == "normal_w":
                assert len(arr.shape) == 3, f"{len(arr.shape)=}"
                assert arr.shape[2] >= 3, f"{arr.shape=}"
                # normalization should consider only the normal part (not the potential hitmap)
                arr[..., :3] = arr[..., :3] * 2 - 1
                arr[..., :3] = torch.nn.functional.normalize(arr[..., :3], dim=-1)
            elif attr_name == "depth":
                filename2 = f"{os.path.splitext(filename)[0]}.pnginfo"
                try:
                    zipfile_obj.getinfo(filename2)
                except KeyError:
                    raise AssertionError(f"{filename2} not found in zip")
                with zipfile_obj.open(filename2) as f:
                    lines = f.read().decode("utf-8").splitlines()
                assert len(lines) == 2, f"{len(lines)=}"
                min_arr = float(lines[0].strip())
                max_arr = float(lines[1].strip())
                arr = arr * (max_arr - min_arr) + min_arr
            elif attr_name == "hit_map":
                assert arr.dtype == torch.float32, f"{arr.dtype=}"
                arr = arr > 1e-6
            elif attr_name == "alpha":
                assert arr.dtype == torch.float32, f"{arr.dtype=}"  # [h, w]
                assert arr.ndim == 2, f"{arr.shape=}"
                arr = arr[..., None]  # we specifically add the channel
            elif attr_name == "roughness_metallic":
                # We do not enforce alpha to be bool as we need its non-binary values to do alpha blending
                assert arr.dtype == torch.float32, f"{arr.dtype=}"  # [h, w]
                arr = arr[..., :2]  # only take the first two (roughness, metallic)
        else:
            raise NotImplementedError
        return arr

    @staticmethod
    def load_from(
        index_filename: str,
        bidxs: T.List[int] = None,
        qidxs: T.Union[T.List[int], T.List[T.List[int]], T.List[np.ndarray]] = None,
        attr_names: T.List[str] = None,
        printout: bool = False,
        zipfile_obj: zipfile.ZipFile | None = None,
    ) -> "RGBDImage":
        if zipfile_obj is None:
            ret_info = RGBDImage.load_from_disk(
                index_filename=index_filename,
                bidxs=bidxs,
                qidxs=qidxs,
                attr_names=attr_names,
                printout=printout,
            )
        else:
            ret_info = RGBDImage.load_from_zip(
                zipfile_obj=zipfile_obj,
                index_filename=index_filename,
                bidxs=bidxs,
                qidxs=qidxs,
                attr_names=attr_names,
                printout=printout,
            )
        return ret_info

    @staticmethod
    def load_from_disk(
        index_filename: str,
        bidxs: T.List[int] = None,
        qidxs: T.Union[T.List[int], T.List[T.List[int]], T.List[np.ndarray]] = None,
        attr_names: T.List[str] = None,
        printout: bool = False,
    ) -> "RGBDImage":
        """
        Load rgbd from the data stored with `save_as_npy`.
        """

        assert os.path.exists(index_filename), f"{index_filename}"
        with open(index_filename, "r") as f:
            index_dict = json.load(f)

        # print(f'loading {index_filename}: {index_dict}')

        root_dir = os.path.dirname(index_filename)

        b, q, h, w = index_dict["b"], index_dict["q"], index_dict["h"], index_dict["w"]
        # note that q is the max number of q supported

        if bidxs is None:
            bidxs = list(range(b))
        assert isinstance(bidxs, (list, tuple, np.ndarray)) and len(bidxs) <= b

        if qidxs is None:
            qidxs = [list(range(q))] * len(bidxs)
        elif len(qidxs) == 0:
            qidxs = [[]] * len(bidxs)
        elif isinstance(qidxs[0], int):
            # same qidxs for all b
            qidxs = [qidxs] * len(bidxs)

        assert isinstance(qidxs, (list, tuple, np.ndarray)) and len(qidxs) == len(bidxs)
        assert isinstance(qidxs[0], (list, tuple, np.ndarray)), f"{type(qidxs[0])}"
        for qidx in qidxs:
            assert len(qidx) == len(qidxs[0]), f"{len(qidx)=}, {len(qidxs[0])=}"

        # change q to be the final q (not the max q)
        b = len(bidxs)
        q = len(qidxs[0])

        if attr_names is None:
            if len(index_dict["sub_index_dicts"]) > 0:
                attr_names = list(index_dict["sub_index_dicts"][0].keys())
            else:
                attr_names = ["rgb", "depth", "normal_w", "hit_map", "alpha"]

        out_dict = dict()

        # read camera
        cam_filename = os.path.join(root_dir, index_dict["camera"])
        if printout:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else None
            print(f"(worker {worker_id}) loading camera file {cam_filename}", flush=True)
            assert os.path.exists(cam_filename), f"{cam_filename} not exist"
        with np.load(cam_filename, allow_pickle=True) as _camera_state_dict:
            camera_state_dict = dict(_camera_state_dict)

        camera_state_dict = utils.to_tensor(camera_state_dict, dtype=torch.float)

        H_c2w = torch.stack(
            [camera_state_dict["H_c2w"][ib][qidxs[idx]] for idx, ib in enumerate(bidxs)],
            dim=0,
        )  # (b', q', 4, 4)
        intrinsic = torch.stack(
            [camera_state_dict["intrinsic"][ib][qidxs[idx]] for idx, ib in enumerate(bidxs)],
            dim=0,
        )  # (b', q', 4, 4)
        if camera_state_dict.get("timestamp", None) is not None and (
            isinstance(camera_state_dict["timestamp"], torch.Tensor) or None not in camera_state_dict["timestamp"]
        ):
            timestamp = torch.stack(
                [camera_state_dict["timestamp"][ib][qidxs[idx]] for idx, ib in enumerate(bidxs)],
                dim=0,
            )  # (b', q')
        else:
            timestamp = None

        if printout:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else None
            print(
                f"(worker {worker_id}) converting H_c2w ({H_c2w.shape}) and intrinsic ({intrinsic.shape}) to tensor",
                flush=True,
            )
        stime = timer()
        ttime = timer() - stime
        if printout:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else None
            print(
                f"(worker {worker_id}) finished converting H_c2w and intrinsic to tensor, used {ttime} secs", flush=True
            )

        camera = Camera(
            H_c2w=H_c2w,
            intrinsic=intrinsic,
            width_px=camera_state_dict["width_px"],
            height_px=camera_state_dict["height_px"],
            timestamp=timestamp,
        )  # (b', q')

        for attr_name in attr_names:
            out_dict[attr_name] = []

        missing_attr_names = set()
        for idx, ib in enumerate(bidxs):
            sub_index_dict = index_dict["sub_index_dicts"][ib]

            for attr_name in attr_names:
                if (
                    attr_name not in sub_index_dict
                    or sub_index_dict[attr_name] is None
                    or len(sub_index_dict[attr_name]) == 0
                ):
                    missing_attr_names.add(attr_name)
                    continue

                # get filenames
                filenames = [os.path.join(root_dir, sub_index_dict[attr_name][iq]) for iq in qidxs[idx]]

                if printout:
                    worker_info = torch.utils.data.get_worker_info()
                    worker_id = worker_info.id if worker_info is not None else None
                    print(f"(worker {worker_id}) loading {attr_name} (len = {len(filenames)})", flush=True)

                stime = timer()
                arr_times = []
                arrs = []
                for filename in filenames:  # iterate over q
                    if printout:
                        assert os.path.exists(filename), f"{filename} not exist"
                    sstime = timer()
                    arr = RGBDImage.load_single_file(filename, attr_name)  # (h, w) or (h, w, 3) or (h, w, 4) tensor
                    arr_times.append(timer() - sstime)
                    arrs.append(arr.contiguous())
                ttime = timer() - stime
                if printout or ttime > 100:
                    worker_info = torch.utils.data.get_worker_info()
                    worker_id = worker_info.id if worker_info is not None else None
                    msg = (
                        f"(worker {worker_id}) finished loading {attr_name} (len = {len(filenames)}), used {ttime} secs"
                    )
                    if ttime > 10:
                        msg += "\n"
                        for ii in range(len(arrs)):
                            msg += (
                                f"  ({ii}) type: {type(arrs[ii])}, shape: {arrs[ii].shape}, "
                                f"dtype: {arrs[ii].dtype}, contiguous: {arrs[ii].is_contiguous()}, "
                                f"{filenames[ii]}\n"
                            )
                    print(msg, flush=True)

                stime = timer()
                if len(arrs) > 1:
                    # arr = np.stack(arrs, axis=0)  # (q, h, w, 3) or (q, h, w)
                    # arr = torch.stack([torch.from_numpy(arr).contiguous() for arr in arrs], dim=0).numpy()  # (q, h, w, 3) or (q, h, w)
                    arr = torch.stack(arrs, dim=0)  # (q, h, w, 3) or (q, h, w) or (q, h, w, 4)
                else:
                    # arr = np.expand_dims(arrs[0], axis=0)  # (q=1, h, w, 3) or (q=1, h, w)
                    arr = arrs[0].unsqueeze(0)  # (q=1, h, w, 3) or (q=1, h, w) or (q=1, h, w, 4)
                ttime = timer() - stime

                if printout or ttime > 10:
                    worker_info = torch.utils.data.get_worker_info()
                    worker_id = worker_info.id if worker_info is not None else None
                    msg = f"(worker: {worker_id}) finished stacking {attr_name} ({arr.shape}), used {ttime} secs "
                    if ttime > 10:
                        msg += "\n"
                        for ii in range(len(arrs)):
                            msg += (
                                f"  ({ii}) type: {type(arrs[ii])}, shape: {arrs[ii].shape}, "
                                f"dtype: {arrs[ii].dtype}, contiguous: {arrs[ii].is_contiguous()}, "
                                f"{filenames[ii]}\n"
                            )
                    print(msg, flush=True)

                out_dict[attr_name].append(arr)

        for attr_name in attr_names:
            if attr_name not in missing_attr_names:
                if printout:
                    worker_info = torch.utils.data.get_worker_info()
                    worker_id = worker_info.id if worker_info is not None else None
                    print(f"(worker: {worker_id}) start stacking {attr_name} along b", flush=True)
                stime = timer()
                if len(out_dict[attr_name]) > 1:
                    # out_dict[attr_name] = np.stack(out_dict[attr_name], axis=0)  # (b, q, h, w)
                    out_dict[attr_name] = torch.stack(out_dict[attr_name], dim=0)  # (b, q, h, w)
                else:
                    # out_dict[attr_name] = np.expand_dims(out_dict[attr_name][0], axis=0)  # (b=1, q, h, w)
                    out_dict[attr_name] = out_dict[attr_name][0].unsqueeze(0)  # (b=1, q, h, w)
                ttime = timer() - stime
                if printout:
                    worker_info = torch.utils.data.get_worker_info()
                    worker_id = worker_info.id if worker_info is not None else None
                    print(
                        f"(worker: {worker_id}) finished stacking {attr_name} ({out_dict[attr_name].shape}), used {ttime} secs",
                        flush=True,
                    )
            else:
                out_dict[attr_name] = None

        if printout:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else None
            print(f"(worker: {worker_id}) converting out_dict ({list(out_dict.keys())}) to tensor", flush=True)
        stime = timer()
        out_dict = utils.to_tensor(out_dict)
        ttime = timer() - stime
        if printout:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else None
            print(
                f"(worker: {worker_id}) finished converting out_dict ({list(out_dict.keys())}) to tensor, used {ttime} secs",
                flush=True,
            )

        # separate into other_maps
        other_maps = dict()

        # handle addon in rgb and normal_w
        if out_dict.get("rgb", None) is not None and out_dict["rgb"].size(-1) == 4:
            other_maps["alpha"] = out_dict["rgb"][..., 3:4]  # (b, q, h, w, 1)
            out_dict["rgb"] = out_dict["rgb"][..., :3]  # (b, q, h, w, 3rgb)
        if out_dict.get("normal_w", None) is not None and out_dict["normal_w"].size(-1) == 4:
            if out_dict.get("hit_map", None) is None:
                out_dict["hit_map"] = out_dict["normal_w"][..., 3] > 0.5  # (b, q, h, w)
            out_dict["normal_w"] = out_dict["normal_w"][..., :3]  # (b, q, h, w, 3xyz)

        # overwrite if actually save
        for key in out_dict:
            if key not in {"rgb", "depth", "normal_w", "hit_map", "feature"}:
                other_maps[key] = out_dict[key]

        if len(other_maps) == 0:
            other_maps = None
        if ("depth" in out_dict) and (out_dict["depth"] is not None):
            out_dict["depth"] = out_dict["depth"].reshape(b, q, h, w)
        if ("hit_map" in out_dict) and (out_dict["hit_map"] is not None):
            out_dict["hit_map"] = out_dict["hit_map"].reshape(b, q, h, w)
        if ("rgb" in out_dict) and (out_dict["rgb"] is not None):
            out_dict["rgb"] = out_dict["rgb"][..., :3]
        if ("normal_w" in out_dict) and (out_dict["normal_w"] is not None):
            out_dict["normal_w"] = out_dict["normal_w"][..., :3]

        # make sure (b, q, h, w) -> (b, q, h, w, 1) in other maps
        if other_maps is not None:
            for key, arr in other_maps.items():
                if arr is not None and arr.ndim == 4:
                    other_maps[key] = arr.unsqueeze(-1)

        rgbd = RGBDImage(
            camera=camera,
            rgb=out_dict.get("rgb", None),
            depth=out_dict.get("depth", None),
            normal_w=out_dict.get("normal_w", None),
            hit_map=out_dict["hit_map"].bool() if out_dict.get("hit_map", None) is not None else None,
            feature=out_dict.get("feature", None),
            other_maps=other_maps,
        )
        return rgbd

    @staticmethod
    def load_from_zip(
        zipfile_obj: zipfile.ZipFile,
        index_filename: str,
        bidxs: T.List[int] = None,
        qidxs: T.Union[T.List[int], T.List[T.List[int]], T.List[np.ndarray]] = None,
        attr_names: T.List[str] = None,
        printout: bool = False,
    ) -> "RGBDImage":
        """
        Load rgbd from the data stored in a zipfile with `save_as_npy` format.
        """

        # Read index file from zipfile
        try:
            with zipfile_obj.open(index_filename) as f:
                index_dict = json.load(f)
        except KeyError:
            raise FileNotFoundError(f"{index_filename} not found in zipfile")

        # print(f'loading {index_filename}: {index_dict}')

        root_dir = os.path.dirname(index_filename)

        b, q, h, w = index_dict["b"], index_dict["q"], index_dict["h"], index_dict["w"]
        # note that q is the max number of q supported

        if bidxs is None:
            bidxs = list(range(b))
        assert isinstance(bidxs, (list, tuple, np.ndarray)) and len(bidxs) <= b

        if qidxs is None:
            qidxs = [list(range(q))] * len(bidxs)
        elif len(qidxs) == 0:
            qidxs = [[]] * len(bidxs)
        elif isinstance(qidxs[0], int):
            # same qidxs for all b
            qidxs = [qidxs] * len(bidxs)

        assert isinstance(qidxs, (list, tuple, np.ndarray)) and len(qidxs) == len(bidxs)
        assert isinstance(qidxs[0], (list, tuple, np.ndarray))
        for qidx in qidxs:
            assert len(qidx) == len(qidxs[0]), f"{len(qidx)=}, {len(qidxs[0])=}"

        # change q to be the final q (not the max q)
        b = len(bidxs)
        q = len(qidxs[0])

        if attr_names is None:
            if len(index_dict["sub_index_dicts"]) > 0:
                attr_names = list(index_dict["sub_index_dicts"][0].keys())
            else:
                attr_names = ["rgb", "depth", "normal_w", "hit_map", "alpha"]

        out_dict = dict()

        # read camera from zipfile
        cam_filename = os.path.join(root_dir, index_dict["camera"])
        if printout:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else None
            print(f"(worker {worker_id}) loading camera file {cam_filename}", flush=True)

        try:
            with zipfile_obj.open(cam_filename) as cam_file:
                # Read the entire file into memory first
                cam_data = cam_file.read()
                # Use BytesIO to create a file-like object that np.load can read
                with np.load(io.BytesIO(cam_data), allow_pickle=True) as _camera_state_dict:
                    camera_state_dict = dict(_camera_state_dict)
        except KeyError:
            raise FileNotFoundError(f"{cam_filename} not found in zipfile")

        H_c2w = np.stack(
            [camera_state_dict["H_c2w"][ib][qidxs[idx]] for idx, ib in enumerate(bidxs)], axis=0
        )  # (b', q', 4, 4)
        intrinsic = np.stack(
            [camera_state_dict["intrinsic"][ib][qidxs[idx]] for idx, ib in enumerate(bidxs)], axis=0
        )  # (b', q', 4, 4)
        if camera_state_dict.get("timestamp", None) is not None and not np.any(camera_state_dict["timestamp"] == None):
            timestamp = np.stack(
                [camera_state_dict["timestamp"][ib][qidxs[idx]] for idx, ib in enumerate(bidxs)], axis=0
            )
            timestamp = torch.from_numpy(timestamp)  # (b', q')
        else:
            timestamp = None

        if printout:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else None
            print(
                f"(worker {worker_id}) converting H_c2w ({H_c2w.shape}) and intrinsic ({intrinsic.shape}) to tensor",
                flush=True,
            )
        stime = timer()
        H_c2w = torch.from_numpy(H_c2w)
        intrinsic = torch.from_numpy(intrinsic)
        ttime = timer() - stime
        if printout:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else None
            print(
                f"(worker {worker_id}) finished converting H_c2w and intrinsic to tensor, used {ttime} secs", flush=True
            )

        camera = Camera(
            H_c2w=H_c2w,
            intrinsic=intrinsic,
            width_px=camera_state_dict["width_px"],
            height_px=camera_state_dict["height_px"],
            timestamp=timestamp,
        )  # (b', q')

        for attr_name in attr_names:
            out_dict[attr_name] = []

        missing_attr_names = set()
        for idx, ib in enumerate(bidxs):
            sub_index_dict = index_dict["sub_index_dicts"][ib]

            for attr_name in attr_names:
                if (
                    attr_name not in sub_index_dict
                    or sub_index_dict[attr_name] is None
                    or len(sub_index_dict[attr_name]) == 0
                ):
                    missing_attr_names.add(attr_name)
                    continue

                # get filenames (relative paths in zipfile)
                filenames = [os.path.join(root_dir, sub_index_dict[attr_name][iq]) for iq in qidxs[idx]]

                if printout:
                    worker_info = torch.utils.data.get_worker_info()
                    worker_id = worker_info.id if worker_info is not None else None
                    print(f"(worker {worker_id}) loading {attr_name} (len = {len(filenames)})", flush=True)

                stime = timer()
                arr_times = []
                arrs = []
                for filename in filenames:  # iterate over q
                    sstime = timer()
                    # Load from zipfile instead of filesystem
                    arr = RGBDImage.load_single_file_from_zip(zipfile_obj, filename, attr_name)
                    arr_times.append(timer() - sstime)
                    # print(f'{filename}: {arr.shape}')
                    arrs.append(arr.contiguous())
                ttime = timer() - stime
                if printout or ttime > 10:
                    worker_info = torch.utils.data.get_worker_info()
                    worker_id = worker_info.id if worker_info is not None else None
                    msg = (
                        f"(worker {worker_id}) finished loading {attr_name} (len = {len(filenames)}), used {ttime} secs"
                    )
                    if ttime > 10:
                        msg += "\n"
                        for ii in range(len(arrs)):
                            msg += (
                                f"  ({ii}) type: {type(arrs[ii])}, shape: {arrs[ii].shape}, "
                                f"dtype: {arrs[ii].dtype}, contiguous: {arrs[ii].is_contiguous()}, "
                                f"{filenames[ii]}\n"
                            )
                    print(msg, flush=True)

                stime = timer()
                if len(arrs) > 1:
                    arr = torch.stack(arrs, dim=0)  # (q, h, w, 3) or (q, h, w) or (q, h, w, 4)
                else:
                    arr = arrs[0].unsqueeze(0)  # (q=1, h, w, 3) or (q=1, h, w) or (q=1, h, w, 4)
                ttime = timer() - stime

                if printout or ttime > 10:
                    worker_info = torch.utils.data.get_worker_info()
                    worker_id = worker_info.id if worker_info is not None else None
                    msg = f"(worker: {worker_id}) finished stacking {attr_name} ({arr.shape}), used {ttime} secs "
                    if ttime > 10:
                        msg += "\n"
                        for ii in range(len(arrs)):
                            msg += (
                                f"  ({ii}) type: {type(arrs[ii])}, shape: {arrs[ii].shape}, "
                                f"dtype: {arrs[ii].dtype}, contiguous: {arrs[ii].is_contiguous()}, "
                                f"{filenames[ii]}\n"
                            )
                    print(msg, flush=True)

                out_dict[attr_name].append(arr)

        for attr_name in attr_names:
            if attr_name not in missing_attr_names:
                if printout:
                    worker_info = torch.utils.data.get_worker_info()
                    worker_id = worker_info.id if worker_info is not None else None
                    print(f"(worker: {worker_id}) start stacking {attr_name} along b", flush=True)
                stime = timer()
                if len(out_dict[attr_name]) > 1:
                    out_dict[attr_name] = torch.stack(out_dict[attr_name], dim=0)  # (b, q, h, w)
                else:
                    out_dict[attr_name] = out_dict[attr_name][0].unsqueeze(0)  # (b=1, q, h, w)
                ttime = timer() - stime
                if printout:
                    worker_info = torch.utils.data.get_worker_info()
                    worker_id = worker_info.id if worker_info is not None else None
                    print(
                        f"(worker: {worker_id}) finished stacking {attr_name} ({out_dict[attr_name].shape}), used {ttime} secs",
                        flush=True,
                    )
            else:
                out_dict[attr_name] = None

        if printout:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else None
            print(f"(worker: {worker_id}) converting out_dict ({list(out_dict.keys())}) to tensor", flush=True)
        stime = timer()
        out_dict = utils.to_tensor(out_dict)
        ttime = timer() - stime
        if printout:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else None
            print(
                f"(worker: {worker_id}) finished converting out_dict ({list(out_dict.keys())}) to tensor, used {ttime} secs",
                flush=True,
            )

        # separate into other_maps
        other_maps = dict()

        # handle addon in rgb and normal_w
        if out_dict.get("rgb", None) is not None and out_dict["rgb"].size(-1) == 4:
            other_maps["alpha"] = out_dict["rgb"][..., 3:4]  # (b, q, h, w, 1)
            out_dict["rgb"] = out_dict["rgb"][..., :3]  # (b, q, h, w, 3rgb)
        if out_dict.get("normal_w", None) is not None and out_dict["normal_w"].size(-1) == 4:
            if out_dict.get("hit_map", None) is None:
                out_dict["hit_map"] = out_dict["normal_w"][..., 3] > 0.5  # (b, q, h, w)
            out_dict["normal_w"] = out_dict["normal_w"][..., :3]  # (b, q, h, w, 3xyz)

        # overwrite if actually save
        for key in out_dict:
            if key not in {"rgb", "depth", "normal_w", "hit_map", "feature"}:
                other_maps[key] = out_dict[key]

        if len(other_maps) == 0:
            other_maps = None
        if ("depth" in out_dict) and (out_dict["depth"] is not None):
            out_dict["depth"] = out_dict["depth"].reshape(b, q, h, w)
        if ("hit_map" in out_dict) and (out_dict["hit_map"] is not None):
            out_dict["hit_map"] = out_dict["hit_map"].reshape(b, q, h, w)
        if ("rgb" in out_dict) and (out_dict["rgb"] is not None):
            out_dict["rgb"] = out_dict["rgb"][..., :3]
        if ("normal_w" in out_dict) and (out_dict["normal_w"] is not None):
            out_dict["normal_w"] = out_dict["normal_w"][..., :3]

        rgbd = RGBDImage(
            camera=camera,
            rgb=out_dict.get("rgb", None),
            depth=out_dict.get("depth", None),
            normal_w=out_dict.get("normal_w", None),
            hit_map=out_dict["hit_map"].bool() if out_dict.get("hit_map", None) is not None else None,
            feature=out_dict.get("feature", None),
            other_maps=other_maps,
        )
        return rgbd

    def save_as_npbgpp_input(
        self,
        output_dirs: T.List[str],
        type: str,  # 'input', 'ground-truth'
        start_idx: int = 0,
        exist_ok: bool = True,
        overwrite: bool = False,
        hit_only: bool = False,
    ):
        """
        Save as the format used by NPBG++ (https://github.com/rakhimovv/npbgpp).
        We use the DTU dataset's format:
        - image:  000000.png ... 000010.png  (all images, input and ground truth target)
        - mask:   000.png ... 010.png  (binary masks, 1: object, 0: background)
        - mvs_pc.ply:  xyz input point cloud (can be without rgb color)
        - cameras.npz:  world_mat_0 ... world_mat_10  (4x4 Projection matrix from world to image, ie, intrinsics * H_w2c)

        image background is white.
        reccommend using b==1.

        Args:
            output_dirs:
                (b,) output folders for each b

        Returns:
            list of output_dir, one per each b
        """
        background_color = 1.0
        assert len(output_dirs) == self.rgb.size(0)
        for output_dir in output_dirs:
            if overwrite:
                try:
                    shutil.rmtree(output_dir)
                except:
                    pass
            if os.path.exists(output_dir) and not exist_ok:
                raise RuntimeError
            os.makedirs(output_dir, exist_ok=True)

        if type == "input":
            save_pcd = True
        else:
            save_pcd = False

        if hit_only and self.hit_map is not None:
            hit_map = self.hit_map.float()  # (b, q, h, w)
        else:
            hit_map = torch.ones_like(self.depth)  # (b, q, h, w)

        if self.hit_map is not None:
            actual_hit_map = self.hit_map.float()  # (b, q, h, w)
        else:
            actual_hit_map = torch.ones_like(self.depth)  # (b, q, h, w)

        b, q, h, w, _3 = self.rgb.shape
        assert self.camera.width_px == w, f"self.camera.width_px = {self.camera.width_px}, w = {w}"
        assert self.camera.height_px == h, f"self.camera.height_px = {self.camera.height_px}, h = {h}"

        H_w2c = self.camera.get_H_w2c()  # (b, q, 4, 4)
        intrinsics_44 = torch.zeros_like(H_w2c)
        intrinsics_44[..., :3, :3] = self.camera.intrinsic
        intrinsics_44[..., 3, 3] = 1
        P_w2c = linalg_utils.matmul(
            intrinsics_44,
            H_w2c,
        )  # (b, q, 4, 4)

        masked_rgb = self.rgb * hit_map.unsqueeze(-1)
        masked_rgb = masked_rgb + (1 - hit_map).unsqueeze(-1).expand_as(masked_rgb) * background_color
        ply_filenames = []
        for ib in range(self.rgb.size(0)):
            output_dir = output_dirs[ib]
            img_dir = os.path.join(output_dir, "image")
            mask_dir = os.path.join(output_dir, "mask")
            ply_filename = os.path.join(output_dir, "mvs_pc.ply")
            camera_filename = os.path.join(output_dir, "cameras.npz")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            camera_dict = dict()
            if os.path.exists(camera_filename):
                tmp = np.load(camera_filename)
                for n in tmp:
                    camera_dict[n] = tmp[n]

            for iq in range(self.rgb.size(1)):
                # rgb
                filename = os.path.join(img_dir, f"{start_idx + iq:06d}.png")
                imageio.imwrite(filename, (masked_rgb[ib, iq] * 255.0).detach().cpu().numpy().astype(np.uint8))
                # mask (actual hit_map if available)
                filename = os.path.join(mask_dir, f"{start_idx + iq:03d}.png")
                imageio.imwrite(
                    filename, ((actual_hit_map[ib, iq] > 0.5) * 255.0).detach().cpu().numpy().astype(np.uint8)
                )
                # camera
                camera_dict[f"world_mat_{start_idx + iq}"] = P_w2c[ib, iq].detach().cpu().numpy()  # (4, 4)

            # save camera npz
            np.savez(camera_filename, **camera_dict)

            # point cloud
            ply_filenames.append(ply_filename)

        # save point cloud
        if save_pcd:
            point_cloud = self.get_pcd()  # (b, n, 3)
            point_cloud.save_as_npbgpp(
                filenames=ply_filenames,
                overwrite=overwrite,
            )

    def save_as_rtmv(
        self,
        output_dirs: T.List[str],
        start_idx: int = 0,
        exist_ok: bool = True,
        overwrite: bool = False,
        srgb_to_linear: bool = True,
        hit_only: bool = False,
    ) -> T.List[str]:
        """
        Save as the format used in the RTMV dataset https://www.cs.umd.edu/~mmeshry/projects/rtmv/.

        Each scene in the dataset contains
        - {id:05d}.depth.exr  # (h, w, 3) grayscale, float32, background is set to -1e10,  "ray traveling distance", not z in camera coordinate
        - {id:05d}.exr  # (h, w, 4), float32,  rgba   (alpha: foreground mask [0, 1])
        - {id:05d}.json  # camera information, see below
        - {id:05d}.seg.exr  # (h, w, 3), float32, [0, 1]  1 foreground, 0 background

        ex:
        - 00000.depth.exr
        - 00000.exr
        - 00000.json
        - 00000.seg.exr

        RTMV uses blender coordinate system (x to right, y to far, z to up).
        We use opengl's (x to right, y to up, z to us).
        So we need to first convert our H_c2w to H_b2w before saving the camera.
        Moreover, Kaolin uses intrinsic matrix to convert world coordinate
        (x to right, y to up, z to us) to image coordinate (x to right, y to down,
        z to far). So we also need to handle this.

        # Camera information
        {
            "camera_data": {
                "cam2world": [  # it is the "transpose" of H_c2w
                    [
                        -0.6331584453582764,
                        0.7740222811698914,
                        0.0,
                        0.0
                    ],
                    [
                        -0.09314906597137451,
                        -0.07619690895080566,
                        0.9927322864532471,
                        0.0
                    ],
                    [
                        0.7683968544006348,
                        0.6285567283630371,
                        0.12034416198730469,
                        0.0
                    ],
                    [
                        0.5917379856109619,
                        0.5100606083869934,
                        0.17243748903274536,
                        1.0
                    ]
                ],
                "camera_look_at": {  # in world coordinate
                    "at": [
                        -0.05198334725487073,
                        -0.016510273653732227,
                        0.0716196402346452
                    ],
                    "eye": [
                        0.5917379969124275,
                        0.5100606335385356,
                        0.17243748276105064
                    ],
                    "up": [
                        0,
                        0,
                        1
                    ]
                },
                "camera_view_matrix": [  # H_w2c.T
                    [
                        -0.6331584453582764,
                        -0.09314906597137451,
                        0.7683968544006348,
                        0.0
                    ],
                    [
                        0.7740222811698914,
                        -0.07619690895080566,
                        0.6285567283630371,
                        0.0
                    ],
                    [
                        0.0,
                        0.9927322864532471,
                        0.12034416198730469,
                        0.0
                    ],
                    [
                        -0.020134389400482178,
                        -0.07719936966896057,
                        -0.7960434556007385,
                        1.0
                    ]
                ],
                "height": 1600,
                "intrinsics": {
                    "cx": 800.0,
                    "cy": 800.0,
                    "fx": 1931.371337890625,
                    "fy": 1931.371337890625
                },
                "location_world": [
                    0.5917379856109619,
                    0.5100606083869934,
                    0.17243748903274536
                ],
                "width": 1600
            },
            "objects": []
        }

        Also note that:
        kaolin-wisp calls linear_to_srgb when loading data (but directly save data in srgb domain).

        Args:
            output_dirs:
                (b,) output folders for each b

        Returns:
            list of output_dir, one per each b
        """

        import wisp.ops.image

        assert len(output_dirs) == self.rgb.size(0)
        for output_dir in output_dirs:
            if overwrite:
                try:
                    shutil.rmtree(output_dir)
                except:
                    pass
            if os.path.exists(output_dir) and not exist_ok:
                raise RuntimeError
            os.makedirs(output_dir, exist_ok=True)

        b, q, h, w, _3 = self.rgb.shape
        assert self.camera.width_px == w, f"self.camera.width_px = {self.camera.width_px}, w = {w}"
        assert self.camera.height_px == h, f"self.camera.height_px = {self.camera.height_px}, h = {h}"
        with torch.no_grad():
            if hit_only and self.hit_map is not None:
                hit_map = self.hit_map.float()  # (b, q, h, w)
            else:
                hit_map = torch.ones_like(self.depth)  # (b, q, h, w)

            # Our H_c2w actually contains two parts:
            # H_c2w (H_i2w) = H_c2l * H_i2c,
            # where i is the image coordinate: c: x to right, y to down, z to far
            #       c is the camara coordinate (our invariant)
            #       l is the world coordinate in OpenGL convention:
            #       l: x to right, y to up, z to us
            # However, rtmv uses blender convention:
            #       b: x to right, y to far, z to up
            # and it should not contain H_i2c.

            H_i2w = self.camera.H_c2w

            H_c2i = torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ]
            ).to(dtype=torch.float, device=self.camera.H_c2w.device)
            H_c2w = linalg_utils.matmul(
                H_i2w,
                H_c2i.view(1, 1, 4, 4),
            )  # (b, q, 4, 4)

            # represent opengl axis in blender axis
            H_w2b = torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ]
            ).to(dtype=torch.float, device=self.camera.H_c2w.device)
            H_c2b = linalg_utils.matmul(
                H_w2b.view(1, 1, 4, 4),
                H_c2w,
            )  # (b, q, 4, 4)

            rgba = (
                torch.cat(
                    [
                        self.rgb,
                        hit_map.unsqueeze(-1).to(dtype=self.rgb.dtype),
                    ],
                    dim=-1,
                )
                .float()
                .detach()
                .cpu()
                .numpy()
            )  # (b, q, h, w, 4)

            xyz_w = utils.compute_3d_xyz(
                z_map=self.depth,  # (b, q, h, w)
                intrinsic=self.camera.intrinsic,  # (b, q, 3, 3)
                H_c2w=self.camera.H_c2w,  # (b, q, 4, 4,)
            )["xyz_w"]  # (b, q, h, w, 3)

        for ib in range(self.rgb.size(0)):
            output_dir = output_dirs[ib]

            for iq in range(self.rgb.size(1)):
                # camera
                camera_filename = os.path.join(output_dir, f"{start_idx + iq:05d}.json")
                camera_dict = dict(camera_data=dict(), objects=[])
                camera_data = camera_dict["camera_data"]
                camera_data["cam2world"] = H_c2b[ib, iq].t().detach().cpu().tolist()
                camera_data["camera_look_at"] = dict(
                    at=(-H_c2b[ib, iq, :3, 2] + H_c2b[ib, iq, :3, 3]).detach().cpu().tolist(),  # notice the negative
                    eye=H_c2b[ib, iq, :3, 3].detach().cpu().tolist(),
                    up=H_c2b[ib, iq, :3, 1].detach().cpu().tolist(),
                )
                camera_data["location_world"] = H_c2b[ib, iq, :3, 3].detach().cpu().tolist()
                camera_data["camera_view_matrix"] = (
                    rigid_motion.inv_homogeneous_tensors(H_c2b[ib, iq]).t().detach().cpu().tolist()
                )
                camera_data["height"] = self.camera.height_px
                camera_data["width"] = self.camera.width_px
                camera_data["intrinsics"] = dict(
                    cx=self.camera.intrinsic[ib, iq, 0, 2].detach().cpu().item(),
                    cy=self.camera.intrinsic[ib, iq, 1, 2].detach().cpu().item(),
                    fx=self.camera.intrinsic[ib, iq, 0, 0].detach().cpu().item(),
                    fy=self.camera.intrinsic[ib, iq, 1, 1].detach().cpu().item(),
                )
                with open(camera_filename, "w") as f:
                    json.dump(camera_dict, f, indent=2)

                # rgb
                filename = os.path.join(output_dir, f"{start_idx + iq:05d}.exr")
                if srgb_to_linear:
                    img = wisp.ops.image.srgb_to_linear(img=torch.from_numpy(rgba[ib, iq, ..., :3]))  # (h, w, 3)
                    img = (
                        torch.cat([img, torch.from_numpy(rgba[ib, iq, ..., 3:4])], dim=-1).detach().cpu().numpy()
                    )  # (h, w, 4)
                    pyexr.write(filename, img)  # (h, w, 4)
                else:
                    pyexr.write(filename, rgba[ib, iq])  # (h, w, 4)

                # mask
                filename = os.path.join(output_dir, f"{start_idx + iq:05d}.seg.exr")
                pyexr.write(
                    filename,
                    hit_map[ib, iq].unsqueeze(-1).expand(-1, -1, 3).detach().cpu().numpy(),
                )  # (h, w, 3)

                # depth (ray travelling distance)
                # we still use H_c2w since the distance is the same under rigid transformation
                filename = os.path.join(output_dir, f"{start_idx + iq:05d}.depth.exr")
                dist = torch.linalg.norm(
                    xyz_w[ib, iq] - self.camera.H_c2w[ib, iq, :3, 3].reshape(1, 1, 3),
                    ord=2,
                    dim=-1,
                    keepdim=True,
                ).expand(-1, -1, 3)  # (h, w, 3)
                dist = dist.masked_fill(hit_map[ib, iq].unsqueeze(-1) < 0.5, -1e10)
                dist = dist.detach().cpu().numpy()

                pyexr.write(filename, dist)  # (h, w, 3)

        return output_dirs

    def save_as_dsnerf(
        self,
        output_dirs: T.List[str],
        type: str,  # 'train', 'test', 'video'
        exist_ok: bool = True,
        overwrite: bool = False,
        background_color: float = 1.0,
        hit_only: bool = False,
    ) -> T.List[str]:
        """
        Save as the format used by DSNeRF https://github.com/dunbar12138/DSNeRF.

        Each scene is composed of
        - train_images.npy:  (num_views, h, w, 3)  float  rgb
        - train_poses.npy: (num_views, 3, 5), [H_c2w,  (h_px, w_px, f_px)^T].
            Note that H_c2w should not contain the H_i2c.

                The intrinsic matrix: [
                    [f_px  0    w_px/2],
                    [0   f_px   h_px/2],
                    [0     0      1   ],
                ]
        - train_depth.npy: (T.List[T.Dist[str, ndarray]]) an array/list (num_views,), each of the element is a dict:
            - "depth":  (num_points,)  z in the camera coordinate (after extrinsic matrix)
            - "coord":  (num_points, 2)  uv in the image coordinate [0, w] [0, h] (where the image center is (w/2, h/2))
            - "error":  (num_points,)  the weights that will be used to weight the depth loss. Not used, so can be set to 1
        - video_poses.npy: (num_views1, 3, 5)  The camera pose [H_c2w, hwf] to render the final video
        - bds.npy:  (20, 2) not sure, but the max(bds) is used to determine "far" (max of z)
                and min(bds) is used to determine "near"
        - test_images.npy:   (num_views, h, w, 3)  float
        - test_poses.npy:  (num_views, 3, 5), [H_c2w,  (h_px, w_px, f_px)^T].

        Args:
            output_dirs:
                (b,) output folders for each b

        Returns:
            list of output_dir, one per each b

        Note:
            pytorch-nerf and DSNeRF uses intrinsic matrix to handle the difference between
            world coordinate (x: right, y: up, z:us) to image coordinate (x: right, y: down, z: far).
            Whereas we use the extrinsic matrix. We need to correct this.
        """
        assert type in {"train", "test", "video"}
        assert len(output_dirs) == self.rgb.size(0)
        for output_dir in output_dirs:
            if overwrite:
                try:
                    shutil.rmtree(output_dir)
                except:
                    pass
            if os.path.exists(output_dir) and not exist_ok:
                raise RuntimeError
            os.makedirs(output_dir, exist_ok=True)

        b, q, h, w, _3 = self.rgb.shape
        assert self.camera.width_px == w, f"self.camera.width_px = {self.camera.width_px}, w = {w}"
        assert self.camera.height_px == h, f"self.camera.height_px = {self.camera.height_px}, h = {h}"
        with torch.no_grad():
            if hit_only and self.hit_map is not None:
                hit_map = self.hit_map.float()  # (b, q, h, w)
            else:
                hit_map = torch.ones_like(self.depth)  # (b, q, h, w)
            rgb = self.rgb * hit_map.unsqueeze(-1) + (1 - hit_map).unsqueeze(-1).expand_as(self.rgb) * background_color
            valid_depth = torch.logical_and(
                hit_map,
                self.depth < 1e6,
            )  # (b, q, h, w)

            # Our H_c2w actually contains two parts:
            # H_c2w (H_i2w) = H_c2l * H_i2c,
            # where i is the image coordinate: c: x to right, y to down, z to far
            #       c is the camara coordinate (our invariant)
            #       l is the world coordinate in OpenGL convention:
            #       l: x to right, y to up, z to us
            # However, ds-nerf only wants H_c2l.

            H_i2w = self.camera.H_c2w
            H_c2i = torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ]
            ).to(dtype=torch.float, device=self.camera.H_c2w.device)
            H_c2w = linalg_utils.matmul(
                H_i2w,
                H_c2i.view(1, 1, 4, 4),
            )  # (b, q, 4, 4)

            # H_l2c = torch.eye(4).to(device=self.camera.H_c2w.device)
            # H_l2c[1, 1] = -1
            # H_l2c[2, 2] = -1
            #
            # H_l2w = self.camera.H_c2w @ H_l2c.view(1, 1, 4, 4)  # (b, q, 4, 4)

        for ib in range(self.rgb.size(0)):
            output_dir = output_dirs[ib]

            # rgb
            imgs = rgb[ib].detach().cpu().numpy()  # (q, h, w, 3)
            filename = os.path.join(output_dir, f"{type}_images.npy")
            np.save(filename, imgs)

            # poses
            poses = torch.zeros(q, 3, 5, dtype=torch.float)  # (q, 3, 5)
            poses[:, :3, :4] = H_c2w[ib, :, :3].detach().cpu()  # (q, 3, 4)
            poses[..., 0, 4] = self.camera.height_px
            poses[..., 1, 4] = self.camera.width_px
            poses[..., 2, 4] = self.camera.intrinsic[ib, :, 0, 0].detach().cpu()
            poses = poses.numpy()
            # print(f'self.camera.height_px = {self.camera.height_px}')
            # print(f'self.camera.width_px = {self.camera.width_px}')
            # print('poses')
            # print(poses)
            filename = os.path.join(output_dir, f"{type}_poses.npy")
            np.save(filename, poses)

            # depth (z in camera coordinate) > 0
            depth_dicts = []
            for iq in range(q):
                depth = self.depth[ib, iq]  # (h, w)
                depth = depth[valid_depth[ib, iq] > 0.5]  # (n,)
                # generate u v on sensor coord
                u, v = torch.meshgrid(
                    torch.arange(0, w),
                    torch.arange(0, h),
                    indexing="xy",
                )  # u: (h, w) for x,  v: (h, w) for y in the sensor coord
                u = u[valid_depth[ib, iq] > 0.5]  # (n,)
                v = v[valid_depth[ib, iq] > 0.5]  # (n,)
                coord = torch.stack((u, v), dim=-1)  # (n, 2)
                #
                weights = torch.ones_like(depth)  # (n,)
                depth_dict = dict(
                    depth=depth.detach().cpu().numpy(),
                    coord=coord.detach().cpu().numpy(),
                    error=weights.detach().cpu().numpy(),
                )
                depth_dicts.append(depth_dict)
            filename = os.path.join(output_dir, f"{type}_depths.npy")
            np.save(filename, depth_dicts, allow_pickle=True)

            if type == "train":
                max_depth = self.depth[ib][valid_depth[ib]].max().detach().cpu().item()
                min_depth = self.depth[ib][valid_depth[ib]].min().detach().cpu().item()
                bds = np.array([[min_depth, max_depth]]).astype(np.float32)
                filename = os.path.join(output_dir, f"bds.npy")
                np.save(filename, bds)

        return output_dirs

    def save_as_llff(
        self,
        output_dirs: T.List[str],
        start_idx: int = 0,
        exist_ok: bool = True,
        overwrite: bool = False,
        hit_only: bool = False,
    ) -> T.List[str]:
        """
        Save as the format used by LLFF.

        LLFF processes the outputs of COLMAP, and we will only output a subset of it.

        See: https://github.com/Fyusion/LLFF#using-your-own-poses-without-running-colmap

        Each scene in the dataset contains
        - images/
            - xxxx.jpg  (total of n (h, w, 3) images)
        - images_2/
            - xxxx.png  (total of n (h, w, 3/4) images
        - poses_bounds.npy: (n, 17)
            the first dimension is ordered by sorted filenames of files in images
            poses[:, 0:12]: vec(H_c2b) (x down, y right, z us)
            poses[:, 12:15]:  hwf in pixel
            poses[:, 15:17]: z_min z_max in the camera coordinate

        We can get the camera poses in opengl coordinate, intrinsics (hwf), z_c range by
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])  # (3, 5, n)
        bds = poses_arr[:, -2:].transpose([1, 0])  # (2, n)  z_near, z_far

        # Convert R matrix from the form [down right back] to [right up back]
        poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)  # (3, 5, n)

        poses[:3, :4] is now H_c2w in the coordinate of x to right, y to up, z to us
        poses[:3, 4] is now hwf in pixel

        intrinsic_matrix = np.array([[f, 0, w/2],
                                     [0, f, h/2],
                                     [0, 0, 1]]).astype(np.float32)

        Args:
            output_dirs:
                (b,) output folders for each b

        Returns:
            list of output_dir, one per each b
        """

        assert len(output_dirs) == self.rgb.size(0)
        for output_dir in output_dirs:
            if overwrite:
                try:
                    shutil.rmtree(output_dir)
                except:
                    pass
            if os.path.exists(output_dir) and not exist_ok:
                raise RuntimeError
            os.makedirs(output_dir, exist_ok=True)

        b, q, h, w, _3 = self.rgb.shape
        assert self.camera.width_px == w, f"self.camera.width_px = {self.camera.width_px}, w = {w}"
        assert self.camera.height_px == h, f"self.camera.height_px = {self.camera.height_px}, h = {h}"
        with torch.no_grad():
            if hit_only and self.hit_map is not None:
                hit_map = self.hit_map.float()  # (b, q, h, w)
            else:
                hit_map = torch.ones_like(self.depth)  # (b, q, h, w)

            if self.hit_map is not None:
                actual_hit_map = self.hit_map.float()  # (b, q, h, w)
            else:
                actual_hit_map = torch.ones_like(self.depth)  # (b, q, h, w)

            # Our H_c2w actually contains two parts:
            # H_c2w (H_i2w) = H_c2l * H_i2c,
            # where i is the image coordinate: c: x to right, y to down, z to far
            #       c is the camara coordinate (our invariant)
            #       l is the world coordinate in OpenGL convention:
            #       l: x to right, y to up, z to us
            # However, llff uses the coodinate:
            #       b: x to down, y to right, z to us
            # and it should not contain H_i2c.

            H_i2w = self.camera.H_c2w

            H_b2i = torch.tensor(
                [
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ]
            ).to(dtype=torch.float, device=self.camera.H_c2w.device)
            H_b2w = linalg_utils.matmul(
                H_i2w,
                H_b2i.view(1, 1, 4, 4),
            )  # (b, q, 4, 4)

            masked_rgb = self.rgb * hit_map.unsqueeze(-1)
            masked_rgb = masked_rgb + (1 - hit_map).unsqueeze(-1)  # background is white

            rgba = (
                torch.cat(
                    [
                        masked_rgb,
                        hit_map.unsqueeze(-1).to(dtype=masked_rgb.dtype),
                    ],
                    dim=-1,
                )
                .float()
                .detach()
                .cpu()
            )  # (b, q, h, w, 4)

            nan_depth = self.depth.clone()
            nan_depth[actual_hit_map < 0.5] = torch.nan

        for ib in range(self.rgb.size(0)):
            output_dir = output_dirs[ib]

            camera_filename = os.path.join(output_dir, f"poses_bounds.npy")
            if os.path.exists(camera_filename):
                ori_poses_arr = np.load(camera_filename)  # (n, 17)
            else:
                ori_poses_arr = np.zeros([0, 17], dtype=np.float32)
            assert start_idx == ori_poses_arr.shape[0]

            os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "images_2"), exist_ok=True)

            dd = self.depth[actual_hit_map > 0.5]
            if dd.numel() > 0:
                z_min_global = dd.min().detach().cpu().item()
                z_max_global = dd.min().detach().cpu().item()
            else:
                z_min_global = 1e-3
                z_max_global = 1

            poses_arrs = []
            for iq in range(self.rgb.size(1)):
                # camera poses, hwf, z_min, z_max
                cam_pose = H_b2w[ib, iq, :3].detach().cpu().numpy()  # (3, 4)
                hwf = np.array(
                    [
                        self.camera.height_px,
                        self.camera.width_px,
                        self.camera.intrinsic[ib, iq, 0, 0].detach().cpu().item(),
                    ]
                )  # (3,)

                dd = self.depth[ib, iq]
                dd = dd[actual_hit_map[ib, iq] > 0.5]
                if dd.numel() > 0:
                    z_min = dd.min().detach().cpu().item()
                    z_max = dd.max().detach().cpu().item()
                else:
                    z_min = z_min_global
                    z_max = z_max_global

                # arr = np.concatenate([cam_pose.ravel(), hwf, np.array([z_min, z_max])], axis=0)  # (17,)
                arr = np.concatenate([cam_pose, hwf.reshape(-1, 1)], axis=1)  # (3, 5)
                arr = np.concatenate([arr.ravel(), np.array([z_min, z_max])], axis=0)  # (17,)
                poses_arrs.append(arr)

                # rgb
                filename = os.path.join(output_dir, "images", f"{start_idx + iq:05d}.jpg")
                imageio.imwrite(filename, (rgba[ib, iq, :, :, :3] * 255.0).detach().cpu().numpy().astype(np.uint8))
                filename = os.path.join(output_dir, "images_2", f"{start_idx + iq:05d}.png")
                imageio.imwrite(filename, (rgba[ib, iq] * 255.0).detach().cpu().numpy().astype(np.uint8))

            # write camera pose to npy
            poses_arrs = np.stack(poses_arrs, axis=0)  # (iq, 17)
            poses_arrs = np.concatenate([ori_poses_arr, poses_arrs], axis=0)  # (n, 17)
            np.save(camera_filename, poses_arrs, allow_pickle=False)

        return output_dirs


class Mesh:
    def __init__(
        self,
        mesh: T.Union[o3d.geometry.TriangleMesh, str],
        scale: T.Optional[float] = 1.0,
        center_w: T.Optional[T.List[float]] = (0.0, 0.0, 0.0),
        preprocess_mesh: bool = True,
        compute_raycasting_scene: bool = True,
    ):
        if isinstance(mesh, str):
            # load mesh
            mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(mesh, enable_post_processing=True)

        # preprocess mesh (clean uv, shift to center, rescale to [-scale, scale])
        mesh = mesh_utils.preprocess_mesh(mesh=mesh, scale=scale, center_w=center_w, clean=preprocess_mesh)

        self.scale = np.max(mesh.get_axis_aligned_bounding_box().get_half_extent())
        self.center_w = mesh.get_axis_aligned_bounding_box().get_center()
        self.mesh = mesh

        # for ray tracing
        if compute_raycasting_scene:
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
            self.scene = o3d.t.geometry.RaycastingScene()
            self.scene.add_triangles(mesh_t)
        else:
            self.scene = None

    def get_raw_mesh(
        self,
        compute_vertex_normal: bool = True,
        merge_texture: bool = True,
        clone: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> "RawMesh":
        """
        Extract information from the mesh and create a RawMesh to be rendered with
        nvdiffrast.

        Returns:
            the raw mesh
        """

        info_dict = mesh_utils.extract_o3d_mesh_raw_information(
            o3d_mesh=self.mesh,
            compute_triangle_normal=False,
            compute_vertex_normal=compute_vertex_normal,
            clean_mesh=True,
            merge_texture=merge_texture,
            clone=clone,
        )

        raw_mesh = RawMesh(
            vertex_xyz_w=info_dict["vertex_xyz_w"].float().to(device).contiguous(),  # (n, 3xyz_w)
            triangles=info_dict["triangles"].int().to(device).contiguous(),  # (nt, 3i)
            vertex_rgb=info_dict["vertex_colors"].float().to(device).contiguous()
            if info_dict["vertex_colors"] is not None
            else None,
            # (n, 3rgb) or None
            texture_rgb=info_dict["texture_map"].float().to(device).contiguous()
            if info_dict["texture_map"] is not None
            else None,
            # (h, w, 3rgb) or None
            vertex_uv=info_dict["vertex_uvs"].float().to(device).contiguous()
            if info_dict["vertex_uvs"] is not None
            else None,
            # (n, 2uv) or None
            vertex_normal_w=info_dict["vertex_normals"].float().to(device).contiguous()
            if info_dict["vertex_normals"] is not None
            else None,
            # (n, 3xyz) or None
            num_horizontal_texture_maps=info_dict["num_horizontal_texture_maps"],
        )
        return raw_mesh

    def replace_texture(self, texture_imgs: T.List[np.ndarray], replace_all: bool = False):
        """
        Replace the texture maps in the o3d mesh.

        Args:
            texture_imgs:
                a list of texture maps (may be in a different shape than the o3d textures)
            # method:
            #     'crop_resize': if a new texture is larger in dimension -> crop; if smaller -> resize.
        """

        texture_maps = self.mesh.textures
        num_textures = len(texture_maps)

        if len(texture_imgs) != num_textures:
            if not replace_all:
                warnings.warn(f"num of texture_imgs {len(texture_imgs)} != number of need {num_textures}")
                return
            else:
                texture_imgs = [texture_imgs] * num_textures

        new_textures = []
        for i in range(len(texture_maps)):
            new_textures.append(o3d.geometry.Image(texture_imgs[i]))
        self.mesh.textures = new_textures

    def get_rgbd_image(
        self,
        camera: Camera,
        render_normal_w: bool = True,
        device: torch.device = torch.device("cpu"),
        render_method: str = "ray_cast",
        camera_for_normal: T.Optional[Camera] = None,
        rasterize_light_on: bool = False,
    ) -> RGBDImage:
        """
        Given camera poses, return the captured RGBD images.

        Args:
            camera:
                (b, q)  already on device
            render_normal_w:
                whether to ray trace to get surface normal in world coordinate
            render_method:
                'rasterization': use o3d rasterization (may have anti-aliasing applied)
                'ray_cast': use ray_casting to sample rgb
            camera_for_normal:
                (b, q) camera for computing normal (in case intrinsic is negative at (2,2))
                used only when using rasterization.

        Returns:
            rgbdimage: (b, q)  on device
        """

        if render_method == "rasterization":
            return self._rasterize_rendering(
                camera=camera,
                render_normal_w=render_normal_w,
                device=device,
                camera_for_normal=camera_for_normal,
                light_on=rasterize_light_on,
            )

        elif render_method == "ray_cast":
            # run ray intersection to get normal
            ray = camera.generate_camera_rays(device=device)  # (b, q, h, w)
            out_dict = self.get_ray_intersection(
                ray=ray,
                device=device,
            )
            rgb = out_dict["ray_rgbs"]  # (b, q, h, w, 3)
            normal_w = out_dict["surface_normals_w"]  # (b, q, h, w, 3)
            hit_map = out_dict["hit_map"]  # (b, q, h, w)
            ray_ts = out_dict["ray_ts"]  # (b, q, h, w)

            # convert ray_ts to depth
            xyz_w = ray.origins_w + ray_ts.unsqueeze(-1) * ray.directions_w  # (b, q, h, w, 3)
            xyz1_w = torch.cat((xyz_w, torch.ones_like(xyz_w[..., 0:1])), dim=-1)  # (b, q, h, w, 4)
            H_w2c = camera.get_H_w2c()  # (b, q, 4, 4)
            xyz1_c = linalg_utils.matmul(
                H_w2c.unsqueeze(2).unsqueeze(2),
                xyz1_w.unsqueeze(-1),
            ).squeeze(-1)  # (b, q, h, w, 4)
            z_map = xyz1_c[..., 2]  # (b, q, h, w)

            valid_mask = torch.logical_and(hit_map > 0.5, z_map.isfinite())
            z_map[valid_mask.logical_not()] = INF

            # # # debug
            # xyz_w2 = utils.compute_3d_xyz(
            #     z_map=z_map,
            #     intrinsic=camera.intrinsic,
            #     H_c2w=camera.H_c2w,
            # )['xyz_w']  # (b, q, h, w, 3)
            # assert torch.allclose(xyz_w, xyz_w2)
            # # end debug

            return RGBDImage(
                rgb=rgb,
                depth=z_map,
                camera=camera,
                normal_w=normal_w,
                hit_map=hit_map,
            )
        else:
            raise NotImplementedError

    def _rasterize_rendering(
        self,
        camera: Camera,
        render_normal_w: bool = True,
        device: torch.device = torch.device("cpu"),
        camera_for_normal: T.Optional[Camera] = None,
        light_on: bool = False,
    ) -> RGBDImage:
        """
        Given camera poses, return the rasterized RGBD images.

        Args:
            camera:
                (b, q)  already on device
            render_normal_w:
                whether to ray trace to get surface normal in world coordinate
            render_method:
                'rasterization': use o3d rasterization (may have anti-aliasing applied)
                'ray_cast': use ray_casting to sample rgb
            camera_for_normal:
                (b, q) camera for computing normal (in case intrinsic is negative at (2,2))

        Returns:
            rgbdimage: (b, q)  on device
        """

        intrinsic = camera.intrinsic.detach().cpu().numpy()  # (b, q, 3, 3)
        H_c2w = camera.H_c2w.detach().cpu().numpy()  # (b, q, 4, 4)
        b, q = H_c2w.shape[0], H_c2w.shape[1]
        assert H_c2w.shape[2] == 4
        assert H_c2w.shape[3] == 4

        # convert (b, q) dimensino to list
        intrinsic_list = []
        H_c2w_list = []
        for i in range(b):
            for j in range(q):
                intrinsic_list.append(intrinsic[i, j])
                H_c2w_list.append(H_c2w[i, j])

        extrinsic_matrices = [rigid_motion.RigidMotion.invert_homogeneous_matrix(H) for H in H_c2w_list]

        out_dict = render.rasterize(
            meshes=[self.mesh],
            intrinsic_matrix=intrinsic_list,
            extrinsic_matrices=extrinsic_matrices,
            width_px=camera.width_px,
            height_px=camera.height_px,
            get_point_cloud=False,
            light_on=light_on,
            dtype=sample_utils.get_np_dtype(camera.H_c2w.dtype),
        )
        # out_dict contains
        #   imgs: a list of (h, w, 3)  rgb
        #   z_maps: a list of (h, w)  z of the scene points in the camera coordinate
        #   hit_maps: a list of (h, w)  true: valid
        # using imgs, cam_pose, intrinsics, and z_maps, we can generate point cloud

        imgs = out_dict["imgs"]  # list of (h, w, 3)
        z_maps = out_dict["z_maps"]  # list of (h, w)
        hit_maps = out_dict["hit_maps"]  # list of (h, w)

        # convert list of b*q back to (b,q)
        rgb = []
        depth = []
        hit_map = []
        current_idx = 0
        for i in range(b):
            tmp = np.stack(imgs[current_idx : current_idx + q], axis=0)  # (q, h, w, 3)
            rgb.append(tmp)
            tmp = np.stack(z_maps[current_idx : current_idx + q], axis=0)  # (q, h, w)
            depth.append(tmp)
            tmp = np.stack(hit_maps[current_idx : current_idx + q], axis=0)  # (q, h, w)
            hit_map.append(tmp)
            current_idx += q
        rgb = torch.from_numpy(np.stack(rgb, axis=0)).to(device=device)  # (b, q, h, w, 3)
        depth = torch.from_numpy(np.stack(depth, axis=0)).to(device=device)  # (b, q, h, w)
        hit_map = torch.from_numpy(np.stack(hit_map, axis=0)).to(device=device)  # (b, q, h, w)

        if render_normal_w:
            # run ray intersection to get normal
            if camera_for_normal is None:
                camera_for_normal = camera
            out_dict = self.get_ray_intersection(
                ray=camera_for_normal.generate_camera_rays(device=device),  # (b, q, h, w)
                device=device,
            )
            normal_w = out_dict["surface_normals_w"]  # (b, q, h, w, 3)
            hit_map = torch.logical_and(hit_map, out_dict["hit_map"])  # (b, q, h, w)
        else:
            normal_w = None

        return RGBDImage(
            rgb=rgb,  # (b, q, h, w, 3)
            depth=depth,  # (b, q, h, w)
            camera=camera,
            hit_map=hit_map,  # (b, q, h, w)
            normal_w=normal_w,  # (b, q, h, w, 3)
        )

    def get_ray_intersection(
        self,
        ray: Ray,
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
        if self.scene is None:
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
            self.scene = o3d.t.geometry.RaycastingScene()
            self.scene.add_triangles(mesh_t)
        raycast_results = self.scene.cast_rays(rays)
        t_hits = raycast_results["t_hit"].numpy()  # (b, *m), inf if not hit the mesh
        hit_map = 1 - np.isinf(t_hits)  # (b, *m)  1 if hit a surface, 0 otherwise

        # render rgb of the ray
        if self.mesh.has_textures():
            ray_rgbs = render.interp_texture_map_from_ray_tracing_results(
                mesh=self.mesh,
                raycast_results=raycast_results,
                texture_maps=[skimage.img_as_float(np.array(img)).astype(np.float32) for img in self.mesh.textures],
                merge_textures=True,  # combine results from multiple textures.
            )[0]
        else:
            ray_rgbs = np.ones((b, *m_shape, 3), dtype=np.float32)

        # note that primitive_normals is the normal of the triangle face
        # we can use uv map to interpolate vertex normal
        # interpolate surface normal using uv map to get better normal estimation
        if self.mesh.has_vertex_normals():
            surface_normals = render.interp_surface_normal_from_ray_tracing_results(
                mesh=self.mesh,
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
        surface_normals = surface_normals * (
            -1 * np.sign(np.sum(surface_normals * ray_directions, axis=-1, keepdims=True))
        )

        # convert to tensor
        ray_rgbs = torch.from_numpy(ray_rgbs).to(dtype=torch.float, device=device)  # (b, *m, 3)
        ray_ts = torch.from_numpy(t_hits).to(dtype=torch.float, device=device)  # (b, *m)
        surface_normals = torch.from_numpy(surface_normals).to(dtype=torch.float, device=device)  # (b, *m, 3)
        hit_map = torch.from_numpy(hit_map).to(
            dtype=torch.bool, device=device
        )  # (b, *m) 1 if hit a surface, 0 otherwise

        return dict(
            ray_rgbs=ray_rgbs,
            ray_ts=ray_ts,
            surface_normals_w=surface_normals,
            hit_map=hit_map,
        )

    def sample_point_cloud(
        self,
        num_points: int,
        method: str = "poisson_disk",
        device: torch.device = torch.device("cpu"),
        width_px: int = 10,
        height_px: int = 10,
        fov: float = 60.0,  # degree
        pinhole_min_r_ratio: float = 1.5,
        pinhole_max_r_ratio: float = 3,
        lookat_r_ratio: float = 0.25,
        dtype: np.dtype = np.float32,
    ) -> T.Dict[str, T.Any]:
        """
        Sample the mesh to create a point cloud
        Args:
            num_points:
                number of point to sample
            method:
                'uniform'
                'poisson_disk'
                'uniform_camera'
                'uniform_camera_random_look_at'
            pinhole_min_r_ratio:
                min radius to sample rgbd camera pinhole location. wrt to scale (radius of mesh)
            pinhole_min_r_ratio:
                max radius to sample rgbd camera pinhole location. wrt to scale (radius of mesh)
            lookat_r_ratio:
                radius to sample camera lookat coordinate. wrt to scale (radius of mesh)
        Returns:
            point_cloud: (1, num_points)
            rgbd_image: (1, num_img)
            camera: (1, num_img)

        """
        if dtype in {np.float32, float}:
            torch_dtype = torch.float32
        elif dtype == np.float64:
            torch_dtype = torch.float64
        else:
            raise NotImplementedError

        if method in ["poisson_disk", "uniform"]:
            if method == "poisson_disk":
                o3d_pcd = self.mesh.sample_points_poisson_disk(int(num_points))
            elif method == "uniform":
                o3d_pcd = self.mesh.sample_points_uniformly(int(num_points))
            else:
                raise NotImplementedError
            # create rays to get uv and texture -> color of points
            xyz_w = np.array(o3d_pcd.points, dtype=dtype)  # (n, 3)
            ray_ends = torch.from_numpy(xyz_w).to(dtype=torch_dtype, device=device).unsqueeze(0)  # (1, n, 3)
            ray_directions = torch.ones_like(ray_ends)  # (1, n, 3)
            ray_ts = torch.ones(1, ray_ends.size(1), dtype=torch_dtype, device=device) * 1.0e-5  # (1, n)
            ray_origins = ray_ends - ray_directions * ray_ts.unsqueeze(-1)
            ray = Ray(
                origins_w=ray_origins,
                directions_w=ray_directions,
            )
            out_dict = self.get_ray_intersection(
                ray=ray,
                device=device,
            )
            xyz_w = ray_origins + out_dict["ray_ts"].unsqueeze(-1) * ray_directions
            idxs = out_dict["hit_map"].squeeze(0) > 0.5  # (n,)

            point_cloud = PointCloud(
                xyz_w=xyz_w[:, idxs],  # (1, n, 3)
                rgb=out_dict["ray_rgbs"][:, idxs],  # (1, n, 3)
                normal_w=out_dict["surface_normals_w"][:, idxs],  # (1, n, 3)
            )
            rgbd_image = None
            camera = None

        elif method == "uniform_camera":
            # adjust resolution settings
            n_imgs = max(1, num_points // (width_px * height_px))
            n_pixels_per_img = num_points / n_imgs
            width_px = max(2, math.floor(n_pixels_per_img / (width_px * height_px) * width_px))
            width_px = max(2, width_px - (width_px % 2))
            height_px = max(2, math.floor(n_pixels_per_img / width_px))
            height_px = max(2, height_px - (height_px % 2))

            # get mesh scale and center
            cs = self.center_w
            s = self.scale

            # create uniformly placed camera
            camera = CameraTrajectory(
                mode="random",
                n_imgs=n_imgs,
                total=1,
                params=dict(
                    max_angle=180,
                    min_r=2 * s,
                    max_r=2 * s + 1.0e-9,
                    origin_w=cs.tolist(),
                    method="LatinHypercube",
                ),
                dtype=dtype,
            ).get_camera(
                fov=fov,
                width_px=width_px,
                height_px=height_px,
                device=device,
            )
            rgbd_image = self.get_rgbd_image(
                camera=camera,
                render_method="ray_cast",  # 'ray_cast',
                device=device,
            )
            point_cloud = rgbd_image.get_pcd()

        elif method == "uniform_camera_random_look_at":
            # adjust resolution settings
            n_imgs = max(1, num_points // (width_px * height_px))
            n_pixels_per_img = num_points / n_imgs
            width_px = max(2, math.floor(n_pixels_per_img / (width_px * height_px) * width_px))
            width_px = max(2, width_px - (width_px % 2))
            height_px = max(2, math.floor(n_pixels_per_img / width_px))
            height_px = max(2, height_px - (height_px % 2))

            # get mesh scale and center
            cs = self.center_w
            s = self.scale  # radius

            # create uniformly placed camera
            camera = CameraTrajectory(
                mode="random_lookat",
                n_imgs=n_imgs,
                total=1,
                params=dict(
                    pinhole_min_r=pinhole_min_r_ratio * s,
                    pinhole_max_r=pinhole_max_r_ratio * s,
                    lookat_r=lookat_r_ratio * s,
                ),
                dtype=dtype,
            ).get_camera(
                fov=fov,
                width_px=width_px,
                height_px=height_px,
                device=device,
            )
            rgbd_image = self.get_rgbd_image(
                camera=camera,
                render_method="ray_cast",  # 'ray_cast',
                device=device,
            )
            point_cloud = rgbd_image.get_pcd()

        else:
            raise NotImplementedError

        return dict(
            point_cloud=point_cloud,
            rgbd_image=rgbd_image,
            camera=camera,
        )


class RawMesh:
    def __init__(
        self,
        vertex_xyz_w: torch.Tensor,  # (n, 3)
        triangles: torch.Tensor,  # (num_triangles, 3)  long or int32 (preferred)
        vertex_rgb: T.Optional[torch.Tensor] = None,  # (n, 3) or (b, n, 3)
        vertex_feature: T.Optional[torch.Tensor] = None,  # (n, d) or (b, n, d)
        texture_rgb: T.Optional[torch.Tensor] = None,  # (ht, wt, 3) or (b, ht, wt, 3)
        texture_feature: T.Optional[torch.Tensor] = None,  # (ht, wt, d) or (b, ht, wt, d)
        vertex_uv: T.Optional[torch.Tensor] = None,  # (n, 2uv)  uv on the texture map
        vertex_normal_w: torch.Tensor = None,  # (n, 3)
        num_horizontal_texture_maps: int = 1,
    ):
        """

        Args:
            vertex_xyz_w:
                (n, 3xyz)  mesh vertices in the world coordinate
            triangles:
                (num_triangles, 3) long, the index of vertex_xyz_w to form each of the triangles
            vertex_normal_w:
                (n, 3xyz), surface normal at each vertex

            texture_rgb:
                (ht, wt, 3) [0, 1]
                Note that we assume the image is already flipped verically (along h).
                See: https://nvlabs.github.io/nvdiffrast/ (coordinate system).

                Open3d mesh already stores flipped image (nothing need to be done).
                Trimesh stores original image as PIL (so please flip it along h before passing).
            vertex_uv:
                (n, 2uv)  uv on the texture map.

                uv = (0, 0) is the top left corner of the texture map.  u to right, v to down.
                (we assume the texture map is flipped if uv is from trimesh/opengl convention).
                To convert from opengl convention without touching the value of uv, simply flipped the
                texture map along h.
        """

        self.vertex_xyz_w = vertex_xyz_w
        self.triangles = triangles.contiguous().int()
        self.vertex_normal_w = vertex_normal_w

        self.vertex_rgb = vertex_rgb
        self.vertex_feature = vertex_feature
        self.texture_rgb = texture_rgb
        self.texture_feature = texture_feature
        self.vertex_uv = vertex_uv
        self.num_horizontal_texture_maps = num_horizontal_texture_maps

    def box_normalize(self):
        """
        Box-normalize the mesh, so that its longest side has length = 2 (-1, 1),
        and the mesh is centered.
        """
        min_xyz_w, _ = self.vertex_xyz_w.min(dim=0)  # (3xyz,)
        max_xyz_w, _ = self.vertex_xyz_w.max(dim=0)  # (3xyz,)

        center_xyz_w = (min_xyz_w + max_xyz_w) * 0.5  # (3xyz,)
        max_length = (max_xyz_w - min_xyz_w).max()
        # shift to origin
        self.vertex_xyz_w = self.vertex_xyz_w - center_xyz_w
        self.vertex_xyz_w = self.vertex_xyz_w * (2 / max_length)

    def get_o3d_mesh(
        self,
        with_vertex_color: bool = False,
        with_texture: bool = False,
        with_vertex_normal_w: bool = False,
    ) -> o3d.geometry.TriangleMesh:
        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(self.vertex_xyz_w.detach().cpu().double().numpy()),
            triangles=o3d.utility.Vector3iVector(self.triangles.detach().cpu().int().numpy()),
        )
        if with_vertex_color and self.vertex_rgb is not None:
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(self.vertex_rgb.detach().cpu().double().numpy())
        if with_vertex_normal_w and self.vertex_normal_w is not None:
            o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(self.vertex_normal_w.detach().cpu().double().numpy())
        if with_texture and self.vertex_uv is not None:
            # compile triangle uv
            triangle_uvs = (
                self.vertex_uv[self.triangles.reshape(-1).long()].detach().cpu().double().numpy()
            )  # (num_triangle*3, 2uv)
            o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)

            # compile triangle_material_ids
            triangle_material_ids = torch.zeros(self.triangles.size(0), dtype=torch.int32).numpy()
            o3d_mesh.triangle_material_ids = o3d.utility.IntVector(triangle_material_ids)

            # compile textures
            if self.texture_rgb is not None:
                img = self.texture_rgb.detach().cpu().float().numpy()
                img = (img * 255).clip(0, 255).astype(np.uint8)
                o3d_mesh.textures = [o3d.geometry.Image(img)]

        return o3d_mesh

    def to(self, device: torch.device):
        if self.vertex_xyz_w is not None:
            self.vertex_xyz_w = self.vertex_xyz_w.to(dtype=torch.float, device=device).contiguous()
        if self.triangles is not None:
            self.triangles = self.triangles.to(dtype=torch.int, device=device).contiguous()
        if self.vertex_normal_w is not None:
            self.vertex_normal_w = self.vertex_normal_w.to(dtype=torch.float, device=device).contiguous()
        if self.vertex_rgb is not None:
            self.vertex_rgb = self.vertex_rgb.to(dtype=torch.float, device=device).contiguous()
        if self.vertex_feature is not None:
            self.vertex_feature = self.vertex_feature.to(dtype=torch.float, device=device).contiguous()
        if self.texture_rgb is not None:
            self.texture_rgb = self.texture_rgb.to(dtype=torch.float, device=device).contiguous()
        if self.texture_feature is not None:
            self.texture_feature = self.texture_feature.to(dtype=torch.float, device=device).contiguous()
        if self.vertex_uv is not None:
            self.vertex_uv = self.vertex_uv.to(dtype=torch.float, device=device).contiguous()
        return self

    @staticmethod
    def get_glctx(method: str = "opengl", device: torch.device = None):
        if method == "opengl":
            return dr.RasterizeGLContext(device=device)
        elif method == "cuda":
            return dr.RasterizeCudaContext(device=device)
        else:
            raise NotImplementedError

    @linalg_utils.disable_tf32_and_autocast()
    def _rasterize(
        self,
        camera: Camera,  # (b, q)
        t_min: float = 1.0e-4,
        t_max: float = None,
        glctx: "dr.RasterizeGLContext" = None,
        need_grad: bool = False,
    ):
        """
        Rasterize an image using nvdiffrast's rasterization.

        Args:
            camera:
                (b, q)
            glctx:
                the nvdiffrast context for the rendering

        Returns:

        """
        if glctx is None:
            glctx = self.get_glctx(device=camera.intrinsic.device)

        # we need to project vertex_xyz_w to the camera coordinate (after perspective projection, in clip space)
        proj_mtx = camera.get_perspective_projection_mtx(t_min=t_min, t_max=t_max, invert_z=True)  # (b, q, 4, 4)
        H_w2c = rigid_motion.inv_homogeneous_tensors(camera.H_c2w)  # (b, q, 4, 4)
        assert H_w2c.dtype == torch.float or H_w2c.dtype == torch.double
        mvp_mtx = linalg_utils.matmul(
            proj_mtx,
            H_w2c,
        )  # (b, q, 4, 4)
        assert mvp_mtx.dtype == torch.float or mvp_mtx.dtype == torch.double

        # project vertex to the clip space
        n, _3xyz = self.vertex_xyz_w.shape
        b, q, _41, _42 = mvp_mtx.shape
        xyzw = torch.cat(
            [
                self.vertex_xyz_w,
                torch.ones(n, 1, dtype=self.vertex_xyz_w.dtype, device=self.vertex_xyz_w.device),
            ],
            dim=-1,
        )  # (n, 4)
        vertex_xyz_c = linalg_utils.matmul(
            mvp_mtx.reshape(b, q, 1, 4, 4).float(),
            xyzw.reshape(1, 1, n, 4, 1).float(),
        ).squeeze(-1)  # (b, q, n, 4)  clip space
        vertex_xyz_c = vertex_xyz_c.reshape(b * q, n, 4).float().contiguous()  # (bq, n, 4)

        # print(self.vertex_xyz_w.max())
        # print(self.vertex_xyz_w.min())
        # print(vertex_xyz_c.max())
        # print(vertex_xyz_c.min())

        # vertex_xyz_c = vertex_xyz_c / vertex_xyz_c[..., -1:]  # (b, q, n, 4)
        # vertex_xyz_c = torch.clamp(vertex_xyz_c, min=-1, max=1)

        w = camera.width_px
        h = camera.height_px
        # The CUDA rasterizer does not support output resolutions greater than 2048×2048,
        # and both dimensions must be multiples of 8.
        # In addition, the number of triangles that can be rendered in one batch
        # is limited to around 16 million. Subpixel precision is limited to 4 bits
        # and depth peeling is less accurate than with OpenGL.
        # if not isinstance(glctx, dr.RasterizeGLContext):
        #     if w % 8 != 0:
        #         w = int((w // 8 + 1) * 8)
        #     if h % 8 != 0:
        #         h = int((h // 8 + 1) * 8)
        #     assert w <= 2048
        #     assert h <= 2048
        #     assert self.triangles.size(0) <= 16_000_000
        #     crop_wh = True
        # else:
        #     crop_wh = False
        crop_wh = False

        # rasterize
        with _nullcontext():
            rast_out, rast_db = dr.rasterize(
                glctx=glctx,
                pos=vertex_xyz_c.float().contiguous(),  # (bq, n, 4)
                tri=self.triangles.int().contiguous(),  # (n, 3)  int
                resolution=[h, w],
                ranges=None,
                grad_db=need_grad,
            )

        return dict(
            rast_out=rast_out,  # (bq, h, w, 4)  (u, v, z/w, triangle_id)
            rast_db=rast_db,
            crop_wh=crop_wh,
            glctx=glctx,
            vertex_xyz_c=vertex_xyz_c,  # (bq, n, 4)  clip space
        )

    @linalg_utils.disable_tf32_and_autocast()
    def render_via_vertex_interpolation(
        self,
        camera: Camera,  # (b, q)
        vertex_feature: torch.Tensor,  # (b, n, d) or (n, d)
        t_min: float = 1.0e-4,
        t_max: float = None,
        glctx: "dr.RasterizeGLContext" = None,
        need_grad: bool = False,
        antialias: bool = False,
        rast_dict: T.Dict[str, T.Any] = None,
        background_color: float = 0,
    ):
        """
        Rasterize and interpolate the feature at the intersection point
        from the vertex features using barycentric interpolation.

        Args:
            camera:
                (b, q)
            vertex_feature:
                (n, d) or (b, n, d)

        Returns:
            glctx
        """

        if glctx is None:
            glctx = self.get_glctx(device=vertex_feature.device)

        # rasterize
        if rast_dict is None:
            rast_dict = self._rasterize(
                camera=camera,
                t_min=t_min,
                t_max=t_max,
                glctx=glctx,
                need_grad=need_grad,
            )
        rast_out = rast_dict["rast_out"]

        b, q, _41, _42 = camera.H_c2w.shape
        n = vertex_feature.size(-2)
        d = vertex_feature.size(-1)
        assert n == self.vertex_xyz_w.size(0)
        if vertex_feature.ndim == 2:
            vertex_feature = vertex_feature.unsqueeze(0)  # (1, n, d)
        elif vertex_feature.ndim == 3:
            vertex_feature = vertex_feature.unsqueeze(1).expand(b, q, -1, -1).reshape(b * q, n, d)  # (bq, n, d)
        else:
            raise RuntimeError

        interp_out, interp_db = dr.interpolate(
            attr=vertex_feature,  # (bq, n, d) or (1, n, d)
            rast=rast_out,  # (bq, h, w, 4)
            tri=self.triangles.int().contiguous(),  # (n, 3)  int
            rast_db=rast_dict["rast_db"] if need_grad else None,
            diff_attrs="all" if need_grad else None,
        )
        # interp_out:  (bq, h, w, d)

        if antialias:
            interp_out = dr.antialias(
                color=interp_out,  # (bq, h, w, d)
                rast=rast_out,  # (bq, h, w, 4)
                pos=rast_dict["vertex_xyz_c"],  # (bq, n, 4)
                tri=self.triangles.int().contiguous(),  # (n, 3)  int
            )  # (bq, h, w, d)

        # rast_out[..., -1] is the triangle index offset by (0 means not hit)
        # bg_mask = torch.clamp(rast_out[..., -1:], min=0, max=1)  # (bq, h, w, 1)  float
        hit_map = rast_out[..., -1:] > 0.5  # (bq, h, w, 1)  bool
        interp_out = interp_out.masked_fill(~hit_map, background_color)

        return dict(
            rast_out=rast_out,  # (bq, h, w, 4)  (u, v, z/w, triangle_id)
            rast_db=rast_dict["rast_db"],
            crop_wh=rast_dict["crop_wh"],
            glctx=glctx,
            interp_out=interp_out,  # (bq, h, w, d)
            interp_db=interp_db,
            hit_map=hit_map.squeeze(-1),  # (bq, h, w) bool  # bg_mask.squeeze(-1) > 0.5,  # (bq, h, w)  bool
        )

    @staticmethod
    @linalg_utils.disable_tf32_and_autocast()
    def prepare_texture(
        texture: torch.Tensor,
        num_horizontal_texture_maps: int = 1,
    ) -> torch.Tensor:
        """
        prepare the texture map.

        Args:
            texture:
                (b, ht, wt, d) or (ht, wt, d)
            num_horizontal_texture_maps:
                number of texture maps concatenated along wt
        Returns:
            (b, ht', wt', d) or (ht', wt', d)

        See: https://nvlabs.github.io/nvdiffrast/#geometry-and-minibatches-range-mode-vs-instanced-mode
            "Mipmaps and texture dimensions"

        We assume the texture map only contains one single texture map (not concatenated of multiple texture maps)
        """
        if texture.ndim == 3:
            ht, wt, d = texture.shape
            texture = texture.unsqueeze(0)  # (1, ht, wt, d)
            squeeze_b = True
        elif texture.ndim == 4:
            b, ht, wt, d = texture.shape
            squeeze_b = False
        else:
            raise RuntimeError

        h2 = 2 ** round(math.log2(ht))
        if num_horizontal_texture_maps is None or num_horizontal_texture_maps == 1:
            w2 = 2 ** round(math.log2(wt))
        else:
            assert wt % num_horizontal_texture_maps == 0
            sub_wt = wt // num_horizontal_texture_maps
            sub_w2 = 2 ** round(math.log2(sub_wt))
            w2 = sub_w2 * num_horizontal_texture_maps

        if h2 != ht or w2 != wt:
            texture = torch.nn.functional.interpolate(
                texture.permute(0, 3, 1, 2),
                size=(h2, w2),
                mode="bilinear",
                align_corners=False,
            )  # (b, d, h', w')
            assert texture.size(-2) == h2
            assert texture.size(-1) == w2
            texture = texture.permute(0, 2, 3, 1)  # (b, h', w', d)

        if squeeze_b:
            texture = texture.squeeze(0)

        return texture.float().contiguous()

    @linalg_utils.disable_tf32_and_autocast()
    def precomute_mip_texture(self, texture: torch.Tensor):
        # make sure texture is of the right size
        texture = self.prepare_texture(texture)

        if texture.ndim == 3:
            ht, wt, d = texture.shape
            texture = texture.unsqueeze(0)  # (1, ht, wt, d)
        elif texture.ndim == 4:
            _b, ht, wt, d = texture.shape
            assert _b == 1, f"we have not implemented b not equal to 1 yet"
        else:
            raise RuntimeError

        mip_texture = dr.texture_construct_mip(
            tex=texture,
        )

        return dict(
            mip_texture=mip_texture,
            texture=texture,  # note that texture is still needed
        )

    @linalg_utils.disable_tf32_and_autocast()
    def render_via_texture_mapping(
        self,
        camera: Camera,  # (b, q)
        texture: torch.Tensor,  # (b, ht, wt, d) or (ht, wt, d)
        vertex_uv: torch.Tensor,  # (n, 2uv)  uv on the texture map
        t_min: float = 1.0e-4,
        t_max: float = None,
        enable_mip: bool = True,
        antialias: bool = False,
        mip_texture: T.List[torch.Tensor] = None,
        filter_mode: str = "auto",
        boundary_mode: str = "wrap",
        glctx: "dr.RasterizeGLContext" = None,
        need_grad: bool = False,
        rast_dict: T.Dict[str, T.Any] = None,
        background_color: float = 0,
    ):
        """
        Rasterize and interpolate the feature at the intersection point
        from the vertex features using barycentric interpolation.

        Args:
            camera:
                (b, q)
            texture:
                (b, ht, wt, d) or (ht, wt, d), float, texture map
            vertex_uv:
                (n, 2uv)  uv of each vertex on the texture map
            mip_texture:
                precomputed mip-texture map
            antialias:
                whether to apply antialiasing on the rendered image
            filter_mode:
                'auto', 'nearest', 'linear', 'linear-mipmap-nearest', and 'linear-mipmap-linear'
            boundary_mode:
                'wrap', 'clamp', 'zero'

        Returns:
            glctx
        """

        if enable_mip:
            need_grad = True

        if glctx is None:
            glctx = self.get_glctx(device=texture.device)

        # rasterize
        if rast_dict is None:
            rast_dict = self._rasterize(
                camera=camera,
                t_min=t_min,
                t_max=t_max,
                glctx=glctx,
                need_grad=need_grad,
            )
        rast_out = rast_dict["rast_out"]  # (bq, h, w, 4)

        n = self.vertex_xyz_w.size(0)
        b, q, _41, _42 = camera.H_c2w.shape

        # make sure texture is of the right size
        texture = self.prepare_texture(
            texture=texture,
            num_horizontal_texture_maps=self.num_horizontal_texture_maps,
        )

        if texture.ndim == 3:
            ht, wt, d = texture.shape
            texture = texture.unsqueeze(0)  # (1, ht, wt, d)
        elif texture.ndim == 4:
            _b, ht, wt, d = texture.shape
            assert b == _b
            texture = texture.unsqueeze(1).expand(b, q, ht, wt, d).reshape(b * q, ht, wt, d)  # (bq, ht, wt, d)
        else:
            raise RuntimeError

        with _nullcontext():
            pixel_uv, pixel_db = dr.interpolate(
                attr=vertex_uv.reshape(1, n, 2).float().contiguous(),  # (1, n, 2)
                rast=rast_out,  # (bq, h, w, 4)
                tri=self.triangles.int().contiguous(),  # (n, 3)  int
                rast_db=rast_dict["rast_db"] if need_grad else None,
                diff_attrs="all" if need_grad else None,
            )
            # pixel_uv:  (bq, h, w, 2uv)

        if enable_mip:
            if filter_mode == "nearest":
                filter_mode = "linear-mipmap-nearest"
            elif filter_mode == "linear":
                filter_mode = "linear-mipmap-linear"

            # print(f'pixel_db = {pixel_db}')
            # print(f'mip_texture = {mip_texture}')
            # print(f'filter_mode = {filter_mode}')

            assert wt % self.num_horizontal_texture_maps == 0
            sub_wt = wt // self.num_horizontal_texture_maps
            sub_ht = ht
            max_mip_level = 0
            while True:
                if sub_wt % 2 == 0 and sub_wt // 2 >= 1 and sub_ht % 2 == 0 and sub_ht // 2 >= 1:
                    max_mip_level += 1
                    sub_wt = sub_wt // 2
                    sub_ht = sub_ht // 2
                else:
                    break

            with _nullcontext():
                interp_out = dr.texture(
                    tex=texture,  # (1, ht, wt, d)
                    uv=pixel_uv.float(),  # (bq, h, w, 2uv)
                    uv_da=pixel_db,
                    mip=mip_texture,
                    filter_mode=filter_mode,
                    boundary_mode=boundary_mode,
                    max_mip_level=max_mip_level,
                )  # (bq, h, w, d)
        else:
            with _nullcontext():
                interp_out = dr.texture(
                    tex=texture,
                    uv=pixel_uv.float(),
                    filter_mode="linear",
                )  # (bq, h, w, d)

        # rast_out[..., -1] is the triangle index offset by (0 means not hit)
        # bg_mask = torch.clamp(rast_out[..., -1:], min=0, max=1)  # (bq, h, w, 1)  float
        # # this assumes black background
        # interp_out = interp_out * bg_mask + (1 - bg_mask) * background_color

        hit_map = rast_out[..., -1:] > 0.5  # (bq, h, w, 1)  bool
        alpha = hit_map.to(dtype=interp_out.dtype)  # (bq, h, w, 1)  float

        if not antialias:
            interp_out = interp_out * alpha + (1 - alpha) * background_color  # (bq, h, w, d)
            interp_out_straight = interp_out  # (bq, h, w, d)

        else:
            # premultiplied (important)
            interp_out = interp_out * alpha  # (bq, h, w, d)

            # antialias both the premultiplied color and the mask.
            interp_out = dr.antialias(
                color=interp_out,  # (bq, h, w, d)
                rast=rast_out,  # (bq, h, w, 4)
                pos=rast_dict["vertex_xyz_c"],  # (bq, n, 4)
                tri=self.triangles.int().contiguous(),  # (n, 3)  int
            )  # (bq, h, w, d)

            alpha = dr.antialias(
                color=alpha.float(),  # (bq, h, w, 1)
                rast=rast_out,  # (bq, h, w, 4)
                pos=rast_dict["vertex_xyz_c"],  # (bq, n, 4)
                tri=self.triangles.int().contiguous(),  # (n, 3)  int
            )  # (bq, h, w, 1)

            interp_out_straight = interp_out / alpha.clamp_min(1e-6)

            # Composite in premultiplied space: C = C_premul + (1 - A)*Bg
            interp_out = interp_out + (1.0 - alpha) * background_color  # (bq, h, w, d)

        return dict(
            rast_out=rast_out,  # (bq, h, w, 4)  (u, v, z/w, triangle_id)
            hit_map=hit_map.squeeze(-1),  # (bq, h, w) bool  # bg_mask.squeeze(-1) > 0.5,  # (bq, h, w)  bool
            alpha=alpha,  # (bq, h, w, 1) [0, 1]
            rast_db=rast_dict["rast_db"],
            crop_wh=rast_dict["crop_wh"],
            glctx=glctx,
            pixel_uv=pixel_uv,  # (bq, h, w, 2)
            pixel_db=pixel_db,
            interp_out=interp_out,  # (bq, h, w, d)  with or without antialiasing
            interp_out_straight=interp_out_straight,  # (bq, h, w, d)  without antialiasing nor bg masking
        )

    @linalg_utils.disable_tf32_and_autocast()
    def _get_z_w(
        self,
        z_ndc: torch.Tensor,
        t_min: float = 1.0e-4,
        t_max: float = None,
    ) -> torch.Tensor:
        """
        Given the z_ndc/w_ndc in the NDC, return the z_w in the world coordinate.

        Args:
            z_ndc:
                (*,)
            t_min:
            t_max:

        Formule:
        z_w = -1 / ((zn/wn * (t_min - t_max)) / (2 * t_min * t_max) + (1 * (t_min + t_max)) / (2 * t_min * t_max))

        Returns:
            z_w:
                (*,)

        Note:
            this assumes invert_z = True
        """
        assert z_ndc.dtype == torch.float or z_ndc.dtype == torch.double
        with _nullcontext():
            if t_max is None or t_max >= INF:
                z_w = (2 * t_min) / (1 - z_ndc)
            else:
                # z_w = 1 / ((z_ndc * (t_min - t_max)) / (2 * t_min * t_max) + (t_min + t_max) / (2 * t_min * t_max))
                z_w = (2 * t_min * t_max) / (t_min + t_max - z_ndc * (t_max - t_min))
            return z_w

    @linalg_utils.disable_tf32_and_autocast()
    def get_rgbd_image(
        self,
        camera: Camera,  # (b, q)
        render_depth: bool = True,
        render_normal_w: bool = True,
        t_min: float = 1.0e-2,
        t_max: float = None,
        enable_mip: bool = False,
        need_grad: bool = False,
        antialias: bool = False,
        render_feature: bool = False,
        force_render_with_texture: bool = True,
        glctx: "dr.RasterizeGLContext" = None,
        max_num_vertices_per_chunk: int = -1,
        background_color: float = 0.0,
    ) -> RGBDImage:
        assert NVDIFFRAST_LOADED

        if glctx is None:
            # use cuda context since opengl context is slow to initiate
            glctx = self.get_glctx(method="cuda", device=camera.H_c2w.device)

        b, q, _41, _42 = camera.H_c2w.shape
        total_vertices = b * q * self.vertex_xyz_w.size(0)

        if max_num_vertices_per_chunk < 0 or total_vertices <= max_num_vertices_per_chunk:
            return self._get_rgbd_image(
                camera=camera,
                render_depth=render_depth,
                render_normal_w=render_normal_w,
                t_min=t_min,
                t_max=t_max,
                enable_mip=enable_mip,
                need_grad=need_grad,
                antialias=antialias,
                render_feature=render_feature,
                force_render_with_texture=force_render_with_texture,
                glctx=glctx,
                background_color=background_color,
            )
        else:
            num_chunks = (total_vertices + max_num_vertices_per_chunk - 1) // max_num_vertices_per_chunk
            cameras = camera.chunk(chunks=num_chunks, dim=1)
            out_rgbds = []
            for cam_idx in range(len(cameras)):
                rgbd = self._get_rgbd_image(
                    camera=cameras[cam_idx],
                    render_depth=render_depth,
                    render_normal_w=render_normal_w,
                    t_min=t_min,
                    t_max=t_max,
                    enable_mip=enable_mip,
                    need_grad=need_grad,
                    antialias=antialias,
                    render_feature=render_feature,
                    force_render_with_texture=force_render_with_texture,
                    glctx=glctx,
                    background_color=background_color,
                )
                out_rgbds.append(rgbd)
            rgbd = RGBDImage.cat(out_rgbds, dim=1)
            return rgbd

    @linalg_utils.disable_tf32_and_autocast()
    def _get_rgbd_image(
        self,
        camera: Camera,  # (b, q)
        render_depth: bool = True,
        render_normal_w: bool = True,
        t_min: float = 1.0e-2,
        t_max: float = None,
        enable_mip: bool = False,
        need_grad: bool = False,
        antialias: bool = False,  # only has effect when rendering vertex properties
        render_feature: bool = False,
        force_render_with_texture: bool = True,
        glctx: "dr.RasterizeGLContext" = None,
        background_color: float = 0.0,
    ) -> RGBDImage:
        if glctx is None:
            glctx = self.get_glctx(device=camera.H_c2w.device)

        b, q, _41, _42 = camera.H_c2w.shape

        # rasterize
        rast_dict = self._rasterize(
            camera=camera,
            t_min=t_min,
            t_max=t_max,
            glctx=glctx,
            need_grad=need_grad,
        )

        # render depth
        if render_depth:
            z_ndc = rast_dict["rast_out"][..., 2]  # (bq, h, w)   z_ndc/w_ndc
            z_w = self._get_z_w(z_ndc=z_ndc, t_min=t_min, t_max=t_max)  # (bq, h, w)
            assert z_w.dtype == torch.float or z_w == torch.double
            if rast_dict["crop_wh"]:
                z_w = z_w[..., : camera.height_px, : camera.width_px]
            z_w = z_w.reshape(b, q, z_w.size(-2), z_w.size(-1))  # (b, q, h, w)
        else:
            z_w = None

        # render rgb
        # we prioritize vertex_rgb (since it is cheaper to render)
        if (not force_render_with_texture) and self.vertex_rgb is not None:
            out_dict = self.render_via_vertex_interpolation(
                camera=camera,
                vertex_feature=self.vertex_rgb,  # (n, 3)
                t_min=t_min,
                t_max=t_max,
                glctx=glctx,
                need_grad=need_grad,
                antialias=antialias,
                rast_dict=rast_dict,
                background_color=background_color,
            )
        else:
            assert self.texture_rgb is not None
            assert self.vertex_uv is not None
            out_dict = self.render_via_texture_mapping(
                camera=camera,
                texture=self.texture_rgb,
                vertex_uv=self.vertex_uv,
                t_min=t_min,
                t_max=t_max,
                enable_mip=enable_mip,
                antialias=antialias,
                mip_texture=None,
                filter_mode="auto",
                boundary_mode="wrap",
                glctx=glctx,
                need_grad=need_grad,
                rast_dict=rast_dict,
                background_color=background_color,
            )

        # print('out_dict:')
        # pprint(out_dict)
        #
        # print(f"out_dict['crop_wh'] = {out_dict['crop_wh']}")
        # print(f"out_dict['interp_out'] = {out_dict['interp_out'].shape}")
        # print(f"out_dict['hit_map'] = {out_dict['hit_map'].shape}")

        hit_map = out_dict["hit_map"]  # (bq, h, w)  bool

        if not antialias:
            rgb = out_dict["interp_out"]  # (bq, h, w, d)  with bg
            alpha = None
        else:
            rgb = out_dict["interp_out_straight"]  # (bq, h, w, d) without bg
            alpha = out_dict["alpha"]  # (bq, h, w, 1)

        if out_dict["crop_wh"]:
            rgb = rgb[..., : camera.height_px, : camera.width_px, :]
            hit_map = hit_map[..., : camera.height_px, : camera.width_px]
            if alpha is not None:
                alpha = alpha[..., : camera.height_px, : camera.width_px]

        rgb = rgb.reshape(b, q, rgb.size(-3), rgb.size(-2), rgb.size(-1))  # (b, q, h, w, 3)
        hit_map = hit_map.reshape(b, q, hit_map.size(-2), hit_map.size(-1))  # (b, q, h, w)
        if alpha is not None:
            alpha = alpha.reshape(b, q, alpha.size(-3), alpha.size(-2), alpha.size(-1))  # (b, q, h, w, 1)

        if z_w is not None:
            z_w = z_w.masked_fill(~hit_map, 0)  # non-hit z_w is 0

        # render feature
        if render_feature and self.vertex_feature is not None:
            out_dict = self.render_via_vertex_interpolation(
                camera=camera,
                vertex_feature=self.vertex_feature,  # (n, 3)
                t_min=t_min,
                t_max=t_max,
                glctx=glctx,
                need_grad=need_grad,
                antialias=antialias,
                rast_dict=rast_dict,
                background_color=background_color,
            )
        elif render_feature and self.texture_feature is not None:
            assert self.vertex_uv is not None
            out_dict = self.render_via_texture_mapping(
                camera=camera,
                texture=self.texture_feature,
                vertex_uv=self.vertex_uv,
                t_min=t_min,
                t_max=t_max,
                enable_mip=True,
                antialias=antialias,
                mip_texture=None,
                filter_mode="auto",
                boundary_mode="wrap",
                glctx=glctx,
                need_grad=need_grad,
                rast_dict=rast_dict,
                background_color=background_color,
            )
        else:
            out_dict = None

        if out_dict is not None:
            feature = out_dict["interp_out"]  # (bq, h, w, d)
            if out_dict["crop_wh"]:
                feature = feature[..., : camera.height_px, : camera.width_px, :]
            feature = feature.reshape(b, q, feature.size(-3), feature.size(-2), feature.size(-1))  # (b, q, h, w, 3)
        else:
            feature = None

        # render normal
        if render_normal_w:
            assert self.vertex_normal_w is not None
            out_dict = self.render_via_vertex_interpolation(
                camera=camera,
                vertex_feature=self.vertex_normal_w,  # (n, 3xyz)
                t_min=t_min,
                t_max=t_max,
                glctx=glctx,
                need_grad=need_grad,
                antialias=antialias,
                rast_dict=rast_dict,
                background_color=0,  # after normalize, stay as 0
            )
            normal_w = out_dict["interp_out"]  # (bq, h, w, 3xyz)
            if out_dict["crop_wh"]:
                normal_w = normal_w[..., : camera.height_px, : camera.width_px, :]
            normal_w = normal_w.reshape(
                b, q, normal_w.size(-3), normal_w.size(-2), normal_w.size(-1)
            )  # (b, q, h, w, 3)
            normal_w = torch.nn.functional.normalize(normal_w, p=2, dim=-1)
        else:
            normal_w = None

        return RGBDImage(
            rgb=rgb,  # (b, q, h, w, 3) without bg
            depth=z_w,  # (b, q, h, w) or None
            camera=camera,
            normal_w=normal_w,  # (b, q, h, w, 3) or None
            hit_map=hit_map,  # (b, q, h, w) bool
            feature=feature,  # (b, q, h, w, d)
            other_maps=None if alpha is None else dict(alpha=alpha),  # (b, q, h, w, 1) or None
        )

    @linalg_utils.disable_tf32_and_autocast()
    def comput_face_normals(self):
        """
        Compute individual faces' normal by cross product
        """
        i0 = self.triangles[..., 0].long()  # (num_triangles, )
        i1 = self.triangles[..., 1].long()  # (num_triangles, )
        i2 = self.triangles[..., 2].long()  # (num_triangles, )

        v0 = self.vertex_xyz_w[i0, :]  # (num_triangles, 3xyz_w)
        v1 = self.vertex_xyz_w[i1, :]  # (num_triangles, 3xyz_w)
        v2 = self.vertex_xyz_w[i2, :]  # (num_triangles, 3xyz_w)
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (num_triangles, 3xyz_w)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)  # (num_triangles, 3xyz_w)
        return face_normals  # (num_triangles, 3xyz_w)

    @linalg_utils.disable_tf32_and_autocast()
    def render(
        self,
        camera: Camera,  # (b, q)
        return_types: T.List[
            T.Literal[
                "vertex_rgb",
                "mask",
                "z_c",
                "vertex_normal",
                "face_normal",
            ]
        ],
        t_min: float = 1.0e-2,
        t_max: float = None,
        glctx: "dr.RasterizeGLContext" = None,
        normalize_vertex_normal: bool = True,
        normalize_face_normal: bool = True,
        max_img_chunk_size: int = -1,
    ) -> T.Dict[str, torch.Tensor]:
        """Render vertex properties."""
        assert NVDIFFRAST_LOADED

        if glctx is None:
            glctx = self.get_glctx(device=camera.H_c2w.device)

        b, q, _41, _42 = camera.H_c2w.shape

        if max_img_chunk_size < 0 or b * q < max_img_chunk_size:
            return self._render(
                camera=camera,
                return_types=return_types,
                t_min=t_min,
                t_max=t_max,
                glctx=glctx,
                normalize_vertex_normal=normalize_vertex_normal,
                normalize_face_normal=normalize_face_normal,
            )
        else:
            assert b == 1
            num_chunks = (q + max_img_chunk_size - 1) // max_img_chunk_size
            cameras = camera.chunk(chunks=num_chunks, dim=1)
            out_dicts = []
            for cam_idx in range(len(cameras)):
                out_dict = self._render(
                    camera=cameras[cam_idx],
                    return_types=return_types,
                    t_min=t_min,
                    t_max=t_max,
                    glctx=glctx,
                    normalize_vertex_normal=normalize_vertex_normal,
                    normalize_face_normal=normalize_face_normal,
                )
                out_dicts.append(out_dict)
            out_dict = utils.cat_dict(out_dicts, dim_dict=1)
            return out_dict

    @linalg_utils.disable_tf32_and_autocast()
    def _render(
        self,
        camera: Camera,  # (b, q)
        return_types: T.List[
            T.Literal[
                "vertex_rgb",
                "mask",
                "z_c",
                "vertex_normal",
                "face_normal",
            ]
        ],
        t_min: float = 1.0e-2,
        t_max: float = None,
        glctx: "dr.RasterizeGLContext" = None,
        normalize_vertex_normal: bool = True,
        normalize_face_normal: bool = True,
    ) -> T.Dict[str, torch.Tensor]:
        """
        Render vertex properties

        Args:
            camera:
                (b, q)

        Returns:
            (b, q, h, w, d)
        """
        if glctx is None:
            glctx = self.get_glctx(device=camera.H_c2w.device)

        b, q, _41, _42 = camera.H_c2w.shape

        if self.vertex_xyz_w.size(0) == 0 or self.triangles.size(0) == 0:
            raise NotImplementedError

        # rasterize
        rast_dict = self._rasterize(
            camera=camera,
            t_min=t_min,
            t_max=t_max,
            glctx=glctx,
            need_grad=False,
        )

        rast = rast_dict["rast_out"]  # (bq, h, w, 4)  (u, v, z/w, triangle_id)
        vertices_clip = rast_dict["vertex_xyz_c"]  # (bq, n, 4) clip space
        crop_wh = rast_dict["crop_wh"]
        faces_int = self.triangles.int().contiguous()

        H_w2c = rigid_motion.inv_homogeneous_tensors(camera.H_c2w)  # (b, q, 4, 4)
        xyz1_w = torch.cat(
            [
                self.vertex_xyz_w,  # (n, 3xyz_w)
                torch.ones_like(self.vertex_xyz_w[..., :1]),  # (n, 1)
            ],
            dim=-1,
        )  # (n, 4)
        # vertices_z_c = linalg_utils.matmul(
        #     H_w2c[..., 2, :].reshape(b * q, 1, 1, 4),  # (bq, 1, 1, 4)
        #     xyz1_w.unsqueeze(-1),  # (n, 4, 1)
        # ).squeeze(-1)  # (bq, n, 1)
        vertices_xyzw_c = linalg_utils.matmul(
            H_w2c.reshape(b * q, 1, 4, 4),  # (bq, 1, 4, 4)
            xyz1_w.unsqueeze(-1),  # (n, 4, 1)
        ).squeeze(-1)  # (bq, n, 4)
        vertices_xyz_c = vertices_xyzw_c[..., :3] / (vertices_xyzw_c[..., 3:] + 1e-8)  # (bq, n, 3)
        vertices_z_c = vertices_xyz_c[..., 2:3]  # (bq, n, 1)

        out_dict = dict()
        for type in return_types:
            img = None
            if type == "mask":
                img = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
            elif type == "z_c":
                img = dr.interpolate(
                    vertices_z_c.contiguous(),  # (bq, n, 1)
                    rast,  # (bq, h, w, 4)
                    faces_int,  # (num_tri, 3)
                )[0]  # (bq, h, w, 1)
                # img = dr.antialias(
                #     img,  # (bq, h, w, 1)
                #     rast,  # (bq, h, w, 4)
                #     vertices_clip,  # (bq, n, 4)
                #     faces_int,
                # )  # (bq, h, w, 1)
            elif type == "xyz_c":
                # https://github.com/nv-tlabs/FlexiCubes/blob/4cc7d6c3d0cee83c011ce36721b81adff0dd7db6/examples/render.py#L106
                img = dr.interpolate(
                    vertices_xyz_c.contiguous(),  # (bq, n, 3)
                    rast,  # (bq, h, w, 4)
                    faces_int,  # (num_tri, 3)
                )[0]  # (bq, h, w, 1)
                # img = dr.antialias(
                #     img,  # (bq, h, w, 1)
                #     rast,  # (bq, h, w, 4)
                #     vertices_clip,  # (bq, n, 4)
                #     faces_int,
                # )  # (bq, h, w, 1)
            elif type == "face_normal":
                # compute face normal
                face_normal = self.comput_face_normals()  # (num_triangles, 3xyz_w)
                face_normal = face_normal[:, None, :].repeat(1, 3, 1)  # (num_triangles, 3, 3xyz_w)
                img = dr.interpolate(
                    face_normal.reshape(1, -1, 3).expand(b * q, -1, -1).contiguous(),  # (bq, num_triangles*3, 3xyz_w)
                    rast,  # (bq, h, w, 4)
                    torch.arange(self.triangles.shape[0] * 3, device=rast.device, dtype=torch.int).reshape(
                        -1, 3
                    ),  # (num_triangles*3, 3)
                )[0]  # (bq, h, w, 3xyz_w)
                img = dr.antialias(
                    img,
                    rast,
                    vertices_clip,
                    faces_int,
                )  # (bq, h, w, 3xyz_w)
                if normalize_face_normal:
                    img = torch.nn.functional.normalize(img, dim=-1)
            elif type == "vertex_normal":
                img = dr.interpolate(
                    self.vertex_normal_w.contiguous(),
                    rast,
                    faces_int,
                )[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
                if normalize_vertex_normal:
                    img = torch.nn.functional.normalize(img, dim=-1)
            elif type == "vertex_rgb":
                img = dr.interpolate(self.vertex_rgb.contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            else:
                raise NotImplementedError

            if crop_wh:
                img = img[..., : camera.height_px, : camera.width_px, :]

            out_dict[type] = img.reshape(b, q, camera.height_px, camera.width_px, img.size(-1))

        return out_dict

    @linalg_utils.disable_tf32_and_autocast()
    def get_p3d_mesh(self) -> pytorch3d.structures.Meshes:
        """
        Construct pytorch3d mesh structure.

        Returns:
            pytorch3d meshes of batch size 1.
            If texture map is None, a gray texture map will be used.
            If vertex uv is None, they will all be set to (u=0, v=0)
        """

        device = self.vertex_xyz_w.device
        if self.texture_rgb is None:
            texture_rgb = torch.ones(512, 512, 3, device=device) * 0.5  # (h, w, 3) [0, 1]
        else:
            texture_rgb = self.texture_rgb  # (h, w, 3) [0, 1] flipped

        if self.vertex_uv is None:
            vertex_uv = torch.zeros(self.vertex_xyz_w.size(0), 2, device=device)  # (n, 2uv)
        else:
            vertex_uv = self.vertex_uv  # (n, 2uv)

        # preprocess texture map (eg, make the texture dimension power of 2)
        img = self.prepare_texture(
            texture=texture_rgb.to(device=device)  # (h, w, 3) [0, 1] flipped
        )  # (h, w, 3) [0, 1] not flipped

        # create p3d_mesh
        tex = pytorch3d.renderer.TexturesUV(
            maps=[torch.flip(img, dims=[0])],  # list of (h, w, 3)  not flipped
            verts_uvs=[vertex_uv],  # list of (n, 2)
            faces_uvs=[self.triangles],  # list of (f, 3)
        )
        p3d_mesh = pytorch3d.structures.Meshes(
            verts=[self.vertex_xyz_w],
            faces=[self.triangles],
            textures=tex,
        )
        return p3d_mesh

    @linalg_utils.disable_tf32_and_autocast()
    def sample_points(
        self,
        num_points: int,
        method: str,
        compute_normal_if_needed: bool,
    ) -> "PointCloud":
        """
        Sample points uniformly on surface of the mesh.

        Args:
            num_points:
                number of points to sample
            method:
                'pytorch3d'
                'open3d'
            compute_normal_if_needed:
                If using open3d to sample points, compute vertex normal
                if not exist in the mesh.
                When using pytorch3d, it returns the face normal of the triangle
                the points are sampled from.

        Returns:
            xyz_w:
                (b=1, num_points, 3xyz_w)
            rgb:
                (b=1, num_points, 3rgb) [0, 1] or None
            normal_w:
                (b=1, num_points, 3xyz_w) or None
            uv:
                (b=1, num_points, 2uv) or None.
                Saved in the `feature` attribute
                Only available if using pytorch3d.

        """

        if method == "pytorch3d":
            p3d_mesh = self.get_p3d_mesh()  # batch_size = 1
            p3d_mesh.faces_areas_packed()
            out_dict = mesh_utils.sample_points_from_p3d_meshes(
                meshes=p3d_mesh,
                num_samples=num_points,
                return_normals=True,
                return_textures=True,
                return_uvs=True,
            )
            point_xyz_w = out_dict["xyz_w"]  # (b=1, n, 3)
            point_normal_w = out_dict["normal_w"]  # (b=1, n, 3)
            point_rgb = out_dict["textures"]  # (b=1, n, 3rgb)  [0, 1]
            point_uv = out_dict["uv"]  # (b=1, n, 2uv)  [0, 1]
            return PointCloud(
                xyz_w=point_xyz_w,  # (b=1, n, 3)
                normal_w=point_normal_w,  # (b=1, n, 3)
                rgb=point_rgb,  # (b=1, n, 3)
                feature=point_uv,  # (b=1, n, 2)
            )
        elif method == "open3d":
            o3d_mesh = self.get_o3d_mesh(
                with_vertex_color=True,
                with_texture=True,
            )
            if compute_normal_if_needed and not o3d_mesh.has_vertex_normals():
                o3d_mesh.compute_vertex_normals()

            o3d_pcd: o3d.geometry.PointCloud = o3d_mesh.sample_points_uniformly(
                number_of_points=num_points,
            )
            point_xyz_w = torch.tensor(
                np.asarray(o3d_pcd.points),
                dtype=self.vertex_xyz_w.dtype,
                device=self.vertex_xyz_w.device,
            ).unsqueeze(0)  # (b=1, n, 3)
            if o3d_pcd.has_normals():
                point_normal_w = torch.tensor(
                    np.asarray(o3d_pcd.normals),
                    dtype=self.vertex_xyz_w.dtype,
                    device=self.vertex_xyz_w.device,
                ).unsqueeze(0)  # (b=1, n, 3)
            else:
                point_normal_w = None
            if o3d_pcd.has_colors():
                point_rgb = torch.tensor(
                    np.asarray(o3d_pcd.colors),
                    dtype=self.vertex_xyz_w.dtype,
                    device=self.vertex_xyz_w.device,
                ).unsqueeze(0)  # (b=1, n, 3)
            else:
                point_rgb = None

            return PointCloud(
                xyz_w=point_xyz_w,  # (b=1, n, 3)
                normal_w=point_normal_w,  # (b=1, n, 3)
                rgb=point_rgb,  # (b=1, n, 3)
            )
        else:
            raise NotImplementedError(method)


class CameraTrajectory:
    """
    CameraTrajectory is a pattern of camera poses
    """

    def __init__(
        self,
        mode: str,
        n_imgs: int,
        total: int,
        rng_seed: T.Union[np.random.RandomState, int] = 0,
        params: T.Dict[str, T.Any] = None,
        dtype: np.dtype = np.float32,
    ):
        """
        Args:
            mode:

            n_imgs:
                number of cameras in a set
            total:
                total number of sets
            rng_seed:
                random seed
            params:
                parameters for the mode
        """
        self.mode = mode
        self.n_imgs = n_imgs
        self.total = total
        self.np_dtype = sample_utils.get_np_dtype(dtype)
        self.torch_dtype = sample_utils.get_torch_dtype(dtype)

        if rng_seed is not None:
            if isinstance(rng_seed, int):
                self.rng = np.random.RandomState(seed=rng_seed)
            elif isinstance(rng_seed, np.random.RandomState):
                self.rng = rng_seed
            else:
                self.rng = rng_seed
        else:
            self.rng = np.random

        if params is None:
            params = dict()

        self.params = params

        if self.mode == "assign":
            assert self.params.get("H_c2w", None) is not None
            H_c2w = self.params["H_c2w"]
            if H_c2w.ndim == 3:
                self.n_imgs = H_c2w.size(0)
                self.cam_poses = H_c2w
            elif H_c2w.ndim == 4:
                self.n_imgs = H_c2w.size(1)
                self.total = H_c2w.size(0)
                self.cam_poses = H_c2w
            else:
                raise NotImplementedError
        elif self.mode == "random":
            # within random camera in a random cone
            self._set_random()
        elif self.mode == "random_lookat":
            # within random camera in a random cone
            self._set_random_lookat()
        elif self.mode == "circle":
            self._set_circle()
        elif self.mode == "udlrfb":
            self._set_udlrfb()
        elif self.mode == "spiral":
            self._set_spiral()
        elif self.mode == "sketchfab_poisson":
            raise NotImplementedError
        elif self.mode == "rex_in":
            raise NotImplementedError
        elif self.mode == "rect":
            raise NotImplementedError
        elif self.mode == "basic":
            raise NotImplementedError
        elif self.mode == "grid":
            raise NotImplementedError
        elif self.mode == "polar_grid":
            raise NotImplementedError
        elif self.mode == "manual":
            self._set_manual()
        else:
            # filename of the camera pose pt file
            camera = Camera(H_c2w=None, intrinsic=None, width_px=None, height_px=None)
            checkpoint = torch.load(self.mode, map_location=torch.device("cpu"))
            camera.load_state_dict(checkpoint)
            # uniformly sample the path
            camera = camera.uniformly_sample(num_samples=self.n_imgs)
            H_c2w = camera.H_c2w  # (b, q, 4, 4)
            self.cam_poses = H_c2w

    def _set_random(self):
        assert "max_angle" in self.params
        assert "min_r" in self.params
        assert "max_r" in self.params

        self.cam_poses: T.List[T.List[np.ndarray]] = [
            rigid_motion.generate_random_camera_poses(
                n=self.n_imgs,
                max_angle=self.params.get("max_angle"),
                min_r=self.params.get("min_r"),
                max_r=self.params.get("max_r"),
                center_direction_w=self.params.get("center_direction_w", None),
                local_max_angle=self.params.get("local_max_angle", 0),
                rand_r=self.params.get("rand_r", 0),
                origin_w=self.params.get("origin_w", None),
                rng=self.rng,
                method=self.params.get("method", "random"),
                dtype=self.np_dtype,
            )
            for _ in range(self.total)
        ]  # list of list of H_c2w
        self.cam_poses = utils.to_dtype(
            utils.to_tensor(self.cam_poses),
            dtype=self.torch_dtype,
        )

    def _set_random_lookat(self):
        assert "pinhole_min_r" in self.params
        assert "pinhole_max_r" in self.params
        assert "lookat_r" in self.params

        self.cam_poses = rigid_motion.generate_random_camera_poses_lookat(
            n=self.n_imgs * self.total,
            pinhole_min_r=self.params.get("pinhole_min_r"),
            pinhole_max_r=self.params.get("pinhole_max_r"),
            lookat_r=self.params.get("lookat_r", 0),
            invert_y=True,
        )  # (n * total, 4, 4)
        self.cam_poses = self.cam_poses.reshape(self.total, self.n_imgs, 4, 4)  # (total, n, 4, 4)

    def _set_circle(self):
        self.cam_poses = []  # (b, q)
        for i in range(self.total):
            poses = []
            center_angles = self.params.get("center_angles", None)  # (2,) in degree,  (to_x, to_z)
            if center_angles is None:
                # determine random center direction
                center_angles = self.rng.rand(2) * 360.0  # angle in degree

            # determine random d for input_imgs
            d = self.params.get("d", None)  # (,)
            if d is None:
                assert "min_r" in self.params
                assert "max_r" in self.params
                max_r = self.params["max_r"]
                min_r = self.params["min_r"]
                d = self.rng.rand(1) * (max_r - min_r) + min_r

            # determine circle r for input_imgs
            r = self.params.get("r", None)  # (,)
            if r is None:
                assert "max_angle" in self.params
                max_angle = self.params["max_angle"]
                r = self.rng.rand(1) * np.tan(max_angle * np.pi / 180.0) * d

            # generate input camera path
            Hs_c2w = utils.generate_camera_circle_path(
                num_poses=self.n_imgs,
                d_to_origin=d,
                r_circle=r,
                center_angles=center_angles,
            )  # (n, 4, 4)
            for j in range(Hs_c2w.size(0)):
                poses.append(Hs_c2w[j])  # list of H_c2w
            self.cam_poses.append(poses)  # list of list of H_c2w

    def _set_udlrfb(self):
        # fixed 6 input views: up down left right front back
        assert "min_r" in self.params
        assert "max_r" in self.params
        max_r = self.params["max_r"]
        min_r = self.params["min_r"]

        assert self.n_imgs == 6
        self.cam_poses = []  # (total, n_imgs, 4, 4)
        for i in range(self.total):
            r = self.rng.rand(1) * (max_r - min_r) + min_r
            poses = []

            Hs_c2w_ud = utils.generate_camera_circle_path(
                num_poses=3,
                d_to_origin=0,
                r_circle=r,
                center_angles=[0, 0],
                alt_yaxis=True,
            )  # (n, 4, 4)
            Hs_c2w_lrfb = utils.generate_camera_circle_path(
                num_poses=5,
                d_to_origin=0,
                r_circle=r,
                center_angles=[0, 90],
                alt_yaxis=True,
            )  # (n, 4, 4)
            poses.append(Hs_c2w_ud[0])  # u
            for j in range(Hs_c2w_lrfb.size(0) - 1):  # lfrb
                poses.append(Hs_c2w_lrfb[j])
            poses.append(Hs_c2w_ud[1])  # d
            self.cam_poses.append(poses)

    def _set_spiral(self):
        assert "min_r" in self.params
        assert "max_r" in self.params
        max_r = self.params["max_r"]
        min_r = self.params["min_r"]
        num_circle = self.params.get("num_circle", 4)
        r_freq = self.params.get("r_freq", 1)

        self.cam_poses = []
        for i in range(self.total):
            Hs_c2w_spiral = utils.generate_camera_spiral_path(
                num_poses=self.n_imgs,
                num_circle=num_circle,
                init_phi=np.pi / 2,
                r_min=min_r,
                r_max=max_r,
                r_freq=r_freq,
                center_angles=[-90, 0],
            )

            poses = []
            for j in range(Hs_c2w_spiral.size(0)):
                poses.append(Hs_c2w_spiral[j])
            self.cam_poses.append(poses)

    def _set_sketchfab_poisson(self):
        self.cam_poses = []

        for i in range(self.total):
            r = 2.2  # fixed, ignore input arguments regarding r
            poses = []
            target_poses_neighbors = []
            # for target camera
            Hs_c2w_lrfb = utils.generate_camera_circle_path(
                num_poses=self.n_imgs,
                d_to_origin=0,
                r_circle=r,
                center_angles=[0, -90],
                alt_yaxis=True,
            )  # (n, 4, 4)
            for j in range(Hs_c2w_lrfb.size(0) - 1):  # lfrb
                poses.append(Hs_c2w_lrfb[j])
            self.cam_poses.append(poses)

    def _set_rex_in(self):
        # assumption:
        # input is from 14 input cameras.
        # mesh size (-4, 4)
        assert "max_r" in self.params
        max_r = self.params["max_r"]

        self.cam_poses = []
        for i in range(self.total):
            # r = max_r
            poses = []
            target_poses_neighbors = []

            # sample input from grid points on polar axis
            # details are in self.cam_path_mode == 'polar_grid':

            num_phi = int(np.ceil((np.sqrt(2 * self.n_imgs - 3) + 3.0) / 2.0))
            num_theta = 2 * (num_phi - 1)
            total_imgs = num_theta * (num_phi - 2) + 2

            Hs_c2w, neighbor_ids = utils.generate_camera_polar_grids(
                num_phi=num_phi, num_theta=num_theta, r=max_r
            )  # (n, 4, 4)

            for j in range(self.n_imgs):
                poses.append(Hs_c2w[j])

            self.cam_poses.append(poses)

    def _set_rect(self):
        assert self.total == 1
        for i in range(self.total):
            poses = []

            # note that the center can be adjusted here
            # camera will face the center
            Hs_c2w_rect = utils.generate_camera_rect_path(
                num_poses=self.n_imgs,
                d_to_origin=0,
                x_length=7,
                y_length=7,
                x_center=0,
                y_center=2,
                center_angles=[-90, 0],
                alt_yaxis=True,
            )

            for j in range(Hs_c2w_rect.size(0)):
                poses.append(Hs_c2w_rect[j])

            self.cam_poses.append(poses)

    def _set_basic(self):
        assert "min_r" in self.params
        assert "max_r" in self.params
        max_r = self.params["max_r"]
        min_r = self.params["min_r"]

        # both input and target are around the same circle path
        self.cam_poses = []
        for i in range(self.total):
            poses = []

            Hs_c2w_input = utils.generate_camera_circle_path(
                num_poses=self.n_imgs,
                d_to_origin=-max_r / 2,
                r_circle=max_r,
                center_angles=[-90, 0],
                alt_yaxis=True,
            )  # (n, 4, 4)

            for j in range(Hs_c2w_input.size(0) - 1):  # lfrb
                poses.append(Hs_c2w_input[j])
            self.cam_poses.append(poses)

    def _set_grid(self):
        # build camera grid around one point
        # currently only used in one case in inverse rendering

        assert "min_r" in self.params
        assert "max_r" in self.params
        max_r = self.params["max_r"]
        min_r = self.params["min_r"]

        grid_width = int(np.ceil(np.sqrt(self.n_imgs)))
        total_imgs = grid_width * grid_width

        self.cam_poses = []
        for i in range(self.total):
            poses = []

            Hs_c2w = utils.generate_camera_grids(
                num_x=grid_width,
                num_y=grid_width,
                cam_position_center=np.array([-max_r, 0, 0]),
            )  # (n, 4, 4)

            for j in np.random.permutation(total_imgs)[range(self.n_imgs)]:
                poses.append(Hs_c2w[j])

            self.cam_poses.append(poses)

    def _set_polar_grid(self):
        # both input and target are around the same circle path
        # sample num_theta points from [0,2*pi]
        # sample num_phi points from [0,pi], including two polars (0,pi)
        # assume sample theta and phi contain same number of angles on circle, 2*(num_phi-1) = num_theta
        # total point: num_theta* (num_phi-2) +2 = 2*num_phi^2 -6*num_phi +6
        # to make total point> self.n_target_imgs, num_phi >= (sqrt(2*self.n_imgs-3)+3)/2
        # common choice: self.n_target_imgs = 6, 14, 26, 42, ...
        # corresponding to num_phi = 3, 4, 5, 6,....
        assert "min_r" in self.params
        assert "max_r" in self.params
        max_r = self.params["max_r"]
        min_r = self.params["min_r"]

        assert self.n_imgs >= 3, "polar grid sample at least 3 points"

        num_phi = int(np.ceil((np.sqrt(2 * self.n_imgs - 3) + 3.0) / 2.0))
        num_theta = 2 * (num_phi - 1)
        total_imgs = num_theta * (num_phi - 2) + 2

        self.cam_poses = []
        # cam_poses_neighbors: used in inverse rendering
        # neighbor camera positions to be updated
        # note that self.cam_poses_neighbors is different from self.target_cam_poses_neighbors
        # the former are used in inverse rendering, the later are used in multivew geometry
        # but their concept are the same
        self.cam_poses_neighbors = []

        for i in range(self.total):
            poses = []

            Hs_c2w, neighbor_ids = utils.generate_camera_polar_grids(
                num_phi=num_phi, num_theta=num_theta, r=max_r
            )  # (n, 4, 4)

            for j in range(self.n_imgs):
                poses.append(Hs_c2w[j])

            self.cam_poses.append(poses)

    def _set_manual(self):
        """
        Manually assign camera
            eye: list of (3,) where the cameras are (before global transform)
            up:  None (assume to be (0,1,0)), (1,3) used for all cameras, or list of (3,)
            look_at:  None (assume to be (0,0,0)), (1, 3) used for all cameras, or list of (3,)
            t_c2w:  (3,) or None
            y_c2w:  (3,) or None
            z_c2w:  (3,) or None
        """
        assert "eye" in self.params
        eyes = self.params["eye"]
        eyes = [[float(i) for i in eye.split(" ")] for eye in eyes]
        eyes = torch.tensor(eyes).float().reshape(-1, 3)  # (q, 3)
        assert self.n_imgs == eyes.size(0)

        ups = self.params.get("up", None)
        if ups is None:
            ups = [0, 1.0, 0]
        else:
            ups = [[float(i) for i in x.split(" ")] for x in ups]
        ups = torch.tensor(ups).float().reshape(-1, 3)  # (q, 3)
        if ups.size(0) == 1:
            ups = ups.expand_as(eyes)  # (q, 3)

        look_ats = self.params.get("look_at", None)
        if look_ats is None:
            look_ats = [0, 0.0, 0]
        else:
            look_ats = [[float(i) for i in x.split(" ")] for x in look_ats]
        look_ats = torch.tensor(look_ats).float().reshape(-1, 3)  # (q, 3)
        if look_ats.size(0) == 1:
            look_ats = look_ats.expand_as(eyes)  # (q, 3)

        t_c2w = self.params.get("t_c2w", None)
        if t_c2w is None:
            t_c2w = torch.zeros(3)  # (3,)
        else:
            t_c2w = [float(i) for i in t_c2w.split(" ")]
            t_c2w = torch.tensor(t_c2w).float()
        y_c2w = self.params.get("y_c2w", None)
        if y_c2w is None:
            y_c2w = torch.tensor([0, 1, 0]).float()  # (3,)
        else:
            y_c2w = [float(i) for i in y_c2w.split(" ")]
            y_c2w = torch.tensor(y_c2w).float()
        z_c2w = self.params.get("z_c2w", None)
        if z_c2w is None:
            z_c2w = torch.tensor([0, 0, 1]).float()  # (3,)
        else:
            z_c2w = [float(i) for i in z_c2w.split(" ")]
            z_c2w = torch.tensor(z_c2w).float()
        R_c2w = rigid_motion.construct_coord_frame(
            z=z_c2w,
            y=y_c2w,
        )
        H_c2w_global = torch.zeros(4, 4)
        H_c2w_global[:3, :3] = R_c2w
        H_c2w_global[:3, 3] = t_c2w
        H_c2w_global[3, 3] = 1

        self.cam_poses = []  # (total, q)
        for i in range(self.total):
            H_c2ws = rigid_motion.get_H_c2w_lookat(
                pinhole_location_w=eyes,  # (q, 3)
                look_at_w=look_ats,  # (q, 3)
                up_w=ups,  # (q, 3)
                invert_y=True,
            )  # (q, 4, 4)

            H_c2ws = linalg_utils.matmul(
                H_c2w_global.unsqueeze(0),
                H_c2ws,
            )  # (q, 4, 4)
            self.cam_poses.append(H_c2ws)

    @staticmethod
    def get_spiral_trajectory(
        H_c2w: torch.Tensor,
        period: int,
        radius: float,
    ) -> "CameraTrajectory":
        """
        Given a trajectory of camera poses, create a trajectory that is
        a spiral near the trajectory.

        The function only moves the camera center. It does not change the
        rotation matrix.

        Args:
            H_c2w:
                (b, q, 4, 4), q >= 2
            period:
                number of cam_poses (q) to finish a full circle
            radius:
                how large the circles are

        Returns:
            a trajectory (containing self.cam_poses (b, q', 4, 4))
        """

        b, q, _41, _42 = H_c2w.shape
        assert q >= 2

        # figure out z direction
        cs = H_c2w[:, :-1, :3, 3]  # (b, q-1, 3)
        cs_next = H_c2w[:, 1:, :3, 3]  # (b, q-1, 3)
        delta_zs = cs_next - cs  # (b, q-1, 3)
        delta_zs = torch.cat([delta_zs, delta_zs[:, -1:]], dim=-2)  # (b, q, 3)
        dzs = torch.nn.functional.normalize(delta_zs, p=2, dim=-1)  # (b, q, 3)

        # decide y and x direction
        dys = torch.zeros_like(dzs)  # (b, q, 3)
        dys[..., 1] = 1  # (b, q, 3)
        coord_frames = rigid_motion.construct_coord_frame(
            z=dzs,  # (b, q, 3)
            y=dys,
        )  # (b, q, 3, 3)
        dxs = coord_frames[..., 0]  # (b, q, 3)
        dys = coord_frames[..., 1]  # (b, q, 3)

        # create circle shift
        thetas = torch.linspace(start=0.0, end=2 * torch.pi, steps=period)  # (period,)
        xs = torch.cos(thetas) * radius  # (period, )
        ys = torch.sin(thetas) * radius  # (period, )

        xs = xs.repeat((q + period - 1) // period)[:q]  # (q, )
        ys = ys.repeat((q + period - 1) // period)[:q]  # (q, )

        shift = dxs * xs.view(1, q, 1) + dys * ys.view(1, q, 1)  # (b, q, 3)

        new_H_c2w = H_c2w.clone()  # (b, q, 4, 4)
        new_H_c2w[:, :, :3, 3] = new_H_c2w[:, :, :3, 3] + shift

        return CameraTrajectory(
            mode="assign",
            n_imgs=None,
            total=None,
            params=dict(H_c2w=new_H_c2w),
        )

    def get_camera(
        self,
        fov: T.Union[float, torch.Tensor],  # in degree,  float, (q,), (b, q)
        width_px: int,
        height_px: int,
        device: torch.device = torch.device("cpu"),
    ) -> Camera:
        """
        Returns cameras in the trajactory
        """

        intrinsics = render.derive_camera_intrinsics(
            width_px=width_px,
            height_px=height_px,
            fov=fov,
            dtype=self.np_dtype,
        )  # (*, 3, 3) np.ndarray
        intrinsics = torch.from_numpy(intrinsics).to(device=device)  # (*, 3, 3)

        if isinstance(self.cam_poses, (list, tuple)):
            H_c2w = []
            for i in range(len(self.cam_poses)):
                poses = [pose for pose in self.cam_poses[i]]
                H = torch.stack(poses, dim=0)  # (n_img, 4, 4)
                H_c2w.append(H)
            H_c2w = torch.stack(H_c2w, dim=0).to(device=device)  # (total, n_img, 4, 4)
        elif isinstance(self.cam_poses, torch.Tensor):
            if self.cam_poses.ndim == 3:
                H_c2w = self.cam_poses.unsqueeze(0)  # (1, q, 4, 4)
            elif self.cam_poses.ndim == 2:
                H_c2w = self.cam_poses.view(1, 1, 4, 4)  # (1, 1, 4, 4)
            else:
                assert self.cam_poses.ndim == 4
                H_c2w = self.cam_poses
        elif isinstance(self.cam_poses, np.ndarray):
            self.cam_poses = torch.tensor(self.cam_poses, dtype=self.torch_dtype)
            if self.cam_poses.ndim == 3:
                H_c2w = self.cam_poses.unsqueeze(0)  # (1, q, 4, 4)
            elif self.cam_poses.ndim == 2:
                H_c2w = self.cam_poses.view(1, 1, 4, 4)  # (1, 1, 4, 4)
            else:
                assert self.cam_poses.ndim == 4
                H_c2w = self.cam_poses
        else:
            raise NotImplementedError

        *b_shape, _, _ = H_c2w.shape

        return Camera(
            H_c2w=H_c2w,
            intrinsic=intrinsics.expand(*b_shape, 3, 3),
            width_px=width_px,
            height_px=height_px,
        )


class ColorCorrector(torch.nn.Module):
    def __init__(
        self,
        correction_type: str = "wrgb",
    ):
        """
        Apply the color correction to an rgbd_image

        Args:
            correction_type:
                'wrgb': the correction is 3 scalars \in [0, 1] that multiply to RGB channels separately
                'identify': do nothing
        """
        super().__init__()
        self.correction_type = correction_type
        if self.correction_type == "wrgb":
            self.wrgb = torch.nn.parameter.Parameter(torch.ones(3))
        elif self.correction_type == "identify":
            self.register_buffer("wrgb", torch.ones(3))
        else:
            raise NotImplementedError

    def get_extra_state(self):
        return dict(
            correction_type=self.correction_type,
        )

    def set_extra_state(self, state):
        self.correction_type = state["correction_type"]

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        apply the color correction
        Args:
            x:
                (*, 3)
        Returns:
            (*, 3) corrected x
        """
        if self.correction_type == "wrgb":
            y = x * self.wrgb.reshape(*([1] * (x.ndim - 1)), -1)
            return y
        elif self.correction_type == "identify":
            return x
        else:
            raise NotImplementedError


class PatchIdxGenerator:
    def __init__(
        self,
        q: int,
        h: int,
        w: int,
        batch_size: int,
        unlimited: bool,
        drop_last: bool,
        patch_width_px: int,
        patch_height_px: int = None,
        seed: int = 0,
    ):
        """
        Generate the top left pixel indexes of random patches.
        We shuffle all the patches so that every pixels will be used.

        Args:
            q, h, w:
                size of rgbd_image
            bacth_size:
                number of patches to return
            unlimited:
                whether we will keep generating patches
            drop_last:
                whether to drop last
            patch_width_px:
                number of pixels in the patch in width
            patch_height_px:
                if None, the same as `patch_width_px`
            required_attributes:
                list of str containing the field wanted. If None: all possible fields
        """
        self.q = q
        self.h = h
        self.w = w
        self.batch_size = batch_size
        self.unlimited = unlimited
        self.drop_last = drop_last
        self.patch_width_px = patch_width_px
        self.patch_height_px = patch_height_px if patch_height_px is not None else patch_width_px
        assert self.w >= self.patch_width_px
        assert self.h >= self.patch_height_px
        self.rng = np.random.default_rng(seed)

        self.h_left_over_idx = 0
        self.w_left_over_idx = 0

        # get all top_left_patch linear indexes (last few cols/rows will be dropped)
        # qidxs = np.arange(start=0, stop=self.q)  # (q,)
        # cidxs = np.arange(start=0, stop=self.w - self.patch_width_px + 1, step=self.patch_width_px)  # (c,)
        # ridxs = np.arange(start=0, stop=self.h - self.patch_height_px + 1, step=self.patch_height_px)  # (r,)
        # Qs, Rs, Cs = np.meshgrid(qidxs, ridxs, cidxs, indexing='ij')  # (q, r, c)
        # qrcs = np.stack([Qs, Rs, Cs], axis=-1)  # (q, r, c, 3qrc)
        # self.qrcs = qrcs.reshape(-1, 3)  # (q * r * c, 3qrc)
        self._create_new_qrcs()
        self.current_idx = 0
        # self.idxs = torch.from_numpy(self.rng.permutation(self.qrcs.shape[0]))

    def _create_new_qrcs(self):
        h_left = self.h % self.patch_height_px
        w_left = self.w % self.patch_width_px

        h_start = self.h_left_over_idx
        self.h_left_over_idx = (self.h_left_over_idx + 1) % (h_left + 1)
        w_start = self.w_left_over_idx
        self.w_left_over_idx = (self.w_left_over_idx + 1) % (w_left + 1)

        qidxs = np.arange(start=0, stop=self.q)  # (q,)
        cidxs = np.arange(start=w_start, stop=self.w - self.patch_width_px + 1, step=self.patch_width_px)  # (c,)
        ridxs = np.arange(start=h_start, stop=self.h - self.patch_height_px + 1, step=self.patch_height_px)  # (r,)
        Qs, Rs, Cs = np.meshgrid(qidxs, ridxs, cidxs, indexing="ij")  # (q, r, c)
        qrcs = np.stack([Qs, Rs, Cs], axis=-1)  # (q, r, c, 3qrc)
        self.qrcs = torch.from_numpy(qrcs).reshape(-1, 3)  # (q * r * c, 3qrc)
        self.idxs = torch.from_numpy(self.rng.permutation(self.qrcs.shape[0]))

    def __iter__(self):
        self._create_new_qrcs()
        # self.idxs = torch.from_numpy(self.rng.permutation(self.qrcs.shape[0]))
        self.current_idx = 0
        return self

    def __len__(self):
        if self.unlimited:
            return int(1e10)
        elif self.drop_last:
            return self.qrcs.shape[0] // self.batch_size
        else:
            return (self.qrcs.shape[0] + self.batch_size - 1) // self.batch_size

    def __next__(self):
        from_idx = self.current_idx
        to_idx = self.current_idx + self.batch_size

        if to_idx <= self.idxs.shape[0]:
            iis = self.idxs[from_idx:to_idx]  # (b,)
            self.current_idx = to_idx
            qrc = self.qrcs[iis]
            return qrc  # (b, 3qrc)
        else:
            if self.unlimited:
                iis = self.idxs[from_idx:to_idx]  # (b,)
                rest = self.batch_size - len(iis)
                qrc = self.qrcs[iis]

                # create new qrc and idx (next epoch)
                self._create_new_qrcs()
                # self.idxs = torch.from_numpy(self.rng.permutation(self.qrcs.shape[0]))
                new_to_idx = min(rest, len(self.idxs))
                jjs = self.idxs[:new_to_idx]
                self.current_idx = new_to_idx

                new_qrc = self.qrcs[jjs]
                qrc = torch.cat([qrc, new_qrc], dim=0)
                return qrc  # (b, 3qrc)
            else:
                if self.drop_last:
                    raise StopIteration
                elif from_idx >= self.idxs.shape[0]:
                    raise StopIteration
                else:
                    iis = self.idxs[from_idx:to_idx]  # (b,)
                    self.current_idx = to_idx
                    qrc = self.qrcs[iis]
                    return qrc  # (b, 3qrc)
