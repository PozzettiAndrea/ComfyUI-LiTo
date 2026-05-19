#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements helper functions to use gaussian splatting.
from __future__ import annotations  # type annotations stay as strings, never evaluated

import math
import os
import traceback
import typing as T

import numpy as np
from plyfile import PlyData, PlyElement

import torch

# Optional heavy imports — only used in helper paths the ComfyUI inference
# wrapper does not exercise (SPZ compression, pytorch3d-based mesh ops).
try:
    import spz
except ImportError:
    spz = None
try:
    import pytorch3d.ops
    import pytorch3d.transforms
except ImportError:
    pytorch3d = None

from plibs import linalg_utils, rigid_motion, sh_utils
from contextlib import nullcontext as _nullcontext

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except:
    try:
        from third_party.diff_gaussian_rasterization.diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
    except:
        GaussianRasterizationSettings = None
        GaussianRasterizer = None


@linalg_utils.disable_tf32_and_autocast()
def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


@linalg_utils.disable_tf32_and_autocast()
def getProjectionMatrix_general(
    focal_x: T.Union[float, torch.Tensor],
    focal_y: T.Union[float, torch.Tensor],
    width: int,
    height: int,
    znear: float,
    zfar: float,
    cx: T.Union[float, torch.Tensor] = 0,
    cy: T.Union[float, torch.Tensor] = 0,
):
    """
    Convert camera intrinsics and return the opengl projection matrix
    that converts the world coordinate to normalized device coordinate
    (normalize +znear to z=0 and +zfar to z=1).

    Args:
        znear:
            min z_c
        zfar:
            max z_c
        focal_x:
            focal length in pixel (intrinsic[0, 0])
        focal_y:
            focal length in pixel (intrinsic[1, 1])
        cx:
            principle point on sensor (intrinsic[0, 2]) in pixel
        cy:
            principle point on sensor (intrinsic[1, 2]) in pixel
        width:
            number of pixels on the sensor horizontally
        height:
            number of pixels on the sensor vertically

    Returns:
        (4, 4) projection matrix that converts camera coordinate
        to NDC where znear is mapped to 0 and zfar is mapped to 1
    """
    if isinstance(focal_x, torch.Tensor):
        dtype = focal_x.dtype
        device = focal_x.device
    else:
        dtype = torch.float
        device = torch.device("cpu")

    tanHalfFovX = 0.5 * width / focal_x
    tanHalfFovY = 0.5 * height / focal_y

    # the origin at center of image plane
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    # shift the frame window due to the non-zero principle point offsets
    offset_x = cx - (width / 2)
    offset_x = (offset_x / focal_x) * znear
    offset_y = cy - (height / 2)
    offset_y = (offset_y / focal_y) * znear

    top = top + offset_y
    left = left + offset_x
    right = right + offset_x
    bottom = bottom + offset_y

    P = torch.zeros(4, 4, dtype=dtype, device=device)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


@linalg_utils.disable_tf32_and_autocast()
def fov2focal(fov, pixels):
    # given fov (in radians) and pixel width, return focal length (in px)
    if isinstance(fov, torch.Tensor):
        return pixels / (2 * torch.tan(fov / 2))
    elif isinstance(fov, (int, float)):
        return pixels / (2 * math.tan(fov / 2))
    else:
        raise NotImplementedError


@linalg_utils.disable_tf32_and_autocast()
def focal2fov(focal, pixels):
    # given focal length (in px) and pixel width, return fov (in radians)
    if isinstance(focal, torch.Tensor):
        return 2 * torch.atan(pixels / (2 * focal))
    elif isinstance(focal, (int, float)):
        return 2 * math.atan(pixels / (2 * focal))
    else:
        raise NotImplementedError


@linalg_utils.disable_tf32_and_autocast()
def build_rotation(quaternion: torch.Tensor, normalized: bool = False):
    """
    convert unit quaternion to rotation matrix, corresponding to R_g2w

    Args:
        quaternion: (*n, 4xyzw) quaternion
        normalized: whether quaternion is already normalized

    Returns:
        (*n, 3, 3)
    """
    *n_shape, _4 = quaternion.shape
    if normalized:
        q = quaternion
    else:
        q = torch.nn.functional.normalize(quaternion, dim=-1)  # (*n, 4)
    R = torch.zeros((*n_shape, 3, 3), dtype=quaternion.dtype, device=quaternion.device)

    r = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - r * z)
    R[..., 0, 2] = 2 * (x * z + r * y)
    R[..., 1, 0] = 2 * (x * y + r * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - r * x)
    R[..., 2, 0] = 2 * (x * z - r * y)
    R[..., 2, 1] = 2 * (y * z + r * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


@linalg_utils.disable_tf32_and_autocast()
def rotation_matrix_to_quaternion(R_g2w: torch.Tensor):
    """
    Convert rotation matrix to unit quaternion.

    Args:
        R_g2w:
            (*, 3, 3)

    Returns:
        (*, 4) normalized
    """
    return pytorch3d.transforms.matrix_to_quaternion(
        matrix=R_g2w,
    )


@linalg_utils.disable_tf32_and_autocast()
def get_cov3d(
    quaternion: torch.Tensor,
    scaling: torch.Tensor,
):
    """
    Returns the 3d covariance matrix (n, 3xyz, 3xyz) of the gaussians.

    Args:
        quaternion:
            (*n, 4xyzw) quaternion, corresponding to R_g2w
        scaling:
            (*n, 3xyz) std of gaussians

    """
    R_g2w = build_rotation(quaternion)  # (*n, 3xyz, 3xyz)
    R_g2w = scaling.unsqueeze(-1) * R_g2w  # (*n, 3, 3)
    cov = R_g2w.transpose(-1, -2) @ R_g2w  # (*n, 3, 3)
    return cov


def convert_ply_to_spz(
    ply_filename: str,
    spz_filename: str,
):
    """
    Convert 3dgs saved as ply to spz.  This is lossy compression.

    Args:
        ply_filename:
            ply saving the original gaussians
        spz_filename:
            spz saving the compressed gaussians
    """
    import spz

    # Load the PLY file
    unpack_options = spz.UnpackOptions()
    cloud = spz.load_splat_from_ply(ply_filename, unpack_options)

    # Save as compressed SPZ format
    pack_options = spz.PackOptions()
    success = spz.save_spz(cloud, pack_options, spz_filename)
    assert success


class Gaussians:
    def __init__(
        self,
        sh_degree: int,
        xyz_w: torch.Tensor,  # (*b, n, 3xyz_w)
        rgb_sh: T.Optional[torch.Tensor],  # (*b, n, (sh+1)**2, 3rgb)
        rgb_sh_dc: T.Optional[torch.Tensor],  # (*b, n, 1, 3rgb)
        rgb_sh_rest: T.Optional[torch.Tensor],  # (*b, n, (sh+1)**2-1, 3rgb)
        scaling_logit: T.Optional[torch.Tensor],  # (*b, n, 3xyz) logit
        quaternion_prenorm: T.Optional[torch.Tensor],  # (*b, n, 4)
        opacity_logit: T.Optional[torch.Tensor],  # (*b, n, 1)  logit
        scaling: T.Optional[torch.Tensor] = None,  # (*b, n, 3xyz)
        quaternion: T.Optional[torch.Tensor] = None,  # (*b, n, 4xyzw)
        opacity: T.Optional[torch.Tensor] = None,  # (*b, n, 1)
        min_scaling: float = 0,
        scaling_activation_type: str = "exp",
    ):
        """
        Data structure to store 3d gaussians.

        Ref: https://github.com/graphdeco-inria/gaussian-splatting/

        Args:
            sh_degree:
                degree of spherical harmonics
            xyz_w:
                (*b, n, 3xyz_w)  gaussian center/mean in the world coordinate
            rgb_sh:
                (*b, n, (sh+1)**2, 3rgb), spherical harmonic coefficients for rgb.
                If given, rgb_sh_dc and rgb_sh_rest will be ignored.
            rgb_sh_dc:
                (*b, n, 1, 3rgb) 0th-degree sh coefficient of rgb color of each gaussian
            rgb_sh_rest:
                (*b, n, (sh+1)**2-1, 3rgb), higher degree spherical harmonic coefficients for rgb
            scaling_logit:
                (*b, n, 3xyz) the logit (before exp()) of the std of the gaussians
            quaternion_prenorm:
                (*b, n, 4xyzw), the quaternion before normalizing to unit norm to represent the rotation R_g2w
            opacity_logit:
                (*b, n, 1) the logit (before sigmoid()) of the opacity of the gaussians
            min_scaling:
                float, the minimum std of the gaussians. It will be added to scaling if given (see code).
            scaling_activation_type:
                'exp', 'softplus'
        """
        self.sh_degree = sh_degree
        self.xyz_w = xyz_w  # (*b, n, 3)
        self.min_scaling = min_scaling
        self.scaling_activation_type = scaling_activation_type

        self.rgb_sh = rgb_sh  # (*b, n, (sh+1)**2, 3rgb) or None
        self._rgb_sh_dc = rgb_sh_dc  # (*b, n, 1, 3) or None
        self._rgb_sh_rest = rgb_sh_rest  # (*b, n, (sh+1)**2-1, 3rgb) or None

        if self.scaling_activation_type == "exp":
            self.scaling_activation = torch.exp
            self.inverse_scaling_activation = torch.log
        elif self.scaling_activation_type == "softplus":
            self.scaling_activation = torch.nn.functional.softplus
            self.inverse_scaling_activation = lambda x: x + torch.log(-torch.expm1(-x))
        elif self.scaling_activation_type == "none":
            self.scaling_activation = None
            self.inverse_scaling_activation = None
        else:
            raise NotImplementedError

        if scaling is None:
            assert scaling_logit is not None
            self.scaling = self.scaling_activation(scaling_logit)  # (*b, n, 3)
        else:
            self.scaling = scaling

        if self.min_scaling > 1e-8:
            self.scaling = (self.scaling**2 + self.min_scaling**2).sqrt()

        if quaternion is None:
            assert quaternion_prenorm is not None
            self.quaternion = torch.nn.functional.normalize(quaternion_prenorm, dim=-1)  # (*b, n, 4)
        else:
            self.quaternion = quaternion

        if opacity is None:
            self.opacity = torch.sigmoid(opacity_logit)  # (*b, n, 1)
        else:
            self.opacity = opacity

    def check(self):
        assert 0 <= self.sh_degree <= 3, f"{self.sh_degree=}"
        assert self.xyz_w.is_cuda
        if self.rgb_sh is not None:
            assert self.rgb_sh.is_cuda
        if self.rgb_sh_dc is not None:
            assert self.rgb_sh_dc.is_cuda
        if self.rgb_sh_rest is not None:
            assert self.rgb_sh_rest.is_cuda
        assert self.scaling.is_cuda
        assert self.quaternion.is_cuda
        assert self.opacity.is_cuda

    @property
    def bshape(self):
        """return *b"""
        return self.xyz_w.shape[:-2]  # if no b_shape, returns [] (len=0)

    @property
    def get_rgb_sh(self):
        if self.rgb_sh is not None:
            shs = self.rgb_sh
        else:
            if self.rgb_sh_rest is not None:
                shs = torch.cat([self.rgb_sh_dc, self.rgb_sh_rest], dim=-2)  # (*b, n, (sh+1)**2, 3rgb)
            else:
                shs = self.rgb_sh_dc
        assert shs.size(-2) >= (self.sh_degree + 1) ** 2
        return shs

    @property
    def rgb_sh_dc(self):
        if self.rgb_sh is None:
            return self._rgb_sh_dc
        else:
            return self.rgb_sh[..., :1, :]  # (*b, n, 1, 3rgb)

    @property
    def rgb_sh_rest(self):
        if self.rgb_sh is None:
            return self._rgb_sh_rest
        else:
            return self.rgb_sh[..., 1:, :]  # (*b, n, (sh+1)**2-1, 3rgb)

    def get_R_g2w(self):
        """
        Return R_g2w  (*b, n, 3, 3)
        """
        return self.build_rotation(self.quaternion)  # (*b, n, 3, 3)

    def get_z_w(self):
        """
        Return the z axis of the gaussian in the world coordinate, ie R_g2w[..., 2].
        (n, 3)
        """
        return self.build_z_w(self.quaternion)  # (n, 3)

    def get_cov3d(self):
        """
        Returns the 3d covariance matrix (*b, n, 3xyz, 3xyz) of the gaussians.
        """
        R_g2w = self.get_R_g2w()  # (*b, n, 3xyz, 3xyz)
        R_g2w = self.scaling.unsqueeze(-1) * R_g2w  # (*b, n, 3, 3)
        cov = R_g2w.transpose(-1, -2) @ R_g2w  # (*b, n, 3, 3)
        return cov

    @staticmethod
    def build_rotation(r: torch.Tensor):
        """
        convert unit quaternion to rotation matrix, corresponing to R_g2w

        Args:
            r: (*n, 4xyzw) quaternion

        Returns:
            (*n, 3, 3)
        """
        return build_rotation(quaternion=r)

    @staticmethod
    def build_z_w(r):
        """
        given unit quaternion, return the third column in the rotation matrix.

        Args:
            r: (*b, n, 4xyzw)

        Returns:
            (*b, n, 3, 3)
        """
        q = torch.nn.functional.normalize(r, dim=-1)  # (*b, n, 4)
        z_w = torch.zeros(*q.shape[:-1], 3, dtype=q.dtype, device=q.device)

        r = q[..., 0]
        x = q[..., 1]
        y = q[..., 2]
        z = q[..., 3]

        z_w[..., 0] = 2 * (x * z + r * y)
        z_w[..., 1] = 2 * (y * z - r * x)
        z_w[..., 2] = 1 - 2 * (x * x + y * y)
        return z_w

    def get_smallest_principal_component(self):
        """
        Return the axis direction that corresponds to the smallest std, ie R_g2w[..., min_idx].
        (*b, n, 3)
        """
        R_g2w = self.get_R_g2w()  # (*b, n, 3, 3)
        stds = self.scaling  # (*b, n, 3)
        min_idx = torch.argmin(stds, dim=-1)  # (*b, n)
        # batch_indices = torch.arange(R_g2w.size(0), device=R_g2w.device)  # (n,)
        # out = R_g2w[batch_indices, :, min_idx]  # (n, 3)

        ndim_b = R_g2w.ndim - 3
        out = torch.gather(
            input=R_g2w,  # (*b, n, 3, 3)
            dim=-1,  # column of R_g2w
            index=min_idx.unsqueeze(-1).unsqueeze(-1).expand(*([-1] * (ndim_b + 1)), 3, 1),  # (*b, n, 3, 1)
        ).squeeze(-1)  # (*b, n, 3)
        return out

    def construct_list_of_attributes(self):
        """Create attribute list for ply files."""
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self.rgb_sh_dc.shape[1] * self.rgb_sh_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        if self.rgb_sh_rest is not None:
            for i in range(3 * ((self.sh_degree + 1) ** 2 - 1)):
                l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self.scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self.quaternion.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, filename: str):
        """
        Save the gaussians as a specialized ply file.
        """
        assert len(self.bshape) == 0
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        xyz_w = self.xyz_w.detach().cpu().float().numpy()  # (n, 3)
        normal_ws = np.zeros_like(xyz_w)  # (n, 3)
        f_dc = (
            self.rgb_sh_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().float().numpy()
        )  # (n, 3rgb*sh)
        if self.rgb_sh_rest is not None:
            f_rest = (
                self.rgb_sh_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().float().numpy()
            )  # (n, ((sh+1)**2-1)*3)
        else:
            f_rest = np.zeros((xyz_w.shape[0], ((self.sh_degree + 1) ** 2 - 1) * 3))  # (n, 3) # (n, ((sh+1)**2-1)*3)

        # opacities is saved as logit
        opacities = inverse_sigmoid(self.opacity).detach().cpu().float().numpy()  # (n, 1)

        # scale is saved as logit
        scale = self.scaling.detach().log().cpu().float().numpy()  # (n, 3)

        rotation = self.quaternion.detach().cpu().float().numpy()  # (n, 4)

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz_w.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            [attr for attr in [xyz_w, normal_ws, f_dc, f_rest, opacities, scale, rotation] if attr is not None],
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(filename)

    @staticmethod
    def load_ply(
        filename: str,
        sh_degree: int = None,
        device: torch.device = torch.device("cpu"),
    ) -> "Gaussians":
        """Load a specialized ply file for gaussians."""
        plydata = PlyData.read(filename)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )  # (n, 3)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]  # (n, 1)

        features_dc = np.zeros((xyz.shape[0], 3, 1))  # (n, 3, 1)
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) % 3 == 0
        if sh_degree is not None:
            assert len(extra_f_names) == 3 * (sh_degree + 1) ** 2 - 3
        else:
            sh_degree = round((len(extra_f_names) // 3 + 1) ** 0.5) - 1
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (sh_degree + 1) ** 2 - 1)
        )  # (n, 3, ((sh+1)**2-1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))  # (n, 3)
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))  # (n, 4)
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # create gaussian
        return Gaussians(
            sh_degree=sh_degree,
            xyz_w=torch.from_numpy(xyz).to(dtype=torch.float, device=device),  # (n, 3)
            rgb_sh=None,
            rgb_sh_dc=torch.from_numpy(features_dc).transpose(-1, -2).to(dtype=torch.float, device=device),  # (n, 1, 3)
            rgb_sh_rest=torch.from_numpy(features_extra)
            .transpose(-1, -2)
            .to(dtype=torch.float, device=device),  # (n, ((sh+1)**2-1), 3)
            scaling_logit=torch.from_numpy(scales).to(dtype=torch.float, device=device),  # (n, 3)
            quaternion_prenorm=torch.from_numpy(rots).to(dtype=torch.float, device=device),  # (n, 4)
            opacity_logit=torch.from_numpy(opacities).to(dtype=torch.float, device=device),  # (n, 1),
        )

    # @staticmethod
    # def from_spz_cloud(
    #     self,
    #     cloud: 'spz.GaussianCloud',
    # ):
    #     """
    #     Convert from the spz gaussian cloud. Deep copy (as we move from npy to tensor)
    #     Args:
    #         cloud:
    #             positions:
    #                 (num_points * 3xyz,)
    #             scales:
    #                 (num_points * 3xyz,)
    #             rotations:
    #                 (num_points * 4xyzw,)
    #             alphas:
    #                 (num_points,)
    #             colors:
    #                 (num_points * 3rgb,)  float32
    #             sh_coeffs:
    #                 (num_points * sh_coeffs_per_point,)
    #             sh_degree:
    #                 int
    #             antialiased:
    #                 bool, whether the gaussian should be rendered with mipsplatting
    #
    #     Returns:
    #
    #     """


class GSCamera:
    def __init__(
        self,
        H_c2w: torch.Tensor,  # (4, 4)
        intrinsic: torch.Tensor,  # (3, 3)
        width_px: int,
        height_px: int,
        znear: float = 0.01,
        zfar: float = 100.0,
    ):
        """
        Coordinate: x to right, y to down, z to far

        Args:
            H_c2w:
            intrinsic:
            width_px:
            height_px:
            znear:
            zfar:
        """
        super().__init__()

        self.H_c2w = H_c2w  # (4, 4)
        self.intrinsic = intrinsic  # (3, 3)
        self.width_px = width_px
        self.height_px = height_px
        self.zfar = zfar
        self.znear = znear

    @property
    def FoVx(self) -> float:
        """fov (in radian) along width"""
        return focal2fov(self.intrinsic[0, 0].item(), pixels=self.width_px)

    @property
    def FoVy(self) -> float:
        return focal2fov(self.intrinsic[1, 1].item(), pixels=self.height_px)

    @property
    def world_view_transform(self):
        """
        Return H_w2c.T.  notice the transpose
        """
        H_w2c = rigid_motion.inv_homogeneous_tensors(self.H_c2w)  # (4, 4)
        return H_w2c.transpose(-1, -2)  # H_w2c.T (4, 4)

    @property
    def projection_matrix(self):
        """
        intrinisc + ndc

        notice the transpose
        """
        projection_matrix = getProjectionMatrix_general(
            focal_x=self.intrinsic[0, 0],
            focal_y=self.intrinsic[1, 1],
            width=self.width_px,
            height=self.height_px,
            znear=self.znear,
            zfar=self.zfar,
            cx=self.intrinsic[0, 2],
            cy=self.intrinsic[1, 2],
        )

        return projection_matrix.transpose(-1, -2)  # P.T  (4, 4)

    @property
    def full_proj_transform(self):
        """
        returns (P @ H_w2c).T

        notice the transpose
        """

        # H_w2c.T @ P.T = (P @ H_w2c).T
        full_proj_transform = self.world_view_transform @ self.projection_matrix  # (4, 4)
        return full_proj_transform

    @property
    def camera_center(self):
        return self.H_c2w[:3, 3].clone()


@linalg_utils.disable_tf32_and_autocast()
def render_3dgs(
    pc: Gaussians,
    H_c2w: torch.Tensor,  # (4, 4)
    intrinsic: torch.Tensor,  # (3, 3)
    width_px: int,
    height_px: int,
    bg_color: T.Union[torch.Tensor, float],
    override_color: T.Optional[torch.Tensor] = None,
    convert_SHs_python: bool = False,
    mip_kernel_size: float = 0,  # > 0 to use mip-gaussian-splatting
    mip_subpixel_offset: torch.Tensor = None,  # (h, w, 2uv)  [-0.5, 0.5]
    debug: bool = False,
):
    """
    Render 3d gaussians with splatting.

    Args:
        pc:
            (n, 3) 3d gaussians
        H_c2w:
            (4, 4) camera pose in the world cooridnate
        intrinsic:
            (3, 3) camera intrinsics
        width_px:
            camera width resolution (px)
        height_px:
            camera height resolution (px)
        bg_color:
            (3,) should be on cuda
        override_color:
            (n, 3rgb) the precomputed color of gaussian
        convert_SHs_python:
            whether to precompute rgb color from rgb spherical harmonics in python
            (instead of inside cuda)
    """

    assert H_c2w.is_cuda
    assert H_c2w.dtype == torch.float
    assert intrinsic.is_cuda
    assert intrinsic.dtype == torch.float
    pc.check()

    if isinstance(bg_color, (float, int)):
        bg_color = torch.ones(3, dtype=torch.float, device=H_c2w.device) * bg_color  # (3,)

    gs_camera = GSCamera(
        H_c2w=H_c2w,
        intrinsic=intrinsic,
        width_px=width_px,
        height_px=height_px,
    )

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.xyz_w,
            dtype=torch.float,  # pc.xyz_w.dtype ,
            requires_grad=True,
            device=pc.xyz_w.device,
        )
        + 0
    )
    try:
        if screenspace_points.requires_grad:
            screenspace_points.retain_grad()
    except:
        traceback.print_exc()
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(gs_camera.FoVx * 0.5)
    tanfovy = math.tan(gs_camera.FoVy * 0.5)
    assert bg_color.is_cuda
    with _nullcontext():
        if mip_kernel_size > 0:
            if mip_subpixel_offset is None:
                mip_subpixel_offset = torch.zeros(
                    (int(gs_camera.height_px), int(gs_camera.width_px), 2),
                    dtype=torch.float32,
                    device=H_c2w.device,
                )
            try:
                raster_settings = GaussianRasterizationSettings(
                    image_height=height_px,
                    image_width=width_px,
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=bg_color.float(),
                    scale_modifier=1.0,
                    viewmatrix=gs_camera.world_view_transform.float(),  # H_w2c.T
                    projmatrix=gs_camera.full_proj_transform.float(),  # (H_w2c -> intrinsic -> ndc).T
                    sh_degree=pc.sh_degree,
                    campos=gs_camera.camera_center.float(),
                    prefiltered=False,
                    kernel_size=mip_kernel_size,
                    subpixel_offset=mip_subpixel_offset,
                    debug=debug,
                )
            except Exception:
                # # NOTE: we should let the error shout out loundly instead of quitely making it work.

                # raster_settings = GaussianRasterizationSettings(
                #     image_height=height_px,
                #     image_width=width_px,
                #     tanfovx=tanfovx,
                #     tanfovy=tanfovy,
                #     bg=bg_color.float(),
                #     scale_modifier=1.0,
                #     viewmatrix=gs_camera.world_view_transform.float(),  # H_w2c.T
                #     projmatrix=gs_camera.full_proj_transform.float(),  # (H_w2c -> intrinsic -> ndc).T
                #     sh_degree=pc.sh_degree,
                #     campos=gs_camera.camera_center.float(),
                #     prefiltered=False,
                #     debug=debug,
                # )
                raise NotImplementedError(
                    f"We have {mip_kernel_size=} but we do not use mip-splatting. "
                    "Make sure you have the right/compatible diff_gaussian_rasterization and mip_kernel_size setup."
                )
        else:
            raster_settings = GaussianRasterizationSettings(
                image_height=height_px,
                image_width=width_px,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg_color.float(),
                scale_modifier=1.0,
                viewmatrix=gs_camera.world_view_transform.float(),  # H_w2c.T
                projmatrix=gs_camera.full_proj_transform.float(),  # (H_w2c -> intrinsic -> ndc).T
                sh_degree=pc.sh_degree,
                campos=gs_camera.camera_center.float(),
                prefiltered=False,
                debug=debug,
            )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.xyz_w  # (n, 3)
    means2D = screenspace_points  # (n, 3) return buffer
    opacity = pc.opacity  # (n, 1)
    scales = pc.scaling  # (n, 3)
    quaterinon = pc.quaternion  # (n, 4)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if convert_SHs_python:
            shs_view = pc.get_rgb_sh.transpose(1, 2).view(-1, 3, (pc.sh_degree + 1) ** 2)  # (n, 3rgb, dim_sh)
            dir_pp = torch.nn.functional.normalize(pc.xyz_w - gs_camera.camera_center.unsqueeze(0), dim=-1)  # (n, 3xyz)
            sh2rgb = sh_utils.eval_sh(
                pc.sh_degree, shs_view, dir_pp
            )  # (n, 3rgb), range [-1, 1], can be larger then 1 and smaller than -1
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_rgb_sh  # (n, dim_sh, 3)  spherical harmonics coeffs (order3) for 3rgb
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    with _nullcontext():
        rendered_image, radii = rasterizer(
            means3D=means3D.float(),  # (n, 3)  float32
            means2D=means2D.float(),  # (n, 3)  float32
            shs=shs.float() if shs is not None else None,  # (n, (sh+1)**2, 3) float 32 or None
            colors_precomp=colors_precomp.float() if colors_precomp is not None else None,  # (n, 3) None
            opacities=opacity.float(),  # (n, 1) float32
            scales=scales.float(),  # (n, 3) float32
            rotations=quaterinon.float(),  # (n, 4) float32
            cov3D_precomp=None,
        )
    # radii:  (n,) int32, the gaussian width on the image plane (in pixel)
    # rendered_image:  # (3rgb, h, w)  float32

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return dict(
        render=rendered_image,  # (3rgb, h, w)  float32
        viewspace_points=screenspace_points,  # (n, 3)
        visibility_filter=radii > 0,  # (n,)  bool
        radii=radii,  # (n,) int32
    )


@linalg_utils.disable_tf32_and_autocast()
def render_3dgs_gsplat(
    # camera
    H_c2w: torch.Tensor,  # (*b, 4, 4)
    intrinsic: torch.Tensor,  # (*b, 3, 3)
    width_px: int,
    height_px: int,
    # gaussian
    sh_degree: int,
    xyz_w: torch.Tensor,  # (*b, n, 3xyz_w)
    scaling: torch.Tensor,  # (*b, n, 3xyz)
    quaternion: torch.Tensor,  # (*b, n, 4xyzw)
    opacity: torch.Tensor,  # (*b, n, 1)
    rgb_sh: T.Optional[torch.Tensor] = None,  # (*b, n, (sh+1)**2, 3rgb)
    feature: T.Optional[torch.Tensor] = None,  # (*b, n, d)
    #
    render_depth: bool = False,
    mip_kernel_size: float = 0.1,  # > 0 to use mip-gaussian-splatting
    t_min: float = 0.01,
    t_max: float = 1.0e6,
    th_2d_radius: float = 0.0,
    depth_mode: str = "expectation",
):
    """
    Render 3d gaussians with splatting.

    Args:
        H_c2w:
            (*b, 4, 4) camera pose in the world cooridnate
        intrinsic:
            (*b, 3, 3) camera intrinsics
        width_px:
            camera width resolution (px)
        height_px:
            camera height resolution (px)

        xyz_w:
            (*b, n, 3xyz_w) gaussian center in the world coordinate
        scaling:
            (*b, n, 3xyz) gaussian std in the world coordinate
        quaternion:
            (*b, n, 4xyzw) gaussian rotation R_g2w but represented as quaternion
        opacity:
            (*b, n, 1) gaussian opacity
        rgb_sh:
            (*b, n, (sh+1)**2, 3rgb) rgb color on sphere, represented as spherical harmonics.
            If None, rgb will not be rendered
        feature:
            (*b, n, d) feature to be rendered, always uses sh_degree = 0

        depth_mode:
            "expectation": expected z-depth.  z_out = (\sum_i wi zi) / (\sum_i wi)
            "accumlation": accumulated z-depth. z_out = \sum_i wi zi

    Returns:
        rgb:
            (*b, h, w, 3rgb) or None, premultiplied (ie, already multiplied with alpha)
        feature:
            (*b, h, w, d) or None, premultiplied (ie, already multiplied with alpha)
        depth:
            (*b, h, w), z_c, or None, premultiplied (ie, already multiplied with alpha)
        alpha:
            (*b, h, w, 1) [0, 1]

    Notes:
        Even though gsplat supports rendering multiple cameras
        at the same time (ie, q > 1). My experience is
        it often goes out of memory and the chunking is
        quite tricky to select.  I decided to support only q=1
        and use for loop in python (instead of c/cuda).
    """

    import gsplat

    if rgb_sh is None and feature is None:
        assert render_depth
        if depth_mode == "expectation":
            render_mode = "ED"
        elif depth_mode == "accumulation":
            render_mode = "D"
        else:
            raise NotImplementedError
    else:
        render_mode = "RGB"
        if render_depth:
            if depth_mode == "expectation":
                render_mode = f"{render_mode}+ED"
            elif depth_mode == "accumulation":
                render_mode = f"{render_mode}+D"
            else:
                raise NotImplementedError

    is_depth_rendered = False
    rendered_rgb = None
    rendered_feature = None
    rendered_depth = None
    rendered_alpha = None
    with _nullcontext():
        H_w2c = rigid_motion.inv_homogeneous_tensors(H_c2w.float())  # (4, 4)
        if rgb_sh is not None:
            if sh_degree == 0:
                rgb_sh = rgb_sh.squeeze(-2)
                _sh_degree = None
            else:
                assert rgb_sh.size(-2) == (sh_degree + 1) ** 2
                _sh_degree = sh_degree

            out, rendered_alpha, meta = gsplat.rasterization(
                means=xyz_w.float(),  # (*b, n, 3)
                quats=quaternion.float(),  # (*b, n, 4)
                scales=scaling.float(),  # (*b, n, 3)
                opacities=opacity.float().squeeze(-1),  # (*b, n)
                colors=rgb_sh.float(),  # (*b, n, (sh+1)**2, 3rgb) or (*b, n, 3)
                viewmats=H_w2c.unsqueeze(0),  # (q=1, 4, 4)
                Ks=intrinsic.float().unsqueeze(0),  # (q=1, 3, 3)
                width=width_px,
                height=height_px,
                near_plane=t_min,
                far_plane=t_max,
                radius_clip=th_2d_radius,
                eps2d=mip_kernel_size,
                sh_degree=_sh_degree,
                packed=xyz_w.requires_grad,  # memory tradeoff seems reasonable https://docs.gsplat.studio/main/tests/profile.html
                backgrounds=None,
                render_mode=render_mode,
                sparse_grad=False,  # usually need sparse optimizer
                absgrad=False,
                rasterize_mode="classic" if mip_kernel_size < 1e-6 else "antialiased",
                channel_chunk=32,
                distributed=False,
                camera_model="pinhole",
                segmented=False,
                covars=None,
            )
            rendered_alpha = rendered_alpha.squeeze(-4)  # (h, w, 1)

            # out: (q, h, w, d)
            if render_mode in ["RGB+D", "RGB+ED"]:
                rendered_rgb = out[..., :-1].squeeze(-4)  # (*b, h, w, c)
                rendered_depth = out[..., -1].squeeze(-3)  # (*b, h, w)
                is_depth_rendered = True
            elif render_mode in ["RGB"]:
                rendered_rgb = out.squeeze(-4)  # (*b, h, w, c)
            elif render_mode in ["D", "ED"]:
                rendered_depth = out.squeeze(-1).squeeze(-3)  # (*b, h, w)
                is_depth_rendered = True
            else:
                raise NotImplementedError(render_mode)

        if feature is not None:
            if is_depth_rendered or (not rendered_depth):
                _render_mode = "RGB"
            else:
                _render_mode = render_mode

            out, rendered_alpha, meta = gsplat.rasterization(
                means=xyz_w.float(),  # (*b, n, 3)
                quats=quaternion.float(),  # (*b, n, 4)
                scales=scaling.float(),  # (*b, n, 3)
                opacities=opacity.float().squeeze(-1),  # (*b, n)
                colors=feature.float(),  # (*b, n, 3)
                viewmats=H_w2c.unsqueeze(0),  # (q=1, 4, 4)
                Ks=intrinsic.float().unsqueeze(0),  # (q=1, 3, 3)
                width=width_px,
                height=height_px,
                near_plane=t_min,
                far_plane=t_max,
                radius_clip=th_2d_radius,
                eps2d=mip_kernel_size,
                sh_degree=None,
                packed=xyz_w.requires_grad,  # memory tradeoff seems reasonable https://docs.gsplat.studio/main/tests/profile.html
                backgrounds=None,
                render_mode=_render_mode,
                sparse_grad=False,  # usually need sparse optimizer
                absgrad=False,
                rasterize_mode="classic" if mip_kernel_size < 1e-6 else "antialiased",
                channel_chunk=32,
                distributed=False,
                camera_model="pinhole",
                segmented=False,
                covars=None,
            )
            rendered_alpha = rendered_alpha.squeeze(-4)  # (h, w, 1)

            # out: (q=1, h, w, d)
            if _render_mode in ["RGB+D", "RGB+ED"]:
                rendered_feature = out[..., :-1].squeeze(-4)  # (*b, h, w, c)
                rendered_depth = out[..., -1].squeeze(-3)  # (*b, h, w)
                is_depth_rendered = True
            elif _render_mode in ["RGB"]:
                rendered_feature = out.squeeze(-4)  # (*b, h, w, c)
            elif _render_mode in ["D", "ED"]:
                rendered_feature = None
                rendered_depth = out.squeeze(-1).squeeze(-3)  # (*b, h, w)
                is_depth_rendered = True
            else:
                raise NotImplementedError(_render_mode)

    return dict(
        premultiplied_rgb=rendered_rgb,  # (*b, h, w, 3rgb) or None
        premultiplied_feature=rendered_feature,  # (*b, h, w, d) or None
        premultiplied_depth=rendered_depth,  # (*b, h, w)  or None
        alpha=rendered_alpha,  # (*b, h, w, 1) [0, 1]
    )


def construct_gaussians_from_point_cloud(
    point_radius: T.Union[float, torch.Tensor],
    xyz_w: torch.Tensor,  # (n, 3)
    rgb: T.Optional[torch.Tensor] = None,  # (n, 3)
    normal_w: T.Optional[torch.Tensor] = None,
    opacity: T.Union[float, torch.Tensor] = 1.0,
    use_2d_gaussian: bool = False,
    use_adaptive_radius: bool = True,
) -> Gaussians:
    """
    Construct a 2d/3d gaussians from point cloud.

    Args:
        point_radius:
            float or (n,)  std of the gaussians
        xyz_w:
            (n, 3) mean/center of the gaussians in the world coordinate
        rgb:
            (n, 3) rgb color (dc) of the gaussians.  If None, use default xyz colormap
        normal_w:
            (n, 3) normal of the gaussians
        opacity:
            float or (n,), (n, 1)  [0, 1]  opacity of the gaussians
        use_2d_gaussian:
            whether to use 2d flat gaussians
        use_adaptive_radius:
            whether to use knn to determine the gaussian std.  In this case, point_radius is the max std.

    Returns:
        Gaussians with sh_degree = 0
    """

    dtype = torch.float  # gaussian supports float only
    device = xyz_w.device
    num_points = xyz_w.size(0)

    if isinstance(opacity, (int, float)):
        opacity = torch.ones(num_points, 1, dtype=dtype, device=device) * opacity
    opacity = opacity.reshape(num_points, 1)

    if isinstance(point_radius, (int, float)):
        point_radius = torch.ones(num_points, 1, dtype=dtype, device=device) * point_radius  # (n, 1)
    point_radius = point_radius.reshape(num_points, 1)

    if use_adaptive_radius:
        knn_out = pytorch3d.ops.knn_points(
            p1=xyz_w.unsqueeze(0),
            p2=xyz_w.unsqueeze(0),
            K=min(8, xyz_w.size(0)),
        )
        rms_dist = knn_out.dists.sqrt().mean(dim=-1).squeeze(0)  # (n,)
        # rms_dist = 0.75 * rms_dist
        point_radius = torch.minimum(rms_dist.unsqueeze(-1), point_radius)  # (n, 1)

    if not use_2d_gaussian:
        scaling = torch.ones(num_points, 3, dtype=dtype, device=device) * point_radius  # (n, 3)
        quaternion_prenorm = torch.ones(num_points, 4, dtype=dtype, device=device)
    else:
        assert normal_w is not None
        scaling = torch.ones(num_points, 3, dtype=dtype, device=device) * point_radius  # (n, 3)
        scaling[:, 2] = 1e-8

        # in case some normal is broken
        normal_w = torch.nn.functional.normalize(normal_w, dim=-1)
        invalid_mask = (torch.linalg.vector_norm(normal_w, dim=-1) - 1).abs() > 1e-6  # (n,)
        normal_w[invalid_mask] = 3**-0.5
        opacity = opacity.clone()
        opacity[invalid_mask] = 0

        R_g2w = rigid_motion.get_min_R(
            v1=torch.tensor(
                [
                    0.0,
                    0.0,
                    1.0,
                ],
                dtype=dtype,
                device=device,
            ).expand(num_points, 3),
            v2=normal_w.to(dtype=dtype, device=device),
        )  # (n, 3, 3)
        quaternion_prenorm = pytorch3d.transforms.matrix_to_quaternion(R_g2w)  # (n, 4)

    gaussians = Gaussians(
        sh_degree=0,
        xyz_w=xyz_w,  # (n, 3)
        rgb_sh_dc=sh_utils.RGB2SH(rgb).unsqueeze(1),  # (n, 1, 3)
        rgb_sh_rest=torch.zeros(num_points, 0, 3),
        scaling_logit=None,
        opacity_logit=None,
        scaling=scaling,  # (n, 1)
        quaternion_prenorm=quaternion_prenorm,  # (n, 4)
        opacity=opacity,
    )

    return gaussians
