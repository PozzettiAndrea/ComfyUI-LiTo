# Modified from TRELLIS https://github.com/microsoft/TRELLIS/tree/main/trellis/representations/mesh/cube2mesh.py

from easydict import EasyDict as edict
from trellis.modules.sparse import SparseTensor
# Vendored FlexiCubes lives next to this file (see flexicubes/) so we don't
# depend on TRELLIS having its own flexicubes git submodule populated.
from .flexicubes.flexicubes import FlexiCubes

import torch

from lito.integrations.trellis.representations.mesh.utils_cube import (
    construct_dense_grid,
    get_defomed_verts,
    get_dense_attrs,
    sparse_cube2verts,
)


def sdf_reg_loss(sdf, all_edges):
    # From https://github.com/nv-tlabs/FlexiCubes/blob/4cc7d6c3d0cee83c011ce36721b81adff0dd7db6/examples/loss.py#L96
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()
    ) + torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff


class MeshExtractResult:
    def __init__(self, vertices, faces, vertex_attrs=None, res=64):
        self.vertices = vertices
        self.faces = faces.long()
        self.vertex_attrs = vertex_attrs
        self.face_normal = self.comput_face_normals(vertices, faces)
        self.res = res
        self.success = vertices.shape[0] != 0 and faces.shape[0] != 0

        # training only
        self.tsdf_v = None
        self.tsdf_s = None
        # For Eq. (8) and others (maybe Sec. 5.2?)
        # https://github.com/nv-tlabs/FlexiCubes/blob/4cc7d6c3d0cee83c011ce36721b81adff0dd7db6/examples/optimize.py#L112-L113
        self.reg_loss = None
        # For Eq. (9) in https://arxiv.org/abs/2308.05371
        # https://github.com/nv-tlabs/FlexiCubes/blob/4cc7d6c3d0cee83c011ce36721b81adff0dd7db6/examples/optimize.py#L111
        self.reg_sdf_loss = None

    def comput_face_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        # print(face_normals.min(), face_normals.max(), face_normals.shape)
        return face_normals[:, None, :].repeat(1, 3, 1)

    def comput_v_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(verts)
        v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

        v_normals = torch.nn.functional.normalize(v_normals, dim=1)
        return v_normals


class SparseFeatures2Mesh:
    def __init__(self, device="cuda", res=64, use_color=True, full_width: float = 1.0):
        """
        a model to generate a mesh from sparse features structures using flexicube
        """
        super().__init__()
        self.device = device
        self.res = res
        self.mesh_extractor = FlexiCubes(device=device)
        self.sdf_bias = -1.0 / res
        verts, cube = construct_dense_grid(self.res, self.device)
        self.reg_c = cube.to(self.device)  # [res^3, 8]
        self.reg_v = verts.to(self.device)  # [(res+1)^3, 8]
        self.use_color = use_color
        self.full_width = full_width
        self._calc_layout()

        #  Retrieve all the edges of the voxel grid; these edges will be utilized to
        #  compute the regularization loss in subsequent steps of the process.
        self.all_edges = self.reg_c[:, self.mesh_extractor.cube_edges].reshape(-1, 2)
        # NOTE: this is too slow on CPU. We moved it to GPU for lazy computing
        self.grid_edges = None  # torch.unique(self.all_edges, dim=0)

    def _calc_layout(self):
        LAYOUTS = {
            "sdf": {"shape": (8, 1), "size": 8},
            "deform": {"shape": (8, 3), "size": 8 * 3},
            "weights": {"shape": (21,), "size": 21},
        }
        if self.use_color:
            """
            6 channel color including normal map
            """
            LAYOUTS["color"] = {
                "shape": (
                    8,
                    6,
                ),
                "size": 8 * 6,
            }
        self.layouts = edict(LAYOUTS)
        start = 0
        for k, v in self.layouts.items():
            v["range"] = (start, start + v["size"])
            start += v["size"]
        self.feats_channels = start

    def get_layout(self, feats: torch.Tensor, name: str):
        if name not in self.layouts:
            return None
        return feats[:, self.layouts[name]["range"][0] : self.layouts[name]["range"][1]].reshape(
            -1, *self.layouts[name]["shape"]
        )

    def __call__(self, cubefeats: SparseTensor, training: bool = False):
        """
        Generates a mesh based on the specified sparse voxel structures.
        Args:
            cubefeats:  sparsetensor, (b, d=101), 4bijk
            cube_attrs [Nx21] : Sparse Tensor attrs about cube weights
            verts_attrs [Nx10] : [0:1] SDF [1:4] deform [4:7] color [7:10] normal
        Returns:
            return the success tag and ni you loss,
        """
        # add sdf bias to verts_attrs
        coords = cubefeats.coords[:, 1:]  # (num_occupied, 3ijk)
        feats = cubefeats.feats  # (num_occupied, d=101)

        sdf, deform, color, weights = [self.get_layout(feats, name) for name in ["sdf", "deform", "color", "weights"]]
        # sdf: (num_occupied, 8corners, 1),  deform: (num_occupied, 8corners, 3),  color: (num_occupied, 8corners, 6),  weights: (num_occupied, 21)
        sdf += self.sdf_bias  # (num_occupied, 8corners, 1)
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_pos, v_attrs, reg_loss = sparse_cube2verts(
            coords, torch.cat(v_attrs, dim=-1), training=training
        )  # v_pos: (num_dual_verts, 3ijk), v_attrs: (num_dual_vert, d), reg_loss: (,)

        # convert sparse voxel to dense voxel
        v_attrs_d = get_dense_attrs(
            v_pos, v_attrs, res=self.res + 1, sdf_init=True
        )  # (num_dense_dual_vert=(res+1)^3, d=10)
        weights_d = get_dense_attrs(coords, weights, res=self.res, sdf_init=False)  # (num_dense_cell=(res)^3, d=21)
        if self.use_color:
            sdf_d, deform_d, colors_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4], v_attrs_d[..., 4:]
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None

        # convert dense ijk [0, res] to xyz [-0.5, 0.5] and deform with tanh
        x_nx3 = get_defomed_verts(
            self.reg_v, deform_d, self.res, full_width=self.full_width
        )  # (num_dense_dual_vert, 3xyz) [-0.5, 0.5]

        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,  # (num_dense_dual_vert=(res_k * res_j * res_i), 3xyz)
            scalar_field=sdf_d,  # (num_dense_dual_vert,)
            cube_idx=self.reg_c,  # (num_dense_cell, 8corners)
            resolution=self.res,  # 256
            beta=weights_d[:, :12],  # (num_dense_cell, 12)
            alpha=weights_d[:, 12:20],  # (num_dense_cell, 8)
            gamma_f=weights_d[:, 20],  # (num_dense_cell, )
            voxelgrid_colors=colors_d,  # (num_dense_dual_vert, 6)
            training=training,
        )  # vertices: (num_v, 3xyz) float,  faces: (num_f, 3vidx) long, colors: (num_v, 6) float

        mesh = MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=colors, res=self.res)
        if training:
            # See
            # - https://github.com/nv-tlabs/FlexiCubes/blob/4cc7d6c3d0cee83c011ce36721b81adff0dd7db6/examples/optimize.py#L78-L81
            # - https://github.com/nv-tlabs/FlexiCubes/blob/4cc7d6c3d0cee83c011ce36721b81adff0dd7db6/examples/optimize.py#L111
            if self.grid_edges is None:
                # lazy computing, torch.unique on GPU is fast while on CPU is extremely slow.
                self.grid_edges = torch.unique(self.all_edges, dim=0)
            reg_sdf_loss = sdf_reg_loss(sdf_d, self.grid_edges).mean()
            mesh.reg_sdf_loss = reg_sdf_loss

            # See https://github.com/nv-tlabs/FlexiCubes/blob/4cc7d6c3d0cee83c011ce36721b81adff0dd7db6/examples/optimize.py#L112-L113
            if mesh.success:
                reg_loss += L_dev.mean() * 0.5
            reg_loss += (weights[:, :20]).abs().mean() * 0.2
            mesh.reg_loss = reg_loss
            mesh.tsdf_v = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)
            mesh.tsdf_s = v_attrs[:, 0]
        return mesh
