#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#
import copy
import math
import sys
import typing as T
import warnings

import cv2
import numpy as np

# preprocess mesh loaded by open3d
import open3d as o3d
import PIL
from scipy.spatial import cKDTree
import skimage
import trimesh

from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.packed_to_padded import packed_to_padded
from pytorch3d.renderer.mesh.rasterizer import Fragments as MeshFragments
import pytorch3d.structures.meshes
import pytorch3d.structures.utils
import torch

from plibs import utils


def load_mesh_using_trimesh(
    filename: str,
    raise_error_if_no_color: bool = False,
) -> T.Dict[str, T.Union[o3d.geometry.TriangleMesh, bool]]:
    """
    Open a mesh file using trimesh. Convert material properties into texture map.

    Args:
        filename:
            filename of the mesh
        raise_error_if_no_color:
            whether to raise RuntimeError if no color can be read
    Returns:
        o3d_mesh:
            open3d mesh
        has_color_texture:
            bool
    """
    scene = trimesh.load(filename, force="scene")
    o3d_meshes = []
    num_texture = 0
    for name in scene.geometry:
        t_mesh: trimesh.Trimesh = scene.geometry[name]
        v_xyz = t_mesh.vertices  # (n, 3) float
        edge = t_mesh.faces  # (e, 3) int

        # create geometry
        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(v_xyz), triangles=o3d.utility.Vector3iVector(edge)
        )
        # print(f'v_xyz.shape {v_xyz.shape}')
        # print(f'edge.shape {edge.shape}')

        if not hasattr(t_mesh, "visual"):
            o3d_meshes.append(o3d_mesh)
            continue

        # convert vertex visual to texture visual
        if isinstance(t_mesh.visual, trimesh.visual.ColorVisuals):
            t_mesh.visual = t_mesh.visual.to_texture()

        uv = np.array(t_mesh.visual.uv, dtype=np.float64)  # (n, 2)  [0, 1]
        # print(f'uv: {np.max(uv)}   {np.min(uv)}')

        if isinstance(t_mesh.visual.material, trimesh.visual.material.SimpleMaterial):
            t_mesh.visual.material = t_mesh.visual.material.to_pbr()

        # get texture map
        if getattr(t_mesh.visual.material, "image", None) is not None:
            # should not happen since we already convert SimpleMaterial to PBRMaterial
            texture: PIL.Image = t_mesh.visual.material.image
            # print(f'texture.shape = {np.array(texture).shape}')
            texture = skimage.img_as_float(np.array(texture)).astype(np.float32)
            texture = np.array(texture[::-1])
            num_texture += 1
        elif getattr(t_mesh.visual.material, "baseColorTexture", None) is not None:
            texture: PIL.Image = t_mesh.visual.material.baseColorTexture
            # print(f'texture.shape = {np.array(texture).shape}')
            texture = skimage.img_as_float(np.array(texture)).astype(np.float32)
            texture = np.array(texture[::-1])
            num_texture += 1
        elif getattr(t_mesh.visual.material, "emissiveTexture", None) is not None:
            texture: PIL.Image = t_mesh.visual.material.emissiveTexture
            texture = skimage.img_as_float(np.array(texture)).astype(np.float32)
            texture = np.array(texture[::-1])
            num_texture += 1
        elif getattr(t_mesh.visual.material, "emissiveTexture", None) is not None:
            texture: PIL.Image = t_mesh.visual.material.emissiveTexture
            texture = skimage.img_as_float(np.array(texture)).astype(np.float32)
            texture = np.array(texture[::-1])
            num_texture += 1
        elif getattr(t_mesh.visual.material, "baseColorFactor", None) is not None:
            color = np.array(t_mesh.visual.material.baseColorFactor[:3]).astype(np.float32) / 255.0
            texture = np.ones((10, 10, 3), dtype=np.float32)
            texture[:, :] = color
            # set uv
            uv = np.ones((v_xyz.shape[0], 2), dtype=np.float64) * 0.5  # (n, 2)  [0.5, 0.5]
            num_texture += 1
        elif getattr(t_mesh.visual.material, "emissiveFactor", None) is not None:
            color = np.array(t_mesh.visual.material.emissiveFactor[:3]).astype(np.float32)
            texture = np.ones((10, 10, 3), dtype=np.float32)
            texture[:, :] = color
            # set uv
            uv = np.ones((v_xyz.shape[0], 2), dtype=np.float64) * 0.5  # (n, 2)  [0.5, 0.5]
            num_texture += 1
        # elif getattr(t_mesh.visual.material, 'main_color', None) is not None:
        #     color = np.array(t_mesh.visual.material.main_color[:3])
        #     if color.dtype == np.uint8:
        #         color = color.astype(np.float32) / 255.
        #     elif color.dtype == np.uint16:
        #         color = color.astype(np.float32) / 65535.
        #
        #     texture = np.ones((10, 10, 3), dtype=np.float32)
        #     texture[:, :] = color
        #     num_texture += 1
        else:
            texture = np.ones((10, 10, 3), dtype=np.float32) * 0.5
            # create uv
            # if len(uv.shape) == 0:
            uv = np.ones((v_xyz.shape[0], 2), dtype=np.float64) * 0.5  # (n, 2)  [0.5, 0.5]

        o3d_mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(t_mesh.faces))
        texture_image = o3d.geometry.Image(texture)
        o3d_mesh.textures = [texture_image]  # (h, w, 3)

        # open3d uv is (num_triangle * 3, 2)
        o3d_uv = uv[np.reshape(edge, -1)]
        o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(o3d_uv)

        o3d_meshes.append(o3d_mesh)

    if raise_error_if_no_color and num_texture == 0:
        raise RuntimeError(f"no color info for {filename}")

    # merge o3d meshes
    o3d_mesh = o3d_meshes[0]
    for i in range(1, len(o3d_meshes)):
        o3d_mesh = o3d_mesh + o3d_meshes[i]
    return dict(
        o3d_mesh=o3d_mesh,
        has_color_texture=(num_texture > 0),
    )


def clean_mesh_uv(triangle_uvs: o3d.utility.Vector2dVector) -> o3d.utility.Vector2dVector:
    """
    ensure mesh uv is wrapped between 0 and 1, and triangles with identical uv in vertices are properly handled
    Args:
        triangle_uvs: input uv vectors  (num_triangles * 3, 2)

    Returns:
        cleaned uv vectors
    """

    uvs = np.asarray(triangle_uvs)
    uvs_mod = np.reshape(uvs, [int(uvs.shape[0] / 3), 3, uvs.shape[1]])  # (num_triangles, 3, 2)
    single_uvs = np.logical_and(
        uvs_mod[:, 0, :] == uvs_mod[:, 1, :], uvs_mod[:, 0, :] == uvs_mod[:, 2, :]
    )  # (num_triangles, 2)
    single_uvs = np.logical_and(single_uvs[:, 0], single_uvs[:, 1])  # (num_triangles,)
    # for identical uvs, change uvs to center of texture maps
    uvs_mod[single_uvs, 0, :] = np.array([0.5, 0.5])
    uvs_mod[single_uvs, 1, :] = np.array([0.5, 0.51])
    uvs_mod[single_uvs, 2, :] = np.array([0.51, 0.5])
    uvs_mod = np.reshape(uvs_mod, uvs.shape)  # (num_triangles*3, 2)

    # wrap uv, note that this would remove repetitive textures
    uvs_mod = uvs_mod - np.floor(uvs_mod)
    return o3d.utility.Vector2dVector(uvs_mod)


def clean_texture(img: T.Union[o3d.geometry.Image, np.ndarray]) -> T.Union[o3d.geometry.Image, np.ndarray]:
    """
    Make sure the texture is a rgb image (not gray one and no alpha channel)
    Args:
        img: input texture image

    Returns:
        img: ensure the output texture image has size (:,:,3)
    """
    img_type = type(img)

    # if img is empty, we need to fake a texture, otherwise open3d stuck during ray casting
    if isinstance(img, o3d.geometry.Image) and img.is_empty():
        img = np.ones((10, 10), dtype=np.uint8) * 128
        img = o3d.geometry.Image(img)
    elif isinstance(img, np.ndarray) and np.prod(img.shape) < 1:
        img = np.ones((10, 10), dtype=np.uint8) * 128

    # convert to np array
    img = np.asarray(img)
    assert len(img.shape) == 2 or len(img.shape) == 3, "wrong image size"

    if len(img.shape) == 2:  # gray image
        img = np.tile(np.expand_dims(img, axis=2), (1, 1, 3))
    elif img.shape[2] == 2:  # gray image with alpha
        img = np.tile(np.expand_dims(img[:, :, 0], axis=2), (1, 1, 3))
    elif img.shape[2] == 4:  # rgb image with alpha
        img = img[:, :, :3]

    # need to copy to a new image, or it would cause problem when convert
    # to o3d.cpu.pybind.geometry.Image
    img1 = copy.deepcopy(img)

    # convert back
    if img_type == o3d.geometry.Image:
        img1 = o3d.geometry.Image(img1)
    return img1


def estimate_mesh_freq(
    mesh: o3d.geometry.TriangleMesh,
) -> float:
    """
        Determine the average mesh length
        Not used anymore
    Args:
        mesh:

    Returns:

    """

    vertex_ids = np.array(mesh.triangles)
    vertex_pos = np.array(mesh.vertices)

    avg_length_func = lambda i, j: np.mean(
        np.linalg.norm(vertex_pos[vertex_ids[:, i], :] - vertex_pos[vertex_ids[:, j], :], axis=1)
    )
    return (avg_length_func(0, 1) + avg_length_func(1, 2) + avg_length_func(0, 2)) / 3.0


def preprocess_mesh(
    mesh: o3d.geometry.TriangleMesh,
    scale: T.Optional[float] = 1.0,
    center_w: T.Optional[T.List[float]] = (0.0, 0.0, 0.0),
    H_c2w: np.ndarray = None,
    clean: bool = True,
) -> o3d.geometry.TriangleMesh:
    """
    Clean the mesh uv and textures, normalize the mesh within [-scale,scale]

    Args:
        mesh: input mesh
        scale: parameter to scaling the mesh
    Returns:
        preprocessed mesh
    """

    # avoid affecting input
    mesh = copy.deepcopy(mesh)

    # center the mesh to (0,0,0)
    if center_w is not None:
        center_w = np.array(center_w)
        cs = mesh.get_axis_aligned_bounding_box().get_center()
        mesh.translate(center_w - cs, relative=True)

    # # get mesh frequency
    # # avg_tri_length = estimate_mesh_freq(mesh)

    # scale the mesh equally along xyz so that it lies within [-scale, scale]
    if scale is not None:
        s = np.max(mesh.get_axis_aligned_bounding_box().get_half_extent())
        if s > 1e-6:
            mesh.scale(scale=scale / s, center=np.zeros((3, 1)))
        else:
            warnings.warn(f"mesh aabb max_s = {s}")

    if clean:
        # wrap mesh uv to [0,1], handle triangles with all vertex have same uv
        mesh.triangle_uvs = clean_mesh_uv(mesh.triangle_uvs)

        # clean non rgb textures, such as ones with alpha or gray images
        mesh.textures = [clean_texture(img) for img in mesh.textures]

        # not sure why exist, not used
        # min_bounds = mesh.get_min_bound()  # (3,)  xyz  top left
        # max_bounds = mesh.get_max_bound()  # (3,)  xyz  bottom right
        # min_t = max(np.max(np.abs(min_bounds)), np.max(np.abs(max_bounds)))
        # max_t = min_t * 1.5
        # print(f'min_t = {min_t}, max_t = {max_t}')

    if H_c2w is not None:
        mesh.transform(H_c2w)

    return mesh


def extract_o3d_mesh_raw_information(
    o3d_mesh: o3d.geometry.TriangleMesh,
    compute_triangle_normal: bool = False,
    compute_vertex_normal: bool = False,
    clean_mesh: bool = False,
    merge_texture: bool = True,
    clone: bool = False,
) -> T.Dict[str, torch.Tensor]:
    """
    Compile the information in an o3d_mesh to create a raw_mesh.

    Args:
        o3d_mesh:
        merge_texture:
            whether to combine individual texture maps into
            a texture atlas by concatenating horizontally

    Returns:
        vertex_xyz_w:
            (num_vertices, 3xyz_w)  vertex xyz_w in the world coordinate
        triangles:
            (num_triangles, 3ijk)  int32.  the vertex index in vertex_xyz_w that form each triangle
        vertex_colors:
            (num_vertex, 3rgb), the vertex color
        texture_map:
            (h, w, 3), the texture map
        triangle_normals:
            (num_triangles, 3xyz), the triangle normals
        vertex_normals:
            (num_vertices, 3xyz), the vertex normals
        triangle_uvs:
            (num_triangles, 3i, 2uv)  the uv coordinate on the texture map for each vertex in the triangle
        vertex_uvs:
            (num_vertex, 2uv), the uv coordinate for each vertex on the texture map
        num_horizontal_texture_maps:
            number of texture maps concatenated horizontally
    Note:
        We only take the first texture map.  In other words, we ignore `triangle_material_ids`
        and assume they are all 0.
    """
    o3d_mesh = copy.deepcopy(o3d_mesh)

    if clean_mesh:
        # wrap mesh uv to [0,1], handle triangles with all vertex have same uv
        o3d_mesh.triangle_uvs = clean_mesh_uv(o3d_mesh.triangle_uvs)
        # clean non rgb textures, such as ones with alpha or gray images
        o3d_mesh.textures = [clean_texture(img) for img in o3d_mesh.textures]

    vertex_xyz_w = torch.from_numpy(np.asarray(o3d_mesh.vertices)).float()  # (num_vertices, 3xyz_w)
    triangles = torch.from_numpy(np.asarray(o3d_mesh.triangles)).int()  # (num_triangles, 3ijk)  int32

    # vertex color
    if o3d_mesh.vertex_colors is not None:
        vertex_colors = torch.from_numpy(np.asarray(o3d_mesh.vertex_colors)).float()  # (num_vertices, 3rgb)
        if vertex_colors.numel() == 0:
            vertex_colors = None
    else:
        vertex_colors = None

    # texture
    if o3d_mesh.textures is not None:  # list of o3d Image
        # textures = o3d_mesh.textures  # list of o3d Image
        if len(o3d_mesh.textures) == 0:
            # print(f'no texture')
            texture = None
            num_horizontal_texture_maps = 1
        elif len(o3d_mesh.textures) == 1:
            # print(f'# texture = 1')
            img = o3d_mesh.textures[0]
            img = np.asarray(img)
            img = prepare_texture_img(img, interp_to_power_of_2=True)  # (h, w, 3)
            texture = torch.from_numpy(img).float()  # (h, w, 3)
            if img.dtype == np.uint8:
                texture = texture / 255.0
            num_horizontal_texture_maps = 1
        elif not merge_texture:
            # print(f'# texture = {len(o3d_mesh.textures)}, not merged')
            warnings.warn(f"num texture maps = {len(o3d_mesh.textures)} > 1, we only take the first texture map")
            img = o3d_mesh.textures[0]
            img = np.asarray(img)
            img = prepare_texture_img(img, interp_to_power_of_2=True)  # (h, w, 3)
            texture = torch.from_numpy(img).float()  # (h, w, 3)
            if img.dtype == np.uint8:
                texture = texture / 255.0
            num_horizontal_texture_maps = 1
        elif merge_texture:
            # print(f'# texture = {len(o3d_mesh.textures)}, merging')
            # concatenate textures horizontally
            # also make sure uv are wrapped and adjusted accordingly
            o3d_mesh.triangle_uvs = clean_mesh_uv(o3d_mesh.triangle_uvs)

            textures = o3d_mesh.textures
            num_textures = len(textures)

            # determine new img resolution for all textures
            imgs = []
            for i in range(num_textures):
                img = textures[i]
                img = np.asarray(img)
                img = prepare_texture_img(img, interp_to_power_of_2=False)  # (h, w, 3)
                imgs.append(img)

            hs = torch.tensor([img.shape[0] for img in imgs], dtype=torch.float)
            ws = torch.tensor([img.shape[1] for img in imgs], dtype=torch.float)
            new_h = 2 ** max(1, round(torch.log2(hs.max()).item()))
            new_w = max(2 ** max(1, round(torch.log2(ws.max()).item())), new_h)

            # to prevent too large or too small a texture
            new_h = min(max(new_h, 128), 2048)
            new_w = min(max(new_w, 128), 2048)

            abs_max_w = 8192
            new_w2 = 2 ** max(1, round(math.floor(math.log2(abs_max_w / len(imgs)))))
            new_w = min(new_w2, new_w)

            # print(f'hs: {hs}')
            # print(f'ws: {ws}')
            # print(f'new_h, new_w: {new_h}, {new_w}')

            # resize all imgs
            for i in range(num_textures):
                h, w, _ = imgs[i].shape
                imgs[i] = prepare_texture_img(
                    imgs[i],
                    interp_to_power_of_2=True,
                    new_h=new_h,
                    new_w=new_w,
                )  # (new_h, new_w, 3)

            # adjust triangle uv for individual texture
            # we assume uv=(0,0) are at the image corner
            if o3d_mesh.triangle_uvs is not None:
                current_u_offset = 0.0
                u_scale = 1.0 / float(num_textures)
                triangle_uvs_np = np.asarray(o3d_mesh.triangle_uvs)
                triangle_uvs = torch.from_numpy(triangle_uvs_np).float()  # (num_triangles*3, 2)

                triangle_material_ids = (
                    torch.from_numpy(np.asarray(o3d_mesh.triangle_material_ids)).long().squeeze(-1)
                )  # (num_trianges,)

                # print(f'triangle_material_ids = {triangle_material_ids}')

                if triangle_uvs.numel() > 0:
                    triangle_uvs = triangle_uvs.reshape(-1, 3, 2)  # (num_triangles, 3i, 2uv)
                    for tidx in range(num_textures):
                        mask = triangle_material_ids == tidx  # (num_trianges,)
                        uvs = triangle_uvs[mask].clone()  # (n, 3i, 2uv)
                        uvs[..., 0] = uvs[..., 0] * u_scale + current_u_offset
                        triangle_uvs[mask] = uvs
                        current_u_offset += u_scale

                    triangle_uvs = triangle_uvs.reshape(-1, 2)  # (num_triangles * 3, 2uv)
                    triangle_uvs = triangle_uvs.detach().cpu().double().numpy()  # flaot64
                    o3d_mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)

            # keep only the merged texture map
            # for i in range(len(imgs)):
            #     print(f'imgs[{i}]: {imgs[i].dtype}  {imgs[i].max()}')
            #     plt.imshow(imgs[i])
            #     plt.pause(1e-6)

            merged_img = np.concatenate(imgs, axis=1)  # (new_h, new_w * num_img, 3)
            merged_img = np.ascontiguousarray(merged_img)

            o3d_mesh.textures = [o3d.geometry.Image(merged_img)]

            # plt.imshow(merged_img)
            # plt.pause(1e-6)

            # modify triangle_material_ids
            o3d_mesh.triangle_material_ids = o3d.utility.IntVector(
                np.zeros(np.asarray(o3d_mesh.triangles).shape[0], dtype=np.int32)
            )

            texture = torch.from_numpy(merged_img).float()  # (h, w, 3)
            if merged_img.dtype == np.uint8:
                texture = texture / 255.0
            num_horizontal_texture_maps = num_textures
        else:
            raise RuntimeError("unexpected condition")
    else:
        texture = None
        num_horizontal_texture_maps = 1

    # triangle_normals
    if compute_triangle_normal:
        o3d_mesh.compute_triangle_normals()

    if o3d_mesh.triangle_normals is not None:
        triangle_normals = torch.from_numpy(np.asarray(o3d_mesh.triangle_normals)).float()  # (num_triangles, 3xyz)
        if triangle_normals.numel() == 0:
            triangle_normals = None
    else:
        triangle_normals = None

    # vertex normal
    if compute_vertex_normal:
        o3d_mesh.compute_vertex_normals()

    if o3d_mesh.vertex_normals is not None:
        vertex_normals = torch.from_numpy(np.asarray(o3d_mesh.vertex_normals)).float()  # (num_vertices, 3xyz)
        if vertex_normals.numel() == 0:
            vertex_normals = None
    else:
        vertex_normals = None

    # triangle_uvs
    if o3d_mesh.triangle_uvs is not None:
        triangle_uvs = torch.from_numpy(np.asarray(o3d_mesh.triangle_uvs)).float()  # (num_triangles*3, 2)
        if triangle_uvs.numel() == 0:
            triangle_uvs = None
        else:
            triangle_uvs = triangle_uvs.reshape(-1, 3, 2)  # (num_triangles, 3i, 2uv)
    else:
        triangle_uvs = None

    # vertex_uvs  (the reason why we only support one texture map is we output vertex_uv, otherwise the triangle
    # can use different texture maps, even though the same vertex is shared across triangles)
    if triangle_uvs is not None:
        # we use torch_scatter with min reduction, because we assume
        # the uv of the same vertex in all triangles are the same
        # vertex_uvs = torch_scatter.scatter(
        #     src=triangle_uvs.reshape(-1, 2),  # (num_triangles * 3i, 2uv)
        #     index=triangles.reshape(-1).long(),    # (num_triangles * 3i,)
        #     dim=0,
        #     dim_size=vertex_xyz_w.size(0),
        #     reduce='min',  # 'min',
        # )   # (num_vertex, 2uv)

        # note that if a vertex is shared between triangles and the vertex is not duplicated
        # so that these triangles uses different vertices, the following code will
        # produce weird texture uv mapping
        vertex_uvs = torch.zeros(vertex_xyz_w.size(0), 2)  # (num_vertex, 2uv)
        vertex_uvs.index_copy_(
            dim=0,
            index=triangles.reshape(-1).long(),  # (num_triangles * 3i,)
            source=triangle_uvs.reshape(-1, 2),  # (num_triangles * 3i, 2uv)
        )  # (num_vertex, 2uv)

        # # debug
        # vertex_uv_list = [[] for _ in range(vertex_xyz_w.size(0))]
        # for triangle_idx in range(triangles.size(0)):
        #     triangle_uv = triangle_uvs[triangle_idx]  # (3, 2uv)
        #     vertex_idxs = triangles[triangle_idx]  # (3,)
        #     assert len(vertex_idxs) == 3
        #     for ii in range(vertex_idxs.size(0)):
        #         vertex_uv_list[vertex_idxs[ii]].append(triangle_uv[ii])
        # # end debug

    else:
        vertex_uvs = None
        # # debug
        # vertex_uv_list = None
        # # end debug

    out_dict = dict(
        vertex_xyz_w=vertex_xyz_w,  # (num_vertices, 3xyz_w)
        triangles=triangles,  # (num_triangles, 3ijk)  int32
        vertex_colors=vertex_colors,  # (num_vertex, 3rgb)
        texture_map=texture,  # (h, w, 3)
        triangle_normals=triangle_normals,  # (num_triangles, 3xyz)
        vertex_normals=vertex_normals,  # (num_vertices, 3xyz)
        triangle_uvs=triangle_uvs,  # (num_triangles, 3i, 2uv)
        vertex_uvs=vertex_uvs,  # (num_vertex, 2uv)
        o3d_mesh=o3d_mesh,
        num_horizontal_texture_maps=num_horizontal_texture_maps,
        # vertex_uv_list=vertex_uv_list,  # debug
    )

    if clone:
        for key in out_dict:
            if out_dict[key] is not None:
                if isinstance(out_dict[key], torch.Tensor):
                    out_dict[key] = out_dict[key].clone()

    return out_dict


def prepare_texture_img(
    img: np.ndarray,
    interp_to_power_of_2: bool = False,
    new_h: int = None,
    new_w: int = None,
) -> np.ndarray:
    assert len(img.shape) == 2 or len(img.shape) == 3, "wrong image size"
    if len(img.shape) == 2:  # gray image
        img = np.tile(np.expand_dims(img, axis=2), (1, 1, 3))
    elif img.shape[2] == 2:  # gray image with alpha
        img = np.tile(np.expand_dims(img[:, :, 0], axis=2), (1, 1, 3))
    elif img.shape[2] == 4:  # rgb image with alpha
        img = img[:, :, :3]

    if interp_to_power_of_2:
        ori_dtype = img.dtype
        # print(f'ori_dtype = {ori_dtype}')
        img = torch.from_numpy(img).float()  # (h, w, 3)
        if ori_dtype == np.uint8:
            img = img / 255.0

        h, w, _3 = img.shape
        if new_w is None or new_h is None:
            new_h = 2 ** max(1, round(math.log2(h)))
            new_w = 2 ** max(1, round(math.log2(w)))

        # print(f'h, w = {h}, {w},  new_h, new_w = {new_h}, {new_w}')
        img = (
            torch.nn.functional.interpolate(
                input=img.permute(2, 0, 1).unsqueeze(0),  # (b=1, 3, h, w)
                size=(new_h, new_w),
                mode="bilinear",
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )  # (new_h, new_w, 3)
        assert img.shape == (new_h, new_w, 3)
        # if ori_dtype == np.uint8:
        #     img = torch.floor(img * 255.)
        img = img.detach().cpu().numpy()
        # img = img.astype(ori_dtype)

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.float64:
        img = img.astype(np.float32)

    return img


def get_point_rgb_and_normal(
    point_triangle_idx: torch.Tensor,  # (m,) long
    point_bary: torch.Tensor,  # (m, 3uvw)
    vertex_xyz_w: torch.Tensor,  #  (n, 3xyz_w)
    triangles: torch.Tensor,  #  (num_triangles, 3ijk)  long or int32 (preferred)
    texture_rgb: torch.Tensor,  # (h, w, 3) [0, 1]
    vertex_uv: torch.Tensor,  #  (n, 2uv)  uv on the texture map
    vertex_normal_w: torch.Tensor,  # (n, 3xyz_w)
):
    """
    Args:
        point_triangle_idx:
            (m,) long, index of the triangle in `triangles`
        point_bary:
            (m, 3uvw) summed to 1, barycentric coefficient of the point in the corresponding triangle

        # mesh_dict from raw_mesh
        vertex_xyz_w:
            (n, 3xyz_w)
        triangles:
            (num_triangles, 3ijk)  long or int32 (preferred)
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

        vertex_normal_w:
            (n, 3xyz_w)
    """
    m = point_triangle_idx.size(0)

    # interpolate vertex uv
    face_uvs = vertex_uv[triangles[point_triangle_idx]]  # (n, 3ijk, 2uv) uv of each vertex in the triangle
    sampled_uvs = torch.einsum("ni,nij->nj", point_bary, face_uvs)  # (m, 2uv)
    sampled_uvs = sampled_uvs % 1.0  # handle uv <0 or > 1 (wrap around)

    # interpolate vertex normals
    if vertex_normal_w is not None:
        face_normals = vertex_normal_w[
            triangles[point_triangle_idx]
        ]  # (n, 3ijk, 3xyz_w) normal of each vertex in the triangle
        point_normal_w = torch.einsum("ni,nij->nj", point_bary, face_normals)  # (m, 3xyz_w)
        point_normal_w = torch.nn.functional.normalize(point_normal_w, dim=-1)
    else:
        point_normal_w = None

    # 4. interpolate texture map to get color
    h, w, _3rgb = texture_rgb.shape
    texture_rgb = texture_rgb.permute(2, 0, 1).unsqueeze(0)  # (b=1, 3, h, w)
    # convert uv [0, 1] -> [-1, 1]
    _uv = sampled_uvs * 2 - 1  # (n, 2uv)
    _uv = _uv.view(1, m, 1, 2)  # shape (1, m, 1, 2)

    # Sample
    point_rgb = torch.nn.functional.grid_sample(
        input=texture_rgb,  # (b=1, 3, h, w)
        grid=_uv.reshape(1, -1, 1, 2),  # (b=1, m, 1, 2)
        mode="bilinear",
        align_corners=False,
    )  # (1, 3, m, 1)
    point_rgb = point_rgb.squeeze(0).squeeze(-1).transpose(0, 1)  # (m, 3)  [0, 1]

    return dict(
        point_normal_w=point_normal_w,  # (m, 3xyz_w) normalized
        point_rgb=point_rgb,  # (m, 3) [0, 1]
        point_uv=sampled_uvs,  # (m, 2uv)  [0, 1]
    )


def sample_triangle_idxs_from_p3d_meshes(
    meshes: pytorch3d.structures.meshes.Meshes,
    num_samples: int,
    pad: int = 0,
):
    """
    Sample triangle index (padded format) using their areas
    as unnormalized probability.

    Args:
        meshes:
            A Meshes object with a batch of N meshes. N includes both valid and invalid meshes.
        num_samples:
            Integer giving the number of point samples per mesh.

    Returns:
        sample_face_idxs_padded:
            (num_meshes, num_samples)  triangle indexes ranges from 0 to max triangle idx in each mesh.
            padded with pad for empty meshes.
        mesh_valid_mask:
            (num_meshes,) bool, whether the mesh has any triangle with nonzero area

    Notes:
        If for a mesh the total surface area is 0, it will raise gpu device assertion when calling `multinomial`.
    """

    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()  # (total_v)
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    faces = meshes.faces_packed()  # (packed_n_triangle, 3)
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()  # (num_meshes,)
    num_meshes = len(meshes)
    num_valid_meshes = torch.sum(meshes.valid)  # Non empty meshes.

    # Only compute samples for non empty meshes
    sample_face_idxs_padded = torch.empty(
        num_meshes,
        num_samples,
        dtype=torch.long,
        device=meshes.device,
    ).fill_(pad)  # (num_meshes, num_samples)

    with torch.autocast(device_type=sample_face_idxs_padded.device.type, enabled=False):
        areas, _ = mesh_face_areas_normals(verts, faces)  # (total_triangle,) packed, Face areas can be zero.
        max_faces = meshes.num_faces_per_mesh().max().item()
        areas_padded = packed_to_padded(areas, mesh_to_face[meshes.valid], max_faces)  # (num_valid_meshes, max_faces)

        # check if total areas > 0
        total_area = areas_padded.sum(dim=-1)  # (num_valid_meshes,)
        valid_area_mask = torch.logical_and(
            total_area.isfinite(),  # (num_valid_meshes,)
            total_area > 1e-6,  # (num_valid_meshes,)
        )  # (num_valid_meshes,)
        total_valid_meshes = torch.sum(valid_area_mask)  # (,)

        valid_area_mask_padded = torch.zeros(num_meshes, dtype=torch.bool, device=meshes.device)  # (num_meshes,)
        valid_area_mask_padded[meshes.valid] = valid_area_mask  # (num_meshes,)

        if total_valid_meshes > 0:
            valid_areas_padded = areas_padded[valid_area_mask]  # (total_valid_meshes, max_faces)
            sample_face_idxs = torch.multinomial(
                input=valid_areas_padded,  # (total_valid_meshes, max_faces)
                num_samples=num_samples,
                replacement=True,
            )  # (total_valid_meshes, num_samples) index of unpacked triangles
            sample_face_idxs_padded[valid_area_mask_padded] = sample_face_idxs  # (num_meshes, num_samples)

        # sample_face_idxs = areas_padded.multinomial(
        #     num_samples, replacement=True
        # )  # (num_valid_meshes, num_samples) index of unpacked triangles
        # sample_face_idxs_padded[meshes.valid] = sample_face_idxs  # (num_meshes, num_samples)

    return dict(
        sample_face_idxs_padded=sample_face_idxs_padded,  # (num_meshes, num_samples)
        mesh_valid_mask=valid_area_mask_padded,  # (num_meshes,)
    )


def sample_points_from_p3d_meshes(
    meshes: pytorch3d.structures.meshes.Meshes,
    num_samples: int = 10000,
    return_normals: bool = False,
    return_textures: bool = False,
    return_uvs: bool = False,
    sample_face_idxs: torch.Tensor = None,
    # mesh_valid_mask: torch.Tensor = None,
    barycentric_coords: torch.Tensor = None,
    debug: bool = False,
) -> T.Dict[str, torch.Tensor]:
    """
    Convert a batch of meshes to a batch of pointclouds by uniformly sampling
    points on the surface of the mesh with probability proportional to the
    face area.

    Args:
        meshes:
            A Meshes object with a batch of N meshes. N includes both valid and invalid meshes.
        num_samples:
            Integer giving the number of point samples per mesh.
        return_normals:
            If True, return normals for the sampled points.
        return_textures:
            If True, return textures for the sampled points.
        return_uvs:
            If True, return uvs for the sampled points.
        sample_face_idxs:
            (N, num_samples) or None. If not None, it contains the triangle index each sample should be from.
            The idx starts from 0 to num_triangle -1 for each mesh
        mesh_valid_mask:
            (N,) whether each mesh contains any triangle with nonzero area.
        barycentric_coords:
            (N, num_samples, 3uvw) or None. If not None, it contains the baricentric coordinates to be used for sampling

    Returns:
        xyz_w:
            FloatTensor of shape (N, num_samples, 3) giving the
            coordinates of sampled points for each mesh in the batch. For empty
            meshes the corresponding row in the samples array will be filled with 0.
        normal_w:
            FloatTensor of shape (N, num_samples, 3) giving a normal vector
            to each sampled point. Only returned if return_normals is True.
            For empty meshes the corresponding row in the normals array will
            be filled with 0.
        textures:
            FloatTensor of shape (N, num_samples, C) giving a C-dimensional
            texture vector to each sampled point. Only returned if return_textures is True.
            For empty meshes the corresponding row in the textures array will
            be filled with 0.
        uv:
            FloatTensor of shape (N, num_samples, 2uv) giving the uv coordinates

    References:
        pytorch3d
        https://github.com/facebookresearch/pytorch3d/blob/e3d3a67a89907476bd5b63289f9669bd427ae550/pytorch3d/ops/sample_points_from_meshes.py
    """
    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()  # (total_v)
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    if return_textures and meshes.textures is None:
        raise ValueError("Meshes do not contain textures.")

    faces = meshes.faces_packed()  # (packed_n_triangle, 3)
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()  # (num_meshes,)
    num_meshes = len(meshes)

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    # Only compute samples for non empty meshes
    if sample_face_idxs is None:
        with torch.no_grad():
            # areas, _ = mesh_face_areas_normals(verts, faces)  # (total_triangle,) packed, Face areas can be zero.
            # max_faces = meshes.num_faces_per_mesh().max().item()
            # areas_padded = packed_to_padded(
            #     areas, mesh_to_face[meshes.valid], max_faces
            # )  # (num_valid_meshes, max_faces)
            #
            # # TODO (gkioxari) Confirm multinomial bug is not present with real data.
            # sample_face_idxs = areas_padded.multinomial(
            #     num_samples, replacement=True
            # )  # (num_valid_meshes, num_samples) index of unpacked triangles

            t_dict = sample_triangle_idxs_from_p3d_meshes(
                meshes=meshes,
                num_samples=num_samples,
            )  # (num_meshes, num_samples)
            sample_face_idxs = t_dict["sample_face_idxs_padded"]  # (num_meshes, num_samples)
            # mesh_valid_mask = t_dict["mesh_valid_mask"]  # (num_meshes,)

    else:
        assert sample_face_idxs.shape == (num_meshes, num_samples)
        # assert mesh_valid_mask.shape == (num_meshes,)

    num_valid_meshes = torch.sum(meshes.valid)  # Non empty meshes.

    # add the offset so sample_face_idxs points to the packed triangle index
    sample_face_idxs = sample_face_idxs[meshes.valid]  # (num_valid_meshes, num_samples)
    sample_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)  # (num_valid_meshes, num_samples)

    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces]  # (packed_n_triangle, 3triangle, 3xyz_w)
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]  # (packed_n_triangle, 3xyz_w)

    # Randomly generate barycentric coords.
    if barycentric_coords is None:
        w0, w1, w2 = rand_barycentric_coords(
            b=num_valid_meshes,
            n=num_samples,
            dtype=verts.dtype,
            device=verts.device,
        )  # (N=num_valid_meshes, num_samples)
    else:
        assert barycentric_coords.shape == (num_meshes, num_samples, 3)
        _tmp = barycentric_coords[meshes.valid]  # (num_valid_mesh, num_samples, 3uvw)
        w0, w1, w2 = _tmp[..., 0], _tmp[..., 1], _tmp[..., 2]  # (num_valid_meshes, num_samples)

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (num_valid_meshes, num_samples, 3xyz_w)
    b = v1[sample_face_idxs]  # (num_valid_meshes, num_samples, 3xyz_w)
    c = v2[sample_face_idxs]  # (num_valid_meshes, num_samples, 3xyz_w)
    samples[meshes.valid] = (w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c).to(dtype=samples.dtype)

    if return_normals:
        # Initialize normals tensor with fill value 0 for empty meshes.
        # Normals for the sampled points are face normals computed from
        # the vertices of the face in which the sampled point lies.
        normals = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)  # (num_meshes, num_samples, 3xyz_w)
        vert_normals = (v1 - v0).cross(v2 - v1, dim=1)  # (packed_n_triangle, 3xyz_w)
        vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(
            min=sys.float_info.epsilon
        )  # (packed_n_triangle, 3xyz_w)
        vert_normals = vert_normals[sample_face_idxs]  # (num_valid_meshes, num_samples, 3xyz_w)
        normals[meshes.valid] = vert_normals.to(dtype=normals.dtype)  # (num_meshes, num_samples, 3xyz_w)
    else:
        normals = None

    if return_textures:
        # Initialize textures tensor with fill value 0 for empty meshes.
        textures = torch.zeros((num_meshes, num_samples, meshes.textures.maps_padded().shape[-1]), device=meshes.device)

        if num_valid_meshes > 0:
            # fragment data are of shape NxHxWxK. Here H=S, W=1 & K=1.
            pix_to_face = sample_face_idxs.view(num_valid_meshes, num_samples, 1, 1)  # NxSx1x1
            bary = torch.stack((w0, w1, w2), dim=2).unsqueeze(2).unsqueeze(2)  # NxSx1x1x3
            # zbuf and dists are not used in `sample_textures` so we initialize them with dummy
            dummy = torch.zeros(
                (num_valid_meshes, num_samples, 1, 1), device=meshes.device, dtype=torch.float32
            )  # NxSx1x1
            fragments = MeshFragments(pix_to_face=pix_to_face, zbuf=dummy, bary_coords=bary, dists=dummy)
            # Create a temporary Meshes object with only valid meshes for texture sampling
            valid_meshes = meshes[meshes.valid]
            sampled_textures = valid_meshes.sample_textures(fragments)  # NxSx1x1xC
            textures[meshes.valid] = (sampled_textures[:, :, 0, 0, :]).to(dtype=textures.dtype)  # NxSxC
    else:
        textures = None

    if return_uvs and meshes.textures is not None:
        _textures = meshes.textures
        # Get UV coordinates (per-vertex-per-face, ie, the same vertex can have different uv in different triangles)
        verts_uvs = _textures.verts_uvs_padded()  # (num_meshes, max(Vt), 2uv)
        # Get indices of UVs for each face
        faces_uvs = _textures.faces_uvs_padded()  # shape: (num_meshes, max(F), 3)  long

        _num_meshes, maxF, _3 = faces_uvs.shape

        uv_per_face = torch.gather(
            verts_uvs,  # (num_meshes, max(Vt), 2)
            dim=1,
            index=faces_uvs.long()
            .reshape(_num_meshes, maxF * 3, 1)
            .expand(_num_meshes, maxF * 3, 2),  # (num_meshes, maxF*3, 2)
        ).reshape(_num_meshes, maxF, 3 * 2)  # (num_meshes, maxF, 3triangle * 2uv)

        packed_uv_per_face = pytorch3d.structures.utils.padded_to_packed(
            x=uv_per_face,  # (num_meshes, maxF, 3triangle * 2uv)
            split_size=meshes.num_faces_per_mesh().tolist(),  # (num_meshes,)
        )  # (packed_n_triangle, 3triangle * 2uv)
        packed_uv_per_face = packed_uv_per_face.reshape(
            packed_uv_per_face.size(0), 3, 2
        )  # (packed_n_triangle, 3triangle, 2uv)

        uv_1 = packed_uv_per_face[:, 0]  # (packed_n_triangle, 2uv)
        uv_2 = packed_uv_per_face[:, 1]  # (packed_n_triangle, 2uv)
        uv_3 = packed_uv_per_face[:, 2]  # (packed_n_triangle, 2uv)

        uv_a = uv_1[sample_face_idxs]  # (num_valid_meshes, num_samples, 2uv)
        uv_b = uv_2[sample_face_idxs]  # (num_valid_meshes, num_samples, 2uv)
        uv_c = uv_3[sample_face_idxs]  # (num_valid_meshes, num_samples, 2uv)

        # barycentric interpolation of uv
        uv = torch.zeros((num_meshes, num_samples, 2), device=meshes.device)  # (num_meshes, n, 2uv)
        uv[meshes.valid] = (w0[:, :, None] * uv_a + w1[:, :, None] * uv_b + w2[:, :, None] * uv_c).to(dtype=uv.dtype)

        # quick test: uv sampling the texture and compare with the sampled texture should get same value
        # we assume one texture map for one mesh
        if debug and textures is not None:
            texture_map = meshes.textures.maps_padded()  # (num_meshes, h, w, c)
            # we need to flip the texture map along y
            # (pytorch3d underlying assumes opengl coordinate system but it stores texture and uv not flipped)
            est_texture = (
                torch.nn.functional.grid_sample(
                    input=torch.flip(texture_map, dims=[1]).permute(0, 3, 1, 2),  # (num_meshes, c, h, w)
                    grid=uv.unsqueeze(1) * 2 - 1,  # (num_meshes, 1, n, 2uv) [-1, 1]
                    mode=meshes.textures.sampling_mode,
                    align_corners=meshes.textures.align_corners,
                    padding_mode=meshes.textures.padding_mode,
                )
                .squeeze(-2)
                .permute(0, 2, 1)
            )  # (num_meshes, n, c)

            # border handling is different in actual uv mapping and
            assert torch.allclose(est_texture, textures, rtol=1e-4, atol=1e-4)

    else:
        uv = None

    return dict(
        xyz_w=samples,  # (N, s, 3xyz_w)
        normal_w=normals,  # (N, s, 3xyz_w)
        textures=textures,  # (N, s, c)
        uv=uv,  # (N, s, 2uv)
    )


def sample_points_from_p3d_meshes_v2(
    meshes: pytorch3d.structures.meshes.Meshes,
    num_samples: int = 10000,
    return_normals: bool = False,
    return_textures: bool = False,
    return_uvs: bool = False,
    sample_face_idxs: torch.Tensor = None,
    mesh_valid_mask: torch.Tensor = None,
    barycentric_coords: torch.Tensor = None,
    debug: bool = False,
) -> T.Dict[str, torch.Tensor]:
    """
    Convert a batch of meshes to a batch of pointclouds by uniformly sampling
    points on the surface of the mesh with probability proportional to the
    face area.

    Args:
        meshes:
            A Meshes object with a batch of N meshes. N includes both valid and invalid meshes.
        num_samples:
            Integer giving the number of point samples per mesh.
        return_normals:
            If True, return normals for the sampled points.
        return_textures:
            If True, return textures for the sampled points.
        return_uvs:
            If True, return uvs for the sampled points.
        sample_face_idxs:
            (N, num_samples) or None. If not None, it contains the triangle index each sample should be from.
            The idx starts from 0 to num_triangle -1 for each mesh
        mesh_valid_mask:
            (N,) whether each mesh contains any triangle with nonzero area.
        barycentric_coords:
            (N, num_samples, 3uvw) or None. If not None, it contains the baricentric coordinates to be used for sampling

    Returns:
        xyz_w:
            FloatTensor of shape (N, num_samples, 3) giving the
            coordinates of sampled points for each mesh in the batch. For empty
            meshes the corresponding row in the samples array will be filled with 0.
        normal_w:
            FloatTensor of shape (N, num_samples, 3) giving a normal vector
            to each sampled point. Only returned if return_normals is True.
            For empty meshes the corresponding row in the normals array will
            be filled with 0.
        textures:
            FloatTensor of shape (N, num_samples, C) giving a C-dimensional
            texture vector to each sampled point. Only returned if return_textures is True.
            For empty meshes the corresponding row in the textures array will
            be filled with 0.
        uv:
            FloatTensor of shape (N, num_samples, 2uv) giving the uv coordinates
    """
    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    if return_textures and meshes.textures is None:
        raise ValueError("Meshes do not contain textures.")

    num_meshes = len(meshes)

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    # Only compute samples for non empty meshes
    if sample_face_idxs is None:
        with torch.no_grad():
            t_dict = sample_triangle_idxs_from_p3d_meshes(
                meshes=meshes,
                num_samples=num_samples,
            )  # (num_meshes, num_samples)
            sample_face_idxs = t_dict["sample_face_idxs_padded"]  # (num_meshes, num_samples)
            mesh_valid_mask = t_dict["mesh_valid_mask"]  # (num_meshes,)
    else:
        assert sample_face_idxs.shape == (num_meshes, num_samples)
        assert mesh_valid_mask.shape == (num_meshes,)

    num_valid_meshes = torch.sum(mesh_valid_mask)  # meshes with nonzero surface area
    mesh_valid_mask_list = mesh_valid_mask.tolist()  # (num_meshes,)
    # valid_idxs = mesh_valid_mask.nonzero(as_tuple=True)[0].tolist()  # (num_valid_mesh,)

    # get packed vertices and triangles from mesh_valid_mask
    vert_list = meshes.verts_list()  # (num_meshes,), each is (num_vertices_i, 3xyz_w)
    face_list = meshes.faces_list()  # (num_mesehs,), each is (num_triangles_i, 3ijk)
    verts = []
    faces = []
    mesh_to_face = []  # (num_meshes,)
    num_triangles_per_mesh = []  # (num_meshes,)
    current_face_idx = 0
    current_vertex_idx = 0
    for ii in range(num_meshes):
        mesh_to_face.append(current_face_idx)
        if mesh_valid_mask_list[ii]:
            verts.append(vert_list[ii])
            faces.append(face_list[ii] + current_vertex_idx)
            current_face_idx = current_face_idx + len(face_list[ii])
            current_vertex_idx = current_vertex_idx + len(vert_list[ii])
            num_triangles_per_mesh.append(len(face_list[ii]))
        else:
            num_triangles_per_mesh.append(0)

    verts = torch.cat(verts, dim=0)  # (packed_n_vertices, 3xyz_w)
    faces = torch.cat(faces, dim=0)  # (packed_n_triangle, 3xyz_w)
    mesh_to_face = torch.tensor(mesh_to_face, dtype=torch.long, device=meshes.device)  # (num_meshes,)
    assert mesh_to_face.shape == (num_meshes,)

    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    # add the offset so sample_face_idxs points to the packed triangle index
    sample_face_idxs = sample_face_idxs[mesh_valid_mask]  # (num_valid_meshes, num_samples)
    sample_face_idxs += mesh_to_face[mesh_valid_mask].view(num_valid_meshes, 1)  # (num_valid_meshes, num_samples)

    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces]  # (packed_n_triangle, 3triangle, 3xyz_w)
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]  # (packed_n_triangle, 3xyz_w)

    # Randomly generate barycentric coords.
    if barycentric_coords is None:
        w0, w1, w2 = rand_barycentric_coords(
            b=num_valid_meshes,
            n=num_samples,
            dtype=verts.dtype,
            device=verts.device,
        )  # (N=num_valid_meshes, num_samples)
    else:
        assert barycentric_coords.shape == (num_meshes, num_samples, 3)
        _tmp = barycentric_coords[mesh_valid_mask]  # (num_valid_mesh, num_samples, 3uvw)
        w0, w1, w2 = _tmp[..., 0], _tmp[..., 1], _tmp[..., 2]  # (num_valid_meshes, num_samples)

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (num_valid_meshes, num_samples, 3xyz_w)
    b = v1[sample_face_idxs]  # (num_valid_meshes, num_samples, 3xyz_w)
    c = v2[sample_face_idxs]  # (num_valid_meshes, num_samples, 3xyz_w)
    samples[mesh_valid_mask] = (w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c).to(dtype=samples.dtype)

    if return_normals:
        # Initialize normals tensor with fill value 0 for empty meshes.
        # Normals for the sampled points are face normals computed from
        # the vertices of the face in which the sampled point lies.
        normals = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)  # (num_meshes, num_samples, 3xyz_w)
        vert_normals = (v1 - v0).cross(v2 - v1, dim=1)  # (packed_n_triangle, 3xyz_w)
        vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(
            min=sys.float_info.epsilon
        )  # (packed_n_triangle, 3xyz_w)
        vert_normals = vert_normals[sample_face_idxs]  # (num_valid_meshes, num_samples, 3xyz_w)
        normals[mesh_valid_mask] = vert_normals.to(dtype=normals.dtype)  # (num_meshes, num_samples, 3xyz_w)
    else:
        normals = None

    if return_textures:
        # Initialize textures tensor with fill value 0 for empty meshes.
        textures = torch.zeros((num_meshes, num_samples, meshes.textures.maps_padded().shape[-1]), device=meshes.device)

        if num_valid_meshes > 0:
            # fragment data are of shape NxHxWxK. Here H=S, W=1 & K=1.
            pix_to_face = sample_face_idxs.view(num_valid_meshes, num_samples, 1, 1)  # (num_valid_mesh, nsample, 1, 1)
            bary = torch.stack((w0, w1, w2), dim=2).unsqueeze(2).unsqueeze(2)  # (num_valid_mesh, nsample, 1, 1, 3ijk)
            # zbuf and dists are not used in `sample_textures` so we initialize them with dummy
            dummy = torch.zeros(
                (num_valid_meshes, num_samples, 1, 1), device=meshes.device, dtype=torch.float32
            )  # (num_valid_mesehs, num_samples, 1, 1)
            fragments = MeshFragments(
                pix_to_face=pix_to_face,  # (num_valid_mesh, nsample, 1, 1)
                zbuf=dummy,  # (num_valid_mesehs, num_samples, 1, 1)
                bary_coords=bary,  # (num_valid_mesh, nsample, 1, 1, 3ijk)
                dists=dummy,  # (num_valid_mesehs, num_samples, 1, 1)
            )
            # Create a temporary Meshes object with only valid meshes for texture sampling
            valid_meshes = meshes[mesh_valid_mask]  # (num_valid_meshes,)
            sampled_textures = valid_meshes.sample_textures(fragments)  # (num_valid_meshes, numsamples, 1, 1, c)
            textures[mesh_valid_mask] = (sampled_textures[:, :, 0, 0, :]).to(dtype=textures.dtype)  # NxSxC
    else:
        textures = None

    if return_uvs and meshes.textures is not None:
        _textures = meshes.textures
        # Get UV coordinates (per-vertex-per-face, ie, the same vertex can have different uv in different triangles)
        verts_uvs = _textures.verts_uvs_padded()  # (num_meshes, max(Vt), 2uv)
        # Get indices of UVs for each face
        faces_uvs = _textures.faces_uvs_padded()  # shape: (num_meshes, max(F), 3)  long

        _num_meshes, maxF, _3 = faces_uvs.shape
        assert _num_meshes == num_meshes

        uv_per_face = torch.gather(
            verts_uvs,  # (num_meshes, max(Vt), 2)
            dim=1,
            index=faces_uvs.long()
            .reshape(_num_meshes, maxF * 3, 1)
            .expand(_num_meshes, maxF * 3, 2),  # (num_meshes, maxF*3, 2)
        ).reshape(_num_meshes, maxF, 3 * 2)  # (num_meshes, maxF, 3triangle * 2uv)

        packed_uv_per_face = pytorch3d.structures.utils.padded_to_packed(
            x=uv_per_face,  # (num_meshes, maxF, 3triangle * 2uv)
            split_size=num_triangles_per_mesh,  # (num_meshes,)
        )  # (packed_n_triangle, 3triangle * 2uv)
        packed_uv_per_face = packed_uv_per_face.reshape(
            packed_uv_per_face.size(0), 3, 2
        )  # (packed_n_triangle, 3triangle, 2uv)

        uv_1 = packed_uv_per_face[:, 0]  # (packed_n_triangle, 2uv)
        uv_2 = packed_uv_per_face[:, 1]  # (packed_n_triangle, 2uv)
        uv_3 = packed_uv_per_face[:, 2]  # (packed_n_triangle, 2uv)

        uv_a = uv_1[sample_face_idxs]  # (num_valid_meshes, num_samples, 2uv)
        uv_b = uv_2[sample_face_idxs]  # (num_valid_meshes, num_samples, 2uv)
        uv_c = uv_3[sample_face_idxs]  # (num_valid_meshes, num_samples, 2uv)

        # barycentric interpolation of uv
        uv = torch.zeros((num_meshes, num_samples, 2), device=meshes.device)  # (num_meshes, n, 2uv)
        uv[mesh_valid_mask] = (w0[:, :, None] * uv_a + w1[:, :, None] * uv_b + w2[:, :, None] * uv_c).to(dtype=uv.dtype)

        # quick test: uv sampling the texture and compare with the sampled texture should get same value
        # we assume one texture map for one mesh
        if debug and textures is not None:
            texture_map = meshes.textures.maps_padded()  # (num_meshes, h, w, c)
            # we need to flip the texture map along y
            # (pytorch3d underlying assumes opengl coordinate system but it stores texture and uv not flipped)
            est_texture = (
                torch.nn.functional.grid_sample(
                    input=torch.flip(texture_map, dims=[1]).permute(0, 3, 1, 2),  # (num_meshes, c, h, w)
                    grid=uv.unsqueeze(1) * 2 - 1,  # (num_meshes, 1, n, 2uv) [-1, 1]
                    mode=meshes.textures.sampling_mode,
                    align_corners=meshes.textures.align_corners,
                    padding_mode=meshes.textures.padding_mode,
                )
                .squeeze(-2)
                .permute(0, 2, 1)
            )  # (num_meshes, n, c)

            # border handling is different in actual uv mapping and
            assert torch.allclose(est_texture, textures, rtol=1e-4, atol=1e-4)

    else:
        uv = None

    return dict(
        xyz_w=samples,  # (num_meshes, num_samples, 3xyz_w)
        normal_w=normals,  # (num_meshes, num_samples, 3xyz_w)
        textures=textures,  # (num_meshes, num_samples, c)
        uv=uv,  # (num_meshes, num_samples, 2uv)
        mesh_valid_mask=mesh_valid_mask,  # (num_meshes,)
    )


def rand_barycentric_coords(
    b: int,
    n: int,
    dtype: torch.dtype,
    device: torch.device,
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function to generate random barycentric coordinates which are uniformly
    distributed over a triangle.

    Args:
        b, n: The number of coordinates generated will be (b, n).
                      Output tensors will each be of shape (b, n).
        dtype: Datatype to generate.
        device: A torch.device object on which the outputs will be allocated.

    Returns:
        w0, w1, w2: Tensors of shape (b, n) giving random barycentric
            coordinates

    Ref:
        pytorch3d
    """
    uv = torch.rand(2, b, n, dtype=dtype, device=device)
    u, v = uv[0], uv[1]  # (b, n)
    u_sqrt = u.sqrt()
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    return w0, w1, w2  # (b, n)


def _to_list(x: np.ndarray) -> T.List[float]:
    """
    Convert a NumPy array-like into a plain Python `list[float]`.

    Why this exists:
        - The analyzer returns JSON/pickle-friendly primitives only.
        - NumPy arrays (and NumPy scalar types) are picklable, but are not JSON-native
          and often get in the way when you want to dump results to JSON.

    Behavior:
        - Flattens `x` to 1D (row-major) via `ravel()`.
        - Casts each element to a Python `float`.

    Args:
        x:
            Any array-like accepted by `np.asarray`, commonly a NumPy array.

    Returns:
        list[float], A 1D list of Python floats.

    Notes
    -----
    - This will happily convert integers to floats.
    - If you pass in an object array or values that cannot be cast to float,
      you will get a `TypeError` or `ValueError`.
    """
    return [float(v) for v in np.asarray(x).ravel().tolist()]


def _stats(x: np.ndarray, percentiles: T.Tuple[int, ...] = (5, 50, 95)) -> T.Dict[str, float]:
    """
    Compute compact descriptive statistics for a 1D numeric array.

    Returned fields (when x is non-empty):
        - count: number of finite values considered
        - min/max/mean/std: basic distribution summary
        - cv: coefficient of variation = std / mean
        - p{K}: percentiles requested (e.g. p5, p50, p95)

    Interpretation tips:
        - `std` measures spread in the same units as the data.
        - `cv` (coefficient of variation) is unitless, making it good for comparing
          variability across different scales.
            * cv ~ 0: very uniform values
            * higher cv: more variation / unevenness
        - Percentiles help detect “long tails” without being as sensitive to outliers
          as max/min.

    Args:
        x:
            Array-like numeric input. Will be converted to float.
            Non-finite values (NaN/inf) are removed before computing stats.
    percentiles:
        Percentiles to compute (0..100). Defaults to (5, 50, 95).

    Returns:
        dict[str, float], Dictionary of stats. If no finite values are present, returns {"count": 0.0}.

    Edge cases
    ----------
    - If mean == 0, `cv` is set to `inf` (division by zero). That’s intentional:
      for some metrics (e.g., areas), mean==0 usually indicates a pathological input.
    """

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"count": 0.0}
    p = np.percentile(x, percentiles)
    out: T.Dict[str, float] = {
        "count": float(x.size),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }
    out["cv"] = float(out["std"] / out["mean"]) if out["mean"] != 0 else float("inf")
    for val, perc in zip(p, percentiles):
        out[f"p{perc}"] = float(val)
    return out


def _gini(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a nonnegative distribution.

    The Gini coefficient is a scalar inequality metric:
        - 0.0 means perfectly equal values (uniform distribution)
        - 1.0 means maximally unequal (one entry dominates)

    In this mesh analyzer, we use Gini on connected-component areas to quantify
    fragmentation:
        - Low Gini (~0): components are similar in area (many similar-sized parts)
        - High Gini (~1): one large component plus many tiny fragments (“debris”)

    Args:
        values:
            Array-like of values. Non-finite values are removed; negatives are clamped
            to 0 (since Gini is typically defined for nonnegative data).

    Returns:
        float, Gini coefficient in [0, 1] (for nonnegative inputs), or NaN if empty.

    Notes
    -----
    - If all values sum to 0 (e.g., all zeros), returns 0.0 (no inequality).
    - This implementation uses the common sorted-sum formula:
        G = 2*sum(i*v_i)/(n*sum(v)) - (n+1)/n
      with i = 1..n.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    v = np.clip(v, 0.0, None)
    s = v.sum()
    if s == 0:
        return 0.0
    v = np.sort(v)
    n = v.size
    i = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(i * v) / (n * s)) - ((n + 1.0) / n))


def _unique_edges_and_counts(faces: np.ndarray) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Return (unique_undirected_edges, face_use_counts_per_edge) for a triangular face array.

    faces: (F,3) int
    unique_edges: (E,2) int, each row sorted [min_vi, max_vi]
    counts: (E,) int, how many triangles reference that edge
    """
    f = np.asarray(faces, dtype=np.int64)
    e = np.vstack((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))  # (3F,2)
    e = np.sort(e, axis=1)  # undirected
    unique_edges, counts = np.unique(e, axis=0, return_counts=True)
    return unique_edges, counts


def analyze_mesh(
    filename: str,
    *,
    box_normalize: bool = True,
    triangulate: bool = True,
    degenerate_area_eps: float = 1e-18,
) -> T.Dict[str, T.Any]:
    """
    Analyze a PLY mesh for geometric and topological “health” and return only
    JSON/pickle-friendly primitives (dicts/lists of ints/floats/bools/strings/None).

    This function is meant to answer:
      - Is the tessellation reasonable (not full of slivers/degenerates)?
      - Is the topology consistent (manifold-ish, watertight when expected)?
      - Is the mesh “fragmented” into many disconnected pieces?
      - Does it *likely* shade smoothly (normals/dihedral behavior), as a proxy?

    Args:
        filename:
            Path to the mesh file.
        box_normalize:
            whether to box normalize the mesh before calculating the statistics
        triangulate:
            whether to triangulate the mesh if not triangle mesh.

    The output is organized into groups:

    --------------------------------------------------------------------------
    basic
    --------------------------------------------------------------------------
    - num_vertices: Vertex count.
    - num_faces / num_triangles: Face count (triangles after optional triangulation).

    Interpretation:
      * Huge triangle counts aren’t “bad”; they just affect performance.
      * Very low triangle counts on curved objects often implies faceting.

    --------------------------------------------------------------------------
    triangles
    --------------------------------------------------------------------------
    - total_area:
        Sum of per-triangle areas.
        Useful for normalization and sanity checks.

    - mean_triangle_area:
        Average triangle area.

    - total_area_over_mean_triangle_area:
        Defined as total_area / mean_triangle_area.
        This behaves like an “effective triangle count”:
          * If all triangle areas are finite and > 0, this should be extremely close
            to num_triangles (up to floating error).
          * If it is noticeably off, it usually means some areas are zero/near-zero
            (degenerate triangles) or invalid (NaNs/infs).

    - degenerate_triangles:
        Count of triangles whose area <= degenerate_area_eps.
        Degenerate triangles often cause:
          * shading artifacts
          * numerical instability
          * failures in downstream geometry processing

    - area_stats:
        Summary statistics of triangle areas: min/max/mean/std/CV and percentiles.

        Key field: area_stats["cv"] = std / mean (coefficient of variation)
          * Low CV (e.g. < ~0.5) suggests more uniform triangle areas
          * High CV suggests a mix of huge triangles and tiny triangles
            (common in sloppy decimation, booleans, remeshing artifacts)

    --------------------------------------------------------------------------
    edges
    --------------------------------------------------------------------------
    - num_unique_edges:
        Count of undirected unique edges in the triangle soup.

    - edge_length_stats:
        Statistics over unique edge lengths.

        Interpreting edge_length_stats["cv"]:
          * Lower CV => more uniform edge lengths (often a “nice” triangulation)
          * High CV => highly irregular tessellation (could be okay, but often a sign
            of uneven remeshing or “micro triangles” clustered in some regions)

    - boundary_edges:
        Count of edges referenced by exactly 1 triangle.
        Interpretation:
          * Boundary edges indicate an open surface / holes.
          * For a mesh expected to be a closed solid, boundary_edges > 0 is a red flag.

    - nonmanifold_edges:
        Count of edges referenced by more than 2 triangles.
        Interpretation:
          * Non-manifold edges can break many algorithms (subdivision, boolean ops,
            normal computation, physics).
          * If nonmanifold_edges > 0, this is frequently “needs repair.”

    - edges_with_face_count_not_2:
        Count of edges that are not used by exactly two triangles.
        Interpretation:
          * For a clean, closed 2-manifold surface: this should be 0.
          * For open surfaces: boundary edges will contribute to this number.

    --------------------------------------------------------------------------
    triangle_quality
    --------------------------------------------------------------------------
    - aspect_proxy_stats:
        Statistics of a triangle “slenderness” proxy.
        In the provided implementation:
          aspect_proxy = (longest_edge) / (2 * inradius)
        where inradius = 2 * area / perimeter.

        Interpretation:
          * Equilateral triangles yield aspect_proxy ≈ 1.732 (baseline “good”)
          * Higher values indicate skinnier/sliver triangles.
          * Large p95/p99 (e.g. > 10–50 depending on domain) usually means many slivers.

    - min_angle_deg_stats:
        Statistics of per-triangle minimum interior angle in degrees.
        Interpretation:
          * Small angles are the classic signature of sliver triangles.
          * Watch p1 / p5:
              - if p5 < ~10°, that’s often “mesh quality problem”
              - if p1 is extremely tiny (< 1°), you almost certainly have slivers

    --------------------------------------------------------------------------
    topology
    --------------------------------------------------------------------------
    - watertight:
        Trimesh’s watertight flag (closed surface with consistent adjacency).
        Interpretation:
          * True is great for “solid” objects
          * False is fine for open surfaces, but suspicious for “closed” assets

    - euler_number (optional):
        Euler characteristic (depends on library support).
        Interpretation:
          * Only really meaningful for manifold surfaces; can be misleading otherwise.

    --------------------------------------------------------------------------
    smoothness_proxies
    --------------------------------------------------------------------------
    These are heuristics — they do not “prove smoothness,” but they correlate with
    common shading/geometry issues.

    - dihedral_angle_deg_stats (optional):
        Distribution stats of dihedral angles between adjacent triangle faces.
        Interpretation (rough intuition):
          * Angles near 180° => faces nearly coplanar => smooth locally
          * Lots of low angles => sharp creases or noisy/irregular surface

        Caveat:
          * Hard-surface models legitimately have many sharp edges — low angles are
            not inherently “bad.”

    - normal_consistency_fraction (optional):
        Fraction of adjacent face pairs whose normals point in roughly the same direction.
        Interpretation:
          * Near 1.0 => consistent orientation
          * Low values => likely flipped triangles / inconsistent winding / messy topology

    --------------------------------------------------------------------------
    components
    --------------------------------------------------------------------------
    Connected components are computed by splitting the mesh into disconnected pieces.

    - num_components:
        Number of disconnected triangle sets.

    - component_areas:
        Area of each component.

    - component_triangle_counts:
        Triangle count of each component.

    - sum_component_area:
        Sum of component areas (should match total_area closely).

    - total_area_over_sum_component_area:
        Sanity check ratio ~= 1.0 (small floating error is normal).
        If far from 1, something is inconsistent (precision, triangulation, or bug).

    - sum_component_area_over_mean_component_area:
        Analogue of (total_area / mean_triangle_area), but for components.
        Interpretation:
          * Equals num_components if component areas are uniform.
          * Becomes smaller than num_components when one component dominates.

    - largest_component_area_fraction:
        max(component_area) / sum_component_area.
        Interpretation:
          * Near 1.0 => one main piece
          * Small => many similar-sized parts, or lots of fragments

    - component_area_stats["cv"]:
        Coefficient of variation of component areas.
        Interpretation:
          * Near 0 => components are similarly sized
          * High => one big component plus many tiny ones (common “debris” pattern)

    - component_area_entropy_normalized:
        Normalized Shannon entropy of component area fractions in [0, 1] (when >1 component).
        Interpretation:
          * Near 0 => one component dominates area (plus tiny scraps)
          * Near 1 => areas are evenly spread across many components

    --------------------------------------------------------------------------
    notes
    --------------------------------------------------------------------------
    Human-readable warnings and hints. These are not exhaustive; they’re there to
    highlight common failure modes (degenerates, non-manifold edges, fragmentation).

    Returns:
        dict: JSON/pickle-friendly nested data structure containing all metrics.

    Practical tips:
      - For “well-made” triangulation, focus on:
          min_angle percentiles, aspect_proxy percentiles, edge length CV, area CV.
      - For “clean topology,” focus on:
          nonmanifold_edges, boundary_edges, watertight.
      - For “fragmentation / junk pieces,” focus on:
          num_components, largest_component_area_fraction, component entropy, component CV.
    """
    mesh = trimesh.load(filename, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError(f"No geometry found in file: {filename}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Loaded object is not a trimesh.Trimesh: {type(mesh)}")

    if triangulate and (mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3):
        mesh = mesh.triangulate()

    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError("Mesh is not triangular (set triangulate=True or triangulate externally).")

    if box_normalize:
        # Axis-aligned bounds: [min(x,y,z), max(x,y,z)]
        bmin, bmax = mesh.bounds  # (3,)
        center = (bmin + bmax) / 2.0  # (3,)
        extents = bmax - bmin  # (3,)
        max_extent = float(extents.max())  # float

        # Avoid divide-by-zero for degenerate meshes
        if max_extent < 1e-6:
            raise ValueError("Mesh has near-zero size; cannot normalize.")

        # Translate so AABB center -> origin
        mesh.apply_translation(-center)

        # Scale so longest AABB side becomes 2 (i.e., fits in [-1,1])
        scale = 2.0 / max_extent
        mesh.apply_scale(scale)

    v = np.asarray(mesh.vertices, dtype=float)
    f = np.asarray(mesh.faces, dtype=np.int64)

    num_vertices = int(len(v))
    num_faces = int(len(f))
    num_triangles = num_faces

    # --- Triangle areas ---
    tri_areas = np.asarray(mesh.area_faces, dtype=float)
    total_area = float(np.sum(tri_areas))
    mean_area = float(np.mean(tri_areas)) if tri_areas.size else 0.0
    total_over_mean = float(total_area / mean_area) if mean_area > 0 else float("inf")
    degenerate_triangles = int(np.sum(tri_areas <= degenerate_area_eps))

    # --- Unique edges + counts (version-proof) ---
    edges_unique, edge_face_count = _unique_edges_and_counts(f)

    # boundary/nonmanifold from counts
    boundary_edges = int(np.sum(edge_face_count == 1))
    nonmanifold_edges = int(np.sum(edge_face_count > 2))

    # edge lengths from unique edges
    e0 = v[edges_unique[:, 0]]
    e1 = v[edges_unique[:, 1]]
    edge_lengths = np.linalg.norm(e1 - e0, axis=1)

    # --- Triangle shape: aspect proxy + min angle ---
    a = np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1)
    b = np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=1)
    c = np.linalg.norm(v[f[:, 0]] - v[f[:, 2]], axis=1)
    perim = a + b + c
    longest = np.maximum(np.maximum(a, b), c)

    r = 2.0 * tri_areas / np.maximum(perim, 1e-30)  # inradius
    aspect_proxy = longest / np.maximum(2.0 * r, 1e-30)  # equilateral ~1.732

    def safe_acos(x: np.ndarray) -> np.ndarray:
        return np.arccos(np.clip(x, -1.0, 1.0))

    cosA = (b * b + c * c - a * a) / np.maximum(2.0 * b * c, 1e-30)
    cosB = (c * c + a * a - b * b) / np.maximum(2.0 * c * a, 1e-30)
    cosC = (a * a + b * b - c * c) / np.maximum(2.0 * a * b, 1e-30)
    A = safe_acos(cosA)
    B = safe_acos(cosB)
    C = safe_acos(cosC)
    min_angle_deg = np.minimum(np.minimum(A, B), C) * (180.0 / np.pi)

    # --- Topology health (watertight + euler if available) ---
    watertight = bool(getattr(mesh, "is_watertight", False))
    euler_number: T.Optional[int] = int(mesh.euler_number) if hasattr(mesh, "euler_number") else None

    # --- Smoothness proxies (optional; depends on trimesh version/availability) ---
    dihedral_stats_deg: T.Optional[T.Dict[str, float]] = None
    try:
        ang = getattr(mesh, "face_adjacency_angles", None)
        if ang is not None and len(ang) > 0:
            dihedral_deg = np.asarray(ang, dtype=float) * (180.0 / np.pi)
            dihedral_stats_deg = _stats(dihedral_deg, percentiles=(5, 50, 95))
    except Exception:
        dihedral_stats_deg = None

    normal_consistency_fraction: T.Optional[float] = None
    try:
        fn = np.asarray(mesh.face_normals, dtype=float)
        adj = getattr(mesh, "face_adjacency", None)
        if adj is not None and len(adj) > 0:
            adj = np.asarray(adj, dtype=np.int64)
            dots = np.einsum("ij,ij->i", fn[adj[:, 0]], fn[adj[:, 1]])
            normal_consistency_fraction = float(np.mean(dots > 0.0))
    except Exception:
        normal_consistency_fraction = None

    # --- Connected components ---
    comps = mesh.split(only_watertight=False)
    num_components = int(len(comps))

    comp_areas = np.array([float(c.area) for c in comps], dtype=float) if num_components else np.array([], dtype=float)
    comp_tris = (
        np.array([int(len(c.faces)) for c in comps], dtype=float) if num_components else np.array([], dtype=float)
    )

    sum_comp_area = float(comp_areas.sum()) if comp_areas.size else 0.0
    mean_comp_area = float(comp_areas.mean()) if comp_areas.size else 0.0
    sum_comp_area_over_mean = float(sum_comp_area / mean_comp_area) if mean_comp_area > 0 else float("inf")

    comp_area_fractions = (comp_areas / sum_comp_area) if sum_comp_area > 0 else np.zeros_like(comp_areas)
    largest_component_area_fraction = float(comp_area_fractions.max()) if comp_area_fractions.size else float("nan")

    entropy_norm = float("nan")
    if comp_area_fractions.size > 0 and sum_comp_area > 0:
        p = comp_area_fractions[comp_area_fractions > 0]
        H = -float(np.sum(p * np.log(p)))
        Hmax = float(np.log(len(comp_area_fractions))) if len(comp_area_fractions) > 1 else 1.0
        entropy_norm = float(H / Hmax) if Hmax > 0 else 0.0

    notes: T.Dict[str, T.Any] = {}
    if degenerate_triangles > 0:
        notes["degenerate_warning"] = (
            f"{degenerate_triangles} triangles have near-zero area (eps={degenerate_area_eps})."
        )
    if nonmanifold_edges > 0:
        notes["nonmanifold_warning"] = f"{nonmanifold_edges} edges look non-manifold (>2 adjacent faces)."
    if num_components > 1:
        notes["component_note"] = (
            "Multiple connected components: could be intentional (separate parts) or an export issue."
        )

    # The “total area over sum(component areas)” sanity check; should be ~1
    total_over_sum_comp = float(total_area / sum_comp_area) if sum_comp_area > 0 else float("inf")

    return {
        "filename": filename,
        "basic": {
            "num_vertices": num_vertices,
            "num_faces": num_faces,
            "num_triangles": num_triangles,
        },
        "triangles": {
            "total_area": total_area,
            "mean_triangle_area": mean_area,
            "total_area_over_mean_triangle_area": total_over_mean,
            "degenerate_triangles": degenerate_triangles,
            "area_stats": _stats(tri_areas, percentiles=(5, 50, 95)),
        },
        "edges": {
            "num_unique_edges": int(len(edges_unique)),
            "edge_length_stats": _stats(edge_lengths, percentiles=(5, 50, 95)),
            "boundary_edges": boundary_edges,
            "nonmanifold_edges": nonmanifold_edges,
            # extra: how many edges are “odd” for a manifold surface
            "edges_with_face_count_not_2": int(np.sum(edge_face_count != 2)),
        },
        "triangle_quality": {
            "aspect_proxy_stats": _stats(aspect_proxy, percentiles=(50, 95, 99)),
            "min_angle_deg_stats": _stats(min_angle_deg, percentiles=(1, 5, 50)),
        },
        "topology": {
            "watertight": watertight,
            "euler_number": euler_number,
        },
        "smoothness_proxies": {
            "dihedral_angle_deg_stats": dihedral_stats_deg,
            "normal_consistency_fraction": normal_consistency_fraction,
        },
        "components": {
            "num_components": num_components,
            "component_areas": _to_list(comp_areas),
            "component_triangle_counts": [int(x) for x in comp_tris.tolist()],
            "sum_component_area": sum_comp_area,
            # sanity check: ~1
            "total_area_over_sum_component_area": total_over_sum_comp,
            # informative analogues
            "sum_component_area_over_mean_component_area": sum_comp_area_over_mean,
            "largest_component_area_fraction": largest_component_area_fraction,
            "component_area_stats": _stats(comp_areas, percentiles=(5, 50, 95)),
            "component_triangle_stats": _stats(comp_tris, percentiles=(5, 50, 95)),
            "component_area_gini": _gini(comp_areas),
            "component_area_entropy_normalized": entropy_norm,
        },
        "notes": notes,
    }


def analyze_mesh_simple(
    filename: str,
    box_normalize: bool = True,
) -> T.Dict[str, T.Any]:
    """
    Analyze a PLY mesh for geometric and topological “health” and return only
    JSON/pickle-friendly primitives (dicts/lists of ints/floats/bools/strings/None).

    This function is meant to answer:
      - Is the tessellation reasonable (not full of slivers/degenerates)?
      - Is the topology consistent (manifold-ish, watertight when expected)?
      - Is the mesh “fragmented” into many disconnected pieces?
      - Does it *likely* shade smoothly (normals/dihedral behavior), as a proxy?

    Args:
        filename:
            Path to the mesh file.
        box_normalize:
            whether to box normalize the mesh before calculating the statistics
        triangulate:
            whether to triangulate the mesh if not triangle mesh.

    The output is organized into groups:

    --------------------------------------------------------------------------
    basic
    --------------------------------------------------------------------------
    - num_vertices: Vertex count.
    - num_faces / num_triangles: Face count (triangles after optional triangulation).

    Interpretation:
      * Huge triangle counts aren’t “bad”; they just affect performance.
      * Very low triangle counts on curved objects often implies faceting.

    --------------------------------------------------------------------------
    triangles
    --------------------------------------------------------------------------
    - total_area:
        Sum of per-triangle areas.
        Useful for normalization and sanity checks.

    - mean_triangle_area:
        Average triangle area.

    - total_area_over_mean_triangle_area:
        Defined as total_area / mean_triangle_area.
        This behaves like an “effective triangle count”:
          * If all triangle areas are finite and > 0, this should be extremely close
            to num_triangles (up to floating error).
          * If it is noticeably off, it usually means some areas are zero/near-zero
            (degenerate triangles) or invalid (NaNs/infs).

    - degenerate_triangles:
        Count of triangles whose area <= degenerate_area_eps.
        Degenerate triangles often cause:
          * shading artifacts
          * numerical instability
          * failures in downstream geometry processing

    - area_stats:
        Summary statistics of triangle areas: min/max/mean/std/CV and percentiles.

        Key field: area_stats["cv"] = std / mean (coefficient of variation)
          * Low CV (e.g. < ~0.5) suggests more uniform triangle areas
          * High CV suggests a mix of huge triangles and tiny triangles
            (common in sloppy decimation, booleans, remeshing artifacts)

    --------------------------------------------------------------------------
    edges
    --------------------------------------------------------------------------
    - num_unique_edges:
        Count of undirected unique edges in the triangle soup.

    - edge_length_stats:
        Statistics over unique edge lengths.

        Interpreting edge_length_stats["cv"]:
          * Lower CV => more uniform edge lengths (often a “nice” triangulation)
          * High CV => highly irregular tessellation (could be okay, but often a sign
            of uneven remeshing or “micro triangles” clustered in some regions)

    - boundary_edges:
        Count of edges referenced by exactly 1 triangle.
        Interpretation:
          * Boundary edges indicate an open surface / holes.
          * For a mesh expected to be a closed solid, boundary_edges > 0 is a red flag.

    - nonmanifold_edges:
        Count of edges referenced by more than 2 triangles.
        Interpretation:
          * Non-manifold edges can break many algorithms (subdivision, boolean ops,
            normal computation, physics).
          * If nonmanifold_edges > 0, this is frequently “needs repair.”

    - edges_with_face_count_not_2:
        Count of edges that are not used by exactly two triangles.
        Interpretation:
          * For a clean, closed 2-manifold surface: this should be 0.
          * For open surfaces: boundary edges will contribute to this number.

    --------------------------------------------------------------------------
    triangle_quality
    --------------------------------------------------------------------------
    - aspect_proxy_stats:
        Statistics of a triangle “slenderness” proxy.
        In the provided implementation:
          aspect_proxy = (longest_edge) / (2 * inradius)
        where inradius = 2 * area / perimeter.

        Interpretation:
          * Equilateral triangles yield aspect_proxy ≈ 1.732 (baseline “good”)
          * Higher values indicate skinnier/sliver triangles.
          * Large p95/p99 (e.g. > 10–50 depending on domain) usually means many slivers.

    - min_angle_deg_stats:
        Statistics of per-triangle minimum interior angle in degrees.
        Interpretation:
          * Small angles are the classic signature of sliver triangles.
          * Watch p1 / p5:
              - if p5 < ~10°, that’s often “mesh quality problem”
              - if p1 is extremely tiny (< 1°), you almost certainly have slivers

    --------------------------------------------------------------------------
    topology
    --------------------------------------------------------------------------
    - watertight:
        Trimesh’s watertight flag (closed surface with consistent adjacency).
        Interpretation:
          * True is great for “solid” objects
          * False is fine for open surfaces, but suspicious for “closed” assets

    - euler_number (optional):
        Euler characteristic (depends on library support).
        Interpretation:
          * Only really meaningful for manifold surfaces; can be misleading otherwise.

    --------------------------------------------------------------------------
    smoothness_proxies
    --------------------------------------------------------------------------
    These are heuristics — they do not “prove smoothness,” but they correlate with
    common shading/geometry issues.

    - dihedral_angle_deg_stats (optional):
        Distribution stats of dihedral angles between adjacent triangle faces.
        Interpretation (rough intuition):
          * Angles near 180° => faces nearly coplanar => smooth locally
          * Lots of low angles => sharp creases or noisy/irregular surface

        Caveat:
          * Hard-surface models legitimately have many sharp edges — low angles are
            not inherently “bad.”

    - normal_consistency_fraction (optional):
        Fraction of adjacent face pairs whose normals point in roughly the same direction.
        Interpretation:
          * Near 1.0 => consistent orientation
          * Low values => likely flipped triangles / inconsistent winding / messy topology

    --------------------------------------------------------------------------
    components
    --------------------------------------------------------------------------
    Connected components are computed by splitting the mesh into disconnected pieces.

    - num_components:
        Number of disconnected triangle sets.

    - component_areas:
        Area of each component.

    - component_triangle_counts:
        Triangle count of each component.

    - sum_component_area:
        Sum of component areas (should match total_area closely).

    - total_area_over_sum_component_area:
        Sanity check ratio ~= 1.0 (small floating error is normal).
        If far from 1, something is inconsistent (precision, triangulation, or bug).

    - sum_component_area_over_mean_component_area:
        Analogue of (total_area / mean_triangle_area), but for components.
        Interpretation:
          * Equals num_components if component areas are uniform.
          * Becomes smaller than num_components when one component dominates.

    - largest_component_area_fraction:
        max(component_area) / sum_component_area.
        Interpretation:
          * Near 1.0 => one main piece
          * Small => many similar-sized parts, or lots of fragments

    - component_area_stats["cv"]:
        Coefficient of variation of component areas.
        Interpretation:
          * Near 0 => components are similarly sized
          * High => one big component plus many tiny ones (common “debris” pattern)

    - component_area_entropy_normalized:
        Normalized Shannon entropy of component area fractions in [0, 1] (when >1 component).
        Interpretation:
          * Near 0 => one component dominates area (plus tiny scraps)
          * Near 1 => areas are evenly spread across many components

    --------------------------------------------------------------------------
    notes
    --------------------------------------------------------------------------
    Human-readable warnings and hints. These are not exhaustive; they’re there to
    highlight common failure modes (degenerates, non-manifold edges, fragmentation).

    Returns:
        dict: JSON/pickle-friendly nested data structure containing all metrics.

    Practical tips:
      - For “well-made” triangulation, focus on:
          min_angle percentiles, aspect_proxy percentiles, edge length CV, area CV.
      - For “clean topology,” focus on:
          nonmanifold_edges, boundary_edges, watertight.
      - For “fragmentation / junk pieces,” focus on:
          num_components, largest_component_area_fraction, component entropy, component CV.
    """
    mesh = o3d.io.read_triangle_mesh(filename)

    if mesh.is_empty():
        raise ValueError("Mesh is empty.")

    if box_normalize:
        # 1) AABB
        aabb = mesh.get_axis_aligned_bounding_box()
        center = aabb.get_center()  # (3,)
        extent = aabb.get_extent()  # (dx, dy, dz)

        max_extent = float(np.max(extent))
        if max_extent <= 0:
            raise ValueError("Degenerate mesh (zero extent).")

        # 2) center at origin
        mesh.translate(-center)

        # 3) scale to fit inside [-1, 1]^3 (largest side becomes 2)
        scale = 2.0 / max_extent
        mesh.scale(scale, center=(0.0, 0.0, 0.0))

    # --- counts ---
    V = np.asarray(mesh.vertices)  # (nV, 3)
    F = np.asarray(mesh.triangles)  # (nF, 3) int indices

    n_vertices = V.shape[0]
    n_triangles = F.shape[0]

    # Total area (Open3D built-in)
    total_area = float(mesh.get_surface_area())  # available on legacy TriangleMesh

    # Per-triangle areas (manual)
    if n_triangles == 0:
        tri_areas = np.array([], dtype=np.float64)
        avg_triangle_area = 0.0
        ratio_avg_over_total = 0.0
    else:
        tri = V[F]  # (nF, 3, 3): triangle vertices a,b,c
        a = tri[:, 0, :]
        b = tri[:, 1, :]
        c = tri[:, 2, :]
        tri_areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

        avg_triangle_area = float(tri_areas.mean())
        ratio_avg_over_total = (avg_triangle_area / total_area) if total_area > 0 else 0.0

    return {
        "n_vertices": n_vertices,
        "n_triangles": n_triangles,
        "total_area": total_area,
        "avg_triangle_area": avg_triangle_area,
        "avg_triangle_area_over_total_area": ratio_avg_over_total,
    }


def _shannon_entropy_from_counts(counts: np.ndarray, base: float = 2.0) -> float:
    """
    Shannon entropy of a discrete distribution given histogram counts.

    counts: (K,) nonnegative
    base: 2 -> bits, np.e -> nats
    """
    total = float(counts.sum())
    if total <= 0:
        return 0.0

    p = counts.astype(np.float64) / total  # (K,)
    p = p[p > 0]
    if p.size == 0:
        return 0.0

    H = -(p * np.log(p)).sum()  # nats
    if base != np.e:
        H /= np.log(base)
    return float(H)


def _weighted_mean(x: np.ndarray, w: np.ndarray, eps: float = 1e-12) -> float:
    """
    Weighted mean.

    x: (...,) float
    w: same shape as x
    returns: scalar
    """
    return float((x * w).sum() / (w.sum() + eps))


def _hann2d(n: int) -> np.ndarray:
    """
    2D Hann window.

    returns: (n, n) float32
    """
    h = np.hanning(n).astype(np.float32)  # (n,)
    return (h[:, None] * h[None, :]).astype(np.float32)  # (n, n)


def _bbox_from_mask(mask: np.ndarray) -> T.Optional[T.Tuple[int, int, int, int]]:
    """
    Compute tight bbox from a boolean mask.

    mask: (H, W) bool
    returns: (x0, y0, x1, y1) or None if empty
    """
    if not np.any(mask):
        return None
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return (x0, y0, x1, y1)


def analyze_texture_metrics_np(
    rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    size: int = 512,  # image resize target
    alpha_threshold: float = 0.5,
    grad_bins: int = 256,
    entropy_base: float = 2.0,
    hf_cut: float = 0.7,
    crop_foreground: bool = False,
    pad_to_square: bool = True,
    use_hann_window: bool = True,
) -> T.Dict[str, T.Any]:
    """
    Compute grayscale entropy, gradient entropy, and FFT-based spectral metrics
    for an RGBA-like image provided as numpy arrays.

    Inputs (REQUIRED):
      rgb:   (H, W, 3) float in [0, 1]
      alpha: (H, W, 1) float in [0, 1]   (transparency mask)

    Pipeline:
      1) Convert RGB -> grayscale luminance (NO premultiplication by alpha)
      2) Crop to foreground bbox (alpha > alpha_threshold),
         pad with black to square, resize to (size, size) using OpenCV (float-preserving)
      3) Compute grayscale entropy on resized image (INTERIOR foreground selection)
      4) Compute gradient-magnitude entropy on resized image (INTERIOR foreground selection)
      5) Compute FFT on resized image using:
           - alpha interior weights (after erosion)  [reduces silhouette-edge cheating]
           - mean subtraction (weighted)
           - optional Hann window
         Then compute:
           - spectral_entropy (on FFT coefficients, DC excluded)
           - spectral_centroid (cycles/pixel)
           - HF ratio using cutoff as a fraction of Nyquist (0.5 cycles/pixel)

    Shapes (conceptual):
      - Inputs:
          rgb:   (H, W, 3)
          alpha: (H, W, 1)
      - After squeeze:
          a:     (H, W)
      - Grayscale:
          gray:  (H, W)
      - After optional crop:
          gray:  (Hc, Wc)
          a:     (Hc, Wc)
      - After optional pad-to-square:
          gray:  (S, S),  a: (S, S) where S = max(Hc, Wc)
      - After resize:
          gray_r:   (size, size)
          alpha_r:  (size, size)

    FFT shapes:
      s:  (size, size) analysis signal
      F:  (size, size) complex FFT (shifted)
      P:  (size, size) power spectrum
      fr: (size, size) radial frequency (cycles/pixel)

    Important behavior fixes incorporated:
      - Avoid alpha^2 bias: grayscale is NOT premultiplied by alpha; alpha is used only for selection/weighting.
      - Mask erosion: use an eroded interior mask for entropy, gradient entropy, and FFT weights.
      - Float-preserving resize: OpenCV resize on float32 (reduces quantization artifacts).
      - HF cutoff uses Nyquist=0.5 cycles/pixel: hf_cutoff = hf_cut * 0.5 (hf_cut in (0,1)).

    Returns:
      A dict with the following keys (all scalar values unless noted):

      Bookkeeping / debug:
        - "size" (int):
            Target resize dimension (size x size).
        - "alpha_threshold" (float):
            Threshold used to define foreground (alpha > threshold).
        - "bbox_xyxy" (Optional[Tuple[int,int,int,int]]):
            Foreground bounding box in ORIGINAL coordinates (x0, y0, x1, y1),
            or None if no foreground found or crop_foreground=False.
        - "foreground_fraction_resized" (float):
            Fraction of pixels considered interior foreground after resize+erosion.
            Very small values mean your object is tiny in the standardized frame,
            which can make metrics less stable.
        - "erode_radius_px" (int):
            Erosion radius (in pixels) used to remove silhouette edge effects.
        - "total_power" (float):
            Sum of FFT power (excluding DC). Near-zero usually means the image is
            almost perfectly uniform within the analyzed region.

      Spatial-domain entropies (units depend on entropy_base; default is bits):
        - "gray_entropy" (float):
            Shannon entropy of grayscale values within the interior foreground.
            Higher -> more diverse intensity values (e.g., multi-tone shading or varied colors
            after grayscale conversion). Note: A smooth gradient can raise this even if texture is low.
        - "gradient_entropy" (float):
            Shannon entropy of gradient magnitudes within the interior foreground.
            Higher -> more diverse edge/texture strengths (often correlates well with “texturedness”).
            Lower -> mostly flat regions or very uniform edge strength.

      Frequency-domain metrics:
        - "spectral_entropy" (float):
            Entropy of the normalized FFT power distribution across frequency coefficients (DC excluded).
            Higher -> power spread across many frequencies (busy/noisy/irregular texture).
            Lower -> power concentrated in fewer frequencies (smooth images or very regular patterns).
        - "spectral_centroid_cyc_per_px" (float):
            Power-weighted average radial frequency in cycles/pixel.
            Higher -> energy shifted toward finer detail (smaller-scale texture).
            Lower -> energy dominated by coarse structure / smooth variation.
            Typical range is ~0 to ~0.35; values near 0.5 are unusual and suggest very sharp/noisy content.
        - "hf_ratio" (float):
            Fraction of total power contained above the high-frequency cutoff.
            Higher -> more high-frequency content (fine detail / sharp edges / noise).
            Lower -> smoother / less fine detail.
        - "hf_cut" (float):
            The user-specified fraction of Nyquist used to define high-frequency.
        - "hf_cutoff_cyc_per_px" (float):
            Actual high-frequency cutoff in cycles/pixel: hf_cut * 0.5.

    How to interpret the outputs together (rules of thumb):
      - High hf_ratio + high spectral_centroid:
          Lots of fine detail (could be real texture or noise). Check gradient_entropy to distinguish:
            * if gradient_entropy is also high -> likely real textured detail
            * if gradient_entropy is high but gray_entropy is low -> could be sharp binary patterns
      - High spectral_entropy:
          Broadband texture (many scales / irregular surfaces / noise). Regular patterns may have
          high hf_ratio but lower spectral_entropy (energy concentrated in peaks).
      - High gray_entropy but low gradient_entropy and low hf_ratio:
          Likely smooth shading/gradients, not “texture”.
      - Very low foreground_fraction_resized:
          Object occupies little area; metrics can be dominated by resizing artifacts or padding.
          Consider increasing crop padding, analyzing at multiple sizes, or rejecting tiny objects.

    Notes:
      - For comparing many images, keep size, hf_cut, alpha_threshold constant.
      - If you change size, you change what “high frequency” means relative to object scale.
    """
    # ----------------------------
    # Validate inputs + normalize
    # ----------------------------
    if not isinstance(rgb, np.ndarray) or not isinstance(alpha, np.ndarray):
        raise TypeError("rgb and alpha must be numpy arrays")

    if rgb.ndim != 3 or (rgb.shape[-1] not in [1, 3]):
        raise ValueError(f"rgb must have shape (H, W, 1/3). Got {rgb.shape}")

    if alpha.ndim != 3 or alpha.shape[-1] != 1:
        raise ValueError(f"alpha must have shape (H, W, 1). Got {alpha.shape}")

    if rgb.shape[:2] != alpha.shape[:2]:
        raise ValueError(f"rgb and alpha spatial shapes must match. Got {rgb.shape[:2]} vs {alpha.shape[:2]}")

    if size <= 8:
        raise ValueError("size should be reasonably large (e.g., 64, 128, 256, 512)")
    if not (0.0 < alpha_threshold < 1.0):
        raise ValueError("alpha_threshold must be in (0, 1)")
    if not (0.0 < hf_cut < 1.0):
        raise ValueError("hf_cut must be in (0, 1)")
    if grad_bins < 8:
        raise ValueError("grad_bins should be >= 8")

    rgb = np.asarray(rgb, dtype=np.float32)  # (H, W, 3)
    a = np.asarray(alpha, dtype=np.float32)[..., 0]  # (H, W)

    if np.isnan(rgb).any() or np.isnan(a).any():
        raise ValueError("rgb/alpha contain NaNs")

    rgb = np.clip(rgb, 0.0, 1.0)  # (H, W, 3)
    a = np.clip(a, 0.0, 1.0)  # (H, W)

    # ----------------------------
    # 1) RGB -> grayscale luminance (no premultiply)
    # ----------------------------
    # gray: (H, W)
    if rgb.shape[-1] == 3:
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]  # (h, w)
    else:
        gray = np.copy(rgb[:, :, 0])  # (h, w)

    # ----------------------------
    # 2) Crop bbox, pad square, resize (OpenCV float-preserving)
    # ----------------------------
    bbox_xyxy: T.Optional[T.Tuple[int, int, int, int]] = None

    fg_mask = a > alpha_threshold  # (H, W) bool
    if crop_foreground:
        bbox_xyxy = _bbox_from_mask(fg_mask)
        if bbox_xyxy is not None:
            x0, y0, x1, y1 = bbox_xyxy
            # gray/a: (Hc, Wc)
            gray = gray[y0:y1, x0:x1]
            a = a[y0:y1, x0:x1]
        else:
            # no foreground found; keep as-is but report bbox None
            bbox_xyxy = None

    if pad_to_square:
        # gray: (Hc, Wc) -> (S, S)
        h, w = gray.shape
        side = max(h, w)

        gray_sq = np.zeros((side, side), dtype=np.float32)  # (S, S) black
        a_sq = np.zeros((side, side), dtype=np.float32)  # (S, S) transparent

        yoff = (side - h) // 2
        xoff = (side - w) // 2

        gray_sq[yoff : yoff + h, xoff : xoff + w] = gray
        a_sq[yoff : yoff + h, xoff : xoff + w] = a
        gray, a = gray_sq, a_sq

    # Resize in float32 using OpenCV (Lanczos)
    # gray_r/alpha_r: (size, size)
    gray_r = cv2.resize(gray.astype(np.float32), (size, size), interpolation=cv2.INTER_LANCZOS4)
    alpha_r = cv2.resize(a.astype(np.float32), (size, size), interpolation=cv2.INTER_LANCZOS4)

    gray_r = np.clip(gray_r, 0.0, 1.0).astype(np.float32)  # (size, size)
    alpha_r = np.clip(alpha_r, 0.0, 1.0).astype(np.float32)  # (size, size)

    # ----------------------------
    # Interior mask via erosion (scale-aware)
    # ----------------------------
    # fg_r: (size, size) bool
    fg_r = alpha_r > alpha_threshold

    # Erosion radius (pixels) chosen as a small fraction of size.
    # For size=512 -> ~8 px, size=256 -> ~4 px, size=128 -> ~2 px, etc.
    erode_radius = max(1, int(round(size / 64)))
    k = 2 * erode_radius + 1
    kernel = np.ones((k, k), dtype=np.uint8)

    # interior_u8: (size, size) uint8 {0,1}
    interior_u8 = cv2.erode(fg_r.astype(np.uint8), kernel, iterations=1)
    interior = interior_u8.astype(bool)  # (size, size) bool

    use_interior = bool(np.any(interior))

    # For weighted computations (FFT), use interior-weighted alpha.
    # w: (size, size) float32
    w = (alpha_r * interior_u8.astype(np.float32)) if use_interior else alpha_r.copy()

    # Foreground fraction (debug): use interior if available, else fg_r
    fg_fraction = float(interior.mean()) if use_interior else float(fg_r.mean())

    # ----------------------------
    # 3) Entropy on resized grayscale (interior selection)
    # ----------------------------
    # vals: (size, size) uint8 in [0,255]
    vals = np.clip(gray_r * 255.0, 0, 255).astype(np.uint8)

    # vals_sel: (N,) uint8
    if use_interior:
        vals_sel = vals[interior]
    elif np.any(fg_r):
        vals_sel = vals[fg_r]
    else:
        vals_sel = vals.ravel()

    counts = np.bincount(vals_sel, minlength=256)  # (256,)
    gray_entropy = _shannon_entropy_from_counts(counts, base=entropy_base)

    # ----------------------------
    # 4) Gradient entropy on resized grayscale (interior selection)
    # ----------------------------
    # Use OpenCV Sobel on float32.
    # gx, gy, grad: (size, size)
    gx = cv2.Sobel(gray_r, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3, borderType=cv2.BORDER_REFLECT101)
    gy = cv2.Sobel(gray_r, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3, borderType=cv2.BORDER_REFLECT101)
    grad = cv2.magnitude(gx, gy)

    # grad_sel: (N,) float32
    if use_interior:
        grad_sel = grad[interior]
    elif np.any(fg_r):
        grad_sel = grad[fg_r]
    else:
        grad_sel = grad.ravel()

    gmax = float(grad_sel.max()) if grad_sel.size else 0.0
    if gmax <= 1e-12:
        grad_entropy = 0.0
    else:
        # g_counts: (grad_bins,)
        g_counts, _ = np.histogram(grad_sel, bins=grad_bins, range=(0.0, gmax))
        grad_entropy = _shannon_entropy_from_counts(g_counts, base=entropy_base)

    # ----------------------------
    # 5) FFT metrics (interior-weighted)
    # ----------------------------
    # Weighted mean subtraction:
    mu = _weighted_mean(gray_r, w) if float(w.sum()) > 1e-8 else float(gray_r.mean())

    # s: (size, size)
    s = (gray_r - mu) * w

    if use_hann_window:
        # window: (size, size)
        s = s * _hann2d(size)

    # FFT:
    # F: (size, size) complex
    # P: (size, size) float64
    F = np.fft.fftshift(np.fft.fft2(s))
    P = (np.abs(F) ** 2).astype(np.float64)

    # Frequency grid in cycles/pixel:
    # f: (size,) in [-0.5, 0.5)
    f = np.fft.fftshift(np.fft.fftfreq(size)).astype(np.float64)  # (size,)

    # fx, fy, fr: each (size, size)
    fy, fx = np.meshgrid(f, f, indexing="ij")
    fr = np.sqrt(fx * fx + fy * fy)

    # Exclude DC
    m = fr > 0
    Pw = P[m]
    frw = fr[m]

    total_power = float(Pw.sum())
    if total_power <= 1e-18:
        spectral_entropy = 0.0
        spectral_centroid = 0.0
        hf_ratio = 0.0
        hf_cutoff = 0.0
    else:
        # Spectral entropy on coefficient distribution
        p_spec = Pw / total_power
        p_nz = p_spec[p_spec > 0]
        spectral_entropy = float(-(p_nz * np.log(p_nz)).sum())
        if entropy_base != np.e:
            spectral_entropy /= np.log(entropy_base)

        # Spectral centroid in cycles/pixel
        spectral_centroid = float((frw * Pw).sum() / total_power)

        # HF ratio: cutoff is a fraction of Nyquist (0.5 cycles/pixel)
        hf_cutoff = float(hf_cut * 0.5)
        hf_ratio = float(Pw[frw > hf_cutoff].sum() / total_power)

    return {
        # debug / bookkeeping
        "size": int(size),
        "alpha_threshold": float(alpha_threshold),
        "bbox_xyxy": bbox_xyxy,  # (x0, y0, x1, y1) in original coords, or None
        "foreground_fraction_resized": fg_fraction,
        "erode_radius_px": int(erode_radius),
        # (3) grayscale entropy
        "gray_entropy": float(gray_entropy),
        # (4) gradient magnitude entropy
        "gradient_entropy": float(grad_entropy),
        # (5) FFT metrics
        "spectral_entropy": float(spectral_entropy),
        "spectral_centroid_cyc_per_px": float(spectral_centroid),
        "hf_ratio": float(hf_ratio),
        "hf_cut": float(hf_cut),
        "hf_cutoff_cyc_per_px": float(hf_cutoff),
        # extra
        "total_power": float(total_power),
        "gray_histogram": counts,  # (256,) int
    }


def check_backprojected_pcd_and_mesh_pcd(
    o3d_mesh: o3d.geometry.TriangleMesh,
    z_map: torch.Tensor,  # (q, h, w)
    hit_map: torch.Tensor,  # (q, h, w)  bool
    intrinsic: torch.Tensor,  # (q, 3, 3)
    H_c2w: torch.Tensor,  # (q, 4, 4)
    num_points: int,
    num_threads: int = -1,
    prs: T.List[float] = (90, 95, 98, 99, 100),  # small to large
    th_dist: float = 0.003,
    printout: bool = False,
):
    """
    randomly backproject pixels from the depth map and
    compare with the point cloud sampled from the mesh surfaces.

    Args:
        o3d_mesh:
        z_map:
            (q, h, w)
        hit_map:
            (q, h, w) bool
        intrinsic:
            (q, 3, 3)
        H_c2w:
            (q, 4, 4)
        num_points:
            number of points to sample from z_map
        num_threads:
            number of threads to use for nearest neighbor search
        prs:
            list of (k,)  each is [0, 100]. The percentile to gather from the dist
        th_dist:
            threshold to count the number of points that is further away from the reference point cloud.

    Returns:

    """
    total = hit_map.sum()
    assert total > 0, "no point to backproject"

    # get backproject pcd
    odict = utils.compute_xyz_w_and_select_random_points(
        z_map=z_map.unsqueeze(0),
        hit_map=hit_map.unsqueeze(0),
        intrinsic=intrinsic.unsqueeze(0),
        H_c2w=H_c2w.unsqueeze(0),
        num_points=num_points,
    )
    proj_xyz_w = odict["xyz_w"].squeeze(0).float().cpu().numpy()  # (n, 3)

    # get mesh pcd
    o3d_pcd = o3d_mesh.sample_points_uniformly(
        number_of_points=int(num_points * 1.5),
    )
    mesh_xyz_w = np.array(o3d_pcd.points, dtype=np.float32)[:num_points]  # (n, 3)

    # find the closest point of mesh_xyz_w in proj_xyz_w
    tree = cKDTree(proj_xyz_w)
    knn_dist, knn_idx = tree.query(mesh_xyz_w, k=1, p=2, workers=num_threads)
    # knn_dist: (n,)  knn_midx: (n,)
    mesh_to_proj_dist = np.reshape(knn_dist, (mesh_xyz_w.shape[0],))  # (n,)  l2_norm
    mesh_to_proj_idx = np.reshape(knn_idx, (mesh_xyz_w.shape[0],))  # (n,) index in proj_xyz_w

    # find the closest point of proj_xyz_w in mesh_xyz_w
    tree = cKDTree(mesh_xyz_w)
    knn_dist, knn_idx = tree.query(proj_xyz_w, k=1, p=2, workers=num_threads)
    # knn_dist: (n,)  knn_midx: (n,)
    proj_to_mesh_dist = np.reshape(knn_dist, (proj_xyz_w.shape[0],))  # (n,)  l2_norm
    proj_to_mesh_idx = np.reshape(knn_idx, (proj_xyz_w.shape[0],))  # (n,) index in proj_xyz_w

    # find the closest point of mesh_xyz_w in mesh_xyz_w
    tree = cKDTree(mesh_xyz_w)
    knn_dist, knn_idx = tree.query(mesh_xyz_w, k=2, p=2, workers=num_threads)
    # knn_dist: (n,2)  knn_midx: (n,2) sorted small to large
    mesh_to_mesh_dist = np.reshape(knn_dist[..., -1], (mesh_xyz_w.shape[0],))  # (n,)  l2_norm
    mesh_to_mesh_idx = np.reshape(knn_idx[..., -1], (mesh_xyz_w.shape[0],))  # (n,) index in proj_xyz_w

    avg_mesh_to_mesh_dist = np.mean(mesh_to_mesh_dist)  # (,)
    std_mesh_to_mesh_dist = np.std(mesh_to_mesh_dist)  # (,)

    pr_vals = np.percentile(mesh_to_proj_dist, prs, method="nearest")
    mesh_to_proj_info_dict = dict(
        th_dist=th_dist,
        prs=prs,
        num_point_greater_than_th=np.sum(mesh_to_proj_dist > th_dist),
        pr_vals=pr_vals,
        normalized_pr_val=pr_vals / np.clip(avg_mesh_to_mesh_dist, a_min=1e-8, a_max=None),
        avg_mesh_to_mesh_dist=avg_mesh_to_mesh_dist,
        std_mesh_to_mesh_dist=std_mesh_to_mesh_dist,
    )

    pr_vals = np.percentile(proj_to_mesh_dist, prs, method="nearest")
    proj_to_mesh_info_dict = dict(
        th_dist=th_dist,
        prs=prs,
        num_point_greater_than_th=np.sum(proj_to_mesh_dist > th_dist),
        pr_vals=pr_vals,
        normalized_pr_val=pr_vals / np.clip(avg_mesh_to_mesh_dist, a_min=1e-8, a_max=None),
        avg_mesh_to_mesh_dist=avg_mesh_to_mesh_dist,
        std_mesh_to_mesh_dist=std_mesh_to_mesh_dist,
    )

    # figure out the ratio of the interior points of the mesh points
    # 1. we first remove all the matched points from the mesh points
    # 2. for the rest of the points, we consider the points whose 1nn-dist > th_dist are interior points (or outliers)
    outlier_mesh_idxs = np.setdiff1d(np.arange(mesh_xyz_w.shape[0]), proj_to_mesh_idx)  # (m,)  index of mesh_xyz_w
    outlier_dists = mesh_to_proj_dist[outlier_mesh_idxs]  # (m,)
    actual_outlier_idxs = outlier_dists > th_dist  # (mo,)
    actual_outlier_dists = outlier_dists[actual_outlier_idxs]  # (mo,)
    actual_outlier_mesh_idxs = outlier_mesh_idxs[actual_outlier_idxs]  # (mo,)

    unseen_info_dict = dict(
        num_unseen_mesh_points=len(actual_outlier_idxs),
        ratio_unseen_mesh_points=len(actual_outlier_idxs) / max(1, mesh_xyz_w.shape[0]),
        avg_unseen_mesh_point_dist=np.mean(actual_outlier_dists),
        std_unseen_mesh_point_dist=np.std(actual_outlier_dists),
    )

    return dict(
        mesh_to_proj_info_dict=mesh_to_proj_info_dict,
        proj_to_mesh_info_dict=proj_to_mesh_info_dict,
        unseen_info_dict=unseen_info_dict,
        mesh_xyz_w=mesh_xyz_w,  # (n, 3) ndarray
        proj_xyz_w=proj_xyz_w,  # (n, 3) ndarray
        mesh_to_proj_dist=mesh_to_proj_dist,  # (n,)
        proj_to_mesh_dist=proj_to_mesh_dist,  # (n,)
        unseen_mesh_point_idxs=actual_outlier_mesh_idxs,  # (mo,)
    )


def check_tranparency(
    alpha: torch.Tensor,  # (b, q, h, w, 1) [0, 1]
    th_alpha_foreground: float = 0.01,
    th_alpha_opaque: float = 0.99,
    th_alpha_transparent: float = 0.9,
    printout: bool = False,
):
    """
    Given alpha map, compute the ratio of opaque pixels (alpha > th_alpha_opaque)
    and transparent pixels (alpha < th_alpha_transparent).

    Args:
        alpha:
            (b, q, h, w, 1)  [0, 1]
        th_alpha_foreground:
        th_alpha_opaque:
        th_alpha_transparent:

    Returns:
        num_foreground:
            (b,) total number of foreground pixels
        num_opaque:
            (b,) total number of opaque pixels
        ratio_opaque:
            (b,) [0, 1]
        num_transparent:
            (b,)
        ratio_transparent:
            (b,)  [0, 1]
    """

    mask_foreground = (alpha > th_alpha_foreground).squeeze(-1)  # (b, q, h, w) bool
    mask_background = ~mask_foreground  # (b, q, h, w)  bool
    b, q, h, w = mask_background.shape
    dilate_kernel_size = 5  # need to be odd
    mask_background = (
        torch.nn.functional.max_pool2d(
            mask_background.reshape(b * q, 1, h, w).float(),
            kernel_size=dilate_kernel_size,
            stride=1,
            padding=dilate_kernel_size // 2,  # to have same size
        )
        > 0.5
    )  # (bq, 1, h, w) bool
    mask_background = mask_background.reshape(b, q, h, w)  # (b, q, h, w) bool
    mask_foreground = ~mask_background  # (b, qhw) bool
    num_foreground = mask_foreground.reshape(b, -1).sum(dim=-1)  # (b,) long
    if printout:
        print(f"num_foreground: {num_foreground}", flush=True)

    mask_opaque = torch.logical_and(
        (alpha >= th_alpha_opaque).squeeze(-1),  # (b, q, h, w) bool
        mask_foreground,  # (b, q, h, w) bool
    )  # (b, q, h, w)
    num_opaque = mask_opaque.reshape(b, -1).sum(dim=-1)  # (b,) long
    ratio_opaque = num_opaque.float() / num_foreground.clamp(min=1).float()  # (b,)

    if printout:
        print(f"num_opaque: {num_opaque}", flush=True)
        print(f"ratio_opaque: {ratio_opaque}", flush=True)

    mask_transparent = torch.logical_and(
        (alpha < th_alpha_transparent).squeeze(-1),  # (b, q, h, w) bool
        mask_foreground,
    )  # (b, q, h, w)
    num_transparent = mask_transparent.reshape(b, -1).sum(dim=-1)  # (b,) long
    ratio_transparent = num_transparent.float() / num_foreground.clamp(min=1).float()  # (b,)

    if printout:
        print(f"num_transparent: {num_transparent}", flush=True)
        print(f"ratio_transparent: {ratio_transparent}", flush=True)

    out_dict = dict(
        num_foreground=num_foreground,  # (b,)
        ratio_foreground=num_foreground / (q * h * w),
        num_opaque=num_opaque,  # (b,)
        ratio_opaque=ratio_opaque,  # (b,)
        num_transparent=num_transparent,  # (b,)
        ratio_transparent=ratio_transparent,  # (b,)
    )

    return out_dict
