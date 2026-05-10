#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
import os
import typing as T
import warnings

import matplotlib.font_manager
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
import xatlas

import torch

from plibs import rigid_motion
from plibs.uv_mapping import UVMap

from . import sample_utils


def sample_point_cloud_with_ray_tracing(
    mesh: o3d.geometry.TriangleMesh,
    cam_pose: T.Dict[str, T.Any],
    texture_maps: T.List[np.ndarray] = None,
) -> T.Dict[str, T.Any]:
    """
    Sample a set of points (with color) on the mesh.

    Args:
        mesh:
            a triangle mesh representing the surface.
            We assume it only have one texture map
        cam_pose:
            a dictionary containing:
                - intrinsic_matrix: the (3,3) intrinsic matrix
                - extrinsic_matrix: the (4,4) homogeneous matrix (from world to local)
                - width_px: number of pixels of the sensor
                - height_px: number of pixels of the sensor
        texture_maps:
            a list of texture maps (h, w, c) that we want to get the values using uv_mapping
            For example, if it is the rgb albedo, it is be retrieved with
            :py:`texture = np.asarray(mesh.textures[0]) / 255.  # (h,w,3)`.
            It can also be surface normals, features, etc.
            If `None`, not uv interpolation is performed.
    Returns:
        raycast_results:
            the output of `RaycastingScene`
        uv_outputs:
            a list of uv-interpolated values from each texture map.
    """

    # set up the raycasting scene
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh_t)

    # create the pinhole camera rays
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=cam_pose["intrinsic_matrix"],
        extrinsic_matrix=cam_pose["extrinsic_matrix"],
        width_px=cam_pose["width_px"],
        height_px=cam_pose["height_px"],
    )

    # cast the rays, get the intersections
    raycast_results = scene.cast_rays(rays)
    # if the ray does not hit the mesh, it goes to inf
    hit_map = 1 - np.isinf(raycast_results["t_hit"].numpy())  # (h',w')

    if texture_maps is None or len(texture_maps) == 0:
        return dict(
            hits=hit_map,  # (h', w')  1: hit, 0: not hit
            rays=rays,  # (h', w', 6)  x, y, z, dx, dy, dz
            raycast_results=raycast_results,
            uv_outputs=[],
        )

    # # barycentric coordinates of the intersection in the intersected triangle
    # barycentric_coords = raycast_results['primitive_uvs'].numpy()  # (h',w',2), the third weight is 1-sum()
    # # (h',w',2) -> (h',w',3)
    # barycentric_coords = np.concatenate(
    #     (barycentric_coords, 1 - np.sum(barycentric_coords, axis=-1, keepdims=True)),
    #     axis=-1,
    # )
    #
    # # the intersected triangle index
    # primitive_ids = raycast_results['primitive_ids'].numpy()  # (h', w'),
    # # fillin a dummy primitive_id for the rays that go to inf
    # primitive_ids[primitive_ids == o3d.t.geometry.RaycastingScene.INVALID_ID] = 0  # (h', w')
    #
    # # get the uv coordinates of the vertices of each triangle
    # triangle_uvs = np.asarray(mesh.triangle_uvs)  # (num_triangles*3, 2)
    # triangle_uvs = np.reshape(triangle_uvs, (-1, 3, 2))  # (num_triangles, 3, 2)
    #
    # # compute the uv coordinate on the texture map
    # vertex_uvs = triangle_uvs[primitive_ids]  # (h', w', 3, 2)
    # uvs = np.sum(np.expand_dims(barycentric_coords, axis=-1) * vertex_uvs, axis=2)  # (h', w', 2)
    #
    # uv_outputs = []
    # for texture in texture_maps:
    #     uv_map = UVMap(texture)
    #     out = uv_map(uvs)  # (h', w', dim)
    #
    #     # set the unintersected rays to zero
    #     axis = list(range(hit_map.ndim, out.ndim))
    #     if len(axis) > 0:
    #         tmp_hit_map = np.expand_dims(hit_map, axis=axis)
    #     else:
    #         tmp_hit_map = hit_map
    #     out = out * tmp_hit_map
    #
    #     uv_outputs.append(out)

    uv_outputs = interp_texture_map_from_ray_tracing_results(
        mesh=mesh,
        raycast_results=raycast_results,
        texture_maps=texture_maps,
    )

    return dict(
        hits=hit_map,  # (h', w')  1: hit, 0: not hit
        rays=rays,  # (h', w', 6)  x, y, z, dx, dy, dz
        raycast_results=raycast_results,
        uv_outputs=uv_outputs,  # list of (h', w', dim)
    )


def interp_texture_map_from_ray_tracing_results(
    mesh: o3d.geometry.TriangleMesh,
    raycast_results: T.Dict[str, T.Any],
    texture_maps: T.List[np.ndarray],
    merge_textures: bool = False,
) -> T.List[np.ndarray]:
    """
    Interpolate a uv_map given ray_tracing results and a mesh

    Args:
        mesh:
            o3d mesh
        raycast_results:
            (*, h', w')  the output of ray casting of o3d,
        texture_maps:
            a list of texture maps (h, w, c) that we want to get the values using uv_mapping
            For example, if it is the rgb albedo, it is be retrieved with
            :py:`texture = np.asarray(mesh.textures[0]) / 255.  # (h,w,3)`.
            It can also be surface normals, features, etc.
            If `None`, not uv interpolation is performed.
        merge_textures:
            If True, assume each texture map corresponds to a material_ids in the mesh
            For each texture map, only apply if the material_id is matched with the pixel
            Sum all contribution from all texture maps

    Returns:
        a list of uv-interpolated values from each texture map.  (*, dim_texture)
        The size `*` is determined by the rays's shape to ray tracing.
        If merge_textures is true, return sum of uv-interpolated values from all texture map instead. (dim_texture)
    """

    hit_map = 1 - np.isinf(raycast_results["t_hit"].numpy())  # (*, h',w')

    # barycentric coordinates of the intersection in the intersected triangle
    barycentric_coords = raycast_results["primitive_uvs"].numpy()  # (*, h',w',2), the third weight is 1-sum()
    # (*, h',w',2) -> (*, h',w',3)
    barycentric_coords = np.concatenate(
        (1 - np.sum(barycentric_coords, axis=-1, keepdims=True), barycentric_coords),
        # note that in open 3d, the primitive_uvs indicates last two coordinates rather than first two
        # (barycentric_coords, 1 - np.sum(barycentric_coords, axis=-1, keepdims=True)),
        axis=-1,
    )  # (*, h', w', 3)

    # the intersected triangle index
    primitive_ids = raycast_results["primitive_ids"].numpy()  # (*, h', w'),
    # fillin a dummy primitive_id for the rays that go to inf
    primitive_ids[primitive_ids == o3d.t.geometry.RaycastingScene.INVALID_ID] = 0  # (*, h', w')

    # get the uv coordinates of the vertices of each triangle
    triangle_uvs = np.asarray(mesh.triangle_uvs)  # (num_triangles*3, 2)
    triangle_uvs = np.reshape(triangle_uvs, (-1, 3, 2))  # (num_triangles, 3, 2)

    # compute the uv coordinate on the texture map
    vertex_uvs = triangle_uvs[primitive_ids]  # (*, h', w', 3, 2)
    uvs = np.sum(np.expand_dims(barycentric_coords, axis=-1) * vertex_uvs, axis=-2)  # (*, h', w', 2)

    # compute the material id for each pixel in the image
    if merge_textures:
        triangle_material_ids = np.asarray(mesh.triangle_material_ids)  # (num_triangle,)
        material_ids = triangle_material_ids[primitive_ids]  # (*, h', w')

    uv_outputs = []  # Now return sum of output from all texture maps []
    for id_t, texture in enumerate(texture_maps):
        uv_map = UVMap(texture)  # texture:  (h, w, dim)
        out = uv_map(uvs)  # (*, h', w', dim)

        # set the unintersected rays to zero
        axis = list(range(hit_map.ndim, out.ndim))
        if len(axis) > 0:
            tmp_hit_map = np.expand_dims(hit_map, axis=axis)  # (*, h', w', 1)
        else:
            tmp_hit_map = hit_map  # (*, h', w')
        out = out * tmp_hit_map  # (*, h', w', dim)

        if merge_textures:
            material_map = material_ids == id_t
            if len(axis) > 0:
                material_map = np.expand_dims((material_ids == id_t), axis=axis)
            out = out * tmp_hit_map * material_map
        uv_outputs.append(out)

    if merge_textures:
        uv_outputs = [sum(uv_outputs)]  # use list to be compatible with legacy usage

    return uv_outputs


def interp_surface_normal_from_ray_tracing_results(
    mesh: o3d.geometry.TriangleMesh,
    raycast_results: T.Dict[str, T.Any],
) -> np.ndarray:
    """
    Interpolate the surface normal of the intersection points using surface normal on the vertices.

    Args:
        mesh:
        raycast_results:
            the output of ray casting of o3d

    Returns:
        (*, 3) surface normal.  Note that it fills in a random value if a ray does not hit a surface

    """
    if not mesh.has_vertex_normals():
        return raycast_results["primitive_normals"].numpy()  # (*,3)

    # barycentric coordinates of the intersection in the intersected triangle
    barycentric_coords = raycast_results["primitive_uvs"].numpy()  # (h,w,2), the third weight is 1-sum()
    barycentric_coords = np.concatenate(
        (1 - np.sum(barycentric_coords, axis=-1, keepdims=True), barycentric_coords),
        # note that in open 3d, the primitive_uvs indicates last two coordinates rather than first two
        # (barycentric_coords, 1 - np.sum(barycentric_coords, axis=-1, keepdims=True)),
        axis=-1,
    )  # (h,w,3)

    # the intersected triangle index
    primitive_ids = raycast_results["primitive_ids"].numpy()  # (h, w)
    # fillin a dummy primitive_id for the rays that go to inf
    primitive_ids[primitive_ids == o3d.t.geometry.RaycastingScene.INVALID_ID] = 0  # (h, w)

    # get triangle vertex
    triangle_vidxs = np.asarray(mesh.triangles)  # (n_triangle, 3)  index of vertices
    vertex_normals = np.asarray(mesh.vertex_normals)  # (n_vertex, 3)  surface normal of each vertex

    vidxs = triangle_vidxs[primitive_ids]  # (h, w, 3)  the vertex id of each camera ray
    v_normals = vertex_normals[vidxs]  # (h, w, 3, 3)  last dimension is normal (dx, dy, dz)
    interped_normals = np.sum(np.expand_dims(barycentric_coords, axis=-1) * v_normals, axis=-2)  # (h, w, 3)
    return interped_normals


def rasterize(
    meshes: T.Union[
        o3d.geometry.TriangleMesh,
        T.List[o3d.geometry.TriangleMesh],
        o3d.geometry.PointCloud,
        T.List[o3d.geometry.PointCloud],
    ],
    intrinsic_matrix: T.Union[np.ndarray, T.List[np.ndarray]],
    extrinsic_matrices: T.Union[np.ndarray, T.List[np.ndarray]],
    width_px: int,
    height_px: int,
    get_point_cloud: bool = True,
    pcd_subsample: int = 1,
    point_size: float = -1,
    show_backface: bool = True,
    dtype: np.dtype = np.float32,
    light_on: bool = False,
    background_color: T.Union[T.List[float], T.Tuple[float]] = (1.0, 1.0, 1.0),
    shade_img=True,
) -> T.Dict[str, T.Any]:
    """
    Use open3d's visualizer to render image and depth_map from the camera.

    Important note:  On mac the function may or may not work due to GLTF support of MacOS.

    Args:
        meshes: a list of meshes
        intrinsic_matrix:
            (3,3) intrinsic matrix shared among all cameras.
            or a list of (3, 3) intrinsic matrices of each cameras.
            We only support fx = fy

            For example:
                intrinsic_matrix=np.array([
                    [128, 0, 128],
                    [0, 128, 128],
                    [0,   0,   1],
                ], dtype=np.float),

        extrinsic_matrices:
            a list of (4,4) homogeneous matrix (from world coordinate to camera coordinate)

            For example:
                extrinsic_matrix=np.array([
                    [1, 0, 0, 0,],
                    [0, 1, 0, 0,],
                    [0, 0, 1, 0,],
                    [0, 0, 0, 1.,],
                ], dtype=np.float),,  # world to camera (cV = H * wV)

        width_px:
            number of pixels of the sensor.  ex: 256
        height_px:
            number of pixels of the sensor.  ex: 256

        # cam_poses:
        #     a list of dictionary containing:
        #         - intrinsic_matrix: the (3,3) intrinsic matrix
        #         - extrinsic_matrix: the (4,4) homogeneous matrix (from world to local)
        #         - width_px: number of pixels of the sensor
        #         - height_px: number of pixels of the sensor
        #
        #     For example:
        #     cam_pose = dict(
        #         intrinsic_matrix=np.array([
        #             [128, 0, 128],
        #             [0, 128, 128],
        #             [0,   0,   1],
        #         ], dtype=np.float),
        #         extrinsic_matrix=np.array([
        #             [1, 0, 0, 0,],
        #             [0, 1, 0, 0,],
        #             [0, 0, 1, 0,],
        #             [0, 0, 0, 1.,],
        #         ], dtype=np.float),,  # world to camera (cV = H * wV)
        #         width_px=256,
        #         height_px=256,
        #     )

        get_point_cloud:
            whether to construct a point cloud (in the world coordinate) from
            the rendered images
        pcd_subsample:
            subsample the point cloud (1 point in every n pixel).  >= 1
        point_size:
            new option when input is a point cloud, change the render size of points
    Returns:
        imgs: a list of (h, w, 3)  rgb
        z_maps:  a list of (h, w)  z of the scene points in the camera coordinate
        pcds: a list of o3d.geometry.PointCloud in the world coordinate (one for each camera pose)
        hit_maps: a list of (h, w)  true: valid, false: not valid
    """

    np_dtype = sample_utils.get_np_dtype(dtype)

    if not isinstance(meshes, (list, tuple)):
        meshes = [meshes]

    # if not isinstance(extrinsic_matrices, (list, tuple)) or (
    if isinstance(extrinsic_matrices, np.ndarray) and extrinsic_matrices.ndim == 2:
        extrinsic_matrices = [extrinsic_matrices]

    if isinstance(intrinsic_matrix, np.ndarray) and intrinsic_matrix.ndim == 2:
        intrinsic_matrix = [intrinsic_matrix] * len(extrinsic_matrices)
    assert len(intrinsic_matrix) == len(extrinsic_matrices)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width_px, height=height_px, visible=False)
    # show back face to make sure ray-casting and rendering results are the same
    vis.get_render_option().mesh_show_back_face = show_backface
    vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
    vis.get_render_option().light_on = light_on
    vis.get_render_option().background_color = background_color
    for mesh in meshes:
        if isinstance(mesh, o3d.geometry.TriangleMesh):
            mesh.compute_vertex_normals()
        vis.add_geometry(mesh)

    all_points = []
    all_colors = []
    imgs = []
    z_maps = []
    hit_maps = []
    normals = []

    for i in range(len(extrinsic_matrices)):
        # assert np.isclose(intrinsic_matrix[i][0, 0], intrinsic_matrix[i][1, 1])
        assert np.isclose(np.abs(intrinsic_matrix[i][0, 0]), np.abs(intrinsic_matrix[i][1, 1]))
        view_ctl = vis.get_view_control()
        cam_pose_ctl = view_ctl.convert_to_pinhole_camera_parameters()
        cam_pose_ctl.intrinsic.height = height_px
        cam_pose_ctl.intrinsic.width = width_px
        cam_pose_ctl.intrinsic.intrinsic_matrix = intrinsic_matrix[i]
        cam_pose_ctl.extrinsic = extrinsic_matrices[i]
        view_ctl.convert_from_pinhole_camera_parameters(cam_pose_ctl, allow_arbitrary=True)

        if point_size > 0:
            vis.get_render_option().point_size = point_size

        # # render
        # vis.poll_events()
        # vis.update_renderer()
        # z_map = vis.capture_depth_float_buffer(do_render=False)
        # z_map = np.asarray(z_map).astype(dtype=np_dtype)
        # hit_map = np.logical_not(z_map == 0)
        # z_map[z_map == 0] = 1e12  # avoid points appear at camera center # not set to inf: avoid numerical problem

        # vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Color
        # img = vis.capture_screen_float_buffer(do_render=False)
        # img = np.asarray(img).astype(dtype=np_dtype)

        # vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
        # normal = vis.capture_screen_float_buffer(do_render=False)
        # normal = np.asarray(normal).astype(dtype=np_dtype)

        # Set mesh color to Color and render RGB image
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Color
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(do_render=False)
        img = np.asarray(img).astype(dtype=np_dtype)

        # Capture depth
        z_map = vis.capture_depth_float_buffer(do_render=False)
        z_map = np.asarray(z_map).astype(dtype=np_dtype)
        hit_map = np.logical_not(z_map == 0)
        z_map[z_map == 0] = 1e12  # to avoid points at camera center

        # Set mesh color to Normal and render normals
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
        vis.poll_events()
        vis.update_renderer()
        normal = vis.capture_screen_float_buffer(do_render=False)
        normal = np.asarray(normal).astype(dtype=np_dtype)

        imgs.append(np.copy(img))
        z_maps.append(np.copy(z_map))
        hit_maps.append(np.copy(hit_map))
        normals.append(normal * 2 - 1)

        # print(f'img.shape = {img.shape}')
        # print(f'z_map.shape = {z_map.shape}')

        # convert point cloud to world coordinate
        if get_point_cloud:
            H_cam_to_world = rigid_motion.RigidMotion.invert_homogeneous_matrix(cam_pose_ctl.extrinsic)
            # print(f'H_cam_to_world: {H_cam_to_world.shape}')
            points, colors = generate_point(
                rgb_image=img,
                depth_image=z_map,
                intrinsic=cam_pose_ctl.intrinsic.intrinsic_matrix,
                subsample=pcd_subsample,
                world_coordinate=True,
                pose=H_cam_to_world,
                hit_map=hit_map,
            )
            all_points.append(points)
            all_colors.append(colors)
            # print(f'points.shape = {points.shape}')
            # print(f'colors.shape = {colors.shape}')

    # clear the visualizer
    vis.clear_geometries()
    vis.destroy_window()
    del cam_pose_ctl
    del view_ctl
    del vis

    # create point cloud
    pcds = []
    for i in range(len(all_points)):
        pcd = o3d.geometry.PointCloud()
        # print(all_points[i].shape)
        pcd.points = o3d.utility.Vector3dVector(all_points[i])
        pcd.colors = o3d.utility.Vector3dVector(all_colors[i])
        pcds.append(pcd)

    return dict(pcds=pcds, imgs=imgs, z_maps=z_maps, hit_maps=hit_maps, normals=normals)


def derive_camera_intrinsics(
    width_px: int,
    height_px: int,
    fov: T.Union[float, torch.Tensor, np.ndarray],
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    derive camera intrinsic matrix
    Args:
        width_px: width (pixel)
        height_px: height (pixel)
        fov: field-of-view (degree)  (*,)

    Returns:
        3x3 intrinsic matrix  or (*, 3, 3)
    """

    if isinstance(fov, (float, int)):
        camera_f = 0.5 * float(width_px) / np.tan(0.5 * fov / 180.0 * np.pi)
        camera_intrinsics = np.array(
            [
                [camera_f, 0.0, width_px * 0.5],
                [0.0, camera_f, height_px * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=sample_utils.get_np_dtype(dtype),
        )
    elif isinstance(fov, (torch.Tensor, np.ndarray)):
        if isinstance(fov, np.ndarray):
            return_np = True
            fov = torch.from_numpy(fov)
        else:
            return_np = False
        camera_f = 0.5 * float(width_px) / torch.tan((0.5 * torch.pi / 180) * fov)  # (*,)
        b_shape = camera_f.shape
        camera_intrinsics = torch.zeros(*b_shape, 3, 3)  # (*, 3, 3)
        camera_intrinsics[..., 0, 0] = camera_f
        camera_intrinsics[..., 1, 1] = camera_f
        camera_intrinsics[..., 0, 2] = width_px * 0.5
        camera_intrinsics[..., 1, 2] = height_px * 0.5
        camera_intrinsics[..., 2, 2] = 1
        if return_np:
            camera_intrinsics = camera_intrinsics.numpy()
    else:
        raise NotImplementedError

    return camera_intrinsics


def compute_fov_from_intrinsics(
    intrinsics: T.Union[np.ndarray, torch.Tensor],
    width_px: int,
    height_px: int,
):
    """
    Given the 3x3 intrinsic matrix, compute the field of view in degree.

    Args:
        intrinsics:
            (*, 3, 3)
        width_px:
        height_px:

    Returns:
        fov_x:
            (*,) in degree
        fov_y:
            (*,) in degree
    """

    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    if isinstance(intrinsics, np.ndarray):
        fov_x = (np.arctan2((width_px - cx), fx) + np.arctan2(cx, fx)) * (180 / np.pi)
        fov_y = (np.arctan2((height_px - cy), fy) + np.arctan2(cy, fy)) * (180 / np.pi)
    elif isinstance(intrinsics, torch.Tensor):
        fov_x = (torch.atan2((width_px - cx), fx) + torch.atan2(cx, fx)) * (180 / torch.pi)
        fov_y = (torch.atan2((height_px - cy), fy) + torch.atan2(cy, fy)) * (180 / torch.pi)
    else:
        raise NotImplementedError

    return dict(
        fov_x=fov_x,  # (*,)
        fov_y=fov_y,  # (*,)
    )


def get_bbox_from_mask(
    mask: torch.Tensor,  # (*, h, w) bool
):
    """
    Given binary mask, compute the bounding box that contains all the True
    in the mask.

    Args:
        mask:
            (*, h, w)

    Returns:
        (*, 4)  y_min, x_min, y_max, x_max. included
        top left is the origin, x to right, y to down.

        If no true, y_min and x_min will be max(h, w), y_max and x_max will be -1.
    """
    *b_shape, h, w = mask.shape
    b = math.prod(b_shape)
    mask = mask.reshape(b, h, w)

    # locations of foreground
    ys, xs = torch.meshgrid(
        torch.arange(h, device=mask.device),
        torch.arange(w, device=mask.device),
        indexing="ij",
    )  # ys: (h, w), xs: (h, w)

    # expand to (t, h, w)
    ys = ys.unsqueeze(0).expand(b, -1, -1)  # (b, h, w)
    xs = xs.unsqueeze(0).expand(b, -1, -1)  # (b, h, w)

    # set invalid positions to big/small so min/max works
    big = torch.full((b,), max(h, w), device=mask.device, dtype=torch.long)
    minus = torch.full((b,), -1, device=mask.device, dtype=torch.long)

    # y-min, x-min
    y_min = torch.where(mask, ys, big[:, None, None]).amin(dim=(1, 2))
    x_min = torch.where(mask, xs, big[:, None, None]).amin(dim=(1, 2))

    # y-max, x-max
    y_max = torch.where(mask, ys, minus[:, None, None]).amax(dim=(1, 2))
    x_max = torch.where(mask, xs, minus[:, None, None]).amax(dim=(1, 2))

    # stack: (b, 4) -> (ymin, xmin, ymax, xmax) per frame, included
    bboxes = torch.stack([y_min, x_min, y_max, x_max], dim=1)  # (b, 4)
    bboxes = bboxes.reshape(*b_shape, 4)  # (*b, 4)

    return bboxes  # (*b, 4)


def adjust_margin(
    img: torch.Tensor,
    mask: torch.Tensor,
    target_margin_ratio: float,
    pad_val: float = 0,
    printout: bool = False,
):
    """
    Given image and mask (alpha or hit_map), adjust the margin.

    Args:
        img:
            (h, w, 3)
        mask:
            (h, w)
        target_margin_ratio:

    Returns:
        img:
            (h', w', 3)
        mask:
            (h', w')
    """
    _h, _w, _3 = img.shape
    assert _h == _w, "currently support only same"
    margin_px = target_margin_ratio * _h  # float

    # compute current bbox
    bbox = get_bbox_from_mask(mask=mask > 1e-6)  # (4,)  y_min, x_min, y_max, x_max. included
    ymin = bbox[0].item()
    xmin = bbox[1].item()
    ymax = bbox[2].item()
    xmax = bbox[3].item()

    y_margin = min(max(0, ymin - 1), max(0, _h - (ymax + 1)))  # float, px
    x_margin = min(max(0, xmin - 1), max(0, _w - (xmax + 1)))  # float, px
    xy_margin = min(x_margin, y_margin)  # int, px

    if printout:
        print(
            f"original margin: {xy_margin} ({xy_margin / _h * 100:.2f}%), "
            f"target ratio: {target_margin_ratio * 100:.2f}%"
        )

    to_pad_px = round(margin_px - xy_margin)  # int
    if to_pad_px >= 1:
        new_img = torch.ones(_h + to_pad_px * 2, _w + to_pad_px * 2, 3, dtype=img.dtype, device=img.device) * pad_val
        new_img[to_pad_px : to_pad_px + _h, to_pad_px : to_pad_px + _w] = img

        new_mask = torch.zeros(_h + to_pad_px * 2, _w + to_pad_px * 2, dtype=mask.dtype, device=mask.device)
        new_mask[to_pad_px : to_pad_px + _h, to_pad_px : to_pad_px + _w] = mask

    elif to_pad_px <= -1:
        new_img = img[-to_pad_px : _h + to_pad_px, -to_pad_px : _w + to_pad_px]
        new_mask = mask[-to_pad_px : _h + to_pad_px, -to_pad_px : _w + to_pad_px]
    else:
        new_img = img
        new_mask = mask

    return dict(
        img=new_img,  # (h, w, 3)
        mask=new_mask,  # (h, w)
    )


def create_gif(
    images: T.Union[torch.Tensor, np.ndarray, T.List[torch.Tensor], T.List[np.ndarray]],
    filename: str,
    fps: float,
    loop: bool = True,
):
    """
    Create a gif from the images

    Args:
        images:
            (n, h, w, 3) or list of (h, w, 3), float, range = 0-1
        filename:
            filename of the output gif
    """

    assert filename.lower().endswith("gif"), f"{filename}"
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().float().numpy()

    if isinstance(images, (list, tuple)):
        images = [img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img for img in images]

    if isinstance(images, np.ndarray):
        images = [images[i] for i in range(images.shape[0])]

    # # (h, w, c) -> (c, h, w)
    # images = [np.transpose(img, axes=(2, 0, 1)) for img in images]

    # make sure the range is 0-255
    images = [(np.clip(img, a_min=0, a_max=1) * 255).astype(np.uint8) for img in images]

    # numpy to pil image
    images = [Image.fromarray(img) for img in images]

    # avoid dithering
    # images = [img.quantize(method=Image.MEDIANCUT) for img in images]
    try:
        images = [img.quantize(method=Image.Quantize.MEDIANCUT) for img in images]
    except:
        images = [img.quantize(method=Image.MEDIANCUT) for img in images]

    # save gif
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=int((1000 + fps - 1) / fps),
        loop=loop,
    )

    # save png for first image to avoid artifact
    images[0].save(filename[:-4] + "_oneimg.png")

    # array2gif.write_gif(images, filename, fps=fps)


def gif_to_nparray(
    filename: str,
    crop_ratio: float = 0,
    crop_dir: str = "left",
) -> np.ndarray:
    """
    load gif to numpy array
    Args:
        filename:  filename of the gif
        crop_ratio: ratio of part to be cropped
        crop_dir: direction to be cropped, default left
    Returns:
        ndarray (n,h,w), n: number of frames
    """
    imglist = []
    if not os.path.exists(filename):
        return None

    imageObject = Image.open(filename)
    for frame in range(0, imageObject.n_frames):
        imageObject.seek(frame)
        tmp = imageObject.convert()  # Make without palette
        tmp = np.asarray(tmp) / 255.0
        # print(tmp.shape)
        if crop_ratio != 0:
            if crop_dir == "left":
                tmp = tmp[:, int(tmp.shape[1] * crop_ratio) :]
            elif crop_dir == "right":
                tmp = tmp[:, : int(tmp.shape[1] * crop_ratio)]
            else:
                raise NotImplementedError

        imglist.append(np.expand_dims(tmp, axis=0))
    gifarray = np.concatenate(imglist, axis=0)  # (n,h,w)
    return gifarray


def add_title_to_image(
    image: np.ndarray,
    title: str,
    font_size: int = 24,
    font_color: T.Union[float, T.List[float], None] = None,  # [0, 1]
    font_name: str = "DejaVuSans",
    background_color: T.Union[float, T.List[float]] = 0.0,  # [0, 1]
    pad_height_px: int = 30,
    align_width: str = "center",
    align_height: str = "center",
) -> np.ndarray:
    """
    Given an image, pad the image at the top and add a title text.
    Note that the function converts image to uint8

    Args:
        image:
            (*, h, w, 3) uint8 [0, 255]
        title:
            the str to add to the image
        font_size:
            font size to add
        font_color:
            font color of the text. If None, it will use the complement color of background_color.
        font_name:
            name of the font.
        background_color:
            background color
        pad_height_px:
            number of pixels to pad at the top of the image
        align_width:
            'center', 'left', 'right'
        align_height:
            'center', 'top', 'bottom'

    Returns:
        the padded image:
            (*, h+pad_height_px, w, 3)  uint8 [0, 255]
    """

    if isinstance(background_color, (int, float)):
        background_color = [background_color] * 3
    background_color = [int(c * 255) for c in background_color]

    if font_color is not None and isinstance(font_color, (int, float)):
        font_color = [int(font_color * 255)] * 3
    if font_color is None:
        font_color = [max(0, 255 - int(c)) for c in background_color]

    *b_shape, h, w, _c = image.shape
    pad_region = np.ones((*b_shape, pad_height_px, w, 3), dtype=image.dtype)
    for c in range(3):
        pad_region[..., c] = background_color[c]
    image = np.concatenate([pad_region, image], axis=-3)  # (*b, h', w, 3)

    if title is None or len(title) == 0:
        return image

    assert image.dtype == np.dtype(np.uint8)

    # get font, image, draw objs
    font_path = matplotlib.font_manager.findfont(font_name)
    font = ImageFont.truetype(font_path, font_size)
    pad_region = np.ones((pad_height_px, w, 3), dtype=image.dtype)  # (hp, w, 3)
    for c in range(3):
        pad_region[..., c] = background_color[c]
    img = Image.fromarray(pad_region)
    draw = ImageDraw.Draw(img)

    # calculate the text width of the resulted text
    text_w_px = draw.textlength(title, font)  # width of the text in pixel (float)

    # compute the starting location of the text image
    if align_width == "center":
        w_start = max(0, int((w - text_w_px) / 2.0))
    elif align_width == "left":
        w_start = 0
    elif align_width == "right":
        w_start = max(0, int(w - text_w_px))
    else:
        raise NotImplementedError

    if align_height == "center":
        h_start = max(0, int((pad_height_px - font_size) / 2.0)) if pad_height_px > 0 else 0
    elif align_height == "top":
        h_start = 0
    elif align_height == "bottom":
        h_start = int(font_size)
    else:
        raise NotImplementedError

    draw.text((w_start, h_start), title, fill=tuple(font_color), font=font)
    pad_region = np.asarray(img)  # (hp, w, 3)
    image[..., :pad_height_px, :, :] = pad_region  # (*b, h', w, 3)

    assert image.dtype == np.dtype(np.uint8)
    return image


def tile_images(
    images: T.Union[T.List[torch.Tensor], T.List[np.ndarray]],
    ncols: int = -1,
    background_color: T.Union[float, T.List[float]] = 0.0,  # [0, 1]
) -> T.Union[torch.Tensor, np.ndarray]:
    """
    Tile a list of images as a matrix (along h and w dimensions).

    Args:
        images:
            list of (*, h, w, c)
        ncols:
            number of images in a column. If -1, unlimited.
        background_color:
            If None, the image will be replaced by (h,w,c) * background

    Returns:
        tiled image:
            (*, h', w', c)
    """
    total = len(images)
    if ncols < 0:
        ncols = total

    # find the first non-None image
    img = None
    for i in range(total):
        if images[i] is not None:
            img = images[i]
            break
    if img is None:
        raise RuntimeError

    if isinstance(img, np.ndarray):
        is_numpy = True
        images = [torch.from_numpy(img) if img is not None else None for img in images]
    else:
        is_numpy = False

    *b_shape, h, w, c = img.shape
    if isinstance(background_color, (int, float)):
        background_color = [background_color] * c

    blank = torch.ones(*b_shape, h, w, c)
    for ic in range(c):
        blank[..., ic] = background_color[ic]

    nrows = math.ceil(total / ncols)
    if nrows == 1:
        ncols = total
    rows = []
    for _ in range(nrows):
        rows.append([blank] * ncols)

    ridx = 0
    cidx = 0
    for i in range(total):
        img = images[i]
        if img is None:
            img = blank
        rows[ridx][cidx] = img
        cidx += 1
        if cidx == ncols:
            cidx = 0
            ridx += 1

    # tile each column, then row
    rows = [torch.cat(row, dim=-2) for row in rows]  # list of (*b, h, w', c)
    img = torch.cat(rows, dim=-3)  # list of (*b, h', w', c)

    if is_numpy:
        img = img.detach().cpu().numpy()
    return img


def create_o3d_plane_mesh(
    top_left: T.List[float],  # (3,)
    top_right: T.List[float],  # (3,)
    bottom_left: T.List[float],  # (3,)
    bottom_right: T.Optional[T.List[float]] = None,  # (3,)
) -> o3d.geometry.TriangleMesh:
    """
    Create an o3d mesh representing a plane.
    The mesh contains two triangles.

    Args:
        top_left:
            the top left coordinate of the plane
        top_right:
            the top right coordinate of the plane
        bottom_left:
            the bottom left coordinate of the plane
        bottom_right:
            the bottom right coordinate of the plane.
            If None, will assumed to be a parallelgram

    Returns:
        o3d mesh
    """
    if isinstance(top_left, list):
        top_left = np.array(top_left, dtype=np.float64)
    if isinstance(top_right, list):
        top_right = np.array(top_right, dtype=np.float64)
    if isinstance(bottom_left, list):
        bottom_left = np.array(bottom_left, dtype=np.float64)
    if isinstance(bottom_right, list):
        bottom_right = np.array(bottom_right, dtype=np.float64)

    if bottom_right is None:
        bottom_right = top_right + (bottom_left - top_left)

    mesh = o3d.geometry.TriangleMesh()
    np_vertices = np.stack(
        [
            top_left,
            top_right,
            bottom_left,
            bottom_right,
        ],
        axis=0,
    )  # (4, 3)
    np_triangles = np.array(
        [
            [0, 2, 1],
            [1, 2, 3],
        ]
    ).astype(np.int32)
    mesh.vertices = o3d.utility.Vector3dVector(np_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(np_triangles)

    return mesh


def create_video(
    images: T.Union[torch.Tensor, np.ndarray, T.List[torch.Tensor], T.List[np.ndarray]],
    filename: str,
    fps: float,
    color_format: str = "rgb",
    val_range: str = "01",
):
    """
    Create a video from the images

    Args:
        images:
            (n, h, w, 3) or list of (h, w, 3), float, range = 0-1
        filename:
            filename of the output video
    """
    import cv2

    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().float().numpy()

    # read images
    if len(images) == 0:
        return

    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(len(images)):
        img = images[i]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        if img.dtype != np.uint8:
            if val_range == "01":
                img = np.clip(img, a_min=0, a_max=0.9999) * 255
            elif val_range == "0255":
                img = np.clip(img, a_min=0, a_max=255)
            else:
                img = img / np.max(img) * 255
            # img = np.clip(img, a_min=0, a_max=0.9999) * 255
            img = img.astype(np.uint8)

        if color_format == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img)

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    video.release()


def remesh_file(
    filename: str,
    out_filename: str = None,
) -> T.Dict[str, T.Any]:
    """
    Uvmap a mesh file using xatlas.  All existing textures are discarded.

    Args:
        filename:
            input mesh file
        out_filename:
            an obj file with uv mapping (without mtl)

    Returns:

    """

    mesh = o3d.io.read_triangle_mesh(filename, enable_post_processing=True)
    vertices = np.asarray(mesh.vertices)
    triangle_ids = np.asarray(mesh.triangles)

    out_dict = remesh(vertices, triangle_ids)

    if out_filename is not None:
        xatlas.export(
            out_filename,
            vertices[out_dict["vmapping"]],
            out_dict["indices"],
            out_dict["uvs"],
        )
    return out_dict


def remesh(
    vertices: np.ndarray,
    triangle_ids: np.ndarray,
) -> T.Dict[str, T.Any]:
    """
    Uvmap a mesh (provided as vertices and triangles) using xatlas.

    Args:
        vertices:
            (n, 3)  float
        triangle_ids:
            (m, 3) int

    Returns:
        vmapping:
            (n,),  uint32, contains the original vertex index for each new vertex.
        indices:
            (num_triangles, 3), uint32, contains the vertex indices of the new triangles.
        uvs:
            (n, 2), contains texture coordinates of the new vertices.
    """

    vmapping, indices, uvs = xatlas.parametrize(
        vertices,
        triangle_ids,
    )
    # `vmapping` contains the original vertex index for each new vertex, (n,), type uint32.
    # `indices` contains the vertex indices of the new triangles, (num_triangles, 3), type uint32.
    # `uvs` contains texture coordinates of the new vertices, (n, 2), type float32.

    return dict(
        vmapping=vmapping,
        indices=indices,
        uvs=uvs,
    )


def remesh_o3d_mesh(
    o3d_mesh: o3d.geometry.TriangleMesh,
) -> o3d.geometry.TriangleMesh:
    """
    Uvmap a mesh using xatlas. All existing textures are discarded.

    Args:
        o3d_mesh:
            triangle mesh
    Returns:
        new triangle mesh
    """

    vertices = np.asarray(o3d_mesh.vertices)
    triangle_ids = np.asarray(o3d_mesh.triangles)

    out_dict = remesh(vertices, triangle_ids)

    new_vertices = np.copy(vertices[out_dict["vmapping"]])  # (num_vertices, 3xyz)
    triangles = out_dict["indices"]  # (num_triangles, 3i)
    vertex_uvs = out_dict["uvs"]  # (num_vertices, 2uv)

    # create triangle_uvs:  # (num_triangles * 3, 2uv)
    triangle_uvs = vertex_uvs[triangles]  # (num_triangles, 3i, 2uv)
    triangle_uvs = np.ascontiguousarray(np.reshape(triangle_uvs, (-1, 2)))  # (num_triangles *3i, 2uv)

    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    new_mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    new_mesh.textures = [o3d.geometry.Image(128 * np.ones((3, 3), dtype=np.uint8))]
    new_mesh.triangle_material_ids = o3d.utility.IntVector(
        np.zeros(np.asarray(new_mesh.triangles).shape[0], dtype=np.int32)
    )

    return new_mesh


def create_coordinate_frame_pcd(
    size: float = 1.0,
    num_points: int = 100,
) -> T.Tuple[np.ndarray, np.ndarray]:
    """
    Create a coordinate frame of `size` centered at the world coordinate's origin.
    The x axis is red, y is green, and z is blue.

    Args:
        size:
            length of each axis
        num_points:
            number of points to form each axis

    Returns:
        points: (3n, 3)  np.ndarray
        colors: (3n, 3)  np.ndarray
    """

    base = np.linspace(start=0, stop=size, num=num_points)  # (n,)
    eye = np.eye(3)  # (3, 3)

    points = []
    colors = []
    for i in range(3):
        point = (eye[:, i : i + 1] @ base[None, :]).T  # (n, 3)
        color = np.tile(eye[:, i][None, :], (num_points, 1))  # (n, 3)
        points.append(point)
        colors.append(color)
    points = np.concatenate(points, axis=0)  # (3n, 3)
    colors = np.concatenate(colors, axis=0)  # (3n, 3)
    return points, colors


def volumetric_integration_bug(
    density: T.Optional[torch.Tensor],  # (*, n, 1)
    delta: T.Optional[torch.Tensor],  # (*, n)
    values: T.Dict[str, torch.Tensor],  # (*, n, d)
    alpha: torch.Tensor = None,  # (*, n, 1)
    background_values: T.Dict[str, T.Union[float, torch.Tensor]] = None,
) -> T.Dict[str, torch.Tensor]:
    """
    Perform volumetric integration using the quadrature sampled
    `density` and bin width `delta` to integrate `values`
    Args:
        density:
            (*, n, 1) where n is the number of samples on a ray. Optional if alpha is given.
        delta:
            (*, n) quadrature bin width for individual samples on the ray.
            Optional if alpha is given.
        values:
            (*, n, d) name -> values to integrate

    Returns:
        dict of str -> (*, d)
    """

    # calculate alpha
    if alpha is None:
        alpha = 1 - torch.exp(-1 * density * delta.unsqueeze(-1))  # (*, n, 1)

    T = (1 - alpha.squeeze(-1)).cumprod(dim=-1)  # (*, n)
    T_1 = torch.cat(
        [
            torch.ones(*T.shape[:-1], 1, dtype=T.dtype, device=T.device),  # (*, 1)
            T[..., 1:],  # (*, n-1),  this is simply a bug,  should be T[..., :-1]
        ],
        dim=-1,
    ).unsqueeze(-1)  # (*, n, 1)

    out_dict = dict()
    for name, vals in values.items():
        out = ((alpha * T_1) * vals).sum(dim=-2)  # (*, d)
        if background_values is not None:
            bg_val = background_values.get(name, None)
            if bg_val is not None:
                if T.shape[-1] >= 1:
                    out = out + T[..., -1:] * bg_val
                else:
                    out = out + bg_val
        out_dict[name] = out

    out_dict["alpha"] = alpha  # (*, n, 1)
    # comment out since we are going
    # out_dict['T_inf'] = T[..., -1:]  # (*, 1)  how much ratio of light reach inf, at most 1
    return out_dict


def volumetric_integration(
    density: T.Optional[torch.Tensor],  # (*, n, 1)
    delta: T.Optional[torch.Tensor],  # (*, n)
    values: T.Dict[str, torch.Tensor],  # (*, n, d)
    alpha: torch.Tensor = None,  # (*, n)
    background_values: T.Dict[str, T.Union[float, torch.Tensor]] = None,
) -> T.Dict[str, torch.Tensor]:
    """
    Perform volumetric integration using the quadrature sampled
    `density` and bin width `delta` to integrate `values`
    Args:
        density:
            (*, n, 1) where n is the number of samples on a ray. Optional if alpha is given.
        delta:
            (*, n) quadrature bin width for individual samples on the ray.
            Optional if alpha is given.
        values:
            (*, n, d) name -> values to integrate

    Returns:
        dict of str -> (*, d)
    """

    if density is not None and delta is not None:
        density_delta = density.squeeze(-1) * delta  # (*, n)
    else:
        density_delta = None

    # calculate alpha
    if alpha is None:
        assert density_delta is not None
        alpha = 1 - torch.exp(-1 * density_delta)  # (*, n)

    # calculate T
    if density_delta is not None:
        logT_1 = torch.cumsum(density_delta[..., :-1], dim=-1)  # (*, n-1)
        logT_1 = torch.cat(
            [
                torch.zeros(*logT_1.shape[:-1], 1, dtype=logT_1.dtype, device=logT_1.device),
                logT_1,
            ],
            dim=-1,
        )  # (*, n)
        T_1 = torch.exp(-logT_1)  # (*, n)
    else:
        T_1 = (1 - alpha[..., :-1]).cumprod(dim=-1)  # (*, n)
        T_1 = torch.cat(
            [
                torch.ones(*T_1.shape[:-1], 1, dtype=T_1.dtype, device=T_1.device),  # (*, 1)
                T_1,
            ],
            dim=-1,
        )  # (*, n)

    # the line works even if alpha is (*, 0) and T_1 is (*, 1)
    weights = (alpha * T_1).unsqueeze(-1)  # (*, n, 1)
    if alpha.size(-1) > 0:
        bg_weight = (T_1[..., -1] * (1 - alpha[..., -1])).unsqueeze(-1)  # (*, 1)
    else:
        bg_weight = 1

    out_dict = dict()
    for name, vals in values.items():
        out = (weights * vals).sum(dim=-2)  # (*, d)
        if background_values is not None:
            bg_val = background_values.get(name, None)
            if bg_val is not None:
                out = out + bg_weight * bg_val
        out_dict[name] = out

    out_dict["alpha"] = alpha.unsqueeze(-1)  # (*, n, 1)
    out_dict["bg_weight"] = bg_weight  # (*, 1)
    return out_dict


def volumetric_integration_with_s_density(
    t_to_surface: torch.Tensor,  # (*, n)
    values: T.Dict[str, torch.Tensor],  # (*, n, d)
    s: T.Union[float, torch.Tensor] = 1.0,
    valid_mask: torch.Tensor = None,  # (*, n)
    background_values: T.Dict[str, T.Union[float, torch.Tensor]] = None,
    check_finite: bool = False,
    mode: str = "mid_only",
    mask_alpha: torch.Tensor = None,  # (*, n)
) -> T.Dict[str, T.Any]:
    """
    Use NeuS's volumetric rendering to integrate value.

    Args:
        t_to_surface:
            (*, n) signed or unsigned distance to surface
        values:
            (*, n, d) dict containing the values to integrate
        s:
            (*,) or (*, 1) or float, to scale the sigmoid
        background_values:
            float, (d,), (*, d)
        mode:
            'all': use all points
            'mid_only': use only the mid points (used by neus)

    Returns:
        alpha:
            (*, n, 1)
        name:
            (*, d)  integrated result
    """

    if t_to_surface.dtype == torch.half:
        warnings.warn("convert t_to_surface from half to float")
        t_to_surface = t_to_surface.float()

    if check_finite:
        assert t_to_surface.isfinite().all(), f"{t_to_surface.isnan().any()}  {t_to_surface.isinf().any()}"

    if mode == "all":
        *b_shape, n = t_to_surface.shape
    elif mode == "mid_only":
        # make sure n is odd
        n = t_to_surface.size(-1)
        if n % 2 == 0:
            n -= 1

        # use the boundary points for dt
        t_to_surface = t_to_surface[..., 0:n:2]  # (*, n'+1)
        *b_shape, _n = t_to_surface.shape

        for name in values:
            # keep the mid points for values
            values[name] = values[name][..., 1:n:2, :]  # (*, n', d)

        if valid_mask is not None:
            _vm = torch.logical_and(
                valid_mask[..., 0 : n - 2 : 2],
                valid_mask[..., 2:n:2],
            )
            valid_mask = torch.logical_and(
                _vm,
                valid_mask[..., 1 : n - 1 : 2],
            )  # (*, n')

    else:
        raise NotImplementedError

    if isinstance(s, torch.Tensor):
        if s.numel() > 1:
            assert s.numel() == math.prod(b_shape)
            if s.ndim == len(b_shape):  # (*,)
                s = s.unsqueeze(-1)  # (*, 1)

    if check_finite and isinstance(s, torch.Tensor):
        assert s.isfinite().all(), f"{s.isnan().any()} {s.isinf().any()}"

    if t_to_surface.dtype == torch.float:
        exp_max = 16.0  # 80.  # if x is greater than the value, exp(x) -> inf
    elif t_to_surface.dtype == torch.half:
        exp_max = 8.0
    else:
        raise NotImplementedError

    # t_to_surface_s = t_to_surface * s
    # t_to_surface_s = torch.clamp(t_to_surface_s.detach(), max=exp_max) + (t_to_surface_s - t_to_surface_s.detach())

    t_to_surface_s = torch.clamp(t_to_surface * s, max=exp_max)

    sigmoid_dt = torch.sigmoid(t_to_surface_s)  # (*, n)

    if check_finite:
        assert sigmoid_dt.isfinite().all(), f"{sigmoid_dt.isnan().any()} {sigmoid_dt.isinf().any()}"

    # we assume the last point has alpha = 0
    if sigmoid_dt.dtype == torch.float:
        eps = 1e-4  # 1e-6
    elif sigmoid_dt.dtype == torch.half:
        eps = 1e-3  # 1e-2
    else:
        raise NotImplementedError

    # sigmoid_dt is very small means dt is very negative (far from surface)
    # we do not care, so we mask the alpha to be 0
    mask = sigmoid_dt[..., :-1] < eps  # (*, n-1)
    sigmoid_dt = torch.clamp(sigmoid_dt, min=eps)
    alpha = (sigmoid_dt[..., :-1] - sigmoid_dt[..., 1:]) / sigmoid_dt[..., :-1]  # (*, n-1)
    alpha = torch.clamp(alpha, min=0)  # (*, n)

    # alpha = (sigmoid_dt[..., :-1] - sigmoid_dt[..., 1:]) / torch.clamp(sigmoid_dt[..., :-1], min=eps)  # (*, n-1)
    if check_finite:
        assert alpha.isfinite().all(), f"{alpha.isnan().any()}  {alpha.isinf().any()}"
    alpha = alpha.masked_fill(mask, 0)  # (*, n-1)

    if mode == "all":
        # need to fake the last point
        alpha = torch.cat(
            [
                alpha,
                torch.zeros(*alpha.shape[:-1], 1, dtype=alpha.dtype, device=alpha.device),
            ],
            dim=-1,
        )  # (*, n)

    if mask_alpha is not None:
        alpha = alpha * mask_alpha  # (*, n)

    if check_finite:
        assert alpha.isfinite().all(), f"{alpha.isnan().any()} {alpha.isinf().any()}"
        assert (alpha < (1 + eps)).all(), f"{alpha.max()} {alpha.min()}"
        assert (alpha > -eps).all(), f"{alpha.max()} {alpha.min()}"

    if valid_mask is not None:
        alpha = alpha.masked_fill(
            ~valid_mask,
            0.0,
        )

    # volumetric rendering
    out_dict = volumetric_integration(
        density=None,
        delta=None,
        values=values,
        alpha=alpha.unsqueeze(-1),  # (*, n, 1)
        background_values=background_values,
    )
    return out_dict


def shade_with_normals(rgb_map, normals, hit_map, light_dir=[0, 0, 1]):
    """
    Shade RGB images using normals and preserve background with hit map.

    Args:
        rgb_map (torch.Tensor): [B, H, W, 3], float32 in [0, 1]
        normals (torch.Tensor): [B, H, W, 3], float32 in [-1, 1] or [0, 1]
        hit_map (torch.Tensor): [B, H, W], bool or 0/1
        light_dir (list): Light direction [x, y, z]

    Returns:
        torch.Tensor: shaded RGB image [B, H, W, 3]
    """
    # Make sure normals are in [-1, 1] and then convert to [0, 1]
    normals = torch.clip(normals, -1, 1)

    # Normalize light direction
    light_dir = torch.tensor(light_dir, dtype=normals.dtype, device=normals.device)
    light_dir = light_dir / light_dir.norm()
    # Compute dot product shading
    shading = normals @ light_dir  # [B, H, W]
    shading = torch.clip((shading + 1) / 2, 0, 1)
    # Apply shading to RGB
    shaded = rgb_map * shading.unsqueeze(-1)

    # Preserve backgrounnd
    shaded[~hit_map.bool()] = rgb_map[~hit_map.bool()]
    return shaded


def rotate_normals(normals, rotation_matrix):
    """
    Rotate normal map using a 3x3 rotation matrix.

    Args:
        normals (torch.Tensor): Shape [1, H, W, 3], values in [-1, 1]
        rotation_matrix (torch.Tensor): Shape [3, 3]

    Returns:
        torch.Tensor: Rotated normals, shape [1, H, W, 3]
    """
    B, H, W, _ = normals.shape
    normals_flat = normals.view(-1, 3)  # [H*W, 3]
    rotated = normals_flat @ rotation_matrix  # [H*W, 3]
    return rotated.view(B, H, W, 3)


def rasterize_shapenet(
    meshes: T.Union[
        o3d.geometry.TriangleMesh,
        T.List[o3d.geometry.TriangleMesh],
        o3d.geometry.PointCloud,
        T.List[o3d.geometry.PointCloud],
    ],
    intrinsic_matrix: T.Union[np.ndarray, T.List[np.ndarray]],
    extrinsic_matrices: T.Union[np.ndarray, T.List[np.ndarray]],
    width_px: int,
    height_px: int,
    get_point_cloud: bool = True,
    pcd_subsample: int = 1,
    point_size: float = -1,
    show_backface=True,
    dtype: np.dtype = np.float32,
    light_on: bool = False,
    background_color: T.Union[T.List[float], T.Tuple[float]] = (1.0, 1.0, 1.0),
    shade_image: bool = True,
    rotation_matrix: T.Optional[np.ndarray] = None,
    light_dir: T.List[float] = [0, 0, 1],
    unrotate_normals: bool = False,
    H_c2w_matrices: T.Optional[T.Union[np.ndarray, T.List[np.ndarray]]] = None,
) -> T.Dict[str, T.Any]:
    """
    Use open3d's visualizer to render image and depth_map from the camera with shading options.

    This is a specialized version of the rasterize function for ShapeNet dataset that includes
    normal-based shading functionality. The shading is applied before point cloud generation
    so that point clouds get the correctly shaded RGB values.

    Args:
        meshes: a list of meshes
        intrinsic_matrix: (3,3) intrinsic matrix shared among all cameras or list of matrices
        extrinsic_matrices: a list of (4,4) homogeneous matrix (from world to camera coordinate)
        width_px: number of pixels of the sensor
        height_px: number of pixels of the sensor
        get_point_cloud: whether to construct a point cloud from rendered images
        pcd_subsample: subsample the point cloud (1 point in every n pixel)
        point_size: render size of points when input is a point cloud
        show_backface: whether to show back faces
        dtype: numpy data type for output
        light_on: whether to enable built-in lighting
        background_color: RGB background color
        shade_image: whether to apply normal-based shading
        rotation_matrix: Optional rotation matrix to apply to normals (3,3)
        light_dir: Light direction for shading [x, y, z]
        unrotate_normals: whether to fix normal coordinate system using H_c2w matrices
        H_c2w_matrices: camera-to-world transformation matrices for normal coordinate fixing

    Returns:
        imgs: a list of (h, w, 3) rgb images
        z_maps: a list of (h, w) depth maps
        pcds: a list of o3d.geometry.PointCloud in world coordinate
        hit_maps: a list of (h, w) valid pixel masks
        normals: a list of (h, w, 3) normal maps
    """
    import torch

    np_dtype = sample_utils.get_np_dtype(dtype)

    if not isinstance(meshes, (list, tuple)):
        meshes = [meshes]

    if isinstance(extrinsic_matrices, np.ndarray) and extrinsic_matrices.ndim == 2:
        extrinsic_matrices = [extrinsic_matrices]

    if isinstance(intrinsic_matrix, np.ndarray) and intrinsic_matrix.ndim == 2:
        intrinsic_matrix = [intrinsic_matrix] * len(extrinsic_matrices)
    assert len(intrinsic_matrix) == len(extrinsic_matrices)

    # Handle H_c2w_matrices for normal coordinate fixing
    if unrotate_normals and H_c2w_matrices is not None:
        if isinstance(H_c2w_matrices, np.ndarray) and H_c2w_matrices.ndim == 2:
            H_c2w_matrices = [H_c2w_matrices] * len(extrinsic_matrices)
        elif isinstance(H_c2w_matrices, np.ndarray) and H_c2w_matrices.ndim == 3:
            H_c2w_matrices = [H_c2w_matrices[i] for i in range(H_c2w_matrices.shape[0])]
        assert len(H_c2w_matrices) == len(extrinsic_matrices)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width_px, height=height_px, visible=False)
    vis.get_render_option().mesh_show_back_face = show_backface
    vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
    vis.get_render_option().light_on = light_on
    vis.get_render_option().background_color = background_color
    for mesh in meshes:
        if isinstance(mesh, o3d.geometry.TriangleMesh):
            mesh.compute_vertex_normals()
        vis.add_geometry(mesh)

    all_points = []
    all_colors = []
    imgs = []
    z_maps = []
    hit_maps = []
    normals = []

    for i in range(len(extrinsic_matrices)):
        assert np.isclose(np.abs(intrinsic_matrix[i][0, 0]), np.abs(intrinsic_matrix[i][1, 1]))
        view_ctl = vis.get_view_control()
        cam_pose_ctl = view_ctl.convert_to_pinhole_camera_parameters()
        cam_pose_ctl.intrinsic.height = height_px
        cam_pose_ctl.intrinsic.width = width_px
        cam_pose_ctl.intrinsic.intrinsic_matrix = intrinsic_matrix[i]
        cam_pose_ctl.extrinsic = extrinsic_matrices[i]
        view_ctl.convert_from_pinhole_camera_parameters(cam_pose_ctl, allow_arbitrary=True)

        if point_size > 0:
            vis.get_render_option().point_size = point_size

        # Set mesh color to Color and render RGB image
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Color
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(do_render=False)
        img = np.asarray(img).astype(dtype=np_dtype)

        # Capture depth
        z_map = vis.capture_depth_float_buffer(do_render=False)
        z_map = np.asarray(z_map).astype(dtype=np_dtype)
        hit_map = np.logical_not(z_map == 0)
        z_map[z_map == 0] = 1e12  # to avoid points at camera center

        # Set mesh color to Normal and render normals
        vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
        vis.poll_events()
        vis.update_renderer()
        normal = vis.capture_screen_float_buffer(do_render=False)
        normal = np.asarray(normal).astype(dtype=np_dtype)

        # # Apply shading if requested
        # if shade_image:
        #     # Convert to torch tensors for shading
        #     img_tensor = torch.from_numpy(img).unsqueeze(0).float()  # [1, H, W, 3]
        #     normal_tensor = torch.from_numpy(normal*2-1).unsqueeze(0).float()  # [1, H, W, 3] in [-1,1]
        #     hit_map_tensor = torch.from_numpy(hit_map).unsqueeze(0).float()  # [1, H, W]

        #     # Apply temporal rotation first (if provided)
        #     if rotation_matrix is not None:
        #         rotation_tensor = torch.from_numpy(rotation_matrix).float()
        #         normal_tensor = rotate_normals(normal_tensor, rotation_tensor)

        #     # Then apply H_c2w transformation to fix normal coordinates (if requested)
        #     if unrotate_normals and H_c2w_matrices is not None:
        #         # Extract rotation part from H_c2w matrix
        #         H_c2w = H_c2w_matrices[i]  # (4, 4)
        #         R_c2w = H_c2w[:3, :3]  # (3, 3) rotation from camera to world
        #         R_c2w_tensor = torch.from_numpy(R_c2w).float()
        #         normal_tensor = rotate_normals(normal_tensor, R_c2w_tensor)

        # Store normals with proper transformations applied
        normal_final = normal * 2 - 1  # Convert to [-1,1] range
        swap_yz = torch.tensor(
            [
                [1, 0, 0],  # x stays x
                [0, -1, 0],  # y becomes z
                [0, 0, -1],  # z becomes y
            ]
        ).float()

        if rotation_matrix is not None or (unrotate_normals and H_c2w_matrices is not None):
            # Apply same transformations to stored normals
            normal_tensor = torch.from_numpy(normal_final).unsqueeze(0).float()  # [1, H, W, 3]

            # Then apply H_c2w transformation to fix normal coordinates (if requested)
            if unrotate_normals and H_c2w_matrices is not None:
                # Extract rotation part from H_c2w matrix
                H_c2w = H_c2w_matrices[i]  # (4, 4)
                R_c2w = H_c2w[:3, :3]  # (3, 3) rotation from camera to world
                R_c2w_tensor = torch.from_numpy(R_c2w).float()
                normal_tensor = rotate_normals(normal_tensor @ swap_yz, R_c2w_tensor.T)

            # Apply temporal rotation first (if provided)
            if rotation_matrix is not None:
                rotation_tensor = torch.from_numpy(rotation_matrix).float()
                normal_tensor = rotate_normals(normal_tensor, rotation_tensor)

            normal_final = normal_tensor.squeeze(0).numpy()

        if shade_image:
            # Apply shading
            normal_tensor = torch.from_numpy(normal_final).unsqueeze(0).float()  # [1, H, W, 3] in [-1,1]
            hit_map_tensor = torch.from_numpy(hit_map).unsqueeze(0).float()  # [1, H, W]
            img_tensor = torch.from_numpy(img).unsqueeze(0).float()  # [1, H, W, 3]
            shaded_img = shade_with_normals(img_tensor, normal_tensor, hit_map_tensor, light_dir)
            img = shaded_img.squeeze(0).numpy().astype(dtype=np_dtype)

        imgs.append(np.copy(img))
        z_maps.append(np.copy(z_map))
        hit_maps.append(np.copy(hit_map))
        normals.append(normal_final)

        # Convert point cloud to world coordinate
        if get_point_cloud:
            H_cam_to_world = rigid_motion.RigidMotion.invert_homogeneous_matrix(cam_pose_ctl.extrinsic)
            points, colors = generate_point(
                rgb_image=img,  # Use shaded image for point cloud colors
                depth_image=z_map,
                intrinsic=cam_pose_ctl.intrinsic.intrinsic_matrix,
                subsample=pcd_subsample,
                world_coordinate=True,
                pose=H_cam_to_world,
                hit_map=hit_map,
            )
            all_points.append(points)
            all_colors.append(colors)

    # clear the visualizer
    vis.clear_geometries()
    vis.destroy_window()
    del cam_pose_ctl
    del view_ctl
    del vis

    # create point cloud
    pcds = []
    for i in range(len(all_points)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points[i])
        pcd.colors = o3d.utility.Vector3dVector(all_colors[i])
        pcds.append(pcd)

    return dict(pcds=pcds, imgs=imgs, z_maps=z_maps, hit_maps=hit_maps, normals=normals)


def generate_point(
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    intrinsic: np.ndarray,
    subsample: int = 1,
    world_coordinate: bool = True,
    pose: np.ndarray = None,
    hit_map: np.ndarray = None,
):
    """
    Generate 3D point coordinates and related rgb feature

    Args:
        rgb_image: (h, w, 3) rgb
        depth_image: (h, w) depth, along z direction (not along individual camera ray)
        intrinsic: (3, 3)
        subsample: int
            resize stride
        world_coordinate: bool
        pose: (4, 4) matrix
            transfer from camera to world coordindate

    Returns:
        points: (N, 3) point cloud coordinates
            in world-coordinates if world_coordinate==True
            else in camera coordinates
        rgb_feat: (N, 3) rgb feature of each point

    Important note:
        The function uses the image coordinate system: x to right, y to "down", z to far.
        If the world coordinate is a different one (say x to right, y to "up", z to us),
        H_c2w need to include the coordinate conversion.
    """
    intrinsic_4x4 = np.identity(4)
    intrinsic_4x4[:3, :3] = intrinsic

    u, v = np.meshgrid(
        range(0, depth_image.shape[1], subsample),
        range(0, depth_image.shape[0], subsample),
    )
    # u: (depth_image.shape[0]//subsample, depth_image.shape[1]//subsample), x
    # v: (depth_image.shape[0]//subsample, depth_image.shape[1]//subsample), y
    if hit_map is not None:
        depth_image[~hit_map] = 0

    d = depth_image[v, u]
    d_filter = d != 0
    mat = np.vstack(
        (
            (u[d_filter] + 0.5) * d[d_filter],
            (v[d_filter] + 0.5) * d[d_filter],
            d[d_filter],
            np.ones_like(u[d_filter]),
        )
    )
    new_points_3d = np.dot(np.linalg.inv(intrinsic_4x4), mat)[:3]
    if world_coordinate:
        new_points_3d_padding = np.vstack((new_points_3d, np.ones((1, new_points_3d.shape[1]))))
        world_coord_padding = np.dot(pose, new_points_3d_padding)
        new_points_3d = world_coord_padding[:3]

    rgb_feat = rgb_image[v, u][d_filter]

    return new_points_3d.T, rgb_feat


def allclose_intrinsic(
    intrinsic1: T.Union[torch.Tensor, np.ndarray],
    intrinsic2: T.Union[torch.Tensor, np.ndarray],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
):
    """
    Check if two homogeneous matrices are close to each other.
    Args:
        intrinsic1:
            (*, 3, 3)
        intrinsic2:
            (*, 3, 3)

    Returns:
        True if the same
    """
    if isinstance(intrinsic1, np.ndarray):
        intrinsic1 = torch.from_numpy(intrinsic1)
    if isinstance(intrinsic2, np.ndarray):
        intrinsic2 = torch.from_numpy(intrinsic2)

    if intrinsic1.shape != intrinsic2.shape:
        return False

    _result = torch.allclose(
        intrinsic1,
        intrinsic2,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )
    if not _result:
        return False

    return True
