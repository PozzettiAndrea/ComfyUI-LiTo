#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# The file implements utils function to use open3d.

import typing as T

import numpy as np
import open3d as o3d

import torch


def create_cylinder(
    x0: np.ndarray,  # (3,)
    x1: np.ndarray,  # (3,)
    radius: float,
) -> o3d.geometry.TriangleMesh:
    """
    Create an open3d cylinder with `radius`. The bottom center is at `x0` and
    the top center is at `x1`.

    Args:
        x0: (3,)  bottom center in the world coordinate
        x1: (3,)  top center in the world coordinate
        radius:

    Returns:
        o3d cylinder
    """
    # Calculate the height of the cylinder
    height = np.linalg.norm(np.array(x1) - np.array(x0))

    # Calculate the center of the cylinder
    center = 0.5 * (np.array(x0) + np.array(x1))

    # Create a cylinder
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)

    # Orient and move the cylinder to the desired position
    rotation_matrix = np.eye(4)
    direction = np.array(x1) - np.array(x0)
    length = np.linalg.norm(direction)
    direction /= length
    # print(f'direction = {direction}')
    rotation_matrix[:3, 2] = direction
    if np.sum(np.array([0, 0, 1.0]) * direction) < 0.9:
        rotation_matrix[:3, 0] = np.cross(np.array([0, 0, 1]), direction)
    else:
        rotation_matrix[:3, 0] = np.cross(np.array([1, 0, 0]), direction)
    rotation_matrix[:3, 1] = np.cross(direction, rotation_matrix[:3, 0])
    # print(f'rotation_matrix = {rotation_matrix}')
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = center
    # print(f'translation_matrix = {translation_matrix}')
    cylinder.transform(rotation_matrix)
    cylinder.transform(translation_matrix)

    cylinder.paint_uniform_color([0.5, 0.5, 0.5])
    return cylinder


def creat_pcd(
    points: np.ndarray,  # (n, 3) np
    color: T.Union[np.ndarray, T.Tuple[float], T.List[float]] = (0.0, 0.0, 1.0),
    normal: T.Union[np.ndarray, None] = None,
) -> o3d.geometry.PointCloud:
    """
    Create an open3d point cloud.

    Args:
        points:
            (n, 3) world coordinate of the points
        color:
            (3,) or (n, 3) the color of the points
        normal:
            (n, 3) or None, vertex normal of the points

    Returns:
        o3d point cloud
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().float().numpy()
    if color is not None and isinstance(color, torch.Tensor):
        color = color.detach().cpu().float().numpy()
    if normal is not None and isinstance(normal, torch.Tensor):
        normal = normal.detach().cpu().float().numpy()

    n, _3 = points.shape
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Set color for all points
    if isinstance(color, (list, tuple)):
        color = np.array(color)
    color = torch.from_numpy(color).expand(n, 3)
    pcd.colors = o3d.utility.Vector3dVector(color.numpy())

    if normal is not None:
        pcd.normals = o3d.utility.Vector3dVector(normal)

    return pcd


def create_spheres(
    points: np.ndarray,  # (n, 3)
    color: T.Union[np.ndarray, T.Tuple[float], T.List[float]] = (0.0, 0.0, 1.0),
    radius: T.Union[float, np.ndarray] = 0.1,
) -> T.List[o3d.geometry.TriangleMesh]:
    """
    Create o3d spheres.

    Args:
        points:
            (n, 3) center locations in the world coordinate
        color:
            (3,) or (n, 3)
        radius:
            (1,) or (n,)
    Returns:
        list of spheres
    """
    n = len(points)
    if isinstance(color, (list, tuple)):
        color = np.array(color)
    color = torch.from_numpy(color).expand(n, 3).numpy()
    assert color.shape == (n, 3)

    if isinstance(radius, (int, float)):
        radius = np.array([radius] * n)
    elif isinstance(radius, (list, tuple)):
        radius = np.array(radius)
    assert radius.shape == (n,)

    # Create spheres for each point
    spheres = []
    for i in range(len(points)):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius[i])
        sphere.translate(points[i], relative=False)
        spheres.append(sphere)

    # Set color for all spheres
    for i in range(len(spheres)):
        spheres[i].paint_uniform_color(color[i])

    return spheres


def create_dumbbell(
    x0: np.ndarray,  # (3,)
    x1: np.ndarray,  # (3,)
    bell_radius: float = 0.1,
    rod_radius: float = 0.01,
    x0_color: T.Union[T.List[float], T.Tuple[float]] = (0.0, 1.0, 0.0),
    x1_color: T.Union[T.List[float], T.Tuple[float]] = (0.0, 0.0, 1.0),
) -> T.List[o3d.geometry.TriangleMesh]:
    """
    Create two connected spheres.
    """
    o3d_s0 = create_spheres(points=[x0], color=[x0_color], radius=bell_radius)[0]
    o3d_s1 = create_spheres(points=[x1], color=[x1_color], radius=bell_radius)[0]
    o3d_rod = create_cylinder(x0=x0, x1=x1, radius=rod_radius)
    return [o3d_s0, o3d_rod, o3d_s1]


def create_ellipsoids(
    center: np.ndarray,  # (n, 3xyz)
    radius: np.ndarray,  # (n, 3xyz)
    R_e2w: np.ndarray,  # (n, 3, 3)
    color: T.Union[np.ndarray, T.Tuple[float], T.List[float]] = (0.0, 0.0, 1.0),
) -> T.List[o3d.geometry.TriangleMesh]:
    """
    Create a list of open3d ellipsoid mesh.

    Args:
        center:
            (n, 3xyz) ellipsoid centers in the world coordinate.
        radius:
            (n, 3xyz) ellipsoid radius for xyz axis (in e coordinate).
            In e coordinate, the 3 axes of the ellipsoid are aligned with xyz axes.
        R_e2w:
            (n, 3, 3) rotation matrix to transform e coordinate to the world coordinate.
        color:
            (3,) (n, 3) color of ellipsoids.

    Returns:
        list of meshes
    """

    if isinstance(center, torch.Tensor):
        center = center.detach().cpu().numpy()
    if isinstance(radius, torch.Tensor):
        radius = radius.detach().cpu().numpy()
    if isinstance(R_e2w, torch.Tensor):
        R_e2w = R_e2w.detach().cpu().numpy()

    n = len(center)
    if isinstance(color, (list, tuple)):
        color = np.array(color)
    color = torch.from_numpy(color).expand(n, 3).numpy()
    assert color.shape == (n, 3)

    if isinstance(radius, (list, tuple)):
        radius = np.reshape(np.array(radius * n), (n, 3))
    assert radius.shape == (n, 3)

    # Create spheres for each point
    meshes = []
    for i in range(n):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1)

        # scale
        vertices = np.asarray(mesh.vertices)
        vertices[:, 0] *= radius[i, 0]  # Scale x-axis
        vertices[:, 1] *= radius[i, 1]  # Scale y-axis
        vertices[:, 2] *= radius[i, 2]  # Scale z-axis
        mesh.vertices = o3d.utility.Vector3dVector(vertices)

        mesh.rotate(R=R_e2w[i])
        mesh.translate(center[i], relative=False)
        meshes.append(mesh)

    # Set color for all spheres
    for i in range(len(meshes)):
        meshes[i].paint_uniform_color(color[i])

    return meshes


def create_normal_ref_sphere(
    center_w: T.List[float] = (0.0, 0.0, 0.0),
    radius: float = 1,
) -> o3d.geometry.TriangleMesh:
    """
    Create a sphere colored from surface normal

    Args:
        center_w:
            (3,) center of the sphere
        radius:
            float, radius of the sphere

    Returns:
        o3d mesh
    """

    # Create a sphere mesh
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=60)

    # Compute normals
    xyz_w = np.asarray(sphere.vertices)
    normal_w = xyz_w / np.linalg.norm(xyz_w, axis=-1)[:, None]
    sphere.vertex_normals = o3d.utility.Vector3dVector(normal_w)  # (n, 3)

    # Convert normals to RGB colors
    colors = (np.asarray(sphere.vertex_normals) + 1) * 0.5  # (n, 3)

    # Assign colors to sphere
    sphere.vertex_colors = o3d.utility.Vector3dVector(colors)  # (n, 3)

    # translate to the center
    sphere.translate(np.array(center_w), relative=False)
    return sphere


def create_dense_voxel_grid_from_o3d_mesh(
    o3d_mesh: o3d.geometry.TriangleMesh,
    num_voxels: T.Union[int, T.List[int]],
    cell_width: float,
    start_xyz_w: T.Union[float, T.List[float]],
) -> np.ndarray:
    """
    Create a dense voxel grid from o3d mesh.  A cell is 1 if it intersects with the mesh,
    and 0 otherwise. Inside the mesh, the cells will be zero.

    Args:
        o3d_mesh:
            open3d mesh
        num_voxels:
            int or (3xyz,) number of voxels along each axis (from x, y, to z).
        cell_width:
            float, width of each cell along all axes.
        start_xyz_w:
            float or (3xyz,) the starting point of the voxel grid (boundary)

    Returns:
        (num_voxels_z, num_voxels_y, num_voxels_z), bool,
        where [0, 0, 0] is the starting voxel (smallest coordinate).
    """

    if isinstance(num_voxels, int):
        num_voxels = [num_voxels]
    if len(num_voxels) == 1:
        num_voxels = num_voxels * 3
    assert len(num_voxels) == 3

    if isinstance(start_xyz_w, float):
        start_xyz_w = [start_xyz_w]
    if len(start_xyz_w) == 1:
        start_xyz_w = start_xyz_w * 3
    assert len(start_xyz_w) == 3

    # create sparse voxel grid (octree)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        input=o3d_mesh,
        voxel_size=cell_width,
    )
    # create center xyz_w of dense voxel grid
    center_xyz_w = create_grid_center_xyz_w(
        num_voxels=num_voxels,
        cell_width=cell_width,
        start_xyz_w=start_xyz_w,
    )  # (res_k, res,j, res_i, 3xyz)
    queries = np.reshape(center_xyz_w, (-1, 3))
    occupied = voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))  # (res_k * res,j * res_i)
    occupied = np.reshape(occupied, center_xyz_w.shape[:-1])  # (res_k, res,j, res_i) bool

    return occupied


def create_dense_voxel_grid_from_o3d_pcd(
    o3d_pcd: o3d.geometry.PointCloud,
    num_voxels: T.Union[int, T.List[int]],
    cell_width: float,
    start_xyz_w: T.Union[float, T.List[float]],
) -> np.ndarray:
    """
    Create a dense voxel grid from o3d point cloud.
    A cell is 1 if it contains a point and 0 otherwise.

    Args:
        o3d_pcd:
            open3d point cloud
        num_voxels:
            int or (3xyz,) number of voxels along each axis (from x, y, to z).
        cell_width:
            float, width of each cell along all axes.
        start_xyz_w:
            float or (3xyz,) the starting point of the voxel grid (boundary)

    Returns:
        (num_voxels_z, num_voxels_y, num_voxels_z), bool,
        where [0, 0, 0] is the starting voxel (smallest coordinate).
    """

    # create sparse voxel grid (octree)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        input=o3d_pcd,
        voxel_size=cell_width,
    )
    # create center xyz_w of dense voxel grid
    center_xyz_w = create_grid_center_xyz_w(
        num_voxels=num_voxels,
        cell_width=cell_width,
        start_xyz_w=start_xyz_w,
    )  # (res_k, res,j, res_i, 3xyz)
    queries = np.reshape(center_xyz_w, (-1, 3))
    occupied = voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))  # (res_k * res,j * res_i)
    occupied = np.reshape(occupied, center_xyz_w.shape[:-1])  # (res_k, res,j, res_i) bool

    return occupied


def create_grid_center_xyz_w(
    num_voxels: T.Union[int, T.List[int]],
    cell_width: float,
    start_xyz_w: T.Union[float, T.List[float]],
):
    """
    Create a grid of xyz_w that correspond to the center of individual
    cell in a dense voxel grid.

    Args:
        num_voxels:
            int or (3xyz,) number of voxels along each axis (from x, y, to z).
        cell_width:
            float, width of each cell along all axes.
        start_xyz_w:
            float or (3xyz,) the starting point of the voxel grid (boundary)

    Returns:
        (num_voxels_z, num_voxels_y, num_voxels_z, 3xyz), float32
        where [0, 0, 0] is the starting voxel (smallest coordinate).
    """

    if isinstance(num_voxels, int):
        num_voxels = [num_voxels]
    if len(num_voxels) == 1:
        num_voxels = num_voxels * 3
    assert len(num_voxels) == 3

    if isinstance(start_xyz_w, (int, float)):
        start_xyz_w = [float(start_xyz_w)]
    if len(start_xyz_w) == 1:
        start_xyz_w = start_xyz_w * 3
    assert len(start_xyz_w) == 3

    xs = (np.arange(num_voxels[0]) + 0.5).astype(np.float32) * cell_width + start_xyz_w[0]  # (res,) [-1, 1]
    ys = (np.arange(num_voxels[1]) + 0.5).astype(np.float32) * cell_width + start_xyz_w[1]  # (res,) [-1, 1]
    zs = (np.arange(num_voxels[2]) + 0.5).astype(np.float32) * cell_width + start_xyz_w[2]  # (res,) [-1, 1]
    Z, Y, X = np.meshgrid(
        zs,
        ys,
        xs,
        indexing="ij",
    )  # (res_k, res,j, res_i)
    center_xyz_w = np.stack([X, Y, Z], axis=-1)  # (res_k, res,j, res_i, 3xyz)

    return center_xyz_w


def save_pointcloud_as_html(
    points: T.Union[torch.Tensor, np.ndarray],
    color: T.Union[torch.Tensor, np.ndarray] = None,
    filename: str = "pointcloud.html",
    point_size: float = 2,
):
    """
    Save point cloud to HTML using Plotly.

    Args:
    - xyz (torch.Tensor or np.ndarray): Point cloud of shape (N, 3).
    - filename (str): Output HTML file name.
    - point_size (int): Size of the points.
    """
    import plotly.graph_objects as go

    import torch

    # Convert to NumPy if it's a torch tensor
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    if color is not None and isinstance(color, torch.Tensor):
        color = color.detach().cpu().numpy()

    # Extract x, y, z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    if color is None:
        color = z
    else:
        color_rgb = np.clip(color, 0, 1)  # Make sure it's in [0, 1]
        color_hex = ["rgb({},{},{})".format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in color_rgb]
        color = color_hex

    # Create scatter plot
    print(f"beginning scatter3d", flush=True)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=point_size, color=color, colorscale="Viridis", opacity=1),
            )
        ]
    )
    print(f"finished scatter3d", flush=True)

    # Customize layout
    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"), margin=dict(l=0, r=0, b=0, t=0))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=[x.min(), x.max()]),
            yaxis=dict(title="Y", range=[y.min(), y.max()]),
            zaxis=dict(title="Z", range=[z.min(), z.max()]),
            aspectmode="manual",  # Use the actual data ranges
            aspectratio=dict(x=(x.max() - x.min()), y=(y.max() - y.min()), z=(z.max() - z.min())),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    # Save to HTML
    if filename is not None:
        fig.write_html(filename)
        print(f"Saved point cloud to {filename}")

    return fig
