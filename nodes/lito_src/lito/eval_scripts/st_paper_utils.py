#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements visualization util functions used by shape tokenization paper.
import copy
import gc
import math
import os
from timeit import default_timer as timer
import typing as T

import imageio.v3 as iio
import matplotlib
import numpy as np
import open3d as o3d

import torch

from plibs import gs_utils, o3d_utils, render as p_render, rigid_motion

try:
    from lito.trainers import st_trainer
except ImportError:
    st_trainer = None


@torch.inference_mode()
def save_pcd(
    xyz_w: torch.Tensor,
    rgb: T.Optional[torch.Tensor],
    normal_w: T.Optional[torch.Tensor],
    out_pcd_filename: T.Optional[str],
    out_logp_filename: T.Optional[str] = None,
    out_gaussian_filename: T.Optional[str] = None,
    gaussian_point_radius: float = 0.005,
    gaussian_opacity: float = 1.0,
    use_2d_gaussian: bool = False,
    realign_normal_w: bool = False,
    realign_k: int = 100,
    use_normal_as_rgb: bool = False,
    use_tonemap: bool = False,
    tonemap_scale: float = 1.0,
    tonemap_gamma: float = 2.2,
    # filter
    outlier_removal_method: T.Union[str, T.List[str]] = "none",
    outlier_nb_neighbors: int = 3,
    outlier_std_ratio: float = 2.0,
    logp: T.Optional[torch.Tensor] = None,
    th_min_logp: float = None,
    th_logp_quantile: float = 0.1,
    # render
    rasterize_pcd: bool = False,
    out_rasterize_dir: str = None,
    H_c2w: torch.Tensor = None,  # (q, 4, 4)
    intrinisc: torch.Tensor = None,  # (q, 3, 3)
    width_px: int = 512,
    height_px: int = 512,
    rasterize_point_size: T.Union[int, T.List[int]] = -1,
    video_filename: str = None,
    gif_filename: str = None,
    fps: float = 10,
    normal_rgb_toward_pinhole: bool = False,
    background_color: T.List[float] = (1.0, 1.0, 1.0),
):
    """
    Save point cloud as ply and optionally gaussian.

    Args:
        xyz_w:
            (n, 3xyz_w)
        rgb:
            (n, 3rgb) or None
        normal_w:
            (n, 3xyz_w)
        realign_normal_w:
            whether to use open3d's to reorient normal_w
        realign_k:
            number of points to use to compute manifold in normal reorientation
        use_normal_as_rgb:
            whether to use normal as rgb
        outlier_removal_method:
            'none'
            'statistical'
            'logp'
        logp:
            (n,)
        th_min_logp:
            min of logp to be considered as valid
        th_logp_quantile:
            threshold for logp filtering [0, 1]

        normal_rgb_toward_pinhole:
            when using normal as rgb, whether to make the rgb (normal) pointing
            toward the camera pinhole

    Returns:

    """

    o3d_pcd = o3d_utils.creat_pcd(
        points=xyz_w.detach().cpu().float().numpy(),  # (n, 3)
        color=rgb.detach().cpu().float().numpy() if rgb is not None else (0.5, 0.5, 0.5),  # (n, 3)
        normal=normal_w.detach().cpu().float().numpy() if normal_w is not None else None,  # (n, 3)
    )

    if realign_normal_w:
        print(f"realigning normal with k = {realign_k}")
        stime = timer()
        o3d_pcd.orient_normals_consistent_tangent_plane(k=realign_k)
        print(f"realigning normal used {timer() - stime:.3f} secs")

    # remove outlier
    if isinstance(outlier_removal_method, str):
        outlier_removal_method = [outlier_removal_method]

    for removal_method in outlier_removal_method:
        if removal_method == "none":
            pass
        elif removal_method == "statistical":
            print(
                f"using remove_statistical_outlier with "
                f"outlier_nb_neighbors = {outlier_nb_neighbors}, "
                f"outlier_std_ratio = {outlier_std_ratio}"
            )
            stime = timer()
            o3d_pcd, ind = o3d_pcd.remove_statistical_outlier(
                nb_neighbors=outlier_nb_neighbors,
                std_ratio=outlier_std_ratio,
            )
            print(f"finished, used {timer() - stime:.3f} secs")
        elif removal_method == "logp":
            assert logp is not None
            th = -np.inf  # (n,)
            if th_logp_quantile is not None:
                th_logp = torch.quantile(logp, q=th_logp_quantile).item()
                th = max(th, th_logp)

            if th_min_logp is not None:
                th = max(th, th_min_logp)

            valid_mask = (logp >= th).detach().cpu().numpy()  # (n,)

            print(
                f"using logp filtering with: \n"
                f"  th_logp_quantile = {th_logp_quantile}\n"
                f"  th_min_logp = {th_min_logp}\n"
                f"  th = {th}\n"
                f"  valid = {torch.from_numpy(valid_mask).float().mean() * 100} %\n"
                f"  after filtering, mean logp = {logp[valid_mask].mean()}"
            )

            o3d_pcd.points = o3d.utility.Vector3dVector(np.asarray(o3d_pcd.points)[valid_mask])
            if o3d_pcd.colors is not None:
                o3d_pcd.colors = o3d.utility.Vector3dVector(np.asarray(o3d_pcd.colors)[valid_mask])
            if o3d_pcd.normals is not None:
                o3d_pcd.normals = o3d.utility.Vector3dVector(np.asarray(o3d_pcd.normals)[valid_mask])

            logp = logp[valid_mask]  # (num_points',)

        else:
            raise NotImplementedError

    if out_logp_filename is not None and logp is not None:
        os.makedirs(os.path.dirname(out_logp_filename), exist_ok=True)
        bb, ext = os.path.splitext(out_logp_filename)
        torch.save(logp, f"{bb}.pth")
        _o3d_pcd = copy.deepcopy(o3d_pcd)
        normalized_logp = (logp.exp() + 1).log()  # (n,)   map to [0, inf]
        normalized_logp = normalized_logp / normalized_logp.max()  # (n,)  map to [0, 1]

        # # plot histogram
        # counts, bin_edges = np.histogram(
        #     normalized_logp.detach().cpu().float().numpy(),
        #     bins=10,
        # )
        # print('score norm at sample points')
        # fig = tpl.figure()
        # fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=False)
        # fig.show()

        colormap = matplotlib.colormaps.get_cmap("viridis")
        logp_rgb = colormap(normalized_logp.detach().cpu().float().numpy())[..., :3]  # (n, 3rgb)
        _o3d_pcd.colors = o3d.utility.Vector3dVector(logp_rgb)
        o3d.io.write_point_cloud(f"{bb}.ply", _o3d_pcd)
        del _o3d_pcd

    o3d_pcds = [o3d_pcd]
    out_pcd_filenames = [out_pcd_filename]
    out_gaussian_filenames = [out_gaussian_filename]
    out_rasterize_dirs = [out_rasterize_dir]
    gif_filenames = [gif_filename]
    video_filenames = [video_filename]
    normal_rgb_toward_pinholes = [False]
    if use_normal_as_rgb:
        postfix = "_normal"
        o3d_pcd_normal = copy.deepcopy(o3d_pcd)
        o3d_pcd_normal.colors = o3d.utility.Vector3dVector((np.asarray(o3d_pcd_normal.normals) + 1) * 0.5)
        o3d_pcds.append(o3d_pcd_normal)
        normal_rgb_toward_pinholes.append(normal_rgb_toward_pinhole)
        if out_pcd_filename is not None:
            nn, ext = os.path.splitext(out_pcd_filename)
            out_pcd_filenames.append(f"{nn}{postfix}{ext}")
        else:
            out_pcd_filenames.append(None)
        if out_gaussian_filename is not None:
            nn, ext = os.path.splitext(out_gaussian_filename)
            out_gaussian_filenames.append(f"{nn}{postfix}{ext}")
        else:
            out_gaussian_filenames.append(None)
        if out_rasterize_dir is not None:
            out_rasterize_dirs.append(f"{out_rasterize_dir.rstrip('/')}{postfix}")
        else:
            out_rasterize_dirs.append(None)
        if gif_filename is not None:
            nn, ext = os.path.splitext(gif_filename)
            gif_filenames.append(f"{nn}{postfix}{ext}")
        else:
            gif_filenames.append(None)
        if video_filename is not None:
            nn, ext = os.path.splitext(video_filename)
            video_filenames.append(f"{nn}{postfix}{ext}")
        else:
            video_filenames.append(None)

    if use_tonemap:
        postfix = "_tonemapped"
        o3d_pcd_tonemap = copy.deepcopy(o3d_pcd)
        _rgb = np.asarray(o3d_pcd_tonemap.colors)
        _rgb = np.power(_rgb, 1.0 / tonemap_gamma) * tonemap_scale
        o3d_pcd_tonemap.colors = o3d.utility.Vector3dVector(_rgb)
        o3d_pcds.append(o3d_pcd_tonemap)
        normal_rgb_toward_pinholes.append(False)
        if out_pcd_filename is not None:
            nn, ext = os.path.splitext(out_pcd_filename)
            out_pcd_filenames.append(f"{nn}{postfix}{ext}")
        else:
            out_pcd_filenames.append(None)
        if out_gaussian_filename is not None:
            nn, ext = os.path.splitext(out_gaussian_filename)
            out_gaussian_filenames.append(f"{nn}{postfix}{ext}")
        else:
            out_gaussian_filenames.append(None)
        if out_rasterize_dir is not None:
            out_rasterize_dirs.append(f"{out_rasterize_dir.rstrip('/')}{postfix}")
        else:
            out_rasterize_dirs.append(None)
        if gif_filename is not None:
            nn, ext = os.path.splitext(gif_filename)
            gif_filenames.append(f"{nn}{postfix}{ext}")
        else:
            gif_filenames.append(None)
        if video_filename is not None:
            nn, ext = os.path.splitext(video_filename)
            video_filenames.append(f"{nn}{postfix}{ext}")
        else:
            video_filenames.append(None)

    # save point cloud
    total = len(o3d_pcds)
    assert len(out_pcd_filenames) == total
    assert len(out_gaussian_filenames) == total
    assert len(out_rasterize_dirs) == total
    assert len(gif_filenames) == total
    assert len(video_filenames) == total
    assert len(normal_rgb_toward_pinholes) == total

    for i in range(total):
        o3d_pcd = o3d_pcds[i]
        out_pcd_filename = out_pcd_filenames[i]
        out_gaussian_filename = out_gaussian_filenames[i]
        out_rasterize_dir = out_rasterize_dirs[i]
        gif_filename = gif_filenames[i]
        video_filename = video_filenames[i]
        _normal_rgb_toward_pinhole = normal_rgb_toward_pinholes[i]

        if o3d_pcd is None:
            continue

        if out_pcd_filename is not None:
            stime = timer()
            os.makedirs(os.path.dirname(out_pcd_filename), exist_ok=True)
            o3d.io.write_point_cloud(out_pcd_filename, o3d_pcd)
            print(f"used {timer() - stime} secs to save point cloud {out_pcd_filename}")

        # save gaussian
        if out_gaussian_filename is not None:
            os.makedirs(os.path.dirname(out_gaussian_filename), exist_ok=True)
            stime = timer()
            gaussians = gs_utils.construct_gaussians_from_point_cloud(
                point_radius=gaussian_point_radius,
                xyz_w=torch.from_numpy(np.asarray(o3d_pcd.points)).float(),
                rgb=torch.from_numpy(np.asarray(o3d_pcd.colors)).float() if o3d_pcd.colors is not None else None,
                normal_w=torch.from_numpy(np.asarray(o3d_pcd.normals)).float() if o3d_pcd.normals is not None else None,
                opacity=gaussian_opacity,
                use_2d_gaussian=use_2d_gaussian,
            )
            gaussians.save_ply(out_gaussian_filename)
            print(f"used {timer() - stime} secs to save gaussian {out_gaussian_filename}")

        # render point cloud
        if rasterize_pcd:
            assert H_c2w is not None
            assert intrinisc is not None
            if isinstance(rasterize_point_size, (int, float)):
                rasterize_point_size = [rasterize_point_size]

            for psize in rasterize_point_size:
                stime = timer()
                # remove normal from o3d_pcd to avoid lighting effect based on normal
                _xyz_w = np.array(o3d_pcd.points)  # (n, 3)
                _ori_normal_w = np.array(o3d_pcd.normals)  # (n, 3)
                if np.prod(_ori_normal_w.shape) <= 1 or not _normal_rgb_toward_pinhole:
                    _o3d_pcd = copy.deepcopy(o3d_pcd)
                    _o3d_pcd.normals = o3d.utility.Vector3dVector([])
                    rdict = p_render.rasterize(
                        meshes=[_o3d_pcd],
                        intrinsic_matrix=intrinisc.detach().cpu().float().numpy(),  # (q, 3, 3)
                        extrinsic_matrices=rigid_motion.inv_homogeneous_tensors(H_c2w).detach().cpu().float().numpy(),
                        # (q, 4, 4)
                        width_px=width_px,
                        height_px=height_px,
                        get_point_cloud=False,
                        point_size=psize,
                        background_color=background_color,
                    )
                    imgs = rdict["imgs"]  # list of (h, w, 3)  rgb  [0, 1]
                else:
                    # rgb is set to normal that points toward the pinhole
                    pinhole_ws = H_c2w[:, :3, 3].detach().cpu().float().numpy()  # (q, 3)
                    imgs = []
                    for iq in range(len(intrinisc)):
                        tmp_o3d_pcd = o3d.geometry.PointCloud()
                        tmp_o3d_pcd.points = o3d.utility.Vector3dVector(_xyz_w)  # (n, 3)

                        mask = np.sum(_ori_normal_w * (pinhole_ws[iq : iq + 1] - _xyz_w), axis=-1) < 0  # (n,)
                        tmp_normal_w = np.copy(_ori_normal_w)
                        tmp_normal_w[mask] *= -1
                        tmp_o3d_pcd.colors = o3d.utility.Vector3dVector((tmp_normal_w + 1) * 0.5)

                        rdict = p_render.rasterize(
                            meshes=[tmp_o3d_pcd],
                            intrinsic_matrix=intrinisc[iq].detach().cpu().float().numpy(),  # (3, 3)
                            extrinsic_matrices=rigid_motion.inv_homogeneous_tensors(H_c2w[iq])
                            .detach()
                            .cpu()
                            .float()
                            .numpy(),
                            # (4, 4)
                            width_px=width_px,
                            height_px=height_px,
                            get_point_cloud=False,
                            point_size=psize,
                            background_color=background_color,
                        )
                        imgs += rdict["imgs"]  # list of (h, w, 3)  rgb  [0, 1]

                print(f"used {timer() - stime} secs to render with psize = {psize}")

                # save individual png
                if out_rasterize_dir is not None:
                    stime = timer()
                    odir = f"{out_rasterize_dir.rstrip('/')}_{psize}"
                    os.makedirs(odir, exist_ok=True)
                    for ii in range(len(imgs)):
                        filename = os.path.join(odir, f"{ii:05d}.png")
                        iio.imwrite(
                            uri=filename,
                            image=np.clip(imgs[ii] * 255, a_min=0, a_max=255).astype(np.uint8),
                        )
                    print(f"used {timer() - stime} secs to save imgs to {odir}")

                # save gif
                if gif_filename is not None:
                    stime = timer()
                    nn, ext = os.path.splitext(gif_filename)
                    gfilename = f"{nn}_{psize}{ext}"
                    p_render.create_gif(
                        images=imgs,
                        filename=gfilename,
                        fps=fps,
                        loop=True,
                    )
                    print(f"used {timer() - stime} secs to save gif to {gfilename}")

                # save video
                if video_filename is not None:
                    stime = timer()
                    nn, ext = os.path.splitext(video_filename)
                    vfilename = f"{nn}_{psize}{ext}"
                    p_render.create_video(
                        images=imgs,
                        filename=vfilename,
                        fps=fps,
                    )
                    print(f"used {timer() - stime} secs to save video to {vfilename}")


@torch.inference_mode()
def realign_normal(
    xyz_w: torch.Tensor,
    normal_w: torch.Tensor,
    k: int = 100,
):
    """
    Reorient normal using minimum spanning tree and manifold method (Hopping et al.)
    implemented in open3d.

    Args:
        xyz_w:
            (b, n, 3) or (n, 3)
        normal_w:
            (b, n, 3) or (n, 3)
        k:
            number of points to use to compute manifold in normal reorientation

    Returns:
        normal_w:
            (b, n, 3) or (n, 3)
    """

    if normal_w.ndim == 2:
        num_dim = 2
        xyz_w = xyz_w.unsqueeze(0)
    else:
        assert normal_w.ndim == 3
        num_dim = 3

    for ib in range(xyz_w.size(0)):
        o3d_pcd = o3d_utils.creat_pcd(
            points=xyz_w[ib].detach().cpu().float().numpy(),  # (n, 3)
            normal=normal_w[ib].detach().cpu().float().numpy(),  # (n, 3)
        )
        o3d_pcd.orient_normals_consistent_tangent_plane(k=k)
        normal_w[ib] = torch.from_numpy(np.asarray(o3d_pcd.normals)).to(dtype=normal_w.dtype, device=normal_w.device)

    if num_dim == 2:
        normal_w = normal_w.squeeze(0)

    return normal_w


@torch.inference_mode()
def sample_point_cloud_from_shape_token(
    model: "st_trainer.ShapeTokenizationTrainer",
    shape_tokens: torch.Tensor,  # (b, num_token, dim_token)
    num_points: int,
    compute_normal_w: bool,
    compute_loglikelihood: bool,
    num_steps: int,
    ode_method: str,
    max_point_chunk: int = 65536,
    logp_num_steps: int = 25,
    logp_ode_method: str = "euler",
    max_logp_point_chunk: int = 16384,
    init_noise_dict: T.Optional[T.Dict[str, T.Optional[torch.Tensor]]] = None,
):
    """
    sample point cloud from shape tokens.

    Args:
        model:
        shape_tokens:
            (b, num_token, dim_token),
        num_points:
            number of points to sample from each shape token
        compute_normal_w:
            whether to estimate normal using score norm
        compute_loglikelihood:
            whether to integrate ODE to get log-likelihood
        num_steps:
            number of ODE steps for sampling point cloud
        ode_method:
            method for point cloud ODE integration

    Returns:
        init_xyz_w:
            (b, num_points, 3)
        xyz_w:
            (b, num_points, 3)
        rgb:
            (b, num_points, 3) or None
        normal_w:
            (b, num_points, 3) or None
        est_normal_w:
            (b, num_points, 3) or None, determined by compute_normal_w
        logp:
            (b, num_points) or None, determined by compute_loglikelihood
    """

    b, num_tokens, dim_token = shape_tokens.shape

    # sample point cloud
    print(f"sampling point cloud", flush=True)
    if init_noise_dict is None:
        init_noise_dict = model.get_conditional_sampling_init_noise(b, num_points)
    sampled_x_dict = model.conditional_sampling(
        shape_latent=shape_tokens,  # (b, num_latent, dim_latent)
        num_steps=num_steps,
        **init_noise_dict,
        method=ode_method,
        max_point_chunk=max_point_chunk,
        printout=True,
        compute_score_direction=compute_normal_w,
        compute_log_likelihood=False,
    )  # (b, num_points, d)

    init_xyz_w = init_noise_dict["init_xyz_w"]  # (b, m, 3) [-1, 1]
    xyz_w = sampled_x_dict["xyz_w"]  # (b, m, 3) or None
    rgb = sampled_x_dict["rgb"]  # (b, m, 3) or None
    normal_w = sampled_x_dict["normal_w"]  # (b, m, 3) or None
    est_normal_w = sampled_x_dict["score_direction_xyz_w"]  # (b, m, 3) or None

    if normal_w is not None:
        normal_w = torch.nn.functional.normalize(normal_w, dim=-1)

    if est_normal_w is not None:
        est_normal_w = torch.nn.functional.normalize(est_normal_w, dim=-1)

    if rgb is not None:
        rgb = (rgb + 1) * 0.5

    # log likelihood
    if compute_loglikelihood:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"computing log likelihood", flush=True)
        logp_dict = model.compute_log_likelihood_of_sample(
            shape_latent=shape_tokens,
            num_steps=logp_num_steps,
            xyz_w=xyz_w,  # (b, num_points, 3)
            rgb=rgb,  # (b, num_points, 3) or None
            normal_w=normal_w,  # (b, num_points, 3) or None
            method=logp_ode_method,
            max_point_chunk=max_logp_point_chunk,
            printout=True,
        )
        logp = logp_dict["log_p1"]  # (b, num_points)
    else:
        logp = None

    return dict(
        init_xyz_w=init_xyz_w,  # (b, num_points, 3)
        xyz_w=xyz_w,  # (b, num_points, 3)
        rgb=rgb,  # (b, num_points, 3) or None
        normal_w=normal_w,  # (b, num_points, 3) or None
        est_normal_w=est_normal_w,  # (b, num_points, 3) or None
        logp=logp,  # (b, num_points) or None
    )


def get_circular_camera(
    fov: float = 40.0,
    up_dir: str = "y",
    r: float = 3.0,
    height: float = 3.0,
    num_imgs: int = 4,
    width_px: int = 512,
    height_px: int = 512,
):
    """
    Create camera pose (H_c2w) and intrinsic
    so that the camera flies on a circular trajectory on a plane
    with height

    Args:
        up_dir:
            'y', 'z'
        r:
            radius of the camera trajectory
        height:
            y or z
        num_imgs:
            number of camera to create on the circle

    Returns:
        H_c2w:
            (q, 4, 4) camera pose
        intrinsic:
            (q, 3, 3) camera intrinsic matrix
        width_px:
            int
        height_px:
            int
    """
    q = num_imgs
    z = height
    H_c2ws = []
    # we assume object is on xy plane (z up)
    for i in range(q):
        if up_dir == "z":
            pinhole_location_w = torch.tensor([r * np.cos(2 * np.pi / q * i), r * np.sin(2 * np.pi / q * i), z]).float()
            up_w = (0.0, 0.0, 1.0)
        elif up_dir == "y":
            pinhole_location_w = torch.tensor([r * np.cos(2 * np.pi / q * i), z, r * np.sin(2 * np.pi / q * i)]).float()
            up_w = (0.0, 1.0, 0.0)
        else:
            raise NotImplementedError

        H_c2w = rigid_motion.get_H_c2w_lookat(
            pinhole_location_w=pinhole_location_w,
            look_at_w=(0.0, 0.0, 0.0),
            up_w=up_w,
            invert_y=True,
        )  # (4, 4)
        H_c2ws.append(H_c2w)
    H_c2w = torch.stack(H_c2ws, dim=0)  # (q, 4, 4)

    fov = np.ones(H_c2w.shape[0]) * fov  # (q,)
    intrinsic = torch.from_numpy(
        p_render.derive_camera_intrinsics(
            width_px=width_px,
            height_px=height_px,
            fov=fov,
        )
    )  # (q, 3, 3)

    return dict(
        H_c2w=H_c2w,  # (q, 4, 4)
        intrinsic=intrinsic,  # (q, 3, 3)
        width_px=width_px,
        height_px=height_px,
    )


def get_circular_camera_v2(
    fov: float = 40.0,
    up_dir: str = "y",
    r: float = 3.5,
    phi: float = 45.0,
    start_theta: float = 0.0,
    end_theta: float = 360.0,
    num_imgs: int = 4,
    width_px: int = 512,
    height_px: int = 512,
    exclude_last: bool = False,
):
    """
    Create camera pose (H_c2w) and intrinsic
    so that the camera flies on a circular trajectory on a plane
    with height

    Args:
        up_dir:
            'y', 'z'
        r:
            radius of the sphere
        theta:
            if z_up:
                the angle (in degree from the +x axis in the anti-clockwise direction, toward +y)
            if y-up:
                the angle (in degree from the +z axis in the anti-clockwise direction, toward +x)
        phi:
            the starting angle from the ground plane (if y up, it is toward +y; if z up, it is toward +z).
            phi=0 is on the ground
        num_imgs:
            number of camera to create on the circle

    Returns:
        H_c2w:
            (q, 4, 4) camera pose
        intrinsic:
            (q, 3, 3) camera intrinsic matrix
        width_px:
            int
        height_px:
            int
    """
    q = num_imgs
    H_c2ws = []

    if not exclude_last:
        thetas = np.linspace(start=start_theta, stop=end_theta, num=num_imgs)  # (num_imgs,)
        phis = np.ones((num_imgs,)) * phi  # (num_imgs,)
    else:
        thetas = np.linspace(start=start_theta, stop=end_theta, num=num_imgs + 1)[:num_imgs]  # (num_imgs,)
        phis = np.ones((num_imgs,)) * phi  # (num_imgs,)

    phis = phis * np.pi / 180.0
    thetas = thetas * np.pi / 180.0

    for i in range(q):
        if up_dir == "z":
            r_ground = r * np.cos(phis[i])
            pinhole_location_w = torch.tensor(
                [
                    r_ground * np.cos(thetas[i]),
                    r_ground * np.sin(thetas[i]),
                    r * np.sin(phis[i]),
                ]
            ).float()
            up_w = (0.0, 0.0, 1.0)
        elif up_dir == "y":
            r_ground = r * np.cos(phis[i])
            pinhole_location_w = torch.tensor(
                [
                    r_ground * np.sin(thetas[i]),
                    r * np.sin(phis[i]),
                    r_ground * np.cos(thetas[i]),  # notice the negative sign
                ]
            ).float()
            up_w = (0.0, 1.0, 0.0)
        else:
            raise NotImplementedError

        H_c2w = rigid_motion.get_H_c2w_lookat(
            pinhole_location_w=pinhole_location_w,
            look_at_w=(0.0, 0.0, 0.0),
            up_w=up_w,
            invert_y=True,
        )  # (4, 4)
        H_c2ws.append(H_c2w)
    H_c2w = torch.stack(H_c2ws, dim=0)  # (q, 4, 4)

    fov = np.ones(H_c2w.shape[0]) * fov  # (q,)
    intrinsic = torch.from_numpy(
        p_render.derive_camera_intrinsics(
            width_px=width_px,
            height_px=height_px,
            fov=fov,
        )
    )  # (q, 3, 3)

    return dict(
        H_c2w=H_c2w,  # (q, 4, 4)
        intrinsic=intrinsic,  # (q, 3, 3)
        width_px=width_px,
        height_px=height_px,
    )


def get_vertical_camera(
    fov: float = 40.0,
    up_dir: str = "y",
    r: float = 3.5,
    theta: float = 0.0,
    start_phi: float = 0.0,
    end_phi: float = 45.0,
    num_imgs: int = 4,
    width_px: int = 512,
    height_px: int = 512,
    exclude_last: bool = False,
):
    """
    Create camera pose (H_c2w) and intrinsic
    so that the camera flies on a vertical circular trajectory
    from start height to end height

    Args:
        up_dir:
            'y', 'z'
        r:
            radius of the sphere
        theta:
            if z_up:
                the angle (in degree from the +x axis in the anti-clockwise direction, toward +y)
            if y-up:
                the angle (in degree from the +z axis in the anti-clockwise direction, toward +x)
        start_phi:
            the starting angle from the ground plane (if y up, it is toward +y; if z up, it is toward +z).
            phi=0 is on the ground
        end_phi:
            the final angle from the ground plane (if y up, it is toward +y; if z up, it is toward +z).
            phi=0 is on the ground
        num_imgs:
            number of camera to create on the circle
        exclude_last:
            if enabled, end_phi will not be included

    Returns:
        H_c2w:
            (q, 4, 4) camera pose
        intrinsic:
            (q, 3, 3) camera intrinsic matrix
        width_px:
            int
        height_px:
            int
    """
    q = num_imgs
    H_c2ws = []

    if not exclude_last:
        phis = np.linspace(start=start_phi, stop=end_phi, num=num_imgs)  # (num_imgs,)
        thetas = np.ones((num_imgs,)) * theta  # (num_imgs,)
    else:
        phis = np.linspace(start=start_phi, stop=end_phi, num=num_imgs + 1)[:num_imgs]  # (num_imgs,)
        thetas = np.ones((num_imgs,)) * theta  # (num_imgs,)

    phis = phis * np.pi / 180.0
    thetas = thetas * np.pi / 180.0

    for i in range(q):
        if up_dir == "z":
            r_ground = r * np.cos(phis[i])
            pinhole_location_w = torch.tensor(
                [
                    r_ground * np.cos(thetas[i]),
                    r_ground * np.sin(thetas[i]),
                    r * np.sin(phis[i]),
                ]
            ).float()
            up_w = (0.0, 0.0, 1.0)
        elif up_dir == "y":
            r_ground = r * np.cos(phis[i])
            pinhole_location_w = torch.tensor(
                [
                    r_ground * np.sin(thetas[i]),
                    r * np.sin(phis[i]),
                    r_ground * np.cos(thetas[i]),  # notice the negative sign
                ]
            ).float()
            up_w = (0.0, 1.0, 0.0)
        else:
            raise NotImplementedError

        H_c2w = rigid_motion.get_H_c2w_lookat(
            pinhole_location_w=pinhole_location_w,
            look_at_w=(0.0, 0.0, 0.0),
            up_w=up_w,
            invert_y=True,
        )  # (4, 4)
        H_c2ws.append(H_c2w)
    H_c2w = torch.stack(H_c2ws, dim=0)  # (q, 4, 4)

    fov = np.ones(H_c2w.shape[0]) * fov  # (q,)
    intrinsic = torch.from_numpy(
        p_render.derive_camera_intrinsics(
            width_px=width_px,
            height_px=height_px,
            fov=fov,
        )
    )  # (q, 3, 3)

    return dict(
        H_c2w=H_c2w,  # (q, 4, 4)
        intrinsic=intrinsic,  # (q, 3, 3)
        width_px=width_px,
        height_px=height_px,
    )


def get_center_up_circle_back_cameras(
    num_imgs: int,
    fov: float = 40.0,
    up_dir: str = "y",
    r: float = 3.5,
    theta: float = 0.0,
    phi: float = 45.0,
    width_px: int = 512,
    height_px: int = 512,
):
    """
    Get a camera trajectory that starts from the (0, 0, r) and then travels upward
    along the sphere, go to phi, then a anti-clockwise circle and back to the starting point.

    Args:
        num_imgs:
            total number of camera poses. We assign the number of images
            for the vertical and circle based on the circumference length.
        fov:
            horizontal field of view of the camera (in degree)
        up_dir:
            'y' or 'z'
        r:
            radius of the sphere
        width_px:
            resolution of the camera
        height_px:
            recolution of the camera

    Returns:
        H_c2w:
            (q, 4, 4) camera pose
        intrinsic:
            (q, 3, 3) camera intrinsic matrix
        width_px:
            int
        height_px:
            int
    """
    assert num_imgs >= 2

    # compute the camera trajectory lengths
    vertical_length = r * phi * np.pi / 180.0  # up
    half_circle_length = r * np.cos(phi * np.pi / 180.0) * np.pi  # half circle
    total_length = 2 * (vertical_length + half_circle_length)

    # assign number cameras
    num_vertical_up_cameras = math.ceil(num_imgs * vertical_length / total_length)
    num_circle_cameras = max(
        0,
        min(
            num_imgs - num_vertical_up_cameras,
            math.ceil(num_imgs * 2 * half_circle_length / total_length),
        ),
    )
    num_vertical_down_cameras = max(0, num_imgs - num_vertical_up_cameras - num_circle_cameras)

    # create cameras
    up_cam_dict = get_vertical_camera(
        fov=fov,
        up_dir=up_dir,
        r=r,
        theta=theta,
        start_phi=0,
        end_phi=phi,
        num_imgs=num_vertical_up_cameras,
        width_px=width_px,
        height_px=height_px,
        exclude_last=True,
    )

    circle_cam_dict = get_circular_camera_v2(
        fov=fov,
        up_dir=up_dir,
        r=r,
        phi=phi,
        start_theta=theta,
        end_theta=theta + 360.0,
        num_imgs=num_circle_cameras,
        width_px=width_px,
        height_px=height_px,
        exclude_last=True,
    )

    down_cam_dict = get_vertical_camera(
        fov=fov,
        up_dir=up_dir,
        r=r,
        theta=theta,
        start_phi=phi,
        end_phi=0,
        num_imgs=num_vertical_down_cameras,
        width_px=width_px,
        height_px=height_px,
        exclude_last=True,
    )

    # combine camera poses
    H_c2w = torch.cat(
        [
            up_cam_dict["H_c2w"],
            circle_cam_dict["H_c2w"],
            down_cam_dict["H_c2w"],
        ],
        dim=0,
    )  # (q, 4, 4)

    intrinsic = torch.cat(
        [
            up_cam_dict["intrinsic"],
            circle_cam_dict["intrinsic"],
            down_cam_dict["intrinsic"],
        ],
        dim=0,
    )  # (q, 3, 3)

    return dict(
        H_c2w=H_c2w,
        intrinsic=intrinsic,
        width_px=up_cam_dict["width_px"],
        height_px=up_cam_dict["height_px"],
    )


def estimate_sphere_trajectory_geodesic_length(
    r: float,
    start_theta: float,
    end_theta: float,
    start_phi: float,
    end_phi: float,
    n_samples: int = 1000,
):
    """
    Estimate the geodesic length of a trajectory on a sphere between two points.

    Uses dense sampling and the haversine formula to approximate the arc length
    of a path that linearly interpolates between the start and end angles.

    Args:
        r: radius of the sphere
        start_theta: starting azimuth angle in degrees
        end_theta: ending azimuth angle in degrees
        start_phi: starting elevation angle in degrees
        end_phi: ending elevation angle in degrees
        n_samples: number of sample points for length estimation (higher = more accurate)

    Returns:
        float: estimated geodesic length of the trajectory
    """
    # Dense sampling of the trajectory
    theta_dense = np.linspace(start_theta, end_theta, n_samples)
    phi_dense = np.linspace(start_phi, end_phi, n_samples)

    # Convert to radians
    theta_rad = np.radians(theta_dense)
    phi_rad = np.radians(phi_dense)

    # Calculate cumulative arc length using haversine formula
    total_length = 0.0
    for i in range(1, n_samples):
        dphi = phi_rad[i] - phi_rad[i - 1]
        dtheta = theta_rad[i] - theta_rad[i - 1]

        # Haversine formula for great circle distance on sphere
        a = np.sin(dphi / 2) ** 2 + np.cos(phi_rad[i - 1]) * np.cos(phi_rad[i]) * np.sin(dtheta / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # Clip to avoid numerical issues
        distance = r * c

        total_length += distance

    return total_length


def get_spiral_camera(
    fov: float = 40.0,
    up_dir: str = "y",
    r: float = 3.5,
    start_theta: float = 0.0,
    end_theta: float = 360.0,
    start_phi: float = 45.0,
    end_phi: float = -45.0,
    num_imgs: int = 10,
    width_px: int = 512,
    height_px: int = 512,
    equal_geodesic: bool = False,
    exclude_last: bool = False,
):
    """
    Create camera poses for a spiral trajectory on a sphere with equal geodesic spacing.

    Args:
        fov: horizontal field of view of the camera (in degree)
        up_dir: 'y' or 'z'
        r: radius of the sphere
        start_theta: starting azimuth angle (in degrees)
        end_theta: ending azimuth angle (in degrees)
        start_phi: starting elevation angle (in degrees)
        end_phi: ending elevation angle (in degrees)
        num_imgs: number of camera poses to create
        width_px: resolution of the camera
        height_px: resolution of the camera
        equal_geodesic: if True, space cameras equally by geodesic distance.
            False is better visually.

    Returns:
        H_c2w: (q, 4, 4) camera pose
        intrinsic: (q, 3, 3) camera intrinsic matrix
        width_px: int
        height_px: int
    """
    assert num_imgs >= 2

    if exclude_last:
        num_imgs = num_imgs + 1

    H_c2ws = []

    # Calculate camera intrinsic matrix using existing function
    intrinsic = torch.from_numpy(
        p_render.derive_camera_intrinsics(
            width_px=width_px,
            height_px=height_px,
            fov=fov,
        )
    )  # (3, 3)

    if equal_geodesic:
        # Calculate total geodesic length of the spiral
        # For a spiral from (start_theta, start_phi) to (end_theta, end_phi)
        # we approximate the path length by sampling densely and summing distances
        n_samples = 1000
        theta_dense = np.linspace(start_theta, end_theta, n_samples)
        phi_dense = np.linspace(start_phi, end_phi, n_samples)

        # Convert to radians for distance calculation
        theta_dense_rad = np.radians(theta_dense)
        phi_dense_rad = np.radians(phi_dense)

        # Calculate cumulative arc lengths
        arc_lengths = [0.0]
        for i in range(1, n_samples):
            # Geodesic distance on sphere between two points
            # Using haversine formula for sphere
            dphi = phi_dense_rad[i] - phi_dense_rad[i - 1]
            dtheta = theta_dense_rad[i] - theta_dense_rad[i - 1]

            # Haversine formula adapted for our coordinate system
            a = (
                np.sin(dphi / 2) ** 2
                + np.cos(phi_dense_rad[i - 1]) * np.cos(phi_dense_rad[i]) * np.sin(dtheta / 2) ** 2
            )
            c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            distance = r * c

            arc_lengths.append(arc_lengths[-1] + distance)

        # Interpolate to get equally spaced points
        total_length = arc_lengths[-1]
        target_lengths = np.linspace(0, total_length, num_imgs)

        # Interpolate theta and phi values for equal arc length spacing
        theta_values = np.interp(target_lengths, arc_lengths, theta_dense)
        phi_values = np.interp(target_lengths, arc_lengths, phi_dense)
    else:
        # Simple linear interpolation in angle space
        theta_values = np.linspace(start_theta, end_theta, num_imgs)
        phi_values = np.linspace(start_phi, end_phi, num_imgs)

    if exclude_last:
        theta_values = theta_values[:-1]
        phi_values = phi_values[:-1]
        num_imgs = num_imgs - 1
    assert len(theta_values) == num_imgs
    assert len(phi_values) == num_imgs

    # Define up vector based on coordinate system
    if up_dir == "y":
        up_w = (0.0, 1.0, 0.0)
    else:  # up_dir == "z"
        up_w = (0.0, 0.0, 1.0)

    for phi_deg, theta_deg in zip(phi_values, theta_values):
        # Convert to radians
        phi_rad = np.radians(phi_deg)
        theta_rad = np.radians(theta_deg)

        # Spherical to Cartesian coordinates
        if up_dir == "y":
            # Y-up coordinate system
            x = r * np.cos(phi_rad) * np.sin(theta_rad)
            y = r * np.sin(phi_rad)
            z = r * np.cos(phi_rad) * np.cos(theta_rad)
        else:  # up_dir == "z"
            # Z-up coordinate system
            x = r * np.cos(phi_rad) * np.cos(theta_rad)
            y = r * np.cos(phi_rad) * np.sin(theta_rad)
            z = r * np.sin(phi_rad)

        pinhole_location_w = (float(x), float(y), float(z))

        # Create camera pose using existing function
        H_c2w = rigid_motion.get_H_c2w_lookat(
            pinhole_location_w=pinhole_location_w,
            look_at_w=(0.0, 0.0, 0.0),
            up_w=up_w,
            invert_y=True,
        )  # (4, 4)

        H_c2ws.append(H_c2w)

    return dict(
        H_c2w=torch.stack(H_c2ws, dim=0),  # (q, 4, 4)
        intrinsic=intrinsic.unsqueeze(0).repeat(num_imgs, 1, 1),  # (q, 3, 3)
        width_px=width_px,
        height_px=height_px,
    )


def get_center_up_spiral_down_cameras(
    num_imgs: int,
    fov: float = 40.0,
    up_dir: str = "y",
    r: float = 3.5,
    num_rotation: int = 1,
    theta: float = 0.0,
    phi: float = 45.0,  # degree
    width_px: int = 512,
    height_px: int = 512,
):
    """
    Get a camera trajectory that starts from the (0, 0, r), travels upward to phi,
    then spirals downward to -phi with equal geodesic spacing throughout the entire trajectory.

    Args:
        num_imgs:
            total number of camera poses
        fov:
            horizontal field of view of the camera (in degree)
        up_dir:
            'y' or 'z'
        r:
            radius of the sphere
        theta:
            starting azimuth angle (in degrees)
        phi:
            maximum elevation angle (in degrees)
        width_px:
            resolution of the camera
        height_px:
            resolution of the camera

    Returns:
        H_c2w:
            (q, 4, 4) camera pose
        intrinsic:
            (q, 3, 3) camera intrinsic matrix
        width_px:
            int
        height_px:
            int
    """
    assert num_imgs >= 2

    # Define the trajectory segments:
    # 1. Vertical up: from phi=0 to phi=+phi (theta stays constant)
    # 2. Spiral down: from phi=+phi to phi=-phi (theta rotates 360 degrees)

    # Estimate geodesic lengths of each segment
    up_length = estimate_sphere_trajectory_geodesic_length(
        r=r,
        start_theta=theta,
        end_theta=theta,  # No rotation during vertical movement
        start_phi=0.0,
        end_phi=phi,
    )

    spiral_down_length = estimate_sphere_trajectory_geodesic_length(
        r=r,
        start_theta=theta,
        end_theta=theta + 360.0 * num_rotation,  # Full rotation
        start_phi=phi,
        end_phi=-phi,
    )

    total_length = up_length + spiral_down_length + up_length

    # Assign cameras proportional to segment lengths
    num_up_cameras = max(1, round(num_imgs * up_length / total_length))
    num_spiral_cameras = num_imgs - num_up_cameras * 2

    # Generate camera poses for each segment
    up_cam_dict = get_vertical_camera(
        fov=fov,
        up_dir=up_dir,
        r=r,
        theta=theta,
        start_phi=0.0,
        end_phi=phi,
        num_imgs=num_up_cameras,
        width_px=width_px,
        height_px=height_px,
        exclude_last=True,  # Avoid duplicate at phi
    )

    spiral_cam_dict = get_spiral_camera(
        fov=fov,
        up_dir=up_dir,
        r=r,
        start_theta=theta,
        end_theta=theta + 360.0 * num_rotation,
        start_phi=phi,
        end_phi=-phi,
        num_imgs=num_spiral_cameras,
        width_px=width_px,
        height_px=height_px,
        equal_geodesic=False,  # it seems equal in angle is better visually
        exclude_last=True,
    )

    up_cam_dict_2 = get_vertical_camera(
        fov=fov,
        up_dir=up_dir,
        r=r,
        theta=theta,
        start_phi=-phi,
        end_phi=0.0,
        num_imgs=num_up_cameras,
        width_px=width_px,
        height_px=height_px,
        exclude_last=False,  # Avoid duplicate at phi
    )

    # Combine camera poses
    H_c2w = torch.cat([up_cam_dict["H_c2w"], spiral_cam_dict["H_c2w"], up_cam_dict_2["H_c2w"]], dim=0)
    intrinsic = torch.cat([up_cam_dict["intrinsic"], spiral_cam_dict["intrinsic"], up_cam_dict_2["intrinsic"]], dim=0)

    return dict(
        H_c2w=H_c2w,
        intrinsic=intrinsic,
        width_px=width_px,
        height_px=height_px,
    )


def determine_crop_and_pad(
    alpha: torch.Tensor,
    keep_optical_axis: bool = True,
    fill_ratio: float = 0.9,
    th_alpha: float = 0.8,
    pad_x_ratio: float = 0.5,
    pad_y_ratio: float = 0.5,
):
    """
    Given an alpha map of an object-centric image,
    determine the crop and pad to create an input
    conditioning image.

    The function determines the crop and pad only ---
    no actually cropping nor resizing is applied.

    Args:
        alpha:
            (h, w) float [0, 1]
        keep_optical_axis:
            bool, if True, it keeps the optical axis at the center of the cropped image.
        fill_ratio:
            float, how large the object occupies the field of view
        th_alpha:
            float
        pad_x_ratio:
            float, [0, 1]. how much padding to be assigned to the left of the padding
        pad_y_ratio:
            float, [0, 1]. how much padding to be assigned to the top of the padding

    Returns:
        crop_x1:
            int, left most along width
        crop_y1:
            int, top most along height
        crop_x2:
            int, right most along width
        crop_y2:
            int, bottom most along height
        pad_left:
            int, how much to pad after cropping on the left
        pad_right:
            int, how much to pad after cropping on the right
        pad_top:
            int, how much to pad after cropping on the top
        pad_bottom:
            int, how much to pad after cropping on the bottom
    Notes:
        the resulted image can be cropped and padded with
        ```
            # rgb: (f, h, w, 3)
            rgb = rgb[:, crop_y1:crop_y2, crop_x1:crop_x2]  # (num_frames, h', w', 3)
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                rgb = torch.nn.functional.pad(
                    rgb,  # (num_frames, h', w', 3)
                    ((0, 0, pad_left, pad_right, pad_top, pad_bottom)),
                    mode="constant",
                    value=0,
                )  # (num_frames, h', w', 3)
        ```
    """
    h = alpha.size(-2)
    w = alpha.size(-1)

    rows = torch.any(alpha > th_alpha, dim=1)  # (h,)  bool
    cols = torch.any(alpha > th_alpha, dim=0)  # (w,)  bool

    y_min, y_max = torch.where(rows)[0][[0, -1]]
    x_min, x_max = torch.where(cols)[0][[0, -1]]

    x_min = x_min.item()
    x_max = x_max.item()
    y_min = y_min.item()
    y_max = y_max.item()

    if keep_optical_axis:
        center_x = w / 2
        center_y = h / 2
        bbox_size = max(
            max(abs(x_max - center_x), abs(x_min - center_x)),
            max(abs(y_max - center_y), abs(y_min - center_y)),
        )
        bbox_size = bbox_size * 2
    else:
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        bbox_size = max(x_max - x_min, y_max - y_min)

    crop_size = int(bbox_size / fill_ratio)  # the image size we should crop before resize
    pad_size = crop_size - bbox_size

    pad_x_front = pad_size * pad_x_ratio
    pad_y_front = pad_size * pad_y_ratio

    crop_x1 = int(center_x - bbox_size / 2 - pad_x_front)
    crop_y1 = int(center_y - bbox_size / 2 - pad_y_front)
    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size

    pad_left = max(0, -crop_x1)
    pad_top = max(0, -crop_y1)
    pad_right = max(0, crop_x2 - w)
    pad_bottom = max(0, crop_y2 - h)

    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(w, crop_x2)
    crop_y2 = min(h, crop_y2)

    x_size = (crop_x2 - crop_x1) + pad_left + pad_right
    y_size = (crop_y2 - crop_y1) + pad_top + pad_bottom
    print(f"x_size: {x_size}, y_size: {y_size}")

    return dict(
        crop_x1=crop_x1,
        crop_y1=crop_y1,
        crop_x2=crop_x2,
        crop_y2=crop_y2,
        pad_left=pad_left,
        pad_right=pad_right,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
    )
