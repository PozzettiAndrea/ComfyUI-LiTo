#
# Copyright (C) 2026 Apple Inc. All rights reserved.
#
# The file implements trainer for Light Tokenization.
from __future__ import annotations  # type annotations stay as strings, never evaluated

import contextlib
import math
import os
import pathlib
import platform
import sys
import tempfile
import time
from timeit import default_timer as timer
import typing as T

import numpy as np
from packaging import version

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

# Optional heavy/training-only imports — only needed for code paths the
# ComfyUI inference wrapper does not call.
try:
    from lightning.pytorch.loggers import TensorBoardLogger
except ImportError:
    TensorBoardLogger = None
try:
    import lpips
except ImportError:
    lpips = None
try:
    import pytorch3d.io
    import pytorch3d.loss
    import pytorch3d.ops
    import pytorch3d.renderer
    import pytorch3d.structures
except ImportError:
    pytorch3d = None
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# Training-only modules removed from the vendored source for the ComfyUI wrapper.
obj_wdset = None
eval_utils_metrics = None

# TRELLIS sparse-structure pipeline IS used at inference time (the gaussian
# decoder seeds its initial coordinates from voxel_ss_pipeline when
# voxel_decoder_config is set in the checkpoint). Wire it up properly instead
# of stubbing.
from lito.integrations.trellis.trellis_sparse_structure import (
    TrellisSparseStructurePipeline,
    get_trellis_sparse_structure_pipeline,
)

from lito.flow import path
from lito.models.point_decoder import GaussianDecoderXv
from lito.models.spoint_encoder import SPointEncoder
from lito.odelibs import ode_solvers
from lito.script_utils import config_utils
from lito.trainers.base import BaseTrainer
from plibs import gs_utils, linalg_utils, ppoint, sh_utils, structures, utils
# lightning_utils removed — only used by `local_rank_first` (distributed
# training helper). Two call sites in this file are already commented out.
lightning_utils = None  # type: ignore
from plibs.ppoint import PackedPoint
from contextlib import nullcontext as _nullcontext

if version.parse(torch.__version__) >= version.parse("2.9.0"):
    torch.backends.fp32_precision = "none"
    torch.backends.cuda.matmul.fp32_precision = "none"
    torch.backends.cudnn.fp32_precision = "none"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.backends.cudnn.rnn.fp32_precision = "tf32"
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
else:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


class LightTokenizationTrainer(BaseTrainer):
    def __init__(
        self,
        mode: str = "tokenizer",
        velocity_outputs: T.List[str] = ("xyz",),
        min_num_encoder_points: int = 1_048_576,
        max_num_encoder_points: int = 1_048_576,
        min_num_flow_points: int = 16384,
        max_num_flow_points: int = 16384,
        center_inputs: T.List[str] = None,
        center_outputs: T.List[str] = None,
        t_eps: float = 1e-3,
        fpoint_encoder_config: T.Dict[str, T.Any] = None,
        time_embedder_config: T.Dict[str, T.Any] = None,
        velocity_decoder_config: T.Dict[str, T.Any] = None,
        gs_decoder_config: T.Dict[str, T.Any] = None,
        voxel_decoder_config: T.Dict[str, T.Any] = None,
        mesh_decoder_config: T.Dict[str, T.Any] = None,
        ode_sampling_method: str = "euler",
        flow_matching_path_type: str = "linear",
        noise_type: str = "gaussian",
        optim_config: T.Dict[str, T.Any] = None,
        mesh_optim_config: T.Dict[str, T.Any] = None,
        debug: bool = False,
        keep_latent_coord: bool = False,
        freeze_encoder: bool = False,
        freeze_velocity_decoder: bool = False,
        freeze_gaussian_decoder: bool = False,
        freeze_voxel_decoder: bool = False,
        freeze_mesh_decoder: bool = False,
    ):
        """
        Args:
            velocity_outputs:
                a list containing 'xyz', 'rgb', 'normal'
                the output of the velocity estimator
            min_num_encoder_points:
                int, min number of points to use as input to the fpoint encoder
            max_num_encoder_points:
                int, max number of points (included) to use as input to the fpoint encoder
            min_num_flow_points:
                int, min number of points to use as input to the flow matching velocity decoder
            max_num_flow_points:
                int, max number of points (included) to use as input to the flow matching velocity decoder
            t_eps:
                a small eps to make sure we do not sample t=0 or t=1 (unstable backward)

            shape_encoder_config:
                target:
                params:

            time_embedder_config:
                target:
                params:

            velocity_decoder_config:
                target:
                params:

            ode_sampling_method:
                see torchdiffeq, e.g, `dopri5`, `euler`.  None: use the default dopri5

            optim_config:
                batch_size:
                    int, suggested batch size of the dataloader
                lr_256:
                    float, learning rate when we have global_batch_size = 256
                gradient_clip_val:
                    float, gradient clipping value
                max_epochs:
                    int, if -1: inf epoch
                max_steps:
                    int, if -1: inf iterations
                num_sanity_val_steps:
                    int, number of validation steps to run before training starts, useful to make sure validation runs ok
                val_check_interval:
                    int, validation is performed every `val_check_interval` iterations
                monitor_loss_name:
                    str, name of the log of monitor, used to save model, e.g, 'loss/total_loss'

                align_normal_first:
                    whether to flip the sign of the gt normal when computing velocity loss
                    default: False

                loss_weight_velocity:
                    default: 1

                loss_weight_kl:
                    loss weight of the kl divergence
                std_posterior:
                    std of the posterior q(s|y), if 0, use mean.

        """
        super().__init__()
        self.save_hyperparameters()
        self.mode = mode
        self.velocity_outputs = velocity_outputs
        self.min_num_encoder_points = min_num_encoder_points
        self.max_num_encoder_points = max_num_encoder_points
        self.min_num_flow_points = min_num_flow_points
        self.max_num_flow_points = max_num_flow_points

        if center_inputs is None:
            center_inputs = set()
        self.center_inputs = center_inputs
        for key in ["xyz_w", "normal_w", "ray_origin_direction_w"]:
            assert key not in self.center_inputs, f"{key}, {self.center_inputs}"

        if center_outputs is None:
            center_outputs = set()
        self.center_outputs = center_outputs
        for key in [
            "xyz_w",
            "quaternion_prenorm",
            "quaternion",
            "scaling_logit",
            "scaling",
            "normal_w",
            "opacity_logit",
            "opacity",
            "rgb_sh",
        ]:
            assert key not in self.center_outputs, f"{key}, {self.center_outputs}"

        self.t_eps = t_eps
        self.fpoint_encoder_config = fpoint_encoder_config

        self.time_embedder_config = time_embedder_config
        self.velocity_decoder_config = velocity_decoder_config
        self.gs_decoder_config = gs_decoder_config
        self.voxel_decoder_config = voxel_decoder_config
        self.mesh_decoder_config = mesh_decoder_config

        self.ode_sampling_method = ode_sampling_method
        self.optim_config = optim_config
        self.mesh_optim_config = mesh_optim_config

        self.noise_type = noise_type
        self.debug = debug

        self.glctx = None

        self.keep_latent_coord = keep_latent_coord

        self.freeze_encoder = freeze_encoder
        self.freeze_velocity_decoder = freeze_velocity_decoder
        self.freeze_gaussian_decoder = freeze_gaussian_decoder
        self.freeze_voxel_decoder = freeze_voxel_decoder
        self.freeze_mesh_decoder = freeze_mesh_decoder

        encoder_module_names = []  # contains the names of encoder modules that we will freeze if required
        velocity_module_names = []
        gaussian_module_names = []
        voxel_module_names = []
        mesh_module_names = []

        # fpoint encoder
        self.fpoint_encoder = config_utils.instantiate_from_config(self.fpoint_encoder_config)
        encoder_module_names.append("fpoint_encoder")
        self.token_shape = self.get_latent_shape()

        # flow matching decoder
        self.use_velocity = self.optim_config["loss_weight_velocity"] > 1e-6
        self.flow_matching_path_type = flow_matching_path_type
        if self.flow_matching_path_type == "linear":
            self.path = path.LinearPath()
        elif self.flow_matching_path_type == "cosine":
            self.path = path.SinusoidalPath()
        else:
            raise NotImplementedError

        if self.time_embedder_config is not None:
            self.flow_t_encoder = config_utils.instantiate_from_config(self.time_embedder_config)
            velocity_module_names.append("flow_t_encoder")
        else:
            self.flow_t_encoder = None

        if self.velocity_decoder_config is not None:
            self.velocity_decoder = config_utils.instantiate_from_config(self.velocity_decoder_config)
            velocity_module_names.append("velocity_decoder")
        else:
            self.velocity_decoder = None

        if self.gs_decoder_config is not None:
            self.gs_decoder = config_utils.instantiate_from_config(self.gs_decoder_config)
            gaussian_module_names.append("gs_decoder")
        else:
            self.gs_decoder = None

        if self.voxel_decoder_config is not None:
            self.voxel_decoder = config_utils.instantiate_from_config(self.voxel_decoder_config)
            voxel_module_names.append("voxel_decoder")
            self.voxel_ss_pipeline: TrellisSparseStructurePipeline = get_trellis_sparse_structure_pipeline()

        else:
            self.voxel_decoder = None
            self.voxel_ss_pipeline = None

        if self.mesh_decoder_config is not None and platform.system() != "Darwin":
            self.mesh_decoder = config_utils.instantiate_from_config(self.mesh_decoder_config)
            mesh_module_names.append("mesh_decoder")

            # sparse structure pipeline
            if getattr(self, "voxel_ss_pipeline", None) is None:
                self.voxel_ss_pipeline: TrellisSparseStructurePipeline = get_trellis_sparse_structure_pipeline()
        else:
            # Mesh decoder uses nvdiffrast/Flexicube which are CUDA-only — skip on macOS.
            self.mesh_decoder = None

        # kl
        self.loss_weight_kl = self.optim_config["loss_weight_kl"]
        self.loss_weight_kl_global = self.optim_config["loss_weight_kl_global"]
        self.std_posterior = self.optim_config["std_posterior"]
        self.sample_posterior = self.std_posterior > 1e-6

        # 3dgs
        self.use_3dgs = self.optim_config["loss_weight_3dgs"] > 1e-9
        self.mip_kernel_size: int = self.optim_config["mip_kernel_size"]

        # lpips
        self.get_lpips_models()

        # timers
        self.data_loading_stime = timer()
        self.compute_stime = timer()

        # freeze encoder, velocity, gaussian decoders
        if self.freeze_encoder:
            for name in encoder_module_names:
                m = getattr(self, name, None)
                if m is None:
                    continue
                else:
                    for param in m.parameters():
                        param.requires_grad = False
                    m.eval()

        if self.freeze_velocity_decoder:
            for name in velocity_module_names:
                m = getattr(self, name, None)
                if m is None:
                    continue
                else:
                    for param in m.parameters():
                        param.requires_grad = False
                    m.eval()

        if self.freeze_gaussian_decoder:
            for name in gaussian_module_names:
                m = getattr(self, name, None)
                if m is None:
                    continue
                else:
                    for param in m.parameters():
                        param.requires_grad = False
                    m.eval()

        if self.freeze_voxel_decoder:
            for name in voxel_module_names:
                m = getattr(self, name, None)
                if m is None:
                    continue
                else:
                    for param in m.parameters():
                        param.requires_grad = False
                    m.eval()

        if self.freeze_mesh_decoder:
            for name in mesh_module_names:
                m = getattr(self, name, None)
                if m is None:
                    continue
                else:
                    for param in m.parameters():
                        param.requires_grad = False
                    m.eval()

    def get_lpips_models(self):
        """
        load lpips models, without register as a nn.module
        """
        # lpips
        self.lpips_model = lpips.LPIPS(net="vgg")
        for name, param in self.lpips_model.named_parameters():
            param.requires_grad = False
        self.lpips_model.eval()
        for name, param in self.lpips_model.named_parameters():
            assert not param.requires_grad, f"{name}"

    def lpips_loss_fn(
        self,
        x: torch.Tensor,  # (b, d, h, w) [-1, 1]
        y: torch.Tensor,  # (b, d, h, w) [-1, 1]
    ):
        """compute lpips with gradient checkpointing"""
        self.lpips_model = self.lpips_model.to(device=x.device)

        b, d, h, w = x.shape
        max_lpips_size = self.optim_config.get("max_lpips_size", -1)

        # crop
        if max_lpips_size > 0 and (h > max_lpips_size or w > max_lpips_size):
            h_last = h - max_lpips_size  # included
            w_last = w - max_lpips_size  # included
            h_start = torch.randint(0, h_last + 1, (1,)).item()  # int
            w_start = torch.randint(0, w_last + 1, (1,)).item()  # int
            h_end = h_start + max_lpips_size
            w_end = w_start + max_lpips_size
        else:
            h_start = 0
            w_start = 0
            h_end = h
            w_end = w

        def fn(a, b, h_start, w_start, h_end, w_end):
            return self.lpips_model(
                a[:, :, h_start:h_end, w_start:w_end],
                b[:, :, h_start:h_end, w_start:w_end],
            )

        if self.optim_config.get("lpips_use_grad_checkpointing", False):
            # use_reentrant=False is recommended on newer PyTorch
            loss = torch.utils.checkpoint.checkpoint(
                fn, x, y, h_start, w_start, h_end, w_end, use_reentrant=False
            )  # (,)
        else:
            # TODO debug
            # loss = torch.nn.functional.l1_loss(x, y, reduction="mean")
            loss = fn(x, y, h_start, w_start, h_end, w_end)
        return loss  # (,)

    # def log(self, *args, **kwargs):
    #     if kwargs.get("sync_dist", False):
    #         name = args[0] if args else kwargs.get("name", "?")
    #         step = self.trainer.global_step if self.trainer else "?"
    #         print(f"[rank {self.global_rank}] sync_dist log: {name} (step={step})", flush=True)
    #     super().log(*args, **kwargs)

        # with lightning_utils.local_rank_first(self):
        #     print(f'finding unused parameters..', flush=True)
        #     unused_params = []
        #     for name, param in self.named_parameters():
        #         if param.grad is None and param.requires_grad:
        #             unused_params.append(name)
        #     if unused_params:
        #         print(f"Unused parameters after backward pass: {unused_params}")

    def estimate_velocity(
        self,
        fpoint_latent: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor,
        max_chunk_size: T.Optional[int] = None,
        latent_coord: T.Optional[ppoint.PackedPoint] = None,
    ):
        """
        Estimate the flow matching velocity at t given xt=x

        Args:
            fpoint_latent:
                (b, num_latent, dim_latent) or (bl, dim_latent)
            t:
                (,) or (b,)
            x:
                (b, m, d)  same dimension as estimated velocity
            latent_coord:
                (bl, dim_latent) needed if fpoint_latent is in packed format

        Returns:
            velocity:
                (b, m, d)
        """

        b, m, d = x.shape
        t = t.expand(b)  # (b,)
        t_encoded = self.flow_t_encoder(t, debug=self.debug)  # (b, dim_cond_feature)

        if self.debug:
            assert t_encoded.isfinite().all(), f"nan: {t_encoded.isnan().any()}, inf: {t_encoded.isinf().any()}"

        # estimate velocity
        if max_chunk_size is None or max_chunk_size < 0 or max_chunk_size >= m:
            est_ut = self.velocity_decoder(
                input_point_cloud=x,  # (b, m, dim_point)
                latent_tokens=fpoint_latent,  # (b, num_latent, dim_latent) or (bl, dim_latent)
                cond_feature=t_encoded,  # (b, dim_cond_feature)
                debug=self.debug,
                latent_coord=latent_coord,  # (bl, dn) or None
            )  # (b, m, d)
        else:
            num_chunks = (m + max_chunk_size - 1) // max_chunk_size
            xs = torch.chunk(x, chunks=num_chunks, dim=1)  # (b, mm, d)
            est_uts = []
            for i in range(len(xs)):
                est_ut = self.velocity_decoder(
                    input_point_cloud=xs[i],  # (b, mm, dim_point)
                    latent_tokens=fpoint_latent,  # (b, num_latent, dim_latent)
                    cond_feature=t_encoded,  # (b, dim_cond_feature)
                    debug=self.debug,
                    latent_coord=latent_coord,  # (bl, dn) or None
                )  # (b, mm, d)
                est_uts.append(est_ut)
            est_ut = torch.cat(est_uts, dim=1)

        return est_ut

    def estimate_gaussians(
        self,
        fpoint_latent: torch.Tensor,
        init_coord: ppoint.PackedPoint,
        latent_coord: T.Optional[ppoint.PackedPoint] = None,
    ) -> T.List[T.Dict[str, torch.Tensor]]:
        """
        Estimate 3d gaussians from shape token.

        Args:
            fpoint_latent:
                (b, num_tokens, dim_token)
            init_coord:
                 (m1+m2+...+mb, dn) the occupied voxel center coordinates in packed format

        Returns:
            xyz_w:
                (b, n, 3xyz_w)  mean of 3d gaussians
            opacity:
                (b, n, 1) [0, 1], opacity after sigmoid
            scaling:
                (b, n, 3xyz) > 0, after exp, std of gaussians
            quaternion:
                (b, n, 4) after normalization.  representing R_g2w
            rgb_sh:
                (b, n, (sh+1)**2, 3rgb)
        """

        if isinstance(self.gs_decoder, (GaussianDecoderXv,)):
            if latent_coord is None:
                b, num_latent, dim_latent = fpoint_latent.shape
                latent_coord = ppoint.PackedPoint(
                    coord=torch.zeros(b * num_latent, 3, dtype=fpoint_latent.dtype, device=fpoint_latent.device),
                    seq_lens=[num_latent] * b,
                )
                fpoint_latent = fpoint_latent.reshape(b * num_latent, dim_latent)
            else:
                assert isinstance(latent_coord, PackedPoint), f"{type(latent_coord)}"
                assert fpoint_latent.ndim == 2, f"{fpoint_latent.shape}"

            gs_dicts = self.gs_decoder(
                latent_coord=latent_coord,
                latent=fpoint_latent,  # (bn, dim_latent)
                given_region_coord=init_coord,  # (m1+m2+...+mb, dn)
                use_grad_checkpointing=self.optim_config.get("gs_decoder_use_grad_checkpointing", False),
            )  # list of (b,), each is a gs_dict: key -> (num_occ_voxels, num_gs_per_voxel, d)

            for key in self.center_outputs:
                for i in range(len(gs_dicts)):
                    if gs_dicts[i].get(key, None) is not None:
                        # (-1, 1) -> (0, 1)
                        gs_dicts[i][key] = (gs_dicts[i][key] + 1) * 0.5

        else:
            raise NotImplementedError

        return gs_dicts

    def inference_init_coords_for_decoder(
        self,
        fpoint_latent: torch.Tensor,
        init_coord_src: str,
        init_coord: T.Optional[ppoint.PackedPoint] = None,
        method_for_sample_xyz: str = "heun",
        steps_for_sample_xyz: int = 100,
        num_points_for_sample_xyz: T.Optional[int] = 100_000,
        given_xyz: T.Optional[torch.Tensor] = None,
        occ_bool_threshold: float = 0.5,
        occ_grid_no_grad: bool = True,
    ) -> T.Dict[str, T.Any]:
        """
        Estimate initial coordinates from shape token during inference for the decoder.
        This assumes
        - encoder is sencoder
        - decoder is GaussianDecoderXv or MeshDecoder
        - not use latent coord

        Args:
            fpoint_latent:
                (b, num_tokens, dim_token)
            init_coord_src:
                'sample_xyz': sample points from velocity decoder and use it to construct init_coord
                'voxel_decoder': use the voxel decoder to estimate voxel
                'given': use the given init_coord
                'given_xyz': compute init_coord based on given xyz. This is useful for oracle evaluation with GT xyz.
            given_xyz:
                (b, num_points, d)
            occ_bool_threshold:
                float, the threshold, above which will be treated as True for occupancy.
            occ_grid_no_grad:
                if True, we disable gradient on occupancy grid readout.

        Returns:
            latent_coord:
                packed point for fake latent coordinates
            init_coord:
                packed point for initial coordinates for the decoder
            extra_info:
                - if init_coord_src == 'sample_xyz':
                    sample_dict:
                        xyz_w: (b, num_points, 3xyz_w)
                    est_occ_grid:
                        (b, 1, res_z, res_y, res_x) bool
                - if init_coord_src == 'voxel_decoder':
                    est_occ_grid:
                        (b, 1, res_z, res_y, res_x) bool
        """
        assert isinstance(self.fpoint_encoder, (SPointEncoder,)), f"{type(self.fpoint_encoder)=}"

        grid_size = 64
        min_xyz_w = -1
        max_xyz_w = 1
        cell_width = (max_xyz_w - min_xyz_w) / grid_size

        # create a fake latent coord (to convery batch size)
        b, num_latent, dim_latent = fpoint_latent.shape
        latent_coord = ppoint.PackedPoint(
            coord=torch.zeros(b * num_latent, 3, dtype=fpoint_latent.dtype, device=fpoint_latent.device),
            seq_lens=[num_latent] * b,
        )

        est_occ_feature = None
        est_occ_grid_logit = None

        out_dict = dict()
        if init_coord_src in ["sample_xyz", "given_xyz"]:
            if init_coord_src == "sample_xyz":
                num_points = num_points_for_sample_xyz
                init_noise_dict = self.get_conditional_sampling_init_noise(
                    b,
                    num_points,
                )  # (b, num_points, d)
                sample_dict = self.conditional_sampling(
                    fpoint_latent=fpoint_latent,  # .reshape(b * num_latent, dim_latent),  # (bl, dim_latent)
                    method=method_for_sample_xyz,  # self.ode_sampling_method,
                    num_steps=steps_for_sample_xyz,  # 100,
                    **init_noise_dict,
                    latent_coord=latent_coord,  # (bl, dn) packed or None
                )  # (b, num_points, d)
                _xyz_w = sample_dict["xyz_w"].float()  # (b, num_points, 3xyz_w)
                # _rgb = sample_dict["rgb"] if sample_dict["rgb"] is not None else None
                # _normal_w = sample_dict["normal_w"] if sample_dict["normal_w"] is not None else None
            elif init_coord_src == "given_xyz":
                assert given_xyz is not None
                assert (given_xyz.ndim == 3) and (given_xyz.shape[0] == b) and (given_xyz.shape[2] == 3), (
                    f"{given_xyz.shape=}"
                )  # (b, num_points, d)
                _xyz_w = given_xyz.to(fpoint_latent.device)
                sample_dict = dict()
            else:
                raise ValueError(f"{init_coord_src=}")

            # # compute occupancy grid from sampled points
            est_occ_grid = []
            for ib in range(_xyz_w.shape[0]):
                tmp_occ_grid = obj_wdset.compute_occupancy_grid(
                    xyz_w=_xyz_w[ib],
                    grid_size=grid_size,
                    min_xyz_w=min_xyz_w,
                    max_xyz_w=max_xyz_w,
                )  # (res_z, res_y, res_x)  bool
                est_occ_grid.append(tmp_occ_grid[None])
            est_occ_grid = torch.stack(est_occ_grid)  # (b, 1, res_z, res_y, res_x)

            # construct init_coord from xyz_w
            # get occupied voxels
            vdict = self.get_voxel(
                xyz_w=_xyz_w,  # (b, num_points, 3)
                cell_width=cell_width,
                return_packed_coord=True,
                min_xyz_w=min_xyz_w,
                max_xyz_w=max_xyz_w,
                grid_size=grid_size,
            )
            init_coord = vdict["coord"]  # (total_occ_cells, 3xyz) packed
            del vdict
            # save intermediate results
            out_dict["sample_dict"] = sample_dict
            out_dict["est_occ_grid"] = est_occ_grid

        elif init_coord_src == "voxel_decoder":
            assert self.voxel_decoder is not None
            # th_occ_prob = 0.5
            vdict = self.estimate_occ_grid(
                latent_token=fpoint_latent,  # (b, l, dl)
                return_occ_grid=True,
                occ_grid_no_grad=occ_grid_no_grad,
            )
            est_occ_grid = vdict["est_occ_grid"]  # (b, 1, res_z, res_y, res_x) [0, 1]
            est_occ_grid = est_occ_grid >= occ_bool_threshold  # (b, 1, res_z, res_y, res_x) bool
            est_occ_feature = vdict["est_ss_latent"]
            est_occ_grid_logit = vdict["est_occ_grid_logit"]

            # convert occ_grid to init_coords
            assert est_occ_grid.size(-3) == grid_size, f"{est_occ_grid.shape=}"
            assert est_occ_grid.size(-2) == grid_size, f"{est_occ_grid.shape=}"
            assert est_occ_grid.size(-1) == grid_size, f"{est_occ_grid.shape=}"
            bijk = torch.nonzero(
                est_occ_grid[:, 0].permute(0, 3, 2, 1),  # (b, res_x, res_y, res_z)
                as_tuple=False,
            )  # (num_voxels, 4bijk)
            xyz = (bijk[:, 1:].float() + 0.5) * cell_width + min_xyz_w  # (num_voxels, 3xyz_w)

            # sort by b
            idx = torch.argsort(bijk[:, 0])  # (num_voxels,)
            xyz = xyz[idx]  # (num_voxels, 3xyz_w)
            seq_lens = torch.bincount(bijk[:, 0], minlength=b)  # (b,)

            init_coord = ppoint.PackedPoint(
                coord=xyz.to(device=fpoint_latent.device),  # (num_voxels, 3xyz_w)
                seq_lens=seq_lens.to(device=fpoint_latent.device),  # (b,)
            )  # (total_occ_cells, 3xyz) packed
            out_dict["est_occ_grid"] = est_occ_grid  # (b, 1, res_z, res_y, res_x) bool

            out_dict["sample_dict"] = None

        elif init_coord_src == "given":
            assert init_coord is not None
        else:
            raise NotImplementedError

        return_dict = dict(
            latent_coord=latent_coord,
            init_coord=init_coord,
            est_occ_grid=out_dict["est_occ_grid"],
            sample_dict=out_dict["sample_dict"],
            est_occ_feature=est_occ_feature,
            est_occ_grid_logit=est_occ_grid_logit,
        )

        return return_dict

    def inference_estimate_gaussians(
        self,
        fpoint_latent: torch.Tensor,
        init_coord_src: T.Optional[str],
        init_coord: T.Optional[ppoint.PackedPoint] = None,
        latent_coord: T.Optional[ppoint.PackedPoint] = None,
        num_points_for_sample_xyz: T.Optional[int] = 100_000,
        given_occ_xyz_w: torch.Tensor = None,  # (b, n, 3xyz_w)
        method_for_sample_xyz: str = "heun",
        steps_for_sample_xyz: int = 100,
    ) -> T.List[T.Dict[str, T.Any]]:
        """
        Estimate 3d gaussians from shape token during inference.

        Args:
            fpoint_latent:
                (b, num_tokens, dim_token)
            init_coord_src:
                'sample_xyz': sample points from velocity decoder and use it to construct init_coord
                'voxel_decoder': use the voxel decoder to estimate voxel
                # 'given': use the given init_coord
                'given_xyz': use the given_occ_xyz_w to compute occ voxel
            init_coord:

            num_points_for_sample_xyz:
                the number of points to sample to obtain occupied voxels for decoder
            given_occ_xyz_w:
                (b, n, 3xyz_w) the point used to compute occ voxels if `given_xyz` is used

        Returns:
            (b,) list of gs_dict
        """
        b, num_latent, dim_latent = fpoint_latent.shape

        if init_coord is None:
            init_coord_ret_dict = self.inference_init_coords_for_decoder(
                fpoint_latent=fpoint_latent,
                init_coord_src=init_coord_src,
                init_coord=init_coord,
                num_points_for_sample_xyz=num_points_for_sample_xyz,
                given_xyz=given_occ_xyz_w,
                method_for_sample_xyz=method_for_sample_xyz,
                steps_for_sample_xyz=steps_for_sample_xyz,
            )
            latent_coord: ppoint.PackedPoint = init_coord_ret_dict["latent_coord"]
            init_coord: ppoint.PackedPoint = init_coord_ret_dict["init_coord"]
            # out_dict = init_coord_ret_dict["extra_info"]
        else:
            if latent_coord is None:
                latent_coord = ppoint.PackedPoint(
                    coord=torch.zeros(b * num_latent, 3, dtype=fpoint_latent.dtype, device=fpoint_latent.device),
                    seq_lens=[num_latent] * b,
                )

        if fpoint_latent.ndim == 3:
            fpoint_latent = fpoint_latent.reshape(b * num_latent, dim_latent)  # (bl, dl)

        gs_dicts = self.estimate_gaussians(
            fpoint_latent=fpoint_latent,
            init_coord=init_coord,
            latent_coord=latent_coord,
        )
        return gs_dicts

    # ------------------------------------------------------------------
    # MLX inference for Gaussian decoder (Apple Silicon)
    # ------------------------------------------------------------------
    _gs_decoder_mlx = None
    _gs_decoder_mlx_step = -1

    def _get_or_build_mlx_gaussian_decoder(self):
        """Lazily construct or refresh the MLX Gaussian decoder.

        Rebuilds when the training step changes.

        Returns:
            MLXGaussianDecoderXv with the current weights.
        """
        from lito.mlx.convert_gaussian_decoder import build_mlx_gaussian_decoder

        current_step = getattr(self, "global_step", 0)
        if self._gs_decoder_mlx is None or self._gs_decoder_mlx_step != current_step:
            self._gs_decoder_mlx = build_mlx_gaussian_decoder(self.gs_decoder)
            self._gs_decoder_mlx_step = current_step
        return self._gs_decoder_mlx

    @torch.no_grad()
    def inference_estimate_gaussians_mlx(
        self,
        fpoint_latent: torch.Tensor,
        init_coord_src: T.Optional[str] = "voxel_decoder",
        init_coord: T.Optional[ppoint.PackedPoint] = None,
        latent_coord: T.Optional[ppoint.PackedPoint] = None,
        num_points_for_sample_xyz: T.Optional[int] = 100_000,
        given_occ_xyz_w: torch.Tensor = None,  # (b, n, 3xyz_w)
        method_for_sample_xyz: str = "heun",
        steps_for_sample_xyz: int = 100,
        mlx_compute_dtype: T.Optional[str] = "bfloat16",
    ) -> T.List[T.Dict[str, T.Any]]:
        """Estimate 3D Gaussians using the MLX backend (Apple Silicon).

        Same interface as ``inference_estimate_gaussians`` but runs the
        GaussianDecoderXv forward pass in MLX to avoid MPS SDPA issues.
        The voxel decoder (init_coord generation) still runs in PyTorch.

        Args:
            fpoint_latent: Shape latent tokens. (b, num_tokens, dim_token)
            init_coord_src: How to initialize coordinates. Only ``"voxel_decoder"``
                is supported.
            init_coord: Pre-computed init coordinates (optional).
            latent_coord: Pre-computed latent coordinates (optional).
            num_points_for_sample_xyz: Num points for sample_xyz path.
            given_occ_xyz_w: GT points for given_xyz path. (b, n, 3xyz_w)
            method_for_sample_xyz: ODE solver method.
            steps_for_sample_xyz: ODE steps.
            mlx_compute_dtype: Compute dtype for MLX (``"bfloat16"`` or ``None``).

        Returns:
            (b,) list of gs_dict, each containing ``xyz_w``, ``scaling``,
            ``quaternion``, ``opacity``, ``rgb_sh``.
        """
        import mlx.core as mx

        b, num_latent, dim_latent = fpoint_latent.shape

        # 1. Compute init_coord via existing PyTorch path
        print(f"inference_init_coords_for_decoder", flush=True)

        stime = timer()
        if init_coord is None:
            init_coord_ret_dict = self.inference_init_coords_for_decoder(
                fpoint_latent=fpoint_latent,
                init_coord_src=init_coord_src,
                init_coord=init_coord,
                num_points_for_sample_xyz=num_points_for_sample_xyz,
                given_xyz=given_occ_xyz_w,
                method_for_sample_xyz=method_for_sample_xyz,
                steps_for_sample_xyz=steps_for_sample_xyz,
            )
            latent_coord = init_coord_ret_dict["latent_coord"]
            init_coord = init_coord_ret_dict["init_coord"]
        else:
            if latent_coord is None:
                latent_coord = ppoint.PackedPoint(
                    coord=torch.zeros(b * num_latent, 3, dtype=fpoint_latent.dtype, device=fpoint_latent.device),
                    seq_lens=[num_latent] * b,
                )
        ttime = timer() - stime
        print(f"  Finished inference_init_coords_for_decoder, took {ttime: .1f} secs", flush=True)

        # Flatten latent to packed format
        with _nullcontext():
            fpoint_latent_packed = fpoint_latent.reshape(b * num_latent, dim_latent)  # (bl, dl)

            # 2. Pre-compute voxelization for localized_voxel self-attention
            print(f"precomute voxel info in packed point", flush=True)
            stime = timer()
            voxel_infos_per_block = None
            perceiver = self.gs_decoder.perceiver
            if perceiver.self_attn_type == "localized_voxel" and perceiver.self_cell_widths is not None:
                init_coord_cpu = ppoint.PackedPoint(
                    coord=init_coord.coord.detach().cpu().float(),
                    seq_lens=init_coord.seq_lens.cpu()
                    if isinstance(init_coord.seq_lens, torch.Tensor)
                    else init_coord.seq_lens,
                )
                num_self_attn = len(perceiver.blocks[0].sa_layers)
                num_blocks = len(perceiver.blocks)

                # Cache voxelization by (cell_width, shift) — typically only 2 unique combos
                voxel_cache = {}
                voxel_infos_per_block = []
                for block_idx in range(num_blocks):
                    self_cell_width = perceiver.self_cell_widths[block_idx]
                    block_voxel_infos = []
                    for sa_idx in range(num_self_attn):
                        shift_ratio = 0.5 * (sa_idx % 2)
                        shift = shift_ratio * self_cell_width
                        cache_key = (self_cell_width, shift)

                        if cache_key not in voxel_cache:
                            bijk_dict = init_coord_cpu.get_bijk_info(
                                cell_width=self_cell_width,
                                shift=shift,
                                attn_backend="pytorch",
                                save_to_cache=False,
                            )
                            # Convert to mx.arrays
                            voxel_info = {
                                "forward_idxs": mx.array(bijk_dict["forward_idxs"].numpy()),
                                "backward_idxs": mx.array(bijk_dict["backward_idxs"].numpy()),
                                "cu_seq_lens": [mx.array(cu.numpy()) for cu in bijk_dict["cu_seq_lens"]],
                                "max_seq_lens": bijk_dict["max_seq_lens"],
                                "chunk_start_idxs": bijk_dict["chunk_start_idxs"],
                            }
                            voxel_cache[cache_key] = voxel_info
                        block_voxel_infos.append(voxel_cache[cache_key])
                    voxel_infos_per_block.append(block_voxel_infos)

            ttime = timer() - stime
            print(f"  Finished compute voxel info, took {ttime: .1f} secs", flush=True)

            # 3. Build/fetch MLX decoder
            print(f"get_or_build_mlx_gaussian_decoder", flush=True)
            stime = timer()
            mlx_decoder = self._get_or_build_mlx_gaussian_decoder()
            ttime = timer() - stime
            print(f"  Finished get_or_build_mlx_gaussian_decoder, took {ttime: .1f} secs", flush=True)

            # Optionally cast to bfloat16 (only on Metal GPU; CPU-only MLX only supports float32)
            _has_metal = hasattr(mx, "metal") and mx.metal.is_available()
            if mlx_compute_dtype == "bfloat16" and _has_metal:
                import mlx.utils as mlx_utils

                def _cast(x):
                    return x.astype(mx.bfloat16) if isinstance(x, mx.array) else x

                print(f"cast mlx decoder to bfloat16", flush=True)
                stime = timer()
                mlx_decoder.update(mlx_utils.tree_map(_cast, mlx_decoder.parameters()))
                mx.eval(mlx_decoder.parameters())
                ttime = timer() - stime
                print(f"  Finished cast to bfloat16, took {ttime: .1f} secs", flush=True)

            # 4. Convert inputs to MLX
            print(f"convert input to mlx", flush=True)
            stime = timer()
            latent_mx = mx.array(fpoint_latent_packed.detach().cpu().float().numpy())  # (bl, dl)
            coord_mx = mx.array(init_coord.coord.detach().cpu().float().numpy())  # (bm, 3)

            q_seq_lens = (
                init_coord.seq_lens.tolist()
                if isinstance(init_coord.seq_lens, torch.Tensor)
                else list(init_coord.seq_lens)
            )
            kv_seq_lens = (
                latent_coord.seq_lens.tolist()
                if isinstance(latent_coord.seq_lens, torch.Tensor)
                else list(latent_coord.seq_lens)
            )

            # Cast inputs if needed (only on Metal GPU)
            if mlx_compute_dtype == "bfloat16" and _has_metal:
                latent_mx = latent_mx.astype(mx.bfloat16)
                coord_mx = coord_mx.astype(mx.bfloat16)

            ttime = timer() - stime
            print(f"  Finished converting input to mlx, took {ttime: .1f} secs", flush=True)

            # 5. Run MLX decoder
            print(f"run mlx gs decoder ", flush=True)
            stime = timer()
            shape_out_mx, color_out_mx = mlx_decoder(
                latent=latent_mx,
                init_query_coord=coord_mx,
                q_seq_lens=q_seq_lens,
                kv_seq_lens=kv_seq_lens,
                voxel_infos_per_block=voxel_infos_per_block,
            )
            mx.eval(shape_out_mx, color_out_mx)
            ttime = timer() - stime
            print(f"  Finished running mlx decoder, took {ttime: .1f} secs", flush=True)

        # 6. Convert outputs back to torch (CPU, float32)
        print(f"convert back to torch ", flush=True)
        stime = timer()
        shape_out = torch.from_numpy(np.array(shape_out_mx.astype(mx.float32)))  # (bm, dim_shape * k)
        color_out = torch.from_numpy(np.array(color_out_mx.astype(mx.float32)))  # (bm, dim_color * k)
        ttime = timer() - stime
        print(f"  Finished converting back to torch, took {ttime: .1f} secs", flush=True)

        # 7. Reshape and run decode_gs in PyTorch CPU
        bm = shape_out.shape[0]
        k = self.gs_decoder.gs_expansion_ratio
        shape_out = shape_out.reshape(bm, k, -1)  # (bm, k, dim_shape)
        color_out = color_out.reshape(bm, k, -1)  # (bm, k, dim_color)

        print(f"decode gs ", flush=True)
        stime = timer()
        shape_out_dict = self.gs_decoder.decode_gs(
            shape_out,
            info=self.gs_decoder.gs_shape_info,
            scaling_logit_bias=self.gs_decoder.scaling_logit_bias,
            scaling_scalar=self.gs_decoder.scaling_scalar,
            min_scaling=self.gs_decoder.min_scaling,
            max_scaling=self.gs_decoder.max_scaling,
        )
        color_out_dict = self.gs_decoder.decode_gs(
            color_out,
            info=self.gs_decoder.gs_color_info,
            scaling_logit_bias=self.gs_decoder.scaling_logit_bias,
            scaling_scalar=self.gs_decoder.scaling_scalar,
            min_scaling=self.gs_decoder.min_scaling,
            max_scaling=self.gs_decoder.max_scaling,
        )
        ttime = timer() - stime
        print(f"  Finished decode gs, took {ttime: .1f} secs", flush=True)

        shape_out_dict.update(color_out_dict)

        if self.gs_decoder.use_unit_opacity:
            opacity = torch.ones(bm, k, 1, dtype=torch.float32)  # (bm, k, 1)
            shape_out_dict["opacity"] = opacity

        # 8. Apply xyz offset + region scaling
        init_coord_cpu = init_coord.coord.detach().cpu().float()  # (bm, 3)
        shape_out_dict["xyz_w"] = (
            shape_out_dict["xyz_w"].sigmoid() * 2 - 1
        ) * self.gs_decoder.region_scaling  # (bm, k, 3) [-r, r]
        shape_out_dict["xyz_w"] = shape_out_dict["xyz_w"] + init_coord_cpu.unsqueeze(-2)  # (bm, k, 3)

        # 9. Pack to list of dicts
        gs_dicts = []
        current_idx = 0
        for ib in range(b):
            seq_len = q_seq_lens[ib]
            end_idx = current_idx + seq_len
            gs_dict = {}
            for key in [
                "xyz_w",
                "scaling",
                "quaternion",
                "opacity",
                "rgb_sh",
                "normal_w",
                "albedo",
                "roughness_metallic",
            ]:
                if shape_out_dict.get(key, None) is None:
                    continue
                gs_dict[key] = shape_out_dict[key][current_idx:end_idx]
            gs_dicts.append(gs_dict)
            current_idx = end_idx

        # 10. Apply center_outputs denormalization
        for key in self.center_outputs:
            for i in range(len(gs_dicts)):
                if gs_dicts[i].get(key, None) is not None:
                    gs_dicts[i][key] = (gs_dicts[i][key] + 1) * 0.5

        return gs_dicts

    def inference_estimate_mesh(
        self,
        fpoint_latent: torch.Tensor,
        init_coord_src: T.Optional[str],
        num_points_for_sample_xyz: T.Optional[int] = 100_000,
        given_occ_xyz_w: torch.Tensor = None,  # (b, n, 3xyz_w)
        method_for_sample_xyz: str = "heun",
        steps_for_sample_xyz: int = 100,
    ) -> T.List[structures.RawMesh]:
        """
        Estimate mesh from shape token during inference.

        Args:
            fpoint_latent:
                (b, num_tokens, dim_token)
            init_coord_src:
                'sample_xyz': sample points from velocity decoder and use it to construct init_coord
                'voxel_decoder': use the voxel decoder to estimate voxel
                # 'given': use the given init_coord
                'given_xyz': use the given_occ_xyz_w to compute occ voxel
            init_coord:

            num_points_for_sample_xyz:
                the number of points to sample to obtain occupied voxels for decoder
            given_occ_xyz_w:
                (b, n, 3xyz_w) the point used to compute occ voxels if `given_xyz` is used

        Returns:
            (b,) list of structures.RawMesh
        """
        b, num_latent, dim_latent = fpoint_latent.shape

        init_coord_ret_dict = self.inference_init_coords_for_decoder(
            fpoint_latent=fpoint_latent,
            init_coord_src=init_coord_src,
            init_coord=None,
            num_points_for_sample_xyz=num_points_for_sample_xyz,
            given_xyz=given_occ_xyz_w,
            method_for_sample_xyz=method_for_sample_xyz,
            steps_for_sample_xyz=steps_for_sample_xyz,
        )
        input_occ_grid = init_coord_ret_dict["est_occ_grid"]  # (b, 1, res_z, res_y, res_x) bool

        # if fpoint_latent.ndim == 3:
        #     fpoint_latent = fpoint_latent.reshape(b * num_latent, dim_latent)  # (bl, dl)

        # convert packed init_coord to occ_bijk
        # min_xyz_w = -1
        # max_xyz_w = 1
        # grid_size = 64
        # cw = (max_xyz_w - min_xyz_w) / grid_size
        # occ_bijk = torch.cat([
        #     init_coord.batch_idxs.unsqueeze(-1),  # (total_occ_cells, 1)
        #     ((init_coord.coord - min_xyz_w) / cw).floor().long(),
        # ], dim=-1)

        raw_meshes = self.estimate_mesh(
            latent_token=fpoint_latent,
            occ_bijk=None,
            input_occ_grid=input_occ_grid,  # (b, 1, res_z, res_y, res_x) bool
        )["raw_meshes"]
        return raw_meshes

    @linalg_utils.disable_tf32_and_autocast()
    def render_gaussians(
        self,
        gs_dicts: T.List[T.Dict[str, torch.Tensor]],  # list of (b,), each is a dict: key -> (*, d)
        H_c2w: torch.Tensor,  # (b, q, 4, 4)
        intrinsic: torch.Tensor,  # (b, q, 3, 3)
        width_px: int,
        height_px: int,
        given_rgb_sh_degree: int | None = None,
    ):
        """
        Render gaussians estimated from shape tokens

        Args:
            gs_dicts:
                list of (b,), each is a dict containing:
                    xyz_w:
                        (n, 3xyz_w)  mean of 3d gaussians
                    opacity:
                        (n, 1)  [0, 1], opacity after sigmoid
                    scaling:
                        (n, 3),  > 0, after exp, std of gaussians
                    quaternion:
                        (n, 4), after normalization.  representing R_g2w
                    rgb_sh:
                        (n, (sh+1)**2, 3rgb)
            H_c2w:
                (b, q, 4, 4)  camera pose in the world coordinate
            intrinsic:
                (b, q, 3, 3)  camera intrinsics
            width_px:
                horizontal resolution
            height_px:
                vertical resolution

            given_rgb_sh_degree:
                If None, we will render with all spherical harmonics (SH) degrees.
                If not None, this function renders image with SH degree up to the given one.

        Returns:
            premultiplied_rgb:
                (b, q, h, w, 3rgb) [0, 1], premultiplied with alpha
            alpha:
                (b, q, h, w, 1) [0, 1]
            normal_w:
                (b, q, h, w, 3xyz_w) normalized, pointing toward camera pinhole, straight
            premultiplied_normal_w_raw:
                (b, q, h, w, 3xyz_w) unnormalized, premultiplied with alpha, raw output from rendering
        """

        b = len(gs_dicts)
        q = H_c2w.size(1)

        all_out_dict = dict()
        for ib in range(b):
            gs_dict = gs_dicts[ib]

            xyz_w = gs_dict["xyz_w"].reshape(-1, 3)  # (n, 3)
            n = xyz_w.size(0)
            scaling = gs_dict["scaling"].reshape(n, 3)  # (n, 3)
            quaternion = gs_dict["quaternion"].reshape(n, 4)  # (n, 4)
            opacity = gs_dict["opacity"].reshape(n, 1)  # (n, 1)
            rgb_sh = gs_dict["rgb_sh"].reshape(n, -1, 3)  # (n, sh, 3)
            sh_degree = sh_utils.get_sh_degree_from_total_dim(rgb_sh.size(-2))

            if given_rgb_sh_degree is not None:
                # only render with specific number of spherical harmonics degree
                assert sh_degree >= given_rgb_sh_degree, f"{sh_degree=}, {given_rgb_sh_degree=}"
                sh_degree = given_rgb_sh_degree
                sh_n_coeffs = sh_utils.get_total_coeffs_for_sh_degree(sh_degree)
                rgb_sh = rgb_sh[..., :sh_n_coeffs, :]

            features = []
            feature_start_dim_dict = dict()
            feature_dim_dict = dict()
            current_start_dim = 0
            if gs_dict.get("normal_w", None) is not None:
                features.append(gs_dict["normal_w"].reshape(n, 3))  # (n, 3)
                feature_start_dim_dict["normal_w"] = current_start_dim
                feature_dim_dict["normal_w"] = 3
                current_start_dim += 3
            if len(features) > 0:
                features = torch.cat(features, dim=-1)  # (n, d)
            else:
                features = None

            # render
            odict = dict()
            for iq in range(q):
                out = gs_utils.render_3dgs_gsplat(
                    H_c2w=H_c2w[ib, iq],  # (4, 4)
                    intrinsic=intrinsic[ib, iq],  # (3, 3)
                    width_px=width_px,
                    height_px=height_px,
                    sh_degree=sh_degree,
                    xyz_w=xyz_w,  # (n, 3)
                    scaling=scaling,  # (n, 3)
                    quaternion=quaternion,  # (n, 4)
                    opacity=opacity,  # (n, 1)
                    rgb_sh=rgb_sh,  # (n, sh, 3)
                    feature=features,  # (n, d)
                    render_depth=False,
                    mip_kernel_size=self.mip_kernel_size,
                )
                for key in out:
                    if out[key] is None:
                        continue

                    if key not in odict:
                        odict[key] = []
                    odict[key].append(out[key])

            for key in odict:
                assert None not in odict[key], f"{key}"
                assert len(odict[key]) == q, f"{len(odict[key]) =}"
                odict[key] = torch.stack(odict[key], dim=0) if len(odict[key]) > 1 else odict[key][0].unsqueeze(0)
                # premultiplied_rgb: (q, h, w, 3rgb) [0, 1]
                # premultiplied_feature: (q, h, w, d)
                # alpha: (q, h, w, 1) [0, 1]

            if odict.get("premultiplied_feature", None) is not None:
                features = odict["premultiplied_feature"]  # (q, h, w, d)
                for key in feature_start_dim_dict:
                    arr = features[
                        ..., feature_start_dim_dict[key] : feature_start_dim_dict[key] + feature_dim_dict[key]
                    ]  # (q, h, w, d')
                    odict[f"premultiplied_{key}"] = arr  # (q, h, w, d')
                del features
                del odict["premultiplied_feature"]

            for key in odict:
                if key not in all_out_dict:
                    all_out_dict[key] = []
                all_out_dict[key].append(odict[key])

        for key in all_out_dict:
            assert None not in all_out_dict[key], f"{key}"
            all_out_dict[key] = (
                torch.stack(
                    all_out_dict[key],
                    dim=0,
                )
                if len(all_out_dict[key]) > 1
                else all_out_dict[key][0].unsqueeze(0)
            )
            # premultiplied_rgb: (b, q, h, w, 3rgb) [0, 1]
            # premultiplied_normal_w: (b, q, h, w, 3xyz_w)
            # alpha: (b, q, h, w, 1) [0, 1]

        # normalize normal map and
        if all_out_dict.get("premultiplied_normal_w", None) is not None:
            est_normal_w_raw = all_out_dict["premultiplied_normal_w"]  # (b, q, h, w, 3xyz_w) unnormalized
            est_normal_w = torch.nn.functional.normalize(est_normal_w_raw, dim=-1)  # (b, q, h, w, 3xyz_w) normalized

            # make sure normal points toward pinhole
            with torch.no_grad():
                ro_w, rd_w = utils.generate_camera_rays(
                    cam_poses=H_c2w.reshape(b * q, 4, 4),
                    intrinsics=intrinsic.reshape(b * q, 3, 3),
                    width_px=width_px,
                    height_px=height_px,
                    use_quick_inv_intrinsic=True,
                )
                rd_w = rd_w.reshape(b, q, height_px, width_px, 3)
                opp_dir = (est_normal_w.detach() * rd_w).sum(dim=-1) <= 0  # (b, q, h, w) {0, 1}
                opp_dir = opp_dir * 2 - 1  # {-1, 1}

            est_normal_w = est_normal_w * opp_dir.unsqueeze(-1)  # (b, q, h, w, 3)
            all_out_dict["normal_w"] = est_normal_w  # (b, q, h, w, 3), straight because of normalization
            all_out_dict["premultiplied_normal_w_raw"] = (
                est_normal_w_raw  # (b, q, h, w, 3), unnormalized, premultiplied
            )

            del all_out_dict["premultiplied_normal_w"]

        return all_out_dict

    def get_conditional_sampling_init_noise(
        self,
        *shape,
        scale: float = 1.0,
    ) -> T.Dict[str, T.Union[torch.Tensor, None]]:
        """
        Get the initial noise for conditional sampling.

        Args:
            *shape:

        Returns:
            init_xyz_w:
                (*shape, 3) or None
            init_rgb:
                (*shape, 3) or None
            init_normal_w:
                (*shape, 3) or None
        """

        if "xyz" in self.velocity_outputs:
            if self.noise_type == "gaussian":
                init_xyz_w = torch.randn(*shape, 3, dtype=self.dtype, device=self.device)
            elif self.noise_type == "uniform":
                init_xyz_w = torch.rand(*shape, 3, dtype=self.dtype, device=self.device) * 2 - 1
            else:
                raise NotImplementedError
            init_xyz_w = init_xyz_w * scale
        else:
            init_xyz_w = None

        if "rgb" in self.velocity_outputs:
            if self.noise_type == "gaussian":
                init_rgb = torch.randn(*shape, 3, dtype=self.dtype, device=self.device)
            elif self.noise_type == "uniform":
                init_rgb = torch.rand(*shape, 3, dtype=self.dtype, device=self.device) * 2 - 1
            else:
                raise NotImplementedError
            init_rgb = init_rgb * scale
        else:
            init_rgb = None

        if "normal" in self.velocity_outputs:
            if self.noise_type == "gaussian":
                init_normal_w = torch.randn(*shape, 3, dtype=self.dtype, device=self.device)
            elif self.noise_type == "uniform":
                init_normal_w = torch.rand(*shape, 3, dtype=self.dtype, device=self.device) * 2 - 1
            else:
                raise NotImplementedError
            init_normal_w = init_normal_w * scale
        else:
            init_normal_w = None

        return dict(
            init_xyz_w=init_xyz_w,
            init_rgb=init_rgb,
            init_normal_w=init_normal_w,
        )

    def conditional_sampling(
        self,
        fpoint_latent: torch.Tensor,
        num_steps: int,
        # x0: torch.Tensor,
        init_xyz_w: torch.Tensor,
        init_rgb: T.Optional[torch.Tensor],
        init_normal_w: T.Optional[torch.Tensor],
        method: str = None,
        rtol: float = 1e-3,
        atol: float = 1e-4,
        max_point_chunk: int = -1,
        compute_log_likelihood: bool = False,
        compute_score_direction: bool = False,
        keep_freq: int = None,
        reverse_time: bool = False,
        printout: bool = False,
        latent_coord: T.Optional[PackedPoint] = None,
    ) -> T.Dict[str, torch.Tensor]:
        """
        Sample a point cloud using flow matching given the shape latent.

        Args:
            fpoint_latent:
                (b, num_latent, dim_latent) or (bl, dim_latent) packed
            num_steps:
                number of samples (suggested for adaptive methods)
            # x0:
            #     (b, num_points, d)  initial noise
            init_xyz_w:
                (b, num_points, 3)
            init_rgb:
                (b, num_points, 3) or None
            init_normal_w:
                (b, num_points, 3) or None
            method:
                see torchdiffeq, e.g, `dopri5`, `euler`.  None: use the default dopri
            compute_log_likelihood:
                whether to compute the log-likelihood of the sample using instantaneous change of variables formula
                from neural ODE / SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS (D.2)
            compute_score_direction:
                whether to compute the score direction at the sampled points
            keep_freq:
                if not None, we will keep the intermediate x every keep_freq iters
            reverse_time:
                if True, we will go from xyz (data, t=1) to uvw (noise, t=0)
            latent_coord:
                (bl, dn) needed if fpoint_latent is packed format

        Returns:
            sampled_x:
                (b, num_points, d)
            xyz_w:
                (b, num_points, 3)
            rgb:
                (b, num_points, 3) or None
            normal_w:
                (b, num_points, 3) or None

        """
        b, num_points, d = init_xyz_w.shape
        dtype = init_xyz_w.dtype
        device = init_xyz_w.device

        if not compute_log_likelihood:
            # construct the velocity function
            func = lambda t, x: self.estimate_velocity(
                fpoint_latent=fpoint_latent,
                t=t,
                x=x,
                latent_coord=latent_coord,
            )
            # x: (b, m, d)
        else:

            def func(t, y):
                # y: (b, m, d+1)
                x = y[..., :-1]  # (b, m, d)
                dx_dt = self.estimate_velocity(
                    fpoint_latent=fpoint_latent,
                    t=t,
                    x=x,
                    latent_coord=latent_coord,
                )  # (b, m, d)
                dlogp_dt = -1 * self.compute_velocity_divergence(fpoint_latent=fpoint_latent, t=t, x=x)  # (b, m)
                # print(f'dlogp_dt: min={dlogp_dt.min()}, mean={dlogp_dt.mean()}, max={dlogp_dt.max()}')
                dy_dt = torch.cat([dx_dt, dlogp_dt.unsqueeze(-1)], dim=-1)  # (b, m, d+1)
                return dy_dt

            raise NotImplementedError

        if method == "euler":
            # construct uniform ts
            ts = torch.linspace(min(1 / num_steps, self.t_eps), 1, num_steps, device=device)
        elif method == "heun":
            # construct uniform ts
            ts = torch.linspace(min(1 / num_steps, self.t_eps), 1, num_steps, device=device)
        elif method.startswith("heun_"):
            # heun_alpha
            a = float(method.split("heun_", 1)[1])

            # construct nonuniform ts (see https://arxiv.org/pdf/2206.00364 eq5)
            s_max, _ = self.path.compute_sigma_t(t=0)
            s_min, _ = self.path.compute_sigma_t(t=1)
            N = num_steps
            stds = [(s_max ** (1 / a) + i / (N - 1) * (s_min ** (1 / a) - s_max ** (1 / a))) ** a for i in range(N)]
            stds = torch.tensor(stds, dtype=dtype, device=device)
            ts = self.path.compute_t(sigma_t=stds)
        else:
            # construct uniform ts
            ts = torch.linspace(min(1 / num_steps, self.t_eps), 1, num_steps, device=device)

        if reverse_time:
            ts = 1 - ts
            assert not compute_score_direction
            assert not compute_log_likelihood

        # determine number of chunks
        if max_point_chunk < 0 or num_points <= max_point_chunk:
            chunk_size = num_points
            num_chunks = 1
        else:
            chunk_size = max_point_chunk
            num_chunks = (num_points + max_point_chunk - 1) // max_point_chunk

        out_dicts = []
        current_point_idx = 0

        for chunk_idx in range(num_chunks):
            if printout:
                print(f"chunk_idx: {chunk_idx} / {num_chunks}", flush=True)

            # compile input
            ddict = dict()
            current_idx = 0
            x0 = [init_xyz_w[:, current_point_idx : current_point_idx + chunk_size]]
            ddict["xyz"] = current_idx
            current_idx += 3
            if "rgb" in self.velocity_outputs:
                assert init_rgb is not None
                x0.append(init_rgb[:, current_point_idx : current_point_idx + chunk_size])
                ddict["rgb"] = current_idx
                current_idx += 3
            if "normal" in self.velocity_outputs:
                assert init_normal_w is not None
                x0.append(init_normal_w[:, current_point_idx : current_point_idx + chunk_size])
                ddict["normal"] = current_idx
                current_idx += 3

            if compute_log_likelihood:
                mvn = torch.distributions.MultivariateNormal(
                    torch.zeros(3, dtype=dtype, device=device),
                    torch.eye(3, dtype=dtype, device=device),
                )
                init_log_p = 0
                for ii in range(len(x0)):
                    log_p = mvn.log_prob(x0[ii])  # (b, m)
                    init_log_p += log_p  # (b, m)

                x0.append(init_log_p.unsqueeze(-1))
                ddict["logp"] = current_idx
                current_idx += 1

            current_point_idx += chunk_size
            x0 = torch.cat(x0, dim=-1)  # (b, m, d) or # (b, m, d+1)

            sampled_out = ode_solvers.odeint(
                func=func,
                x0=x0,
                ts=ts,
                method=method,
                rtol=rtol,
                atol=atol,
                printout=printout,
                keep_freq=keep_freq,
            )  # (b, num_points, d)
            if keep_freq is not None:
                sampled_x, xs_intermediate = sampled_out
            else:
                sampled_x = sampled_out
                xs_intermediate = None

            sampled_xyz_w = sampled_x[..., ddict["xyz"] : ddict["xyz"] + 3]
            if "rgb" in self.velocity_outputs:
                sampled_rgb = sampled_x[..., ddict["rgb"] : ddict["rgb"] + 3]
            else:
                sampled_rgb = None
            if "normal" in self.velocity_outputs:
                sampled_normal_w = sampled_x[..., ddict["normal"] : ddict["normal"] + 3]
            else:
                sampled_normal_w = None

            if compute_log_likelihood:
                assert "logp" in ddict
                logp = sampled_x[..., ddict["logp"]]  # (b, m)
            else:
                logp = None

            # compute score
            if compute_score_direction:
                _t = torch.ones(sampled_x.size(0), dtype=sampled_x.dtype, device=sampled_x.device)  # (b,)

                if compute_log_likelihood:
                    assert ddict["logp"] == (sampled_x.size(-1) - 1)
                    _sampled_x = sampled_x[..., :-1]
                else:
                    _sampled_x = sampled_x
                sdict = self.compute_score_numerator(
                    shape_latent=fpoint_latent,
                    t=_t,
                    x=_sampled_x,
                )
                score_numerator = sdict["score_numerator"]  # (b, n, d)
                score_numerator_xyz_w = sdict["score_numerator_xyz_w"]  # (b, n, 3) or None
                score_numerator_rgb = sdict["score_numerator_rgb"]  # (b, n, 3) or None
                score_numerator_normal_w = sdict["score_numerator_normal_w"]  # (b, n, 3) or None

            else:
                score_numerator = None
                score_numerator_xyz_w = None
                score_numerator_rgb = None
                score_numerator_normal_w = None

            out_dict = dict(
                x=sampled_x,  # may include xyz_w, rgb, normal, logp
                xyz_w=sampled_xyz_w,
                rgb=sampled_rgb,
                normal_w=sampled_normal_w,
                logp=logp,  # (b, m) or None
                score_direction_xyz_w=score_numerator_xyz_w,  # (b, m, 3) or None,  not normalized
                score_direction_rgb=score_numerator_rgb,  # (b, m, 3) or None
                score_direction_normal_w=score_numerator_normal_w,  # (b, m, 3) or None
                xs_intermediate=xs_intermediate,  # list of (b, m, d) or None
            )
            out_dicts.append(out_dict)

        # concat
        if num_chunks == 1:
            out_dict = out_dicts[0]
        else:
            out_dict = utils.cat_dict(out_dicts, dim_dict=1)

        out_dict["ts"] = ts  # (num_steps,)
        return out_dict

    def get_latent_shape(self) -> T.Dict[str, int]:
        """
        Returns number of latents and dimension of latents.
        """
        if isinstance(self.fpoint_encoder, SPointEncoder):
            return dict(
                # num_latent=self.fpoint_encoder.perceiver_num_latent,
                dim_latent=self.fpoint_encoder.dim_output,
            )
        else:
            raise NotImplementedError

    def get_latents(
        self,
        xyz_w: torch.Tensor,  # (b, m, 3)
        rgb: torch.Tensor,  # (b, m, 3)  [0, 1]
        ray_origin_direction_w: torch.Tensor,  # (b, m, 6)
        normal_w: torch.Tensor = None,  # (b, m, 3)
        alpha: torch.Tensor = None,  # (b, n, 1) [0, 1]
        num_latent: T.Optional[int] = None,
    ) -> T.Dict[str, T.Any]:
        """
        Encode and get latents.

        Args:
            xyz_w:
                (b, m, 3xyz_w) point positions [-1, 1]
            rgb:
                (b, m, 3rgb) or None, point rgb, [0, 1]
            normal_w:
                (b, m, 3xyz_w) or None, point normal,
            alpha:
                (b m, 1) [0, 1]

            num_latent:
                number of latents to use, only if using SPointEncoder. If None, uses the default number used in model training.

        Returns:
            latent_tokens:
                (b, num_shape_latents, dim_shape_latent) if format == 'batch' or
                (bn, dim_latent) if format == 'packed'
            latent_coord:
                (bn, dim_latent) if format == 'packed' or None if the encoder does not output coord
            format:
                'batch', 'packed'
        """

        assert isinstance(self.fpoint_encoder, SPointEncoder)
        out_dict = self.fpoint_encoder(
            xyz_w=xyz_w,
            rgb=rgb * 2 - 1 if "rgb" in self.center_inputs and rgb is not None else rgb,
            normal_w=normal_w,
            ray_origin_direction_w=ray_origin_direction_w,  # (b, n, 6_ro_rd)
            alpha=alpha * 2 - 1 if "alpha" in self.center_inputs and alpha is not None else alpha,  # (b, n, 1)
            tao=None,
            use_grad_checkpointing=self.optim_config.get("spoint_encoder_use_grad_checkpointing", False),
            debug=self.debug,
            num_latent=num_latent,
        )  # (b, num_latent, dim_latent) or (b, n, dim_out)

        if self.keep_latent_coord:
            latent_coord = out_dict["latent_coord"]  # (bl, dn) or (b, num_latent, dn) or None
        else:
            latent_coord = None

        fpoint_latents = out_dict["latent_tokens"]  # (bl, dim_latent) or (b, num_latent, dim_latent)
        format = out_dict["format"]

        return dict(
            latent_tokens=fpoint_latents,  # (b, num_shape_latents, dim_shape_latent) or (bn, dim_latent)
            latent_coord=latent_coord,  # (b, num_shape_latents, dn) or (bn, dn) or None
            format=format,  # 'batch' or 'packed'
        )

    def get_voxel(
        self,
        xyz_w: torch.Tensor,  # (b, n, 3xyz_w)
        cell_width: float,
        return_packed_coord: bool,
        min_xyz_w: T.Optional[float] = None,
        max_xyz_w: T.Optional[float] = None,
        grid_size: T.Optional[int] = None,
    ):
        """
        Convert xyz to voxel indices, find out occupied voxels and remove duplicates.

        Args:
            xyz_w:
                (b, n, 3xyz_w)
            cell_width:
                cell width of each voxel
            return_packed_coord:
                whether to return the xyz_w coordinate of the voxel centers
                in packed format
            min_xyz_w:
                float or (3,)
            max_xyz_w:
                float or (3,)
            grid_size:
                if given, it will clip the ijk to range from 0 to grid_size -1

        Returns:

        """
        b, n, _3xyz = xyz_w.shape

        # convert xyz to ijk
        if min_xyz_w is not None and max_xyz_w is not None:
            ijk = torch.floor((xyz_w - min_xyz_w) / cell_width).long()  # (b, n, 3ijk)
            if grid_size is not None:
                ijk = torch.clamp(ijk, min=0, max=grid_size - 1)  # (b, n, 3ijk)
        else:
            ijk = torch.floor(xyz_w / cell_width).long()  # (b, n, 3ijk)

        bijk = torch.cat(
            [
                torch.arange(b, device=xyz_w.device).reshape(b, 1, 1).expand(b, ijk.size(1), 1),  # (b, n, 1)
                ijk,  # (b, n, 3ijk)
            ],
            dim=-1,
        )  # (b, n, 4bijk)

        cell_bijk = torch.unique(
            bijk.reshape(-1, 4),  # (bn, 4) long
            sorted=True,
            return_inverse=False,
            return_counts=False,
            dim=0,
        )  # cell_bijk: (total_occupied_cells, 4bijk),  linear_idx: (bn,)  num_points_in_cells: (total_cells,)
        del bijk  # bijk has double meaning, so delete to prevent misuse

        if return_packed_coord:
            cell_xyz_w = (cell_bijk[..., 1:] + 0.5) * cell_width + min_xyz_w  # (total_occupied_cells, 3xyz_w)
            seq_lens = torch.bincount(input=cell_bijk[..., 0], minlength=b)  # (b,)

            # create packed point
            coord = ppoint.PackedPoint(
                coord=cell_xyz_w,  # (total_occupied_cells, 3xyz_w)
                seq_lens=seq_lens,  # (b,)
            )
        else:
            coord = None

        return dict(
            coord=coord,  # (total_occupied_cells, 3xyz_w)
            cell_bijk=cell_bijk,  # (total_occupied_cells, 4bijk)
        )

    def estimate_occ_grid(
        self,
        latent_token: torch.Tensor,
        return_occ_grid: bool = False,
        occ_grid_no_grad: bool = True,
    ):
        """
        Estimate the sparse structure latent and optionally the dense occupancy grid.

        Args:
            latent_token:
                (b, num_tokens, dim_tokens)
            return_occ_grid:
                whether to return occupancy grid
            occ_grid_no_grad:
                if True, we disable gradient on occupancy grid readout.

        Returns:
            est_ss_latent:
                (b, d, lowres_z, lowres_y, lowres_x)
            est_occ_grid:
                (b, 1, res_z, res_y, res_x) [0, 1]
        """
        assert self.voxel_decoder is not None

        odict = self.voxel_decoder(latent_token)
        est_ss_latent = odict["ss_latent"]  # (b, d, lowres_z, lowres_y, lowres_x)

        if return_occ_grid:
            assert self.voxel_ss_pipeline is not None
            if self.voxel_ss_pipeline.device != self.device:
                self.voxel_ss_pipeline.to(device=self.device)

            with torch.no_grad() if occ_grid_no_grad else contextlib.nullcontext():
                est_occ_grid_logit = self.voxel_ss_pipeline.decode_lowres_latent_to_logits(
                    est_ss_latent,
                )  # (b, 1, res_z, res_y, res_x)
                est_occ_grid = est_occ_grid_logit.sigmoid()  # (b, 1, res_z, res_y, res_x) [0, 1]
        else:
            est_occ_grid = None
            est_occ_grid_logit = None

        return dict(
            est_ss_latent=est_ss_latent,  # (b, d, lowres_z, lowres_y, lowres_x)
            est_occ_grid=est_occ_grid,  # (b, 1, res_z, res_y, res_x) [0, 1]
            est_occ_grid_logit=est_occ_grid_logit,  # (b, 1, res_z, res_y, res_x)
        )

    def estimate_mesh(
        self,
        latent_token: torch.Tensor,
        occ_bijk: T.Optional[torch.Tensor],
        grid_size: int = 64,
        input_occ_grid: T.Optional[torch.Tensor] = None,
        th_occ: float = 0.5,
        sdf_bias: float = None,
    ) -> T.Dict[str, T.Any]:
        """
        Estimate the sparse structure latent and optionally the dense occupancy grid.

        Args:
            latent_token:
                (b, num_tokens, dim_tokens)
            occ_bijk:
                (total_num_occupied_cells, 4bijk) int, packed format, the occupied cell indexes
            grid_size:
                number of cell of each dimension in the dense occ grid
            input_occ_grid:
                (b, 1, res_z, res_y, res_x) bool, where we will be computing
                the SDF and other flexicube parameters

        Returns:
            list of (b,) raw_meshes
                vertex_xyz_w:
                    (n, 3xyz_w)  [-1, 1], the vertex xyz coordinates
                triangles:
                    (num_triangles, 3idx)  long
                vertex_rgb:
                    (n, 3rgb)
                vertex_normal_w:
                    (n, 3xyz_w) real valued, not normalized
                grid_size:
                    int, number of cells per side
                success:
                    bool, whether the extraction is successful
        """
        assert self.mesh_decoder is not None

        if occ_bijk is None:
            assert input_occ_grid is not None
            assert input_occ_grid.size(2) == input_occ_grid.size(3) == input_occ_grid.size(4), (
                f"{input_occ_grid.shape=}"
            )
            # convert dense occ grid to sparse tensor
            occ_bijk = torch.argwhere(input_occ_grid > th_occ)[:, [0, 4, 3, 2]].int()  # (n, 4bijk)
            grid_size = input_occ_grid.size(2)

        assert grid_size == self.mesh_decoder.resolution, f"{grid_size=}, {self.mesh_decoder.resolution=}"

        if sdf_bias is not None:
            ori_sdf_bias = self.mesh_decoder.mesh_extractor.sdf_bias
            self.mesh_decoder.mesh_extractor.sdf_bias = sdf_bias
        mesh_dicts = self.mesh_decoder(
            latent_token=latent_token,
            occ_bijk=occ_bijk,  # (n, 4bijk)
            grid_min_xyz_w=self.mesh_optim_config.get("min_xyz_w", -1),
            grid_max_xyz_w=self.mesh_optim_config.get("max_xyz_w", 1),
        )

        if sdf_bias is not None:
            self.mesh_decoder.mesh_extractor.sdf_bias = ori_sdf_bias

        raw_meshes = []
        reg_losses = []
        reg_sdf_losses = []
        for ii in range(len(mesh_dicts)):
            if mesh_dicts[ii]["success"]:
                raw_mesh = structures.RawMesh(
                    vertex_xyz_w=mesh_dicts[ii]["vertex_xyz_w"].float(),  # (n, 3)
                    triangles=mesh_dicts[ii]["triangles"],  # (num_triangles, 3)
                    vertex_rgb=mesh_dicts[ii]["vertex_rgb"].float(),  # (n, 3)
                    vertex_normal_w=mesh_dicts[ii]["vertex_normal_w"].float(),  # (n, 3) not normalized
                )
            else:
                raw_mesh = None

            reg_losses.append(mesh_dicts[ii]["reg_loss"])
            reg_sdf_losses.append(mesh_dicts[ii]["reg_sdf_loss"])

            if self.debug:
                for key in mesh_dicts[ii]:
                    if isinstance(mesh_dicts[ii][key], torch.Tensor):
                        assert mesh_dicts[ii][key].isfinite().all(), (
                            f"{ii}, {key}: "
                            f"nan: {mesh_dicts[ii][key].isnan().any()}, "
                            f"inf: {mesh_dicts[ii][key].isinf().any()}"
                        )

            raw_meshes.append(raw_mesh)

        return dict(
            raw_meshes=raw_meshes,
            reg_losses=reg_losses,  # None if is_training = False
            reg_sdf_losses=reg_sdf_losses,  # None if is_training = False
        )

