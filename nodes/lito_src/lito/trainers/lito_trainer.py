#
# Copyright (C) 2026 Apple Inc. All rights reserved.
#
# The file implements trainer for Light Tokenization.

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

from lightning.pytorch.loggers import TensorBoardLogger
import lpips
import numpy as np
import open3d as o3d
from packaging import version

import pytorch3d.io
import pytorch3d.loss
import pytorch3d.ops
import pytorch3d.renderer
import pytorch3d.structures
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter

from lito.datasets import obj_wdset
from lito.eval_scripts import eval_utils_metrics
from lito.flow import path
from lito.integrations.trellis.trellis_sparse_structure import (
    TrellisSparseStructurePipeline,
    get_trellis_sparse_structure_pipeline,
)
from lito.models.point_decoder import GaussianDecoderXv
from lito.models.spoint_encoder import SPointEncoder
from lito.odelibs import ode_solvers
from lito.script_utils import config_utils
from lito.trainers.base import BaseTrainer
from plibs import gs_utils, lightning_utils, linalg_utils, ppoint, sh_utils, structures, utils
from plibs.ppoint import PackedPoint

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

    def configure_optimizers(self):
        """construct optimizer"""

        lr = self.optim_config["lr"]

        # optimizer
        if self.optim_config.get("optim_alg", "adamw") == "adamw":
            print("using adamw optimizer", flush=True)
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=self.optim_config["weight_decay"],
                betas=(self.optim_config["beta1"], self.optim_config["beta2"]),
            )
        else:
            raise NotImplementedError

        # scheduler
        if self.optim_config["lr_scheduler_config"] is None:
            return optimizer

        scheduler = config_utils.instantiate_from_config(
            config=self.optim_config["lr_scheduler_config"],
            optimizer=optimizer,
            last_epoch=self.trainer.global_step - 1,  # total number of batches
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # def log(self, *args, **kwargs):
    #     if kwargs.get("sync_dist", False):
    #         name = args[0] if args else kwargs.get("name", "?")
    #         step = self.trainer.global_step if self.trainer else "?"
    #         print(f"[rank {self.global_rank}] sync_dist log: {name} (step={step})", flush=True)
    #     super().log(*args, **kwargs)

    def on_train_batch_start(
        self,
        batch: T.Any,
        batch_idx: int,
    ):
        if isinstance(batch, (tuple, list)):
            b = len(batch)
            data_decoding_times = [batch[ib].get("total_decoding_time", [0.0]) for ib in range(b)]  # list of float
            data_decoding_times = torch.tensor(data_decoding_times, dtype=torch.float, device=self.device)
        else:
            raise NotImplementedError

        self.log(
            "data_decoding_time",
            data_decoding_times.mean(),
            prog_bar=True,
            logger=True,
            rank_zero_only=False,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            reduce_fx="mean",
            batch_size=b,
        )
        self.log(
            "data_decoding_time_min",
            data_decoding_times.min(),
            prog_bar=True,
            logger=True,
            rank_zero_only=False,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            reduce_fx="min",
            batch_size=b,
        )
        self.log(
            "data_decoding_time_max",
            data_decoding_times.max(),
            prog_bar=True,
            logger=True,
            rank_zero_only=False,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            reduce_fx="max",
            batch_size=b,
        )

        data_loading_time = timer() - self.data_loading_stime
        self.log(
            "data_loading_time",
            data_loading_time,
            prog_bar=True,
            logger=True,
            rank_zero_only=False,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            reduce_fx="mean",
            batch_size=b,
        )
        self.log(
            "data_loading_time_max",
            data_loading_time,
            prog_bar=True,
            logger=True,
            rank_zero_only=False,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            reduce_fx="max",
            batch_size=b,
        )
        self.log(
            "data_loading_time_min",
            data_loading_time,
            prog_bar=True,
            logger=True,
            rank_zero_only=False,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            reduce_fx="min",
            batch_size=b,
        )
        # print(f'data_loading_time: {data_loading_time} sec')
        self.compute_stime = timer()

    def on_train_batch_end(
        self,
        outputs: T.Union[torch.Tensor, T.Mapping[str, T.Any], None],
        batch: T.Any,
        batch_idx: int,
    ):
        if isinstance(batch, (tuple, list)):
            batch_size = len(batch)
        else:
            raise NotImplementedError

        # scheduler = self.lr_schedulers()
        optimizer = self.optimizers()
        self.log(
            "hparams/lr",
            optimizer.param_groups[0]["lr"],
            on_epoch=False,
            on_step=True,
            logger=True,
            prog_bar=True,
            rank_zero_only=True,
            batch_size=batch_size,
        )

        compute_time = timer() - self.compute_stime
        self.log(
            "compute_time",
            compute_time,
            prog_bar=True,
            logger=True,
            rank_zero_only=False,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            reduce_fx="mean",
            batch_size=batch_size,
        )
        self.log(
            "compute_time_max",
            compute_time,
            prog_bar=True,
            logger=True,
            rank_zero_only=False,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            reduce_fx="max",
            batch_size=batch_size,
        )
        self.log(
            "compute_time_min",
            compute_time,
            prog_bar=True,
            logger=True,
            rank_zero_only=False,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            reduce_fx="min",
            batch_size=batch_size,
        )
        # print(f'compute_time: {compute_time} sec')
        self.data_loading_stime = timer()

    def on_fit_start(self):
        # check if any module are in eval mode
        print(
            f"rank {self.global_rank}: "
            f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'not set')}, "
            f"PYTORCH_ALLOC_CONF: {os.environ.get('PYTORCH_ALLOC_CONF', 'not set')}, "
        )

        eval_mods = [(n, m) for n, m in self.named_modules() if not m.training]
        if eval_mods:
            names = [n for n, _ in eval_mods]
            if self.local_rank == 0:
                print(f"[on_fit_start] {len(names)} modules in eval mode:\n  " + "\n  ".join(names))

    def on_after_backward(self):
        if self.debug:
            current_min = math.inf
            current_max = -math.inf
            for name, param in self.named_parameters():
                if param.grad is None:
                    print(f"Parameter {name} has no gradient")
                else:
                    pmin = param.grad.min().item()
                    pmax = param.grad.max().item()
                    # print(f"Parameter {name} gradient: min: {pmin}, max: {pmax}")
                    current_min = min(current_min, pmin)
                    current_max = max(current_max, pmax)
            print(f"All gradient: min: {current_min}, max: {current_max}")

        # with lightning_utils.local_rank_first(self):
        #     print(f'finding unused parameters..', flush=True)
        #     unused_params = []
        #     for name, param in self.named_parameters():
        #         if param.grad is None and param.requires_grad:
        #             unused_params.append(name)
        #     if unused_params:
        #         print(f"Unused parameters after backward pass: {unused_params}")

    def training_step(
        self,
        batch: T.Dict[str, T.Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        r"""
        Training_step defines the train loop.

        Args:
            batch:
                latent_token:
                    (b, num_latent, dim_latent)
            batch_idx:

        """

        loss = self._step(
            batch=batch,
            batch_idx=batch_idx,
            split="train",
            run_velocity=self.use_velocity,
            run_3dgs=self.use_3dgs,
            mode=self.mode,
        )
        return loss

    @torch.no_grad()
    def validation_step(
        self,
        batch: T.Dict[str, T.Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        loss = self._step(
            batch=batch,
            batch_idx=batch_idx,
            split="valid",
            run_velocity=self.use_velocity,
            run_3dgs=self.use_3dgs,
            sample_flow=True,
            mode=self.mode,
            dataloader_idx=dataloader_idx,
        )
        return loss

    @torch.no_grad()
    def predict_step(
        self,
        batch: T.Dict[str, T.Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """a dummy predict step"""
        pass

    def _step(
        self,
        batch: T.Union[T.Dict[str, T.Any], T.List[T.Dict[str, T.Any]]],
        batch_idx: int,
        split: str,
        run_velocity: bool = False,
        run_3dgs: bool = False,
        sample_flow: bool = False,
        print_timing: bool = False,
        mode: str = "tokenizer",
        dataloader_idx: int = 0,
    ):
        r"""
        Training_step defines the train loop.

        Args:
            batch:
                uid:
                    uid of the mesh
                point_xyz_w:
                    (n, 3) the point xyz in the n-coordinate
                point_rgb:
                    (n, 3) or -1, the point rgb [0, 1]
                point_normal_w:
                    (n, 3) or -1, the point normal in the n-coordinate

            batch_idx:

            split:
                'train', 'valid', 'test'

        Returns:
            loss: (,)

        """
        stime = timer()
        assert isinstance(batch, list)

        if batch[0].get("dset_type", None) in ["tokenizer", "lito_tokenizer"]:
            batch = default_collate(batch)

        else:
            raise NotImplementedError(batch[0]["dset_type"])

        if self.debug:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    if not torch.isfinite(batch[key]).all():
                        print(f"uid: {batch.get('uid')}")
                        print(f"shard_url: {batch.get('shard_url')}")
                        print(f"{key}: nan: {torch.isnan(batch[key]).any()}, inf: {torch.isinf(batch[key]).any()}")

        # # debug
        # if batch_idx < 10:
        #     batch_filename = f"/mnt/batch_{batch_idx}.pth"
        #     torch.save(batch, batch_filename)
        # # end debug

        if mode == "tokenizer":
            loss = self._step_tokenizer(
                batch=batch,
                batch_idx=batch_idx,
                split=split,
                run_velocity=run_velocity,
                run_3dgs=run_3dgs,
                sample_flow=sample_flow,
                print_timing=print_timing,
                dataloader_idx=dataloader_idx,
            )
        else:
            raise NotImplementedError

        # to releive the stress of cpu (dataloader)
        if self.optim_config.get("min_iter_ms", 0) > 0:
            wait_sec = self.optim_config["min_iter_ms"] / 1000
            ttime = timer() - stime  # secs
            if wait_sec > (ttime - 1e-6):
                time.sleep(max(0, wait_sec - ttime))

        if self.debug:
            assert loss.isfinite().all(), f"{loss.isnan()}  {loss.isinf()}"

        return loss

    def compute_velocity_loss(
        self,
        fpoint_latent: torch.Tensor,
        xyz_w: torch.Tensor,
        rgb: T.Optional[torch.Tensor] = None,
        normal_w: T.Optional[torch.Tensor] = None,
        latent_coord: ppoint.PackedPoint = None,
    ) -> T.Dict[str, T.Any]:
        """
        Randomly sample t, compute xt and ut_gt, estimate ut, then compute loss

        Args:
            fpoint_latent:
                (b, num_latent, dim_latent) or (bl, dim_latent)
            xyz_w:
                (b, m, 3)
            rgb:
                (b, m, 3) or None [0, 1]
            normal_w:
                (b, m, 3) or None
            latent_coord:
                (bl, dn) needed if fpoint_latent is packed

        Returns:
            loss:
                (,) mse loss on the velocity
            est_ut:
                (b, m, d) estimated velocity
            t:
                (b,) sampled t [0, 1]
        """

        b, m, d = xyz_w.shape
        device = xyz_w.device

        # construct decoder input and gt decoder output
        # sample t
        sampled_t = torch.rand(b, device=device)
        t_flow = sampled_t * (1 - 2 * self.t_eps) + self.t_eps  # (b,)

        # compile input as a whole
        ddict = dict()
        current_idx = 0
        x = [xyz_w]
        ddict["xyz"] = current_idx
        current_idx += 3
        if "rgb" in self.velocity_outputs:
            assert rgb is not None
            x.append(rgb)
            ddict["rgb"] = current_idx
            current_idx += 3
        if "normal" in self.velocity_outputs:
            assert normal_w is not None
            x.append(normal_w)
            ddict["normal"] = current_idx
            current_idx += 3
        x = torch.cat(x, dim=-1)  # (b, m, d)

        # sample noise (we choose to use standard gaussian)
        if self.noise_type == "gaussian":
            x0 = torch.randn_like(x)  # (b, num_flow_points, d)
        elif self.noise_type == "uniform":
            x0 = torch.rand_like(x) * 2 - 1  # (b, num_flow_points, d)  [-1, 1]
        else:
            raise NotImplementedError
        xt, ut_gt = self.path.compute_xt_ut(t=t_flow, x0=x0, x1=x)  # (b, num_flow_points, d)

        if self.debug:
            assert t_flow.isfinite().all(), f"nan: {t_flow.isnan().any()}, inf: {t_flow.isinf().any()}"
            assert xt.isfinite().all(), f"nan: {xt.isnan().any()}, inf: {xt.isinf().any()}"
            assert ut_gt.isfinite().all(), f"nan: {ut_gt.isnan().any()}, inf: {ut_gt.isinf().any()}"

        # estimate velocity
        est_ut = self.estimate_velocity(
            fpoint_latent=fpoint_latent,
            t=t_flow,  # (b,)
            x=xt,  # (b, num_flow_points, d)
            latent_coord=latent_coord,
        )  # (b, m, d)
        if self.debug:
            assert est_ut.isfinite().all(), f"nan: {est_ut.isnan().any()}, inf: {est_ut.isinf().any()}"

        # compute velocity loss for xyz_w
        if "xyz" in self.velocity_outputs:
            didx = ddict["xyz"]
            loss_xyz = F.mse_loss(
                input=est_ut[..., didx : didx + 3], target=ut_gt[..., didx : didx + 3], reduction="mean"
            )  # (,)
            loss = loss_xyz.clone()
        else:
            raise RuntimeError

        if "rgb" in self.velocity_outputs:
            didx = ddict["rgb"]
            loss_rgb = F.mse_loss(
                input=est_ut[..., didx : didx + 3], target=ut_gt[..., didx : didx + 3], reduction="mean"
            )  # (,)
            loss += loss_rgb
        else:
            loss_rgb = None

        if "normal" in self.velocity_outputs:
            didx = ddict["normal"]
            est_normal = est_ut[..., didx : didx + 3]  # (b, m, 3)
            gt_normal = ut_gt[..., didx : didx + 3]  # (b, m, 3)
            if self.optim_config.get("align_normal_first", False):
                with torch.no_grad():
                    mask = (est_normal * gt_normal).sum(dim=-1) < 0  # (b, m)
                    # directly overwrite the memory (does not matter for normal)
                    gt_normal[mask] *= -1

            loss_normal = F.mse_loss(input=est_normal, target=gt_normal, reduction="mean")  # (,)
            loss += loss_normal
        else:
            loss_normal = None

        loss = loss / len(self.velocity_outputs)

        return dict(
            loss=loss,  # (,)
            loss_xyz=loss_xyz,  # (,)
            loss_rgb=loss_rgb,  # (,)
            loss_normal=loss_normal,  # (,)
            est_ut=est_ut,  # (b, m, d)
            t=t_flow,  # (b,)
        )

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
        with torch.autocast(device_type=fpoint_latent.device.type, enabled=False):
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

    def compute_loss_recon(
        self,
        est: torch.Tensor,
        tgt: torch.Tensor,
        loss_weight_l1: float = 1.0,
        loss_weight_ssim: float = 0,
        loss_weight_lpips: float = 0.2,
        valid_mask: torch.Tensor = None,
        compute_psnr: bool = False,
    ) -> T.Dict[str, torch.Tensor]:
        """
        Args:
            est:
                (*, h, w, d)   [0, 1]
            tgt:
                (*, h, w, d)   [0, 1]
            valid_mask:
                (*,)
        Returns:
            (,)
        """
        *b_shape, h, w, d = est.shape
        bq = math.prod(b_shape)

        loss = 0.0

        if valid_mask is not None:
            l1_loss = torch.nn.functional.l1_loss(
                input=est,
                target=tgt,
                reduction="none",
            )  # (*, h, w, d)
            l1_loss = l1_loss.masked_fill(~valid_mask.reshape(*b_shape, 1, 1, 1), 0)
            l1_loss = l1_loss.mean()  # (,)
        else:
            l1_loss = torch.nn.functional.l1_loss(
                input=est,
                target=tgt,
                reduction="mean",
            )
        if loss_weight_l1 > 1e-7:
            loss = loss + loss_weight_l1 * l1_loss

        est = est.reshape(bq, h, w, d).permute(0, 3, 1, 2)  # (bq, d, h, w)  [0, 1]
        tgt = tgt.reshape(bq, h, w, d).permute(0, 3, 1, 2)  # (bq, d, h, w)  [0, 1]

        # ssim expect [0, 1]
        # ssim is slow -- don't compute if not needed
        # if loss_weight_ssim > 1e-6:
        if valid_mask is not None:
            l_ssim = 1 - eval_utils_metrics.fast_ssim(
                img1=est,  # (bq, d, h, w)  [0, 1]
                img2=tgt,  # (bq, d, h, w)  [0, 1]
                reduction=None,
            )  # (bq, d, h, w)
            l_ssim = l_ssim.masked_fill(~valid_mask.reshape(bq, 1, 1, 1), 0)
            l_ssim = l_ssim.mean()  # (,)
        else:
            l_ssim = 1 - eval_utils_metrics.fast_ssim(
                img1=est,  # (bq, d, h, w)  [0, 1]
                img2=tgt,  # (bq, d, h, w)  [0, 1]
                reduction="mean",
            )  # (,)
        if loss_weight_ssim > 1e-7:
            loss = loss + loss_weight_ssim * l_ssim
        # else:
        #     l_ssim = None

        # lpips wants input to be [-1, 1]
        if loss_weight_lpips > 1e-6:
            l_lpips = self.lpips_loss_fn(
                est * 2 - 1,  # (bq, d, h, w) [-1, 1]
                tgt * 2 - 1,  # (bq, d, h, w) [-1, 1]
            )  # (bq, 1, 1, 1)
            if valid_mask is not None:
                l_lpips = l_lpips.masked_fill(~valid_mask.reshape(bq, 1, 1, 1), 0)
            l_lpips = l_lpips.mean()
            if loss_weight_lpips > 1e-7:
                # assert self.lpips_loss_fn is not None
                loss = loss + loss_weight_lpips * l_lpips
        else:
            l_lpips = None

        if compute_psnr:
            mse = ((est - tgt) ** 2).reshape(bq, -1).mean(dim=-1)  # (bq,)
            psnr_val = -20 * torch.log10(torch.sqrt(torch.clamp(mse, min=1e-8)))  # (bq,)
            psnr_val = psnr_val.mean()  # (,)
        else:
            psnr_val = None

        return dict(
            loss=loss,
            l1_loss=l1_loss,  # (,) or None
            l_ssim=l_ssim,  # (,) or None
            l_lpips=l_lpips,  # (,) or None
            psnr=psnr_val,  # (,) or None
        )

    def compute_3dgs_loss_given_gt_and_est(
        self,
        # gt
        H_c2w: torch.Tensor,  # (b, q, 4, 4)
        intrinsic: torch.Tensor,  # (b, q, 3, 3)
        rgb_gt: torch.Tensor,  # (b, q, h, w, 3) [0, 1]
        rgb_mask_gt: torch.Tensor,  # (b, q, h, w, 1) [0, 1]
        hit_mask_gt: torch.Tensor,  # (b, q, h, w) [0, 1]
        normal_w_gt: torch.Tensor,  # (b, q, h, w, 3)
        # est
        est_rgb: torch.Tensor,  # (b, q, h, w, 3) [0, 1], premultiplied with est_alpha
        est_alpha: torch.Tensor,  # (b, q, h, w, 1) [0, 1]
        est_normal_w: torch.Tensor,  # (b, q, h, w, 3), straight (renormalized)
        est_normal_w_raw: torch.Tensor,  # (b, q, h, w, 3), unnormalized, premultiplied
        # weight rgb
        loss_weight_rgb: float = 1.0,
        loss_weight_rgb_l1: float = 1.0,
        loss_weight_rgb_ssim: float = 0,
        loss_weight_rgb_lpips: float = 0.2,
        # weight normal
        loss_weight_normal: float = 1.0,
        loss_weight_normal_l1: float = 1.0,
        loss_weight_normal_ssim: float = 0,
        loss_weight_normal_lpips: float = 0.2,
        loss_weight_normal_sin: float = 1.0,
        use_random_bg: bool = True,
        compute_psnr: bool = False,
    ):
        loss = 0
        loss_3dgs_lpips = 0
        num_loss_3dgs_lpips = 0
        loss_dict = dict()

        # rgb
        if est_rgb is not None:
            if use_random_bg:
                random_bg = torch.rand_like(rgb_gt)  # (b, q, h, w, 3)  [0, 1]
            else:
                random_bg = torch.zeros_like(rgb_gt)
            ldict = self.compute_loss_recon(
                # est=est_rgb * est_alpha + (1 - est_alpha) * random_bg,  # (b, q, h, w, 3)  [0, 1]
                est=est_rgb + (1 - est_alpha) * random_bg,  # (b, q, h, w, 3)  [0, 1]  # rgb already premultiplied
                tgt=rgb_gt * rgb_mask_gt + (1 - rgb_mask_gt) * random_bg,  # (b, q, h, w, 3) [0, 1]
                loss_weight_l1=loss_weight_rgb_l1,
                loss_weight_ssim=loss_weight_rgb_ssim,
                loss_weight_lpips=loss_weight_rgb_lpips,
                compute_psnr=compute_psnr,
            )
            loss = loss + loss_weight_rgb * ldict["loss"]  # (,)
            loss_dict["loss_rgb_l1"] = ldict["l1_loss"]  # (,)
            loss_dict["loss_rgb_ssim"] = ldict["l_ssim"]  # (,)
            loss_dict["loss_rgb_lpips"] = ldict["l_lpips"]  # (,)
            loss_dict["rgb_psnr"] = ldict["psnr"]  # (,)
            loss_dict["rgb_ssim"] = 1 - ldict["l_ssim"] if ldict["l_ssim"] is not None else None  # (,)
            if ldict["l_lpips"] is not None:
                loss_3dgs_lpips += ldict["l_lpips"].detach()
                num_loss_3dgs_lpips += 1

        # compute normal_w loss
        if est_normal_w is not None and est_normal_w_raw is not None:
            assert normal_w_gt is not None
            assert hit_mask_gt is not None
            assert H_c2w is not None

            normal_w_gt = normal_w_gt * hit_mask_gt.float().unsqueeze(-1) + (1 - hit_mask_gt.float().unsqueeze(-1)) * (
                1 / math.sqrt(3)
            )  # (b, q, h, w, 3xyz_w) premultiplied
            est_normal_w_raw = est_normal_w_raw + (1 - est_alpha) * (1 / math.sqrt(3))

            normal_w_gt = torch.nn.functional.normalize(normal_w_gt, dim=-1)  # (b, q, h, w, 3xyz_w)
            est_normal_w = torch.nn.functional.normalize(est_normal_w_raw, dim=-1)  # (b, q, h, w, 3xyz_w)

            # make sure gt points toward -z in camera coordinate
            with torch.no_grad(), torch.autocast(device_type=normal_w_gt.device.type, enabled=False):
                b, q, h, w, _ = normal_w_gt.shape
                bq = b * q

                same_dir = (est_normal_w * normal_w_gt).sum(dim=-1) <= 0  # (b, q, h, w) {0, 1}
                same_dir = same_dir * 2 - 1  # {-1, 1}

            est_normal_w = est_normal_w * same_dir.unsqueeze(-1)  # (b, q, h, w, 3)
            del same_dir

            # compute loss: sin(theta)^2
            loss_normal_w_sin = 1.0 - ((normal_w_gt * est_normal_w).sum(dim=-1) ** 2)  # (b, q, h, w)
            loss_normal_w_sin = loss_normal_w_sin.mean()
            loss = loss + (loss_weight_normal * loss_weight_normal_sin) * loss_normal_w_sin
            loss_dict["loss_normal_w_sin"] = loss_normal_w_sin

            # recon loss
            ldict = self.compute_loss_recon(
                est=(est_normal_w + 1) * 0.5,  # (b, q, h, w, 3)  [0, 1]
                tgt=(normal_w_gt + 1) * 0.5,  # (b, q, h, w, 3) [0, 1]
                loss_weight_l1=loss_weight_normal_l1,
                loss_weight_ssim=loss_weight_normal_ssim,
                loss_weight_lpips=loss_weight_normal_lpips,
                compute_psnr=compute_psnr,
            )
            loss = loss + loss_weight_normal * ldict["loss"]  # (,)
            loss_dict["loss_normal_l1"] = ldict["l1_loss"]  # (,)
            loss_dict["loss_normal_ssim"] = ldict["l_ssim"]  # (,)
            loss_dict["loss_normal_lpips"] = ldict["l_lpips"]  # (,)
            loss_dict["normal_psnr"] = ldict["psnr"]  # (,)
            loss_dict["normal_ssim"] = 1 - ldict["l_ssim"] if ldict["l_ssim"] is not None else None  # (,)
            if ldict["l_lpips"] is not None:
                loss_3dgs_lpips += ldict["l_lpips"].detach()
                num_loss_3dgs_lpips += 1

        gt_maps = dict(
            rgb=rgb_gt * rgb_mask_gt,  # (b, q, h, w, 3) [0, 1] premultiplied
            normal_w=normal_w_gt * rgb_mask_gt if normal_w_gt is not None else None,  # (b, q, h, w, 3) toward pinhole
            alpha=rgb_mask_gt,  # (b, q, h, w, 1) [0, 1]
            hit_map=hit_mask_gt,  # (b, q, h, w) [0, 1]
        )
        est_maps = dict(
            rgb=est_rgb,  # (b, q, h, w, 3) [0, 1] premultiplied with est_alpha
            normal_w=est_normal_w * est_alpha
            if est_normal_w is not None
            else None,  # (b, q, h, w, 3) toward pinhole, premultiplied
            alpha=est_alpha,  # (b, q, h, w, 1) [0, 1]
            hit_map=est_alpha[..., 0] > 0.5,  # (b, q, h, w) [0, 1]
        )

        loss_dict["loss_3dgs_lpips"] = loss_3dgs_lpips / max(1, num_loss_3dgs_lpips)  # (,)

        return dict(
            loss=loss,
            loss_dict=loss_dict,
            gt_maps=gt_maps,
            est_maps=est_maps,
        )

    def compute_3dgs_loss(
        self,
        fpoint_latent: torch.Tensor,
        init_coord: ppoint.PackedPoint,
        # gt
        H_c2w: torch.Tensor,
        intrinsic: torch.Tensor,
        rgb_gt: torch.Tensor,  # (b, q, h, w, 3) [0, 1]
        rgb_mask_gt: torch.Tensor,  # (b, q, h, w, 1) [0, 1]
        hit_mask_gt: torch.Tensor,  # (b, q, h, w, 1) [0, 1]
        normal_w_gt: torch.Tensor,  # (b, q, h, w, 3)
        # rgb
        loss_weight_rgb: float = 1.0,
        loss_weight_rgb_l1: float = 1.0,
        loss_weight_rgb_ssim: float = 0,
        loss_weight_rgb_lpips: float = 0.2,
        # normal
        loss_weight_normal: float = 1.0,
        loss_weight_normal_l1: float = 1.0,
        loss_weight_normal_ssim: float = 0,
        loss_weight_normal_lpips: float = 0.2,
        loss_weight_normal_sin: float = 1.0,
        latent_coord: ppoint.PackedPoint = None,
    ):
        """
        Compute pointersect loss given the ground truth

        Args:
            fpoint_latent:
                (b, num_latent, dim_latent) or (bl, dim_latent) packed format
            H_c2w:
                (b, q, 4, 4) the camera pose of the corresponding gt_rgb images
            intrinsic:
                (b, q, 3, 3) the camera intrinsic of the corresponding gt_rgb images
            rgb_mask_gt:
                (b, q, 1, h, w) the RGB mask / alpha of the rgb
            hit_mask_gt:
                (b, q, 1, h, w) the hit map of the rgb

            normal_w_gt:
                (b, q, 3, h, w)
            rgb_gt:
                (b, q, 3, h, w) [0, 1]
            random_bg:
                whether to randomly use black or white background
            latent_coord:
                (bl, dn) or None
            normal_bg_type:
                'full': use the full image
                'valid_erode-5': valid mask + erode kernel size = 5

        Returns:
            loss:
                (,) total loss

            est_ray_t:
                (b, q, h, w) or None
            est_normal_w:
                (b, q, h, w, 3xyz) or None
            est_hit:
                (b, q, h, w) [0, 1]

            gt_ray_t:
                (b, q, h, w) or None
            gt_normal_w:
                (b, q, h, w, 3xyz) or None
            gt_hit:
                (b, q, h, w)  [0, 1]

        """

        assert self.gs_decoder is not None
        if rgb_gt is not None:
            b, q, h, w, _3rgb = rgb_gt.shape
        elif normal_w_gt is not None:
            b, q, h, w, _3xyz = normal_w_gt.shape
        else:
            raise NotImplementedError

        bq = b * q

        # estimate gaussians
        gs_dicts = self.estimate_gaussians(
            fpoint_latent=fpoint_latent,  # (b, #token, dim)
            init_coord=init_coord,  # (total_occ_cells, 3xyz_w) or None
            latent_coord=latent_coord,  # (b, #token, dn), (bl, dn) or None
        )  # list of (b,), each is a gs_dict

        # render gaussians
        out_dict = self.render_gaussians(
            gs_dicts=gs_dicts,  # list of (b,)
            H_c2w=H_c2w,  # (b, q, 4, 4)
            intrinsic=intrinsic,  # (b, q, 3, 3)
            width_px=w,
            height_px=h,
            given_rgb_sh_degree=None,
        )
        # rgb: (b, q, h, w, 3rgb) [0, 1]
        # alpha: (b, q, h, w, 1) [0, 1]
        # normal_w: (b, q, h, w, 3xyz_w) normalized, pointing toward camera pinhole

        # compute loss
        loss_dict = self.compute_3dgs_loss_given_gt_and_est(
            H_c2w=H_c2w,
            intrinsic=intrinsic,
            rgb_gt=rgb_gt,  # (b, q, h, w, 3) [0, 1]
            rgb_mask_gt=rgb_mask_gt,  # (b, q, h, w, 1) [0, 1]
            hit_mask_gt=hit_mask_gt,  # (b, q, h, w) [0, 1]
            normal_w_gt=normal_w_gt,  # (b, q, h, w, 3)
            # est
            est_rgb=out_dict.get("premultiplied_rgb", None),  # (b, q, h, w, 3) [0, 1]
            est_alpha=out_dict["alpha"],  # (b, q, h, w, 1) [0, 1]
            est_normal_w=out_dict.get("normal_w", None),  # (b, q, h, w, 3)  normalized
            est_normal_w_raw=out_dict.get(
                "premultiplied_normal_w_raw", None
            ),  # (b, q, h, w, 3)  unnormalized, premultiplied
            # weight rgb
            loss_weight_rgb=loss_weight_rgb,
            loss_weight_rgb_l1=loss_weight_rgb_l1,
            loss_weight_rgb_ssim=loss_weight_rgb_ssim,
            loss_weight_rgb_lpips=loss_weight_rgb_lpips,
            # weight normal
            loss_weight_normal=loss_weight_normal,
            loss_weight_normal_l1=loss_weight_normal_l1,
            loss_weight_normal_ssim=loss_weight_normal_ssim,
            loss_weight_normal_lpips=loss_weight_normal_lpips,
            loss_weight_normal_sin=loss_weight_normal_sin,
        )

        return dict(
            gs_dicts=gs_dicts,
            loss=loss_dict["loss"],
            other_loss_dict=loss_dict["loss_dict"],
            est_maps=loss_dict["est_maps"],
            # rgb: (b, q, h, w, 3rgb) [0, 1]
            # alpha: (b, q, h, w, 1) [0, 1]
            # normal_w: (b, q, h, w, 3xyz_w) normalized, pointing toward camera pinhole
            gt_maps=loss_dict["gt_maps"],
        )

    def _step_3dgs(
        self,
        batch: T.Dict[str, T.Any],
        batch_idx: int,
        split: str,
        fpoint_latent: torch.Tensor = None,  # (b, num_latent, dim_latent)
        dataloader_idx: int = 0,
        latent_coord: T.Optional[ppoint.PackedPoint] = None,
        save_debug_gaussians: bool = False,
    ):
        r"""
        Only train the 3dgs decoder.

        Args:
            batch:
                uid:
                    uid of the mesh
                point_xyz_w:
                    (n, 3) the point xyz in the n-coordinate
                point_rgb:
                    (n, 3) or -1
                point_normal_w:
                    (n, 3) or -1

                text:
                    str or -1, the text caption of mesh
                rgb_ori:
                    (b, q, 3rgb, h, w) or -1, [0, 1] the loaded views
                normal_w:
                    (b, q, 3xyz, h, w)
                z_c:
                    (b, q, 1, h, w)
                rgb_mask:
                    (b, q, 1, h, w)  bool
                H_c2w:
                    (b, q, 4, 4) or -1, camera pose of the loaded views
                intrinsic:
                    (b, q, 3, 3) or -1, camera intrinsics of the loaded views
                extra_rgbd_dict:
                    rgb_map:
                        (b, q', 3, h', w')  [0, 1]
                    normal_w_map:
                        (b, q', 3xyz, h', w')
                    z_c_map:
                        (b, q', 1, h', w')
                    rgb_mask:
                        (b, q', 1, h', w')  bool
                    H_c2w:
                        (b, q', 4, 4) or -1, camera pose of the loaded views
                    intrinsic:
                        (b, q', 3, 3) or -1, camera intrinsics of the loaded views
                patch_feature:
                    (b, q, ph, pw, d) patch feature map computed from rgb_map

            batch_idx:

            split:
                'train', 'valid', 'test'

            fpoint_latent:
                (b, num_latent, dim_latent)

        Returns:
            loss: (,)

        """
        batch_size = batch["point_xyz_w"].shape[0]

        # get init_coord (voxel center of occupied voxels, 64^3 from [-1, 1])
        grid_size = 64
        min_xyz_w = -1
        max_xyz_w = 1
        cell_width = (max_xyz_w - min_xyz_w) / grid_size

        # get occupied voxels
        vdict = self.get_voxel(
            xyz_w=batch["point_xyz_w"],  # (b, n, 3)
            cell_width=cell_width,
            return_packed_coord=True,
            min_xyz_w=min_xyz_w,
            max_xyz_w=max_xyz_w,
            grid_size=grid_size,
        )
        init_coord = vdict["coord"]  # (total_occ_cells, 3xyz) packed
        del vdict

        rgbd_dict_random = batch["rgbd_dict_random"]

        # # debug
        # with lightning_utils.local_rank_first(self):
        #     for key, arr in rgbd_dict_random.items():
        #         if isinstance(arr, torch.Tensor):
        #             print(f"{key}: {arr.shape} {arr.dtype}, {arr.device}")
        # # end debug

        out_dict = self.compute_3dgs_loss(
            fpoint_latent=fpoint_latent,
            init_coord=init_coord,
            H_c2w=rgbd_dict_random["H_c2w"],
            intrinsic=rgbd_dict_random["intrinsic"],
            rgb_gt=rgbd_dict_random["rgb"],
            rgb_mask_gt=rgbd_dict_random["alpha"],
            hit_mask_gt=rgbd_dict_random["hit_map"],
            normal_w_gt=rgbd_dict_random["normal_w"]
            if rgbd_dict_random["normal_w"] is not None and rgbd_dict_random["normal_w"].ndim > 1
            else None,
            # rgb
            loss_weight_rgb=self.optim_config.get("loss_weight_rgb", 1),
            loss_weight_rgb_l1=self.optim_config.get("loss_weight_rgb_l1", 1),
            loss_weight_rgb_ssim=self.optim_config.get("loss_weight_rgb_ssim", 0),
            loss_weight_rgb_lpips=self.optim_config.get("loss_weight_rgb_lpips", 0.2),
            # normal
            loss_weight_normal=self.optim_config.get("loss_weight_normal", 0.1),
            loss_weight_normal_l1=self.optim_config.get("loss_weight_normal_l1", 0),
            loss_weight_normal_ssim=self.optim_config.get("loss_weight_normal_ssim", 0),
            loss_weight_normal_lpips=self.optim_config.get("loss_weight_normal_lpips", 0),
            loss_weight_normal_sin=self.optim_config.get("loss_weight_normal_sin", 1),
            latent_coord=latent_coord,
        )
        loss = self.optim_config["loss_weight_3dgs"] * out_dict["loss"]  # (,)

        ## logging and plotting (sync_dict = True)
        need_sync_names = ["loss_3dgs_lpips"]
        for name in need_sync_names:
            ll = out_dict["other_loss_dict"].get(name, None)
            if ll is not None:
                self.log(
                    name=f"{split}/{name}",  # f"{name}_dset{dataloader_idx}" if dataloader_idx > 0 else name,
                    value=ll,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    batch_size=batch_size,
                    sync_dist=True,
                )

        ## logging and plotting (sync_dict = False)
        for name, ll in out_dict["other_loss_dict"].items():
            if name in need_sync_names:
                continue
            if ll is not None:
                self.log(
                    name=f"{split}/{name}",  # f"{name}_dset{dataloader_idx}" if dataloader_idx > 0 else name,
                    value=ll,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    batch_size=batch_size,
                    sync_dist=False,  # False if split == "train" else True,
                )

        # plot 3dgs results
        if batch_idx % 500 == 0:
            num_to_plot = 8
            plot_idx = 0
            try:
                if self.local_rank == 0:
                    tensorboard_logger: SummaryWriter = self.loggers[0].experiment

                    for key in ["rgb", "normal_w"]:
                        est = out_dict["est_maps"][key]
                        gt = out_dict["gt_maps"][key]

                        if est is None or gt is None:
                            continue

                        if key in ["normal_w"]:
                            est = (est + 1) * 0.5
                            gt = (gt + 1) * 0.5

                        _b, _q, _h, _w, d = est.shape
                        for ib in range(_b):
                            if plot_idx >= num_to_plot:
                                break

                            ee = est[ib]  # (q, h, w, d)
                            gg = gt[ib]  # (q, h, w, d)
                            img = torch.cat([ee, gg], dim=-2)  # (q, h, w*2, d)
                            tensorboard_logger.add_images(
                                tag=f"{split}_dset{dataloader_idx}_{key}/{plot_idx}"
                                if dataloader_idx > 0
                                else f"{split}_{key}/{plot_idx}",
                                img_tensor=img,
                                dataformats="NHWC",
                                global_step=self.trainer.global_step,
                            )
                            plot_idx += 1
            except:
                pass

            # save gaussian as ply
            if save_debug_gaussians:
                gs_dicts = out_dict["gs_dicts"]
                for ib in range(batch_size):
                    out_dir = self.optim_config.get("artifact_dir", "artifacts")

                    if gs_dicts[ib].get("rgb_sh", None) is not None:
                        filename = os.path.join(
                            out_dir, self.config["name"], f"gaussians_dset{dataloader_idx}", f"{ib}_rgb.ply"
                        )
                        os.makedirs(os.path.dirname(filename), exist_ok=True)

                        _sh_degree = sh_utils.get_sh_degree_from_total_dim(gs_dicts[ib]["rgb_sh"].size(-2))
                        ngs = math.prod(gs_dicts[ib]["xyz_w"].shape[:-1])
                        gs = gs_utils.Gaussians(
                            sh_degree=_sh_degree,
                            xyz_w=gs_dicts[ib]["xyz_w"].reshape(ngs, 3),  # (n, 3xyz)
                            rgb_sh=gs_dicts[ib]["rgb_sh"].reshape(ngs, -1, 3),
                            rgb_sh_dc=None,
                            rgb_sh_rest=None,
                            scaling_logit=None,
                            quaternion_prenorm=None,
                            opacity_logit=None,
                            scaling=gs_dicts[ib]["scaling"].reshape(ngs, 3),  # (n, 3xyz)
                            quaternion=gs_dicts[ib]["quaternion"].reshape(ngs, 4),  # (n, 4xyzw)
                            opacity=gs_dicts[ib]["opacity"].reshape(ngs, 1),  # (n, 1)
                            min_scaling=0,  # handled by network
                            scaling_activation_type="none",
                        )
                        gs.save_ply(filename=filename)

                    if gs_dicts[ib].get("normal_w", None) is not None:
                        filename = os.path.join(
                            out_dir, self.config["name"], f"gaussians_dset{dataloader_idx}", f"{ib}_normal_w.ply"
                        )
                        os.makedirs(os.path.dirname(filename), exist_ok=True)

                        gs = gs_utils.Gaussians(
                            sh_degree=0,
                            xyz_w=gs_dicts[ib]["xyz_w"].reshape(ngs, 3),  # (n, 3xyz)
                            rgb_sh=None,
                            rgb_sh_dc=sh_utils.RGB2SH((gs_dicts[ib]["normal_w"] + 1) * 0.5).reshape(ngs, 1, 3),
                            rgb_sh_rest=None,
                            scaling_logit=None,
                            quaternion_prenorm=None,
                            opacity_logit=None,
                            scaling=gs_dicts[ib]["scaling"].reshape(ngs, 3),  # (n, 3xyz)
                            quaternion=gs_dicts[ib]["quaternion"].reshape(ngs, 4),  # (n, 4xyzw)
                            opacity=gs_dicts[ib]["opacity"].reshape(ngs, 1),  # (n, 1)
                            min_scaling=0,  # handled by network
                            scaling_activation_type="none",
                        )
                        gs.save_ply(filename=filename)

        return loss

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

    def _step_tokenizer(
        self,
        batch: T.Dict[str, T.Any],
        batch_idx: int,
        split: str,
        run_velocity: bool = False,
        run_3dgs: bool = False,
        sample_flow: bool = False,
        print_timing: bool = False,
        dataloader_idx: int = 0,
    ):
        r"""
        Training_step defines the train loop.

        Args:
            batch:
                uid:
                    uid of the mesh
                point_xyz_w:
                    (n, 3) the point xyz in the n-coordinate
                point_rgb:
                    (n, 3) or -1, the point rgb [0, 1]
                point_normal_w:
                    (n, 3) or -1, the point normal in the n-coordinate
                point_alpha:
                    (n, 1) or -1, [0, 1]

                point_index_filename:
                    str, index_filename of the point data
                text:
                    str or -1, the text caption of mesh
                rgb_ori:
                    (b, q, 3rgb, h, w) or -1, [0, 1] the loaded views
                normal_w:
                    (b, q, 3xyz, h, w)
                z_c:
                    (b, q, 1, h, w)
                rgb_mask:
                    (b, q, 1, h, w)  bool
                H_c2w:
                    (b, q, 4, 4) or -1, camera pose of the loaded views
                intrinsic:
                    (b, q, 3, 3) or -1, camera intrinsics of the loaded views
                extra_rgbd_dict:
                    rgb_map:
                        (b, q', 3, h', w')  [0, 1]
                    normal_w_map:
                        (b, q', 3xyz, h', w')
                    z_c_map:
                        (b, q', 1, h', w')
                    rgb_mask:
                        (b, q', 1, h', w')  bool
                    H_c2w:
                        (b, q', 4, 4) or -1, camera pose of the loaded views
                    intrinsic:
                        (b, q', 3, 3) or -1, camera intrinsics of the loaded views
                patch_feature:
                    (b, q, ph, pw, d) patch feature map computed from rgb_map

            batch_idx:

            split:
                'train', 'valid', 'test'

        Returns:
            loss: (,)

        """

        total_start_event = torch.cuda.Event(enable_timing=True)
        total_end_event = torch.cuda.Event(enable_timing=True)
        start_event_img_feat = torch.cuda.Event(enable_timing=True)
        end_event_img_feat = torch.cuda.Event(enable_timing=True)
        start_event_3dgs = torch.cuda.Event(enable_timing=True)
        end_event_3dgs = torch.cuda.Event(enable_timing=True)

        total_start_event.record()

        xyz_w = batch["point_xyz_w"]  # (b, n, 3xyz)  [-1, 1]
        rgb = batch["point_rgb"]  # (b, n, 3rgb)  [0, 1]
        normal_w = batch.get("point_normal_w", None)  # (b, n, 3xyz)  [-1, 1]
        ray_origin_direction_w = batch.get("point_ray_origin_direction_w", None)  # (b, n, 6)
        point_alpha = batch.get("point_alpha", None)  # (b, n, 1)  [0, 1]

        # selecting random points for encoder, flow
        batch_size, m, d = xyz_w.shape
        device = xyz_w.device

        # randomly select encoder and decoder points
        ridxs_full = torch.randperm(m, device=device)
        ridxs_current_idx = 0
        num_encoder_points = np.random.randint(
            low=self.min_num_encoder_points,
            high=self.max_num_encoder_points + 1,
        )
        ridxs_encoder = ridxs_full[ridxs_current_idx : ridxs_current_idx + num_encoder_points]  # (m,)
        ridxs_current_idx += num_encoder_points
        num_flow_points = np.random.randint(
            low=self.min_num_flow_points,
            high=self.max_num_flow_points + 1,
        )
        ridxs_flow = ridxs_full[ridxs_current_idx : ridxs_current_idx + num_flow_points]  # (m,)
        ridxs_current_idx += num_flow_points

        # compute latent
        out_dict = self.get_latents(
            xyz_w=xyz_w[:, ridxs_encoder] if xyz_w is not None and xyz_w.ndim > 1 else None,
            rgb=rgb[:, ridxs_encoder] if rgb is not None and rgb.ndim > 1 else None,
            normal_w=normal_w[:, ridxs_encoder] if normal_w is not None and normal_w.ndim > 1 else None,
            ray_origin_direction_w=ray_origin_direction_w[:, ridxs_encoder]
            if ray_origin_direction_w is not None and ray_origin_direction_w.ndim > 1
            else None,
            alpha=point_alpha[:, ridxs_encoder] if point_alpha is not None and point_alpha.ndim > 1 else None,
        )
        fpoint_latents_coord = out_dict["latent_coord"]  # (bl, dn) or (b, num_latent, dn) or None
        fpoint_latents = out_dict["latent_tokens"]  # (bl, d) or (b, num_latent, d)
        fpoint_latents_mean = fpoint_latents

        if self.debug:
            assert fpoint_latents.isfinite().all(), (
                f"nan: {fpoint_latents.isnan().any()}, inf: {fpoint_latents.isinf().any()}"
            )

        # sample shape_latent from q(s|y)
        if self.sample_posterior:
            # print(f'sampling posterior with std {self.std_posterior}', flush=True)
            fpoint_latents = fpoint_latents_mean + self.std_posterior * torch.randn_like(
                fpoint_latents_mean
            )  # (b, num_latent, dim_latent)

        loss = 0

        # compute velocity loss
        if run_velocity and self.max_num_flow_points > 0:
            out_dict = self.compute_velocity_loss(
                fpoint_latent=fpoint_latents,
                xyz_w=xyz_w[:, ridxs_flow],
                latent_coord=fpoint_latents_coord,
            )
            loss_velocity = out_dict["loss"]  # (,)
            loss = loss + self.optim_config["loss_weight_velocity"] * loss_velocity  # (,)
        else:
            loss_velocity = None

        if run_3dgs:
            loss_3dgs = self._step_3dgs(
                batch=batch,
                batch_idx=batch_idx,
                split=split,
                fpoint_latent=fpoint_latents,
                dataloader_idx=dataloader_idx,
                latent_coord=fpoint_latents_coord,
            )
            loss = loss + loss_3dgs

        # compute kl_global divergence
        # note that we use mean (so already divided by dimension)
        if self.loss_weight_kl_global > 1e-7:
            loss_kl_global = torch.nn.functional.mse_loss(
                input=fpoint_latents_mean,
                target=torch.zeros_like(fpoint_latents_mean),
                reduction="mean",
            )
            # print(f'loss_kl_global = {loss_kl_global.item()}', flush=True)
            loss = loss + self.loss_weight_kl_global * loss_kl_global  # (,)
        else:
            loss_kl_global = None

        if self.debug:
            assert loss.isfinite().all(), f"nan: {loss.isnan().any()}, inf: {loss.isinf().any()}"

        total_end_event.record()
        if print_timing and batch_idx % 1 == 0:
            torch.cuda.synchronize()
            print(
                f"total forward takes: {total_start_event.elapsed_time(total_end_event):.2f} ms, "
                f"extracting img feature takes: {start_event_img_feat.elapsed_time(end_event_img_feat):.2f} ms, "
                f"3dgs takes: {start_event_3dgs.elapsed_time(end_event_3dgs) if run_3dgs else 0.0:.2f} ms, "
            )

        ## logging and plotting
        self.log(
            name=f"{split}/{self.optim_config['monitor_loss_name']}",
            value=loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        # loss dict
        losses = [
            [f"{split}/loss_velocity", loss_velocity],
            [f"{split}/loss_kl_global", loss_kl_global],
        ]  # 'name', loss (tensor)

        for name, ll in losses:
            if ll is not None:
                self.log(
                    name=name,  # f"{name}_dset{dataloader_idx}" if dataloader_idx > 0 else name,
                    value=ll,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    batch_size=batch_size,
                    sync_dist=False if split == "train" else True,
                )

        # sample a few point cloud
        if sample_flow and self.use_velocity:
            max_num_sample_points = 65536
            ridxs_sample = torch.randperm(xyz_w.size(1), device=xyz_w.device)[:max_num_sample_points]
            xyz_w_flow = xyz_w[:, ridxs_sample]
            init_noise_dict = self.get_conditional_sampling_init_noise(*xyz_w_flow.shape[:-1])
            sampled_x_flow_dict = self.conditional_sampling(
                fpoint_latent=fpoint_latents,  # (b, num_latent, dim_latent) or (bl, dim_latent)
                num_steps=100,  # 100,
                **init_noise_dict,
                method=self.ode_sampling_method,
                latent_coord=fpoint_latents_coord,  # (bl, dn) packed or None
            )  # (b, num_flow_points, d)

            # plot sampled and reference point cloud to tensorboard
            try:
                if self.local_rank == 0 and batch_idx == 0:
                    num_to_plot = 20
                    tensorboard_logger: SummaryWriter = self.loggers[0].experiment

                    # sampled point cloud
                    tensorboard_logger.add_mesh(
                        tag=f"valid_sampled_x_dset{dataloader_idx}/{batch_idx}"
                        if dataloader_idx > 0
                        else f"valid_sampled_x/{batch_idx}",
                        vertices=sampled_x_flow_dict["xyz_w"][:num_to_plot].float(),
                        colors=None,
                        global_step=self.trainer.global_step,
                    )
                    # reference point cloud
                    tensorboard_logger.add_mesh(
                        tag=f"valid_ref_x_dset{dataloader_idx}/{batch_idx}"
                        if dataloader_idx > 0
                        else f"valid_ref_x/{batch_idx}",
                        vertices=xyz_w_flow[:num_to_plot].float(),
                        colors=None,
                        global_step=self.trainer.global_step,
                    )
            except:
                pass

            # compute metrics like chamfer distance
            with torch.autocast(device_type="cuda", enabled=False):
                num_chamfer_points = 2048
                loss_cf_xyz, _ = pytorch3d.loss.chamfer_distance(
                    x=sampled_x_flow_dict["xyz_w"][:, :num_chamfer_points].float(),  # (b, num_flow_points, 3)
                    y=xyz_w_flow[
                        :, :num_chamfer_points
                    ].float(),  # (b, num_flow_points, 3), note it is indep to sampled_x_flow
                    batch_reduction="mean",
                    point_reduction="mean",
                    single_directional=False,
                    abs_cosine=True,
                )  # (,) or None

                xyz_w_input = xyz_w[:, ridxs_encoder]  # (b, n, 3)
                if xyz_w_input.size(1) >= num_chamfer_points:
                    xyz_w_input = xyz_w_input[:, :num_chamfer_points]
                else:
                    rr = torch.randint(high=xyz_w_input.size(1), size=(num_chamfer_points,), device=xyz_w_input.device)
                    xyz_w_input = xyz_w_input[:, rr]
                loss_cf_xyz_input, _ = pytorch3d.loss.chamfer_distance(
                    x=xyz_w_input[:, :num_chamfer_points].float(),  # (b, num_flow_points, 3)
                    y=xyz_w_flow[
                        :, :num_chamfer_points
                    ].float(),  # (b, num_flow_points, 3), note it is indep to sampled_x_flow
                    batch_reduction="mean",
                    point_reduction="mean",
                    single_directional=False,
                    abs_cosine=True,
                )  # (,) or None

                self.log(
                    name=f"{split}/chamfer_xyz",  # f"{split}_dset{dataloader_idx}/chamfer_xyz" if dataloader_idx > 0 else f"{split}/chamfer_xyz",
                    value=loss_cf_xyz,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    batch_size=batch_size,
                    sync_dist=True,
                )
                self.log(
                    name=f"{split}/chamfer_xyz_input",  # f"{split}_dset{dataloader_idx}/chamfer_xyz_input" if dataloader_idx > 0 else f"{split}/chamfer_xyz_input",
                    value=loss_cf_xyz_input,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    batch_size=batch_size,
                    sync_dist=False if split == "train" else True,
                )
        return loss

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

    def compute_voxel_loss(
        self,
        fpoint_latent: torch.Tensor,
        gt_occ_grid: torch.Tensor,
        loss_weight_voxel_l2: float = 0,
        loss_weight_voxel_l1: float = 0,
        loss_weight_voxel_huber: float = 1,
        return_occ_grid: bool = False,
    ):
        """
        Compute voxel estimation loss given the ground truth

        Args:
            fpoint_latent:
                (b, num_latent, dim_latent)
            gt_occ_grid:
                (b, 1, res_z, res_y, res_x) bool
            return_occ_grid_logit:
                if True, we will run the sparse_structure_decoder to convert
                ss_latent to dense occ grid [0, 1]

        Returns:
            loss:
                (,) total loss
            est_occ_grid:
                (b, 1, res_z, res_y, res_x) [0, 1]
            gt_occ_grid:
                (b, 1, res_z, res_y, res_x) bool
        """
        assert self.voxel_decoder is not None
        assert self.voxel_ss_pipeline is not None
        b, _1, res_z, res_y, res_x = gt_occ_grid.shape
        assert res_z == res_y == res_x == 64

        # compute sparse latent
        with torch.no_grad():
            if self.voxel_ss_pipeline.device != self.device:
                self.voxel_ss_pipeline.to(device=self.device)
            gt_ss_latent = self.voxel_ss_pipeline.encode_highres_grid(
                dense_grid=gt_occ_grid.float()
            )  # (b, d, low_res_k, low_res_j, low_res_i)

        # estimate occ_grid
        occ_dict = self.estimate_occ_grid(
            latent_token=fpoint_latent,
            return_occ_grid=return_occ_grid,
        )
        est_ss_latent = occ_dict["est_ss_latent"]  # (b, d, low_res_k, low_res_j, low_res_i) float
        est_occ_grid = occ_dict.get("est_occ_grid", None)  # (b, 1, res_k, res_j, res_i) float

        if self.debug:
            assert est_ss_latent.isfinite().all(), (
                f"nan: {est_ss_latent.isnan().any()}, inf: {est_ss_latent.isinf().any()}"
            )
            if est_occ_grid is not None:
                assert est_occ_grid.isfinite().all(), (
                    f"nan: {est_occ_grid.isnan().any()}, inf: {est_occ_grid.isinf().any()}"
                )

        # compute loss
        loss_ss_latent_l2 = torch.nn.functional.mse_loss(
            input=est_ss_latent,  # (b, d, low_res_k, low_res_j, low_res_i)
            target=gt_ss_latent.to(dtype=est_ss_latent.dtype),  # (b, d, low_res_k, low_res_j, low_res_i)
            reduction="mean",
        )  # (,)
        loss_ss_latent_l1 = torch.nn.functional.l1_loss(
            input=est_ss_latent,  # (b, d, low_res_k, low_res_j, low_res_i)
            target=gt_ss_latent.to(dtype=est_ss_latent.dtype),  # (b, d, low_res_k, low_res_j, low_res_i)
            reduction="mean",
        )  # (,)
        loss_ss_latent_huber = torch.nn.functional.huber_loss(
            input=est_ss_latent,  # (b, d, low_res_k, low_res_j, low_res_i)
            target=gt_ss_latent.to(dtype=est_ss_latent.dtype),  # (b, d, low_res_k, low_res_j, low_res_i)
            reduction="mean",
        )  # (,)

        loss = (
            loss_weight_voxel_l2 * loss_ss_latent_l2
            + loss_weight_voxel_l1 * loss_ss_latent_l1
            + loss_weight_voxel_huber * loss_ss_latent_huber
        )

        if self.debug:
            assert loss_ss_latent_l2.isfinite().all(), (
                f"nan: {loss_ss_latent_l2.isnan().any()}, inf: {loss_ss_latent_l2.isinf().any()}"
            )

        if est_occ_grid is not None:
            acc = ((est_occ_grid > 0.5) == gt_occ_grid.bool()).float().mean()  # (,)
        else:
            acc = None

        return dict(
            loss=loss,  # (,)
            loss_voxel_l2=loss_ss_latent_l2,  # (,)
            loss_voxel_l1=loss_ss_latent_l1,  # (,)
            loss_voxel_huber=loss_ss_latent_huber,  # (,)
            acc=acc,  # (,) or None
            gt_occ_grid=gt_occ_grid,  # (b, 1, res_k, res_j, res_i) bool
            est_occ_grid=est_occ_grid,  # (b, 1, res_k, res_j, res_i) float [0, 1]
        )

    def _step_voxel(
        self,
        batch: T.Dict[str, T.Any],
        batch_idx: int,
        split: str,
        fpoint_latent: torch.Tensor = None,  # (b, num_latent, dim_latent)
        print_timing: bool = False,
        dataloader_idx: int = 0,
        latent_coord: T.Optional[ppoint.PackedPoint] = None,
    ):
        """
        Train/valid step of pointersect.

        Args:
            batch:
                latent_token:
                    (num_latent, dim_latent)
                H_w2n:
                    (4, 4) or None.  The coordinate transform that was used to compute latent_token
                rgb_map:
                    (b=1, q, h, w, 3rgb) [0, 1] or None.  The image taken in the n-coordinate.
                depth_map:
                    (b=1, q, h, w) or None. The z_c of the camera.
                normal_w_map:
                    (b=1, q, h, w, 3xyz_n) or None. The normal map in the n-coordinate.
                hit_map:
                    (b=1, q, h, w) or None. whether a pixel hit the shape.
                uid:
                    uid of the mesh
                point_xyz_w:
                    (n, 3) the point xyz in the n-coordinate
                point_rgb:
                    (n, 3) or -1, the point rgb [0, 1]
                point_normal_w:
                    (n, 3) or -1, the point normal in the n-coordinate
                point_index_filename:
                    str, index_filename of the point data
                text:
                    str or -1, the text caption of mesh
                rgb_ori:
                    (b, q, 3rgb, h, w) or -1, [0, 1] the loaded views
                H_c2w:
                    (b, q, 4, 4) or -1, camera pose of the loaded views in the n-coordinate
                intrinsic:
                    (b, q, 3, 3) or -1, camera intrinsics of the loaded views
                rgb_mask:
                    (b, q, 1, h, w) bool, the valid mask of rgb_ori
                normal_w:
                    (b, q, 3xyz, h, w) or -1, [-1, 1] normal map of the loaded views in the n-coordinate
                ray_t:
                    (b, q, 1, h, w) or -1.  If keep_ray_t_z_c_bug is True, z_c (z coordinate in the camera coordinate).
                    Else, typical ray travelling distance.
            batch_idx:

            split:
                'train', 'valid', 'test'
        Returns:
            loss:
                (,)
        """
        assert batch["point_xyz_w"].ndim > 1
        # assert batch["occ_grid"].ndim > 1  # (b, res_z, res_y, res_x) bool

        # compute shape latent
        if fpoint_latent is None:
            xyz_w = batch["point_xyz_w"]  # (b, m, 3)
            rgb = batch["point_rgb"]  # (b, m, 3)
            normal_w = batch.get("point_normal_w", None)  # (b, n, 3xyz)  [-1, 1]
            ray_origin_direction_w = batch.get("point_ray_origin_direction_w", None)  # (b, n, 6)
            point_alpha = batch.get("point_alpha", None)  # (b, n, 1)  [0, 1]

            m = xyz_w.size(2)
            device = xyz_w.device

            # randomly select encoder and decoder points
            ridxs_full = torch.randperm(m, device=device)
            ridxs_current_idx = 0
            num_encoder_points = np.random.randint(
                low=self.min_num_encoder_points,
                high=self.max_num_encoder_points + 1,
            )
            ridxs_encoder = ridxs_full[ridxs_current_idx : ridxs_current_idx + num_encoder_points]  # (m,)
            ridxs_current_idx += num_encoder_points

            with torch.no_grad():
                out_dict = self.get_latents(
                    xyz_w=xyz_w[:, ridxs_encoder],
                    rgb=rgb[:, ridxs_encoder] if rgb is not None and rgb.ndim > 1 else None,
                    normal_w=normal_w[:, ridxs_encoder] if normal_w is not None and normal_w.ndim > 1 else None,
                    ray_origin_direction_w=ray_origin_direction_w[:, ridxs_encoder]
                    if ray_origin_direction_w is not None and ray_origin_direction_w.ndim > 1
                    else None,
                    alpha=point_alpha[:, ridxs_encoder] if point_alpha is not None and point_alpha.ndim > 1 else None,
                )
                fpoint_latent_coord = out_dict["latent_coord"]  # (bl, dn) or (b, num_latent, dn) or None
                fpoint_latent = out_dict["latent_tokens"]  # (bl, d) or (b, num_latent, d)

                # sample shape_latent from q(s|y)
                if self.sample_posterior:
                    # print(f'sampling posterior with std {self.std_posterior}', flush=True)
                    fpoint_latent = fpoint_latent + self.std_posterior * torch.randn_like(
                        fpoint_latent
                    )  # (b, num_latent, dim_latent)

        # construct gt_occ_grid
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):
            min_xyz_w, max_xyz_w, grid_size = -1.0, 1.0, 64
            cell_width = (max_xyz_w - min_xyz_w) / grid_size
            # randomly select 100000 points
            _xyz_w = batch["point_xyz_w"]  # (b, n, 3)
            ridx = torch.randperm(_xyz_w.size(1), device=self.device)[:100000]
            _xyz_w = _xyz_w[:, ridx]  # (b, n, 3)

            # get occupied voxels. since we need occ_grid, it is faster if we
            # directly get the occ_grid instead of calling get_voxel
            _ijk = torch.floor((_xyz_w - min_xyz_w) / cell_width).long()  # (b, n, 3ijk)
            _ijk = torch.clamp(_ijk, min=0, max=grid_size - 1)  # (b, n, 3ijk)

            gt_occ_grid = torch.zeros(
                _ijk.size(0),
                1,
                grid_size,
                grid_size,
                grid_size,
                dtype=torch.bool,
                device=self.device,
            )  # (b, 1, res_z, res_y, res_x) bool
            i, j, k = _ijk.unbind(-1)  # (b, n)
            # create a batch index to match shape
            bidx = torch.arange(_ijk.shape[0], device=_ijk.device).unsqueeze(-1).expand(-1, _ijk.shape[1])  # (b, n)
            gt_occ_grid[bidx, 0, k, j, i] = True  # intentional overlapping write for efficiency

            del i
            del j
            del k
            del bidx
            del _ijk
            del _xyz_w
            del ridx

        # compute loss
        voxel_out_dict = self.compute_voxel_loss(
            fpoint_latent=fpoint_latent,
            gt_occ_grid=gt_occ_grid,  # (b, 1, res_z, res_y, res_x) bool
            loss_weight_voxel_l2=self.optim_config["loss_weight_voxel_l2"],
            loss_weight_voxel_l1=self.optim_config["loss_weight_voxel_l1"],
            loss_weight_voxel_huber=self.optim_config["loss_weight_voxel_huber"],
            return_occ_grid=(split == "valid"),
        )
        loss = voxel_out_dict["loss"]  # (,)
        acc = voxel_out_dict["acc"]  # (,)

        if self.debug:
            assert loss.isfinite().all(), f"nan: {loss.isnan().any()}, inf: {loss.isinf().any()}"

        # loss dict
        losses = [
            [f"{split}/loss_voxel", loss],
            [f"{split}/loss_voxel_l2", voxel_out_dict["loss_voxel_l2"]],
            [f"{split}/loss_voxel_l1", voxel_out_dict["loss_voxel_l1"]],
            [f"{split}/loss_voxel_huber", voxel_out_dict["loss_voxel_huber"]],
            [f"{split}/acc_voxel", acc],
        ]  # 'name', loss (tensor)

        for name, ll in losses:
            if ll is not None:
                self.log(
                    name=name,
                    value=ll,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    batch_size=fpoint_latent.size(0),
                    sync_dist=False if split == "train" else True,
                )

        # visualize
        if split == "valid":
            num_to_plot = 20
            num_batch_to_plot = max(1, num_to_plot // gt_occ_grid.size(0))
            plot_idx = 0
            if self.trainer.local_rank == 0 and batch_idx < num_batch_to_plot:
                tensorboard_logger: SummaryWriter = self.loggers[0].experiment

                for ib in range(gt_occ_grid.size(0)):
                    if plot_idx >= num_to_plot:
                        break

                    for name in [
                        "est_occ_grid",
                        "gt_occ_grid",
                    ]:
                        voxel_grid = voxel_out_dict[name]  # (b, 1, r, r, r)

                        kji = torch.argwhere(voxel_grid[ib, 0] > 0.5)  # (num, 3kji)
                        color = kji.float() / voxel_grid.size(-1)  # (num, 3rgb)

                        tensorboard_logger.add_mesh(
                            tag=f"{split}_{name}_{dataloader_idx}/{plot_idx}",
                            vertices=kji.float().unsqueeze(0),  # (1, num, 3kji)
                            colors=(color.float().clamp(min=0, max=1) * 255).to(dtype=torch.uint8).unsqueeze(0),
                            global_step=self.trainer.global_step,
                        )

                    plot_idx += 1

        return loss

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

    def compute_mesh_loss(
        self,
        latent_token: torch.Tensor,
        input_occ_grid: torch.Tensor,
        camera: structures.Camera,
        use_masked_loss: bool,
        gt_hit_map: torch.Tensor,
        gt_depth_map: torch.Tensor,
        gt_xyz_c: torch.Tensor,
        gt_normal_map: torch.Tensor,
        normal_align_mode: str = "gt",
        normalize_loss_by_valid_region_size: bool = True,
        loss_weight_mask: float = 1,
        loss_weight_xyz_c: float = 10,
        loss_weight_xyz_c_l2_sqrt: float = 1,
        loss_weight_z_c: float = 10,
        loss_weight_z_c_huber: float = 1,
        loss_weight_z_c_l2: float = 0,
        loss_weight_face_normal: float = 1,
        loss_weight_face_normal_sinusoid: float = 0,
        loss_weight_face_normal_l1: float = 1,
        loss_weight_face_normal_ssim: float = 0.2,
        loss_weight_face_normal_lpips: float = 0.2,
        loss_weight_vertex_normal: float = 0.1,
        loss_weight_vertex_normal_sinusoid: float = 0,
        loss_weight_vertex_normal_l1: float = 1,
        loss_weight_vertex_normal_ssim: float = 0.2,
        loss_weight_vertex_normal_lpips: float = 0.2,
        loss_weight_reg: float = 1,
        loss_weight_reg_sdf: float = 0.2,
    ):
        """
        Compute voxel estimation loss given the ground truth

        Args:
            latent_token:
                (b, num_latent, dim_latent)
            input_occ_grid:
                (b, 1, res_z, res_y, res_x) bool, where we will be computing
                the SDF and other flexicube parameters
            use_masked_loss:
                bool, if True, we enforce depth/normal loss in masked region based on GT mask.
                Otherwise, we compute depth/normal loss on the whole image.
            gt_hit_map:
                (b, q, h, w) bool
            gt_depth_map:
                (b, q, h, w)  z_c
            gt_normal_map:
                (b, q, h, w, 3xyz_w)  normal in the world coordinates

        Returns:
            loss:
                (,) total loss
            est_mesh:

        """

        # print(f'normal_align_mode = {normal_align_mode}')
        b, _1, res_z, res_y, res_x = input_occ_grid.shape

        # estimate mesh
        out_dict = self.estimate_mesh(
            latent_token=latent_token,
            occ_bijk=None,
            grid_size=input_occ_grid.size(2),
            input_occ_grid=input_occ_grid,
        )
        raw_meshes: T.List[structures.RawMesh] = out_dict["raw_meshes"]
        reg_losses: T.List[torch.Tensor] = out_dict["reg_losses"]
        reg_sdf_losses: T.List[torch.Tensor] = out_dict["reg_sdf_losses"]

        # tmp_dir = pathlib.Path("/mnt/test/code/shape_tokenization/debug/debug_model")
        # tmp_dir.mkdir(parents=True, exist_ok=True)
        # for tmp_i, tmp_raw_mesh in enumerate(raw_meshes):
        #     tmp_mesh = tmp_raw_mesh.get_o3d_mesh(with_vertex_normal_w=True)
        #     tmp_f = tmp_dir / f"save_in_trainer_{tmp_i:04d}.ply"
        #     o3d.io.write_triangle_mesh(
        #         filename=str(tmp_f),
        #         mesh=tmp_mesh,
        #     )

        # tmp_hash = hash_state_dict(self.state_dict())
        # print(f"\n\n{tmp_hash=}\n\n")

        # make sure gt_depth_map and gt_normal_map have black background (which our rendering will be)
        gt_depth_map = gt_depth_map * gt_hit_map.to(dtype=gt_depth_map.dtype)
        gt_normal_map = gt_normal_map * gt_hit_map.unsqueeze(-1).to(dtype=gt_depth_map.dtype)

        if self.glctx is None:
            if self.mesh_optim_config.get("glctx", "opengl") == "opengl":
                # opengl is slow to initialize, but does not return memory
                self.glctx = structures.RawMesh.get_glctx(
                    method="opengl",
                    device=self.device,
                )
                glctx = self.glctx
            elif self.mesh_optim_config.get("glctx", "opengl") == "cuda":
                # cuda is fast to initialize, so we initialize every iter
                glctx = structures.RawMesh.get_glctx(
                    method="cuda",
                    device=self.device,
                )
            else:
                raise NotImplementedError
        else:
            glctx = self.glctx

        # render z_c and normal maps
        cameras = camera.chunk(b, dim=0)
        rdicts = []
        valids = []

        total_est_hit = 0
        for ib in range(len(raw_meshes)):
            raw_mesh = raw_meshes[ib]
            if raw_mesh is not None:
                with torch.autocast(device_type=self.device.type, enabled=False):
                    rdict = raw_mesh.render(
                        camera=cameras[ib].to(device=self.device, dtype=torch.float),
                        return_types=[
                            "mask",
                            "z_c",
                            "xyz_c",
                            "vertex_normal",
                            "face_normal",
                        ],
                        t_min=1e-2,
                        t_max=1000,
                        glctx=glctx,
                        normalize_vertex_normal=True,  # False,
                        normalize_face_normal=True,  # False,
                        max_img_chunk_size=10,
                    )  # (1, q, h, w, d), background is (0, 0, 0) even for vertex or face normal
                    valids.append(1)
                    total_est_hit += rdict["mask"].detach().sum()
            else:
                # use gt to zero-out loss
                rdict_z_c = gt_depth_map[ib : ib + 1].unsqueeze(-1)
                rdict_xyz_c = rdict_z_c.expand(*rdict_z_c.shape[:-1], 3)
                rdict = dict(
                    mask=gt_hit_map[ib : ib + 1].float().unsqueeze(-1),  # (1, q, h, w, 1)
                    z_c=rdict_z_c,  # (1, q, h, w, 1)
                    xyz_c=rdict_xyz_c,  # (1, q, h, w, 1)
                    vertex_normal=torch.ones_like(gt_normal_map[ib : ib + 1]) / math.sqrt(3),  # (1, q, h, w, 3)
                    face_normal=torch.ones_like(gt_normal_map[ib : ib + 1]) / math.sqrt(3),  # (1, q, h, w, 3)
                )
                valids.append(0)
            rdicts.append(rdict)

        rdict = utils.cat_dict(rdicts, dim_dict=0)

        valids = torch.tensor(valids, dtype=torch.bool, device=latent_token.device)  # (b,)

        gt_masks = gt_hit_map.float().unsqueeze(-1)  # (b, q, h, w, 1)
        gt_masks = gt_masks.masked_fill(~valids.reshape(b, 1, 1, 1, 1), 0)

        if normalize_loss_by_valid_region_size:
            # scale = rdict["mask"].numel() / total_est_hit
            total_denom = torch.sum(gt_masks)
        else:
            # scale = 1
            total_denom = gt_masks.numel()

        # if isinstance(total_est_hit, (int, float)):
        #     total_est_hit = max(1, total_est_hit)
        # elif isinstance(total_est_hit, torch.Tensor):
        #     total_est_hit = total_est_hit.clamp(min=1)
        # else:
        #     raise NotImplementedError

        if self.debug:
            for key in rdict:
                if isinstance(rdict[key], torch.Tensor):
                    assert rdict[key].isfinite().all(), (
                        f"{key}: nan: {rdict[key].isnan().any()}, inf: {rdict[key].isinf().any()}"
                    )

        # align face_normal and vertex_normal
        if normal_align_mode == "gt":
            for name in ["face_normal", "vertex_normal"]:
                est_normal = rdict[name].detach()  # (b, q, h, w, 3) not normalized
                _dot = (est_normal * gt_normal_map).sum(dim=-1, keepdim=True)  # (b, q, h, w, 1)
                _s = torch.ones_like(_dot)
                _s[_dot < 0] = -1
                rdict[name] = rdict[name] * _s.detach()

                if self.debug:
                    assert gt_normal_map.isfinite().all(), (
                        f"nan: {gt_normal_map.isnan().any()}, inf: {gt_normal_map.isinf().any()}"
                    )
        elif normal_align_mode == "origin":
            with torch.no_grad():
                _est_xyz_w = utils.compute_3d_xyz(
                    z_map=rdict["z_c"].squeeze(-1).detach(),  # (b, q, h, w)
                    intrinsic=camera.intrinsic,  # (b, q, 3, 3)
                    H_c2w=camera.H_c2w,  # (b, q, 4, 4)
                )["xyz_w"]  # (b, q, h, w, 3)

                if self.debug:
                    assert _est_xyz_w.isfinite().all(), (
                        f"nan: {_est_xyz_w.isnan().any()}, inf: {_est_xyz_w.isinf().any()}"
                    )

            # est
            for name in ["face_normal", "vertex_normal"]:
                est_normal = rdict[name].detach()  # (b, q, h, w, 3) not normalized
                _dot = (est_normal * _est_xyz_w).sum(dim=-1, keepdim=True)  # (b, q, h, w, 1)
                _s = torch.ones_like(_dot)
                _s[_dot < 0] = -1
                rdict[name] = rdict[name] * _s.detach()

                if self.debug:
                    assert rdict[name].isfinite().all(), (
                        f"{name}: nan: {rdict[name].isnan().any()}, inf: {rdict[name].isinf().any()}"
                    )

            del _est_xyz_w

            # gt
            with torch.no_grad():
                _gt_xyz_w = utils.compute_3d_xyz(
                    z_map=gt_depth_map,  # (b, q, h, w)
                    intrinsic=camera.intrinsic,  # (b, q, 3, 3)
                    H_c2w=camera.H_c2w,  # (b, q, 4, 4)
                )["xyz_w"]  # (b, q, h, w, 3)

                if self.debug:
                    assert _gt_xyz_w.isfinite().all(), f"nan: {_gt_xyz_w.isnan().any()}, inf: {_gt_xyz_w.isinf().any()}"

                _dot = (gt_normal_map * _gt_xyz_w).sum(dim=-1, keepdim=True)  # (b, q, h, w, 1)
                _s = torch.ones_like(_dot)
                _s[_dot < 0] = -1
                gt_normal_map = gt_normal_map * _s.detach()

                if self.debug:
                    assert gt_normal_map.isfinite().all(), (
                        f"nan: {gt_normal_map.isnan().any()}, inf: {gt_normal_map.isinf().any()}"
                    )

        else:
            raise NotImplementedError

        # mask:  l1 on the entire image
        loss_mask = torch.nn.functional.l1_loss(
            input=rdict["mask"].squeeze(-1).float(),
            target=gt_hit_map.float(),
            reduction="none",
        )  # (b, q, h, w)
        loss_mask = loss_mask.masked_fill(~valids.reshape(b, 1, 1, 1), 0)
        loss_mask = loss_mask.sum() / (total_denom + 1e-8)  # (,)

        # xyz in camera coordinates
        est_xyz_c = rdict["xyz_c"].float()  # (b, q, h, w, 3)
        ref_xyz_c = gt_xyz_c.float()
        if use_masked_loss:
            # print(f"\n\n{est_z_c.shape=}, {ref_z_c.shape=}, {gt_masks.shape=}\n\n")
            est_xyz_c = est_xyz_c * gt_masks
            ref_xyz_c = ref_xyz_c * gt_masks
        # We compute Euclidean distance
        # See https://github.com/nv-tlabs/FlexiCubes/blob/4cc7d6c3d0cee83c011ce36721b81adff0dd7db6/examples/optimize.py#L107
        loss_xyz_c_l2_sqrt = (((est_xyz_c - ref_xyz_c) ** 2).sum(-1) + 1e-8).sqrt()  # (b, q, h, w)
        loss_xyz_c_l2_sqrt = loss_xyz_c_l2_sqrt.masked_fill(~valids.reshape(b, 1, 1, 1), 0)
        loss_xyz_c_l2_sqrt = loss_xyz_c_l2_sqrt.sum() / (total_denom + 1e-8)  # (,)
        loss_xyz_c = loss_weight_xyz_c_l2_sqrt * loss_xyz_c_l2_sqrt

        # z_c: 10 * huber (on the entire image)
        est_z_c = rdict["z_c"].squeeze(-1).float()
        ref_z_c = gt_depth_map.float()
        if use_masked_loss:
            # print(f"\n\n{est_z_c.shape=}, {ref_z_c.shape=}, {gt_masks.shape=}\n\n")
            est_z_c = est_z_c * gt_masks[..., 0]
            ref_z_c = ref_z_c * gt_masks[..., 0]
        loss_z_c_huber = torch.nn.functional.huber_loss(input=est_z_c, target=ref_z_c, reduction="none")  # (b, q, h, w)
        loss_z_c_huber = loss_z_c_huber.masked_fill(~valids.reshape(b, 1, 1, 1), 0)
        loss_z_c_huber = loss_z_c_huber.sum() / (total_denom + 1e-8)  # (,)

        loss_z_c_l2 = torch.nn.functional.mse_loss(input=est_z_c, target=ref_z_c, reduction="none")  # (b, q, h, w)
        loss_z_c_l2 = loss_z_c_l2.masked_fill(~valids.reshape(b, 1, 1, 1), 0)
        loss_z_c_l2 = loss_z_c_l2.sum() / (total_denom + 1e-8)  # (,)
        loss_z_c = loss_weight_z_c_l2 * loss_z_c_l2 + loss_weight_z_c_huber * loss_z_c_huber

        # face_normal:  l1 + 0.2 * (1 - ssim) + 0.2 * lpips
        est_face_normal = torch.nn.functional.normalize(rdict["face_normal"].float(), p=2.0, dim=-1)
        ref_face_normal = torch.nn.functional.normalize(gt_normal_map.float(), p=2.0, dim=-1)  # (b, #view, h, w, 3)
        if use_masked_loss:
            # print(f"\n\n{est_face_normal.shape=}, {ref_face_normal.shape=}, {gt_masks.shape=}\n\n")
            est_face_normal = est_face_normal * gt_masks
            ref_face_normal = ref_face_normal * gt_masks
        loss_face_normal = self.compute_loss_recon(
            est=est_face_normal,  #  [-1, 1]
            tgt=ref_face_normal,  #  [-1, 1]
            valid_mask=valids.unsqueeze(-1).expand(b, ref_face_normal.size(1)),  # (b, q)
            loss_weight_l1=loss_weight_face_normal_l1,
            loss_weight_ssim=loss_weight_face_normal_ssim,
            loss_weight_lpips=loss_weight_face_normal_lpips,
        )["loss"]

        # sin(theta)^2, full_bg (make sure the background in gt is black)
        # note that when both gt_normal_map and est_normal_map are (0, 0, 0),
        # the dot product does not work. we get max loss but it is ok since they do not have gradient
        loss_face_normal_sinusoid = 1 - ((ref_face_normal * est_face_normal).sum(dim=-1) ** 2)  # (b, q, h, w)
        loss_face_normal_sinusoid = loss_face_normal_sinusoid.mean()  # full bg
        loss_face_normal = loss_face_normal + loss_weight_face_normal_sinusoid * loss_face_normal_sinusoid

        # vertex_normal:  l1 + 0.2 * (1 - ssim) + 0.2 * lpips
        est_vertex_normal = torch.nn.functional.normalize(rdict["vertex_normal"].float(), p=2.0, dim=-1)
        ref_vertex_normal = torch.nn.functional.normalize(gt_normal_map.float(), p=2.0, dim=-1)
        if use_masked_loss:
            # print(f"\n\n{est_vertex_normal.shape=}, {ref_vertex_normal.shape=}, {gt_masks.shape=}\n\n")
            est_vertex_normal = est_vertex_normal * gt_masks
            ref_vertex_normal = ref_vertex_normal * gt_masks
        loss_vertex_normal = self.compute_loss_recon(
            est=est_vertex_normal,  #  [-1, 1]
            tgt=ref_vertex_normal,  #  [-1, 1]
            valid_mask=valids.unsqueeze(-1).expand(b, ref_vertex_normal.size(1)),  # (b, q)
            loss_weight_l1=loss_weight_vertex_normal_l1,
            loss_weight_ssim=loss_weight_vertex_normal_ssim,
            loss_weight_lpips=loss_weight_vertex_normal_lpips,
        )["loss"]

        # sin(theta)^2, full_bg (make sure the background in gt is black)
        loss_vertex_normal_sinusoid = 1 - ((ref_vertex_normal * est_vertex_normal).sum(dim=-1) ** 2)  # (b, q, h, w)
        loss_vertex_normal_sinusoid = loss_vertex_normal_sinusoid.mean()  # full bg
        loss_vertex_normal = loss_vertex_normal + loss_weight_vertex_normal_sinusoid * loss_vertex_normal_sinusoid

        # loss_reg
        loss_reg = 0.0
        _count_reg = 0
        for tmp_l in reg_losses:
            if (tmp_l is not None) and (torch.isfinite(tmp_l).all()):
                loss_reg += tmp_l
                _count_reg += 1
        # if normalize_loss_by_valid_region_size:
        loss_reg = loss_reg / max(1, _count_reg)
        # else:
        #     loss_reg = loss_reg / len(reg_losses)

        # loss_reg_sdf
        loss_reg_sdf = 0.0
        _count_reg_sdf = 0
        for tmp_l in reg_sdf_losses:
            if (tmp_l is not None) and (torch.isfinite(tmp_l).all()):
                loss_reg_sdf += tmp_l
                _count_reg_sdf += 1
        loss_reg_sdf = loss_reg_sdf / max(1, _count_reg_sdf)

        loss = (
            loss_weight_mask * loss_mask
            + loss_weight_xyz_c * loss_xyz_c
            + loss_weight_z_c * loss_z_c
            + loss_weight_face_normal * loss_face_normal
            + loss_weight_vertex_normal * loss_vertex_normal
            + loss_weight_reg * loss_reg
            + loss_weight_reg_sdf * loss_reg_sdf
        )
        if self.debug:
            assert loss.isfinite().all(), f"nan: {loss.isnan().any()}, inf: {loss.isinf().any()}"

        # in case no valid mesh
        flag_loss_nan = loss.isnan().any()
        flag_loss_infinite = not loss.isfinite().all()
        flag_param_nan = torch.any(torch.BoolTensor([p.data.isnan().any() for p in self.parameters()]))
        if (not isinstance(loss, torch.Tensor)) or (not loss.requires_grad) or flag_loss_nan or flag_loss_infinite:
            loss = sum([0 * p.mean() for p in self.parameters()])

        return dict(
            loss=loss,  # (,)
            loss_mask=loss_mask.item(),  # (,)
            loss_z_c=loss_z_c.item(),  # (,)
            loss_xyz_c=loss_xyz_c.item(),
            loss_face_normal=loss_face_normal.item(),  # (,)
            loss_vertex_normal=loss_vertex_normal.item(),  # (,)
            loss_reg=loss_reg.item() if isinstance(loss_reg, torch.Tensor) else loss_reg,  # (,)
            loss_reg_sdf=loss_reg_sdf.item() if isinstance(loss_reg_sdf, torch.Tensor) else loss_reg_sdf,  # (,)
            raw_meshes=raw_meshes,
            camera=camera,  # (b, q)
            gt_hit_map=gt_hit_map,  # (b, q, h, w)
            gt_depth_map=gt_depth_map,  # (b, q, h, w)  black background
            gt_normal_map=gt_normal_map,  # (b, q, h, w, 3)  black background
            est_hit_map=rdict["mask"].squeeze(-1),  # (b, q, h, w)  [0, 1] float
            est_depth_map=rdict["z_c"].squeeze(-1),  # (b, q, h, w) black background, unmasked version
            est_face_normal_map=rdict["face_normal"].float(),  # (b, q, h, w, 3)  black background, unmasked version
            est_vertex_normal_map=rdict["vertex_normal"].float(),  # (b, q, h, w, 3)  black background, unmasked version
            flag_loss_nan=float(flag_loss_nan),
            flag_loss_infinite=float(flag_loss_infinite),
            flag_param_nan=float(flag_param_nan),
            valid_ratio=torch.mean(valids.float()),
        )

    def _step_mesh(
        self,
        batch: T.Dict[str, T.Any],
        batch_idx: int,
        split: str,
        dataloader_idx: int = 0,
    ):
        """
        Train/valid step of flexicube meshing.

        Args:
            batch:
                latent_token:
                    (num_latent, dim_latent)
                H_w2n:
                    (4, 4) or None.  The coordinate transform that was used to compute latent_token
                rgb_map:
                    (b=1, q, h, w, 3rgb) [0, 1] or None.  The image taken in the n-coordinate.
                depth_map:
                    (b=1, q, h, w) or None. The z_c of the camera.
                normal_w_map:
                    (b=1, q, h, w, 3xyz_n) or None. The normal map in the n-coordinate.
                hit_map:
                    (b=1, q, h, w) or None. whether a pixel hit the shape.
                uid:
                    uid of the mesh
                point_xyz_w:
                    (n, 3) the point xyz in the n-coordinate
                point_rgb:
                    (n, 3) or -1, the point rgb [-1, 1]
                point_normal_w:
                    (n, 3) or -1, the point normal in the n-coordinate
                point_index_filename:
                    str, index_filename of the point data
                text:
                    str or -1, the text caption of mesh
                rgb_ori:
                    (b, q, 3rgb, h, w) or -1, [0, 1] the loaded views
                H_c2w:
                    (b, q, 4, 4) or -1, camera pose of the loaded views in the n-coordinate
                intrinsic:
                    (b, q, 3, 3) or -1, camera intrinsics of the loaded views
                rgb_mask:
                    (b, q, 1, h, w) bool, the valid mask of rgb_ori
                normal_w:
                    (b, q, 3xyz, h, w) or -1, [-1, 1] normal map of the loaded views in the n-coordinate
                ray_t:
                    (b, q, 1, h, w) or -1.  If keep_ray_t_z_c_bug is True, z_c (z coordinate in the camera coordinate).
                    Else, typical ray travelling distance.
            batch_idx:

            split:
                'train', 'valid', 'test'
        Returns:
            loss:
                (,)
        """
        assert batch["point_xyz_w"].ndim > 1

        xyz_w = batch["point_xyz_w"]  # (b, n, 3xyz)  [-1, 1]
        rgb = batch["point_rgb"]  # (b, n, 3rgb)  [0, 1]
        normal_w = batch.get("point_normal_w", None)  # (b, n, 3xyz)  [-1, 1]
        ray_origin_direction_w = batch.get("point_ray_origin_direction_w", None)  # (b, n, 6)
        point_alpha = batch.get("point_alpha", None)  # (b, n, 1)  [0, 1]

        # selecting random points for encoder
        batch_size, m, d = xyz_w.shape
        device = xyz_w.device

        # randomly select encoder and decoder points
        ridxs_full = torch.randperm(m, device=device)
        ridxs_current_idx = 0
        num_encoder_points = np.random.randint(
            low=self.min_num_encoder_points,
            high=self.max_num_encoder_points + 1,
        )
        ridxs_encoder = ridxs_full[ridxs_current_idx : ridxs_current_idx + num_encoder_points]  # (m,)
        ridxs_current_idx += num_encoder_points

        # compute latent
        with torch.no_grad():
            out_dict = self.get_latents(
                xyz_w=xyz_w[:, ridxs_encoder] if xyz_w is not None else None,
                rgb=rgb[:, ridxs_encoder] if rgb is not None and rgb.ndim > 1 else None,
                normal_w=normal_w[:, ridxs_encoder] if normal_w is not None and normal_w.ndim > 1 else None,
                ray_origin_direction_w=ray_origin_direction_w[:, ridxs_encoder]
                if ray_origin_direction_w is not None and ray_origin_direction_w.ndim > 1
                else None,
                alpha=point_alpha[:, ridxs_encoder] if point_alpha is not None and point_alpha.ndim > 1 else None,
            )
            fpoint_latents_coord = out_dict["latent_coord"]  # (bl, dn) or (b, num_latent, dn) or None
            fpoint_latents = out_dict["latent_tokens"]  # (bl, d) or (b, num_latent, d)
            fpoint_latents_mean = fpoint_latents

            if self.debug:
                assert fpoint_latents.isfinite().all(), (
                    f"nan: {fpoint_latents.isnan().any()}, inf: {fpoint_latents.isinf().any()}"
                )

            # sample shape_latent from q(s|y)
            if self.sample_posterior:
                # print(f'sampling posterior with std {self.std_posterior}', flush=True)
                fpoint_latents = fpoint_latents_mean + self.std_posterior * torch.randn_like(
                    fpoint_latents_mean
                )  # (b, num_latent, dim_latent)

        # get input occ_grid
        if self.mesh_optim_config["mesh_occ_grid_source"] == "data":
            # use the gt occ_grid from dataset
            assert batch["occ_grid"].ndim == 4, f"{batch['occ_grid'].shape=}"  # (b, res_z, res_y, res_x) bool
            occ_grid = batch["occ_grid"][:, None]  # (b, 1, res_z, res_y, res_x) bool
        elif self.mesh_optim_config["mesh_occ_grid_source"] == "voxel_decoder":
            # use the occ_grid estimated from voxel estimator
            assert self.voxel_decoder is not None
            with torch.no_grad():
                occ_grid = self.estimate_occ_grid(
                    latent_token=fpoint_latents,  # (b, num_latent, dim_latent)
                    return_occ_grid=True,
                )["est_occ_grid"]  # (b, 1, res_z, res_y, res_x) [0, 1]
                occ_grid = occ_grid > 0.5  # (b, 1, res_z, res_y, res_x) bool
        else:
            raise NotImplementedError(f"{self.mesh_optim_config.get('mesh_occ_grid_source')=}")

        # randomly add a few cells
        num_occ_before_expand = torch.sum(occ_grid)

        num_to_add = np.random.randint(
            self.mesh_optim_config.get("min_num_occ_to_add", 0),
            self.mesh_optim_config.get("max_num_occ_to_add", 0) + 1,
        )  # int, can be used as number of voxels to add, or dilation kernel size, depending on occ_expand_type

        if num_to_add > 0:
            input_occ_grid = obj_wdset.expand_occ_grid_for_mesh(
                occ_grid=occ_grid,  # (b, 1, res_z, res_y, res_x) bool
                num_to_add=num_to_add,
                occ_expand_type=self.mesh_optim_config.get("occ_expand_type", "dilate"),
                dtype=fpoint_latents.dtype,
                device=fpoint_latents.device,
            )  # (b, 1, res_z, res_y, res_x)
            num_occ_after_expand = torch.sum(input_occ_grid)
        else:
            input_occ_grid = occ_grid
            num_occ_after_expand = num_occ_before_expand

        del occ_grid

        # compute loss
        rgbd_dict_random = batch["rgbd_dict_random"]
        gt_hit_map = rgbd_dict_random["hit_map"]  # (b, q, h, w)
        gt_depth_map = rgbd_dict_random["depth"]  # (b, q, h, w)
        gt_normal_map = batch["normal_w"]  # (b, q, h, w, 3xyz_w)
        gt_depth_map = gt_depth_map * gt_hit_map.to(dtype=gt_depth_map.dtype)
        gt_normal_map = gt_normal_map * gt_hit_map.to(dtype=gt_depth_map.dtype).unsqueeze(-1)
        gt_xyz_c = batch["xyz_c"]  # (b, q, h, w, 3xyz_c)

        # gt_hit_map = batch["rgb_mask"].squeeze(2)  # (b, q, h, w)
        # gt_depth_map = batch["z_c"].squeeze(2)  # (b, q, h, w)
        # gt_xyz_c = batch["xyz_c"].permute(0, 1, 3, 4, 2)  # (b, q, h, w, 3xyz)
        # gt_normal_map = batch["normal_w"].permute(0, 1, 3, 4, 2)  # (b, q, h, w, 3xyz_w)
        # gt_depth_map = gt_depth_map * gt_hit_map.to(dtype=gt_depth_map.dtype)
        # gt_normal_map = gt_normal_map * gt_hit_map.to(dtype=gt_depth_map.dtype).unsqueeze(-1)

        mesh_out_dict = self.compute_mesh_loss(
            latent_token=fpoint_latents,  # (b, num_tokens, dim_tokens)
            input_occ_grid=input_occ_grid,  # (b, 1, res_z, res_y, res_x)
            camera=structures.Camera(
                H_c2w=batch["H_c2w"],  # (b, q, 4, 4)
                intrinsic=batch["intrinsic"],  # (b, q, 3, 3)
                width_px=batch["rgb_mask"].size(-1),
                height_px=batch["rgb_mask"].size(-2),
            ),
            use_masked_loss=self.mesh_optim_config.get("use_masked_loss", True),
            gt_hit_map=gt_hit_map,  # (b, q, h, w)
            gt_depth_map=gt_depth_map,
            gt_xyz_c=gt_xyz_c,
            gt_normal_map=gt_normal_map,  # (b, q, h, w, 3xyz_w)
            normal_align_mode=self.mesh_optim_config.get("normal_align_mode", "gt"),
            normalize_loss_by_valid_region_size=self.mesh_optim_config.get("normalize_loss_by_valid_region_size", True),
            # See https://github.com/microsoft/TRELLIS/blob/6b0d64751ad54d9c32d7b05fec482eb29178f56f/configs/vae/slat_vae_dec_mesh_swin8_B_64l8_fp16.json#L65-L70
            loss_weight_mask=self.mesh_optim_config.get("loss_weight_mask", 1),
            # depth loss
            loss_weight_z_c=self.mesh_optim_config.get("loss_weight_z_c", 0),
            loss_weight_z_c_huber=self.mesh_optim_config.get("loss_weight_z_c_huber", 1),
            loss_weight_z_c_l2=self.mesh_optim_config.get("loss_weight_z_c_l2", 0),
            # xyz in camera coordinates
            loss_weight_xyz_c=self.mesh_optim_config.get("loss_weight_xyz_c", 10),
            loss_weight_xyz_c_l2_sqrt=self.mesh_optim_config.get("loss_weight_xyz_c_l2_sqrt", 1),
            # face normal
            loss_weight_face_normal=self.mesh_optim_config.get("loss_weight_face_normal", 0.0),
            loss_weight_face_normal_sinusoid=self.mesh_optim_config.get("loss_weight_face_normal_sinusoid", 0),
            loss_weight_face_normal_l1=self.mesh_optim_config.get("loss_weight_face_normal_l1", 1),
            loss_weight_face_normal_ssim=self.mesh_optim_config.get("loss_weight_face_normal_ssim", 0.2),
            loss_weight_face_normal_lpips=self.mesh_optim_config.get("loss_weight_face_normal_lpips", 0.2),
            # vertex normal
            loss_weight_vertex_normal=self.mesh_optim_config.get("loss_weight_vertex_normal", 0.0),
            loss_weight_vertex_normal_sinusoid=self.mesh_optim_config.get("loss_weight_vertex_normal_sinusoid", 0),
            loss_weight_vertex_normal_l1=self.mesh_optim_config.get("loss_weight_vertex_normal_l1", 1),
            loss_weight_vertex_normal_ssim=self.mesh_optim_config.get("loss_weight_vertex_normal_ssim", 0.2),
            loss_weight_vertex_normal_lpips=self.mesh_optim_config.get("loss_weight_vertex_normal_lpips", 0.2),
            # regularization
            loss_weight_reg=self.mesh_optim_config.get("loss_weight_reg", 1),
            loss_weight_reg_sdf=self.mesh_optim_config.get("loss_weight_reg_sdf", 0.2),
        )
        loss = mesh_out_dict["loss"]  # (,)
        loss_mask = mesh_out_dict["loss_mask"]  # (,)
        loss_z_c = mesh_out_dict["loss_z_c"]  # (,)
        loss_xyz_c = mesh_out_dict["loss_xyz_c"]  # (,)
        loss_face_normal = mesh_out_dict["loss_face_normal"]  # (,)
        loss_vertex_normal = mesh_out_dict["loss_vertex_normal"]  # (,)
        loss_reg = mesh_out_dict["loss_reg"]  # (,)
        loss_reg_sdf = mesh_out_dict["loss_reg_sdf"]  # (,)
        flag_loss_nan = mesh_out_dict["flag_loss_nan"]  # (,)
        flag_loss_infinite = mesh_out_dict["flag_loss_infinite"]  # (,)
        flag_param_nan = mesh_out_dict["flag_param_nan"]  # (,)
        valid_ratio = mesh_out_dict["valid_ratio"]  # (,)

        if self.debug:
            assert loss.isfinite().all(), f"nan: {loss.isnan().any()}, inf: {loss.isinf().any()}"

        # loss dict
        info_to_log = [
            [f"{split}/num_occ_before_expand", num_occ_before_expand],
            [f"{split}/num_occ_after_expand", num_occ_after_expand],
            # losses
            [f"{split}/loss_mesh", loss],
            [f"{split}/loss_mesh_mask", loss_mask],
            [f"{split}/loss_mesh_z_c", loss_z_c],
            [f"{split}/loss_mesh_xyz_c", loss_xyz_c],
            [f"{split}/loss_mesh_face_normal", loss_face_normal],
            [f"{split}/loss_mesh_vertex_normal", loss_vertex_normal],
            [f"{split}/loss_mesh_reg", loss_reg],
            [f"{split}/loss_mesh_reg_sdf", loss_reg_sdf],
            [f"{split}/flag_loss_nan", flag_loss_nan],
            [f"{split}/flag_loss_infinite", flag_loss_infinite],
            [f"{split}/flag_param_nan", flag_param_nan],
            [f"{split}/valid_ratio", valid_ratio],
        ]

        for name, ll in info_to_log:
            if ll is not None:
                self.log(
                    name=name,
                    value=ll,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    batch_size=fpoint_latents.size(0),
                    sync_dist=True,
                )

        # visualize
        if split == "valid":
            num_to_plot = 5  # 20
            num_batch_to_plot = max(1, num_to_plot // input_occ_grid.size(0))
            if self.trainer.local_rank == 0 and batch_idx < num_batch_to_plot:
                tensorboard_logger: SummaryWriter = self.loggers[0].experiment

                for vis_name, pkey, gkey in [
                    ["hit_map", "est_hit_map", "gt_hit_map"],
                    ["depth_map", "est_depth_map", "gt_depth_map"],
                    ["vertex_normal_map", "est_vertex_normal_map", "gt_normal_map"],
                    ["face_normal_map", "est_face_normal_map", "gt_normal_map"],
                ]:
                    est = mesh_out_dict[pkey]
                    gt = mesh_out_dict[gkey]
                    if est is None or gt is None:
                        continue

                    if pkey in ["est_vertex_normal_map", "est_face_normal_map"]:
                        est = (est + 1) * 0.5
                        gt = (gt + 1) * 0.5

                    if pkey in ["est_depth_map"]:
                        est = est / 6
                        gt = gt / 6

                    if pkey in ["est_hit_map", "est_depth_map"]:
                        est = est.unsqueeze(-1).expand(-1, -1, -1, -1, 3)
                        gt = gt.unsqueeze(-1).expand(-1, -1, -1, -1, 3)

                    _b, _q, _h, _w, d = est.shape
                    for ib in range(min(_b, num_to_plot)):
                        plot_idx = batch_idx * _b + ib
                        ee = est[ib].reshape(-1, _h, _w, d)
                        gg = gt[ib].reshape(-1, _h, _w, d)
                        img = torch.cat([ee, gg], dim=-2)
                        tensorboard_logger.add_images(
                            tag=f"{split}_{vis_name}_{dataloader_idx}/{plot_idx}",
                            img_tensor=img,
                            dataformats="NHWC",
                            global_step=self.trainer.global_step,
                        )

                # mesh
                tensorboard_logger: TensorBoardLogger = self.loggers[0]
                assert isinstance(tensorboard_logger, TensorBoardLogger), f"{type(tensorboard_logger)=}"
                mesh_save_dir = pathlib.Path(tensorboard_logger.log_dir).parent / "meshes"
                mesh_save_dir.mkdir(parents=True, exist_ok=True)

                for ib in range(min(len(mesh_out_dict["raw_meshes"]), num_to_plot)):
                    plot_idx = batch_idx * len(mesh_out_dict["raw_meshes"]) + ib
                    raw_mesh: structures.RawMesh = mesh_out_dict["raw_meshes"][ib]
                    if raw_mesh is not None:
                        # tensorboard cannot display meshes with >= 65535 vertices
                        # https://github.com/tensorflow/tensorboard/issues/5329
                        # tensorboard_logger.add_mesh(
                        #     tag=f"{split}_mesh/{plot_idx}",
                        #     vertices=raw_mesh.vertex_xyz_w.float().unsqueeze(0),  # (1, num, 3xyz)
                        #     faces=raw_mesh.triangles.int().unsqueeze(0),  # (1, num_triangles, 3idx)
                        #     colors=(((raw_mesh.vertex_normal_w + 1) * 0.5).float().clamp(min=0, max=1) * 255)
                        #     .to(dtype=torch.uint8)
                        #     .unsqueeze(0),
                        #     global_step=self.trainer.global_step,
                        # )

                        # save mesh to artifact
                        filename = os.path.join(mesh_save_dir, "mesh", f"{plot_idx}_step_{self.global_step:09d}.obj")
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                        o3d.io.write_triangle_mesh(
                            filename=filename,
                            mesh=raw_mesh.get_o3d_mesh(with_vertex_normal_w=True),
                        )

        return loss
