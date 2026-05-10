#
# Copyright (C) 2026 Apple Inc. All rights reserved.
#
# The file implements the pytorch lightning module (trainer) for
# learning a generative model of shape tokens.

import contextlib
import copy
import gc
import math
import os
import pathlib
import pprint
import shutil
import tempfile
import typing as T

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    mx = None
    HAS_MLX = False

from timeit import default_timer as timer

from cleanfid import fid
import numpy as np
import PIL.Image

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter

from lito.flow import path
from lito.models import dit

try:
    from lito.models import intrinsic_predictor
except:
    intrinsic_predictor = None
from lito.models.dino import SpatialDinov2
from lito.models.ema import ModelEmaV2
from lito.odelibs import ode_solvers
from lito.script_utils import config_utils, pl_utils
from lito.trainers import base as base_trainer, lito_trainer
from plibs import ppoint, rigid_motion, utils


class LiToDiTTrainer(base_trainer.BaseTrainer):
    def __init__(
        self,
        # for pretrained model
        pretrained_model_checkpoint_url: str,
        num_latent: int,
        latent_mean: float,
        latent_std: float,
        sample_posterior: bool,
        std_posterior: float,
        # velocity decoder
        velocity_estimator_config: T.Dict[str, T.Any],
        #
        optim_config: T.Dict[str, T.Any],
        # img condition
        patch_encoder_name: str,
        num_cond_views: int = 1,
        cond_width_px: T.Union[int, T.List[int]] = 518,
        cond_height_px: T.Union[int, T.List[int]] = 518,
        # misc
        ema_decay: float = 0.999,
        eval_ema: bool = True,
        cfg_scale: float = 1.0,
        t_eps: float = 1e-4,
        debug: bool = False,
    ):
        """
        Args:
            cond_width_px / cond_height_px:
                int or list of int, resolution of the conditioning image.
                if a list is given, we randomly choose one during training every iteration
                and during inference we choose the closest resolution in the list.

            t_eps:
                a small eps to make sure we do not sample t=0 or t=1 (unstable backward)

            velocity_estimator_config:
                target:
                params:

            pretrained_model_checkpoint_url:
                str, S3 path for pretrained tokenizer.

            pretrained_model_num_latent:
                int, number of latents used for pretrained tokenizer.

            pretrained_model_fps_multiplier:
                int, -1 means that we do not use farthest point sampling (fps).
                Otherwise, first we subsample `num_latent * fps_multiplier` of points from input points
                and then use fps to sample the `num_latent` points.

            optim_config:
                num_tokenizer_points:
                    int, number of points to compute the latent. if None, use max_num_encoder_points in the tokenizer
                st_init_coord_src:
                    "sample_xyz"
                st_num_sample_points_for_occ:
                    int, 100_000

                stg_sampling_method:
                    "heun",  how to sample latent during validation
                stg_sampling_steps:
                    int, 100
                st_sampling_method:
                    "heun", how to sample point cloud from sampled latent during validation
                st_sampling_steps:
                    int, 100

                batch_size:
                    int, suggested batch size of the dataloader
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

            num_latent:
                int, this specifies the number of latents that we use for the tokenization

            patch_encoder_name:

            latent_mean / latent_std:
                mean / std of the latent tokens' elements.

        """
        super().__init__()
        self.save_hyperparameters()
        self.t_eps = t_eps
        self.num_latent = num_latent
        self.velocity_estimator_config = velocity_estimator_config

        self.optim_config = optim_config

        self.patch_encoder_name = patch_encoder_name
        self.num_cond_views = num_cond_views
        self.cond_width_px = cond_width_px
        self.cond_height_px = cond_height_px

        self.debug = debug

        self.sample_posterior = sample_posterior
        # self.latent_scale = latent_scale
        self.cfg_scale = cfg_scale
        self.use_cfg = self.cfg_scale > 1.0 and self.use_img_cond
        self.ema_decay = ema_decay
        self.eval_ema = eval_ema
        self.pretrained_model_checkpoint_url = pretrained_model_checkpoint_url
        self.std_posterior = std_posterior

        # load img encoder
        self.load_img_encoder()

        # load the pretrained shape tokenizer model (on cpu first)
        if pretrained_model_checkpoint_url is not None:
            print(f"Loading pretrained tokenizer from {pretrained_model_checkpoint_url}")
            with tempfile.TemporaryDirectory(dir=".") as tmpdir:
                from lito.eval_scripts.st_model_utils import load_model  # put here to prevent circular import

                self.pretrained_tokenizer: lito_trainer.LightTokenizationTrainer = load_model(
                    checkpoint_url=pretrained_model_checkpoint_url,
                    download_dir_root=tmpdir,
                    eval=True,
                    freeze=True,
                )["model"]
        else:
            # rely on the weight stored in the checkpoint
            import yaml

            lito_config_filename = os.path.normpath(os.path.join(__file__, "..", "..", "configs", "lito.yaml"))
            with open(lito_config_filename, "r") as f:
                config = yaml.safe_load(f)
            self.pretrained_tokenizer = lito_trainer.LightTokenizationTrainer(**config)

        self.pretrained_tokenizer.freeze()
        self.pretrained_tokenizer.eval()

        if self.optim_config.get("num_tokenizer_points", None) is None:
            self.optim_config["num_tokenizer_points"] = self.pretrained_tokenizer.max_num_encoder_points

        self.std_posterior = std_posterior if std_posterior is not None else self.pretrained_tokenizer.std_posterior

        if latent_mean is None:
            self.latent_mean = 0.0
        else:
            self.latent_mean = latent_mean

        if latent_std is None:
            self.latent_std = 1.0
        else:
            self.latent_std = latent_std

        # flow matching
        self.path = path.LinearPath()

        # update the latent shape based on pretrained shape tokenizer
        self.latent_shape = self.pretrained_tokenizer.get_latent_shape()
        assert "num_latent" not in self.latent_shape, f"{list(self.latent_shape.keys())=}"
        self.latent_shape["num_latent"] = self.num_latent
        self.dim_latent = self.latent_shape["dim_latent"]
        self.velocity_estimator_config["params"].update(self.latent_shape)

        # create dit model
        self.velocity_estimator: dit.DiffusionTransformer = config_utils.instantiate_from_config(
            self.velocity_estimator_config,
        )
        self.velocity_estimator.init_positional_embedding()

        # exponential moving average
        self.velocity_estimator_ema = ModelEmaV2(self.velocity_estimator, decay=ema_decay)

        self.validation_rgb_gt = []
        self.validation_rgb_est = []

        self.is_configure_model_called = False

        # MLX inference (lazy construction)
        self._velocity_estimator_mlx = None
        self._mlx_model_step = None
        self._mlx_use_ema = None
        self._mlx_compute_dtype = None

    # def configure_model(self):
    #     """
    #     When training sharded models with FSDP or DeepSpeed, models should not be initialized in __init__.
    #     Instead, override the configure_model() hook.
    #
    #     Note:
    #         This hook is called during each of fit/val/test/predict stages in the same process,
    #         so ensure that implementation of this hook is idempotent,
    #         i.e., after the first time the hook is called, subsequent calls to it should be a no-op.
    #     """
    #     if self.is_configure_model_called:
    #         return None
    #     self.is_configure_model_called = True
    #
    #     # load img encoder
    #     self.load_img_encoder()
    #
    #     # load the pretrained shape tokenizer model (on cpu first)
    #     with tempfile.TemporaryDirectory(dir=".") as tmpdir:
    #         self.pretrained_tokenizer: lito_trainer.LightTokenizationTrainer = load_model(
    #             checkpoint_url=self.pretrained_model_checkpoint_url,
    #             download_dir_root=tmpdir,
    #             eval=True,
    #             freeze=True,
    #         )["model"]
    #     self.pretrained_tokenizer.freeze()
    #     self.pretrained_tokenizer.eval()
    #
    #     if self.optim_config.get("num_tokenizer_points", None) is None:
    #         self.optim_config["num_tokenizer_points"] = self.pretrained_tokenizer.max_num_encoder_points
    #
    #     if self.std_posterior is None:
    #         self.std_posterior = self.pretrained_tokenizer.std_posterior
    #
    #     # flow matching
    #     self.path = path.LinearPath()
    #
    #     # update the latent shape based on pretrained shape tokenizer
    #     self.latent_shape = self.pretrained_tokenizer.get_latent_shape()
    #     assert "num_latent" not in self.latent_shape, f"{list(self.latent_shape.keys())=}"
    #     self.latent_shape["num_latent"] = self.num_latent
    #     self.dim_latent = self.latent_shape["dim_latent"]
    #     self.velocity_estimator_config["params"].update(self.latent_shape)
    #
    #     # create dit model
    #     self.velocity_estimator: dit.DiffusionTransformer = config_utils.instantiate_from_config(
    #         self.velocity_estimator_config,
    #     )
    #     self.velocity_estimator.init_positional_embedding()
    #
    #     # exponential moving average
    #     self.velocity_estimator_ema = ModelEmaV2(self.velocity_estimator, decay=self.ema_decay)

    def normalize_latent(self, latents: torch.Tensor):
        return (latents - self.latent_mean) / self.latent_std

    def unnormalize_latents(self, normalized_latentss: torch.Tensor):
        return normalized_latentss * self.latent_std + self.latent_mean

    def compute_velocity_loss(
        self,
        x: torch.Tensor,
        cond_tokens: torch.Tensor = None,
    ) -> T.Dict[str, T.Any]:
        """
        Randomly sample t, compute xt and ut_gt, estimate ut, then compute loss

        Args:
            x:
                (b, m, d)
            cond_tokens:
                (b, num_cond_tokens, dim_cond_token) or None

        Returns:
            loss:
                (,) mse loss on the velocity
            est_ut:
                (b, m, d) estimated velocity
            t:
                (b,) sampled t [0, 1]
        """

        b, m, d = x.shape
        device = x.device

        # construct decoder input and gt decoder output
        # sample t
        t_flow = torch.rand(b, device=device) * (1 - 2 * self.t_eps) + self.t_eps  # (b,)

        # sample noise (we choose to use standard gaussian)
        x0 = torch.randn_like(x)  # (b, num_flow_points, d)
        xt, ut_gt = self.path.compute_xt_ut(t=t_flow, x0=x0, x1=x)  # (b, num_flow_points, d)

        if self.debug:
            assert t_flow.isfinite().all(), f"nan: {t_flow.isnan().any()}, inf: {t_flow.isinf().any()}"
            assert xt.isfinite().all(), f"nan: {xt.isnan().any()}, inf: {xt.isinf().any()}"
            assert ut_gt.isfinite().all(), f"nan: {ut_gt.isnan().any()}, inf: {ut_gt.isinf().any()}"

        # estimate velocity
        est_ut = self.estimate_velocity(
            t=t_flow,  # (b,)
            x=xt,  # (b, num_flow_points, d)
            cond_tokens=cond_tokens,  # (b, num_cond_tokens, dim_cond_token)
            cond_use_grad_checkpointing=self.optim_config["dit_cond_use_grad_checkpointing"],
        )  # (b, m, d)

        if self.debug:
            assert est_ut.isfinite().all(), f"nan: {est_ut.isnan().any()}, inf: {est_ut.isinf().any()}"

        # compute velocity loss for xyz_w
        loss = F.mse_loss(input=est_ut, target=ut_gt, reduction="mean")  # (,)

        if self.debug:
            assert loss.isfinite().all(), f"nan: {loss.isnan().any()}, inf: {loss.isinf().any()}"

        return dict(
            loss=loss,  # (,)
            est_ut=est_ut,  # (b, m, d)
            t=t_flow,  # (b,)
        )

    def estimate_velocity(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        cond_tokens: torch.Tensor = None,
        cond_drop_ids: torch.Tensor = None,
        cond_use_grad_checkpointing: bool = False,
    ):
        """
        Estimate the flow matching velocity at t given xt=x

        Args:
            t:
                (,) or (b,)
            x:
                (b, m, d)
            cond_tokens:
                (b, num_cond_tokens, dim_cond_tokens)

        Returns:
            velocity:
                (b, m, d)
        """

        b, m, d = x.shape
        t = t.expand(b)  # (b,)

        est_ut = self.velocity_estimator(
            tokens=x,  # (b, m, dim_point)
            t=t,  # (b,)
            cond=cond_tokens,  # (b, num_cond_tokens, dim_cond_token)
            cond_drop_ids=cond_drop_ids,  # (b, )
            cond_use_grad_checkpointing=cond_use_grad_checkpointing,
            debug=self.debug,
        )  # (b, m, d)

        if self.debug:
            assert est_ut.isfinite().all(), f"nan: {est_ut.isnan().any()}, inf: {est_ut.isinf().any()}"

        return est_ut

    @torch.no_grad()
    def estimate_velocity_sampling(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        use_ema: bool,
        use_cfg: bool,
        cfg_scale: float,
        cond_tokens: torch.Tensor = None,
        cond_drop_ids: torch.Tensor = None,
    ):
        """
        Estimate the flow matching velocity at t given xt=x in inference

        Args:
            t:
                (,) or (b,)
            x:
                (b, m, d)
            cond_tokens:
                (b, num_cond_tokens, dim_cond_tokens)

        Returns:
            velocity:
                (b, m, d)
        """

        b, m, d = x.shape
        t = t.expand(b)  # (b,)

        # determine the forward function
        if use_ema:
            if use_cfg:
                model_fn = self.velocity_estimator_ema.module.forward_with_cfg
            else:
                model_fn = self.velocity_estimator_ema.module
        else:
            if use_cfg:
                model_fn = self.velocity_estimator.forward_with_cfg
            else:
                model_fn = self.velocity_estimator

        # import pdb; pdb.set_trace()

        if use_cfg:
            est_ut = model_fn(
                tokens=x,  # (b, m, dim_point)
                t=t,  # (b,)
                cond=cond_tokens,  # (b, num_cond_tokens, dim_cond_token)
                cond_drop_ids=cond_drop_ids,  # (b, )
                debug=self.debug,
                cfg_scale=cfg_scale,
            )
        else:
            est_ut = model_fn(
                tokens=x,  # (b, m, dim_point)
                t=t,  # (b,)
                cond=cond_tokens,  # (b, num_cond_tokens, dim_cond_token)
                cond_drop_ids=cond_drop_ids,  # (b, )
                debug=self.debug,
            )  # (b, m, d)

        if self.debug:
            assert est_ut.isfinite().all(), f"nan: {est_ut.isnan().any()}, inf: {est_ut.isinf().any()}"

        return est_ut

    def sampling(
        self,
        num_steps: int,
        x0: torch.Tensor,
        use_ema: bool,
        cfg_scale: float,
        cond_tokens: torch.Tensor = None,
        method: str = None,
        rtol: float = 1e-3,
        atol: float = 1e-4,
        printout: bool = False,
    ):
        """
        Sample shape token using flow matching

        Args:
            num_steps:
                number of samples (suggested for adaptive methods)
            x0:
                (b, num_tokens, d)  initial noise
            cond_tokens:
                (b, num_cond_tokens, dim_cond_token)  conditioning tokens
            method:
                see torchdiffeq, e.g, `dopri5`, `euler`.  None: use the default dopri5

        Returns:
            sampled_x:
                (b, num_tokens, d)

        """

        b, num_tokens, d = x0.shape
        device = x0.device

        use_cfg = (cfg_scale > 1.0) and (cond_tokens is not None)

        if use_cfg:
            assert cond_tokens is not None

            n = len(x0)
            x0 = torch.cat([x0, x0], 0)
            cond_tokens = torch.cat([cond_tokens, cond_tokens], 0)
            drop_ids_1 = torch.zeros(n, dtype=torch.bool, device=device)
            drop_ids_2 = torch.ones(n, dtype=torch.bool, device=device)
            cond_drop_ids = torch.cat([drop_ids_1, drop_ids_2], 0)  # drop cond_tokens for the second half
        else:
            cond_drop_ids = None

        # construct the velocity function
        func = lambda t, x: self.estimate_velocity_sampling(
            t=t,
            x=x,
            use_ema=use_ema,
            use_cfg=use_cfg,
            cfg_scale=cfg_scale,
            cond_tokens=cond_tokens,
            cond_drop_ids=cond_drop_ids,
        )

        # construct the ts
        if method == "euler":
            ts = torch.linspace(self.t_eps, 1, num_steps, device=device)
        elif method == "heun":
            ts = torch.linspace(self.t_eps, 1, num_steps, device=device)
        elif method.startswith("heun_"):
            # heun_alpha
            a = float(method.split("heun_", 1)[1])

            # construct nonuniform ts (see https://arxiv.org/pdf/2206.00364 eq5)
            s_max, _ = self.path.compute_sigma_t(t=0)
            s_min, _ = self.path.compute_sigma_t(t=1)
            N = num_steps
            stds = [(s_max ** (1 / a) + i / (N - 1) * (s_min ** (1 / a) - s_max ** (1 / a))) ** a for i in range(N)]
            stds = torch.tensor(stds, dtype=x0.dtype, device=device)
            ts = self.path.compute_t(sigma_t=stds)
        else:
            ts = torch.linspace(self.t_eps, 1, num_steps, device=device)

        sampled_x = ode_solvers.odeint(
            func=func,
            x0=x0,
            ts=ts,
            method=method,
            rtol=rtol,
            atol=atol,
            printout=printout,
        )  # (b, num_points, d)

        if use_cfg:
            sampled_x, _ = sampled_x.chunk(2, dim=0)

        return sampled_x

    def load_img_encoder(self):
        # image encoder
        if self.patch_encoder_name == "dinov2_vitl14_reg_rgb":
            # dim_token: 1024(dino) + 1024(linear)

            # use frozen dino
            self.patch_encoder = SpatialDinov2(
                model_type="dinov2_vitl14_reg",
                dino_layer_idxs=[-1],
                dino_normalize_tokens=False,
                dino_normalize_concat_tokens=True,
                dino_use_cls=True,
                dino_use_registers=True,
                learnable_model_type="linear",
                learnable_model_params=dict(
                    out_channels=1024,
                    input_types=["rgb"],
                    add_layer_norm=False,
                ),
                learnable_model_first_transforms_rgb=True,
                learnable_add_joint_layernorm=False,
                width_px=self.cond_width_px,
                height_px=self.cond_height_px,
            )
        elif self.patch_encoder_name == "dinov2_vitl14_reg_rgba_nonorm":
            # dim_token: 1024(dino) + 1024(linear)

            # use frozen dino
            self.patch_encoder = SpatialDinov2(
                model_type="dinov2_vitl14_reg",
                dino_layer_idxs=[-1],
                dino_normalize_tokens=False,
                dino_normalize_concat_tokens=False,
                dino_use_cls=True,
                dino_use_registers=True,
                learnable_model_type="linear",
                learnable_model_params=dict(
                    out_channels=1024,
                    input_types=["rgb", "alpha"],
                    add_layer_norm=False,
                ),
                learnable_model_first_transforms_rgb=True,
                learnable_add_joint_layernorm=False,
                width_px=self.cond_width_px,
                height_px=self.cond_height_px,
            )
        elif self.patch_encoder_name == "dinov2_vitl14_reg_rgba":
            # dim_token: 1024(dino) + 1024(linear)

            # use frozen dino
            self.patch_encoder = SpatialDinov2(
                model_type="dinov2_vitl14_reg",
                dino_layer_idxs=[-1],
                dino_normalize_tokens=False,
                dino_normalize_concat_tokens=True,
                dino_use_cls=True,
                dino_use_registers=True,
                learnable_model_type="linear",
                learnable_model_params=dict(
                    out_channels=1024,
                    input_types=["rgb", "alpha"],
                    add_layer_norm=False,
                ),
                learnable_model_first_transforms_rgb=True,
                learnable_add_joint_layernorm=False,
                width_px=self.cond_width_px,
                height_px=self.cond_height_px,
            )
        else:
            raise NotImplementedError(self.patch_encoder_name)

    @torch.no_grad()
    @torch.compile
    def inference_sample_latent(
        self,
        cond_rgba: torch.Tensor,  # (b, q, h, w, 4rgba) [0, 1] rgb is straight
        ode_sampling_method: str = "heun",
        ode_num_steps: int = 20,
        cfg_scale: float = 3.0,
        use_ema: bool = True,
    ):
        """
        Sampling a latent given input conditioning.

        Args:
            cond_rgba:
                (b, q, h, w, 4rgba) [0, 1], rgb is straight.
            ode_sampling_method:
                str, "heun", "euler"
            ode_num_steps:
                int, eg, 20
            cfg_scale:
                float, eg, 3
            use_ema:
                bool, if True, we use the EMA model

        Returns:
            unnormalized_latent:
                (b, nl, dl)  sampled latent, already unnormalized
        """
        assert cond_rgba.ndim == 5
        b, q, h, w, _4rgb = cond_rgba.shape
        assert _4rgb == 4
        cond_rgba = cond_rgba.to(device=self.device)

        # compute conditioning tokens
        cond_tokens = self.get_image_conditioning(
            straight_rgb=cond_rgba[..., :3],  # (b, q, h, w, 3rgb) [0, 1]
            alpha=cond_rgba[..., 3:4],  # (b, q, h, w, 1) [0, 1]
        )  # (b, num_cond_tokens, d)

        # sample shape latent
        sampled_x = self.sampling(
            use_ema=use_ema,
            cfg_scale=cfg_scale,
            num_steps=ode_num_steps,
            x0=torch.randn(b, self.num_latent, self.dim_latent, device=self.device),  # (b, num_latent, dim_latent)
            cond_tokens=cond_tokens,  # (b, num_cond_tokens, dim_cond_token)
            method=ode_sampling_method,
        )  # (b, num_latent, dim_latent)

        # map back to the original scale
        sampled_x = self.unnormalize_latents(sampled_x)  # (b, nl, dl)

        return dict(
            unnormalized_latent=sampled_x,  # (b, num_cond_tokens, d)
        )

    def _was_trained_with_cond_dropout(self) -> bool:
        """Whether the velocity estimator was trained with random conditioning dropout.

        Used to decide if classifier-free guidance is meaningful at inference. If
        no dropout was applied during training, the model never saw "unconditional"
        inputs, so the CFG interpolation amplifies noise rather than guidance.

        Returns:
            True if the model has a usable unconditional branch.
        """
        cond_embedder = getattr(self.velocity_estimator, "cond_embedder", None)
        return cond_embedder is not None and float(cond_embedder.cond_drop_prob) > 0.0

    def _get_or_build_mlx_model(
        self,
        use_ema: bool = True,
        mlx_compute_dtype: T.Optional[str] = None,
    ):
        """Lazily construct or refresh the MLX velocity estimator.

        Rebuilds when ``use_ema``, ``mlx_compute_dtype``, or the training step
        changes.

        Args:
            use_ema: If True, use EMA weights; otherwise use the live model weights.
            mlx_compute_dtype: Compute dtype string (``"bfloat16"``, ``"float16"``,
                ``"float32"``, or ``None`` for f32).  When set to a reduced-precision
                dtype, model weights are cast accordingly so that MLX linear ops
                run in that precision — analogous to ``torch.autocast``.

        Returns:
            MLX DiffusionTransformer with the requested weights and dtype.
        """
        assert HAS_MLX, "MLX is not installed. Install with: pip install mlx"
        current_step = getattr(self, "global_step", 0)
        need_rebuild = (
            self._velocity_estimator_mlx is None
            or self._mlx_model_step != current_step
            or self._mlx_use_ema != use_ema
            or self._mlx_compute_dtype != mlx_compute_dtype
        )
        if need_rebuild:
            from lito.mlx.convert import build_mlx_model

            torch_source = self.velocity_estimator_ema.module if use_ema else self.velocity_estimator
            mlx_model = build_mlx_model(torch_source)

            # Cast params to compute dtype.
            _dtype_map = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}
            mlx_fwd_dtype = _dtype_map.get(mlx_compute_dtype) if mlx_compute_dtype else None
            if mlx_fwd_dtype is not None and mlx_fwd_dtype != mx.float32:
                import mlx.utils as mlx_utils

                def _cast(x):
                    if isinstance(x, mx.array):
                        return x.astype(mlx_fwd_dtype)
                    return x

                mlx_model.update(mlx_utils.tree_map(_cast, mlx_model.parameters()))
                mx.eval(mlx_model.parameters())

            self._velocity_estimator_mlx = mlx_model
            self._mlx_model_step = current_step
            self._mlx_use_ema = use_ema
            self._mlx_compute_dtype = mlx_compute_dtype

        return self._velocity_estimator_mlx

    @torch.no_grad()
    def inference_sample_latent_mlx(
        self,
        cond_rgba: torch.Tensor,  # (b, q, h, w, 4rgba) [0, 1] rgb is straight
        ode_sampling_method: str = "heun",
        ode_num_steps: int = 20,
        cfg_scale: float = 3.0,
        use_ema: bool = True,
        mlx_compute_dtype: T.Optional[str] = "bfloat16",
    ):
        """Sample a latent using MLX on Apple Silicon.

        Same interface as ``inference_sample_latent``.  Image conditioning
        is computed in PyTorch (runs once); the ODE sampling loop runs
        entirely in MLX.

        Args:
            cond_rgba: Conditioning RGBA image. (b, q, h, w, 4) [0, 1], straight RGB.
            ode_sampling_method: ODE solver method ("euler", "heun", "heun_<alpha>").
            ode_num_steps: Number of ODE steps.
            cfg_scale: Classifier-free guidance scale.
            use_ema: Whether to use EMA model weights.
            mlx_compute_dtype: Compute dtype for the MLX forward pass.
                Use ``"bfloat16"`` (default) to match CUDA ``torch.autocast(bf16)``
                behaviour, ``"float16"`` for half-precision, or ``None`` / ``"float32"``
                for full precision.

        Returns:
            Dict with ``unnormalized_latent``: (b, nl, dl) sampled latent.
        """
        assert HAS_MLX, "MLX is not installed. Install with: pip install mlx"
        from lito.mlx.flow.path import LinearPath as MLXLinearPath
        from lito.mlx.odelibs import ode_solvers as mlx_ode_solvers

        assert cond_rgba.ndim == 5
        b, q, h, w, _4rgb = cond_rgba.shape
        assert _4rgb == 4
        cond_rgba = cond_rgba.to(device=self.device)

        # ---- Step 1: Image conditioning in PyTorch (runs once) ----
        print(
            f"running dino ({next(self.patch_encoder.parameters()).device}, {next(self.patch_encoder.parameters()).dtype})",
            flush=True,
        )
        stime = timer()
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
            cond_tokens = self.get_image_conditioning(
                straight_rgb=cond_rgba[..., :3],  # (b, q, h, w, 3rgb) [0, 1]
                alpha=cond_rgba[..., 3:4],  # (b, q, h, w, 1) [0, 1]
            )  # (b, num_cond_tokens, d)
        print(f"finished running dino, took {timer() - stime} secs", flush=True)

        # ---- Step 2: Convert to MLX ----
        cond_tokens_mx = mx.array(cond_tokens.detach().cpu().float().numpy())  # (b, num_cond_tokens, d)

        # ---- Step 2b: CFG validity check ----
        # If the model wasn't trained with random conditioning dropout, CFG produces nonsense
        # (the "unconditional" branch was never trained). Silently fall back to cfg_scale=1.0.
        if cfg_scale > 1.0 and not self._was_trained_with_cond_dropout():
            print(
                f"[inference_sample_latent_mlx] cfg_scale={cfg_scale} requested but model was not "
                f"trained with conditioning dropout — falling back to cfg_scale=1.0",
                flush=True,
            )
            cfg_scale = 1.0

        # ---- Step 3: Build / fetch MLX model (with dtype casting) ----
        use_cfg = cfg_scale > 1.0
        mlx_model = self._get_or_build_mlx_model(
            use_ema=use_ema,
            mlx_compute_dtype=mlx_compute_dtype,
        )

        # Resolve the compute dtype for input casting
        _dtype_map = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}
        mlx_fwd_dtype = _dtype_map.get(mlx_compute_dtype) if mlx_compute_dtype else None

        # ---- Step 4: Generate initial noise in MLX ----
        x0 = mx.random.normal(shape=(b, self.num_latent, self.dim_latent))  # (b, nl, dl)

        # ---- Step 5: Setup CFG ----
        # Cast conditioning tokens to match compute dtype
        if mlx_fwd_dtype is not None and mlx_fwd_dtype != mx.float32:
            cond_tokens_mx = cond_tokens_mx.astype(mlx_fwd_dtype)

        if use_cfg:
            n = b
            x0 = mx.concatenate([x0, x0], axis=0)  # (2b, nl, dl)
            cond_tokens_mx = mx.concatenate([cond_tokens_mx, cond_tokens_mx], axis=0)  # (2b, m, d)
            drop_ids_1 = mx.zeros((n,), dtype=mx.bool_)
            drop_ids_2 = mx.ones((n,), dtype=mx.bool_)
            cond_drop_ids = mx.concatenate([drop_ids_1, drop_ids_2], axis=0)  # (2b,)
        else:
            cond_drop_ids = None

        # ---- Step 6: Construct velocity function ----
        def _cast_in(arr):
            if mlx_fwd_dtype is not None and mlx_fwd_dtype != mx.float32:
                return arr.astype(mlx_fwd_dtype)
            return arr

        if use_cfg:

            def velocity_fn(t_val, x_val):
                return mlx_model.forward_with_cfg(
                    tokens=_cast_in(x_val),
                    t=_cast_in(t_val),
                    cond=cond_tokens_mx,
                    cfg_scale=cfg_scale,
                    cond_drop_ids=cond_drop_ids,
                ).astype(mx.float32)
        else:

            def velocity_fn(t_val, x_val):
                return mlx_model(
                    tokens=_cast_in(x_val),
                    t=_cast_in(t_val),
                    cond=cond_tokens_mx,
                    cond_drop_ids=cond_drop_ids,
                ).astype(mx.float32)

        # ---- Step 7: Construct timesteps in MLX ----
        if ode_sampling_method == "euler":
            ts = mx.linspace(self.t_eps, 1.0, ode_num_steps)
        elif ode_sampling_method == "heun":
            ts = mx.linspace(self.t_eps, 1.0, ode_num_steps)
        elif ode_sampling_method.startswith("heun_"):
            a = float(ode_sampling_method.split("heun_", 1)[1])
            mlx_path = MLXLinearPath()
            s_max, _ = mlx_path.compute_sigma_t(mx.array(0.0))
            s_min, _ = mlx_path.compute_sigma_t(mx.array(1.0))
            s_max_val = float(s_max)
            s_min_val = float(s_min)
            N = ode_num_steps
            stds = mx.array(
                [
                    (s_max_val ** (1 / a) + i / (N - 1) * (s_min_val ** (1 / a) - s_max_val ** (1 / a))) ** a
                    for i in range(N)
                ]
            )
            ts = mlx_path.compute_t(stds)
        else:
            ts = mx.linspace(self.t_eps, 1.0, ode_num_steps)

        # ---- Step 8: ODE sampling in MLX ----
        print(f"running odeint", flush=True)
        stime = timer()
        sampled_x = mlx_ode_solvers.odeint(
            func=velocity_fn,
            x0=x0,
            ts=ts,
            method=ode_sampling_method,
        )  # (2b or b, nl, dl)

        if use_cfg:
            mid = sampled_x.shape[0] // 2
            sampled_x = sampled_x[:mid]  # (b, nl, dl)

        print(f"finished odeint, took {timer() - stime} secs", flush=True)

        # ---- Step 9: Convert back to PyTorch ----
        sampled_x_torch = torch.from_numpy(np.array(sampled_x)).to(
            device=self.device, dtype=torch.float32
        )  # (b, nl, dl)

        # ---- Step 10: Unnormalize ----
        sampled_x_torch = self.unnormalize_latents(sampled_x_torch)  # (b, nl, dl)

        return dict(
            unnormalized_latent=sampled_x_torch,  # (b, nl, dl)
        )

    def on_train_epoch_start(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # reset dataset
        # print(f'(rank {self.global_rank}) reset dataset', flush=True)
        try:
            dset = self.trainer.train_dataloader.dataset
            dset.reset()
        except Exception:
            pass

    # def on_after_backward(self):
    #     print(f'finding unused parameters..', flush=True)
    #     unused_params = []
    #     for name, param in self.named_parameters():
    #         if param.grad is None and param.requires_grad:
    #             unused_params.append(name)
    #     if unused_params:
    #         print(f"Unused parameters after backward pass: {unused_params}")

    def on_train_batch_end(
        self,
        outputs: T.Union[torch.Tensor, T.Mapping[str, T.Any], None],
        batch: T.Any,
        batch_idx: int,
    ):
        # get batch size
        if isinstance(batch, dict):
            batch_size = batch["point_xyz_w"].shape[0]
        elif isinstance(batch, (list, tuple)):
            batch_size = len(batch)
        else:
            raise NotImplementedError

        # log learning rate
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

    def compute_img_conditioning(
        self,
        img: torch.Tensor,  # (b, q, 3rgb, h, w)
        H_c2w: torch.Tensor,  # (b, q, 4, 4)
        intrinsic: torch.Tensor,  # (b, q, 3, 3)
    ) -> torch.Tensor:
        """
        Compute the image conditioning tokens

        Args:
            img:
                (b, q, 3rgb, h, w)
            H_c2w:
                (b, q, 4, 4)
            intrinsic:
                (b, q, 3, 3)

        Returns:
            cond_tokens:
                (b, num_cond_tokens, dim_cond_token)
        """

        cond_tokens_info = self.extract_raypatch_tokens_from_img(
            img=img,  # (b, q, 3rgb, h, w)
            H_c2w=H_c2w,  # (b, q, 4, 4)
            intrinsic=intrinsic,  # (b, q, 3, 3)
            min_num_views=-1,
            max_num_tokens=-1,
        )
        cond_tokens = cond_tokens_info["patch_feature"]  # (b, num_cond_tokens, dim_cond_token)

        if self.debug:
            assert cond_tokens.isfinite().all(), f"nan: {cond_tokens.isnan().any()}, inf: {cond_tokens.isinf().any()}"

        return cond_tokens

    def get_tokenizer_unnormalized_latent(
        self,
        xyz_w: torch.Tensor,  # (b, n, 3)
        rgb: torch.Tensor,  # (b, n, 3) [0, 1]
        normal_w: T.Optional[torch.Tensor] = None,  # (b, n, 3)
        ray_origin_direction_w: T.Optional[torch.Tensor] = None,  # (b, n, 6_origin_w_dir_w)
        alpha: T.Optional[torch.Tensor] = None,  # (b, n, 1)  [0, 1]
        num_points: int = None,
    ) -> torch.Tensor:
        """
        Compute tokenizer latent given the points. Returns unnormalized latents.

        Args:
            xyz_w:
                (b, n, 3) the point xyz in the n-coordinate
            rgb:
                (b, n, 3) the point rgb [0, 1]
            normal_w:
                (b, n, 3) the point normal in the n-coordinate
            alpha:
                (b, n, 1) [0, 1]

            num_points:
                int, number of points to use to compute the latent

        Returns:
            latent:
                (b, nl, dl), not normalized yet
        """

        # selecting random points for encoder, flow
        batch_size, m, d = xyz_w.shape
        device = xyz_w.device

        if num_points is None:
            num_points = self.pretrained_tokenizer.max_num_encoder_points

        # randomly select encoder and decoder points
        ridxs_encoder = utils.get_subsample_idx(
            n=m,
            num_samples=num_points,
            repeat_if_not_enough=True,
            device=device,
        )  # (num_points,)

        # compute latent
        out_dict = self.pretrained_tokenizer.get_latents(
            xyz_w=xyz_w[:, ridxs_encoder] if xyz_w is not None else None,
            rgb=rgb[:, ridxs_encoder] if rgb is not None else None,
            normal_w=normal_w[:, ridxs_encoder] if normal_w is not None and normal_w.ndim > 1 else None,
            ray_origin_direction_w=ray_origin_direction_w[:, ridxs_encoder]
            if ray_origin_direction_w is not None and ray_origin_direction_w.ndim > 1
            else None,
            alpha=alpha[:, ridxs_encoder] if alpha is not None and alpha.ndim > 1 else None,
            num_latent=self.num_latent,
        )
        latents = out_dict["latent_tokens"]  # (b, num_latent, d)
        assert latents.size(-2) == self.num_latent
        latents_mean = latents

        if self.debug:
            assert latents.isfinite().all(), f"nan: {latents.isnan().any()}, inf: {latents.isinf().any()}"

        # sample shape_latent from q(s|y)
        if self.sample_posterior:
            latents = latents_mean + self.std_posterior * torch.randn_like(latents_mean)  # (b, num_latent, dim_latent)

        return latents  # (b, nl, dl)

    def get_image_conditioning(
        self,
        straight_rgb: torch.Tensor,  # (b, q, h, w, 3rgb) [0, 1]
        alpha: torch.Tensor,  # (b, q, h, w, 1) [0, 1]
    ):
        """
        Compute image conditioning tokens.

        Args:
            straight_rgb:
                (b, q, h, w, 3rgb) [0, 1], before multiplied with alpha to remove background
            alpha:
                (b, q, h, w, 1) [0, 1], alpha map that can be multiplied to remove background

        Returns:
            cond_tokens:
                (b, num_cond_tokens, dim_cond_token)
        """

        b, q, h, w, _3rgb = straight_rgb.shape

        # no need to wrap with torch.no_grad(). we handle it inside
        # with torch.autocast(device_type=straight_rgb.device.type, dtype=torch.bfloat16, enabled=True):
        # assert not self.patch_encoder.dinov2_model.training
        # for name, param in self.patch_encoder.dinov2_model.named_parameters():
        #     assert not param.requires_grad, f"{name} requires grad"

        out_dict = self.patch_encoder(
            premultiplied_rgb=(straight_rgb * alpha).permute(0, 1, 4, 2, 3),  # (b, q, 3rgb, h, w) [0, 1]
            xyz_w=None,
            plucker=None,
            alpha=alpha.permute(0, 1, 4, 2, 3),  # (b, q, 1, h, w)  [0, 1]
            use_grad_checkpointing=self.optim_config["patch_encoder_use_grad_checkpointing"],
        )
        cond_feature = out_dict["out_tokens"]  # (b, q, num_extra + phpw, d)
        _b, _q, _ntoken_per_view, d = cond_feature.shape
        cond_feature = cond_feature.reshape(b, q * _ntoken_per_view, d)  # (b, num_tokens, d)

        return cond_feature  # (b, num_tokens, d)

    def _collate_batch(self, batch: T.Dict[str, T.Any]):
        # on the fly rendering the raw mesh to get rgbd images
        if isinstance(batch, dict):
            pass
        elif isinstance(batch, list):
            if batch[0].get("dset_type", None) in [
                "img_cond_ldm",
                "lito_img_cond_ldm",
                "lito_img_cond_ldm_with_rgbd_cond_or_sphere",
            ]:
                batch = default_collate(batch)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError(batch["dset_type"])
        return batch

    def _step(
        self,
        batch: T.Dict[str, T.Any],
        batch_idx: int,
    ):
        """
        Args:
            batch:
                a dict, containing:

                point_xyz_w:
                    (b, n, 3) the point xyz in the n-coordinate
                point_rgb:
                    (b, n, 3) or -1, the point rgb [0, 1]
                point_normal_w:
                    (b, n, 3) or -1, the point normal in the n-coordinate
                point_alpha:
                    (b, n, 1) or -1, [0, 1]

                rgbd_dict_cond:
                    H_c2w:
                        (b, q, 4, 4)
                    intrinsic:
                        (b, q, 3, 3)
                    rgb:
                        (b, q, h, w, 3rgb) [0, 1], straight
                    alpha:
                        (b, q, h, w, 1) [0, 1]

            batch_idx:

        Returns:

        """
        # get tokenizer latent
        unnormalized_latent = self.get_tokenizer_unnormalized_latent(
            xyz_w=batch["point_xyz_w"],  # (b, n, 3xyz_w)
            rgb=batch["point_rgb"],  # (b, n, 3rgb) [0, 1]
            normal_w=batch.get("point_normal_w", None),
            ray_origin_direction_w=batch.get("point_ray_origin_direction_w", None),  # (b, n, 6_origin_w_dir_w)
            alpha=batch.get("point_alpha", None),  # (b, n, 1)  [0, 1]
            num_points=self.optim_config["num_tokenizer_points"],
        )  # (b, nl, dl)  unnormalized
        assert self.num_latent == unnormalized_latent.size(1)

        # get image conditioning
        cond_tokens = self.get_image_conditioning(
            straight_rgb=batch["rgbd_dict_cond"]["rgb"][:, : self.num_cond_views],  # (b, q, h, w, 3rgb) [0, 1]
            alpha=batch["rgbd_dict_cond"]["alpha"][:, : self.num_cond_views],  # (b, q, h, w, 1) [0, 1]
        )  # (b, num_cond_tokens, d)

        # compute velocity loss
        out_dict = self.compute_velocity_loss(
            x=self.normalize_latent(latents=unnormalized_latent),  # (b, nl, dl) normalized
            cond_tokens=cond_tokens,  # (b, num_cond_tokens, d)
        )
        loss = out_dict["loss"]  # (,)

        if self.debug:
            assert loss.isfinite().all(), f"nan: {loss.isnan().any()}, inf: {loss.isinf().any()}"

        return dict(
            loss=loss,  # (,)
            # unnormalized_latent=unnormalized_latent,  # (b, nl, dl)  unnormalized
            cond_tokens=cond_tokens,  # (b, num_cond_tokens, d)
        )

    def training_step(
        self,
        batch: T.Dict[str, T.Any],
        batch_idx: int,
    ):
        # run collate function
        batch = self._collate_batch(batch)

        out_dict = self._step(
            batch=batch,
            batch_idx=batch_idx,
        )
        loss = out_dict["loss"]  # (,)

        self.log(
            name=self.optim_config["monitor_loss_name"],
            value=loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=out_dict["cond_tokens"].shape[0],
            sync_dist=True,
        )
        return loss

    @torch.no_grad()
    def validation_step(
        self,
        batch: T.Dict[str, T.Any],
        batch_idx: int,
    ):
        # print(f"({self.global_rank}) validation_step:  self.device: {self.device}")
        # for name, param in self.named_parameters():
        #     if param is not None and isinstance(param, torch.Tensor):
        #         if param.device != self.device:
        #             print(f"({self.global_rank}) name: {name}, device: {param.device}")

        # run collate function
        batch = self._collate_batch(batch)

        out_dict = self._step(
            batch=batch,
            batch_idx=batch_idx,
        )
        loss = out_dict["loss"]  # (,)
        cond_tokens = out_dict["cond_tokens"]  # (b, num_cond_tokens, d)
        b = cond_tokens.size(0)

        self.log(
            name=f"valid/{self.optim_config['monitor_loss_name']}",
            value=loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=b,
            sync_dist=True,
        )

        # sample shape latent
        sampled_x = self.sampling(
            use_ema=self.eval_ema,
            cfg_scale=self.cfg_scale,
            num_steps=self.optim_config["stg_sampling_steps"],
            x0=torch.randn(b, self.num_latent, self.dim_latent, device=self.device),  # (b, num_latent, dim_latent)
            cond_tokens=cond_tokens,  # (b, num_cond_tokens, dim_cond_token)
            method=self.optim_config["stg_sampling_method"],
        )  # (b, num_latent, dim_latent)

        # map back to the original scale
        sampled_x = self.unnormalize_latents(sampled_x)  # (b, nl, dl)

        # get gaussian
        gs_dicts = self.pretrained_tokenizer.inference_estimate_gaussians(
            fpoint_latent=sampled_x,  # (b, nl, dl)
            init_coord_src=self.optim_config["st_init_coord_src"],  # "sample_xyz"
            init_coord=None,
            latent_coord=None,
            num_points_for_sample_xyz=self.optim_config["st_num_sample_points_for_occ"],
            method_for_sample_xyz=self.optim_config["st_sampling_method"],
            steps_for_sample_xyz=self.optim_config["st_sampling_steps"],
        )

        # render
        rgbd_dict_cond = batch["rgbd_dict_cond"]
        render_dict = self.pretrained_tokenizer.render_gaussians(
            gs_dicts=gs_dicts,
            H_c2w=rgbd_dict_cond["H_c2w"],  # (b, q, 4, 4)
            intrinsic=rgbd_dict_cond["intrinsic"],  # (b, q, 3, 3)
            width_px=rgbd_dict_cond["rgb"].size(-2),
            height_px=rgbd_dict_cond["rgb"].size(-3),
        )
        est_rgb = render_dict["premultiplied_rgb"]  # (b, q, h, w, 3rgb) [0, 1]  premultiplied, black bg
        rgb_gt = rgbd_dict_cond["rgb"] * rgbd_dict_cond["alpha"]  # (b, q, h, w, 3rgb) [0, 1]  premultiplied, black bg

        rgb_gt_all_gather = self.gather_images_from_all_global_ranks(
            images=rgb_gt,
            new_width_px=self.optim_config["fid_width_px"],
            new_height_px=self.optim_config["fid_height_px"],
        )  # (b * world_size, q, h, w, 3rgb) uint8, premultiplied, black bg
        rgb_est_all_gather = self.gather_images_from_all_global_ranks(
            images=est_rgb,
            new_width_px=self.optim_config["fid_width_px"],
            new_height_px=self.optim_config["fid_height_px"],
        )  # (b * world_size, q, h, w, 3rgb) uint8, premultiplied, black bg

        if self.global_rank == 0:
            assert rgb_gt_all_gather is not None
            assert rgb_est_all_gather is not None
            self.validation_rgb_gt.append(rgb_gt_all_gather.detach().cpu())  # (b * world_size, q, h, w, 3), uint8
            self.validation_rgb_est.append(rgb_est_all_gather.detach().cpu())  # (b * world_size, q, h, w, 3), uint8

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

    @torch.no_grad()
    def on_validation_epoch_end(self):
        # print(f"({self.global_rank}) on_validation_epoch_end:  self.device: {self.device}")
        # for name, param in self.named_parameters():
        #     if param is not None and isinstance(param, torch.Tensor):
        #         if param.device != self.device:
        #             print(f"({self.global_rank}) name: {name}, device: {param.device}")

        if self.global_rank == 0:
            rgb_gt = torch.cat(self.validation_rgb_gt, dim=0)  # [(b, q, h, w, 3)] -> (n x b, q, h, w, 3) uint8
            rgb_est = torch.cat(self.validation_rgb_est, dim=0)  # (n x b, q, h, w, 3) uint8

            assert rgb_gt.shape == rgb_est.shape, f"{rgb_gt.shape=}, {rgb_est.shape=}"
            b, q, h, w, _3 = rgb_gt.shape

            # compute fid
            mdict = self.compute_gen_metrics_given_imgs(
                rgb_gt=rgb_gt.reshape(-1, h, w, 3),  # (bq, h, w, 3) uint8
                rgb_est=rgb_est.reshape(-1, h, w, 3),  # (bq, h, w, 3) uint8
            )

            # log
            for tmp_name, tmp_val in [
                ["fid", mdict["fid"]],
                ["kid", mdict["kid"]],
            ]:
                self.log(
                    f"valid/{tmp_name}",
                    tmp_val,
                    on_epoch=True,
                    on_step=False,
                    logger=True,
                    prog_bar=True,
                    rank_zero_only=True,
                    batch_size=1,
                    sync_dist=False,
                )

            num_samples_to_plot = 8
            num_views_to_plot = 4
            rgb_gt_vis = rgb_gt[:num_samples_to_plot, :num_views_to_plot]  # (b', q', h, w, 3) uint8
            rgb_est_vis = rgb_est[:num_samples_to_plot, :num_views_to_plot]  # (b', q', h, w, 3) uint8

            try:
                tensorboard_logger: SummaryWriter = self.loggers[0].experiment

                _b, _q, _h, _w, _d = rgb_est_vis.shape

                img_vis = torch.cat(
                    [
                        rgb_est_vis.reshape(_b * _q, _h, _w, _d),
                        rgb_gt_vis.reshape(_b * _q, _h, _w, _d),
                    ],
                    dim=-2,
                )  # (b' * q', h, w*2, d), uint8

                tensorboard_logger.add_images(
                    tag=f"validation/gs_render_{self.global_rank}",
                    img_tensor=img_vis,
                    dataformats="NHWC",
                    global_step=self.trainer.global_step,
                )

            except Exception:
                pass

        # clean up data for the current validation
        self.validation_rgb_gt = []
        self.validation_rgb_est = []

    @torch.no_grad()
    def gather_images_from_all_global_ranks(
        self,
        images: torch.Tensor,  # (b, q, h, w, 3rgb) [0, 1] or (b, h, w, 3rgb) [0, 1]
        new_width_px: int,
        new_height_px: int,
    ):
        """
        Gatheer images from all ranks
        Args:
            images:
                (b, q, h, w, 3rgb), same shape across all ranks
            new_width_px:
                int, the resolution we will resize the image to
            new_height_px:
                int, the resolution we will resize the image to
        Returns:
             (only on rank 0): Tensor of shape
                (b * world_size, q, h, w, 3rgb)  uint8
                (b * world_size, h, w, 3rgb)  uint8
        """

        *b_shape, h, w, _3rgb = images.shape
        b = math.prod(b_shape)

        if h != new_height_px or w != new_width_px:
            images = torch.nn.functional.interpolate(
                input=images.reshape(b, h, w, 3).permute(0, 3, 1, 2),  # (b, 3, h, w) [0, 1]
                size=(new_height_px, new_width_px),
                mode="bilinear",
                align_corners=False,
            )
            assert images.shape == (b, 3, new_height_px, new_width_px)
            images = images.permute(0, 2, 3, 1).reshape(
                *b_shape, new_height_px, new_width_px, 3
            )  # (b, q, h, w, 3) [0, 1]

        # convert to uint8 to save memory
        local_uint8 = (images.clamp(min=0, max=1) * 255).to(torch.uint8)  # (b, q, h, w, 3rgb) uint8

        local_uint8: torch.Tensor = local_uint8.to(
            device=self.device,
            non_blocking=True,
        ).contiguous()  # (b, q, h, w, 3rgb) uint8

        if not dist.is_initialized():
            return local_uint8  # (b, q, h, w, 3rgb) uint8

        # get world size
        world_size = dist.get_world_size()

        # Prepare receive buffers
        gather_list = (
            [torch.empty_like(local_uint8) for _ in range(world_size)] if self.global_rank == 0 else None
        )  # (world_size,) list.  each is (b, q, h, w, 3rgb)

        # gather from
        dist.gather(local_uint8, gather_list=gather_list, dst=0)

        if self.global_rank == 0:
            return torch.cat(gather_list, dim=0)  # (b * world_size, q, h, w, 3rgb) uint8
        else:
            return None

    def compute_gen_metrics_given_imgs(
        self,
        *,
        rgb_gt: torch.Tensor,  # (b, h, w, 3) uint8, premultiplied, black bg
        rgb_est: torch.Tensor,  # (b, h, w, 3) uint8, premultiplied, black bg
    ):
        assert self.global_rank == 0, f"{self.global_rank=}"

        assert (rgb_gt.ndim == 4) and (rgb_gt.shape[3] == 3), f"{rgb_gt.shape=}"
        b, h, w, _3 = rgb_gt.shape
        assert rgb_est.shape[1:] == (h, w, 3), f"{rgb_est.shape=}"

        assert rgb_gt.dtype == torch.uint8, f"{rgb_gt.dtype=}"
        assert rgb_est.dtype == torch.uint8, f"{rgb_est.dtype=}"
        rgb_gt_np = rgb_gt.cpu().numpy()
        rgb_est_np = rgb_est.cpu().numpy()

        repo_root = pathlib.Path(__file__).parent.parent
        run_dir = repo_root / f"step_{self.global_step}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        root_gt = run_dir / "gt"
        root_est = run_dir / "est"
        root_gt.mkdir(parents=True, exist_ok=True)
        root_est.mkdir(parents=True, exist_ok=True)

        for tmp_i, tmp_img in enumerate(rgb_gt_np):
            PIL.Image.fromarray(tmp_img).save(root_gt / f"rank_{self.local_rank}_i_{tmp_i}.png")

        for tmp_i, tmp_img in enumerate(rgb_est_np):
            PIL.Image.fromarray(tmp_img).save(root_est / f"rank_{self.local_rank}_i_{tmp_i}.png")

        n_gt = len(list(root_gt.glob("*.png")))
        n_est = len(list(root_est.glob("*.png")))

        metric_fid = fid.compute_fid(
            fdir1=str(root_gt),
            fdir2=str(root_est),
            num_workers=0,  # NOTE: important! avoids spawning multiprocessing dataloaders when computing FID
            batch_size=32,
            verbose=True,
        )  # float
        metric_kid = fid.compute_kid(
            fdir1=str(root_gt),
            fdir2=str(root_est),
            num_workers=0,  # NOTE: important! avoids spawning multiprocessing dataloaders when computing FID
            batch_size=32,
            verbose=True,
        )  # float

        if run_dir.exists():
            shutil.rmtree(run_dir)

        return dict(
            fid=metric_fid,  # float
            kid=metric_kid,  # float
            n_gt=n_gt,  # int
            n_est=n_est,  # int
        )

    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self.velocity_estimator_ema:
            self.velocity_estimator_ema.update(self.velocity_estimator)

    def configure_optimizers(self):
        """construct optimizer"""

        lr = self.optim_config["lr"]
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=self.optim_config["weight_decay"],
            betas=(self.optim_config["beta1"], self.optim_config["beta2"]),
        )
        if self.optim_config["lr_scheduler_config"] is None:
            return optimizer

        scheduler = config_utils.instantiate_from_config(
            config=self.optim_config["lr_scheduler_config"],
            optimizer=optimizer,
            last_epoch=self.trainer.global_step - 1,  # total number of batches
        )

        # plot lr schedule in command line
        if self.trainer.local_rank == 0 and self.optim_config["plot_lr"]:
            print("plotting learning rate:", flush=True)
            pl_utils.plot_lr_schedule(
                optimizer=optimizer,
                scheduler=scheduler,
                num_iters=int(1e6),
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
