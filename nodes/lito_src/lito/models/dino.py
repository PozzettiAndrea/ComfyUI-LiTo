#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#
# The file implements codes to use pretrained dinov2 model.

import copy
import math
import typing as T

import numpy as np

import torch
from torchvision.transforms import Compose, Normalize

from plibs import utils
_OPS = None
def _ops():
    global _OPS
    if _OPS is None:
        import comfy.ops
        _OPS = comfy.ops.manual_cast
    return _OPS



def resize(
    image: torch.Tensor,
    size: T.Union[int, T.Tuple[int, int]],
):
    r"""
    Resize image or feature map using bilinear interpolation.

    Args:
        image:
            (b, c, h, w) image or feature map to be resized
        size:
            (2,) target size (h', w') to resize to

    Returns:
        (b, c, h', w')
    """

    return torch.nn.functional.interpolate(
        image,
        size=size,
        mode="bilinear",
        align_corners=False,
    )


def get_dino_tranform():
    """
    Return a torchvision transform callable
    that takes input image tensor (*, c, h, w) [0, 1]
    and output image (*, c, h, w) that can be
    fed to dino as input.
    """
    image_transforms = Compose(
        [
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return image_transforms


class SpatialDino(torch.nn.Module):
    """
    Use dinov2 to encode an image as patches.
    Note that dino uses a patch size = 14.
    """

    def __init__(
        self,
        freeze_weights: bool = True,
        model_type: str = "dinov2_vits14",
        learnable_model_type: str = "none",
        learnable_params: T.Dict[str, T.Any] = None,
        use_prenorm_feature: bool = True,  # whether to use dino feature before (True) or after layernorm
        use_prenrom_cls_and_register_feature: bool = False,
        use_prenorm_feature_with_layernorm: bool = False,
        prob_drop_dino: float = 0,  # probability to provide all zero instead of dino feature
        check_dino_output_nograd: bool = True,
        dino_shift: float = None,
        dino_scale: float = None,
        dino_clip_min: float = None,
        dino_clip_max: float = None,
    ):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_type)
        self.feature_dim = self.model.embed_dim  # output dimension of the model
        self.patch_size = self.model.patch_size
        self.learnable_model_type = learnable_model_type
        self.learnable_params = learnable_params
        self.use_prenorm_feature = use_prenorm_feature
        self.use_prenrom_cls_and_register_feature = use_prenrom_cls_and_register_feature
        self.use_prenorm_feature_with_layernorm = use_prenorm_feature_with_layernorm
        self.prob_drop_dino = prob_drop_dino
        self.check_dino_output_nograd = check_dino_output_nograd

        self.dino_shift = dino_shift
        self.dino_scale = dino_scale
        self.dino_clip_min = dino_clip_min
        self.dino_clip_max = dino_clip_max

        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

        # create learnable part
        if self.learnable_model_type == "none":
            self.learnable_model = None
        elif self.learnable_model_type == "linear":
            assert self.learnable_params is not None
            assert "out_channels" in self.learnable_params, f"{list(self.learnable_params.keys())=}"
            if "input_types" not in self.learnable_params:
                self.learnable_params["input_types"] = ["rgb"]
            in_channels = 0
            for input_type in self.learnable_params["input_types"]:
                if input_type == "rgb":
                    in_channels += 3
                elif input_type == "xyz_w":
                    in_channels += 3
                elif input_type == "plucker":
                    in_channels += 6
                elif input_type == "hit":
                    in_channels += 1
                else:
                    raise NotImplementedError
            self.learnable_model = _ops().Conv2d(
                in_channels=in_channels,
                out_channels=self.learnable_params["out_channels"],
                kernel_size=self.model.patch_size,
                stride=self.model.patch_size,
            )
        else:
            raise NotImplementedError

        if self.use_prenorm_feature_with_layernorm:
            # We need to create learnable parameters for the CLS and register tokens in DINO feature
            n_cls = 1
            if model_type in [
                "dinov2_vits14_reg",
                "dinov2_vitb14_reg",
                "dinov2_vitl14_reg",
                "dinov2_vitg14_reg",
            ]:
                # see https://github.com/facebookresearch/dinov2/blob/b8931f7bf91576930313be2c6d6af376033b35f0/dinov2/hub/backbones.py#L136
                n_reg = 4
            elif model_type in [
                "dinov2_vits14",
                "dinov2_vitb14",
                "dinov2_vitl14",
                "dinov2_vitg14",
            ]:
                n_reg = 0
            else:
                raise NotImplementedError(model_type)

            self.learnable_pad_params = torch.nn.Parameter(
                torch.zeros(n_cls + n_reg, self.learnable_params["out_channels"])
            )

            self.output_flattened = True
        else:
            self.learnable_pad_params = None

            self.output_flattened = False

    def forward(
        self,
        x: torch.Tensor,  # (b, c, h, w)
        xyz_w: torch.Tensor = None,
        plucker: torch.Tensor = None,
        hit: torch.Tensor = None,
    ):
        """
        Spatial dimensions of output will be H // patch_size, W // patch_size.

        Args:
            x (torch.Tensor):
                Images (b, c, h, w). Should be normalized by
                the above transform (created by `get_dino_tranform()`).
                Additionally, h and w should also be a multiple of patch_size.
            xyz_w:
                (b, 3xyz_w, h, w).  xyz_w for individual pixels
            plucker:
                (b, 6, h, w) plucker ray for individual pixels
            hit:
                (b, 1, h, w). hit for individual pixels. float
        Returns:
            feature_map (torch.tensor): (b, c, h // 14, w // 14)
        """
        *b_shape, c, h, w = x.shape

        assert h % self.patch_size == 0, f"Input image height {h} is not a multiple of patch height {self.patch_size}"
        assert w % self.patch_size == 0, f"Input image width {w} is not a multiple of patch width: {self.patch_size}"

        x = x.reshape(-1, c, h, w)
        if xyz_w is not None:
            assert xyz_w.shape == (*b_shape, 3, h, w)
            xyz_w = xyz_w.reshape(-1, 3, h, w)
        if plucker is not None:
            assert plucker.shape == (*b_shape, 6, h, w)
            plucker = plucker.reshape(-1, 6, h, w)
        if hit is not None:
            assert hit.shape == (*b_shape, 1, h, w)
            hit = hit.reshape(-1, 1, h, w)

        ph = h // self.patch_size
        pw = w // self.patch_size

        # dino
        if self.prob_drop_dino < 1e-6 or torch.rand(1).item() >= self.prob_drop_dino:
            # assert sum([p.requires_grad for name, p in self.model.named_parameters() if p is not None]) == 0
            if not self.use_prenorm_feature:
                features = self.model.forward_features(x)["x_norm_patchtokens"]  # (b, h'w', c)
            else:
                features = self.model.forward_features(x)["x_prenorm"]  # (b, #cls + #reg + h'w', c)
                if not self.use_prenrom_cls_and_register_feature:
                    features = features[:, self.model.num_register_tokens + 1 :]  # (b, h'w', c)

                if self.use_prenorm_feature_with_layernorm:
                    features = torch.nn.functional.layer_norm(features, features.shape[-1:])  # (b, h'w', c)

            # clip, shift, scale
            if self.dino_clip_min is not None or self.dino_clip_max is not None:
                features = torch.clamp(
                    features,
                    min=self.dino_clip_min,
                    max=self.dino_clip_max,
                )  # (b, h'w', c)
            if self.dino_shift is not None:
                features = features + self.dino_shift
            if self.dino_scale is not None:
                features = features * self.dino_scale

            if self.check_dino_output_nograd:
                assert not features.requires_grad

            features = features.permute(0, 2, 1)  # (b, c, h'w')
            if not self.use_prenrom_cls_and_register_feature:
                features = features.reshape(*b_shape, self.feature_dim, ph, pw)  # (*b, c, h//ps, w//ps)

        else:
            features = torch.zeros(
                *b_shape,
                self.feature_dim,
                h // self.patch_size,
                w // self.patch_size,
                dtype=x.dtype,
                device=x.device,
            )  # (*b, c, h//ps, w//ps)

        # learnable model
        if self.learnable_model is not None:
            x_input = []
            for input_type in self.learnable_params["input_types"]:
                if input_type == "rgb":
                    x_input.append(x)
                elif input_type == "xyz_w":
                    x_input.append(xyz_w)
                elif input_type == "plucker":
                    x_input.append(plucker)
                elif input_type == "hit":
                    x_input.append(hit)
                else:
                    raise NotImplementedError
            if len(x_input) == 1:
                x_input = x_input[0]
            else:
                x_input = torch.cat(x_input, dim=1)  # (b, d, h, w)
            learnable_feature = self.learnable_model(x_input)  # (b, d, h//ps, w//ps)
            _b, d, _ph, _pw = learnable_feature.shape

            if not self.use_prenrom_cls_and_register_feature:
                assert _ph == features.size(-1), f"{ph=}, {features.shape=}"
                assert _pw == features.size(-2), f"{pw=}, {features.shape=}"
                learnable_feature = learnable_feature.reshape(*b_shape, d, ph, pw)  # (*b, d, ph, pw)
                features = torch.cat([features, learnable_feature], dim=-3)  # (*b, c+d, ph, pw)
            else:
                learnable_feature = learnable_feature.reshape(*b_shape, d, ph * pw)  # (*b, d, ph * pw)

                learnable_padding = self.learnable_pad_params.permute(1, 0).expand(*b_shape, -1, -1)  # (*b, d, #pad)

                # NOTE: the padding should come first as DINOv2 puts CLS/registers before patch features.
                learnable_feature = torch.cat((learnable_padding, learnable_feature), dim=-1)  # (*b, d, #pad + ph * pw)

                features = torch.cat([features, learnable_feature], dim=-2)  # (*b, c+d, #pad + ph * pw)

        return features


class Dinov2(torch.nn.Module):
    """
    Use dinov2 to encode an image as patches.
    Note that dino uses a patch size = 14.
    """

    def __init__(
        self,
        model_type: str = "dinov2_vitl14_reg",
        layer_idxs: T.List[int] = (-1,),
        normalize: bool = False,
    ):
        """
        Args:
            model_type:
                'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14' 'dinov2_vitg14',
                'dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg'
            layer_idxs:
                list of int, the index of the layers to get the output.
            normalize:
                bool, whether to normalize individual tokens with the same layernorm.
        """
        super().__init__()

        # load dino
        self.model = torch.hub.load("facebookresearch/dinov2", model_type)

        self.feature_dim = self.model.embed_dim  # output dimension of the model
        self.patch_size = self.model.patch_size
        self.num_layers = len(self.model.blocks)
        self.num_cls_tokens = 1
        self.num_register_tokens = self.model.num_register_tokens
        self.normalize = normalize

        if isinstance(layer_idxs, int):
            layer_idxs = [layer_idxs]

        # make sure layer_idxs is non-negative
        layer_idxs = [l % self.num_layers for l in layer_idxs]
        self.layer_idxs = layer_idxs

        # normalization function
        self.img_transform = get_dino_tranform()

    def get_cls_tokens(
        self,
        x: T.Union[torch.Tensor, T.List[torch.Tensor]],  # (b, seq_len, d)
    ):
        """
        Get the class token from the output tokens.

        Args:
            x:
                (b, seq_len, d) or list of (b, seq_len, d)

        Returns:
            (b, ncls, d) or list of (b, ncls, d)
        """

        if isinstance(x, torch.Tensor):
            return x[:, 0 : self.num_cls_tokens]  # (b, ncls, d)
        else:
            return [arr[:, 0 : self.num_cls_tokens] for arr in x]  # list of (b, ncls, d)

    def get_register_tokens(
        self,
        x: T.Union[torch.Tensor, T.List[torch.Tensor]],
    ):
        """
        Get the register token from the output tokens.

        Args:
            x:
                (b, seq_len, d) or list of (b, seq_len, d)

        Returns:
            (b, num_reg_tokens, d) or list of (b, num_reg_tokens, d)
        """

        if isinstance(x, torch.Tensor):
            return x[:, self.num_cls_tokens : self.num_cls_tokens + self.num_register_tokens]  # (b, n, d)
        else:
            return [
                arr[:, self.num_cls_tokens : self.num_cls_tokens + self.num_register_tokens] for arr in x
            ]  # list of (b, n, d)

    def get_patch_tokens(
        self,
        x: T.Union[torch.Tensor, T.List[torch.Tensor]],
    ):
        """
        Get the patch tokens from the output tokens.

        Args:
            x:
                (b, seq_len, d) or list of (b, seq_len, d)

        Returns:
            (b, num_patches, d) or list of (b, num_patches, d)
        """

        if isinstance(x, torch.Tensor):
            return x[:, (self.num_cls_tokens + self.num_register_tokens) :]  # (b, n, d)
        else:
            return [arr[:, (self.num_cls_tokens + self.num_register_tokens) :] for arr in x]  # list of (b, n, d)

    def forward(
        self,
        premultiplied_rgb: torch.Tensor,  # (b, c, h, w) [0, 1]
    ) -> T.Dict[str, T.Any]:
        """
        Spatial dimensions of output will be H // patch_size, W // patch_size.

        Args:
            premultiplied_rgb:
                (b, c, h, w) [0, 1].  not yet normalized by get_dino_transform.
                Additionally, h and w should also be a multiple of patch_size.
        Returns:
            cls_tokens:
                (num_layer,) list, each is (b, 1, d)
            reg_tokens:
                (num_layer,) list, each is (b, num_reg, d)
            patch_tokens:
                (num_layer,) list, each is (b, num_patches, d)
            ph:
                int, can be used to reshape patch tokens
            pw:
                int, can be used to reshape patch tokens
        """
        *b_shape, c, h, w = premultiplied_rgb.shape
        assert h % self.patch_size == 0, f"Input image height {h} is not a multiple of patch height {self.patch_size}"
        assert w % self.patch_size == 0, f"Input image width {w} is not a multiple of patch width: {self.patch_size}"
        ph = h // self.patch_size
        pw = w // self.patch_size

        x = premultiplied_rgb.reshape(-1, c, h, w)

        # get feature
        with torch.no_grad():
            assert not self.model.bag_of_channels

            # normalize mean and std
            x = self.img_transform(x)  # (b, c, h, w)

            if self.model.chunked_blocks:
                outputs = self.model._get_intermediate_layers_chunked(
                    x, self.layer_idxs
                )  # (num_layer,) list, each is (b, seq_len, d)
            else:
                outputs = self.model._get_intermediate_layers_not_chunked(
                    x, self.layer_idxs
                )  # (num_layer,) list, each is (b, seq_len, d)

            # normalize the mean and std of individual tokens
            # it uses the same layernorm -- this makes all tokens to have the same statistics
            if self.normalize:
                outputs = [self.model.norm(out) for out in outputs]

        cls_tokens = self.get_cls_tokens(outputs)  # (num_layer,) list, each is (b, 1, d)
        reg_tokens = self.get_register_tokens(outputs)  # (num_layer,) list, each is (b, num_reg, d)
        patch_tokens = self.get_patch_tokens(outputs)  # (num_layer,) list, each is (b, num_patches, d)

        return dict(
            cls_tokens=cls_tokens,  # (num_layer,) list, each is (b, 1, d)
            reg_tokens=reg_tokens,  # (num_layer,) list, each is (b, num_reg, d)
            patch_tokens=patch_tokens,  # (num_layer,) list, each is (b, num_patches, d)
            ph=ph,
            pw=pw,
        )


class SpatialDinov2(torch.nn.Module):
    """
    Use dinov2 to encode an image as patches.
    Note that dino uses a patch size = 14.

    Compared to SpatialDino, the class cleans up the complex interface.
    """

    def __init__(
        self,
        model_type: str = "dinov2_vitl14_reg",
        dino_layer_idxs: T.List[int] = (-1,),  # (4, 11, 17, 23),
        dino_normalize_tokens: bool = False,
        dino_normalize_concat_tokens: bool = False,  # only exists for backward compatibility
        dino_use_cls: bool = True,
        dino_use_registers: bool = False,
        learnable_model_type: str = "linear",
        learnable_model_params: T.Dict[str, T.Any] = dict(
            out_channels=1024,
            input_types=("rgb", "alpha"),
            add_layer_norm=True,
        ),
        learnable_model_first_transforms_rgb: bool = False,
        learnable_add_joint_layernorm: bool = False,  # joint layernorm with dino and linear
        width_px: T.Union[int, T.List[int]] = None,  # None: use input image as is
        height_px: T.Union[int, T.List[int]] = None,  # None: use input image as is
    ):
        super().__init__()

        self.width_px = width_px
        self.height_px = height_px

        # load dino
        self.dinov2_model = Dinov2(
            model_type=model_type,
            layer_idxs=dino_layer_idxs,
            normalize=dino_normalize_tokens,
        )
        # make dino in eval and does not require grad
        self.dinov2_model.eval()
        for param in self.dinov2_model.parameters():
            param.requires_grad = False

        self.dino_normalize_concat_tokens = dino_normalize_concat_tokens

        self.dino_use_cls = dino_use_cls
        self.dino_use_registers = dino_use_registers
        self.num_extra_tokens = 0
        if self.dino_use_cls:
            self.num_extra_tokens += self.dinov2_model.num_cls_tokens
        if self.dino_use_registers:
            self.num_extra_tokens += self.dinov2_model.num_register_tokens

        # get learnable model
        self.learnable_model_type = learnable_model_type
        self.learnable_model_params = learnable_model_params
        self.learnable_model_first_transforms_rgb = learnable_model_first_transforms_rgb
        self.learnable_add_joint_layernorm = learnable_add_joint_layernorm
        self.get_learnable_model()

    def get_learnable_model(self):
        dim_out_feature = 0

        # create learnable part
        if self.learnable_model_type == "none":
            self.learnable_model = None
        elif self.learnable_model_type == "linear":
            assert self.learnable_model_params is not None
            assert self.learnable_model_params.get("out_channels", None) is not None
            assert self.learnable_model_params.get("input_types", None) is not None
            assert self.learnable_model_params.get("add_layer_norm", None) is not None

            in_channels = 0
            for input_type in self.learnable_model_params["input_types"]:
                if input_type == "rgb":
                    in_channels += 3
                elif input_type == "xyz_w":
                    in_channels += 3
                elif input_type == "plucker":
                    in_channels += 6
                elif input_type == "alpha":
                    in_channels += 1
                else:
                    raise NotImplementedError(input_type)

            self.learnable_model = _ops().Conv2d(
                in_channels=in_channels,
                out_channels=self.learnable_model_params["out_channels"],
                kernel_size=self.dinov2_model.patch_size,
                stride=self.dinov2_model.patch_size,
            )

            if self.learnable_model_params["add_layer_norm"]:
                self.learnable_linear_layernorm = _ops().LayerNorm(
                    self.learnable_model_params["out_channels"],
                    eps=1e-6,
                )
            else:
                self.learnable_linear_layernorm = None

            dim_out_feature = self.learnable_model_params["out_channels"]
        else:
            raise NotImplementedError

        # learn padding
        if self.num_extra_tokens > 0 and dim_out_feature > 0:
            self.learnable_paddings = torch.nn.Parameter(
                torch.zeros(self.num_extra_tokens, dim_out_feature)
            )  # (num_extra_tokens, d)
        else:
            self.learnable_paddings = None

        # joint layernorm
        if self.learnable_add_joint_layernorm:
            self.learnable_joint_layernorm = _ops().LayerNorm(dim_out_feature, eps=1e-6)
        else:
            self.learnable_joint_layernorm = None

        # dino transform for premultiplied rgb
        if self.learnable_model_first_transforms_rgb:
            self.dino_img_transform = get_dino_tranform()
        else:
            self.dino_img_transform = None

    def _forward_learnable_model(
        self,
        b: int,
        ph: int,
        pw: int,
        premultiplied_rgb: torch.Tensor,  # (b, 3rgb, h, w)
        xyz_w: torch.Tensor = None,
        plucker: torch.Tensor = None,
        alpha: torch.Tensor = None,
    ):
        assert self.learnable_model is not None

        # construct input
        x_input = []
        for input_type in self.learnable_model_params["input_types"]:
            if input_type == "rgb":
                if self.learnable_model_first_transforms_rgb:
                    assert self.dino_img_transform is not None
                    x_input.append(self.dino_img_transform(premultiplied_rgb))
                else:
                    x_input.append(premultiplied_rgb)
            elif input_type == "xyz_w":
                x_input.append(xyz_w)
            elif input_type == "plucker":
                x_input.append(plucker)
            elif input_type == "alpha":
                x_input.append(alpha)
            else:
                raise NotImplementedError
        x_input = torch.cat(x_input, dim=1) if len(x_input) > 1 else x_input[0]  # (b, d, h, w)

        if self.learnable_model_type == "linear":
            x_input = self.learnable_model(x_input)  # (b, d, ph, pw)
            _b, d, _ph, _pw = x_input.shape
            assert _ph == ph, f"{_ph} != {ph}"
            assert _pw == pw, f"{_pw} != {pw}"

            if self.learnable_linear_layernorm is not None:
                x_input = self.learnable_linear_layernorm(
                    x_input.permute(0, 2, 3, 1)  # (b, ph, pw, d)
                ).permute(0, 3, 1, 2)  # (b, d, ph, pw)

        else:
            raise NotImplementedError

        x_input = x_input.reshape(b, d, ph * pw).permute(0, 2, 1)  # (b, phpw, d)
        if self.learnable_paddings is not None:
            x_input = torch.cat(
                [
                    self.learnable_paddings.expand(b, self.num_extra_tokens, d),  # (b, n, d)
                    x_input,  # (b, phpw, d)
                ],
                dim=1,
            )  # (b, num_extra+ phpw, d)

        return x_input  # (b, num_extra+ phpw, d)

    def resize_input(
        self,
        premultiplied_rgb: torch.Tensor,  # (b, 3rgb, h, w)
        xyz_w: torch.Tensor = None,
        plucker: torch.Tensor = None,
        alpha: torch.Tensor = None,
    ):
        """
        Resize the input to target resolution.

        Args:
            premultiplied_rgb:
                (b, 3rgb, h, w) [0, 1].  Not yet normalized.  should be premultiplied!
            xyz_w:
                (b, 3xyz_w, h, w).  xyz_w for individual pixels
            plucker:
                (b, 6, h, w) plucker ray for individual pixels
            alpha:
                (b, 1, h, w). alpha for individual pixels. float. [0, 1]

        Returns:
            dict of key -> (b, c, target_height_px, target_width_px)
        """

        b, _3rgb, h, w = premultiplied_rgb.shape

        width_px = copy.deepcopy(self.width_px)
        height_px = copy.deepcopy(self.height_px)

        if width_px is None:
            width_px = w
        if height_px is None:
            height_px = h

        if isinstance(width_px, int):
            width_px = [width_px]
        if isinstance(height_px, int):
            height_px = [height_px]

        assert len(width_px) == len(height_px)

        if self.training:
            target_width_px = np.random.choice(width_px).item()
            target_height_px = np.random.choice(height_px).item()
        else:
            hs = np.array(height_px)
            if (hs >= h).any():
                # closest higher resolution
                idx = np.argmin([_h - h for _h in height_px if _h >= h]).item()
            else:
                # closest resolution
                idx = np.argmin([abs(_h - h) for _h in height_px]).item()
            target_width_px = width_px[idx]
            target_height_px = height_px[idx]

        out_dict = dict()
        for key, arr in [
            ["premultiplied_rgb", premultiplied_rgb],
            ["xyz_w", xyz_w],
            ["plucker", plucker],
            ["alpha", alpha],
        ]:
            if arr is None:
                out_dict[key] = None
            elif target_height_px == h and target_width_px == w:
                out_dict[key] = arr
            else:
                arr = resize(
                    image=arr,  # (b, c, h, w)
                    size=(target_height_px, target_width_px),
                )  # (b, c, h', w')
                out_dict[key] = arr

        return out_dict

    def forward(
        self,
        premultiplied_rgb: torch.Tensor,  # (b, 3rgb, h, w)
        xyz_w: torch.Tensor = None,
        plucker: torch.Tensor = None,
        alpha: torch.Tensor = None,
        use_grad_checkpointing: bool = False,
    ) -> T.Dict[str, T.Any]:
        """
        Args:
            premultiplied_rgb:
                (b, 3rgb, h, w) [0, 1].  Not yet normalized.  should be premultiplied!
            xyz_w:
                (b, 3xyz_w, h, w).  xyz_w for individual pixels
            plucker:
                (b, 6, h, w) plucker ray for individual pixels
            alpha:
                (b, 1, h, w). alpha for individual pixels. float. [0, 1]
        Returns:
            out_tokens:
                (b, num_extra_tokens + ph * pw, d)
            ph:
                int
            pw:
                int
            num_extra_tokens:
                int
        """
        *b_shape, c, h, w = premultiplied_rgb.shape
        b = math.prod(b_shape)
        premultiplied_rgb = premultiplied_rgb.reshape(-1, c, h, w)
        if xyz_w is not None:
            assert xyz_w.shape == (*b_shape, 3, h, w)
            xyz_w = xyz_w.reshape(-1, 3, h, w)
        if plucker is not None:
            assert plucker.shape == (*b_shape, 6, h, w)
            plucker = plucker.reshape(-1, 6, h, w)
        if alpha is not None:
            assert alpha.shape == (*b_shape, 1, h, w)
            alpha = alpha.reshape(-1, 1, h, w)

        if self.width_px is not None or self.height_px is not None:
            idict = self.resize_input(
                premultiplied_rgb=premultiplied_rgb,
                xyz_w=xyz_w,
                plucker=plucker,
                alpha=alpha,
            )
            premultiplied_rgb = idict["premultiplied_rgb"]  # (b, 3, h, w)
            xyz_w = idict["xyz_w"]  # (b, 3, h, w)
            plucker = idict["plucker"]  # (b, 6, h, w)
            alpha = idict["alpha"]  # (b, 1, h, w)

        # dino
        with torch.no_grad():
            # # debug
            # assert not self.dinov2_model.training
            # for name, param in self.dinov2_model.named_parameters():
            #     assert not param.requires_grad, f"{name} requires grad"
            # # end debug

            dino_out_dict = self.dinov2_model(premultiplied_rgb=premultiplied_rgb)
            ph = dino_out_dict["ph"]
            pw = dino_out_dict["pw"]

            out_feature = []
            # the order is important (affects padding order below)
            if self.dino_use_cls:
                out_feature.append(
                    torch.cat(dino_out_dict["cls_tokens"], dim=-1)  # (num_layer,) list, each is (b, 1, d)
                    if len(dino_out_dict["cls_tokens"]) > 1
                    else dino_out_dict["cls_tokens"][0]
                )  # (b, 1, d)
            if self.dino_use_registers:
                out_feature.append(
                    torch.cat(dino_out_dict["reg_tokens"], dim=-1)  # (num_layer,) list, each is (b, num_reg, d)
                    if len(dino_out_dict["reg_tokens"]) > 1
                    else dino_out_dict["reg_tokens"][0]
                )  # (b, num_reg, d)
            out_feature.append(
                torch.cat(dino_out_dict["patch_tokens"], dim=-1)  # (num_layer,) list, each is (b, num_patch, d)
                if len(dino_out_dict["patch_tokens"]) > 1
                else dino_out_dict["patch_tokens"][0]
            )  # (b, num_patch, d)

            out_feature = (
                torch.cat(
                    out_feature,
                    dim=-2,
                )
                if len(out_feature) > 1
                else out_feature[0]
            )  # (b, num_extra+num_patches, d)
            del dino_out_dict

            if self.dino_normalize_concat_tokens:
                out_feature = torch.nn.functional.layer_norm(
                    out_feature,
                    out_feature.shape[-1:],
                )  # (b, num_extra+num_patches, d)

        # learnable model
        if self.learnable_model is not None:
            if not use_grad_checkpointing:
                out_learnable = self._forward_learnable_model(
                    b=b,
                    ph=ph,
                    pw=pw,
                    premultiplied_rgb=premultiplied_rgb,
                    xyz_w=xyz_w,
                    plucker=plucker,
                    alpha=alpha,
                )  # (b, num_extra + phpw, d)
            else:
                out_learnable = torch.utils.checkpoint.checkpoint(
                    self._forward_learnable_model,
                    b,
                    ph,
                    pw,
                    premultiplied_rgb,
                    xyz_w,
                    plucker,
                    alpha,
                    use_reentrant=False,
                )  # (b, num_extra + phpw, d)

            # concat along feature dimension
            out_feature = torch.cat(
                [
                    out_feature,  # (b, num_extra+num_patches, d)
                    out_learnable,  # (b, num_extra+ phpw, d')
                ],
                dim=-1,
            )  # (b, num_extra+ phpw, d)

        if self.learnable_joint_layernorm is not None:
            out_feature = self.learnable_joint_layernorm(out_feature)  # (b, num_extra+ phpw, d)

        out_dict = dict(
            out_tokens=out_feature.reshape(*b_shape, *out_feature.shape[1:]),  # (*b, num_extra + phpw, d)
            ph=ph,
            pw=pw,
            num_extra_tokens=self.num_extra_tokens,
        )
        return out_dict
