#
# Copyright (C) 2025 Apple Inc. All rights reserved.
#
# The file implements resnet with conv2d.

import torch
import torch.nn.functional as F

from lito.models.perceiver_encoder import PerceiverEncoderBlock
_OPS = None
def _ops():
    global _OPS
    if _OPS is None:
        import comfy.ops
        _OPS = comfy.ops.manual_cast
    return _OPS



class LayerNorm2d(torch.nn.Module):
    def __init__(self, num_features: int):
        super(LayerNorm2d, self).__init__()
        self.norm = _ops().LayerNorm(num_features)

    def forward(self, x: torch.Tensor):
        # x shape: [batch_size, channels, height, width]
        # Reshape for layer norm
        x = x.permute(0, 2, 3, 1)  # [batch_size, height, width, channels]
        x = self.norm(x)
        # Reshape back
        x = x.permute(0, 3, 1, 2)  # [batch_size, channels, height, width]
        return x


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation=torch.nn.ReLU(),
    ):
        super(ResidualBlock, self).__init__()

        # Save activation function
        self.activation = activation

        # First convolutional layer
        self.conv1 = _ops().Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.ln1 = LayerNorm2d(out_channels)

        # Second convolutional layer
        self.conv2 = _ops().Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.ln2 = LayerNorm2d(out_channels)

        # Skip connection
        self.shortcut = torch.nn.Sequential()
        # If dimensions change, we need to adjust the shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                _ops().Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                LayerNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (b, c, h, w)

        Returns:
            (b, co, h', w')
        """
        # Store input for the skip connection
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.ln1(out)
        out = self.activation(out)

        # Second conv block
        out = self.conv2(out)
        out = self.ln2(out)

        # Add skip connection
        out += self.shortcut(identity)

        # Final activation
        out = self.activation(out)

        return out


class ResidualBlocks(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        kernel_size: int = 3,
        stride: int = 1,  # downsample
        upsample_factor: int = 1,  # upsample
        activation=torch.nn.ReLU(),
    ):
        super(ResidualBlocks, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.stride = stride
        self.upsample_factor = upsample_factor

        layers = []
        # Only the first block in each layer might change dimensions
        layers.append(
            ResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
            )
        )

        # Remaining blocks maintain dimensions
        for _ in range(1, num_layers):
            layers.append(
                ResidualBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    activation=activation,
                )
            )

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # x: (b, c, h, w)

        if self.upsample_factor > 1:
            x = torch.nn.functional.interpolate(
                input=x,  # (b, c, h, w)
                scale_factor=self.upsample_factor,
                mode="nearest-exact",
                align_corners=None,
                antialias=False,
            )  # (b, c, s*h, s*w)

        return self.layers(x)


class ResUpNet(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_blocks: int,
        num_layers_per_block: int,
        upsample_factor: int = 1,  # upsample starts from the second block
        reduce_channels_when_upsample: bool = True,
        kernel_size: int = 3,
        activation=torch.nn.ReLU(),
        # # cross
        # add_perceiver: bool = False,
        # dim_kv_tokens: int = None,
        # num_self_attn: int = 1,
        # num_self_heads: int = 8,
        # num_cross_heads: int = 8,
        # dropout_prob: float = 0.0,
        # use_rmsnorm: bool = True,
        # mlp_ratio: float = 2,
        # mlp_type: str = "swiglu",
        # linear_in_attn_add_bias: bool = True,
        # mlp_add_bias: bool = True,
    ):
        super().__init__()
        self.overall_upsample_factor = int(upsample_factor ** (num_blocks - 1))
        # self.add_perceiver = add_perceiver

        # perceiver_blocks = []
        # if self.add_perceiver:
        #     assert dim_kv_tokens is not None
        #     block = PerceiverEncoderBlock(
        #         dim_latent=dim,
        #         dim_token=dim_kv_tokens,
        #         dim_qkv=dim,
        #         num_self_attn=num_self_attn,
        #         num_self_heads=num_self_heads,
        #         num_cross_heads=num_cross_heads,
        #         dropout_prob=dropout_prob,
        #         use_rmsnorm=use_rmsnorm,
        #         mlp_ratio=mlp_ratio,
        #         mlp_type=mlp_type,
        #         linear_in_attn_add_bias=linear_in_attn_add_bias,
        #         mlp_add_bias=mlp_add_bias,
        #     )
        #     perceiver_blocks.append(block)

        blocks = []
        block = ResidualBlocks(
            in_channels=dim,
            out_channels=dim,
            num_layers=num_layers_per_block,
            kernel_size=kernel_size,
            stride=1,
            upsample_factor=1,
            activation=activation,
        )
        blocks.append(block)

        current_dim = dim
        for i in range(num_blocks - 1):
            if reduce_channels_when_upsample:
                out_dim = current_dim // upsample_factor
            else:
                out_dim = current_dim
            assert out_dim >= 1

            block = ResidualBlocks(
                in_channels=current_dim,
                out_channels=out_dim,
                num_layers=num_layers_per_block,
                kernel_size=kernel_size,
                stride=1,
                upsample_factor=upsample_factor,
                activation=activation,
            )
            blocks.append(block)
            current_dim = out_dim

        self.blocks = torch.nn.ModuleList(blocks)
        self.out_channels = current_dim

    def forward(self, x: torch.Tensor):
        """
        Args:
            x:
                (b, ci, h, w)

        Returns:
            (b, co, hs, ws)
        """

        for layer_idx in range(len(self.blocks)):
            x = self.blocks[layer_idx](x)
        return x
