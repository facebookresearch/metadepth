# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the FAIR Noncommercial Research License. See LICENSE file for details.


"""
Neural network modules for MoGe.

Includes:
- DINOv2Encoder using DA2 DINOv2 implementation
- ResidualConvBlock, ConvStack, MLP, Resampler
"""

import functools
import itertools
from typing import Callable, List, Literal, Optional, Sequence, Tuple, TypeAlias, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ..da2.dpt import DinoVisionTransformer, DPT_MODEL_CONFIGS

PaddingMode: TypeAlias = Literal["zeros", "reflect", "replicate", "circular"]


__all__ = [
    "DINOv2Encoder",
    "ResidualConvBlock",
    "ConvStack",
    "MLP",
    "Resampler",
]


class ResidualConvBlock(nn.Module):
    """
    Residual convolutional block for feature refinement.
    """

    def __init__(  # noqa: C901
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        kernel_size: int = 3,
        padding_mode: PaddingMode = "replicate",
        activation: Literal["relu", "leaky_relu", "silu", "elu"] = "relu",
        in_norm: Literal[
            "group_norm", "layer_norm", "instance_norm", "none"
        ] = "layer_norm",
        hidden_norm: Literal[
            "group_norm", "layer_norm", "instance_norm", "none"
        ] = "group_norm",
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        _activation_map: dict[str, Callable[[], nn.Module]] = {
            "relu": nn.ReLU,
            "leaky_relu": functools.partial(nn.LeakyReLU, negative_slope=0.2),
            "silu": nn.SiLU,
            "elu": nn.ELU,
        }
        if activation not in _activation_map:
            raise ValueError(f"Unsupported activation function: {activation}")
        activation_fn = _activation_map[activation]

        # Build normalization layers
        if in_norm == "group_norm":
            in_norm_layer = nn.GroupNorm(max(1, in_channels // 32), in_channels)
        elif in_norm == "layer_norm":
            in_norm_layer = nn.GroupNorm(1, in_channels)
        elif in_norm == "instance_norm":
            in_norm_layer = nn.InstanceNorm2d(in_channels)
        else:
            in_norm_layer = nn.Identity()

        if hidden_norm == "group_norm":
            hidden_norm_layer = nn.GroupNorm(
                max(1, hidden_channels // 32), hidden_channels
            )
        elif hidden_norm == "layer_norm":
            hidden_norm_layer = nn.GroupNorm(1, hidden_channels)
        elif hidden_norm == "instance_norm":
            hidden_norm_layer = nn.InstanceNorm2d(hidden_channels)
        else:
            hidden_norm_layer = nn.Identity()

        self.layers = nn.Sequential(
            in_norm_layer,
            activation_fn(),
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
            ),
            hidden_norm_layer,
            activation_fn(),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
            ),
        )

        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip_connection(x)
        x = self.layers(x)
        x = x + skip
        return x


class Resampler(nn.Sequential):
    """
    Feature resampling module for upsampling/downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        type_: Literal[
            "pixel_shuffle",
            "nearest",
            "bilinear",
            "conv_transpose",
            "pixel_unshuffle",
            "avg_pool",
            "max_pool",
        ],
        scale_factor: int = 2,
    ) -> None:
        if type_ == "pixel_shuffle":
            nn.Sequential.__init__(
                self,
                nn.Conv2d(
                    in_channels,
                    out_channels * (scale_factor**2),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
            conv0 = self[0]
            assert isinstance(conv0, nn.Conv2d)
            conv0_bias = conv0.bias
            assert conv0_bias is not None
            for i in range(1, scale_factor**2):
                conv0.weight.data[i :: scale_factor**2] = conv0.weight.data[
                    0 :: scale_factor**2
                ]
                conv0_bias.data[i :: scale_factor**2] = conv0_bias.data[
                    0 :: scale_factor**2
                ]
        elif type_ in ["nearest", "bilinear"]:
            nn.Sequential.__init__(
                self,
                nn.Upsample(
                    scale_factor=scale_factor,
                    mode=type_,
                    align_corners=False if type_ == "bilinear" else None,
                ),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
        elif type_ == "conv_transpose":
            nn.Sequential.__init__(
                self,
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=scale_factor,
                    stride=scale_factor,
                ),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
            conv0 = self[0]
            assert isinstance(conv0, nn.ConvTranspose2d)
            conv0.weight.data[:] = conv0.weight.data[:, :, :1, :1]
        elif type_ == "pixel_unshuffle":
            nn.Sequential.__init__(
                self,
                nn.PixelUnshuffle(scale_factor),
                nn.Conv2d(
                    in_channels * (scale_factor**2),
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
            )
        elif type_ == "avg_pool":
            nn.Sequential.__init__(
                self,
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor),
            )
        elif type_ == "max_pool":
            nn.Sequential.__init__(
                self,
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor),
            )
        else:
            raise ValueError(f"Unsupported resampler type: {type_}")


class MLP(nn.Sequential):
    """
    Simple MLP (Multi-Layer Perceptron).
    """

    def __init__(self, dims: Sequence[int]) -> None:
        nn.Sequential.__init__(
            self,
            *itertools.chain(
                *[
                    (nn.Linear(dim_in, dim_out), nn.ReLU(inplace=True))
                    for dim_in, dim_out in zip(dims[:-2], dims[1:-1])
                ]
            ),
            nn.Linear(dims[-2], dims[-1]),
        )


class ConvStack(nn.Module):
    """
    Multi-scale convolutional feature processing stack.
    """

    def __init__(
        self,
        dim_in: List[Optional[int]],
        dim_res_blocks: List[int],
        dim_out: List[Optional[int]],
        resamplers: Union[
            Literal[
                "pixel_shuffle",
                "nearest",
                "bilinear",
                "conv_transpose",
                "pixel_unshuffle",
                "avg_pool",
                "max_pool",
            ],
            List,
        ],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: Union[int, List[int]] = 1,
        res_block_in_norm: Literal[
            "layer_norm", "group_norm", "instance_norm", "none"
        ] = "layer_norm",
        res_block_hidden_norm: Literal[
            "layer_norm", "group_norm", "instance_norm", "none"
        ] = "group_norm",
        activation: Literal["relu", "leaky_relu", "silu", "elu"] = "relu",
    ) -> None:
        super().__init__()

        self.input_blocks = nn.ModuleList(
            [
                (
                    nn.Conv2d(
                        int(dim_in_),
                        dim_res_block_,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                    if dim_in_ is not None
                    else nn.Identity()
                )
                for dim_in_, dim_res_block_ in zip(
                    dim_in
                    if isinstance(dim_in, Sequence)
                    else itertools.repeat(dim_in),
                    dim_res_blocks,
                )
            ]
        )

        self.resamplers = nn.ModuleList(
            [
                # pyre-fixme[6]: Expected literal string type for `type_`.
                Resampler(dim_prev, dim_succ, scale_factor=2, type_=resampler)
                for i, (dim_prev, dim_succ, resampler) in enumerate(
                    zip(
                        dim_res_blocks[:-1],
                        dim_res_blocks[1:],
                        (
                            resamplers
                            if isinstance(resamplers, Sequence)
                            else itertools.repeat(resamplers)
                        ),
                    )
                )
            ]
        )

        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *(
                        ResidualConvBlock(
                            dim_res_block_,
                            dim_res_block_,
                            dim_times_res_block_hidden * dim_res_block_,
                            activation=activation,
                            in_norm=res_block_in_norm,
                            hidden_norm=res_block_hidden_norm,
                        )
                        for _ in range(
                            num_res_blocks[i]
                            if isinstance(num_res_blocks, list)
                            else num_res_blocks
                        )
                    )
                )
                for i, dim_res_block_ in enumerate(dim_res_blocks)
            ]
        )

        self.output_blocks = nn.ModuleList(
            [
                (
                    nn.Conv2d(
                        dim_res_block_,
                        int(dim_out_),
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                    if dim_out_ is not None
                    else nn.Identity()
                )
                for dim_out_, dim_res_block_ in zip(
                    dim_out
                    if isinstance(dim_out, Sequence)
                    else itertools.repeat(dim_out),
                    dim_res_blocks,
                )
            ]
        )

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""

        def make_checkpoint_wrapper(module: nn.Module) -> nn.Module:
            class CheckpointWrapper(nn.Module):
                def __init__(self, inner: nn.Module) -> None:
                    super().__init__()
                    self.inner = inner

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return torch.utils.checkpoint.checkpoint(
                        self.inner, x, use_reentrant=False
                    )

            # pyre-fixme[16]: Pyre does not resolve nested function references.
            return CheckpointWrapper(module)

        for i in range(len(self.resamplers)):
            self.resamplers[i] = make_checkpoint_wrapper(self.resamplers[i])
        for i in range(len(self.res_blocks)):
            for j in range(len(self.res_blocks[i])):
                self.res_blocks[i][j] = make_checkpoint_wrapper(self.res_blocks[i][j])

    def forward(self, in_features: List[torch.Tensor]) -> List[torch.Tensor]:
        out_features = []
        x = None
        for i in range(len(self.res_blocks)):
            feature = self.input_blocks[i](in_features[i])
            if i == 0:
                x = feature
            elif feature is not None:
                x = x + feature
            x = self.res_blocks[i](x)
            out_features.append(self.output_blocks[i](x))
            if i < len(self.res_blocks) - 1:
                x = self.resamplers[i](x)
        return out_features


class DINOv2Encoder(nn.Module):
    """
    DINOv2 encoder for MoGe.
    Wraps the DinoVisionTransformer from da2.dpt and provides
    a compatible DINOv2Encoder interface.
    """

    def __init__(
        self,
        backbone: str,
        intermediate_layers: Union[int, List[int]],
        dim_out: int,
        **deprecated_kwargs,
    ) -> None:
        super().__init__()

        self.intermediate_layers = intermediate_layers
        self.backbone_name = backbone

        # Map backbone name to config
        backbone_to_config = {
            "vitl_dinov2": "vitl",
            "vitb_dinov2": "vitb",
            "vits_dinov2": "vits",
            "dinov2_vitl14": "vitl",
        }

        if backbone not in backbone_to_config:
            raise ValueError(
                f"Unknown backbone: {backbone}. "
                f"Available: {list(backbone_to_config.keys())}"
            )

        config_name = backbone_to_config[backbone]
        config = DPT_MODEL_CONFIGS[config_name]

        # Build the DINOv2 backbone
        self.backbone = DinoVisionTransformer(
            img_size=518,
            patch_size=14,
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            init_values=1.0,
            ffn_layer="mlp" if config_name != "vitg" else "swiglu",
            interpolate_antialias=False,
            interpolate_offset=0.1,
        )

        self.dim_features = config["embed_dim"]
        self.num_features = (
            intermediate_layers
            if isinstance(intermediate_layers, int)
            else len(intermediate_layers)
        )

        # Output projections
        self.output_projections = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.dim_features,
                    out_channels=dim_out,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for _ in range(self.num_features)
            ]
        )

        # Image normalization
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def init_weights(self) -> None:
        """Initialize weights (placeholder for compatibility)."""
        pass

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        for i in range(len(self.backbone.blocks)):

            def make_checkpoint_wrapper(module: nn.Module) -> nn.Module:
                class CheckpointWrapper(nn.Module):
                    def __init__(self, inner: nn.Module) -> None:
                        super().__init__()
                        self.inner = inner

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        return torch.utils.checkpoint.checkpoint(
                            self.inner, x, use_reentrant=False
                        )

                # pyre-fixme[16]: Pyre does not resolve nested function references.
                return CheckpointWrapper(module)

            self.backbone.blocks[i] = make_checkpoint_wrapper(self.backbone.blocks[i])

    def enable_pytorch_native_sdpa(self) -> None:
        """Enable PyTorch native SDPA (already used in OSS implementation)."""
        pass

    def forward(
        self,
        image: torch.Tensor,
        token_rows: Union[int, torch.LongTensor],
        token_cols: Union[int, torch.LongTensor],
        return_class_token: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Tuple]]:
        """
        Forward pass through encoder.

        Args:
            image: Input image tensor (B, 3, H, W) in range [0, 1]
            token_rows: Number of token rows
            token_cols: Number of token columns
            return_class_token: Whether to return class token

        Returns:
            Feature tensor and optionally class token
        """
        # Resize to match token grid
        image_14 = F.interpolate(
            image,
            (int(token_rows) * 14, int(token_cols) * 14),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        image_14 = (image_14 - self.image_mean) / self.image_std

        # Get intermediate layers from backbone
        features = self.backbone.get_intermediate_layers(
            image_14, n=self.intermediate_layers, return_class_token=True
        )

        # Project features to output dimensionality
        x = torch.stack(
            [
                proj(
                    feat.permute(0, 2, 1)
                    .unflatten(2, (int(token_rows), int(token_cols)))
                    .contiguous()
                )
                for proj, (feat, clstoken) in zip(self.output_projections, features)
            ],
            dim=1,
        ).sum(dim=1)

        if return_class_token:
            # pyre-fixme[7]: Expected return type but tuple structure differs.
            return x, features[-1][1], features[-1]
        else:
            return x
