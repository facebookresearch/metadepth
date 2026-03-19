# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the FAIR Noncommercial Research License. See LICENSE file for details.


"""
Pure PyTorch CNN encoder architectures for depth estimation.

This module provides lightweight CNN encoders designed for efficient
depth estimation.

Currently implements:
- CNN_Large_HTP: Large model with HTP optimization (7.37M params)

All building blocks (Conv2dNormActivation, InvertedResidual, CNNEncoderBase)
are included in this single file for portability.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops.misc import Conv2dNormActivation as TorchvisionConv2dNormActivation


logger = logging.getLogger(__name__)


__all__ = [
    "CNNEncoderArch",
    "CNNEncoderBase",
    "CNN_Large_HTP",
    "Conv2dNormActivation",
    "InvertedResidual",
    "ShapeSpec",
    "build_cnn_encoder",
]


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class ShapeSpec:
    """
    Basic shape specification for a feature map tensor.

    Attributes:
        channels: Number of channels in the feature map
        stride: Stride/downsampling factor relative to input
    """

    channels: Optional[int] = None
    stride: Optional[int] = None


class CNNEncoderArch(str, Enum):
    """Available CNN encoder architectures."""

    large_htp = "large_htp"


# =============================================================================
# Base Class
# =============================================================================


class CNNEncoderBase(nn.Module):
    """
    Base class for CNN encoder architectures.

    CNN encoders are composed of multiple stages, where each stage is a
    sequence of layers. This base class manages these stages and provides
    a unified forward pass that returns features from all stages.
    """

    def __init__(
        self,
        stages: Sequence[nn.Module],
        compile: bool = False,
    ) -> None:
        """
        Args:
            stages: List of stage modules (nn.Module instances)
            compile: Whether to use torch.compile
        """
        super().__init__()
        self._stages = stages
        self.compile: bool = compile

        for idx, stage in enumerate(self._stages):
            self.add_module(f"stage{idx}", stage)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all stages.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Dictionary mapping stage names to feature tensors.
        """
        outputs: Dict[str, torch.Tensor] = {}

        for stage_name, stage in self.named_children():
            x = stage(x)
            outputs[stage_name] = x

        return outputs

    @property
    def output_shapes(self) -> Dict[str, ShapeSpec]:
        """
        Returns the output shape specification for each stage.
        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "output_shapes property must be implemented by subclasses"
        )


# =============================================================================
# Conv-Norm-Activation Block
# =============================================================================


class Conv2dNormActivation(TorchvisionConv2dNormActivation):
    """
    Conv2d -> Normalization -> Activation fused block.
    Inherits from torchvision.ops for compatibility.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            dilation=dilation,
            inplace=inplace,
            bias=bias,
        )


# =============================================================================
# Inverted Residual Block
# =============================================================================


class InvertedResidual(nn.Sequential):
    """
    Inverted Residual block (MobileNetV2 style).

    Components:
    1. Pointwise expansion (1x1 conv)
    2. Depthwise convolution (3x3 or 5x5)
    3. Pointwise projection (1x1 conv)
    4. Residual connection (when stride=1 and in_channels==out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        group_size: int = 1,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        activation_layer: Callable[..., nn.Module] = nn.ReLU,
        bias: bool = False,
        se_layer: Optional[Callable[..., nn.Module]] = None,
        use_residual_identity: bool = False,
        requires_pw_layer: bool = False,
        less_se: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        assert stride in [1, 2], f"stride should be 1 or 2 instead of {stride}"

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect: bool = self.stride == 1 and in_channels == out_channels

        self.res_conn_processor: Optional[nn.Module] = (
            nn.Identity() if self.use_res_connect and use_residual_identity else None
        )

        # 1. Pointwise expansion
        self.pw: Optional[nn.Module] = (
            Conv2dNormActivation(
                in_channels,
                hidden_dim,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                bias=bias,
            )
            if expand_ratio != 1 or requires_pw_layer
            else None
        )

        # 2. Depthwise convolution
        self.dw = Conv2dNormActivation(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            groups=hidden_dim // group_size,
            norm_layer=None,
            activation_layer=None,
            bias=bias,
        )

        # 3. Squeeze-Excitation (optional)
        self.se: Optional[nn.Module] = None
        if se_layer is not None:
            if less_se:
                self.se = se_layer(hidden_dim, expand_ratio)
            else:
                self.se = se_layer(hidden_dim)

        # 4. Pointwise projection
        self.pwl = Conv2dNormActivation(
            hidden_dim,
            out_channels,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=None,
            bias=bias,
        )

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass with optional residual connection."""
        y = input

        if self.pw is not None:
            y = self.pw(y)

        y = self.dw(y)

        if self.se is not None:
            y = self.se(y)

        y = self.pwl(y)

        if self.use_res_connect:
            y = torch.add(y, input)
            if self.res_conn_processor is not None:
                y = self.res_conn_processor(y)
            return y
        else:
            return y


# =============================================================================
# CNN_Large_HTP Architecture
# =============================================================================


class CNN_Large_HTP(CNNEncoderBase):
    """
    Large CNN encoder with HTP (Heterogeneous Tensor Processor) optimization.

    An efficient mobile CNN architecture with 7.37M parameters,
    designed for edge deployment while maintaining strong performance.

    Architecture:
    - 5 stages with progressive downsampling (stride 2, 4, 8, 16, 32)
    - Inverted Residual blocks (MobileNetV2 style)
    - Group convolutions for efficiency
    - Output channels: 32 -> 32 -> 64 -> 160 -> 288
    """

    def __init__(
        self,
        in_channels: int = 3,
        norm_layer: Callable[..., torch.nn.Module] = nn.BatchNorm2d,
        activation_layer: Callable[..., torch.nn.Module] = nn.ReLU,
        **kwargs: Any,
    ) -> None:
        ir = partial(
            InvertedResidual,
            kernel_size=3,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            use_residual_identity=True,
        )

        ir5 = partial(
            InvertedResidual,
            kernel_size=5,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            use_residual_identity=True,
        )

        ir_group = partial(
            InvertedResidual,
            kernel_size=3,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            use_residual_identity=True,
            group_size=32,
        )

        ir5_group = partial(
            InvertedResidual,
            kernel_size=5,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            use_residual_identity=True,
            group_size=32,
        )

        first_conv = Conv2dNormActivation(
            in_channels,
            32,
            stride=2,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            bias=True,
        )

        stages = [
            # stage0: 32 channels, stride 2
            nn.Sequential(
                first_conv,
                ir_group(32, 32, 2),
                ir_group(32, 32, 2),
                ir_group(32, 32, 2),
            ),
            # stage1: 32 channels, stride 4
            nn.Sequential(
                ir5_group(32, 32, 4, stride=2),
                ir_group(32, 32, 2),
                ir_group(32, 32, 2),
                ir_group(32, 32, 2),
                ir_group(32, 32, 2),
            ),
            # stage2: 64 channels, stride 8
            nn.Sequential(
                ir5(32, 64, 4, stride=2),
                ir(64, 64, 3),
                ir(64, 64, 3),
                ir(64, 64, 3),
                ir(64, 64, 3),
            ),
            # stage3: 160 channels, stride 16
            nn.Sequential(
                ir5(64, 96, 5, stride=2),
                ir(96, 96, 3),
                ir(96, 96, 3),
                ir(96, 96, 3),
                ir(96, 96, 3),
                ir(96, 160, 5),
                ir(160, 160, 3),
                ir(160, 160, 3),
                ir(160, 160, 3),
                ir(160, 160, 3),
                ir(160, 160, 3),
                ir(160, 160, 3),
                ir(160, 160, 3),
                ir(160, 160, 3),
            ),
            # stage4: 288 channels, stride 32
            nn.Sequential(
                ir(160, 256, 6, stride=2),
                ir(256, 256, 5),
                ir(256, 256, 5),
                ir(256, 256, 5),
                ir(256, 256, 5),
                ir(256, 256, 5),
                ir(256, 256, 5),
                ir(256, 288, 6),
            ),
        ]

        super().__init__(stages=stages, **kwargs)

    @property
    def output_shapes(self) -> Dict[str, ShapeSpec]:
        return {
            "stage0": ShapeSpec(channels=32, stride=2),
            "stage1": ShapeSpec(channels=32, stride=4),
            "stage2": ShapeSpec(channels=64, stride=8),
            "stage3": ShapeSpec(channels=160, stride=16),
            "stage4": ShapeSpec(channels=288, stride=32),
        }


# =============================================================================
# Builder Function
# =============================================================================


_ARCH_MAP: Dict[CNNEncoderArch, type] = {
    CNNEncoderArch.large_htp: CNN_Large_HTP,
}


def build_cnn_encoder(  # noqa: C901
    arch: Union[CNNEncoderArch, str] = CNNEncoderArch.large_htp,
    load_pretrained: bool = False,
    pretrained_weights_path: Optional[str] = None,
    dim_in: int = 3,
    compile: bool = False,
) -> CNNEncoderBase:
    """
    Build a CNN encoder with optional pretrained weights.

    Args:
        arch: Architecture to build (CNNEncoderArch enum or string like "large_htp")
        load_pretrained: Whether to load pretrained weights
        pretrained_weights_path: Local path to .pth checkpoint file
        dim_in: Number of input channels (default: 3 for RGB)
        compile: Whether to use torch.compile

    Returns:
        CNN encoder model instance

    Example:
        >>> model = build_cnn_encoder(CNNEncoderArch.large_htp)
        >>> model = build_cnn_encoder("large_htp", load_pretrained=True,
        ...     pretrained_weights_path="/path/to/checkpoint.pth")
    """
    if isinstance(arch, str):
        arch = CNNEncoderArch(arch)

    model_class = _ARCH_MAP[arch]
    logger.info(f"Building CNN encoder: {arch.value}")
    model = model_class(in_channels=dim_in, compile=compile)

    if load_pretrained:
        if pretrained_weights_path is None:
            raise ValueError(
                "load_pretrained=True but pretrained_weights_path is None. "
                "Please provide a local path to the checkpoint file."
            )

        logger.info(f"Loading pretrained weights from: {pretrained_weights_path}")
        checkpoint = torch.load(
            pretrained_weights_path, map_location="cpu", weights_only=True
        )

        if isinstance(checkpoint, dict):
            if "student" in checkpoint:
                state_dict = checkpoint["student"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "ema_state" in checkpoint:
                state_dict = checkpoint["ema_state"]
            else:
                # Try to extract encoder weights by prefix
                state_dict = {
                    k.replace("encoder.model.", ""): v
                    for k, v in checkpoint.items()
                    if "encoder.model" in k
                }
                if not state_dict:
                    state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Map checkpoint keys to match architecture
        load_first_layer = dim_in == 3
        state_dict = _map_checkpoint_keys(state_dict, load_first_layer)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(
                f"Unexpected keys when loading checkpoint: {unexpected_keys}"
            )
        logger.info("Successfully loaded pretrained weights")

    if compile:
        model = torch.compile(model)

    return model


def _map_checkpoint_keys(
    state_dict: Dict[str, torch.Tensor], load_first: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Map checkpoint keys to match CNN encoder architecture.

    Handles key name differences between checkpoints
    and the model structure.
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        if not load_first:
            if "stage0" in key and "layer_0" in key:
                continue

        new_key = key
        # Handle backbone.block.backbone prefix (from vizard models)
        new_key = new_key.replace("backbone.block.backbone.", "")
        # Handle encoder.model prefix (from metanext models)
        new_key = new_key.replace("encoder.model.", "")

        new_state_dict[new_key] = value

    return new_state_dict
