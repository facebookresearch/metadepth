# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the FAIR Noncommercial Research License. See LICENSE file for details.


"""
Depth Anything 2 (DA2) depth estimation models.

Includes:
- DINOv2 ViT + DPT decoder (dpt.py)

Also re-exports CNN encoder components for convenience.
"""

from ..cnn import (
    build_cnn_encoder,
    CNN_Large,
    CNNEncoderArch,
    CNNEncoderBase,
    ShapeSpec,
)
from .dpt import (
    build_dinov2,
    DepthAnything,
    DPT_MODEL_CONFIGS,
    DPTHead,
    HyDenDepthAnything,
)


__all__ = [
    # CNN encoder (re-exported)
    "build_cnn_encoder",
    "CNN_Large",
    "CNNEncoderArch",
    "CNNEncoderBase",
    "ShapeSpec",
    # DPT
    "build_dinov2",
    "DPTHead",
    "DepthAnything",
    "HyDenDepthAnything",
    "DPT_MODEL_CONFIGS",
]
