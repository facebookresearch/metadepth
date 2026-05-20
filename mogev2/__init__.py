# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the FAIR Noncommercial Research License. See LICENSE file for details.


"""
MoGe V2 (Monocular Geometry Estimation) models.

Includes:
- MoGeModel: Complete MoGe V2 model
- DINOv2Encoder: DINOv2 encoder for MoGe
- ConvStack, MLP: Decoder components
- Geometry utilities: normalized_view_plane_uv, recover_focal_shift, etc.

Example:
    >>> from mogev2 import (
    ...     MoGeModel,
    ...     MODEL_CONFIGS,
    ... )
    >>> model = MoGeModel(**MODEL_CONFIGS["vitl_dinov2"])
    >>> output = model(image)  # image: (B, 3, H, W) in [0, 1]
    >>> points = output["points"]  # (B, H, W, 3)
"""

# Modules
from .modules import ConvStack, DINOv2Encoder, MLP, Resampler, ResidualConvBlock

# Main model
from .moge_v2 import HyDenMoGe, MODEL_CONFIGS, MoGeModel

# Utilities
from .utils import (
    depth_to_points,
    intrinsics_from_focal_center,
    normalized_view_plane_uv,
    recover_focal_shift,
    solve_optimal_focal_shift,
    solve_optimal_shift,
    weighted_mean,
)


__all__ = [
    # Main model
    "MoGeModel",
    "HyDenMoGe",
    "MODEL_CONFIGS",
    # Modules
    "DINOv2Encoder",
    "ConvStack",
    "MLP",
    "Resampler",
    "ResidualConvBlock",
    # Utilities
    "normalized_view_plane_uv",
    "recover_focal_shift",
    "intrinsics_from_focal_center",
    "depth_to_points",
    "solve_optimal_focal_shift",
    "solve_optimal_shift",
    "weighted_mean",
]
