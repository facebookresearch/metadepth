# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the FAIR Noncommercial Research License. See LICENSE file for details.


"""
Standalone MoGe V2 Monocular Geometry Estimation model.

This module provides a complete, self-contained implementation of MoGe V2.
It predicts:
- 3D point maps (metric depth + camera intrinsics)
- Surface normals
- Mask predictions
- Metric scale estimation

Usage:
    >>> from mogev2 import MoGeModel, MODEL_CONFIGS
    >>> model = MoGeModel(**MODEL_CONFIGS["vitl_dinov2"])
    >>> output = model(image)  # image: (B, 3, H, W)
    >>> points = output["points"]  # (B, H, W, 3)
"""

import copy
import logging
import warnings
from numbers import Number
from pathlib import Path
from typing import Any, Dict, IO, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ConvStack, DINOv2Encoder, MLP
from .utils import (
    depth_to_points,
    intrinsics_from_focal_center,
    normalized_view_plane_uv,
    recover_focal_shift,
)

logger = logging.getLogger(__name__)


__all__ = [
    "MoGeModel",
    "MODEL_CONFIGS",
]

# CNN encoder channel dimensions (sum of all 5 stages: 32+32+64+160+288)
_CNN_TOTAL_CHANNELS = 576
# Last CNN encoder stage channels
_CNN_LAST_CHANNEL = 288


MODEL_CONFIGS = {
    "vitl_dinov2": {
        "encoder": {
            "backbone": "vitl_dinov2",
            "intermediate_layers": [5, 11, 17, 23],
            "dim_out": 1024,
        },
        "neck": {
            "dim_in": [1026, 2, 2, 2, 2],
            "dim_out": None,
            "dim_res_blocks": [1024, 256, 128, 64, 32],
            "num_res_blocks": [0, 2, 2, 2, 0],
            "res_block_in_norm": "none",
            "res_block_hidden_norm": "none",
            "resamplers": [
                "conv_transpose",
                "conv_transpose",
                "conv_transpose",
                "bilinear",
            ],
        },
        "points_head": {
            "dim_in": [1024, 256, 128, 64, 32],
            "dim_out": [None, None, None, None, 3],
            "dim_res_blocks": [1024, 256, 128, 64, 32],
            "num_res_blocks": [0, 1, 1, 1, 0],
            "res_block_in_norm": "none",
            "res_block_hidden_norm": "none",
            "resamplers": [
                "conv_transpose",
                "conv_transpose",
                "conv_transpose",
                "bilinear",
            ],
        },
        "mask_head": {
            "dim_in": [1024, 256, 128, 64, 32],
            "dim_out": [None, None, None, None, 1],
            "dim_res_blocks": [1024, 256, 128, 64, 32],
            "num_res_blocks": [0, 1, 1, 1, 0],
            "res_block_in_norm": "none",
            "res_block_hidden_norm": "none",
            "resamplers": [
                "conv_transpose",
                "conv_transpose",
                "conv_transpose",
                "bilinear",
            ],
        },
        "scale_head": {"dims": [1024, 1024, 1024, 1]},
        "normal_head": {
            "dim_in": [1024, 256, 128, 64, 32],
            "dim_out": [None, None, None, None, 3],
            "dim_res_blocks": [1024, 256, 128, 64, 32],
            "num_res_blocks": [0, 1, 1, 1, 0],
            "res_block_in_norm": "none",
            "res_block_hidden_norm": "none",
            "resamplers": [
                "conv_transpose",
                "conv_transpose",
                "conv_transpose",
                "bilinear",
            ],
        },
        "remap_output": "exp",
        "num_tokens_range": [1200, 3600],
    },
}


class MoGeModel(nn.Module):
    """
    MoGe V2 Monocular Geometry Estimation model.

    This is a standalone implementation that combines:
    - DINOv2 encoder for feature extraction
    - Multi-scale ConvStack decoder (neck)
    - Task-specific heads (points, normals, mask, scale)

    Example:
        >>> model = MoGeModel(**MODEL_CONFIGS["vitl_dinov2"])
        >>> output = model(image)  # image: (B, 3, H, W) in [0, 1]
        >>> points = output["points"]  # (B, H, W, 3)
        >>> normal = output["normal"]  # (B, H, W, 3) if available
    """

    def __init__(
        self,
        encoder: Dict[str, Any],
        neck: Dict[str, Any],
        points_head: Optional[Dict[str, Any]] = None,
        mask_head: Optional[Dict[str, Any]] = None,
        normal_head: Optional[Dict[str, Any]] = None,
        scale_head: Optional[Dict[str, Any]] = None,
        remap_output: Literal["linear", "sinh", "exp", "sinh_exp"] = "linear",
        num_tokens_range: Optional[List[int]] = None,
        adaptive_tokens_mode: Literal[
            "fixed_range", "image_size_based"
        ] = "image_size_based",
        default_patch_size: int = 14,
        vit_internal_resolution: Optional[Tuple[int, int]] = None,
        add_cnn_encoder: str = "",
        **deprecated_kwargs,
    ) -> None:
        """
        Args:
            add_cnn_encoder: CNN encoder architecture for HyDen mode (e.g., "large")
        """
        super().__init__()

        if num_tokens_range is None:
            num_tokens_range = [1200, 3600]

        if deprecated_kwargs:
            warnings.warn(
                f"The following deprecated/invalid arguments are ignored: {deprecated_kwargs}",
                stacklevel=2,
            )

        self.remap_output = remap_output
        self.num_tokens_range = num_tokens_range
        self.adaptive_tokens_mode = adaptive_tokens_mode
        self.default_patch_size = default_patch_size
        self.vit_internal_resolution = vit_internal_resolution
        self.encoder_name = encoder["backbone"]

        logger.info(
            f"[MoGeModel] Building with ADAPTIVE_TOKENS_MODE: {self.adaptive_tokens_mode}"
        )

        # Build encoder
        self.encoder = DINOv2Encoder(**encoder)

        self.add_cnn_encoder = add_cnn_encoder
        if add_cnn_encoder:
            from ..cnn import build_cnn_encoder

            self.cnn_encoder = build_cnn_encoder(arch=add_cnn_encoder)

            vit_channels = encoder["dim_out"]  # e.g., 1024 for vitl

            # Fusion layer: concat CNN + ViT features, reduce to ViT channels
            self.image_feats = nn.Sequential(
                nn.Conv2d(
                    in_channels=_CNN_TOTAL_CHANNELS + vit_channels,
                    out_channels=_CNN_TOTAL_CHANNELS + vit_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=_CNN_TOTAL_CHANNELS + vit_channels,
                    out_channels=vit_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )

            # Scale fusion: only if we have a points head (need metric scale)
            if points_head is not None:
                self.scale_feats = nn.Sequential(
                    nn.Linear(
                        _CNN_LAST_CHANNEL + vit_channels,
                        _CNN_LAST_CHANNEL + vit_channels,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Linear(_CNN_LAST_CHANNEL + vit_channels, vit_channels),
                )
            else:
                self.scale_feats = None
        else:
            self.add_cnn_encoder = ""

        # Build neck
        self.neck = ConvStack(**neck)

        # Build task heads
        if points_head is not None:
            self.points_head = ConvStack(**points_head)
        if mask_head is not None:
            self.mask_head = ConvStack(**mask_head)
        if normal_head is not None:
            self.normal_head = ConvStack(**normal_head)
        if scale_head is not None:
            self.scale_head = MLP(**scale_head)

        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(f"[MoGeModel] Total parameters: {total_params:.2f}M")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Union[str, Path, IO[bytes]],
        config_name: str = "vitl_dinov2",
        train_pointmap: bool = True,
        train_normal: bool = True,
        vit_internal_resolution: Optional[Tuple[int, int]] = (518, 518),
        add_cnn_encoder: str = "",
    ) -> "MoGeModel":
        """
        Load a MoGe model from a checkpoint file.

        Args:
            pretrained_model_path: Path to .pth checkpoint file
            config_name: Model configuration name from MODEL_CONFIGS
            train_pointmap: Whether to include points/mask/scale heads
            train_normal: Whether to include normal head
            vit_internal_resolution: Internal resolution for ViT

        Returns:
            Loaded MoGeModel instance
        """
        if config_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown config: {config_name}. "
                f"Available: {list(MODEL_CONFIGS.keys())}"
            )

        model_config = copy.deepcopy(MODEL_CONFIGS[config_name])

        # Remove heads if not needed
        if not train_pointmap:
            model_config.pop("points_head", None)
            model_config.pop("mask_head", None)
            model_config.pop("scale_head", None)
        if not train_normal:
            model_config.pop("normal_head", None)

        model_config["vit_internal_resolution"] = vit_internal_resolution
        model_config["add_cnn_encoder"] = add_cnn_encoder
        model = cls(**model_config)

        # Load checkpoint
        checkpoint_path = str(pretrained_model_path)
        logger.info(f"[MoGeModel] Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif checkpoint and list(checkpoint.keys())[0].startswith("model."):
                state_dict = {k.replace("model.", ""): v for k, v in checkpoint.items()}
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"[MoGeModel] Missing keys: {len(missing)}")
        if unexpected:
            logger.warning(f"[MoGeModel] Unexpected keys: {len(unexpected)}")
        if missing and unexpected and len(missing) > 10:
            raise RuntimeError(
                f"Checkpoint appears incompatible: {len(missing)} missing keys and "
                f"{len(unexpected)} unexpected keys. "
                f"Use an OSS-converted checkpoint."
            )

        return model

    def init_weights(self) -> None:
        """Initialize encoder weights."""
        self.encoder.init_weights()

    def _remap_points(self, points: torch.Tensor) -> torch.Tensor:
        """Remap point predictions based on output type."""
        if self.remap_output == "linear":
            pass
        elif self.remap_output == "sinh":
            points = torch.sinh(points)
        elif self.remap_output == "exp":
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output == "sinh_exp":
            xy, z = points.split([2, 1], dim=-1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Invalid remap output type: {self.remap_output}")
        return points

    def _recover_intrinsics(
        self,
        points: torch.Tensor,
        mask_binary: Optional[torch.Tensor],
        fov_x: Optional[Union[Number, torch.Tensor]],
        aspect_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Recover focal length, shift, intrinsics, and depth from predicted points."""
        focal: torch.Tensor
        shift: torch.Tensor
        if fov_x is None:
            focal, shift = recover_focal_shift(points, mask_binary)
        else:
            fov_tensor = torch.as_tensor(
                fov_x, device=points.device, dtype=points.dtype
            )
            tan_half_fov = torch.tan(torch.deg2rad(fov_tensor / 2))
            aspect_ratio_t = torch.as_tensor(
                aspect_ratio, device=points.device, dtype=points.dtype
            )
            ar_sq_plus_1 = 1 + aspect_ratio_t.pow(2)
            focal = aspect_ratio_t / torch.sqrt(ar_sq_plus_1) / tan_half_fov
            if focal.ndim == 0:
                focal = focal[None].expand(points.shape[0])
            _, shift = recover_focal_shift(points, mask_binary, focal=focal)

        ar_factor = torch.sqrt(
            torch.as_tensor(
                1 + aspect_ratio**2, device=points.device, dtype=points.dtype
            )
        )
        ar_t = torch.as_tensor(aspect_ratio, device=points.device, dtype=points.dtype)
        fx = focal / 2 * ar_factor / ar_t
        fy = focal / 2 * ar_factor
        intrinsics = intrinsics_from_focal_center(fx, fy, 0.5, 0.5)

        points[..., 2] += shift[..., None, None]
        if mask_binary is not None:
            mask_binary = mask_binary & (points[..., 2] > 0)
        depth = points[..., 2].clone()
        return points, depth, intrinsics, mask_binary

    def _apply_mask_to_outputs(
        self,
        points: Optional[torch.Tensor],
        depth: Optional[torch.Tensor],
        normal: Optional[torch.Tensor],
        mask_binary: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Apply binary mask to outputs, setting masked regions to inf/zero."""
        if points is not None:
            points = torch.where(mask_binary[..., None], points, torch.inf)
        if depth is not None:
            depth = torch.where(mask_binary, depth, torch.inf)
        if normal is not None:
            normal = torch.where(
                mask_binary[..., None], normal, torch.zeros_like(normal)
            )
        return points, depth, normal

    def _postprocess_outputs(
        self,
        points: Optional[torch.Tensor],
        normal: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        metric_scale: Optional[torch.Tensor],
        orig_h: int,
        orig_w: int,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Resize outputs to original resolution and apply remapping."""
        points, normal, mask = [
            F.interpolate(v, (orig_h, orig_w), mode="nearest")
            if v is not None
            else None
            for v in [points, normal, mask]
        ]
        if points is not None:
            points = points.permute(0, 2, 3, 1)
            points = self._remap_points(points)
        if normal is not None:
            normal = normal.permute(0, 2, 3, 1)
            normal = F.normalize(normal, dim=-1)
        if mask is not None:
            mask = mask.squeeze(1).sigmoid()
        if metric_scale is not None:
            metric_scale = metric_scale.squeeze(1)
        return points, normal, mask, metric_scale

    def forward(
        self,
        image: torch.Tensor,
        return_features: bool = False,
        training: bool = False,
        return_mask_and_scale: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            image: Input RGB image tensor of shape ``(B, 3, H, W)`` with
                values in ``[0, 1]``.
            return_features: Whether to return intermediate features.
            training: Whether in training mode.
            return_mask_and_scale: Whether to return mask and scale.

        Returns:
            dict with keys (presence depends on model config):
                - ``"points"``: 3-D point map ``(B, H, W, 3)``.
                - ``"normal"``: unit surface normals ``(B, H, W, 3)``.
                - ``"mask"``: confidence mask ``(B, H, W)``
                  (only when ``return_mask_and_scale=True``).
                - ``"metric_scale"``: scalar scale ``(B, 1)``
                  (only when ``return_mask_and_scale=True``).
                - ``"depth"``: depth map ``(B, H, W)``
                  (only when ``return_features=True``).
        """
        if return_features and return_mask_and_scale:
            raise ValueError(
                "Cannot set both return_features and return_mask_and_scale to True"
            )

        batch_size, _, img_h, img_w = image.shape
        device, dtype = image.device, image.dtype
        orig_img_h, orig_img_w = img_h, img_w
        image_cnn = image  # Save full-res image for CNN encoder

        # Resize to internal resolution if specified (for ViT only)
        if self.vit_internal_resolution is not None:
            image = F.interpolate(
                image,
                size=self.vit_internal_resolution,
                mode="bilinear",
                align_corners=True,
            )
            _, _, img_h, img_w = image.shape

        num_patches_h = img_h // self.default_patch_size
        num_patches_w = img_w // self.default_patch_size
        num_tokens = num_patches_h * num_patches_w

        aspect_ratio = img_w / img_h
        base_h = int((num_tokens / aspect_ratio) ** 0.5)
        base_w = int((num_tokens * aspect_ratio) ** 0.5)

        # Encoder forward pass
        features, cls_token, vit_features = self.encoder(
            image, base_h, base_w, return_class_token=True
        )

        # CNN encoder fusion (HyDen mode)
        if self.add_cnn_encoder:
            cnn_out = self.cnn_encoder(image_cnn)  # Use full-res image
            # Compute patch grid from original resolution for CNN features
            num_patches_h_cnn = orig_img_h // self.default_patch_size
            num_patches_w_cnn = orig_img_w // self.default_patch_size
            # Interpolate all CNN stage features to full-res patch grid and concat
            sorted_keys = sorted(cnn_out.keys())
            cnn_features_list = []
            for stage_name in sorted_keys:
                feat = F.interpolate(
                    cnn_out[stage_name],
                    size=(num_patches_h_cnn, num_patches_w_cnn),
                    mode="bilinear",
                    align_corners=False,
                )
                cnn_features_list.append(feat)
            cnn_features = torch.cat(cnn_features_list, dim=1)  # (B, 576, H, W)

            # Fuse scale tokens
            if self.scale_feats is not None:
                last_stage = cnn_out[sorted_keys[-1]]
                cls_token_cnn = (
                    F.avg_pool2d(
                        last_stage,
                        kernel_size=last_stage.shape[2:],
                    )
                    .squeeze(-1)
                    .squeeze(-1)
                )
                cls_token = torch.cat([cls_token, cls_token_cnn], dim=1)
                cls_token = self.scale_feats(cls_token)

            # Fuse image features: upsample ViT features to full-res patch grid
            features = F.interpolate(
                features,
                size=(num_patches_h_cnn, num_patches_w_cnn),
                mode="bilinear",
                align_corners=False,
            )
            features = torch.cat([features, cnn_features], dim=1)
            features = self.image_feats(features)

            base_h, base_w = num_patches_h_cnn, num_patches_w_cnn

        # Build multi-scale features with UV coordinates
        multi_features = [features, None, None, None, None]
        for level in range(5):
            uv = normalized_view_plane_uv(
                width=base_w * 2**level,
                height=base_h * 2**level,
                aspect_ratio=aspect_ratio,
                dtype=dtype,
                device=device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)

            if multi_features[level] is None:
                multi_features[level] = uv
            else:
                if multi_features[level].shape[2:] != uv.shape[2:]:
                    new_feat = F.interpolate(
                        multi_features[level],
                        size=(uv.shape[2], uv.shape[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
                    multi_features[level] = torch.cat([new_feat, uv], dim=1)
                else:
                    multi_features[level] = torch.cat(
                        [multi_features[level], uv], dim=1
                    )

        # Neck forward pass
        features = self.neck(multi_features)

        # Head predictions
        points = (
            self.points_head(features)[-1] if hasattr(self, "points_head") else None
        )
        normal = (
            self.normal_head(features)[-1] if hasattr(self, "normal_head") else None
        )

        if return_features or return_mask_and_scale:
            mask = self.mask_head(features)[-1] if hasattr(self, "mask_head") else None
            metric_scale = (
                self.scale_head(cls_token) if hasattr(self, "scale_head") else None
            )
        else:
            mask = None
            metric_scale = None

        # Resize and remap outputs to original resolution
        points, normal, mask, metric_scale = self._postprocess_outputs(
            points, normal, mask, metric_scale, orig_img_h, orig_img_w
        )

        return_dict = {
            "points": points,
            "normal": normal,
            "mask": mask,
            "metric_scale": metric_scale,
        }
        if return_features:
            return_dict["feature"] = vit_features
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        return return_dict

    @torch.inference_mode()
    def infer(  # noqa: C901
        self,
        image: torch.Tensor,
        num_tokens: Optional[int] = None,
        resolution_level: int = 9,
        force_projection: bool = True,
        apply_mask: Union[bool, Literal["blend"]] = True,
        fov_x: Optional[Union[Number, torch.Tensor]] = None,
        use_fp16: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        User-friendly inference function with post-processing.

        Args:
            image: Input image (B, 3, H, W) or (3, H, W) in range [0, 1]
            num_tokens: Number of base ViT tokens (None = auto)
            resolution_level: Resolution level 0-9 (if num_tokens is None)
            force_projection: Recompute points from depth + intrinsics
            apply_mask: Apply mask to outputs
            fov_x: Known horizontal FoV in degrees (None = infer)
            use_fp16: Use FP16 for inference

        Returns:
            Dictionary with: points, depth, intrinsics, mask, normal
        """
        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False

        image = image.to(dtype=self.dtype, device=self.device)
        original_height, original_width = image.shape[-2:]
        aspect_ratio = original_width / original_height

        # Determine number of tokens
        if num_tokens is None:
            if self.adaptive_tokens_mode == "image_size_based":
                num_patches_h = original_height // self.default_patch_size
                num_patches_w = original_width // self.default_patch_size
                num_tokens = num_patches_h * num_patches_w
            else:
                assert self.num_tokens_range is not None
                min_tokens, max_tokens = self.num_tokens_range
                num_tokens = int(
                    min_tokens + (resolution_level / 9) * (max_tokens - min_tokens)
                )

        # Forward pass
        if use_fp16:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                output = self.forward(image)
        else:
            output = self.forward(image)

        points = output.get("points")
        normal = output.get("normal")
        mask = output.get("mask")
        metric_scale = output.get("metric_scale")

        # Post-processing
        if mask is not None:
            mask_binary = mask > 0.5
        else:
            mask_binary = None

        if points is not None:
            points, depth, intrinsics, mask_binary = self._recover_intrinsics(
                points, mask_binary, fov_x, aspect_ratio
            )
        else:
            depth, intrinsics = None, None

        # Force projection constraint
        if force_projection and depth is not None and intrinsics is not None:
            points = depth_to_points(depth, intrinsics=intrinsics)

        # Apply metric scale
        if metric_scale is not None:
            if points is not None:
                points = points * metric_scale[:, None, None, None]
            if depth is not None:
                depth = depth * metric_scale[:, None, None]

        # Apply mask
        if apply_mask and mask_binary is not None:
            points, depth, normal = self._apply_mask_to_outputs(
                points, depth, normal, mask_binary
            )

        return_dict = {
            "points": points,
            "intrinsics": intrinsics,
            "depth": depth,
            "mask": mask_binary,
            "normal": normal,
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}

        if omit_batch_dim:
            return_dict = {k: v.squeeze(0) for k, v in return_dict.items()}

        return return_dict


class HyDenMoGe(MoGeModel):
    """HyDen (Hybrid DINOv2 + CNN Encoder) model for monocular geometry estimation.

    This is a convenience wrapper around MoGeModel with the CNN encoder
    pre-configured for HyDen mode.
    """

    def __init__(self, cnn_encoder: str = "large", **kwargs) -> None:
        super().__init__(add_cnn_encoder=cnn_encoder, **kwargs)
