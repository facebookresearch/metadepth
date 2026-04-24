# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the FAIR Noncommercial Research License. See LICENSE file for details.

"""
MetaDepth / HyDen — Load and Inference

Demonstrates how to load and run inference with the three HyDen model variants:
  1. HyDen-DA2          — Relative depth estimation
  2. HyDen-MoGeV2-MetricPoint  — Metric 3D point map estimation
  3. HyDen-MoGeV2-SurfaceNormal — Surface normal prediction

IMPORTANT — Image normalization differs between model families:
  - DA2 does NOT apply ImageNet normalization internally.
    You must normalize the input with ImageNet mean/std before the forward pass.
  - MoGeV2 applies ImageNet normalization inside DINOv2Encoder,
    so you should pass images in [0, 1] range without additional normalization.
"""

import torch
from da2 import HyDenDepthAnything
from mogev2 import HyDenMoGe, MODEL_CONFIGS
from PIL import Image
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NOTE: DA2 does NOT apply ImageNet normalization internally, so we must
# normalize here. The model expects (B, 3, H, W) with ImageNet-normalized values.
da2_transform = transforms.Compose(
    [
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# NOTE: MoGeV2 applies ImageNet normalization internally in DINOv2Encoder,
# so just pass [0, 1] range images — do NOT normalize again.
mogev2_transform = transforms.Compose(
    [
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
    ]
)

raw_image = Image.open("your/image/path.jpg").convert("RGB")


# ===========================
#  1. HyDen-DA2 (Relative Depth)
# ===========================

da2_model = HyDenDepthAnything(encoder="vitl")
da2_model.load_state_dict(
    torch.load("checkpoints/hyden_da2_vitl.pth", map_location="cpu", weights_only=True)
)
da2_model = da2_model.to(DEVICE).eval()

da2_input = da2_transform(raw_image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    depth = da2_model(da2_input)  # (1, H, W) relative depth map

print(f"[HyDen-DA2] depth shape: {depth.shape}")


# ===========================
#  2. HyDen-MoGeV2 — Metric Point Map
# ===========================

metric_point_model = HyDenMoGe(**MODEL_CONFIGS["vitl_dinov2"])
metric_point_model.load_state_dict(
    torch.load(
        "checkpoints/hyden_mogev2_metric_point_vitl.pth",
        map_location="cpu",
        weights_only=True,
    )
)
metric_point_model = metric_point_model.to(DEVICE).eval()

mogev2_input = mogev2_transform(raw_image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = metric_point_model(mogev2_input)

points = output["points"]  # (1, H, W, 3)
print(f"[HyDen-MoGeV2-MetricPoint] points shape: {points.shape}")


# ===========================
#  3. HyDen-MoGeV2 — Surface Normal
# ===========================

surface_normal_model = HyDenMoGe(**MODEL_CONFIGS["vitl_dinov2"])
surface_normal_model.load_state_dict(
    torch.load(
        "checkpoints/hyden_mogev2_surface_normal_vitl.pth",
        map_location="cpu",
        weights_only=True,
    )
)
surface_normal_model = surface_normal_model.to(DEVICE).eval()

sn_input = mogev2_transform(raw_image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = surface_normal_model(sn_input)

normals = output["normal"]  # (1, H, W, 3)
print(f"[HyDen-MoGeV2-SurfaceNormal] normals shape: {normals.shape}")
