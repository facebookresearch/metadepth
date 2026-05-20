# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the FAIR Noncommercial Research License. See LICENSE file for details.


"""
Geometry and utility functions for MoGe.
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


__all__ = [
    "normalized_view_plane_uv",
    "recover_focal_shift",
    "intrinsics_from_focal_center",
    "depth_to_points",
    "solve_optimal_focal_shift",
    "solve_optimal_shift",
    "weighted_mean",
]


def weighted_mean(
    x: torch.Tensor,
    w: Optional[torch.Tensor] = None,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute weighted mean of a tensor."""
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(
            dim=dim, keepdim=keepdim
        ).add(eps)


def normalized_view_plane_uv(
    width: int,
    height: int,
    aspect_ratio: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create normalized UV coordinates for a view plane.

    UV with left-top corner as (-width / diagonal, -height / diagonal)
    and right-bottom corner as (width / diagonal, height / diagonal).

    Args:
        width: Image width
        height: Image height
        aspect_ratio: Width/height ratio (computed if None)
        dtype: Output dtype
        device: Output device

    Returns:
        UV coordinates tensor of shape (H, W, 2)
    """
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_y = 1 / (1 + aspect_ratio**2) ** 0.5

    u = torch.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        width,
        dtype=dtype,
        device=device,
    )
    v = torch.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        height,
        dtype=dtype,
        device=device,
    )
    u, v = torch.meshgrid(u, v, indexing="xy")
    uv = torch.stack([u, v], dim=-1)
    return uv


def intrinsics_from_focal_center(
    fx: Union[float, torch.Tensor],
    fy: Union[float, torch.Tensor],
    cx: Union[float, torch.Tensor],
    cy: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Create camera intrinsics matrix from focal lengths and principal point.

    Args:
        fx: Focal length in x direction
        fy: Focal length in y direction
        cx: Principal point x coordinate (normalized to [0, 1])
        cy: Principal point y coordinate (normalized to [0, 1])

    Returns:
        Intrinsics matrix of shape (B, 3, 3) or (3, 3)
    """
    if isinstance(fx, torch.Tensor):
        B = fx.shape[0]
        device = fx.device
        dtype = fx.dtype
    else:
        B = 1
        device = "cpu"
        dtype = torch.float32
        fx = torch.tensor([fx], device=device, dtype=dtype)
        fy = torch.tensor([fy], device=device, dtype=dtype)
        cx = torch.tensor([cx], device=device, dtype=dtype)
        cy = torch.tensor([cy], device=device, dtype=dtype)

    intrinsics = torch.zeros(B, 3, 3, device=device, dtype=dtype)
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy
    intrinsics[:, 2, 2] = 1.0
    return intrinsics


def depth_to_points(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """
    Convert depth map to 3D points using camera intrinsics.

    Args:
        depth: Depth map of shape (B, H, W)
        intrinsics: Camera intrinsics of shape (B, 3, 3)

    Returns:
        3D points of shape (B, H, W, 3)
    """
    B, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    # Create normalized pixel grid [0, 1]
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    x = x / W
    y = y / H

    # Extract intrinsics
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    # Unproject to 3D
    X = (x[None] - cx[:, None, None]) * depth / fx[:, None, None]
    Y = (y[None] - cy[:, None, None]) * depth / fy[:, None, None]
    Z = depth

    points = torch.stack([X, Y, Z], dim=-1)
    return points


def solve_optimal_shift(
    uv: np.ndarray,
    points: np.ndarray,
    focal: float,
) -> float:
    """
    Solve for optimal z-shift given fixed focal length.

    Args:
        uv: UV coordinates of shape (N, 2) or (H, W, 2)
        points: 3D points of shape (N, 3) or (H, W, 3)
        focal: Fixed focal length

    Returns:
        Optimal z-shift value
    """
    uv = uv.reshape(-1, 2)
    points = points.reshape(-1, 3)

    xy = points[:, :2]
    z = points[:, 2]

    # Handle division by zero
    uv_norm = np.linalg.norm(uv, axis=-1, keepdims=True)
    valid = uv_norm.squeeze() > 1e-6

    if valid.sum() < 2:
        return 0.0

    target_z = focal * np.linalg.norm(xy[valid], axis=-1) / uv_norm[valid].squeeze()
    shift = float(np.median(target_z - z[valid]))

    return shift


def solve_optimal_focal_shift(
    uv: np.ndarray,
    points: np.ndarray,
) -> Tuple[float, float]:
    """
    Solve for optimal focal length and z-shift jointly.

    Args:
        uv: UV coordinates of shape (N, 2) or (H, W, 2)
        points: 3D points of shape (N, 3) or (H, W, 3)

    Returns:
        Tuple of (optimal z-shift, optimal focal length)
    """
    uv = uv.reshape(-1, 2)
    points = points.reshape(-1, 3)

    xy = points[:, :2]
    z = points[:, 2]

    A = np.stack([xy, -uv], axis=-1).reshape(-1, 2)
    b = (uv * z[:, None]).reshape(-1)

    # Least squares solution
    try:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        focal = float(result[0])
        shift = float(result[1])
    except np.linalg.LinAlgError:
        focal = 1.0
        shift = 0.0

    # Ensure focal is positive
    if focal <= 0:
        focal = 1.0

    return shift, focal


def recover_focal_shift(
    points: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    focal: Optional[torch.Tensor] = None,
    downsample_size: Tuple[int, int] = (64, 64),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.

    Note that it assumes:
    - the optical center is at the center of the map
    - the map is undistorted
    - the map is isometric in the x and y directions

    Args:
        points: Point map of shape (..., H, W, 3)
        mask: Optional validity mask of shape (..., H, W)
        focal: Optional fixed focal length of shape (...)
        downsample_size: Size for downsampled computation

    Returns:
        Tuple of (focal, shift) tensors
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(
        width, height, dtype=points.dtype, device=points.device
    )

    points_lr = F.interpolate(
        points.permute(0, 3, 1, 2), downsample_size, mode="nearest"
    ).permute(0, 2, 3, 1)
    uv_lr = (
        F.interpolate(
            uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode="nearest"
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )
    mask_lr = (
        None
        if mask is None
        else F.interpolate(
            mask.to(torch.float32).unsqueeze(1), downsample_size, mode="nearest"
        ).squeeze(1)
        > 0
    )

    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_lr_np = None if mask_lr is None else mask_lr.cpu().numpy()

    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        points_lr_i_np = (
            points_lr_np[i] if mask_lr_np is None else points_lr_np[i][mask_lr_np[i]]
        )
        uv_lr_i_np = uv_lr_np if mask_lr_np is None else uv_lr_np[mask_lr_np[i]]

        if uv_lr_i_np.shape[0] < 2:
            optim_focal.append(1.0)
            optim_shift.append(0.0)
            continue

        if focal is None:
            optim_shift_i, optim_focal_i = solve_optimal_focal_shift(
                uv_lr_i_np, points_lr_i_np
            )
            optim_focal.append(float(optim_focal_i))
        else:
            assert focal_np is not None
            optim_shift_i = solve_optimal_shift(uv_lr_i_np, points_lr_i_np, focal_np[i])
        optim_shift.append(float(optim_shift_i))

    optim_shift = torch.tensor(
        optim_shift, device=points.device, dtype=points.dtype
    ).reshape(shape[:-3])

    if focal is None:
        optim_focal = torch.tensor(
            optim_focal, device=points.device, dtype=points.dtype
        ).reshape(shape[:-3])
    else:
        optim_focal = focal.reshape(shape[:-3])

    return optim_focal, optim_shift
