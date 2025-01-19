from typing import Any, Optional, Tuple
import os
import sys

import gymnasium as gym
import gymnasium_robotics
import torch as th
import torchvision as thv
import numpy as np


@th.compile
def depth_to_pointcloud(
    rgbd: th.Tensor, intrinsics: th.Tensor
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Convert an RGBD image to a point cloud using a pinhole camera model.

    Args:
        rgbd (torch.Tensor): Tensor of shape (B, 4, H, W).
                            Channels 0–2 => R, G, B
                            Channel 3 => Depth
        intrinsics (torch.Tensor): Tensor of shape (B, 4) or (4,).
                                   [fx, fy, cx, cy] per batch or shared.

    Returns:
        points_3d (torch.Tensor): 3D points of shape (B, H*W, 3).
        colors (torch.Tensor): RGB colors of shape (B, H*W, 3).
    """
    # ----------------------------------------------------------------
    # 1. Parse shapes and intrinsics
    # ----------------------------------------------------------------
    B, C, H, W = rgbd.shape
    assert C == 4, "rgbd must have 4 channels: R, G, B, Depth"

    # If intrinsics is not per batch but a single set of 4, expand it
    if intrinsics.ndim == 1 and intrinsics.shape[0] == 4:
        intrinsics = intrinsics.unsqueeze(0).expand(B, 4)
    assert intrinsics.shape == (B, 4), "intrinsics must be (B, 4) or (4,)"

    fx = intrinsics[:, 0]  # (B,)
    fy = intrinsics[:, 1]  # (B,)
    cx = intrinsics[:, 2]  # (B,)
    cy = intrinsics[:, 3]  # (B,)

    # ----------------------------------------------------------------
    # 2. Separate RGB and depth
    # ----------------------------------------------------------------
    rgb = rgbd[:, 0:3, ...]  # (B, 3, H, W)
    depth = rgbd[:, 3:4, ...]  # (B, 1, H, W)

    # ----------------------------------------------------------------
    # 3. Create a meshgrid of pixel coordinates
    # ----------------------------------------------------------------
    # y_coords: (H, W), x_coords: (H, W)
    y_coords, x_coords = th.meshgrid(
        th.arange(H, device=rgbd.device, dtype=rgbd.dtype),
        th.arange(W, device=rgbd.device, dtype=rgbd.dtype),
        indexing="ij",
    )
    # Expand to batch dimension and flatten
    x_coords = x_coords.flatten().unsqueeze(0).repeat(B, 1)  # (B, H*W)
    y_coords = y_coords.flatten().unsqueeze(0).repeat(B, 1)  # (B, H*W)

    # Flatten depth
    depth_flat = depth.view(B, -1)  # (B, H*W)

    # ----------------------------------------------------------------
    # 4. Compute 3D coordinates (X, Y, Z)
    # ----------------------------------------------------------------
    # Z = depth
    Z = depth_flat

    # X = (x - cx) * Z / fx
    # Y = (y - cy) * Z / fy
    # We'll need to unsqueeze cx, fx, etc. so they broadcast along the points dimension.
    X = (x_coords - cx.unsqueeze(1)) * Z / fx.unsqueeze(1)
    Y = (y_coords - cy.unsqueeze(1)) * Z / fy.unsqueeze(1)

    points_3d = th.stack([X, Y, Z], dim=-1)  # (B, H*W, 3)

    # ----------------------------------------------------------------
    # 5. Flatten and collect RGB
    # ----------------------------------------------------------------
    # Flatten to (B, H*W, 3)
    colors = rgb.view(B, 3, H * W).permute(0, 2, 1).contiguous()

    return points_3d, colors
