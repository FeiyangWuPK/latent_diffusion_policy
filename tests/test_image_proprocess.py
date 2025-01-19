import pytest
import torch
import torchvision as thv

from latent_diffusion_policy.utils.preprocess import depth_to_pointcloud


def test_depth_to_pointcloud():
    """
    Tests the depth_to_pointcloud function using a small, hand-crafted RGBD image
    and very simple intrinsics. Ensures that outputs match expected values.
    """

    # -----------------------------------------------------------
    # 1. Create a small synthetic RGBD image and intrinsics
    # -----------------------------------------------------------
    # Batch=1, Height=2, Width=2, Channels=4 (R,G,B,Depth)
    B, H, W = 1, 2, 2
    rgbd = torch.zeros(B, 4, H, W)

    # Fill RGB channels with some values:
    # For simplicity, let's keep them constant across all pixels
    rgbd[:, 0, :, :] = 0.1  # R
    rgbd[:, 1, :, :] = 0.2  # G
    rgbd[:, 2, :, :] = 0.3  # B

    # Depth channel with a small pattern:
    # pixel (0,0) -> 0.1, (0,1) -> 0.2, (1,0) -> 0.3, (1,1) -> 0.4
    depth_values = torch.tensor([[0.1, 0.2], [0.3, 0.4]]).unsqueeze(
        0
    )  # shape: (1, 2, 2)
    rgbd[:, 3, :, :] = depth_values

    # Intrinsics: fx=1, fy=1, cx=0, cy=0
    # This simplifies X = x*Z, Y = y*Z, Z=Z
    intrinsics = torch.tensor([1.0, 1.0, 0.0, 0.0])

    # -----------------------------------------------------------
    # 2. Run depth_to_pointcloud
    # -----------------------------------------------------------
    points_3d, colors = depth_to_pointcloud(rgbd, intrinsics)

    # -----------------------------------------------------------
    # 3. Check shapes
    # -----------------------------------------------------------
    # Expected shapes: (1, H*W, 3) => (1, 4, 3) for both points and colors
    assert points_3d.shape == (
        B,
        H * W,
        3,
    ), f"points_3d shape mismatch: {points_3d.shape}"
    assert colors.shape == (B, H * W, 3), f"colors shape mismatch: {colors.shape}"

    # -----------------------------------------------------------
    # 4. Check numerical outputs for points
    # -----------------------------------------------------------
    # Flatten the depth, recall the pixel ordering (row-major):
    # (0,0): Z=0.1, => X=0*0.1=0,   Y=0*0.1=0   => [0.0,   0.0,   0.1]
    # (0,1): Z=0.2, => X=1*0.2=0.2, Y=0*0.2=0   => [0.2,   0.0,   0.2]
    # (1,0): Z=0.3, => X=0*0.3=0,   Y=1*0.3=0.3 => [0.0,   0.3,   0.3]
    # (1,1): Z=0.4, => X=1*0.4=0.4, Y=1*0.4=0.4 => [0.4,   0.4,   0.4]
    expected_points = torch.tensor(
        [[0.0, 0.0, 0.1], [0.2, 0.0, 0.2], [0.0, 0.3, 0.3], [0.4, 0.4, 0.4]],
        dtype=points_3d.dtype,
    )

    # Compare with actual output from the function
    assert torch.allclose(
        points_3d[0], expected_points, atol=1e-6
    ), f"Points mismatch:\nExpected:\n{expected_points}\nGot:\n{points_3d[0]}"

    # -----------------------------------------------------------
    # 5. Check numerical outputs for colors
    # -----------------------------------------------------------
    # Each pixel had R=0.1, G=0.2, B=0.3
    # So the flattened shape is (4, 3) with identical rows.
    expected_colors = torch.tensor(
        [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
        dtype=colors.dtype,
    )

    assert torch.allclose(
        colors[0], expected_colors, atol=1e-6
    ), f"Colors mismatch:\nExpected:\n{expected_colors}\nGot:\n{colors[0]}"

    # If the test reaches here, everything passed
    print("test_depth_to_pointcloud passed!")


if __name__ == "__main__":
    pytest.main([__file__])
