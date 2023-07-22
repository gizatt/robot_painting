import pytest
import os
import numpy as np
from robot_painting.models.stroke_model import BlittingStrokeModel
import torch
from torchcubicspline import natural_cubic_spline_coeffs
import matplotlib.pyplot as plt
from robot_painting.models.sprite_blitting import torch_images_to_numpy


def test_blitting_stroke_model():
    model = BlittingStrokeModel(
        brush_path=os.path.join(os.path.split(
            __file__)[0], "data", "test_brush.png")
    )
    images = torch.ones((1, 3, 128, 128))
    trajectories = torch.tensor([[
        [0., 1., 2., 3.],
        [50., 20., 30., 40.],
        [50., 20., 30., 40.],
        [1., 0.5, 0., -1.]
    ]])
    colors = torch.tensor([
        [1., 0., 0., 0.5]
    ])
    out_images = model.forward(images, trajectories, colors)
    assert out_images.shape == images.shape
    assert torch.min(out_images) < 0.9


def test_stroke_viz(show: bool = False):
    model = BlittingStrokeModel(
        brush_path=os.path.join(os.path.split(
            __file__)[0], "data", "test_brush.png")
    )
    image = torch.ones((3, 512, 512))

    # Create a couple of interesting strokes.

    # Straight line
    trajectories = []
    colors = []

    # Straight line
    N = 21
    trajectories.append(
        torch.stack([
            torch.linspace(0, 10, N),
            torch.linspace(100, 400, N),
            torch.full((N,), 400),
            torch.linspace(0., 0.1, N)
        ], axis=0)
    )
    colors.append(torch.tensor(
        [1., 0., 0., 1.]
    ))

    # Circle
    N = 21
    ts = torch.linspace(0, 10, N)
    trajectories.append(
        torch.stack([
            ts,
            torch.cos(ts) * 50 + 200,
            torch.sin(ts) * 50 + 200,
            torch.cos(ts) * 0.01 + 0.05
        ], axis=0)
    )
    colors.append(torch.tensor(
        [0., 0., 1., 0.5]
    ))

    out_images = model.forward(image.unsqueeze(0).repeat(len(
        trajectories), 1, 1, 1), torch.stack(trajectories, dim=0), torch.stack(colors, dim=0))
    total_out, _ = torch.min(out_images, dim=0)

    print(total_out)

    if show:
        plt.imshow(
            torch_images_to_numpy(total_out)
        )
        plt.show()


if __name__ == "__main__":
    # If (and only if) we directly run this file, we'll show the interesting stroke viz results.
    test_stroke_viz(show=True)
