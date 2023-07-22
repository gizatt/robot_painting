import pytest
import os
from robot_painting.models.stroke_model import BlittingStrokeModel
import torch
from torchcubicspline import natural_cubic_spline_coeffs


def test_blitting_stroke_model():
    model = BlittingStrokeModel(
        brush_path=os.path.join(os.path.split(__file__)[0], "data", "test_brush.png")
    )

    images = torch.zeros((1, 3, 128, 128))
    trajectories = torch.tensor([[
        [0., 1., 2., 3.],
        [0., 0., 1., 1.],
        [0., -1., 0., 1.],
        [1., 0.5, 0., -1.]
    ]])
    colors = torch.tensor([
        [1., 0., 0., 0.5]
    ])
    out_images = model.forward(images, trajectories, colors)
    assert out_images.shape == images.shape
    assert torch.sum(out_images) > 0.
