import time
import pytest
import os
import numpy as np
from robot_painting.models.stroke_model import BlittingStrokeModel
import torch
from torchcubicspline import natural_cubic_spline_coeffs
import matplotlib.pyplot as plt
from robot_painting.models.sprite_blitting import torch_images_to_numpy

def test_stroke_viz(show: bool = False):
    model = BlittingStrokeModel(
        brush_path=os.path.join(os.path.split(
            __file__)[0], "data", "test_brush.png")
    )
    image = torch.ones((3, 256, 256))

    # Create a couple of interesting strokes.

    # All trajectories must share common # of knots, I think...
    coeffs = []
    colors = []

    N = 10
    ts = torch.linspace(0, 1, N)
    
    # Straight line
    pts = []
    pts.append(torch.stack([
            torch.linspace(50, 200, N),
            torch.full((N,), 200),
            torch.linspace(0., 0.1, N)
        ], axis=0).T)
    colors.append(torch.tensor(
        [1., 0., 0., 1.]
    ))

    # Circle
    ts = torch.linspace(0, 1, N)
    pts.append(torch.stack([
            torch.cos(ts * np.pi) * 25 + 100,
            torch.sin(ts * np.pi) * 25 + 100,
            torch.cos(ts *  np.pi) * 0.01 + 0.05
        ], axis=0).T)
    colors.append(torch.tensor(
        [0., 0., 1., 0.5]
    ))


    start = time.time()

    coeffs = natural_cubic_spline_coeffs(ts, torch.stack(pts, dim=0))
    
    traj_gen_end = time.time()

    N_ts = 100
    ts = torch.linspace(0, 1, N_ts)
    out_images = model.forward(image.unsqueeze(0).repeat(len(
        colors), 1, 1, 1), coeffs, torch.stack(colors, dim=0), ts=ts)
    total_out, _ = torch.min(out_images, dim=0)
    end = time.time()
    print(f"Elapsed in draw: {traj_gen_end - start} in traj gen, and {end - traj_gen_end} in rendering.")

    if show:
        plt.imshow(
            torch_images_to_numpy(total_out)
        )
        plt.show()


if __name__ == "__main__":
    # If (and only if) we directly run this file, we'll show the interesting stroke viz results.
    test_stroke_viz(show=True)
