'''
  Numpy (accelerated with numba)-only utilities for generating
  modeled stroke data, used to feed a stroke model.
'''

import cv2
import numpy as np
import time
from dataclasses import dataclass
import mixbox
import imageio
from scipy.interpolate import BSpline, make_interp_spline
import matplotlib.pyplot as plt
from numba import njit
from mixbox_numba import lerp_float

from oil_paint_mixing_simple import submix_power
from stroke_sampling import make_random_spline

submix, b2p, p2b = submix_power(np.array([13, 3, 7.]))

@njit
def transform_to_image_coordinates(pts: np.ndarray, im_size: np.ndarray) -> np.ndarray:
  '''
    Transform Nx3 float coordinates on [-1, 1] to image coordinates.
  '''
  pts[:, :2] = ((pts[:, :2] + 1.) / 2. *
                np.array([im_size[0], im_size[1]])).astype(np.int32)
  pts[:, 0] = np.clip(pts[:, 0], 0, im_size[0] - 1)
  pts[:, 1] = np.clip(pts[:, 1], 0, im_size[1] - 1)
  return pts


@njit
def draw_brushstroke_from_pts(img: np.ndarray, pts, color: np.ndarray, brush: np.ndarray, brush_opacity: float, interp_type: str = None):
  img_size = np.array(img.shape, dtype=np.int32)
  pts = transform_to_image_coordinates(pts, img_size)

  # Accumulated alpha values
  brush_mask = np.zeros((img_size[0], img_size[1]))
  brush_size = np.array(brush.shape, dtype=np.int32)

  for pt in pts:
    # Blit brush at this location of the image. We should have rejection sampled so we never go over bounds.
    scale = pt[2]
    lb = (pt[:2] - scale * brush_size[0] / 2).astype(np.int32)
    ub = (pt[:2] + scale * brush_size[1] / 2).astype(np.int32)

    for x in range(lb[0], ub[0]):
      if x < 0 or x >= img_size[0]:
        continue
      for y in range(lb[1], ub[1]):
        if y < 0 or y >= img_size[1]:
          continue
        brush_xy = (
            brush_size * (np.array([x, y], dtype=np.int32) - lb)) / (ub - lb).astype(np.int32)
        brush_x = int(brush_xy[0])
        brush_y = int(brush_xy[1])
        brush_mask[x, y] += brush[brush_x, brush_y]

  brush_mask = np.clip(brush_mask, 0., 1.) * brush_opacity
  if interp_type == "simple_oil":
    print("Suppressed for faster JIT")
    # brush_color_oil = b2p(color)
    # img_oil = b2p(img)
    # for k in range(3):
    #   img_oil[:, :, k] = img_oil[:, :, k] * \
    #       (1. - brush_mask) + brush_mask * brush_color_oil[k]
    # img = p2b(img_oil)
  elif interp_type == "mixbox":
    print("Suppressed for faster JIT")
    # for x in range(img_size[0]):
    #   for y in range(img_size[1]):
    #     img[x, y, :] = lerp_float(img[x, y, :], color, brush_mask[x, y])
  else:
    for k in range(3):
      img[:, :, k] = img[:, :, k] * (1. - brush_mask) + brush_mask * color[k]
  return img


def draw_brushstroke(img: np.ndarray, spline: BSpline, color: np.ndarray, N_samples: int, brush: np.ndarray, brush_opacity: float, interp_type: str = "naive"):
  pts = spline(np.linspace(0., 1., N_samples))
  return draw_brushstroke_from_pts(img=img, pts=pts, color=color, brush=brush, brush_opacity=brush_opacity, interp_type=interp_type)


if __name__ == "__main__":
  # Test demonstration: load up a brush and draw many paint strokes overlaid with each other.
  brush = imageio.imread("tests/data/test_brush.png")
  brush = brush[:, :, 3] / 255.
  BRUSH_SIZE = np.array([32, 32], dtype=np.int32)  # This needs to be even
  brush = cv2.resize(brush, BRUSH_SIZE,
                    interpolation=cv2.INTER_LINEAR)

  IM_HEIGHT = 128
  IM_WIDTH = 128
  img = np.full((IM_HEIGHT, IM_WIDTH, 3), 1.)


  times = []
  for k in range(100):
    bgr = np.random.uniform(0., 1., size=3)
    start = time.time()
    spline = make_random_spline()
    img = draw_brushstroke(img, spline, bgr, N_samples=100, brush=brush, brush_opacity=0.8)
    times.append(time.time() - start)
    cv2.imshow("image", (img*255).astype(np.uint8))
    print(
        f"First time {times[0]}, , total time after JIT {np.sum(times[1:])}, average time {np.mean(times[1:])}")
    cv2.waitKey(1)