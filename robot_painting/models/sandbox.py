import cv2
import numpy as np
import time
import mixbox
import imageio
from scipy.interpolate import BSpline, make_interp_spline
import matplotlib.pyplot as plt
from numba import njit
from mixbox_numba import lerp_float

# Generate random strokes, as splines that start at the origin towards (+x, 0.)
# and curve randomly.
# The image is in unit bounds.

N_SPLINE_POINTS = 5
MAX_SPLINE_CURVE = np.deg2rad(60)
MAX_SPLINE_DISTANCE = 0.4
MIN_SPLINE_DISTANCE = MAX_SPLINE_DISTANCE / 2.
STROKE_START_XY = np.array([-0.5, 0.])
MIN_SPLINE_HEIGHT = 0.4
MAX_SPLINE_HEIGHT = 1.0
MAX_SPLINE_HEIGHT_STEP = 0.3
SPLINE_T = np.linspace(0., 1., N_SPLINE_POINTS, endpoint=True)


def make_random_spline(bounds_xy=np.array([-0.9, 0.9])) -> BSpline:
  '''
    Generates an Nx3 array of knot points. Rejection samples a spline that stays in the supplied bounds.
  '''
  while 1:
    angles = np.random.uniform(-MAX_SPLINE_CURVE,
                               MAX_SPLINE_CURVE, size=N_SPLINE_POINTS)
    distances = np.random.uniform(
        MIN_SPLINE_DISTANCE, MAX_SPLINE_DISTANCE, size=N_SPLINE_POINTS)
    heading = np.cumsum(angles)
    offsets = np.stack(
        [np.cos(heading), np.sin(heading)], axis=0
    ) * distances
    offsets[:2, 0] = STROKE_START_XY # Force all strokes to start at target location
    pts = np.cumsum(offsets, axis=1).T
    if np.min(pts) <= bounds_xy[0] or np.max(pts) >= bounds_xy[1]:
      continue
    # We're good on XY, so generate out heights in a single pass.
    height_steps = np.random.uniform(-MAX_SPLINE_HEIGHT_STEP,
                                     MAX_SPLINE_HEIGHT_STEP, size=N_SPLINE_POINTS - 1)
    heights = np.zeros(N_SPLINE_POINTS)
    heights[0] = np.random.uniform(MIN_SPLINE_HEIGHT, MAX_SPLINE_HEIGHT)
    for k, height_step in enumerate(height_steps):
      heights[k+1] = np.clip(
          heights[k] + height_step,
          MIN_SPLINE_HEIGHT, MAX_SPLINE_HEIGHT
      )
    pts = np.c_[pts, heights]
    return make_interp_spline(SPLINE_T, pts, k=3, bc_type="clamped")


def submix_power(i):
    # From https://github.com/ctmakro/opencv_playground/blob/master/colormixer.py
    i = np.array(i)

    @njit
    def BGR2PWR(c):
        c = np.clip(c, a_max=1.-1e-6, a_min=1e-6)  # no overflow allowed
        c = np.power(c, 2.2/i)
        u = 1. - c
        return u  # unit absorbance

    @njit
    def PWR2BGR(u):
        c = 1. - u
        c = np.power(c, i/2.2)
        return c  # rgb color

    @njit
    def submix(c1, c2, ratio):
        uabs1, uabs2 = BGR2PWR(c1), BGR2PWR(c2)
        mixuabs = (uabs1 * ratio) + (uabs2*(1-ratio))
        return PWR2BGR(mixuabs)

    return submix, BGR2PWR, PWR2BGR


submix, b2p, p2b = submix_power(np.array([13, 3, 7.]))

brush = imageio.imread("tests/data/test_brush.png")
brush = brush[:, :, 3] / 255.
BRUSH_SIZE = np.array([32, 32], dtype=np.int32)  # This needs to be even
brush = cv2.resize(brush, BRUSH_SIZE,
                   interpolation=cv2.INTER_LINEAR)

IM_HEIGHT = 512
IM_WIDTH = 512
img = np.full((IM_HEIGHT, IM_WIDTH, 3), 1.)


@njit
def transform_to_image_coordinates(pts: np.ndarray) -> np.ndarray:
  '''
    Transform Nx3 float coordinates on [-1, 1] to image coordinates.
  '''
  pts[:, :2] = ((pts[:, :2] + 1.) / 2. *
                np.array([IM_WIDTH, IM_HEIGHT])).astype(np.int32)
  pts[:, 0] = np.clip(pts[:, 0], 0, IM_WIDTH - 1)
  pts[:, 1] = np.clip(pts[:, 1], 0, IM_HEIGHT - 1)
  return pts


@njit
def draw_brushstroke_from_pts(img: np.ndarray, pts, color: np.ndarray, brush_opacity: float, interp_type: str = None):
  pts = transform_to_image_coordinates(pts)

  # Accumulated alpha values
  brush_mask = np.zeros((img.shape[0], img.shape[1]))

  for pt in pts:
    # Blit brush at this location of the image. We should have rejection sampled so we never go over bounds.
    scale = pt[2]
    lb = (pt[:2] - scale * BRUSH_SIZE / 2).astype(np.int32)
    ub = (pt[:2] + scale * BRUSH_SIZE / 2).astype(np.int32)

    for x in range(lb[0], ub[0]):
      if x < 0 or x >= IM_WIDTH:
        continue
      for y in range(lb[1], ub[1]):
        if y < 0 or y >= IM_HEIGHT:
          continue
        brush_xy = (
            BRUSH_SIZE * (np.array([x, y], dtype=np.int32) - lb)) / (ub - lb).astype(np.int32)
        brush_x = int(brush_xy[0])
        brush_y = int(brush_xy[1])
        brush_mask[x, y] += brush[brush_x, brush_y]

  brush_mask = np.clip(brush_mask, 0., 1.) * brush_opacity
  if interp_type == "simple_oil":
    brush_color_oil = b2p(color)
    img_oil = b2p(img)
    for k in range(3):
      img_oil[:, :, k] = img_oil[:, :, k] * \
          (1. - brush_mask) + brush_mask * brush_color_oil[k]
    img = p2b(img_oil)
  elif interp_type == "mixbox":
    for x in range(IM_WIDTH):
      for y in range(IM_HEIGHT):
        img[x, y, :] = lerp_float(img[x, y, :], color, brush_mask[x, y])
  else:
    for k in range(3):
      img[:, :, k] = img[:, :, k] * (1. - brush_mask) + brush_mask * color[k]
  return img


def draw_brushstroke(img: np.ndarray, spline: BSpline, color: np.ndarray, N_samples: int, brush_opacity: float, interp_type: str = "naive"):
  pts = spline(np.linspace(0., 1., N_samples))
  return draw_brushstroke_from_pts(img=img, pts=pts, color=color, brush_opacity=brush_opacity, interp_type=interp_type)

times = []
for k in range(100):
  bgr = np.random.uniform(0., 1., size=3)
  start = time.time()
  spline = make_random_spline()
  img = draw_brushstroke(img, spline, bgr, N_samples=100, brush_opacity=0.8)
  times.append(time.time() - start)
  cv2.imshow("image", (img*255).astype(np.uint8))
  print(
      f"First time {times[0]}, , total time after JIT {np.sum(times[1:])}, average time {np.mean(times[1:])}")
  cv2.waitKey(1)