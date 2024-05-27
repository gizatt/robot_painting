import numpy as np
from dataclasses import dataclass

from scipy.interpolate import BSpline, make_interp_spline

@dataclass
class SplineSamplingParams():
  N_SPLINE_POINTS: int = 5
  MAX_SPLINE_CURVE: float = np.deg2rad(60)
  MAX_SPLINE_DISTANCE = 0.4
  MIN_SPLINE_DISTANCE = 0.2
  STROKE_START_XY = np.array([-0.5, 0.])
  MIN_SPLINE_HEIGHT = 0.4
  MAX_SPLINE_HEIGHT = 1.0
  MAX_SPLINE_HEIGHT_STEP = 0.3

def make_random_spline_pts(sampling_params: SplineSamplingParams = SplineSamplingParams(), bounds_xy=np.array([-0.9, 0.9])) -> np.ndarray:
  '''
    Generates an Nx3 array of knot points. Rejection samples a spline that stays in the supplied bounds.
    Splines start at STROKE_START_XY moving towards (+x, 0.), and then curve randomly. The image is in unit bounds.
  '''
  while 1:
    angles = np.random.uniform(-sampling_params.MAX_SPLINE_CURVE,
                               sampling_params.MAX_SPLINE_CURVE, size=sampling_params.N_SPLINE_POINTS)
    distances = np.random.uniform(
        sampling_params.MIN_SPLINE_DISTANCE, sampling_params.MAX_SPLINE_DISTANCE, size=sampling_params.N_SPLINE_POINTS)
    heading = np.cumsum(angles)
    offsets = np.stack(
        [np.cos(heading), np.sin(heading)], axis=0
    ) * distances
    offsets[:2, 0] = sampling_params.STROKE_START_XY # Force all strokes to start at target location
    pts = np.cumsum(offsets, axis=1).T
    if np.min(pts) <= bounds_xy[0] or np.max(pts) >= bounds_xy[1]:
      continue
    # We're good on XY, so generate out heights in a single pass.
    height_steps = np.random.uniform(-sampling_params.MAX_SPLINE_HEIGHT_STEP,
                                     sampling_params.MAX_SPLINE_HEIGHT_STEP, size=sampling_params.N_SPLINE_POINTS - 1)
    heights = np.zeros(sampling_params.N_SPLINE_POINTS)
    heights[0] = np.random.uniform(sampling_params.MIN_SPLINE_HEIGHT, sampling_params.MAX_SPLINE_HEIGHT)
    for k, height_step in enumerate(height_steps):
      heights[k+1] = np.clip(
          heights[k] + height_step,
          sampling_params.MIN_SPLINE_HEIGHT, sampling_params.MAX_SPLINE_HEIGHT
      )
    pts = np.c_[pts, heights]
    return pts

def make_spline_from_pts(pts: np.ndarray) -> BSpline:
  t = np.linspace(0., 1., pts.shape[0])
  return make_interp_spline(t, pts, k=3, bc_type="clamped")
  
def make_random_spline(sampling_params: SplineSamplingParams = SplineSamplingParams(), bounds_xy=np.array([-0.9, 0.9])) -> BSpline:
  '''
    See make_random_spline_pts. Returns a BSpline object.
  '''
  pts = make_random_spline_pts(sampling_params, bounds_xy)
  return make_spline_from_pts(pts)
