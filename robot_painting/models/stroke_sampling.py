import numpy as np
from dataclasses import dataclass

from scipy.interpolate import BSpline, make_interp_spline

'''
  OK, so issue: if I want to search over strokes via gradient descent, I need a way 
  to constrain the optimized parameters to the legal search space. So ideally strokes
  are parameterized by a vector of (WLOG) unit-range values, and a uniform distribution over
  those values creates a uniform distribution over "reasonable" strokes.

  I think this space could be:
  - Distance delta (xy), distance delta (z), and heading change at each knot, as we presently do.
  So 3xN values.

  This turns into a trajectory via integration (with saturation on z), and I guess we keep the trajectory
  short enough that we don't hit the walls
'''

@dataclass
class SplineSamplingParams():
  MAX_SPLINE_CURVE = np.deg2rad(60)
  MAX_SPLINE_DISTANCE = 0.2
  MIN_SPLINE_DISTANCE = 0.1
  MIN_SPLINE_HEIGHT = 0.4
  MAX_SPLINE_HEIGHT = 1.0
  MAX_SPLINE_HEIGHT_STEP = 0.3

def integrate_knot_velocities(q0: np.ndarray, v: np.ndarray, sampling_params: SplineSamplingParams) -> np.ndarray:
  '''
    q0: [z, heading]. Will always start at xy=[0, 0].
    v: (N)x3 [xy speed, z speed, angular speed].
  '''
  assert v.shape[1] == 3
  N = v.shape[0]
  headings = np.cumsum(v[:, 2]) + q0[1]
  distances = v[:, 0]
  offsets = np.stack(
      [np.cos(headings), np.sin(headings)], axis=0
  ) * distances
  pts = np.cumsum(offsets, axis=1).T
  
  height_steps = v[:, 1]
  heights = np.empty(N)
  for k, height_step in enumerate(height_steps):
    if k == 0:
      unclipped_next_height = q0[0] + height_step
    else:
      unclipped_next_height = heights[k-1] + height_step
    heights[k] = np.clip(
        unclipped_next_height,
        sampling_params.MIN_SPLINE_HEIGHT, sampling_params.MAX_SPLINE_HEIGHT
    )
  pts = np.c_[pts, heights]
  return pts


def make_spline_from_unit_parameter_vector(q0_unit: np.ndarray, v_unit: np.ndarray, sampling_params: SplineSamplingParams) -> BSpline:
  '''
    Parameter vector should be 2 + N*3 values, ordered 
  '''
  q0 = np.empty(2)
  q0[0] = q0_unit[0] * (sampling_params.MAX_SPLINE_HEIGHT - sampling_params.MIN_SPLINE_HEIGHT) + sampling_params.MIN_SPLINE_HEIGHT
  q0[1] = q0_unit[1] * np.pi * 2.
  n_knots = v_unit.shape[0]
  v = np.empty_like(v_unit)
  v[:, 0] = v_unit[:, 0] * (sampling_params.MAX_SPLINE_DISTANCE - sampling_params.MIN_SPLINE_DISTANCE) + sampling_params.MIN_SPLINE_DISTANCE
  v[:, 1] = (v_unit[:, 1] - 0.5) * 2. * sampling_params.MAX_SPLINE_HEIGHT_STEP 
  v[:, 2] = (v_unit[:, 2] - 0.5) * 2. * sampling_params.MAX_SPLINE_CURVE
  t = np.linspace(0., 1., n_knots, endpoint=True)
  pts = integrate_knot_velocities(q0, v, sampling_params)
  return make_interp_spline(t, pts, k=3, bc_type="clamped")

def make_random_spline_unit_parameters(n_knots):
  q0 = np.random.uniform(0., 1., (2,))
  v = np.random.uniform(0., 1., (n_knots, 3))
  return q0, v

def make_random_spline(n_knots: int, sampling_params: SplineSamplingParams) -> BSpline:
  '''
    See make_random_spline_pts. Returns a BSpline object.
  '''
  q0, v = make_random_spline_unit_parameters(n_knots)
  return make_spline_from_unit_parameter_vector(q0, v, sampling_params)
