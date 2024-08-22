import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional

import json
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PPoly
import matplotlib.pyplot as plt


@dataclass
class SplineGenerationParams:
    min_move_time: float = 0.5
    max_move_time: float = 2.0
    min_move_length: float = 10
    max_move_length: float = 20.0
    max_turn_amount: float = np.pi
    min_velocity: float = 0.0
    max_velocity: float = 50.0
    max_velocity_angle_from_dir: float = np.pi / 2.0
    n_steps: int = 2


def make_random_spline(
    params: SplineGenerationParams = SplineGenerationParams(),
    rng: Optional[np.random.Generator] = None,
) -> PPoly:
    """
    Produces a 3D (x-y-height) Cubic Hermite Spline with N_STEPS+1 control
    points. The first is at the origin with zero velocity and a random height
    (in [0, 1]). The subsequent knots are chosen by:
      1) Picking a random move time, and incrementing time by that much.
      2) Picking a random move length and heading, and moving that far from the
         previous point, in polar coordinates.
      3) Picking a velocity for that control point randomly in the allowed
         velocity range and heading delta from the move heading.
    """

    if rng is None:
        rng = np.random.default_rng()

    # Pts will be Nx3.
    ts = [0.0]
    qs = [np.array([0.0, 0.0, rng.uniform(0.0, 1.0)])]
    vs = [np.zeros(3)]

    heading = rng.uniform(0., 2*np.pi)
    for k in range(params.n_steps):
        dt = rng.uniform(params.min_move_time, params.max_move_time)
        ts.append(ts[-1] + dt)

        dist = rng.uniform(params.min_move_length, params.max_move_length)
        if k > 0:
            heading += rng.uniform(-params.max_turn_amount, params.max_turn_amount)
        height = rng.uniform(0.0, 1.0)
        q = np.array(
            [
                qs[-1][0] + dist * np.cos(heading),
                qs[-1][1] + dist * np.sin(heading),
                height,
            ]
        )
        qs.append(q)

        if k < params.n_steps - 1:
            speed = rng.uniform(params.min_velocity, params.max_velocity)
            vel_heading = heading + rng.uniform(
                -params.max_velocity_angle_from_dir, params.max_velocity_angle_from_dir
            )
            v = np.array(
                [speed * np.cos(vel_heading), speed * np.sin(vel_heading), 0.0]
            )
        else:
            v = np.zeros(3)
        vs.append(v)

    return CubicHermiteSpline(
        np.array(ts), np.stack(qs), np.stack(vs), extrapolate=False
    )


def spline_to_dict(spline: PPoly) -> dict:
    """
    Serializes a PPoly object to a JSON string.
    """
    ppoly_dict = {
        "c": spline.c.tolist(),
        "x": spline.x.tolist(),
    }
    return ppoly_dict


def spline_from_dict(spline_dict: dict) -> PPoly:
    """
    Deserializes a JSON string back to a PPoly object.
    """
    c = np.array(spline_dict["c"])
    x = np.array(spline_dict["x"])
    return PPoly(c=c, x=x, extrapolate=False)

@dataclass
class SplineAndOffset():
    '''
        Representation of spline with an x and y shift.
    '''
    spline: PPoly
    offset: np.ndarray # Should match dimension of PPoly.
    
    def sample(self, N: int) -> np.ndarray:
        '''
            Returns Nxspline_dim array.
        '''
        t = np.linspace(self.spline.x[0], self.spline.x[-1], N)
        xs = self.spline(t)
        return xs + self.offset
    
    def to_dict(self) -> dict:
        return {
            "spline": spline_to_dict(self.spline),
            "offset": self.offset.tolist()
        }

    @staticmethod
    def from_dict(data_dict: dict) -> 'SplineAndOffset':
        return SplineAndOffset(
            spline=spline_from_dict(data_dict["spline"]),
            offset=np.array(data_dict["offset"])
        )

if __name__ == "__main__":
    fig, axs = plt.subplots(2)

    for k in range(5):
        spline = make_random_spline()
        t = np.linspace(spline.x[0], spline.x[-1], 100)

        xs = spline(t)
        axs[0].plot(xs[:, 0], xs[:, 1])
        axs[1].plot(t, xs[:, 2])

    plt.show()
