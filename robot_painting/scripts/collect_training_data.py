"""
    Collects training data by rolling out random strokes. Saves training data following the format
"""

from typing import Optional
import robot_painting.models.spline_generation as spline_generation
import robot_painting.hardware_drivers.canvas_imager as canvas_imager
import robot_painting.hardware_drivers.printer_controller as printer_controller
import robot_painting.hardware_drivers.calibration as calibration
import argparse
from dataclasses import asdict
import datetime
import pathlib
import numpy as np
import logging
import ujson as json
import matplotlib.pyplot as plt
import cv2
import sys
import tqdm

LOG = logging.getLogger()

DATASET_FORMAT_VERSION = "v1"


def get_timestamp() -> str:
    """
    Generates a second-resolution string timestamp of format `YYYYMMDD_HHMMSS`.
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_image(im: np.ndarray, out_directory: pathlib.Path) -> str:
    """
    Saves a timestamped image, and returns the name generated for that image within `out_directory`.
    """
    im_name = pathlib.Path(f"{get_timestamp()}_000.png")
    k = 1
    while (out_directory / im_name).exists():
        im_name = pathlib.Path(f"{get_timestamp()}_{k:03d}.png")
        k += 1
    im_path = out_directory / im_name
    cv2.imwrite(im_path, im)
    return str(im_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pentype",
        required=True,
        help="Unique name for the pen to be used for this run.",
    )
    parser.add_argument(
        "--logdir",
        required=True,
        help="Log directory. This run will be saved in a timestamped subdirectory.",
    )
    parser.add_argument(
        "--n", type=int, default=100, help="Number of paintstrokes to record."
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    return parser.parse_args()


def sample_spline_within_robot_bounds(
    calib_data: calibration.Calibration,
    generation_params: spline_generation.SplineGenerationParams,
    rng: np.random.Generator,
) -> Optional[spline_generation.SplineAndOffset]:
    # Randomly sample a spline and pick a random translation to place it somewhere in the canvas without
    # overflowing an edge.
    spline = spline_generation.make_random_spline(params=generation_params, rng=rng)
    t = np.linspace(spline.x[0], spline.x[-1], 100)
    xs = spline(t)
    min_extent = np.min(xs, axis=0)[:2]
    max_extent = np.max(xs, axis=0)[:2]
    lb = calib_data.robot_lb - min_extent
    ub = calib_data.robot_ub - max_extent
    if any(lb >= ub):
        LOG.error("Spline is too large to fit in bounds!")
        return None
    offset = np.r_[rng.uniform(lb, ub), 0.0]
    return spline_generation.SplineAndOffset(spline, offset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = parse_args()

    rng = np.random.default_rng(args.seed)

    # Load up calibration data.
    calib_data = calibration.Calibration.from_file(
        "robot_painting/hardware_drivers/calibration.npz"
    )

    # Get output directoy ready.
    out_directory = pathlib.Path(args.logdir) / pathlib.Path(
        get_timestamp() + "_" + args.pentype
    )
    LOG.info(f"Making output directory {out_directory}")
    out_directory.mkdir(parents=True, exist_ok=False)

    # Load pen.
    LOG.error("TODO: Move to location to insert pen.")
    input("Press enter when pen has been loaded with the tip touching the surface.")

    # Save out initial config file.
    generation_params = spline_generation.SplineGenerationParams()
    # Start the info YAML file.
    info_dict = {
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "pen_type": args.pentype,
        "generation_params": asdict(generation_params),
        "actions": [],
    }
    info_file = out_directory / pathlib.Path("info.json")
    with open(info_file, "w") as f:
        json.dump(info_dict, f, indent=2)

    # Get initial image.
    im = np.zeros((calib_data.im_size[0], calib_data.im_size[1], 3), dtype=np.uint8)
    before_image_path = save_image(im, out_directory=out_directory)
    for sample_k in tqdm.tqdm(range(args.n)):
        color = rng.uniform(0, 255, size=(3,)).astype(int)

        # Pick a spline and execute it.
        spline_and_offset = sample_spline_within_robot_bounds(
            calib_data=calib_data, generation_params=generation_params, rng=rng
        )
        xs = spline_and_offset.sample(100)
        uvs = (calib_data.uv_M_robot @ calibration.make_homog(xs[:, :2].T)).T
        uvs = np.flip(uvs, axis=1)
        for k in range(xs.shape[0] - 1):
            im = cv2.line(
                im,
                uvs[k, :2].astype(int),
                uvs[k + 1, :2].astype(int),
                color=color.tolist(),
                thickness=int(xs[k, 2] * 10 + 1),
            )
        after_image_path = save_image(im, out_directory=out_directory)

        # Save the action we just took.
        action = {
            "action": {
                "action_type": "SplineAndOffset",
                "SplineAndOffset": spline_and_offset.to_dict(),
            },
            "before_im": before_image_path,
            "after_im": after_image_path,
        }
        # TODO(gizatt) Horribly inefficient to be writing and rewriting this
        # file like this. But we do writing infrequently enough that this
        # doesn't bottleneck us until we have thousands of entries, which would
        # be more strokes that we'd do in a single sitting (it'd take hours).
        info_dict["actions"].append(action)
        with open(info_file, "w") as f:
            json.dump(info_dict, f, indent=2)

        # Proceed to the next action.
        before_image_path = after_image_path
