"""
    Collects training data by rolling out random strokes. Saves training data following the format
"""

import robot_painting.models.spline_generation as spline_generation
import robot_painting.hardware_drivers.canvas_imager as canvas_imager
import robot_painting.hardware_drivers.printer_controller as printer_controller
import argparse
from dataclasses import asdict
import datetime
import pathlib
import numpy as np
import logging
import ujson as json
import matplotlib.pyplot as plt
import cv2
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


if __name__ == "__main__":
    LOG.setLevel(logging.INFO)

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
    args = parser.parse_args()

    out_directory = pathlib.Path(args.logdir) / pathlib.Path(
        get_timestamp() + "_" + args.pentype
    )
    LOG.info(f"Making output directory {out_directory}")
    out_directory.mkdir(parents=True, exist_ok=False)

    LOG.error("TODO: Move to location to insert pen.")
    input("Press enter when pen has been loaded with the tip touching the surface.")

    generation_params = spline_generation.SplineGenerationParams()
    # Start the info YAML file.
    info_dict = {
        "dataset_format_version": DATASET_FORMAT_VERSION,
        "pen_type": args.pentype,
        "generation_params": asdict(generation_params),
        "actions": []
    }
    info_file = out_directory / pathlib.Path("info.json")
    with open(info_file, "w") as f:
        json.dump(info_dict, f, indent=2)

    # Get initial image.
    im = np.zeros((256, 256, 3), dtype=np.uint8)
    before_image_path = save_image(im, out_directory=out_directory)
    for sample_k in tqdm.tqdm(range(args.n)):
        color = np.random.uniform(0, 255, size=(3,)).astype(int)

        # Pick a spline and execute it.
        spline = spline_generation.make_random_spline(params=generation_params)
        t = np.linspace(spline.x[0], spline.x[-1], 100)
        xs = spline(t)
        xs[:, 0] += im.shape[0] / 2.
        xs[:, 1] += im.shape[1] / 2.
        for k in range(len(t) - 1):
            im = cv2.line(
                im,
                xs[k, :2].astype(int),
                xs[k + 1, :2].astype(int),
                color=color.tolist(),
                thickness=int(xs[k, 2] * 10 + 1),
            )
        after_image_path = save_image(im, out_directory=out_directory)

        # Save the action we just took.
        action = {
            "action": {
                "action_type": "PPoly",
                "PPoly": spline_generation.spline_to_dict(spline),
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
