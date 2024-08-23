"""
    Collects training data by rolling out random strokes. Saves training data following the format
"""

import time
from typing import Optional
import robot_painting.models.spline_generation as spline_generation
import robot_painting.hardware_drivers.canvas_imager as canvas_imager
import robot_painting.hardware_drivers.grbl_gcode_streamer as grbl_code_streamer
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
        "--port",
        required=True,
        help="Serial port for comms.",
    )
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
    # Note extra extent to keep the robot from leaving the canvas. It can get *very* close to the edge without this.
    lb = calib_data.robot_lb - min_extent + 10.
    ub = calib_data.robot_ub - max_extent - 10.
    if any(lb >= ub):
        LOG.error("Spline is too large to fit in bounds!")
        return None
    offset = np.r_[rng.uniform(lb, ub), 0.0]
    return spline_generation.SplineAndOffset(spline, offset)


def get_canvas_image(
    imager: canvas_imager.CanvasImager, timeout: float = 10.0
) -> canvas_imager.CanvasImagerOutput:
    start_time = time.time()
    while time.time() - start_time < timeout:
        output = imager.update(include_debug_drawing=True)
        cv2.imshow("Image", output.debug_image)
        cv2.waitKey(1)
        if output.rectified_canvas is not None:
            return output
    LOG.error("Could not get rectified output in time.")
    return None


def do_printer_setup(port: str) -> grbl_code_streamer.GRBLGCodeStreamer:
    input("Press enter when print head has been zero'd.")
    interface = grbl_code_streamer.GRBLGCodeStreamer(args.port, verbose=False)

    interface.send_setting("$32=1")
    interface.send_command("G90")
    interface.send_command("G92 X0 Y0")
    interface.send_command("M3")
    interface.update()
    interface.send_move_command(x=0, y=0, stroke=0.5, feed_rate=5000)
    interface.update()
    # Load pen.
    input("Press enter when pen has been loaded with the tip touching the surface.")
    return interface


def run_stroke(
    interface: grbl_code_streamer.GRBLGCodeStreamer,
    spline_and_offset: spline_generation.SplineAndOffset,
    timeout: float = 10.0,
) -> bool:
    xs = spline_and_offset.sample(100)
    dxs = spline_and_offset.sample_derivative(100)

    lb = np.min(xs[:, :2], axis=0)
    ub = np.max(xs[:, :2], axis=0)
    if lb[0] < 0 or lb[1] < 0:
        LOG.error(f"Bad lb on spline interpolation: {lb}")
        return False
    if ub[0] >= interface.X_MAX or ub[1] >= interface.Y_MAX:
        LOG.error(f"Bad ub on spline interpolation: {lb}")
        return False

    speeds_mm_per_s = np.linalg.norm(dxs[:, :2], axis=1)
    # Take a random stroke.
    LOG.info(f"Taking stroke of length {xs.shape[1]}")

    # Move to start, and wait to get there.
    xy_start = xs[0, :2]
    start_time = time.time()
    interface.send_move_command(
        x=xy_start[0], y=xy_start[1], stroke=0.0, feed_rate=6000
    )
    while not np.allclose(interface.xyz[:2], xy_start, atol=1):
        interface.update()
        if time.time() - start_time > timeout:
            LOG.error("Timed out waiting for stroke to start.")
            return False

    # Drop pen to just about surface, wait a blip
    interface.send_move_command(
        x=xy_start[0], y=xy_start[1], stroke=0.45, feed_rate=6000
    )
    interface.update()
    time.sleep(0.25)

    # Send the stroke itself.
    feed_rate_mm_p_min = np.clip(speeds_mm_per_s * 60.0, 100.0, 6000)
    print(feed_rate_mm_p_min)
    for k in range(xs.shape[0]):
        # Remap stroke heights from 0->1 to 0.5 -> 1, since on HW we consider 0.5 to be zero-width stroke.
        interface.send_move_command(
            x=xs[k, 0],
            y=xs[k, 1],
            stroke=xs[k, 2] * 0.5 + 0.5,
            feed_rate=feed_rate_mm_p_min[min(k, xs.shape[0] - 2)],
        )

    # Wait to get to the end.
    start_time = time.time()
    interface.update()
    time.sleep(0.25)
    xy_end = xs[-1, :2]
    while not np.allclose(interface.xyz[:2], xy_end, atol=1):
        interface.update()
        if time.time() - start_time > timeout:
            LOG.error("Timed out waiting for stroke to end.")
            return False

    # Home and wait for it to home.
    interface.send_move_command(x=xy_end[0], y=xy_end[1], stroke=0.0, feed_rate=6000)
    interface.send_move_command(x=0, y=0, stroke=0.0, feed_rate=6000)
    while not np.allclose(interface.xyz[:2], np.zeros(2), atol=1):
        interface.update()
        if time.time() - start_time > timeout:
            LOG.error("Timed out waiting for final homing.")
            return False

    return True


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

    # Make sure get get images as expected.
    imager = canvas_imager.CanvasImager()
    image = get_canvas_image(imager)
    assert image is not None
    assert (
        image.rectified_canvas.shape[0] == calib_data.im_size[1]
    ), f"Calibration looks out of date: {image.rectified_canvas.shape} vs {calib_data.im_size}"
    assert (
        image.rectified_canvas.shape[1] == calib_data.im_size[0]
    ), f"Calibration looks out of date: {image.rectified_canvas.shape} vs {calib_data.im_size}"

    # Setup printer.
    interface = do_printer_setup(port=args.port)

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
    image = get_canvas_image(imager)
    assert image is not None
    im = image.rectified_canvas
    before_image_path = save_image(im, out_directory=out_directory)
    for sample_k in tqdm.tqdm(range(args.n)):
        color = rng.uniform(0, 255, size=(3,)).astype(int)

        # Pick a spline and execute it.
        spline_and_offset = sample_spline_within_robot_bounds(
            calib_data=calib_data, generation_params=generation_params, rng=rng
        )
        stroke_successful = run_stroke(interface=interface, spline_and_offset=spline_and_offset)

        image = get_canvas_image(imager)
        assert image is not None
        im = image.rectified_canvas

        # uvs = (calib_data.uv_M_robot @ calibration.make_homog(xs[:, :2].T)).T
        # uvs = np.flip(uvs, axis=1)
        # for k in range(xs.shape[0] - 1):
        #     im = cv2.line(
        #         im,
        #         uvs[k, :2].astype(int),
        #         uvs[k + 1, :2].astype(int),
        #         color=color.tolist(),
        #         thickness=int(xs[k, 2] * 10 + 1),
        #     )
        after_image_path = save_image(im, out_directory=out_directory)

        if stroke_successful:
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
