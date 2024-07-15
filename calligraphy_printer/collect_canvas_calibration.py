"""
    Collects 

"""

import time
import argparse
import numpy as np
import logging
from pathlib import Path
import cv2
import sys
import shutil

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

from grbl_gcode_streamer import GRBLGCodeStreamer, get_servo
from canvas_imager import CanvasImager, CanvasImagerOutput


show_debug = False


def show_image(output: CanvasImagerOutput):
    global show_debug
    if not show_debug and output.rectified_canvas is not None:
        cv2.imshow("canvas", output.rectified_canvas)
    elif output.debug_image is not None:
        cv2.imshow("canvas", output.debug_image)
    else:
        LOG.error("No image to show!")

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        LOG.error("Terminating early, on GUI request.")
        sys.exit(0)
    elif key == ord("f"):
        show_debug = not show_debug


if __name__ == "__main__":
    logging.basicConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=str)
    parser.add_argument("--port", type=str, default="COM3")
    parser.add_argument("-f", action="store_true", help="Overwrite output directory.")
    args = parser.parse_args()

    # Set up imager and ensure it's working
    imager = CanvasImager()
    output = imager.update(include_debug_drawing=True)
    if output.rectified_canvas is None:
        raise RuntimeError(
            "Couldn't get rectified canvas. Run canvas_imager and fix the webcam location."
        )
    show_image(output)

    # Talk to the robot and make sure that's working.
    interface = GRBLGCodeStreamer(args.port, verbose=False)
    interface.send_setting("$32=1")
    interface.send_command("G90")
    interface.send_command("G92 X0 Y0")
    interface.send_command("M3")
    interface.send_move_command(x=0, y=0, s=0)
    interface.update_until_done(timeout=2.0)

    # Now set up our output directory.
    output_directory = Path(args.output_directory)
    if output_directory.exists():
        if args.f:
            LOG.info("Removing output directory %s that exists.", output_directory)
            shutil.rmtree(output_directory)
        else:
            raise ValueError("Output directory %s already exists.", output_directory)
    LOG.info("Making output directory %s.", output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)

    # Take initial image
    output = imager.update(include_debug_drawing=True)
    show_image(output)
    assert output.rectified_canvas is not None
    cv2.imwrite(output_directory / f"im_start.png", output.rectified_canvas)
    LOG.info("Grabbed initial image.")

    # Calibration pattern.
    lb = np.array([80, 60])
    ub = np.array([280, 180])
    n_steps = 5
    k = 0
    xs = []
    for x in np.linspace(lb[0], ub[0], n_steps):
        for y in np.linspace(lb[1], ub[1], n_steps):
            xs.append([x, y])
            LOG.info(f"Move to {x} {y}")

            # Make mark
            interface.send_move_command(x=x, y=y, s=get_servo(0.0))
            interface.send_move_command(x=x, y=y, s=get_servo(0.6))
            interface.update()
            while not np.allclose(interface.xyz[:2], np.array([x, y]), atol=1e-3):
                interface.update()
            time.sleep(0.25)
            
            # Get head out of way
            interface.send_move_command(x=x, y=y, s=get_servo(0.0))
            interface.send_move_command(x=0, y=0, s=get_servo(0.0))
            while not np.allclose(interface.xyz[:2], np.array([0, 0]), atol=1e-3):
                interface.update()

            # Read
            output = imager.update(include_debug_drawing=True)
            show_image(output)
            if output.rectified_canvas is not None:
                cv2.imwrite(output_directory / f"im_{k}.png", output.rectified_canvas)
            LOG.info("Grabbed image, continuing.")
            k += 1
    targets_file = output_directory / "pts.npy"
    np.save(targets_file, xs)

    LOG.info(f"Done. Saved points to {targets_file}")
    interface.send_move_command(x=0, y=0, s=0)
    interface.update_until_done(timeout=5.0)
