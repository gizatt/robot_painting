from copy import deepcopy
import cv2
import serial
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import logging
from calibration import make_homog
from stroke_planner import get_brush_stroke_masked_sd, convert_masked_sd_to_strokes, draw_stroke
from calibration import make_homog
from canvas_imager import CanvasImager
from pathlib import Path

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

from grbl_gcode_streamer import GRBLGCodeStreamer

FEED_RATE_WHILE_DRAWING = 1000

# Width is 0 at s=0.5, and this at s=0.75. Any further down and bristles get crazy
BRUSH_WIDTH_MAX = 6
def convert_pixel_width_to_stroke(x):
    return np.clip((x / BRUSH_WIDTH_MAX) * 0.2 + 0.5, 0., 1.)

def make_stroke(
    interface,
    canvas_imager,
    calib_data,
    target_image: np.ndarray,
    save_prefix: Path = None,
):
    # Take an image.
    canvas_output = canvas_imager.update()
    if canvas_output.rectified_canvas is None:
        LOG.error("Couldn't get canvas image.")
        return False

    # Threshold it and take diff with the target.
    bw_canvas = cv2.cvtColor(canvas_output.rectified_canvas, cv2.COLOR_BGR2GRAY)
    thresholded_canvas = bw_canvas >= 128

    kernel = np.ones((3,3),np.uint8)
    thresholded_canvas = cv2.erode(thresholded_canvas.astype(np.uint8)*255, kernel, iterations=1)

    # Make an image that's 0 only where the target image is 0 and the thresholded canvas is not already dark.
    diff_im = np.invert(np.logical_and(target_image == 0, thresholded_canvas != 0))
    if save_prefix is not None:
        cv2.imwrite(str(save_prefix) + "_before.png", canvas_output.rectified_canvas)
        cv2.imwrite(str(save_prefix) + "_before_thresholded.png", thresholded_canvas)
        cv2.imwrite(str(save_prefix) + "_diff.png", diff_im.astype(np.uint8)*255)

    # to_show = np.concatenate([bw_canvas, 255*thresholded_canvas, 255*diff_im], axis=0).astype(np.uint8)
    # cv2.imshow("images", to_show)
    # cv2.waitKey(0)

    # Plan strokes.
    start_time = time.time()
    masked_sd = get_brush_stroke_masked_sd(diff_im, threshold=0.5, brush_width=BRUSH_WIDTH_MAX)
    sd_time = time.time()
    strokes = convert_masked_sd_to_strokes(masked_sd, point_spacing=3)
    strokes_time = time.time()
    print(
        f"Took %f sec for signed distance, %f sec for stroke gen."
        % (sd_time - start_time, strokes_time - sd_time)
    )
    strokes = sorted(strokes, key=lambda x: x.shape[1], reverse=True)
    for stroke_k, stroke in enumerate(strokes):
        original_stroke = deepcopy(stroke)

        if stroke.shape[1] < 5 and np.max(stroke[2, :]) < BRUSH_WIDTH_MAX / 2.:
            print(f"Skipping short stroke.")
            continue

        # Take a random stroke.
        print(f"Taking stroke of length {stroke.shape[1]}")

        # Convert to robot coords, adding a final point that lifts the pen.
        # Why do I need a flip here? Where did I go wrong...
        stroke[[0,1]] = stroke[[1, 0]]
        stroke = np.c_[stroke, np.array([stroke[0, -1], stroke[1, -1], 0.0])]
        stroke[:2, :] = robot_M_uv @ make_homog(stroke[:2, :])
        stroke[2, :] = convert_pixel_width_to_stroke(stroke[2, :])

        # Move to start, and wait to get there.
        xy_target = stroke[:2, 0]
        interface.send_move_command(x=stroke[0, 0], y=stroke[1, 0], stroke=0.0, feed_rate=6000)
        while not np.allclose(interface.xyz[:2], xy_target, atol=1):
            interface.update()

        # Drop pen to just about surface, wait a blip
        interface.send_move_command(x=stroke[0, 0], y=stroke[1, 0], stroke=0.45, feed_rate=FEED_RATE_WHILE_DRAWING)
        interface.update()
        time.sleep(0.25)

        # Send the stroke itself
        for x, y, s in stroke.T:
            interface.send_move_command(x=x, y=y, stroke=s, feed_rate=FEED_RATE_WHILE_DRAWING)

        # Wait to get to the end.
        interface.update()
        time.sleep(0.25)
        xy_target = stroke[:2, -1]
        while not np.allclose(interface.xyz[:2], xy_target, atol=1):
            interface.update()


        print(f"Done with stroke {stroke_k}")

        if save_prefix is not None:
            # Save out the before image before we ovewrite it.
            cv2.imwrite(str(save_prefix) + f"_s{stroke_k:03d}_before.png", canvas_output.rectified_canvas)

            # Draw the intended stroke over it and save that.
            before_plus_intended_stroke = canvas_output.rectified_canvas.copy()
            draw_stroke(before_plus_intended_stroke, original_stroke, color=[0, 0, 0, 255])
            cv2.imwrite(str(save_prefix) + f"_s{stroke_k:03d}_before_plus_plan.png", before_plus_intended_stroke)


            interface.send_move_command(x=0, y=0, stroke=0, feed_rate=6000)
            while not np.allclose(interface.xyz[:2], np.zeros(2), atol=1):
                interface.update()
            canvas_output = canvas_imager.update()
            if canvas_output.rectified_canvas is None:
                LOG.error("Couldn't get canvas image after stroke %d.", stroke_k)
                return False
            cv2.imwrite(str(save_prefix) + f"_s{stroke_k:03d}_after.png", canvas_output.rectified_canvas)
            cv2.imwrite(str(save_prefix) + f"_s{stroke_k:03d}_after_residual.png", ((canvas_output.rectified_canvas.astype(np.float32) - before_plus_intended_stroke.astype(np.float32)) / 2 + 128).astype(np.uint8))
            
            
    # Go home.
    interface.send_move_command(x=0, y=0, stroke=0, feed_rate=6000)
    while not np.allclose(interface.xyz[:2], np.zeros(2), atol=1):
        interface.update()
    interface.send_move_command(x=0, y=0, stroke=0.5, feed_rate=6000)
    interface.update()
    time.sleep(0.1)
    interface.update()

if __name__ == "__main__":
    logging.basicConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=str)
    parser.add_argument("image", type=str)
    parser.add_argument("--n-iters", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()

    img_bgr = cv2.imread(args.image)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_threshold = img_gray > 128


    calib_data = np.load("calibration.npz", allow_pickle=True)
    im_size = calib_data["im_size"]
    robot_M_uv = calib_data["robot_M_uv"]

    assert (
        img_gray.shape[1] == im_size[0] and img_gray.shape[0] == im_size[1]
    ), f"Input image of wrong size: {img_gray.shape} vs desired {im_size}"

    canvas_imager = CanvasImager()

    interface = GRBLGCodeStreamer(args.port, verbose=False)
    interface.send_setting("$32=1")
    interface.send_command("G90")
    interface.send_command("G92 X0 Y0")
    interface.send_command("M3")
    interface.update()

    if args.log_dir is not None:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=False)
        cv2.imwrite(log_dir / "target.png", img_threshold.astype(np.uint8)*255)
    else:
        log_dir = None

    for k in range(args.n_iters):
        if log_dir is not None:
            save_prefix = log_dir / ("%03d" % k)
        else:
            save_prefix = None
        make_stroke(
            interface, canvas_imager, calib_data, img_threshold, save_prefix=save_prefix
        )
