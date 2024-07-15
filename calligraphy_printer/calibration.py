"""
    Utilities for performing calibration between a canvas with AprilTag corners
    and an X/Y gantry. The primary output of this calibration is a linear transformation
    matrix `robot_M_uv` that maps `robot_coordinate = robot_M_pixel * pixel_coordinate`.

    Calibration is performed by capturing a sequence of images of the robot drawing a dot
    at each of a sequence of grid locations `robot_coord_i`. An observed location of dot
    `pixel_coord_i` is extracted from the difference image of image `i` and image `i-1`. This
    yields a dataset of calibration pairs `robot_coord_i = robot_M_uv * pixel_coord_i` to which
    we fit `robot_M_uv`.
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from pathlib import Path
from canvas_imager import CANVAS_IM_WIDTH, CANVAS_IM_HEIGHT

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def invert_affine_tf(M: np.ndarray) -> np.ndarray:
    Rinv = np.linalg.inv(M[:2, :2])
    return np.c_[Rinv, -Rinv @ M[:, 2]]


def make_homog(pts: np.ndarray) -> np.ndarray:
    return np.r_[pts, np.ones((1, pts.shape[1]))]


def fit_affine_transform(
    robot_coords: np.ndarray, pixel_coords: np.ndarray
) -> np.ndarray:
    """
    Inputs should be 2xN. Output is 3x3. (We'll padd pixel_coords to homog coordinations.)
    """
    # return make_homog(robot_coords) * np.linalg.pinv(make_homog(pixel_coords))[:2, :]
    robot_coord_mean = np.mean(robot_coords, axis=1, keepdims=True)
    pixel_coords_mean = np.mean(pixel_coords, axis=1, keepdims=True)
    t = robot_coord_mean - pixel_coords_mean
    R = np.linalg.lstsq(
        (pixel_coords - pixel_coords_mean).T, (robot_coords - robot_coord_mean).T
    )[0].T
    return np.c_[R, -R @ pixel_coords_mean + robot_coord_mean]


def add_dot_to_image(image: cv2.Mat, uv: np.ndarray, radius: int):
    cv2.circle(image, uv.astype(int)[[1, 0]], radius, color=(0, 0, 0), thickness=-1)
    print(np.max(image))


class ImageBlobDetector:
    def __init__(self, min_radius: float = 3):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = np.pi * min_radius**2
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.filterByConvexity = True
        params.minConvexity = 0.2
        params.filterByInertia = True
        params.minInertiaRatio = 0.01
        self.detector = cv2.SimpleBlobDetector_create(params)

    def detect_difference_image_peak(
        self, image: cv2.Mat, last_image: cv2.Mat
    ) -> np.ndarray:
        difference = 1.0 - ((1.0 - image) - (1.0 - last_image))
        keypoints = self.detector.detect((difference * 255.0).astype(np.uint8))
        if len(keypoints) != 1:
            LOG.error(f"Detected {len(keypoints)} != 1 keypoints.")
            return None
        return np.asarray(keypoints[0].pt)


def do_test():
    ROBOT_BOUNDS = [297.0, 210.0]  # mm
    X, Y = np.meshgrid(
        np.linspace(0.0, ROBOT_BOUNDS[0], 5), np.linspace(0.0, ROBOT_BOUNDS[1], 5)
    )
    robot_coords = np.stack([X.ravel(), Y.ravel()], axis=0)

    # For testing, generate noisy observations that exercise our image detection pipeline.
    IM_SIZE = [640, 480]
    M_gt = np.array(
        [
            [0.0, 0.8 * ROBOT_BOUNDS[0] / IM_SIZE[1], -30],
            [1.2 * ROBOT_BOUNDS[1] / IM_SIZE[0], 0.0, -30],
        ]
    )
    pixel_coords = invert_affine_tf(M_gt) @ make_homog(robot_coords) + np.random.normal(
        0.0, 5, size=robot_coords.shape
    )
    robot_coords_were_matched = []
    pixel_coords_detected = []
    im = np.ones(IM_SIZE)
    detector = ImageBlobDetector()
    for pixel_coord, robot_coord in zip(pixel_coords.T, robot_coords.T, strict=True):
        last_im = im.copy()
        add_dot_to_image(im, pixel_coord, radius=int(np.random.uniform(1, 10)))
        detection = detector.detect_difference_image_peak(im, last_im)
        robot_coords_were_matched.append(detection is not None)
        if detection is not None:
            pixel_coords_detected.append(detection)

    if len(pixel_coords_detected) > 0:
        pixel_coords_detected = np.stack(pixel_coords_detected, axis=1)
        robot_coords_matched = robot_coords[:, robot_coords_were_matched]
        M_fit = fit_affine_transform(robot_coords_matched, pixel_coords_detected)
        robot_coords_reprojected = M_fit @ make_homog(pixel_coords_detected)
    else:
        LOG.error("Detected no keypoints ever.")
        plt.figure()
        plt.imshow(im.T)
        plt.show()
        sys.exit(-1)

    print(f"Ground truth: {M_gt}")
    print(f"Fit: {M_fit}")
    print(f"Errors: {M_gt - M_fit}")
    errors = robot_coords_reprojected - robot_coords_matched
    error_scales = np.linalg.norm(errors, axis=0)
    mse = np.mean(error_scales)
    print(f"MSE: {mse}")

    plt.figure()

    # Plot 1: robot coordinates with reprojections. Color coding indicates which samples were observed (red if not observed).
    MAX_ALLOWED_ERROR = 3.0
    plt.subplot(2, 1, 1)
    plt.title(f"Fit MSE (in robot coords): {mse: 0.2f}")
    plt.scatter(
        robot_coords[0, :],
        robot_coords[1, :],
        c=["green" if matched else "red" for matched in robot_coords_were_matched],
        label="Robot Coordinates",
    )
    nonzero_errors = error_scales >= MAX_ALLOWED_ERROR
    plt.scatter(
        robot_coords_reprojected[0, nonzero_errors],
        robot_coords_reprojected[1, nonzero_errors],
        c="blue",
        label="Reprojected Coordinates",
        alpha=0.5,
    )
    plt.quiver(
        robot_coords_matched[0, nonzero_errors],
        robot_coords_matched[1, nonzero_errors],
        errors[0, nonzero_errors],
        errors[1, nonzero_errors],
        color="blue",
        angles="xy",
        scale=1.0,
        scale_units="xy",
        alpha=0.5,
    )

    # Plot 2: Detected image coordinates over the final image. Color coding by reprojection quality.
    plt.subplot(2, 1, 2)
    plt.imshow(np.repeat(im.T[:, :, np.newaxis], 3, axis=2))
    quality = error_scales / np.max(error_scales)
    # Map 0 to green and 1 to red in gist_rainbow
    cmap = plt.get_cmap("gist_rainbow")
    plt.scatter(
        pixel_coords_detected[1, :],
        pixel_coords_detected[0, :],
        c=cmap(0.3 - (quality * 0.3)),
        alpha=1.0,
        marker="x",
        s=50,
    )

    plt.show()


def run_calibration_on_directory(calibration_directory: str):
    calibration_directory = Path(calibration_directory)
    assert calibration_directory.exists()
    robot_coords = np.load(calibration_directory / "pts.npy").T

    def load_im(path):
        img = cv2.imread(calibration_directory / path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(img.shape)
        img = cv2.resize(img, (CANVAS_IM_WIDTH, CANVAS_IM_HEIGHT))
        print(img.shape)
        return (img > 128).astype(np.float32)

    im_start = load_im("im_start.png")
    ims = [load_im(f"im_{k}.png") for k in range(len(robot_coords.T))]
    LOG.info("Loaded %d datapoints from %s.", len(ims), calibration_directory)
    detector = ImageBlobDetector(min_radius=2)

    robot_coords_were_matched = []
    pixel_coords_detected = []
    last_im = im_start
    for xy, next_im in zip(robot_coords.T, ims, strict=True):
        detection = detector.detect_difference_image_peak(next_im, last_im)
        last_im = next_im
        robot_coords_were_matched.append(detection is not None)
        if detection is not None:
            pixel_coords_detected.append(detection)

    im = ims[-1]
    if len(pixel_coords_detected) > 0:
        pixel_coords_detected = np.stack(pixel_coords_detected, axis=1)
        robot_coords_matched = robot_coords[:, robot_coords_were_matched]
        M_fit = fit_affine_transform(robot_coords_matched, pixel_coords_detected)
        robot_coords_reprojected = M_fit @ make_homog(pixel_coords_detected)
    else:
        LOG.error("Detected no keypoints at all.")
        plt.figure()
        plt.imshow(im.T)
        plt.show()
        sys.exit(-1)

    LOG.info(f"Fit: {M_fit}")
    errors = robot_coords_reprojected - robot_coords_matched
    error_scales = np.linalg.norm(errors, axis=0)
    mse = np.mean(error_scales)
    LOG.info(f"MSE: {mse}")

    # Spit out a calibration file
    calibration_data = {
        "robot_M_uv": M_fit,
        "im_size": [im.shape[1], im.shape[0]],
        "lb": M_fit @ np.array([0., 0., 1.]),
        "ub": M_fit @ np.array([im.shape[1], im.shape[0], 1.]),
    }
    LOG.info("Final calibration data: %s", calibration_data)
    calibration_output_path = calibration_directory / "calibration.npz"
    np.savez(calibration_output_path, **calibration_data)
    LOG.info("Saved calibration to %s", calibration_output_path)

    plt.figure()

    # Plot 1: robot coordinates with reprojections. Color coding indicates which samples were observed (red if not observed).
    MAX_ALLOWED_ERROR = 3.0
    plt.subplot(2, 1, 1)
    plt.title(f"Fit MSE (in robot coords): {mse: 0.2f}")
    plt.scatter(
        robot_coords[0, :],
        robot_coords[1, :],
        c=["green" if matched else "red" for matched in robot_coords_were_matched],
        label="Robot Coordinates",
    )
    nonzero_errors = error_scales >= MAX_ALLOWED_ERROR
    plt.scatter(
        robot_coords_reprojected[0, nonzero_errors],
        robot_coords_reprojected[1, nonzero_errors],
        c="blue",
        label="Reprojected Coordinates",
        alpha=0.5,
    )
    plt.quiver(
        robot_coords_matched[0, nonzero_errors],
        robot_coords_matched[1, nonzero_errors],
        errors[0, nonzero_errors],
        errors[1, nonzero_errors],
        color="blue",
        angles="xy",
        scale=1.0,
        scale_units="xy",
        alpha=0.5,
    )

    # Plot 2: Detected image coordinates over the final image. Color coding by reprojection quality.
    plt.subplot(2, 1, 2)
    plt.imshow(np.repeat(im.T[:, :, np.newaxis], 3, axis=2))
    quality = error_scales / np.max(error_scales)
    # Map 0 to green and 1 to red in gist_rainbow
    cmap = plt.get_cmap("gist_rainbow")
    plt.scatter(
        pixel_coords_detected[1, :],
        pixel_coords_detected[0, :],
        c=cmap(0.3 - (quality * 0.3)),
        alpha=1.0,
        marker="x",
        s=50,
    )

    plt.show()


if __name__ == "__main__":
    logging.basicConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration_directory", type=str, default="")
    args = parser.parse_args()

    if len(args.calibration_directory) == 0:
        run_test()
    else:
        run_calibration_on_directory(args.calibration_directory)
