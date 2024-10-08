import argparse
import logging
import pathlib
from copy import deepcopy
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.interpolate
import torch
import torchvision.transforms
import ujson as json
from torch.utils.data import Dataset, IterableDataset
from torchvision.io import read_image

import robot_painting.hardware_drivers.calibration as calibration
import robot_painting.models.spline_generation as spline_generation

LOG = logging.getLogger()


class StrokeDatasetRandomization(object):
    """
    Apply domain randomization on a stroke dataset entry.

    At present, this will:
    """

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def __call__(
        self, before_image, after_image, action: spline_generation.SplineAndOffset
    ):
        assert (
            np.min(np.abs(action.offset)) < 1e-3
        )  # Spline should start at origin now.
        yaw = 45 + 0 * self.rng.uniform(0.0, 360.0)
        yaw_rad = np.deg2rad(yaw)
        before_image = torchvision.transforms.functional.rotate(
            before_image,
            angle=yaw,
            fill=(1.0, 1.0, 1.0),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        after_image = torchvision.transforms.functional.rotate(
            after_image,
            angle=yaw,
            fill=(1.0, 1.0, 1.0),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )

        # This is opposite the rotation I expect... but matches visually... :P
        rotmat = np.array(
            [
                [np.cos(yaw_rad), np.sin(yaw_rad), 0],
                [-np.sin(yaw_rad), np.cos(yaw_rad), 0],
                [0, 0, 1],
            ]
        )
        for i in range(action.spline.c.shape[0]):
            for j in range(action.spline.c.shape[1]):
                action.spline.c[i, j, :] = rotmat @ action.spline.c[i, j, :]
        return before_image, after_image, action


class SplineToSamples(object):
    """
    Convert a spline to a set of samples by sampling it at regular intervals and
    normalizing the samples to the max distance the spline can be drawn.
    """

    def __init__(
        self,
        max_stroke_duration: float,
        max_stroke_distance: float,
        num_stroke_time_samples: int = 32,
    ):
        self.max_stroke_duration = max_stroke_duration
        self.num_stroke_time_samples = num_stroke_time_samples
        self.max_stroke_distance = max_stroke_distance

    def __call__(self, spline: spline_generation.SplineAndOffset):
        t = np.linspace(
            0.0, self.max_stroke_duration, self.num_stroke_time_samples, endpoint=True
        )
        t = np.clip(t, a_min=spline.spline.x[0], a_max=spline.spline.x[-1])
        return (spline.spline(t) + spline.offset) / self.max_stroke_distance

    def invert(self, spline_params: np.ndarray) -> np.ndarray:
        return spline_params * self.max_stroke_distance

    @staticmethod
    def make_from_spline_generation_params(
        spline_generation_params: spline_generation.SplineGenerationParams,
        num_stroke_time_samples: int = 32,
    ):
        return SplineToSamples(
            max_stroke_duration=spline_generation_params.max_move_time
            * spline_generation_params.n_steps,
            max_stroke_distance=spline_generation_params.max_move_length
            * spline_generation_params.n_steps,
            num_stroke_time_samples=num_stroke_time_samples,
        )


class StrokeDataset(Dataset):

    def __init__(
        self,
        dataset_path: str | pathlib.Path,
        dataset_assignment: str,
        spline_transform: Callable[[spline_generation.SplineAndOffset], np.ndarray],
        output_image_size: int = 128,
        transform=None,
        crop_halfwidth_mm: int = 64,  # Output image will be 1mm per pixel.
    ):
        if isinstance(dataset_path, str):
            dataset_path = pathlib.Path(dataset_path)
        assert dataset_path.exists()
        self.dataset_path = dataset_path
        self.dataset_assignment = dataset_assignment

        self.spline_transform = spline_transform
        self.output_image_size = output_image_size

        calibration_data_path = dataset_path / "calibration.npz"
        assert calibration_data_path.exists()
        self.calibration = calibration.Calibration.from_file(calibration_data_path)
        self.crop_halfwidth_mm = crop_halfwidth_mm

        # Walk subdirectories, loading each `info.json`, and collecting dataset entries.
        self.before_image_paths = []
        self.after_image_paths = []
        self.actions = []
        self.pen_type_indices = []
        self.pen_type_to_index = {}

        subdirectories = [x for x in self.dataset_path.iterdir() if x.is_dir()]
        for subdirectory in subdirectories:
            info_json = subdirectory / "info.json"
            if not info_json.exists():
                continue

            with open(info_json, "r") as f:
                data = json.load(f)

            for entry in data["actions"]:
                if "dataset_assignment" not in entry:
                    LOG.error("Missing `dataset_assignment` in %s", info_json)
                    continue
                if entry["dataset_assignment"] != self.dataset_assignment:
                    continue

                action = entry["action"]
                if action["action_type"] != "SplineAndOffset":
                    LOG.error("Dataset %s has incorrect action type.", info_json)
                    continue

                before_im = subdirectory / entry["before_im"]
                assert before_im.exists()
                self.before_image_paths.append(before_im)
                after_im = subdirectory / entry["after_im"]
                assert after_im.exists()
                self.after_image_paths.append(after_im)
                self.actions.append(
                    spline_generation.SplineAndOffset.from_dict(
                        action["SplineAndOffset"]
                    )
                )
                pen_type = data["pen_type"]
                if pen_type not in self.pen_type_to_index:
                    self.pen_type_to_index[pen_type] = len(self.pen_type_to_index)
                self.pen_type_indices.append(self.pen_type_to_index[pen_type])

        LOG.info(
            f"Loaded StrokeDataset::{dataset_assignment} with {len(self.actions)} entries from {dataset_path}."
        )
        self.transform = transform

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        before_image = cv2.imread(self.before_image_paths[idx])
        after_image = cv2.imread(self.after_image_paths[idx])
        action: spline_generation.SplineAndOffset = deepcopy(self.actions[idx])
        pen_type_index = self.pen_type_indices[idx]

        # Crop around the stroke origin (first knot) by the desired amount.
        # Resize to the desired mm/pixel amount.
        center_xy = action.offset[:2]
        top_left = center_xy + np.array(
            [-self.crop_halfwidth_mm, -self.crop_halfwidth_mm]
        )
        top_right = center_xy + np.array(
            [-self.crop_halfwidth_mm, self.crop_halfwidth_mm]
        )
        bottom_right = center_xy + np.array(
            [self.crop_halfwidth_mm, self.crop_halfwidth_mm]
        )
        bottom_left = center_xy + np.array(
            [self.crop_halfwidth_mm, -self.crop_halfwidth_mm]
        )
        startpoints_robot = np.array([top_left, top_right, bottom_right, bottom_left])
        startpoints_uv = (
            self.calibration.uv_M_robot @ calibration.make_homog(startpoints_robot.T)
        ).T
        endpoints_robot = np.array(
            [
                [0, 0],
                [0, 2 * self.crop_halfwidth_mm],
                [self.crop_halfwidth_mm * 2, self.crop_halfwidth_mm * 2],
                [2 * self.crop_halfwidth_mm, 0],
            ]
        )
        # CV2 is very picky that these are float32s.
        M = cv2.getPerspectiveTransform(
            startpoints_uv.astype(np.float32), endpoints_robot.astype(np.float32)
        )
        cropped_before_image = cv2.warpPerspective(
            before_image,
            M,
            (self.crop_halfwidth_mm * 2, self.crop_halfwidth_mm * 2),
            borderValue=(255, 255, 255),
        )
        cropped_after_image = cv2.warpPerspective(
            after_image,
            M,
            (self.crop_halfwidth_mm * 2, self.crop_halfwidth_mm * 2),
            borderValue=(255, 255, 255),
        )

        # Convert to torch images. These want the channels first.
        cropped_before_image = torch.tensor(
            cropped_before_image.transpose([2, 0, 1]).astype(np.float32) / 255.0
        )
        cropped_after_image = torch.tensor(
            cropped_after_image.transpose([2, 0, 1]).astype(np.float32) / 255.0
        )
        action.offset[:] = 0.0

        if self.transform:
            cropped_before_image, cropped_after_image, action = self.transform(
                cropped_before_image, cropped_after_image, action
            )

        # Resize the cropped images to the latent image size
        resize_transform = torchvision.transforms.Resize(
            (self.output_image_size, self.output_image_size)
        )
        cropped_before_image = resize_transform(cropped_before_image)
        cropped_after_image = resize_transform(cropped_after_image)

        # Convert spline to a vector of (x, y, height) tuples, normalized to the output image bound.
        spline_params = torch.tensor(
            self.spline_transform(action).flatten().astype(np.float32)
        )

        return cropped_before_image, cropped_after_image, spline_params, pen_type_index


class StrokeRenderingDataset(Dataset):
    """
    Randomly generates splines and "renders" them to images.
    """

    def __init__(
        self,
        batch_size: int,
        latent_image_size: int = 128,
        max_stroke_width_fraction: float = 0.2,
        fixed_seeding: bool = False,
        num_stroke_time_samples: int = 32,
    ):
        self.latent_image_size = latent_image_size
        self.spline_generation_params = spline_generation.SplineGenerationParams()
        self.spline_transform = SplineToSamples.make_from_spline_generation_params(
            self.spline_generation_params
        )
        self.max_stroke_width_fraction = max_stroke_width_fraction
        self.batch_size = batch_size
        self.fixed_seeding = fixed_seeding

    @property
    def spline_vectorization_length(self):
        return 3 * self.spline_transform.num_stroke_time_samples

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        if self.fixed_seeding:
            rng = np.random.default_rng(idx)
        else:
            rng = None
        spline = spline_generation.make_random_spline(
            self.spline_generation_params, rng=rng
        )

        # Do simple rendering.
        im = np.full(
            (self.latent_image_size, self.latent_image_size, 1),
            fill_value=255,
            dtype=np.uint8,
        )
        N_samples = 30
        t = np.linspace(spline.x[0], spline.x[-1], N_samples)
        assert np.isclose(spline.x[0], 0.0), (
            "Stroke didn't start at t=0, but instead %f!" % spline.x[0]
        )
        assert (
            spline.x[-1] <= self.spline_transform.max_stroke_duration
        ), "Stroke has duration > max duration! %f vs %f" % (
            spline.x[-1],
            self.spline_transform.max_stroke_duration,
        )
        xs = spline(t)
        xs[:, :2] += self.latent_image_size / 2.0
        stroke_width = self.max_stroke_width_fraction * self.latent_image_size
        for k in range(xs.shape[0] - 1):
            im = cv2.line(
                im,
                xs[k, :2].astype(int),
                xs[k + 1, :2].astype(int),
                color=[0, 0, 0],
                thickness=int(xs[k, 2] * stroke_width + 1),
            )
        im = torch.tensor(im.astype(np.float32) / 255.0).permute([2, 0, 1])

        # Convert spline to a vector of (x, y, height) tuples, and normalize them all to the image bounds.
        spline_params = torch.tensor(
            self.spline_transform(
                spline_generation.SplineAndOffset(spline, np.zeros(3))
            )
            .flatten()
            .astype(np.float32)
        )
        return spline_params, im


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_directory", type=str, help="Dataset directory to process"
    )
    parser.add_argument("--dataset_assignment", type=str, default="train")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    spline_transform = SplineToSamples(
        max_stroke_duration=5.0,
        max_stroke_distance=100.0,
        num_stroke_time_samples=32,
    )
    dataset = StrokeDataset(
        args.dataset_directory,
        dataset_assignment=args.dataset_assignment,
        spline_transform=spline_transform,
        output_image_size=64,
        crop_halfwidth_mm=64,
        transform=StrokeDatasetRandomization(rng=np.random.default_rng(42)),
    )

    fig = plt.figure()
    gs = plt.GridSpec(3, 2, figure=fig)
    for k in range(3):
        before, after, spline_params, pen_type_index = dataset[k]
        spline_params = spline_params.numpy().reshape([-1, 3])
        ax = fig.add_subplot(gs[k, 0])
        im = before.numpy().transpose([1, 2, 0])
        ax.imshow(im)
        spline_params = spline_transform.invert(spline_params)
        # Account for the image rescaling.
        spline_params[:, :2] *= dataset.output_image_size / (
            dataset.crop_halfwidth_mm * 2
        )
        # Center the spline on the image.
        spline_params[:, :2] += dataset.output_image_size / 2.0
        plt.plot(spline_params[:, 0], spline_params[:, 1])
        ax = fig.add_subplot(gs[k, 1])
        ax.imshow(after.numpy().transpose([1, 2, 0]))
        plt.plot(spline_params[:, 0], spline_params[:, 1])
    plt.show()
