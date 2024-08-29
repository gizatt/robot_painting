"""
Model pieces for encoding a stroke from a flat vector input, and auxiliary
decoder and direct-supervision pieces for training that model. See
MODEL_ARCHITECTURE.md for more details.
"""

from typing import Iterable

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from robot_painting.models.stroke_dataset import StrokeDataset
from robot_painting.models.stroke_encoding_model import StrokeEncoder


class StrokePredictionConvnet(nn.Sequential):
    """
    Takes an encoded stroke image, a before image, and a pen embedding feature, and predicts an after image.
    """

    # Pair of (n_channels, kernel_size) for intermediate conv layers. A final
    # 1x1 conv layer will reduce to 3 channels.
    DEFAULT_CONV_LAYERS = [
        (64, 7),
        (128, 5),
        (64, 3),
        (32, 3),
        (16, 3),
        (8, 3),
        (3, 3),
    ]

    def __init__(
        self,
        image_size: int,
        encoded_image_channels: int,
        pen_feature_size: int,
        conv_layers: list[tuple[int, int]] = DEFAULT_CONV_LAYERS,
    ):
        self.image_size = image_size
        self.encoded_image_channels = encoded_image_channels
        self.n_input_channels = encoded_image_channels + 3 + pen_feature_size
        layers = []
        in_channels = self.n_input_channels
        for i, (out_channels, kernel_size) in enumerate(conv_layers):
            layers.append(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=kernel_size, padding="same"
                )
            )
            in_channels = out_channels
        layers.append(nn.Conv2d(in_channels, 3, kernel_size=1, padding="same"))
        super().__init__(*layers)

    def forward(
        self,
        encoded_stroke_image: torch.Tensor,
        before_image: torch.Tensor,
        pen_embedding: torch.Tensor,
    ) -> torch.Tensor:
        # Ensure encoded_stroke_image and before_image have the expected size
        assert (
            encoded_stroke_image.shape[2:] == before_image.shape[2:]
        ), f"Encoded stroke image size {encoded_stroke_image.shape[2:]} does not match before image size {before_image.shape[2:]}"
        assert (
            encoded_stroke_image.shape[1] == self.encoded_image_channels
        ), f"Expected {self.encoded_image_channels} channels for encoded stroke image, but got {encoded_stroke_image.shape[1]}"
        assert (
            before_image.shape[1] == 3
        ), f"Expected 3 channels for before image, but got {before_image.shape[1]}"

        # Tile pen_embedding to match image dimensions
        pen_embedding_tiled = pen_embedding[:, :, None, None].expand(
            -1, -1, *encoded_stroke_image.shape[2:]
        )

        # Concatenate along channel dimension
        x = torch.cat([encoded_stroke_image, before_image, pen_embedding_tiled], dim=1)

        assert (
            x.size(1) == self.n_input_channels
        ), f"Expected {self.n_input_channels} channels, but got {x.size(1)}"
        return super().forward(x)


class StrokePredictionModel(L.LightningModule):
    """
    Training module for a stroke predictor.
    """

    def __init__(
        self,
        stroke_encoder: StrokeEncoder,
        stroke_dataset: StrokeDataset,
        stroke_parameterization_size: int,
        pen_feature_size: int,
        image_size: int,
        encoded_image_channels: 3,
    ):
        super().__init__()
        self.stroke_encoder = stroke_encoder
        self.pen_embedding = nn.Embedding(
            num_embeddings=len(stroke_dataset.pen_type_to_index),
            embedding_dim=pen_feature_size,
        )
        self.prediction_model = StrokePredictionConvnet(
            image_size=image_size,
            encoded_image_channels=encoded_image_channels,
            pen_feature_size=pen_feature_size,
        )

    def shared_step(self, batch, batch_idx, step_name: str, log_images: bool = False):
        before_image, after_image, spline_params, pen_type_index = batch
        encoded_stroke_image = self.encoder(spline_params)
        pen_embedding = self.pen_embedding(pen_type_index)
        predicted_after_image = self.prediction_model(
            encoded_stroke_image, before_image, pen_embedding
        )

        reconstruction_loss = nn.functional.mse_loss(predicted_after_image, after_image)
        self.log(f"{step_name}_reconstruction_loss", reconstruction_loss)

        if log_images and self.logger is not None:
            N_images = min(predicted_after_image.shape[0], 9)

            predicted_after_image_for_viz = predicted_after_image[:N_images]

            grid = torchvision.utils.make_grid(predicted_after_image_for_viz, nrow=3)
            self.logger.experiment.add_image(
                f"{step_name}_predicted_after_image", grid, self.current_epoch
            )

            after_image_for_viz = after_image[:N_images]
            grid = torchvision.utils.make_grid(after_image_for_viz, nrow=3)
            self.logger.experiment.add_image(
                f"{step_name}_after_image", grid, self.current_epoch
            )

        return reconstruction_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, step_name="train", log_images=True)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, step_name="val", log_images=True)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, step_name="test", log_images=True)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = L.pytorch.utilities.grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=1000, gamma=0.1
        )
        return [optimizer], [scheduler]
