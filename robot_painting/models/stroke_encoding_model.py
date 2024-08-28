"""
Model pieces for encoding a stroke from a flat vector input, and auxiliary
decoder and direct-supervision pieces for training that model.

See `MODEL_ARCHITECTURE.md`.     
"""

from typing import Iterable

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class StrokeEncoder(nn.Sequential):
    """
    Input: `batch_size` x `input_dimension`  vector describing the stroke.
    Output: `batch_size` images of size `encoded_image_size` with `encoded_image_channels` channels.
    """

    def __init__(
        self,
        input_dimension: int,
        encoded_image_size: torch.Size | int,
        encoded_image_channels: int,
        fc_sizes: Iterable[int] = [128, 256, 512],
    ):
        if isinstance(encoded_image_size, int):
            encoded_image_size = torch.Size([encoded_image_size, encoded_image_size])
        assert len(encoded_image_size) == 2
        self.encoded_image_size = encoded_image_size
        self.encoded_image_channels = encoded_image_channels
        self.image_n_elements = (
            self.encoded_image_size[0]
            * self.encoded_image_size[1]
            * self.encoded_image_channels
        )

        # Grow the input to the requested total size.
        layers = []
        last_vector_size = input_dimension
        for fc_size in fc_sizes:
            layers.append(nn.Linear(last_vector_size, fc_size))
            layers.append(nn.ReLU())
            last_vector_size = fc_size 
        layers.append(nn.Linear(last_vector_size, self.image_n_elements))
        layers.append(
            nn.Unflatten(
                dim=1,
                unflattened_size=(
                    self.encoded_image_channels,
                    self.encoded_image_size[0],
                    self.encoded_image_size[1],
                ),
            )
        )
        super().__init__(*layers)


class StrokeDecoder(nn.Sequential):
    """
    Input: `batch_size` images of size `encoded_image_size` with `encoded_image_channels` channels.
    Output: `batch_size` x `input_dimension`  vector describing the stroke.
    """

    def __init__(
        self,
        output_dimension: int,
        encoded_image_size: torch.Size | int,
        encoded_image_channels: int,
        fc_sizes: Iterable[int] = [512, 256, 128],
    ):
        if isinstance(encoded_image_size, int):
            encoded_image_size = torch.Size([encoded_image_size, encoded_image_size])
        assert len(encoded_image_size) == 2
        self.encoded_image_size = encoded_image_size
        self.encoded_image_channels = encoded_image_channels
        self.image_n_elements = (
            self.encoded_image_size[0]
            * self.encoded_image_size[1]
            * self.encoded_image_channels
        )

        # Grow the input to the requested total size.
        layers = []
        layers.append(
            nn.Flatten()
        )
        last_vector_size = self.image_n_elements
        for fc_size in fc_sizes:
            layers.append(nn.Linear(last_vector_size, fc_size))
            layers.append(nn.ReLU())
            last_vector_size = fc_size
        layers.append(nn.Linear(last_vector_size, output_dimension))
        super().__init__(*layers)


class StackedConvnets(nn.Sequential):
    """
    Input: `batch_size` images with `N_input_channels` channels.
    Output: `batch_size` images  with `N_output_channels` channels.
    """

    def __init__(
        self,
        N_input_channels: int,
        N_output_channels: int,
        convnet_channels: Iterable[int] = [
            16, 32, 16,
        ],
    ):
        self.N_input_channels = N_input_channels
        self.N_output_channels = N_output_channels
        self.kernel_size = 1

        layers = []
        last_n_channels = self.N_input_channels
        for n_channels in convnet_channels:
            layers.append(
                nn.Conv2d(
                    last_n_channels,
                    n_channels,
                    kernel_size=self.kernel_size,
                    padding="same",
                )
            )
            layers.append(nn.ReLU())
            last_n_channels = n_channels
        layers.append(
            nn.Conv2d(
                last_n_channels,
                N_output_channels,
                kernel_size=self.kernel_size,
                padding="same",
            )
        )
        super().__init__(*layers)


class StrokeSupervisedAutoEncoder(L.LightningModule):
    """
    Training module for a standalone stroke encoder with optional direct
    supervision of stroke rendering from the encoded image.
    """

    def __init__(
        self,
        stroke_parameterization_size: int,
        encoded_image_size: int,
        encoded_image_channels: 3,
        with_stroke_rendering: bool = True,
        lr: float = 1E-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.encoder = StrokeEncoder(
            input_dimension=stroke_parameterization_size,
            encoded_image_size=encoded_image_size,
            encoded_image_channels=encoded_image_channels,
        )
        self.decoder = StrokeDecoder(
            output_dimension=stroke_parameterization_size,
            encoded_image_size=encoded_image_size,
            encoded_image_channels=encoded_image_channels,
        )
        if with_stroke_rendering:
            self.encoded_image_converter = StackedConvnets(
                N_input_channels=encoded_image_channels, N_output_channels=1
            )
        else:
            self.encoded_image_converter = None

        self.autoencoder_loss_weight = 1.0
        self.rendering_loss_weight = 10.0

    def shared_step(self, batch, batch_idx, step_name: str, log_images: bool = False):
        # training_step defines the train loop.
        # it is independent of forward
        spline_params, rendered_stroke_image = batch

        encoded_stroke_image = self.encoder(spline_params)
        reconstructed_spline_params = self.decoder(encoded_stroke_image)
        autoencoder_loss = nn.functional.mse_loss(
            reconstructed_spline_params, spline_params
        )
        self.log(f"{step_name}_autoencoder_loss", autoencoder_loss)
        total_loss = autoencoder_loss * self.autoencoder_loss_weight

        if self.encoded_image_converter is not None:
            reconstructed_stroke_image = self.encoded_image_converter(
                encoded_stroke_image
            )
            rendering_loss = nn.functional.binary_cross_entropy_with_logits(
                reconstructed_stroke_image, rendered_stroke_image
            )
            self.log(f"{step_name}_rendering_loss", rendering_loss)
            total_loss += rendering_loss * self.rendering_loss_weight

            if log_images and self.logger is not None:
                N_images = min(reconstructed_stroke_image.shape[0], 9)

                rendered_stroke_image_for_viz = rendered_stroke_image[:N_images]
                
                grid = torchvision.utils.make_grid(
                    rendered_stroke_image_for_viz, nrow=3
                )
                self.logger.experiment.add_image(
                    f"{step_name}_rendered_stroke_image", grid, self.current_epoch
                )

                reconstructed_stroke_image_for_viz = reconstructed_stroke_image[
                    :N_images
                ]
                grid = torchvision.utils.make_grid(
                    reconstructed_stroke_image_for_viz, nrow=3
                )
                self.logger.experiment.add_image(
                    f"{step_name}_reconstructed_stroke_image", grid, self.current_epoch
                )

        self.log(f"{step_name}_total_loss", total_loss)
        return total_loss

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=1E-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=1000, gamma=0.5
        )
        return [optimizer], [scheduler]
