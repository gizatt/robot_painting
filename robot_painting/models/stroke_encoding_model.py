"""
Model pieces for encoding a stroke from a flat vector input, and auxiliary
decoder and direct-supervision pieces for training that model.

                                                        ┌────────────────┐   
                                                        │                │   
                              (extra supervision)       │Rendered stroke │   
        ┌──────────────────────────────────────────────►│      image     │   
        │                                               │     NxNx1      │   
        │                                               │                │   
        │                                               └────────▲───────┘   
        │                                                        │           
        │                                                    CNN layers      
        │                                                        ▲           
        │                                               ┌────────┼───────┐   
        │                                               │                │   
 ┌──────┼──────┐                                        │  Stroke image  │   
 │Stroke params├─────► FCN+RELU layers ────► Reshape ── ►     NxNxM      ┼   
 └──────▲──────┘                                        │                │   
        │                                               └─────────┬──────┘   
        │                                                         │          
        │                                                         │          
   Reconstruction◄─────FCN+RELU layers◄──────Reshape◄─────────────┘          
      error                                                                  
"""

from typing import Iterable

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class StrokeEncoder(nn.Module):
    """
    Input: `batch_size` x `input_dimension`  vector describing the stroke.
    Output: `batch_size` images of size `encoded_image_size` with `encoded_image_channels` channels.
    """

    def __init__(
        self,
        input_dimension: int,
        encoded_image_size: torch.Size | int,
        encoded_image_channels: int,
        fc_sizes: Iterable[int] = [512, 1024],
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
        super().__init__()

        # Grow the input to the requested total size.
        self.fcs = torch.nn.ModuleList()
        last_vector_size = input_dimension
        for fc_size in fc_sizes:
            self.fcs.append(nn.Linear(last_vector_size, fc_size))
            last_vector_size = fc_size
        self.fcs.append(nn.Linear(last_vector_size, self.image_n_elements))

    def forward(self, x):
        for fc in self.fcs:
            x = F.relu(fc(x))
        return x.view(
            -1,
            self.encoded_image_channels,
            self.encoded_image_size[0],
            self.encoded_image_size[1],
        )


class StrokeDecoder(nn.Module):
    """
    Input: `batch_size` images of size `encoded_image_size` with `encoded_image_channels` channels.
    Output: `batch_size` x `input_dimension`  vector describing the stroke.
    """

    def __init__(
        self,
        output_dimension: int,
        encoded_image_size: torch.Size | int,
        encoded_image_channels: int,
        fc_sizes: Iterable[int] = [1024, 512],
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
        super().__init__()

        # Grow the input to the requested total size.
        self.fcs = torch.nn.ModuleList()
        last_vector_size = self.image_n_elements
        for fc_size in fc_sizes:
            self.fcs.append(nn.Linear(last_vector_size, fc_size))
            last_vector_size = fc_size
        self.fcs.append(nn.Linear(last_vector_size, output_dimension))

    def forward(self, x):
        x = x.view(-1, self.image_n_elements)
        for fc in self.fcs:
            x = F.relu(fc(x))
        return x


class StackedConvnets(nn.Module):
    """
    Input: `batch_size` images with `N_input_channels` channels.
    Output: `batch_size` images  with `N_output_channels` channels.
    """

    def __init__(
        self,
        N_input_channels: int,
        N_output_channels: int,
        convnet_channels: Iterable[int] = [
            3,
        ],
    ):
        super().__init__()

        self.N_input_channels = N_input_channels
        self.N_output_channels = N_output_channels
        self.kernel_size = 3

        self.convnets = torch.nn.ModuleList()
        last_n_channels = self.N_input_channels
        for n_channels in convnet_channels:
            self.convnets.append(
                nn.Conv2d(
                    last_n_channels,
                    n_channels,
                    kernel_size=self.kernel_size,
                    padding="same",
                )
            )
            last_n_channels = n_channels
        self.convnets.append(
            nn.Conv2d(
                last_n_channels,
                N_output_channels,
                kernel_size=self.kernel_size,
                padding="same",
            )
        )

    def forward(self, x):
        for convnet in self.convnets:
            x = F.relu(convnet(x))
        return x


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
    ):
        super().__init__()
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
        self.rendering_loss_weight = 1.0

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
            rendering_loss = nn.functional.mse_loss(
                reconstructed_stroke_image, rendered_stroke_image
            )
            self.log(f"{step_name}_rendering_loss", rendering_loss)
            total_loss += rendering_loss * self.rendering_loss_weight

            if log_images and self.logger is not None:
                N_images = min(reconstructed_stroke_image.shape[0], 9)

                rendered_stroke_image_for_viz = rendered_stroke_image[:N_images]
                grid = torchvision.utils.make_grid(rendered_stroke_image_for_viz, nrow=3)
                self.logger.experiment.add_image(f"{step_name}_rendered_stroke_image", grid, self.current_epoch)
                
                reconstructed_stroke_image_for_viz = reconstructed_stroke_image[:N_images]
                grid = torchvision.utils.make_grid(reconstructed_stroke_image_for_viz, nrow=3)
                self.logger.experiment.add_image(f"{step_name}_reconstructed_stroke_image", grid, self.current_epoch)
                

        self.log(f"{step_name}_total_loss", total_loss)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, step_name="train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, step_name="val", log_images=True)
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, step_name="test", log_images=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
