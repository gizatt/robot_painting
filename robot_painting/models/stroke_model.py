'''
Note: Generated with help from with GPT-4, 20230721.
'''

from abc import ABC, abstractmethod
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import typing
import glob
import logging
import imageio.v2 as imageio
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

LOG = logging.getLogger()

from .sprite_blitting import (
    numpy_images_to_torch,
    torch_images_to_numpy,
    draw_sprites_at_poses,
    oilpaint_converters
)

class StrokeModel(nn.Module, ABC):
    '''
        Definition of trajectory:
            A trajectory should be an `N_knots x 4` tensor, with fields [t, x, y, z].
            `t` is in seconds, and should be strictly ascending. `x` and `y` are in arbitrary
            coordinates, usually pixels. `z` is height (or pressure), with 0 being "no pressure"
            and 1 being "full pressure".
        '''
    @abstractmethod
    def forward(self, images, trajectories, out_images = None):
        '''
            images: N_batch x N_channels x N_rows x N_cols
            trajectories: N_batch x N_knots x 4
        '''
        ...



    



class BlittingStrokeModel(StrokeModel):
    '''
        Simple model of paint strokes that renders trajectories by blitting a sprite
        along the stroke trajectory, using an appealing oil model for paint mixing.
    '''
    def __init__(self, brush_search_paths: typing.Iterable[str]):
        super(BlittingStrokeModel, self).__init__()
        self.load_brushes(brush_search_paths)

    def load_brushes(self, brush_search_paths: typing.Iterable[str]):
        # Open every image in the brush dir`
        brush_paths = sum([glob.glob(sp) for sp in brush_search_paths], [])
        LOG.info(f"Found brushes: {brush_paths}")

        raw_brushes = [imageio.imread(brush_path).astype(float) / 256. for brush_path in brush_paths]
        self.register_buffer("brushes", numpy_images_to_torch(raw_brushes))

    def forward(self, images, trajectories, out_images = None):
        '''
            images: N_batch x N_channels x N_rows x N_cols
            trajectories: N_batch x N_knots x 4

            N_channels should match the preloaded brushes.
        '''
        if out_images is None:
            out_images = torch.zeros_like(images)
        else:
            assert out_images.shape == images.shape
    
        b2p, p2b = oilpaint_converters(device=self.device)

        # For each image in our batch...
        for i in range(images.shape[0]):
            # Turn the trajectory into a series of poses by 

            # Loop over the points in the trajectory
            for j in range(1, trajectory.shape[0]):
                rr, cc, val = line_aa(*trajectory[j-1], *trajectory[j])

                # Modify the image along the line
                for k in range(-self.line_width, self.line_width + 1):
                    image[rr + k, cc] = 1

            # Add the resulting image to the tensor
            resulting_images[i] = torch.from_numpy(image)

        return resulting_images
    
class NeuralStrokeModel(StrokeModel):
    def __init__(self):
        super(NeuralStrokeModel, self).__init__()

        # Convolutional layers to encode the patches
        self.conv = nn.Sequential(
            # Add your convolutional layers here
            # The output should have `conv_out_features` features
        )

        # Fully connected layer to further encode the patches
        self.fc = nn.Linear(conv_out_features + trajectory_features, fc_out_features)

        # Transformer encoder to process the encoded patches
        transformer_layer = TransformerEncoderLayer(transformer_features, 8)
        self.transformer = TransformerEncoder(transformer_layer, transformer_layers)

    def forward(self, images, trajectories):
        # Create an ImagePatcher
        image_patcher = ImagePatcher(patch_size=5)  # For example

        # Extract the patches from the images
        patches = image_patcher.extract_patches(images, trajectories)

        # Apply the convolutional layers to the patches
        x = self.conv(patches)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Combine the encoded patches and trajectory parameters
        x = torch.cat((x, trajectories), dim=1)

        # Apply the fully connected layer
        x = self.fc(x)

        # Apply the transformer encoder
        x = self.transformer(x)

        return x
