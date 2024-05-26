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
    def forward(self, images, trajectories, colors, out_images = None):
        '''
            images: N_batch x N_channels x N_rows x N_cols
            trajectories: N_batch x N_spline_pts x 4 [t, x, y, z] spline points
            colors: N_batch x 4 brush colors [rgba]
        '''
        ...


class BlittingStrokeModel(StrokeModel):
    '''
        Simple model of paint strokes that renders trajectories by blitting a sprite
        along the stroke trajectory, using an appealing oil model for paint mixing.
    '''
    def __init__(self, brush_path: str):
        super(BlittingStrokeModel, self).__init__()
        brush = imageio.imread(brush_path).astype(float) / 256.
        self.register_buffer("brush", numpy_images_to_torch(brush))

    def forward(self, images: torch.Tensor, coeffs: torch.Tensor, colors: torch.Tensor, ts: torch.Tensor, out_images = None):
        '''
            images: N_batch x 4 x N_rows x N_cols
            trajectories: N_batch x spline coeffs for use in NaturalCubicSpline
            colors: N_batch x 4 brush colors [rgba]
            ts: 
        '''
        if out_images is None:
            out_images = torch.empty_like(images)
        else:
            assert out_images.shape == images.shape
    
        b2p, p2b = oilpaint_converters(device=images.device)

        # Evaluate position and velocity at the knots of the spline
        splines = NaturalCubicSpline(coeffs)
        q = splines.evaluate(ts)
        qd = splines.derivative(ts, order=1)
        assert not torch.any(torch.isnan(q))
        assert not torch.any(torch.isnan(qd))

        # Render each image in our batch. Very slow as we're making this differentible...
        # this should just be a shader or something.
        for i in range(images.shape[0]):                        
            # Use xy's to figure out brush center, and direction of velocity
            # to figure out yaw
            poses = torch.stack([
                q[i, :, 0], q[i, :, 1], -torch.atan2(qd[i, :, 1], qd[i, :, 0])
            ], dim=1)
            brush = self.brush.clone()
            for k in range(3):
                brush[k, :, :] = colors[i, k]
            brush[3, :, :] *= colors[i, 3]
            brushes = brush.unsqueeze(0).expand((q.shape[1], *brush.shape))

            scales = torch.clip(q[i, :, 2], 1E-3, 1.)
            sprite_ims = draw_sprites_at_poses(
                brushes, poses, scales,
                out_images.shape[2], out_images.shape[3]
            )

            # Pass through oil transform
            image_pre_oil = b2p(images[i])
            sprite_ims_oil = b2p(sprite_ims)
        
            alphas = 1. - sprite_ims_oil[:, 3:, :, :]  # alphas inverted in absorb space
            inv_alphs = sprite_ims_oil[:, 3:, :, :]

            # Reduce down to one image
            image = image_pre_oil[:3, :, :]
            scaled_sprites = sprite_ims_oil[:, :3, :, :] * alphas
            for k, z in enumerate(q[i, :, 2]):
                # Skip those brush strokes that had no pressure.
                if z > 0:
                    image = image * inv_alphs[k, ...] + scaled_sprites[k, ...]
        
            # Save out rendered image
            out_images[i, ...] = p2b(image)

        return out_images
    

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
