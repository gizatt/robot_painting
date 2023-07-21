import glob
import time

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def submix_power(i):
    # From https://github.com/ctmakro/opencv_playground,
    # converted to torch.
    def BGR2PWR(c):
        # no overflow allowed
        c = torch.clamp(c, min=1e-6, max=1.-1e-6)
        c_new = torch.empty_like(c)
        for k in range(3):
            c_new[..., k, :, :] = torch.pow(c[..., k, :, :], 2.2/i[k])
        if c.shape[1] > 3:
            c_new[..., 3:, :, :] = c[..., 3:, :, :]
        u = 1. - c_new
        return u  # unit absorbance

    def PWR2BGR(u):
        c = 1. - u
        c_new = torch.empty_like(c)
        for k in range(3):
            c_new[..., k, :, :] = torch.pow(c[..., k, :, :], i[k]/2.2)
        if c.shape[1] > 3:
            c_new[..., 3:, :, :] = c[..., 3:, :, :]
        return c_new  # rgb color

    def submix(c1, c2, ratio):
        uabs1, uabs2 = BGR2PWR(c1), BGR2PWR(c2)
        mixuabs = (uabs1 * ratio) + (uabs2*(1-ratio))
        return PWR2BGR(mixuabs)

    return submix, BGR2PWR, PWR2BGR


def oilpaint_converters():
    submix, b2p, p2b = submix_power(torch.tensor([13, 3, 7.]))
    return b2p, p2b



def convert_pose_to_matrix(pose):
    '''
     Converts a batch of [x, y, theta] poses
     to a tf matrix:
     [[cos(theta) -sin(theta) x]]
      [sin(theta) cos(theta)  y]]
     Input is n x 3, output is n x 2 x 3.
    '''
    n = pose.size(0)
    out = torch.cat((torch.cos(pose[:, 2]).view(n, -1),
                     -torch.sin(pose[:, 2]).view(n, -1),
                     pose[:, 0].view(n, -1),
                     torch.sin(pose[:, 2]).view(n, -1),
                     torch.cos(pose[:, 2]).view(n, -1),
                     pose[:, 1].view(n, -1)), 1)
    out = out.view(n, 2, 3)
    return out


def invert_tf_matrix(tf):
    ''' Inverts a nx2x3 affine TF matrix. '''
    new_tf = torch.empty(tf.shape, dtype=tf.dtype)
    new_tf[:, 0:2, 0:2] = tf[:, 0:2, 0:2].transpose(dim0=1, dim1=2)
    new_tf[:, :, 2] = -1.*torch.bmm(
        tf[:, 0:2, 0:2].transpose(dim0=1, dim1=2),
        tf[:, 0:2, 2].unsqueeze(2)).squeeze(2)
    return new_tf


def draw_sprites_at_poses(sprites, im_T_sprites, scales, image_rows, image_cols):
    '''
    Given a batch of sprites (n x n_channels x sprite_size_x x sprite_size_y), 
    a batch of x-y-theta poses (n x 3), scales (n), and an output image size 
    returns a [n x image_size_x x image_size_y batch of images] from
    rendering those sprites at those locations.

    Poses are defined as offsets in pixels + angle, in theta: [x, y, theta]

    Uses Spatial Transformer Network, heavily referencing the Pyro AIR tutorial.
    '''

    n = sprites.size(0)
    n_channels = sprites.size(1)
    assert sprites.dim() == 4
    sprite_rows = sprites.size(2)
    sprite_cols = sprites.size(3)

    im_T_sprites = convert_pose_to_matrix(im_T_sprites)
    sprites_T_im = invert_tf_matrix(im_T_sprites)
    # Scale down the offset to move by pixel rather than
    # by a factor of the half image size
    # (A movement of "+1" moves the sprite by image_size/2 in the given
    # dimension).
    # X and Y are flipped here, as the "x" dim is the column dimension
    sprites_T_im[:, 0, 2] = sprites_T_im[:, 0, 2] / (float(sprite_cols) / 2.) - 0.5
    sprites_T_im[:, 1, 2] = sprites_T_im[:, 1, 2] / (float(sprite_rows) / 2.) - 0.5
    # And scale the whole transformation by the ratio of sprite to image
    # ratio -- the default affine_grid will fill the output image
    # with the input image.
    sprites_T_im[:, 0, 0:2] /= (float(sprite_cols) / float(image_cols))
    sprites_T_im[:, 1, 0:2] /= (float(sprite_rows) / float(image_rows))

    # Apply global sprite scaling
    sprites_T_im[:, 0:2, 0:3] /= scales.view(-1, 1, 1).repeat(1, 2, 3)

    grid = F.affine_grid(sprites_T_im, torch.Size((n, n_channels, image_rows, image_cols)), align_corners=False)
    out = F.grid_sample(sprites, grid,
                        padding_mode="zeros", mode="bilinear", align_corners=False)
    return out.view(n, n_channels, image_rows, image_cols)


def numpy_images_to_torch(images):
    # Torch generally wants ordering [channels, x, y]
    # while numpy as has [x, y, channels]
    return torch.stack([torch.Tensor(image).permute(2, 0, 1)
                        for image in images], dim=0)


def torch_images_to_numpy(images):
    assert images.dim() == 4, \
           "Images must be batched n x channels x x_dim x y_dim"
    images_out = []
    for k in range(images.shape[0]):
        images_out.append(
            images[k, ...].permute(1, 2, 0).cpu().detach().numpy())
    return np.stack(images_out, axis=0)