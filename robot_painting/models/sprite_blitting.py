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


def oilpaint_converters(device=torch.device('cpu')):
    submix, b2p, p2b = submix_power(torch.tensor([13, 3, 7.], device=device))
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

def compose_tf_matrices(a_T_b, b_T_c):
    # [R t] * [R]
    assert a_T_b.shape[0] == b_T_c.shape[0]
    a_T_c = torch.empty(a_T_b.shape, dtype=a_T_b.dtype, device=a_T_b.device)
    a_T_c[:, 0:2, 0:2] = torch.bmm(a_T_b[:, 0:2, 0:2], b_T_c[:, 0:2, 0:2])
    a_T_c[:, :, 2] = torch.bmm(a_T_b[:, 0:2, 0:2], b_T_c[:, :, 2].unsqueeze(2)).squeeze(2) + a_T_b[:, :, 2]
    return a_T_c

def invert_tf_matrix(tf):
    ''' Inverts a nx2x3 affine TF matrix. '''
    new_tf = torch.empty(tf.shape, dtype=tf.dtype, device=tf.device)
    new_tf[:, 0:2, 0:2] = torch.linalg.inv(tf[:, 0:2, 0:2])
    new_tf[:, :, 2] = -1.*torch.bmm(
        new_tf[:, 0:2, 0:2],
        tf[:, 0:2, 2].unsqueeze(2)).squeeze(2)
    return new_tf

def draw_sprites_at_poses(sprites, im_T_sprites, scales, image_rows, image_cols):
    '''
    Given a batch of sprites (n x n_channels x sprite_size_x x sprite_size_y), 
    a batch of x-y-theta poses (n x 3), scales (n), and an output image size 
    returns a [n x image_size_x x image_size_y batch of images] from
    rendering those sprites at those locations.

    Poses are defined as offsets in pixels + angle, in theta: [x, y, theta]. We'll
    put the *center* of the sprite at that pose, rotated about the center by that
    rotation, scaled from its original size by <scale>.

    Uses Spatial Transformer Network, heavily referencing the Pyro AIR tutorial.
    '''

    n = sprites.size(0)
    n_channels = sprites.size(1)
    assert sprites.dim() == 4
    sprite_rows = sprites.size(2)
    sprite_cols = sprites.size(3)

    im_unit_dims_T_im = torch.tensor([[
        [1. / (float(image_cols) / 2.), 0., -1.],
        [0., 1. / (float(image_rows) / 2.), -1.],
    ]], device=sprites.device).expand(n, -1, -1)
    sprite_unit_dims_T_sprite = torch.tensor([[
        [1. / (float(sprite_cols) / 2.), 0., -1.],
        [0., 1. / (float(sprite_rows) / 2.), -1.]
    ]], device=sprites.device).expand(n, -1, -1)

    # Flip x and y, since "x" is the column dim in `affine_grid`. Offset for
    # sprite size.
    im_T_sprite_centers = convert_pose_to_matrix(
        torch.stack([im_T_sprites[:, 1], im_T_sprites[:, 0], im_T_sprites[:, 2]], dim=1)
    )
    
    # sprite_centers_T_sprite = translation of [-sprite_cols * scales, -sprite_rows * scales]
    sprite_centers_T_sprites = convert_pose_to_matrix(
        torch.stack([
            torch.full_like(scales, -sprite_cols / 2.), torch.full_like(scales, -sprite_rows / 2.), torch.full_like(scales, 0)], dim=1)
    )
    # Apply scaling
    sprite_centers_T_sprites[:, 0:2, 0:3] *= scales.view(-1, 1, 1).expand(-1, 2, 3)

    im_T_sprites = compose_tf_matrices(im_T_sprite_centers, sprite_centers_T_sprites)
    im_unit_dims_T_sprites = compose_tf_matrices(im_unit_dims_T_im, im_T_sprites)
    sprites_T_im_unit_dims = invert_tf_matrix(im_unit_dims_T_sprites)
    sprites_unit_dims_T_im_unit_dims = compose_tf_matrices(sprite_unit_dims_T_sprite, sprites_T_im_unit_dims)

    grid = F.affine_grid(sprites_unit_dims_T_im_unit_dims, torch.Size((n, n_channels, image_rows, image_cols)), align_corners=False)
    out = F.grid_sample(sprites, grid,
                        padding_mode="zeros", mode="bilinear", align_corners=False)
    return out.view(n, n_channels, image_rows, image_cols)


def numpy_images_to_torch(images):
    # Torch generally wants ordering [channels, x, y]
    # while numpy as has [x, y, channels]
    if isinstance(images, np.ndarray) and len(images.shape) == 3:
        return torch.Tensor(images).permute(2, 0, 1)
    else:
        return torch.stack([torch.Tensor(image).permute(2, 0, 1)
                            for image in images], dim=0)


def torch_images_to_numpy(images):
    if images.dim() == 3:
        return images.permute(1, 2, 0).cpu().detach().numpy()
    
    assert images.dim() == 4, \
           "Images must be batched n x channels x x_dim x y_dim"
    images_out = []
    for k in range(images.shape[0]):
        images_out.append(
            images[k, ...].permute(1, 2, 0).cpu().detach().numpy())
    return np.stack(images_out, axis=0)


if __name__ == "__main__":
    # debug for grid sample
    out_rows = 5
    out_cols = 7

    image_to_sample = torch.arange(110).float().reshape((1, 1, 10, 11))

    im_to_sample_T_im_to_produce = torch.tensor([[
        [0.1, 0., 0.1],
        [0., 0.1, 0.]
    ]])
    print("im_to_produce_T_im_to_sample: ", invert_tf_matrix(im_to_sample_T_im_to_produce))
    grid = F.affine_grid(im_to_sample_T_im_to_produce, torch.Size((1, 1, out_rows, out_cols)), align_corners=False)
    print(grid)
    out = F.grid_sample(image_to_sample, grid)
    print(out.shape)
    