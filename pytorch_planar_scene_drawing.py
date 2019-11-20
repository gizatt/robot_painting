import imageio
import numpy as np
import os
import time
import torch
import torch.nn.functional as F

device = torch.device('cpu')

def submix_power(i):
    # From https://github.com/ctmakro/opencv_playground,
    # converted to torch.
    def BGR2PWR(c):
        # no overflow allowed
        c = torch.clamp(c, min=1e-6, max=1.-1e-6)
        for k in range(3):
            c[..., k, :, :] = torch.pow(c[..., k, :, :], 2.2/i[k])
        u = 1. - c
        return u  # unit absorbance

    def PWR2BGR(u):
        c = 1. - u
        for k in range(3):
            c[..., k, :, :] = torch.pow(c[..., k, :, :], i[k]/2.2)
        return c  # rgb color

    def submix(c1, c2, ratio):
        uabs1, uabs2 = BGR2PWR(c1), BGR2PWR(c2)
        mixuabs = (uabs1 * ratio) + (uabs2*(1-ratio))
        return PWR2BGR(mixuabs)

    return submix, BGR2PWR, PWR2BGR


def oilpaint_converters(device):
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


def invert_tf_matrix(tf):
    ''' Inverts a nx2x3 affine TF matrix. '''
    new_tf = torch.empty(tf.shape, dtype=tf.dtype)
    new_tf[:, 0:2, 0:2] = tf[:, 0:2, 0:2].transpose(dim0=1, dim1=2)
    new_tf[:, :, 2] = -1.*torch.bmm(
        new_tf[:, 0:2, 0:2],
        tf[:, 0:2, 2].unsqueeze(2)).squeeze(2)
    return new_tf


def draw_sprites_at_poses(pose, sprite_rows, sprite_cols,
                          image_rows, image_cols, sprites):
    '''
    Given a batch of poses (n x 3), a shared sprite + output image size,
    and a batch of sprites (n x n_channels x sprite_size_x x sprite_size_y),
    returns a n x image_size_x x image_size_y batch of images from
    rendering those sprites at those locations.

    Poses are defined as offsets from the image center, in pixels,
    and angle, in theta: [x, y, theta]

    Uses Spatial Transformer Network, heavily referencing the Pyro AIR
    tutorial.
    '''

    n = sprites.size(0)
    n_channels = sprites.size(1)
    assert sprites.dim() == 4
    assert sprites.size(2) == sprite_rows, 'Sprite input size x mismatch.'
    assert sprites.size(3) == sprite_cols, 'Sprite input size y mismatch.'

    tf = convert_pose_to_matrix(pose)
    tf = invert_tf_matrix(tf)

    # Scale down the offset to move by pixel rather than
    # by a factor of the half image size
    # (A movement of "+1" moves the sprite by image_size/2 in the given
    # dimension).
    # X and Y are flipped here, as the "x" dim is the column dimension
    tf[:, 0, 2] /= (float(sprite_cols) / 2.)
    tf[:, 1, 2] /= (float(sprite_rows) / 2.)
    # And scale the whole transformation by the ratio of sprite to image
    # ratio -- the default affine_grid will fill the output image
    # with the input image.
    tf[:, 0, 0:2] /= (float(sprite_cols) / float(image_cols))
    tf[:, 1, 0:2] /= (float(sprite_rows) / float(image_rows))
    print("Rows: ", sprite_rows, image_rows)
    print("Cols: ", sprite_cols, image_cols)

    print(sprites.shape)
    grid = F.affine_grid(tf.cuda(), torch.Size((n, n_channels, image_rows, image_cols)))
    out = F.grid_sample(sprites, grid,
                        padding_mode="zeros", mode="nearest")
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
    return images_out


if __name__ == "__main__":

    # Test out by generating some sprites and drawing them into a larger image.
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    print("CUDA? ", torch.cuda.is_available())

    demo_image = "none" # "data/kara.png"
    if os.path.isfile(demo_image):
        sprite_1 = imageio.imread("data/kara.png").astype(np.float)/256.
        sprite_2 = imageio.imread("data/kara.png").astype(np.float)/256.
        sprite_rows, sprite_cols, n_channels = sprite_1.shape
    else:
        sprite_rows = 100
        sprite_cols = 100
        n_channels = 4
        sprite_1 = np.zeros([sprite_rows, sprite_cols, n_channels])
        sprite_1[:, :, 3] = 0.75
        sprite_red = sprite_1.copy()
        sprite_red[:, :, 0] = 1.
        sprite_blue = sprite_1.copy()
        sprite_blue[:, :, 2] = 1.
        sprite_green = sprite_1.copy()
        sprite_green[:, :, 1] = 1.

    image_rows = 1000
    image_cols = 1000
    image_pre = torch.ones(1, n_channels, image_rows, image_cols)
    sprite_poses = torch.stack(
        [torch.Tensor([torch.randint(low=-500, high=500, size=(1,)),
                       torch.randint(low=-500, high=500, size=(1,)),
                       torch.rand(1)*np.pi*2.]) for k in range(100)])
    print(sprite_poses.shape)
    print("Starting...")
    start_time = time.time()
    sprite_ims = draw_sprites_at_poses(
          sprite_poses,
          sprite_rows, sprite_cols,
          image_rows, image_cols,
          numpy_images_to_torch([sprite_red, sprite_blue, sprite_green]*100)[:100, :, :, :].cuda())
    print("done in %f seconds " % (time.time() - start_time))

    start_time = time.time()

    # Reduce down to one image
    b2p, p2b = oilpaint_converters(device=torch.device('cuda'))
    image_pre_oil = b2p(image_pre.cuda())
    sprite_ims_oil = b2p(sprite_ims.cuda())
    # Permute for easier alpha combo
    image = image_pre_oil[0, :3, :, :]
    for k in range(sprite_ims_oil.shape[0]):
        alpha_map = 1. - sprite_ims_oil[k, 3, :, :]  # alphas inverted in absorb space
        image = image * (1 - alpha_map) + sprite_ims_oil[k, :3, :, :] * alpha_map
    result = p2b(image)

    print("Reductions done in %f seconds " % (time.time() - start_time))
    image = torch_images_to_numpy(result.unsqueeze(0).cpu())[0]

    plt.imshow(image)
    plt.show()