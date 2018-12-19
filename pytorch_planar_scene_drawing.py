import imageio
import numpy as np
import os
import torch
import torch.nn.functional as F

device = torch.device('cpu')


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


def draw_sprites_at_poses(pose, sprite_size_x, sprite_size_y,
                          image_size_x, image_size_y, sprites):
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
    assert sprites.size(2) == sprite_size_x, 'Sprite input size x mismatch.'
    assert sprites.size(3) == sprite_size_y, 'Sprite input size y mismatch.'

    tf = convert_pose_to_matrix(pose)
    tf = invert_tf_matrix(tf)

    # Scale down the offset to move by pixel rather than
    # by a factor of the half image size
    # (A movement of "+1" moves the sprite by image_size/2 in the given
    # dimension).
    tf[:, 0, 2] /= float(sprite_size_x)
    tf[:, 1, 2] /= float(sprite_size_y)
    # And scale the whole transformation by the ratio of sprite to image
    # ratio -- the default affine_grid will fill the output image
    # with the input image.
    tf[:, 0, 0:2] /= (float(sprite_size_x)/float(image_size_x))
    tf[:, 1, 0:2] /= (float(sprite_size_y)/float(image_size_y))

    grid = F.affine_grid(tf, torch.Size((n, n_channels, image_size_x, image_size_y)))
    out = F.grid_sample(sprites.view(n, n_channels, sprite_size_x, sprite_size_y), grid,
                        padding_mode="zeros", mode="nearest")
    return out.view(n, n_channels, image_size_x, image_size_y)


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
    poses = torch.FloatTensor([[0., 1., 2.],
                               [1.57, 4., 5.]]).view(2, -1)
    print "Poses: ", poses
    print "Pose matrix: ", convert_pose_to_matrix(poses)

    # Test out by generating some sprites and drawing them into a larger image.
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    demo_image = "data/kara.png"
    if os.path.isfile(demo_image):
        sprite_1 = imageio.imread("data/kara.png").astype(np.float)/256.
        sprite_2 = imageio.imread("data/kara.png").astype(np.float)/256.
        sprite_size_x, sprite_size_y, n_channels = sprite_1.shape
    else:
        sprite_size_x = 100
        sprite_size_y = 100
        n_channels = 3
        sprite_1 = np.ones([sprite_size_x, sprite_size_y, n_channels])
        sprite_2 = np.ones([sprite_size_x, sprite_size_y, n_channels])

    image_size_x = 1000
    image_size_y = 1000
    image_pre = torch.zeros(1, n_channels, image_size_x, image_size_y)
    print image_pre.shape
    images_post = draw_sprites_at_poses(
          torch.stack([torch.Tensor(torch.Tensor([50.0, 100.0, np.pi/4.])),
                       torch.Tensor(torch.Tensor([100, 100, np.pi/2]))]),
          sprite_size_x, sprite_size_y,
          image_size_x, image_size_y,
          numpy_images_to_torch([sprite_1, sprite_2]))

    image = torch_images_to_numpy(image_pre + images_post.sum(dim=0))[0]

    plt.imshow(image)
    plt.show()