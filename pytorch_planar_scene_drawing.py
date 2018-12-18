import torch
import torch.nn.functional as F
import pyro

device = torch.device('cpu')

# Use Spatial Transformer Network strategy from the 
# Pyro AIR tutorial to construct a differentiable
# drawing system.

expansion_indices = torch.LongTensor([0, 1, 2, 3, 4, 5])
def convert_pose_to_matrix(pose):
    # Converts a [x, y, theta] pose to a tf matrix:
    # [[cos(theta) -sin(theta) x]]
    #  [sin(theta) cos(theta)  y]]
    # (except in vectorized form -- so input is n x 3,
    #  output is n x 2 x 3)
    n = pose.size(0)
    out = torch.cat((torch.cos(pose[:, 2]).view(n, -1),
    				 -torch.sin(pose[:, 2]).view(n, -1),
    				 pose[:, 0].view(n, -1),
    				 torch.sin(pose[:, 2]).view(n, -1),
    				 torch.cos(pose[:, 2]).view(n, -1),
    				 pose[:, 1].view(n, -1)), 1)
    out = out.view(n, 2, 3)
    return out

def draw_sprites_at_poses(pose, sprite_size_x, sprite_size_y, image_size_x, image_size_y, sprites):
    n = sprites.size(0)
    assert sprites.size(1) == sprite_size_x * sprite_size_y, 'Size mismatch.'
    tf = convert_pose_to_matrix(pose)

    # Modify the TF to shrink the sprite by sprite_size / image_size,
    # and offset it so that pose [0, 0] puts it at the top left of the image.

    # The coordinate system for grid_sample puts -1, -1 at the top left,
    # and 0, 0 at the middle. Awkward...

    # what the fuck, why am I such an idiot
    # this should be totally trivial
    # and I don't even *need* or care about rotations, or even subimages,
    # for anything other than foveation so my method can scale to arbitrarily
    # large images.
    # All I need to care about is cropping and how effects apply to small subimages.
    # Nuke this and restart from AIR's version, being more careful about how they
    # handle scaling.

    x_scaling = float(sprite_size_x) / image_size_x
    y_scaling = float(sprite_size_y) / image_size_y
    # Scale the sprite down...
    # this is *reversed* as the image coordinate in the sprite is
    # multiplied by this TF to get the image coordinate in the
    # complete image.
    tf[:, 0, 0] /= x_scaling
    tf[:, 1, 0] /= x_scaling
    tf[:, 0, 1] /= y_scaling
    tf[:, 1, 1] /= y_scaling
    
    grid = F.affine_grid(tf, torch.Size((n, 1, image_size_x, image_size_y)))
    out = F.grid_sample(sprites.view(n, 1, sprite_size_x, sprite_size_y), grid)
    return out.view(n, image_size_x, image_size_y)

if __name__ == "__main__":
    poses = torch.FloatTensor([[0., 1., 2.],
                           [1.57, 4., 5.]]).view(2, -1)
    print "Poses: ", poses
    print "Pose matrix: ", convert_pose_to_matrix(poses)


    # Test out by generating some sprites and drawing them into a larger image.
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    sprite_size_x = 100
    sprite_size_y = 100
    sprite_1 = np.random.random([sprite_size_x, sprite_size_y])
    sprite_2 = np.random.random([sprite_size_x, sprite_size_y])

    image_size_x = 500
    image_size_y = 500
    image_pre = torch.Tensor(np.zeros((image_size_x, image_size_y)))
    print torch.stack
    images_post = draw_sprites_at_poses(
          torch.stack([torch.Tensor(torch.Tensor([0.3, 0, -np.pi/2.])),
                     torch.Tensor(torch.Tensor([0, 0, np.pi/4]))]),
          sprite_size_x, sprite_size_y,
          image_size_x, image_size_y,
          torch.stack([torch.Tensor(sprite_1).view(-1, 1),
                      torch.Tensor(sprite_2).view(-1, 1)]))


    print images_post.shape
    image = (image_pre + images_post.sum(dim=0)).cpu().detach().numpy()
    print image, np.max(image)
    plt.imshow(image)
    plt.show()