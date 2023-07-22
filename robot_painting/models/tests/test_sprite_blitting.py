import pytest
import torch
import robot_painting.models.sprite_blitting as mut
import numpy as np

def test_oilpaint_converters():
    '''
        Just make sure we can instantiate the oilpaint converters.
    '''
    mut.oilpaint_converters()

def test_pose_utils():
    pose = torch.tensor([[
        1., 2., np.pi/2.
    ]])
    tf_mat = mut.convert_pose_to_matrix(pose)
    assert tf_mat.shape == (1, 2, 3)
    assert torch.allclose(tf_mat, torch.tensor([[
        [0, -1, 1.],
        [1., 0., 2.]
    ]]), atol=1E-6)
    
    inv_tf_mat = mut.invert_tf_matrix(tf_mat)
    assert torch.allclose(inv_tf_mat, torch.tensor([[
        [0, 1, -2.],
        [-1., 0., 1.]
    ]]), atol=1E-6)

    composed_tf = mut.compose_tf_matrices(tf_mat, inv_tf_mat)
    assert torch.allclose(composed_tf, torch.tensor([[
        [1., 0., 0.],
        [0., 1., 0]
    ]]))

def test_numpy_torch_conversions():
    ims = np.random.random(size=(10, 32, 64, 3))
    ims_torch = mut.numpy_images_to_torch(ims)
    assert ims_torch.shape == (10, 3, 32, 64)
    ims_again = mut.torch_images_to_numpy(ims_torch)
    assert np.allclose(ims, ims_again)

def test_blitting():
    sprite_size = (3, 12, 13)
    N_sprites = 10
    sprites = torch.rand(size=(N_sprites, *sprite_size))
    im_T_sprites = torch.rand(size=(N_sprites, 3))
    scales = torch.rand(size=(N_sprites,)) + 1.
    image_rows = 64
    image_cols = 127
    im = mut.draw_sprites_at_poses(sprites, im_T_sprites, scales, image_rows, image_cols)
    assert im.shape == (N_sprites, sprite_size[0], image_rows, image_cols)

