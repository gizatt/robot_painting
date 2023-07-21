'''
Note: Generated with help from with GPT-4, 20230721.
'''

import pytest

import torch
from robot_painting.models.patcher import ImagePatcher


def test_image_patcher_with_single_patch():
    image_patcher = ImagePatcher(patch_size=3)
    images = torch.arange(16, dtype=torch.float).reshape(1, 1, 4, 4)
    trajectories = torch.tensor([[[1, 1]]])
    patches = image_patcher.extract_patches(images, trajectories)
    expected_patches = torch.tensor([[[[[0., 1., 2.], [4., 5., 6.], [8., 9., 10.]]]]])
    assert torch.equal(patches, expected_patches)

def test_image_patcher_with_multiple_patches_and_padding():
    image_patcher = ImagePatcher(patch_size=2)
    images = torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)
    trajectories = torch.tensor([[[0, 0], [1, 1], [0, 2], [2, 0]]])
    patches = image_patcher.extract_patches(images, trajectories)
    expected_patches = torch.tensor(
        [
            [
                [[[0., 1.], [3., 4.]]],
                [[[4., 5.], [7., 8.]]],
                [[[6., 7.], [0., 0.]]],
                [[[2., 0.], [5., 0.]]]
            ]
        ]
    )
    assert patches.shape == expected_patches.shape, f"{patches.shape} vs {expected_patches.shape}"
    assert torch.equal(patches, expected_patches)
