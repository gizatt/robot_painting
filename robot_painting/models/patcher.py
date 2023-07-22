'''
Generated primarily with GPT-4, 20230721.
'''

import torch
from torch.nn.functional import pad

class ImagePatcher:
    def __init__(self, patch_size):
        '''
            `patch_size`: Desired image patch in pixels. If `patch_size` is odd, we'll center exactly
            at requested coordinates. If `patch_size` is even, we'll center positively-shifted by
            half a pixel.
        '''
        self.patch_size = patch_size

    def extract_patches(self, images, locations):
        """
        Extract patches centered at each point in the trajectory.

        Parameters:
        images (Tensor): A tensor of shape (batch_size n_channels, height (y), width (x)).
        locations (Tensor): A tensor of shape (batch_size, N_locations, 2),
                               where the last dimension represents the (x, y) coordinates.

        Returns:
        patches (Tensor): A tensor of shape (batch_size, N_locations, n_channels, patch_size, patch_size),
                          containing the patches centered at each point in the trajectory.
        """
        # Determine the amount of padding needed on each side
        left_pad_size = self.patch_size // 2 - 1 + self.patch_size % 2
        right_pad_size = self.patch_size // 2
        assert left_pad_size + right_pad_size + 1 == self.patch_size, f"{left_pad_size}, {right_pad_size}, {self.patch_size}"

        # Pad the images
        images_padded = pad(images, (left_pad_size, right_pad_size, left_pad_size, right_pad_size))

        # Initialize a tensor to hold the patches
        patches = torch.zeros(
            (images.shape[0], locations.shape[1], images.shape[1], self.patch_size, self.patch_size)
        )

        # Loop over the images & locations in the batch and extract the patches
        for i in range(images.shape[0]):
            for j in range(locations.shape[1]):
                x, y = locations[i, j].to(int)
                patches[i, j, :] = images_padded[i, :, y:y+self.patch_size, x:x+self.patch_size]

        return patches