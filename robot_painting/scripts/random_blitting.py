import numpy as np
import torch
import imageio.v2 as imageio
import glob
import matplotlib.pyplot as plt
import torchvision

from robot_painting.models.sprite_blitting import (
    numpy_images_to_torch,
    torch_images_to_numpy,
    draw_sprites_at_poses,
    oilpaint_converters
)

def load_brushes():
    brush_search_paths = ["../../data/brushes/*.png"]
    # Open every image in that folder
    brush_paths = sum([glob.glob(sp) for sp in brush_search_paths], [])
    print("Found brushes: ", brush_paths)

    raw_brushes = [imageio.imread(brush_path).astype(float) / 256. for brush_path in brush_paths]
    raw_brushes = numpy_images_to_torch(raw_brushes)
    return raw_brushes

def load_target_image():
    target_image_path = "../../data/target.jpeg"
    target_image_np = imageio.imread(target_image_path).astype(float)/256.
    target_image = torch.tensor(target_image_np).permute(2, 0, 1)
    return target_image

if __name__ == "__main__":
    device = torch.device('cuda')
    N_sprites_in_batch = 100
    brushes = load_brushes().to(device)
    target_image = load_target_image().to(device)
    b2p, p2b = oilpaint_converters(device=device)

    n_channels, image_rows, image_cols = target_image.shape
    current_image = torch.ones(3, image_rows, image_cols, device=device)

    orig_brush_mask = brushes[0][3, :]
    brushes = torch.empty(N_sprites_in_batch, 4, orig_brush_mask.shape[0], orig_brush_mask.shape[1], device=device)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.imshow(torch_images_to_numpy(target_image))
    imshow_data = ax2.imshow(torch_images_to_numpy(current_image))

    iter_k = 0
    while (1):
        iter_k += 1

        brush_scale = max(0.025, (0.5 / np.sqrt(iter_k)))
        blurred_target_image = torchvision.transforms.GaussianBlur(kernel_size=11, sigma=brush_scale * 2).forward(target_image)

        sprite_poses = torch.stack([
            torch.randint(low=0, high=image_rows, size=(N_sprites_in_batch,)),
            torch.randint(low=0, high=image_cols, size=(N_sprites_in_batch,)),
            torch.rand(N_sprites_in_batch) * np.pi * 2.
        ]).T.to(device)
        # Take color from source image, and use fixed opacity.
        for k in range(N_sprites_in_batch):
            center = sprite_poses[k, 0:2]
            for i in range(3):
                brushes[k, i, :, :] = blurred_target_image[i, int(center[0]), int(center[1])]
            brushes[k, 3, :, :] = orig_brush_mask * torch.full((1,), 0.5, device=device)
            
        sprites_scales = torch.ones(N_sprites_in_batch, 1, device=device) * brush_scale
        sprite_ims = draw_sprites_at_poses(
            brushes, sprite_poses, sprites_scales,
            image_rows, image_cols
        )
        
        image_pre_oil = b2p(current_image)
        sprite_ims_oil = b2p(sprite_ims)
        
        alphas = 1. - sprite_ims_oil[:, 3:, :, :]  # alphas inverted in absorb space
        inv_alphs = sprite_ims_oil[:, 3:, :, :]
            
        # Reduce down to one image
        # Permute for easier alpha combo
        image = image_pre_oil[:3, :, :]
        scaled_sprites = sprite_ims_oil[:, :3, :, :] * alphas
        for k in range(sprite_ims_oil.shape[0]):
            image = image * inv_alphs[k, ...] + scaled_sprites[k, ...]
    
        current_image = p2b(image)
        imshow_data.set_data(torch_images_to_numpy(current_image))
        plt.pause(1E-3)
    
