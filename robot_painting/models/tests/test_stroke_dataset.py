import matplotlib.pyplot as plt
import torch

from robot_painting.models.stroke_dataset import StrokeRenderingDataset


def test_stroke_rendering_dataset(draw: bool = False):
    latent_image_size = 128
    dataset = StrokeRenderingDataset(latent_image_size=128)

    N_samples = 3
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=N_samples)
    if draw:
        fig = plt.figure()
        gs = plt.GridSpec(N_samples, 1, figure=fig)

    spline_params, rendered_images = next(iter(dataloader))
    assert (
        isinstance(spline_params, torch.Tensor)
        and len(spline_params.shape) == 2
        and spline_params.shape[0] == N_samples
    )
    assert isinstance(rendered_images, torch.Tensor) and rendered_images.shape == (
        N_samples,
        1,
        latent_image_size,
        latent_image_size,
    )

    if draw:
        for k, rendered_image in enumerate(rendered_images):
            ax = fig.add_subplot(gs[k, 0])
            ax.imshow(rendered_image.permute([1, 2, 0]).numpy())
            print(torch.min(rendered_image))

    if draw:
        plt.show()


if __name__ == "__main__":
    test_stroke_rendering_dataset(draw=True)
