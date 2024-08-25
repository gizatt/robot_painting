import pytest
import torch
from robot_painting.models.stroke_encoding_model import (
    StrokeEncoder,
    StrokeDecoder,
    StackedConvnets,
    StrokeSupervisedAutoEncoder,
)


def test_stroke_encoder():
    input_dimension = 128
    encoded_image_size = (16, 16)
    encoded_image_channels = 3

    model = StrokeEncoder(
        input_dimension=input_dimension,
        encoded_image_size=encoded_image_size,
        encoded_image_channels=encoded_image_channels,
    )

    # Create a dummy input tensor
    batch_size = 4
    input_tensor = torch.randn(batch_size, input_dimension)

    # Forward pass
    output = model(input_tensor)

    # Check output shape
    assert output.shape == (
        batch_size,
        encoded_image_channels,
        encoded_image_size[0],
        encoded_image_size[1],
    )


def test_stroke_decoder():
    output_dimension = 128
    encoded_image_size = (16, 16)
    encoded_image_channels = 3

    model = StrokeDecoder(
        output_dimension=output_dimension,
        encoded_image_size=encoded_image_size,
        encoded_image_channels=encoded_image_channels,
    )

    # Create a dummy input tensor
    batch_size = 4
    input_tensor = torch.randn(
        batch_size, encoded_image_channels, encoded_image_size[0], encoded_image_size[1]
    )

    # Forward pass
    output = model(input_tensor)

    # Check output shape
    assert output.shape == (batch_size, output_dimension)


def test_stacked_convnets():
    N_input_channels = 3
    N_output_channels = 1
    convnet_channels = [8, 16]

    model = StackedConvnets(
        N_input_channels=N_input_channels,
        N_output_channels=N_output_channels,
        convnet_channels=convnet_channels,
    )

    # Create a dummy input tensor
    batch_size = 4
    height, width = 32, 32
    input_tensor = torch.randn(batch_size, N_input_channels, height, width)

    # Forward pass
    output = model(input_tensor)

    # Check output shape
    assert output.shape == (batch_size, N_output_channels, height, width)


def test_autoencoder_with_rendering():
    stroke_parameterization_size = 128
    encoded_image_size = 16
    encoded_image_channels = 3

    model = StrokeSupervisedAutoEncoder(
        stroke_parameterization_size=stroke_parameterization_size,
        encoded_image_size=encoded_image_size,
        encoded_image_channels=encoded_image_channels,
        with_stroke_rendering=True,
    )

    # Create dummy data for the test
    batch_size = 4
    spline_params = torch.randn(batch_size, stroke_parameterization_size)
    rendered_stroke_image = torch.randn(
        batch_size, 1, encoded_image_size, encoded_image_size
    )

    batch = (spline_params, rendered_stroke_image)

    # Perform a forward pass through training and val steps.
    train_loss = model.training_step(batch, batch_idx=0)
    validation_loss = model.validation_step(batch, batch_idx=0)

    # Check that the loss is a scalar tensor
    for loss in [train_loss, validation_loss]:
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0


if __name__ == "__main__":
    pytest.main()
