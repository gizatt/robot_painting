import pytest
import torch
from robot_painting.models.stroke_prediction_model import StrokePredictionConvnet

def test_stroke_prediction_convnet():
    torch.manual_seed(42)

    batch_size = 2
    image_size = 64
    encoded_image_channels = 16
    pen_feature_size = 8

    # Run the model with fake inputs
    model = StrokePredictionConvnet(
        image_size=image_size,
        encoded_image_channels=encoded_image_channels,
        pen_feature_size=pen_feature_size
    )
    encoded_stroke_image = torch.rand(batch_size, encoded_image_channels, image_size, image_size)
    before_image = torch.rand(batch_size, 3, image_size, image_size)
    pen_embedding = torch.rand(batch_size, pen_feature_size)

    output = model(encoded_stroke_image, before_image, pen_embedding)

    # Sanity-check output size.
    expected_shape = (batch_size, 3, image_size, image_size)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"
