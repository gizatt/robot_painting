import argparse
import lightning as L
import torch
import robot_painting.models.spline_generation as spline_generation
import robot_painting.models.stroke_dataset as stroke_dataset
import robot_painting.models.stroke_encoding_model as stroke_encoding_model
from lightning.pytorch.loggers import TensorBoardLogger
import pathlib



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=str)
    parser.add_argument("--batch-size", default=1024)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    stroke_generation_params = spline_generation.SplineGenerationParams()
    encoded_image_size = 128
    train_dataset = stroke_dataset.StrokeRenderingDataset(latent_image_size=encoded_image_size, batch_size=args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataset = stroke_dataset.StrokeRenderingDataset(latent_image_size=encoded_image_size, batch_size=args.batch_size, fixed_seeding=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    root_dir = pathlib.Path(args.root_dir)
    root_dir.mkdir(exist_ok=True, parents=True)
    logger = TensorBoardLogger(root_dir, name="stroke_encoder")
    trainer = L.Trainer(logger=logger, max_epochs=50, default_root_dir=root_dir, log_every_n_steps=1, benchmark=True)
    model = stroke_encoding_model.StrokeSupervisedAutoEncoder(
        stroke_parameterization_size=stroke_generation_params.spline_vectorization_length,
        encoded_image_size=encoded_image_size,
        encoded_image_channels=3,
        with_stroke_rendering=True
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)