import argparse
import pathlib

import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger

import robot_painting.models.spline_generation as spline_generation
import robot_painting.models.stroke_dataset as stroke_dataset
import robot_painting.models.stroke_encoding_model as stroke_encoding_model
import robot_painting.models.stroke_prediction_model as stroke_prediction_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--encoder-checkpoint", required=True, type=str)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--pen-feature-size", default=8)
    parser.add_argument("--finetune-encoder", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load stroke encoder from checkpoint and optionally freeze it.
    stroke_supervised_autoencoder = (
        stroke_encoding_model.StrokeSupervisedAutoEncoder.load_from_checkpoint(
            args.encoder_checkpoint
        )
    )
    stroke_encoder = stroke_supervised_autoencoder.encoder

    # TODO(gizatt) This should be serialized with the stroke encoder so we
    # ensure we're using compatible spline generation parameters.
    spline_transform = (
        stroke_dataset.SplineToSamples.make_from_spline_generation_params(
            spline_generation.SplineGenerationParams(), num_stroke_time_samples=32
        )
    )

    encoded_image_size = stroke_encoder.encoded_image_size[0]
    assert stroke_encoder.encoded_image_size[0] == stroke_encoder.encoded_image_size[1]
    encoded_image_channels = stroke_encoder.encoded_image_channels
    datasets = {}
    loaders = {}
    for dataset_assignment in ["train", "val"]:
        datasets[dataset_assignment] = stroke_dataset.StrokeDataset(
            dataset_path=args.dataset_path,
            dataset_assignment=dataset_assignment,
            spline_transform=spline_transform,
            output_image_size=encoded_image_size,
            transform=stroke_dataset.StrokeDatasetRandomization(),
            crop_halfwidth_mm=64,
        )
        loaders[dataset_assignment] = torch.utils.data.DataLoader(
            datasets[dataset_assignment],
            batch_size=args.batch_size,
            shuffle=dataset_assignment == "train",
            num_workers=4,
            persistent_workers=True,
        )

    stroke_prediction_model = stroke_prediction_model.StrokePredictionModel(
        stroke_encoder=stroke_encoder,
        stroke_dataset=datasets["train"],
        pen_feature_size=args.pen_feature_size,
        image_size=encoded_image_size,
        encoded_image_channels=encoded_image_channels,
        freeze_encoder=not args.finetune_encoder,
    )

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    logger = TensorBoardLogger(save_dir, name="stroke_prediction_model")
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=save_dir / "stroke_prediction_model" / "checkpoints_of_last_run",
        save_top_k=2,
        monitor="val_reconstruction_loss",
    )

    trainer = L.Trainer(
        logger=logger,
        default_root_dir=save_dir,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        max_epochs=5000,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
    )

    trainer.fit(
        stroke_prediction_model,
        train_dataloaders=loaders["train"],
        val_dataloaders=loaders["val"],
    )
