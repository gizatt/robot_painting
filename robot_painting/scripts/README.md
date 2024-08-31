## Dataset intake:
```
python .\robot_painting\scripts\do_dataset_intake.py .\data\dataset_20240823\ --seed 42 --test_fraction 0.0 --val_fraction 0.2
```

## Training stroke encoder:
```
python .\robot_painting\scripts\train_stroke_encoder.py training_logs
```

## Training prediction model:
```
python .\robot_painting\scripts\train_stroke_prediction_model.py --batch-size 128 --save-dir training_logs --dataset-path .\data\dataset_20240823\ --encoder-checkpoint .\data\pretrained_models\20240829_stroke_encoder_128x3.ckpt --finetune-encoder
```


