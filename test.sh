#!/bin/bash
python run.py \
    --ckpt models/training-video-final/best_model_wts.pt \
    --n_classes 50 \
    --model_type lrcn \
    --batch_size 16 \
    --output_dir models/training-video-final \
    --run_name training \
    --wandb \
    --mode eval
