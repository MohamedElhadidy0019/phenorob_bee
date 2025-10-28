#!/bin/bash

# Example usage:
# ./run_rf_detr.sh

python infer_video_rfdetr.py \
    --input_folder "/datan/bee_data/video_records_2024" \
    --output_folder "/home/s52melba/phenorob_bee/data_24_tiled" \
    --model_ckpt "/scratch/s52melba/detr_train_2/checkpoint_best_total.pth" \
    --tile_size 256 \
    --overlap 64 \
    --conf_threshold 0.5 \
    --fps 1.0

