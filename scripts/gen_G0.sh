#!/bin/bash

export MODEL_ID="path with trained weight"
export OUTPUT_DIRECTORY="path with output images"

CUDA_VISIBLE_DEVICES=0 python gen_img_G0.py \
  --model_id=$MODEL_ID \
  --prompt="a photo of tomato_bacterial_spot_mild leaf" \
  --output_directory=$OUTPUT_DIRECTORY \
  --num_images=200
