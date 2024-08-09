#!/bin/bash

export MODEL_NAME="path/ComVis/stable-diffusion-v1-4"
export TRAIN_DIR="path to train data"
export OUTPUT_DIR="path to output of weight"

CUDA_VISIBLE_DEVICES=3 python /home/swu/bgf/dreambooth/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=8000 \
  --learning_rate=1e-06 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --noise_offset=0.1 \
  --use_8bit_adam \
  --checkpointing_steps=10000

