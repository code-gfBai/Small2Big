#!/bin/bash

export MODEL_NAME="/path/ComVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path to instance images"
export CLASS_DIR="path to class images"
export OUTPUT_DIR="path to output of your weight"

accelerate launch ../train_Small2Big.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--train_text_encoder \
--instance_data_dir=$INSTANCE_DIR \
--class_data_dir=$CLASS_DIR \
--output_dir=$OUTPUT_DIR \
--with_prior_preservation --prior_loss_weight=1.0 \
--instance_prompt="a photo of sks leaf" \
--class_prompt="a photo of leaf" \
--resolution=512 \
--train_batch_size=1 \
--use_8bit_adam \
--gradient_accumulation_steps=1 \
--gradient_checkpointing \
--learning_rate=1e-6 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--num_class_images=200 \
--max_train_steps=300 \
--offset_noise

