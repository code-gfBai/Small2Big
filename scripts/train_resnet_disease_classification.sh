#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_disease_classification.py \
  --train_data_dir="path to train data" \
  --val_data_dir="path to val data" \
  --test_data_dir="path to test data" \
  --output_folder_name="output name" \
  --net_name="resnet50" \
  --batch_size=16 \
  --epochs=30 \
  --resume_epoch=1 \
  --output_path="path to output" \
  --class_num=4 \
  --lr=0.001 \
  --momentum=0.9

