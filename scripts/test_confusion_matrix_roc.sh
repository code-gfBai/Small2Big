#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python /home/swu/bgf/ResNet/test_cjh.py \
  --train_data_dir="path to train data" \
  --val_data_dir="path to val data" \
  --test_data_dir="path to test data" \
  --batch_size=16 \
  --epochs=30 \
  --resume_epoch=1 \
  --output_path="path to output" \
  --class_num=4 \
  --lr=0.001 \
  --momentum=0.9 \
  --weights_loc="path to trained weight of resnet50"

