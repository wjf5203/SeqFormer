#!/usr/bin/env bash

set -x

python3 -u main.py \
    --dataset_file jointcoco \
    --epochs 12 \
    --lr 2e-4 \
    --lr_drop 4 10\
    --batch_size 2 \
    --num_workers 2 \
    --coco_path ../coco \
    --ytvis_path ../ytvis \
    --num_queries 300 \
    --num_frames 5 \
    --with_box_refine \
    --masks \
    --rel_coord \
    --backbone resnet101 \
    --pretrain_weights weights/r101_weight.pth \
    --output_dir r101_joint \

