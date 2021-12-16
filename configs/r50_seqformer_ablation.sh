#!/usr/bin/env bash

set -x

python3 -u main.py \
    --dataset_file YoutubeVIS \
    --epochs 12 \
    --lr 2e-4 \
    --lr_drop 2 10\
    --batch_size 2 \
    --num_workers 2 \
    --coco_path ../coco \
    --ytvis_path ../ytvis \
    --num_queries 300 \
    --num_frames 5 \
    --with_box_refine \
    --masks \
    --rel_coord \
    --backbone resnet50 \
    --pretrain_weights weights/r50_weight.pth \
    --output_dir r50_ablation \

