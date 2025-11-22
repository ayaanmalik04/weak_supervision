#!/bin/bash

# Run Baseline 3: Full Supervised Linear Probe
# This establishes the UPPER BOUND (uses all training labels)

echo "=========================================="
echo "Baseline 3: Full Supervised Linear Probe"
echo "=========================================="
echo ""
echo "This baseline uses ALL training labels"
echo "Trains on ~9,000 videos (100% of training data)"
echo ""
echo "Estimated time: ~2-3 hours"
echo "=========================================="
echo ""

python3 baseline3_fullsupervised.py \
    --output_dir /mnt/amlfs-03/shared/ayaanm/yanav/results/baseline3 \
    --cache_dir /mnt/amlfs-03/shared/ayaanm/yanav/cache \
    --train_subset 1.0 \
    --model_name microsoft/xclip-base-patch16 \
    --num_frames 32 \
    --input_size 224 \
    --scale_resize 256 \
    --use_train_augmentation \
    --use_multi_gpu_extraction \
    --batch_size 16 \
    --num_workers 16 \
    --C 1.0 \
    --max_iter 1000 \
    --patience 20 \
    --min_delta 0.001

echo ""
echo "=========================================="
echo "✓ Baseline 3 Complete!"
echo "=========================================="
echo ""
echo "Results saved to: results/baseline3/"

