#!/bin/bash

# Run Baseline 1 for Milestone Report
# Tests multiple K-shot values with multiple seeds for robust statistics

echo "=========================================="
echo "Running Baseline 1 for CS229 Milestone"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  K-shot values: 2, 5, 8, 10"
echo "  Seeds: 42, 43, 44"
echo "  Total experiments: 12"
echo ""
echo "This will take approximately 2-3 hours"
echo "=========================================="
echo ""

python3 baseline1_fewshot.py \
    --output_dir /mnt/amlfs-03/shared/ayaanm/yanav/results/baseline1 \
    --cache_dir /mnt/amlfs-03/shared/ayaanm/yanav/cache \
    --train_subset 1.0 \
    --model_name microsoft/xclip-base-patch16 \
    --num_frames 32 \
    --input_size 224 \
    --scale_resize 256 \
    --use_train_augmentation \
    --batch_size 4 \
    --num_workers 4 \
    --k_shots 2 5 8 10 \
    --seeds 42 43 44 \
    --val_ratio 0.15 \
    --C 1.0 \
    --max_iter 1000 \
    --patience 20 \
    --min_delta 0.001

echo ""
echo "=========================================="
echo "Experiments complete!"
echo "=========================================="
echo ""
echo "Now generating figures..."
bash make_figures.sh

echo ""
echo "✓ All done! Check results/baseline1/figures/ for outputs"

