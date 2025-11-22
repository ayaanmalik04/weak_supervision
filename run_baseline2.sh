#!/bin/bash

# Run Baseline 2: Zero-Shot X-CLIP Evaluation
# This establishes the LOWER BOUND (no training needed)

echo "=========================================="
echo "Baseline 2: Zero-Shot X-CLIP"
echo "=========================================="
echo ""
echo "This baseline uses text prompts only"
echo "No training required - just inference!"
echo ""
echo "Estimated time: ~1-2 minutes (with 8 GPUs in parallel)"
echo "=========================================="
echo ""

python3 baseline2_zeroshot.py \
    --output_dir /mnt/amlfs-03/shared/ayaanm/yanav/results/baseline2 \
    --cache_dir /mnt/amlfs-03/shared/ayaanm/yanav/cache \
    --model_name microsoft/xclip-base-patch16 \
    --num_frames 32 \
    --input_size 224 \
    --scale_resize 256 \
    --use_multi_gpu_extraction \
    --batch_size 16 \
    --num_workers 16

echo ""
echo "=========================================="
echo "✓ Baseline 2 Complete!"
echo "=========================================="
echo ""
echo "Results saved to: results/baseline2/"

