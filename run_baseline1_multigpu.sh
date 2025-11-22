#!/bin/bash
# Run Baseline 1 on multiple GPUs in parallel
# Each K-shot/seed combination runs on a separate GPU

# Usage: bash run_baseline1_multigpu.sh

# This script launches 9 experiments (3 K-shots × 3 seeds) across 8 GPUs
# Multiple jobs can share the same GPU if needed

echo "Running Baseline 1 experiments across multiple GPUs..."
echo "Total experiments: 9 (K=5,10,20 × seeds=42,43,44)"
echo ""

OUTPUT_DIR="/mnt/amlfs-03/shared/ayaanm/yanav/results/baseline1"

# K=5
CUDA_VISIBLE_DEVICES=0 python baseline1_fewshot.py --k_shots 5 --seeds 42 --output_dir $OUTPUT_DIR &
CUDA_VISIBLE_DEVICES=1 python baseline1_fewshot.py --k_shots 5 --seeds 43 --output_dir $OUTPUT_DIR &
CUDA_VISIBLE_DEVICES=2 python baseline1_fewshot.py --k_shots 5 --seeds 44 --output_dir $OUTPUT_DIR &

# K=10
CUDA_VISIBLE_DEVICES=3 python baseline1_fewshot.py --k_shots 10 --seeds 42 --output_dir $OUTPUT_DIR &
CUDA_VISIBLE_DEVICES=4 python baseline1_fewshot.py --k_shots 10 --seeds 43 --output_dir $OUTPUT_DIR &
CUDA_VISIBLE_DEVICES=5 python baseline1_fewshot.py --k_shots 10 --seeds 44 --output_dir $OUTPUT_DIR &

# K=20
CUDA_VISIBLE_DEVICES=6 python baseline1_fewshot.py --k_shots 20 --seeds 42 --output_dir $OUTPUT_DIR &
CUDA_VISIBLE_DEVICES=7 python baseline1_fewshot.py --k_shots 20 --seeds 43 --output_dir $OUTPUT_DIR &
CUDA_VISIBLE_DEVICES=0 python baseline1_fewshot.py --k_shots 20 --seeds 44 --output_dir $OUTPUT_DIR &  # Reuse GPU 0

echo "All jobs launched! Check with: nvidia-smi"
echo "Logs will be in the respective output directories."
echo ""
echo "To wait for all jobs to finish, run: wait"

# Wait for all background jobs to complete
wait

echo ""
echo "All experiments completed!"
echo "Results are in ./results/baseline1/"

# Merge all results into a single JSON
python -c "
import json
import glob
from pathlib import Path

# This is a placeholder - you'd need to manually merge or run the aggregation script
print('Note: Results are saved separately. You may need to aggregate them manually.')
"

