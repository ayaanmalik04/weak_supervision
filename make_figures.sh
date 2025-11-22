#!/bin/bash

# Quick script to generate all figures for milestone report
# Run this after your experiments complete

echo "=========================================="
echo "Generating Milestone Figures"
echo "=========================================="

python3 visualize_results.py \
    --results_dir /mnt/amlfs-03/shared/ayaanm/yanav/results/baseline1 \
    --output_dir /mnt/amlfs-03/shared/ayaanm/yanav/results/baseline1/figures

echo ""
echo "Done! Check the figures directory for outputs."
echo "You can now include these in your milestone report."

