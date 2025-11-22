python3 baseline1_fewshot.py \
    --output_dir /mnt/amlfs-03/shared/ayaanm/yanav/results/baseline1 \
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
    --k_shots 8 \
    --seeds 42 \
    --val_ratio 0.15 \
    --C 1.0 \
    --max_iter 1000 \
    --patience 20 \
    --min_delta 0.001

