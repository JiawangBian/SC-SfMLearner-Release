# Export environment variable.
export DECORD_EOF_RETRY_MAX=200000480

# GPU device ID
GPU_DEVICE_ID=0

# Train the model.
CUDA_VISIBLE_DEVICES=$GPU_DEVICE_ID python train_model_efficient_loader.py \
--config "configs/dynamic_scenes_single_scale_29_videos.toml" \
--batch-size 1 \
--epochs 25 \
--max-train-iterations 25 \
--max-val-iterations 5 \
--gpu