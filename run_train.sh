export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export MODEL_NAME="pt-sk/stable-diffusion-1.5"
export OUTPUT_DIR="/workspace/LCM/saved"

accelerate launch train_lcm_distill_sd_wds.py \
    --pretrained_teacher_model=$MODEL_NAME \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=512 \
    --learning_rate=1e-6 --loss_type="huber" --ema_decay=0.95 --adam_weight_decay=0.0 \
    --max_train_steps=100 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=8 \
    --train_shards_path_or_url="pipe:aws s3 cp s3://muse-datasets/laion-aesthetic6plus-min512-data/{00000..01210}.tar -" \
    --validation_steps=200 \
    --checkpointing_steps=200 --checkpoints_total_limit=1 \
    --train_batch_size=12 \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --report_to=wandb \
    --seed=453645634
    # --resume_from_checkpoint=latest \
    # --push_to_hub