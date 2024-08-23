#!/bin/bash

# Set variables
TRAIN_DATA0="./data/e2e/train0.jsonl"
TRAIN_DATA1="./data/e2e/train1.jsonl"
TRAIN_DATA2="./data/e2e/train2.jsonl"
VALID_DATA="./data/e2e/valid.jsonl"
INIT_CHECKPOINT="./pretrained_checkpoints/gpt2-medium-pytorch_model.bin"
WORK_DIR="./trained_models/GPT2_M/e2e"

# Activate the virtual environment (uncomment if needed)
# source .venv_dp-lora/bin/activate

# Run the training script
python src-dp/gpt2_ft_sfl.py \
    --train_data0 $TRAIN_DATA0 \
    --train_data1 $TRAIN_DATA1 \
    --train_data2 $TRAIN_DATA2 \
    --valid_data $VALID_DATA \
    --train_batch_size 4 \
    --grad_acc 1 \
    --valid_batch_size 4 \
    --seq_len 512 \
    --model_card gpt2.md \
    --init_checkpoint $INIT_CHECKPOINT \
    --platform local \
    --clip 0.0 \
    --lr 0.0002 \
    --weight_decay 0.01 \
    --correct_bias \
    --adam_beta2 0.999 \
    --scheduler linear \
    --warmup_step 500 \
    --max_epoch 5 \
    --save_interval 400000 \
    --lora_dim 2 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --label_smooth 0.1 \
    --work_dir $WORK_DIR \
    --random_seed 40