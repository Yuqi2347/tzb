#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12324

MODEL_MAX_Length=4096 # 保持完整信息，不截断

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


###################################################  sft  ############################################
# 使用原始 LoRA 模型（tokenizer 已添加 special tokens）
MODEL="${MODEL_PATH:-models/FM9G4B-V}"
DATA="${TRAIN_DATA:-examples/data/sample_train.json}"
EVAL_DATA="${EVAL_DATA:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" torchrun $DISTRIBUTED_ARGS scripts/train/finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --use_lora True \
    --tune_vision False \
    --tune_llm False \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 9 \
    --num_train_epochs 2 \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$OUTPUT_DIR" \
    --logging_strategy "steps" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 10 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed configs/ds_config_zero2.json \
    --report_to "tensorboard"


############################################################  GRPO  #######################################################################

# PPO / GRPO 超参
# BATCH_SIZE=1
# MAX_NEW_TOKENS=1024
# LR=1e-5
# EPOCHS=4
# SAVE_INTERVAL=2000
# LOG_INTERVAL=1

# # 分阶段奖励 & baseline
# WARMUP_STEPS=100           # 只用格式奖励的 warm‑up
# SCHEDULE_STEPS=500         # 从 warm‑up 到全混合的 schedule
# BETA_ENTROPY=1e-3          # 熵正则权重 β
# BASELINE_INIT=0.0          # baseline 初始值
# BASELINE_DECAY=0.9         # baseline 平滑系数
# FORMAT_ONLY_STEPS=100      # 第一 200 步只算格式奖励
# MAX_REWARD=1.0             # 奖励 clip 上限
# MAX_ADV=1.0                # 优势 clip 上限
# EPSILON=1e-3               # 防 0 常数

# # 模型 & 数据路径
# SFT_MODEL="models/FM9G4B-V"
# PROMPTS_DATA="examples/data/sample_train.json"
# OUTPUT_DIR="output_grpo"
# IMAGE_PATH="datasets/vrs/Images_train/"
# # RESUME_CKPT="$SFT_MODEL"   # 把 merged_model 当作 resume checkpoint

# # Reward 权重（并入 compute_reward）
# THINK_R=0.5
# ANSWER_R=0.5
# VQA_R=1.0
# CAPTION_R=1.0
# GROUNDING_R=1.0
# PENALTY_UNIT=1.2

# # LoRA 参数
# USE_LORA=true
# LORA_R=16
# LORA_ALPHA=32
# LORA_DROPOUT=0.1
# LORA_TARGET_MODULES="q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
# Q_LORA=false

# # DeepSpeed / 其它
# DS_CONFIG="ds_config_zero2.json"

# # 构建命令参数
# CMD_ARGS=""
# CMD_ARGS="$CMD_ARGS --model_name_or_path $SFT_MODEL"
# CMD_ARGS="$CMD_ARGS --prompts_path $PROMPTS_DATA"
# CMD_ARGS="$CMD_ARGS --output_dir $OUTPUT_DIR"
# CMD_ARGS="$CMD_ARGS --image_path $IMAGE_PATH"
# # CMD_ARGS="$CMD_ARGS --resume_from_checkpoint $RESUME_CKPT"

# CMD_ARGS="$CMD_ARGS --batch_size $BATCH_SIZE"
# CMD_ARGS="$CMD_ARGS --max_new_tokens $MAX_NEW_TOKENS"
# CMD_ARGS="$CMD_ARGS --lr $LR"
# CMD_ARGS="$CMD_ARGS --epochs $EPOCHS"
# CMD_ARGS="$CMD_ARGS --save_interval $SAVE_INTERVAL"
# CMD_ARGS="$CMD_ARGS --log_interval $LOG_INTERVAL"

# # reward 分段 & baseline
# CMD_ARGS="$CMD_ARGS --warmup_steps $WARMUP_STEPS"
# CMD_ARGS="$CMD_ARGS --schedule_steps $SCHEDULE_STEPS"
# CMD_ARGS="$CMD_ARGS --beta_entropy $BETA_ENTROPY"
# CMD_ARGS="$CMD_ARGS --baseline_init $BASELINE_INIT"
# CMD_ARGS="$CMD_ARGS --baseline_decay $BASELINE_DECAY"
# CMD_ARGS="$CMD_ARGS --format_only_steps $FORMAT_ONLY_STEPS"
# CMD_ARGS="$CMD_ARGS --max_reward $MAX_REWARD"
# CMD_ARGS="$CMD_ARGS --max_adv $MAX_ADV"
# CMD_ARGS="$CMD_ARGS --epsilon $EPSILON"

# # 传递 compute_reward 中的权重
# CMD_ARGS="$CMD_ARGS --think_r $THINK_R"
# CMD_ARGS="$CMD_ARGS --answer_r $ANSWER_R"
# CMD_ARGS="$CMD_ARGS --vqa_r $VQA_R"
# CMD_ARGS="$CMD_ARGS --caption_r $CAPTION_R"
# CMD_ARGS="$CMD_ARGS --grounding_r $GROUNDING_R"
# CMD_ARGS="$CMD_ARGS --penalty_unit $PENALTY_UNIT"

# # LoRA 相关
# if [ "$USE_LORA" = "true" ]; then
#     CMD_ARGS="$CMD_ARGS --use_lora"
#     CMD_ARGS="$CMD_ARGS --lora_r $LORA_R"
#     CMD_ARGS="$CMD_ARGS --lora_alpha $LORA_ALPHA"
#     CMD_ARGS="$CMD_ARGS --lora_dropout $LORA_DROPOUT"
#     CMD_ARGS="$CMD_ARGS --lora_target_modules $LORA_TARGET_MODULES"
# fi
# if [ "$Q_LORA" = "true" ]; then
#     CMD_ARGS="$CMD_ARGS --q_lora"
# fi

# # DeepSpeed
# CMD_ARGS="$CMD_ARGS --deepspeed $DS_CONFIG"

# # 启动
# torchrun $DISTRIBUTED_ARGS scripts/train/train_grpo.py $CMD_ARGS \
#   --logging_dir $OUTPUT_DIR \
#   --bf16 \
#   --gradient_checkpointing

#   --resume_from_checkpoint outputs/grpo/step_2000 \
