#!/usr/bin/env bash
# set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ========= 使用说明 =========
# PPO (Proximal Policy Optimization) 专用训练脚本
# 基于PPO算法的强化学习训练
#
# 特点：
# - 使用裁剪目标函数，防止策略更新过大
# - 支持GAE (Generalized Advantage Estimation)
# - 包含价值函数训练
# - 稳定性好，适合复杂任务

# ========= 分布式配置 =========
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12325  # 使用不同端口避免冲突

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=OFF
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ========= 优化策略 =========
OPTIMIZATION_STRATEGY="ppo"

# ========= 超参 =========
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
MAX_NEW_TOKENS=512
LR=1e-5
EPOCHS=1
SAVE_INTERVAL=50
LOG_INTERVAL=1

# ========= PPO 特定参数 =========
PPO_CLIP_RATIO=0.2            # PPO裁剪比率，防止策略更新过大
PPO_VALUE_LOSS_COEF=0.5       # 价值损失系数
PPO_ENTROPY_COEF=0.01         # 熵系数，鼓励探索
PPO_MAX_GRAD_NORM=0.5         # 最大梯度范数
PPO_GAE_LAMBDA=0.95           # GAE lambda参数
PPO_NUM_EPOCHS=4              # PPO内部epoch数
PPO_BATCH_SIZE=4              # PPO批次大小
PPO_MINI_BATCH_SIZE=1         # PPO小批次大小

# ========= 模型 & 数据 =========
SFT_MODEL="${MODEL_PATH:-models/FM9G4B-V}"
PROMPTS_DATA="${PROMPTS_DATA:-examples/data/sample_train.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ppo}"
IMAGE_PATH="${IMAGE_ROOT:-.}"

# Reward 权重
THINK_R=0.5
ANSWER_R=0.5
VQA_R=1.0
CAPTION_R=1.0
GROUNDING_R=1.0
PENALTY_UNIT=1.2

# LoRA
USE_LORA=true
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1
LORA_TARGET_MODULES="q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
Q_LORA=false

# DeepSpeed
DS_CONFIG="configs/ds_config_zero2.json"

# ========= 组装命令行 =========
CMD_ARGS=""
CMD_ARGS="$CMD_ARGS --model_name_or_path $SFT_MODEL"
CMD_ARGS="$CMD_ARGS --prompts_path $PROMPTS_DATA"
CMD_ARGS="$CMD_ARGS --output_dir $OUTPUT_DIR"
CMD_ARGS="$CMD_ARGS --image_path $IMAGE_PATH"

# 优化策略
CMD_ARGS="$CMD_ARGS --optimization_strategy $OPTIMIZATION_STRATEGY"

# 基础参数
CMD_ARGS="$CMD_ARGS --batch_size $BATCH_SIZE"
CMD_ARGS="$CMD_ARGS --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"
CMD_ARGS="$CMD_ARGS --max_new_tokens $MAX_NEW_TOKENS"
CMD_ARGS="$CMD_ARGS --lr $LR"
CMD_ARGS="$CMD_ARGS --epochs $EPOCHS"
CMD_ARGS="$CMD_ARGS --save_interval $SAVE_INTERVAL"
CMD_ARGS="$CMD_ARGS --log_interval $LOG_INTERVAL"

# PPO 特定参数
echo "🚀 使用 PPO 优化策略"
CMD_ARGS="$CMD_ARGS --ppo_clip_ratio $PPO_CLIP_RATIO"
CMD_ARGS="$CMD_ARGS --ppo_value_loss_coef $PPO_VALUE_LOSS_COEF"
CMD_ARGS="$CMD_ARGS --ppo_entropy_coef $PPO_ENTROPY_COEF"
CMD_ARGS="$CMD_ARGS --ppo_max_grad_norm $PPO_MAX_GRAD_NORM"
CMD_ARGS="$CMD_ARGS --ppo_gae_lambda $PPO_GAE_LAMBDA"
CMD_ARGS="$CMD_ARGS --ppo_num_epochs $PPO_NUM_EPOCHS"
CMD_ARGS="$CMD_ARGS --ppo_batch_size $PPO_BATCH_SIZE"
CMD_ARGS="$CMD_ARGS --ppo_mini_batch_size $PPO_MINI_BATCH_SIZE"

# 通用奖励参数
CMD_ARGS="$CMD_ARGS --think_r $THINK_R"
CMD_ARGS="$CMD_ARGS --answer_r $ANSWER_R"
CMD_ARGS="$CMD_ARGS --vqa_r $VQA_R"
CMD_ARGS="$CMD_ARGS --caption_r $CAPTION_R"
CMD_ARGS="$CMD_ARGS --grounding_r $GROUNDING_R"
CMD_ARGS="$CMD_ARGS --penalty_unit $PENALTY_UNIT"

# LoRA
if [ "$USE_LORA" = "true" ]; then
  CMD_ARGS="$CMD_ARGS --use_lora"
  CMD_ARGS="$CMD_ARGS --lora_r $LORA_R"
  CMD_ARGS="$CMD_ARGS --lora_alpha $LORA_ALPHA"
  CMD_ARGS="$CMD_ARGS --lora_dropout $LORA_DROPOUT"
  CMD_ARGS="$CMD_ARGS --lora_target_modules $LORA_TARGET_MODULES"
fi
if [ "$Q_LORA" = "true" ]; then
  CMD_ARGS="$CMD_ARGS --q_lora"
fi

# DeepSpeed
CMD_ARGS="$CMD_ARGS --deepspeed $DS_CONFIG"

# ========= torchrun 启动 =========
DISTRIBUTED_ARGS="\
  --nproc_per_node $GPUS_PER_NODE \
  --nnodes $NNODES \
  --node_rank $NODE_RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
"

mkdir -p "$OUTPUT_DIR"

echo "🚀 启动 PPO 训练..."
echo "📊 参数配置:"
echo "   - 优化策略: $OPTIMIZATION_STRATEGY"
echo "   - 批次大小: $BATCH_SIZE"
echo "   - 学习率: $LR"
echo "   - PPO裁剪比率: $PPO_CLIP_RATIO"
echo "   - PPO内部epoch: $PPO_NUM_EPOCHS"
echo "   - 输出目录: $OUTPUT_DIR"

torchrun $DISTRIBUTED_ARGS scripts/train/train_grpo.py $CMD_ARGS \
  --logging_dir $OUTPUT_DIR \
  --bf16 \
  --gradient_checkpointing
