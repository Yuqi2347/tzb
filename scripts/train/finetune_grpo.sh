#!/usr/bin/env bash
# set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ========= 使用说明 =========
# 本脚本支持两种优化策略：
# 1. GRPO (Group Relative Policy Optimization) - 默认
# 2. PPO (Proximal Policy Optimization)
#
# 使用方法：
# - 修改 OPTIMIZATION_STRATEGY 变量来选择策略
# - GRPO: 适合组内相对比较，计算效率高
# - PPO: 经典策略优化，稳定性好
#
# 示例：
# OPTIMIZATION_STRATEGY="grpo"  # 使用GRPO
# OPTIMIZATION_STRATEGY="ppo"   # 使用PPO
#sudo nvidia-smi -i 6 -c EXCLUSIVE_PROCESS  sudo nvidia-smi -i 0 -c DEFAULT
# ========= 分布式配置 =========
GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12325

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=INFO            # 如需安静可改为 WARN
export TORCH_DISTRIBUTED_DEBUG=OFF # 需要排错可改 DETAIL
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# 🔥 显存碎片整理
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ========= 优化策略选择 =========
OPTIMIZATION_STRATEGY="rgrpo"  # 可选: "grpo" 或 "ppo" 或 "rgrpo"
# GRPO: Group Relative Policy Optimization
# PPO:  Proximal Policy Optimization

# ========= 超参 =========
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1  # 梯度累积，等效batch_size=4
GROUP_SIZE=4                   # GRPO组采样：现在每个response立即backward，可以用4！
MAX_NEW_TOKENS=1024             # 从512改为256以节省显存
LR=1e-5
EPOCHS=3
SAVE_INTERVAL=50
LOG_INTERVAL=1

# ========= PPO 特定参数 =========
PPO_CLIP_RATIO=0.2            # PPO裁剪比率
PPO_VALUE_LOSS_COEF=0.5       # 价值损失系数
PPO_ENTROPY_COEF=0.01         # 熵系数
PPO_MAX_GRAD_NORM=0.5         # 最大梯度范数
PPO_GAE_LAMBDA=0.95           # GAE lambda参数
PPO_NUM_EPOCHS=4              # PPO内部epoch数

# 分阶段奖励 & baseline
WARMUP_STEPS=10
SCHEDULE_STEPS=10
BETA_ENTROPY=0.01
BASELINE_INIT=0.0
BASELINE_DECAY=0.9
FORMAT_ONLY_STEPS=10
MAX_REWARD=10.0
MAX_ADV=5.0
EPSILON=1e-3

# ========= RGRPO 参数 =========
RGRPO_MAX_ITERATIONS=4
RGRPO_SUCCESS_THRESHOLD=1.0
RGRPO_API_BASE="https://xiaoai.plus/v1"
RGRPO_API_MODEL="gpt-5"
RGRPO_API_KEY_ENV="RGRPO_API_KEY"
RGRPO_FEEDBACK_TEMPERATURE=0.2
RGRPO_FEEDBACK_TOP_P=0.9
RGRPO_RETRY_LIMIT=2

# ========= 模型 & 数据 =========
SFT_MODEL="${MODEL_PATH:-models/FM9G4B-V}"
PROMPTS_DATA="${PROMPTS_DATA:-examples/data/sample_train.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo}"
IMAGE_PATH="${IMAGE_ROOT:-.}"
# RESUME_CKPT="$SFT_MODEL"

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

# DeepSpeed（占位，可选）
DS_CONFIG="configs/ds_config_zero2.json"

# ========= 组装命令行 =========
CMD_ARGS=""
CMD_ARGS="$CMD_ARGS --model_name_or_path $SFT_MODEL"
CMD_ARGS="$CMD_ARGS --prompts_path $PROMPTS_DATA"
CMD_ARGS="$CMD_ARGS --output_dir $OUTPUT_DIR"
CMD_ARGS="$CMD_ARGS --image_path $IMAGE_PATH"
# CMD_ARGS="$CMD_ARGS --resume_from_checkpoint $RESUME_CKPT"

# 添加优化策略参数
CMD_ARGS="$CMD_ARGS --optimization_strategy $OPTIMIZATION_STRATEGY"

CMD_ARGS="$CMD_ARGS --batch_size $BATCH_SIZE"
CMD_ARGS="$CMD_ARGS --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"
CMD_ARGS="$CMD_ARGS --max_new_tokens $MAX_NEW_TOKENS"
CMD_ARGS="$CMD_ARGS --lr $LR"
CMD_ARGS="$CMD_ARGS --epochs $EPOCHS"
CMD_ARGS="$CMD_ARGS --save_interval $SAVE_INTERVAL"
CMD_ARGS="$CMD_ARGS --log_interval $LOG_INTERVAL"

# 根据优化策略添加特定参数
if [ "$OPTIMIZATION_STRATEGY" = "grpo" ]; then
    echo "🚀 使用 GRPO 优化策略"
    CMD_ARGS="$CMD_ARGS --group_size $GROUP_SIZE"
    CMD_ARGS="$CMD_ARGS --warmup_steps $WARMUP_STEPS"
    CMD_ARGS="$CMD_ARGS --schedule_steps $SCHEDULE_STEPS"
    CMD_ARGS="$CMD_ARGS --beta_entropy $BETA_ENTROPY"
    CMD_ARGS="$CMD_ARGS --baseline_init $BASELINE_INIT"
    CMD_ARGS="$CMD_ARGS --baseline_decay $BASELINE_DECAY"
    CMD_ARGS="$CMD_ARGS --format_only_steps $FORMAT_ONLY_STEPS"
    CMD_ARGS="$CMD_ARGS --max_reward $MAX_REWARD"
    CMD_ARGS="$CMD_ARGS --max_adv $MAX_ADV"
    CMD_ARGS="$CMD_ARGS --epsilon $EPSILON"
elif [ "$OPTIMIZATION_STRATEGY" = "ppo" ]; then
    echo "🚀 使用 PPO 优化策略"
    CMD_ARGS="$CMD_ARGS --ppo_clip_ratio $PPO_CLIP_RATIO"
    CMD_ARGS="$CMD_ARGS --ppo_value_loss_coef $PPO_VALUE_LOSS_COEF"
    CMD_ARGS="$CMD_ARGS --ppo_entropy_coef $PPO_ENTROPY_COEF"
    CMD_ARGS="$CMD_ARGS --ppo_max_grad_norm $PPO_MAX_GRAD_NORM"
    CMD_ARGS="$CMD_ARGS --ppo_gae_lambda $PPO_GAE_LAMBDA"
    CMD_ARGS="$CMD_ARGS --ppo_num_epochs $PPO_NUM_EPOCHS"
elif [ "$OPTIMIZATION_STRATEGY" = "rgrpo" ]; then
    echo "🚀 使用 RGRPO 优化策略"
    CMD_ARGS="$CMD_ARGS --group_size $GROUP_SIZE"
    CMD_ARGS="$CMD_ARGS --warmup_steps $WARMUP_STEPS"
    CMD_ARGS="$CMD_ARGS --schedule_steps $SCHEDULE_STEPS"
    CMD_ARGS="$CMD_ARGS --beta_entropy $BETA_ENTROPY"
    CMD_ARGS="$CMD_ARGS --baseline_init $BASELINE_INIT"
    CMD_ARGS="$CMD_ARGS --baseline_decay $BASELINE_DECAY"
    CMD_ARGS="$CMD_ARGS --format_only_steps $FORMAT_ONLY_STEPS"
    CMD_ARGS="$CMD_ARGS --max_reward $MAX_REWARD"
    CMD_ARGS="$CMD_ARGS --max_adv $MAX_ADV"
    CMD_ARGS="$CMD_ARGS --epsilon $EPSILON"
    CMD_ARGS="$CMD_ARGS --rgrpo_max_iterations $RGRPO_MAX_ITERATIONS"
    CMD_ARGS="$CMD_ARGS --rgrpo_success_threshold $RGRPO_SUCCESS_THRESHOLD"
    CMD_ARGS="$CMD_ARGS --rgrpo_api_base $RGRPO_API_BASE"
    CMD_ARGS="$CMD_ARGS --rgrpo_api_model $RGRPO_API_MODEL"
    CMD_ARGS="$CMD_ARGS --rgrpo_api_key_env $RGRPO_API_KEY_ENV"
    CMD_ARGS="$CMD_ARGS --rgrpo_feedback_temperature $RGRPO_FEEDBACK_TEMPERATURE"
    CMD_ARGS="$CMD_ARGS --rgrpo_feedback_top_p $RGRPO_FEEDBACK_TOP_P"
    CMD_ARGS="$CMD_ARGS --rgrpo_retry_limit $RGRPO_RETRY_LIMIT"
else
    echo "❌ 不支持的优化策略: $OPTIMIZATION_STRATEGY"
    echo "   支持的策略: grpo, ppo, rgrpo"
    exit 1
fi

# 通用奖励参数（两种策略都使用）
CMD_ARGS="$CMD_ARGS --think_r $THINK_R"
CMD_ARGS="$CMD_ARGS --answer_r $ANSWER_R"
CMD_ARGS="$CMD_ARGS --vqa_r $VQA_R"
CMD_ARGS="$CMD_ARGS --caption_r $CAPTION_R"
CMD_ARGS="$CMD_ARGS --grounding_r $GROUNDING_R"
CMD_ARGS="$CMD_ARGS --penalty_unit $PENALTY_UNIT"

# 注意：奖励权重参数已在上面设置

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

# DeepSpeed（这里只是透传参数，不启用 DS 引擎）
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

torchrun $DISTRIBUTED_ARGS scripts/train/train_grpo.py $CMD_ARGS \
  --logging_dir $OUTPUT_DIR \
  --bf16 \
  --gradient_checkpointing
