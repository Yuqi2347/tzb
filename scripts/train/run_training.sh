#!/usr/bin/env bash
# 训练策略选择脚本

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 强化学习训练策略选择"
echo "================================"

# 检查参数
if [ $# -eq 0 ]; then
    echo "使用方法: $0 [grpo|ppo]"
    echo ""
    echo "支持的策略:"
    echo "  grpo  - Group Relative Policy Optimization (默认)"
    echo "  ppo   - Proximal Policy Optimization"
    echo ""
    echo "示例:"
    echo "  $0 grpo    # 使用GRPO策略"
    echo "  $0 ppo     # 使用PPO策略"
    exit 1
fi

STRATEGY=$1

case $STRATEGY in
    "grpo")
        echo "🚀 启动 GRPO 训练..."
        echo "特点: 组内相对比较，计算效率高"
        echo "================================"
        bash scripts/train/finetune_grpo.sh
        ;;
    "ppo")
        echo "🚀 启动 PPO 训练..."
        echo "特点: 经典策略优化，稳定性好"
        echo "================================"
        bash scripts/train/finetune_ppo.sh
        ;;
    *)
        echo "❌ 不支持的策略: $STRATEGY"
        echo "支持的策略: grpo, ppo"
        exit 1
        ;;
esac

echo "✅ 训练完成!"

