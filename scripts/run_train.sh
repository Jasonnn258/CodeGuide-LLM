#!/usr/bin/env bash
# GRPO 后训练脚本
# 用法：
#   bash scripts/run_train.sh                        # 使用默认配置
#   bash scripts/run_train.sh configs/my_config.yaml # 指定配置
#   RESUME=models/grpo_final/checkpoint-200 bash scripts/run_train.sh  # 从断点恢复
set -euo pipefail

CONFIG=${1:-configs/train_config.yaml}
RESUME=${RESUME:-""}

echo "启动 GRPO 训练，配置文件: $CONFIG"

RESUME_ARG=""
if [ -n "$RESUME" ]; then
    echo "从 checkpoint 恢复：$RESUME"
    RESUME_ARG="--resume_from_checkpoint $RESUME"
fi

# 单卡训练（RTX 4090）
# 若有多卡可改为：torchrun --nproc_per_node=N src/training/grpo_train.py ...
CUDA_VISIBLE_DEVICES=0 python src/training/grpo_train.py \
    --config "$CONFIG" \
    $RESUME_ARG

echo "[done] 训练完成，产物见 models/grpo_final/ 和 models/codeguide_llm_merged/"
