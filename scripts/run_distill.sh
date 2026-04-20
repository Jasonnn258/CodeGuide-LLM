#!/usr/bin/env bash
# GPT-4o 蒸馏脚本：生成步进式讲解数据
set -euo pipefail

: "${OPENAI_API_KEY:?请先 export OPENAI_API_KEY=sk-...}"

LIMIT=${1:-500}   # 默认蒸馏前 500 条，可传参覆盖

echo "开始蒸馏，最多处理 $LIMIT 条题目..."

python src/data/distill.py \
    --input data/processed/train_raw.jsonl \
    --output data/distilled/ \
    --limit "$LIMIT"

echo "[done] 蒸馏完成，输出至 data/distilled/distilled.jsonl"
