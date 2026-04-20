#!/usr/bin/env bash
# 数据准备脚本：下载 LeetCode 题目并清洗
set -euo pipefail

DATA_DIR="data/raw"
PROCESSED_DIR="data/processed"
mkdir -p "$DATA_DIR" "$PROCESSED_DIR"

echo "[1/3] 下载 LeetCode 公开题目数据集..."
# 使用 Hugging Face datasets 中的公开镜像
python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("greengerong/leetcode", split="train")
ds.to_json("data/raw/leetcode_raw.jsonl")
print(f"下载完成，共 {len(ds)} 条")
EOF

echo "[2/3] 清洗并过滤..."
python - <<'EOF'
import jsonlines, re

def clean(item):
    return {
        "id": item.get("id") or item.get("frontend_question_id", ""),
        "title": item.get("title", ""),
        "description": re.sub(r"<[^>]+>", "", item.get("content", "")),
        "difficulty": item.get("difficulty", "").lower(),
        "tags": [t["name"] for t in item.get("topicTags", [])],
        "test_cases": item.get("sampleTestCase", ""),
    }

with jsonlines.open("data/raw/leetcode_raw.jsonl") as r, \
     jsonlines.open("data/processed/problems.jsonl", "w") as w:
    n = 0
    for item in r:
        c = clean(item)
        if len(c["description"]) > 50:   # 过滤空题
            w.write(c)
            n += 1
print(f"清洗完成，保留 {n} 条")
EOF

echo "[3/3] 切分 train/eval..."
python - <<'EOF'
import jsonlines, random
random.seed(42)

with jsonlines.open("data/processed/problems.jsonl") as r:
    items = list(r)

random.shuffle(items)
split = int(len(items) * 0.95)
train, eval_ = items[:split], items[split:]

with jsonlines.open("data/processed/train_raw.jsonl", "w") as w:
    for it in train: w.write(it)
with jsonlines.open("data/processed/eval_raw.jsonl", "w") as w:
    for it in eval_: w.write(it)

print(f"train: {len(train)}, eval: {len(eval_)}")
EOF

echo "[done] 数据准备完成"
