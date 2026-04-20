"""
GPT-4o 蒸馏：将原始题目转化为步进式讲解数据
用法: python src/data/distill.py --input data/raw/problems.jsonl --output data/distilled/
"""
import argparse
import json
import os
from pathlib import Path

import jsonlines
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

SYSTEM_PROMPT = """你是一位经验丰富的 OI/ACM 算法教练，擅长向初学者讲解算法题。
请按以下格式逐步讲解题目，不要直接给出完整代码：

**第一步：理解题意**
（用自己的话复述题目，明确输入输出）

**第二步：分析暴力解**
（先想最简单的解法，分析时间复杂度）

**第三步：寻找优化方向**
（暴力哪里慢？能否用特定数据结构或算法优化？）

**第四步：设计最优解**
（详细推导思路，包括为什么这样做）

**第五步：实现代码**
（给出完整 Python 代码，并逐行注释关键逻辑）

**复杂度分析**
时间复杂度：O(...)
空间复杂度：O(...)"""

USER_TMPL = """请按格式讲解以下算法题：

{problem}

难度：{difficulty}
标签：{tags}"""


def distill_one(problem: dict) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TMPL.format(
                problem=problem["description"],
                difficulty=problem.get("difficulty", "unknown"),
                tags=", ".join(problem.get("tags", [])),
            )},
        ],
    )
    teaching = resp.choices[0].message.content
    return {**problem, "teaching": teaching}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "distilled.jsonl"

    with jsonlines.open(args.input) as reader, jsonlines.open(out_file, "w") as writer:
        items = list(reader)
        if args.limit:
            items = items[: args.limit]
        for item in tqdm(items, desc="蒸馏"):
            try:
                result = distill_one(item)
                writer.write(result)
            except Exception as e:
                print(f"[skip] {item.get('id', '?')}: {e}")

    print(f"[done] 蒸馏完成，输出至 {out_file}")


if __name__ == "__main__":
    main()
