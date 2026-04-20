#!/usr/bin/env python3
"""
Seed 样本构建：从题库中随机采样 50 道题，调用 GPT-4o 生成步进式 CoT 讲解

输出：data/seed_50.jsonl
每条格式：
  {
    "id": str,
    "source": str,
    "difficulty": str,
    "problem": str,          # 题目描述
    "cot_solution": str,     # GPT-4o 完整输出（含逐步讲解）
    "code": str | null       # 从输出中提取的 Python 代码块
  }

用法：
    python scripts/build_seed.py [--n 50] [--source both] [--out data/seed_50.jsonl]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from tqdm import tqdm

from src.data.code_validator import extract_code, validate_syntax
from src.data.loader import Problem, load_problems

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 系统提示 ─────────────────────────────────────────────────

SYSTEM_PROMPT = """\
你是一位经验丰富的 OI/ACM 算法教练，专门面向初学者讲解算法题。
请按照以下格式，用中文逐步拆解解题思路。每一步都要给出推理过程，\
不要跳步骤，让初学者能完全跟上。

## 格式要求

**第一步：理解题意**
- 用自己的话复述题目，明确输入格式、输出格式、数据规模
- 举一个具体的小例子走一遍

**第二步：分析暴力解法**
- 给出最朴素的思路
- 分析其时间/空间复杂度，指出瓶颈在哪里

**第三步：寻找优化方向**
- 暴力的哪个步骤可以加速？
- 给出关键洞察（例如：单调性、数学性质、数据结构加速等）

**第四步：设计最优解**
- 详细推导最优算法的思路
- 给出关键伪代码片段（非完整代码，重点突出核心逻辑）

**第五步：完整 Python 实现**
- 给出可直接运行的 Python 代码
- 每个关键语句都要有注释，解释为什么这样写

**复杂度分析**
- 时间复杂度：O(...)，原因是...
- 空间复杂度：O(...)，原因是...

请严格遵守格式，代码必须放在 ```python ... ``` 代码块中。\
"""

USER_TMPL = """\
请按上述格式讲解以下算法题：

【题目描述】
{description}

【难度】{difficulty}
【标签】{tags}\
"""


# ── GPT-4o 调用 ───────────────────────────────────────────────

def call_gpt4o(client: OpenAI, problem: Problem, retry: int = 3) -> str:
    """调用 GPT-4o，失败时指数退避重试。"""
    tags_str = ", ".join(problem.tags) if problem.tags else "暂无"
    user_content = USER_TMPL.format(
        description=problem.description,
        difficulty=problem.difficulty,
        tags=tags_str,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    for attempt in range(retry):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
            )
            return resp.choices[0].message.content
        except Exception as e:
            wait = 2 ** attempt
            logger.warning("GPT-4o 调用失败（%s），%ds 后重试…", e, wait)
            time.sleep(wait)

    raise RuntimeError(f"GPT-4o 调用失败，已重试 {retry} 次")


# ── 单条处理 ─────────────────────────────────────────────────

def process_one(client: OpenAI, problem: Problem) -> dict:
    cot = call_gpt4o(client, problem)
    code = extract_code(cot)

    # 语法验证：如果提取的代码有语法错误，记录但不丢弃（seed 阶段保留供人工审查）
    syntax_ok = True
    syntax_err = None
    if code:
        syntax_ok, syntax_err = validate_syntax(code)

    return {
        "id":          problem.id,
        "source":      problem.source,
        "difficulty":  problem.difficulty,
        "tags":        problem.tags,
        "problem":     problem.description,
        "cot_solution": cot,
        "code":        code,
        "syntax_ok":   syntax_ok,
        "syntax_error": syntax_err,
    }


# ── 主流程 ───────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="构建 50 条 Seed 数据集")
    parser.add_argument("--n",      type=int,  default=50,               help="采样数量（默认 50）")
    parser.add_argument("--source", type=str,  default="both",
                        choices=["code_contests", "taco", "both"],        help="数据来源")
    parser.add_argument("--out",    type=str,  default="data/seed_50.jsonl", help="输出文件路径")
    parser.add_argument("--seed",   type=int,  default=42,               help="随机种子")
    parser.add_argument("--max_pool", type=int, default=5000,            help="题库采样池大小")
    args = parser.parse_args()

    # 检查 API Key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("请先 export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    client = OpenAI()

    # 1. 加载题库
    logger.info("加载题库（source=%s, pool=%d）…", args.source, args.max_pool)
    pool = load_problems(source=args.source, max_items=args.max_pool)
    logger.info("题库大小：%d 条", len(pool))

    if len(pool) < args.n:
        logger.warning("题库(%d)不足 %d 条，全部使用", len(pool), args.n)
        args.n = len(pool)

    # 2. 随机采样（固定 seed 保证可复现）
    random.seed(args.seed)
    sampled = random.sample(pool, args.n)
    diff_dist = {d: sum(1 for p in sampled if p.difficulty == d) for d in ("easy", "medium")}
    logger.info("采样完成：%s", diff_dist)

    # 3. 逐条调用 GPT-4o
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    failed = 0
    with tqdm(sampled, desc="GPT-4o 标注", unit="题") as pbar:
        for problem in pbar:
            try:
                record = process_one(client, problem)
                results.append(record)
                pbar.set_postfix(
                    ok=len(results),
                    fail=failed,
                    syntax_ok=sum(1 for r in results if r["syntax_ok"]),
                )
            except Exception as e:
                failed += 1
                logger.error("题目 %s 处理失败：%s", problem.id, e)

    # 4. 保存
    with open(out_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 5. 统计报告
    syntax_ok_count = sum(1 for r in results if r["syntax_ok"])
    code_found      = sum(1 for r in results if r["code"] is not None)
    logger.info(
        "完成！共 %d 条 → 输出至 %s\n"
        "  ├ 成功调用 GPT-4o: %d / %d\n"
        "  ├ 提取到代码块:     %d / %d\n"
        "  └ 代码语法正确:     %d / %d",
        len(results), out_path,
        len(results), args.n,
        code_found, len(results),
        syntax_ok_count, code_found,
    )


if __name__ == "__main__":
    main()
