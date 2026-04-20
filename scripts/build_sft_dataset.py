#!/usr/bin/env python3
"""
SFT 数据集全量构建（异步并发版）

流程：
  1. 加载 10k 题目池（load_problems）
  2. 断点续传：跳过 out_path 中已有的 ID
  3. 异步并发调用 GPT-4o（semaphore 控制并发数）
  4. 代码校验：语法检查 + 可选沙箱执行
  5. 组装 ChatML 格式，逐行写入 sft_train.jsonl

输出：data/sft_train.jsonl
每条格式（ChatML）：
  {
    "id": str,
    "messages": [
      {"role": "system",    "content": "..."},
      {"role": "user",      "content": "题目描述"},
      {"role": "assistant", "content": "步骤讲解 + 代码"}
    ],
    "metadata": {"source": ..., "difficulty": ..., "pass_rate": ...}
  }

用法：
    python scripts/build_sft_dataset.py \\
        [--max_items 10000] \\
        [--concurrency 10] \\
        [--run_code] \\
        [--out data/sft_train.jsonl]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

from src.data.code_validator import validate
from src.data.loader import Problem, load_problems

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 提示词 ───────────────────────────────────────────────────

SFT_SYSTEM = """\
你是 CodeGuide，一位专为 OI/ACM 初学者设计的算法教学助手。
当用户提出一道算法题时，你会：
1. 先用简单的语言理解题意，举例说明
2. 从暴力解出发，逐步优化到最优解
3. 每一步都解释"为什么这样想"，而不只是"怎么做"
4. 最后给出带详细注释的完整 Python 代码

你的讲解应当通俗易懂，适合刚开始学习算法竞赛的初学者。\
"""

SFT_SYSTEM_PROMPT = """\
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


# ── 异步 GPT-4o 调用 ──────────────────────────────────────────

async def call_gpt4o_async(
    client: AsyncOpenAI,
    problem: Problem,
    semaphore: asyncio.Semaphore,
    retry: int = 3,
) -> Optional[str]:
    """异步调用 GPT-4o，semaphore 限制并发，失败时指数退避重试。"""
    tags_str = ", ".join(problem.tags) if problem.tags else "暂无"
    user_content = USER_TMPL.format(
        description=problem.description,
        difficulty=problem.difficulty,
        tags=tags_str,
    )
    messages = [
        {"role": "system", "content": SFT_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    async with semaphore:
        for attempt in range(retry):
            try:
                resp = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2048,
                )
                return resp.choices[0].message.content
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(
                    "[%s] 调用失败（%s），%.1fs 后重试…", problem.id, e, wait
                )
                await asyncio.sleep(wait)

    logger.error("[%s] 已重试 %d 次，放弃", problem.id, retry)
    return None


# ── ChatML 组装 ──────────────────────────────────────────────

def to_chatml(problem: Problem, assistant_content: str, pass_rate: float) -> dict:
    """将题目 + GPT-4o 输出组装为 ChatML 格式记录。"""
    return {
        "id": problem.id,
        "messages": [
            {"role": "system",    "content": SFT_SYSTEM},
            {"role": "user",      "content": problem.description},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source":     problem.source,
            "difficulty": problem.difficulty,
            "tags":       problem.tags,
            "pass_rate":  round(pass_rate, 3),
        },
    }


# ── 断点续传：加载已完成的 ID ─────────────────────────────────

def load_done_ids(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                done.add(record["id"])
            except Exception:
                pass
    return done


# ── 统计计数器（线程安全替代品，asyncio 单线程无锁） ───────────

class Counter:
    def __init__(self):
        self.total = 0
        self.skipped = 0
        self.gpt_failed = 0
        self.no_code = 0
        self.syntax_fail = 0
        self.exec_fail = 0
        self.saved = 0

    def summary(self) -> str:
        return (
            f"总计处理: {self.total}\n"
            f"  ├ 断点跳过:   {self.skipped}\n"
            f"  ├ GPT-4o失败: {self.gpt_failed}\n"
            f"  ├ 无代码块:   {self.no_code}\n"
            f"  ├ 语法错误:   {self.syntax_fail}\n"
            f"  ├ 执行失败:   {self.exec_fail}\n"
            f"  └ 最终写入:   {self.saved}"
        )


# ── 单条异步处理 ─────────────────────────────────────────────

async def process_one(
    client: AsyncOpenAI,
    problem: Problem,
    semaphore: asyncio.Semaphore,
    run_code: bool,
    counter: Counter,
    out_file,
    lock: asyncio.Lock,
) -> None:
    counter.total += 1

    # GPT-4o 标注
    cot = await call_gpt4o_async(client, problem, semaphore)
    if cot is None:
        counter.gpt_failed += 1
        return

    # 代码校验
    result = validate(
        llm_output=cot,
        test_cases=problem.public_tests if run_code else None,
        run_code=run_code,
    )

    if result.code is None:
        counter.no_code += 1
        return

    if not result.ok:
        # 区分语法错误 vs 执行错误
        if "Syntax" in (result.error or ""):
            counter.syntax_fail += 1
        else:
            counter.exec_fail += 1
        # 语法错误直接丢弃；执行错误（如无测试用例导致的低分）视情况保留
        if "Syntax" in (result.error or ""):
            return

    # 组装 ChatML
    record = to_chatml(problem, cot, result.pass_rate)

    # 原子写入（asyncio 单线程，lock 防止多条记录交错）
    async with lock:
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_file.flush()
        counter.saved += 1


# ── 主流程 ───────────────────────────────────────────────────

async def async_main(args: argparse.Namespace) -> None:
    # 检查 API Key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("请先 export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    client = AsyncOpenAI()

    # 1. 加载题库
    logger.info("加载题库（source=%s, max=%d）…", args.source, args.max_items)
    all_problems = load_problems(source=args.source, max_items=args.max_items)
    logger.info("题库大小：%d 条", len(all_problems))

    # 2. 断点续传
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(out_path)
    if done_ids:
        logger.info("断点续传：跳过已完成 %d 条", len(done_ids))

    pending = [p for p in all_problems if p.id not in done_ids]
    logger.info("待处理：%d 条，并发数：%d", len(pending), args.concurrency)

    if not pending:
        logger.info("所有题目已处理完毕！")
        return

    # 3. 异步并发处理
    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    counter = Counter()
    counter.skipped = len(done_ids)

    with open(out_path, "a", encoding="utf-8") as out_file:
        tasks = [
            process_one(client, p, semaphore, args.run_code, counter, out_file, lock)
            for p in pending
        ]
        # tqdm 进度条
        for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="蒸馏进度"):
            await coro

    logger.info("\n%s", counter.summary())
    logger.info("输出文件：%s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 SFT 全量数据集（异步并发）")
    parser.add_argument("--max_items",   type=int,  default=10_000,
                        help="最大题目数（默认 10000）")
    parser.add_argument("--source",      type=str,  default="both",
                        choices=["code_contests", "taco", "both"])
    parser.add_argument("--out",         type=str,  default="data/sft_train.jsonl")
    parser.add_argument("--concurrency", type=int,  default=10,
                        help="并发 GPT-4o 请求数（建议 5-20，避免触发限速）")
    parser.add_argument("--run_code",    action="store_true",
                        help="启用沙箱执行校验（有测试用例时），会过滤执行错误样本")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
