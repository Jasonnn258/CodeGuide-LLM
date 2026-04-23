#!/usr/bin/env python3
"""
evals/blind_eval.py — CodeGuide-LLM 双盲评测流程

四个可独立运行的 Phase（--phase 控制）：
  prepare  — 从验证集构建 100 道未见过的测试题 → evals/test_100.jsonl
  generate — 两个模型依次生成回答（串行加载节省 VRAM）→ evals/responses_*.jsonl
  judge    — 异步调用 GPT-4o 盲测评审 → evals/judge_results.jsonl
  report   — 汇总统计并生成 evals/eval_report.md
  all      — 依次执行以上全部（默认）

中间产物（均支持断点续传）：
  evals/test_100.jsonl          — 100 道测试题（含 public_tests）
  evals/responses_codeguide.jsonl  — CodeGuide-LLM 的回答
  evals/responses_baseline.jsonl   — 基座模型 Qwen2.5-Coder-7B 的回答
  evals/judge_results.jsonl        — GPT-4o 裁判原始输出 + 映射关系
  evals/eval_report.md             — 最终评测报告

用法：
  # 一键运行完整评测（约 2~4 小时）
  python evals/blind_eval.py --phase all

  # 单独重跑某阶段（例如只重新生成报告）
  python evals/blind_eval.py --phase report

  # 自定义路径
  python evals/blind_eval.py --phase all \\
      --codeguide_model models/codeguide_llm_merged \\
      --baseline_model  Qwen/Qwen2.5-Coder-7B-Instruct
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
import random
import re
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("blind_eval")

# ── 路径常量 ─────────────────────────────────────────────────────
EVAL_DIR        = Path(__file__).resolve().parent
TEST_SET_PATH   = EVAL_DIR / "test_100.jsonl"
RESP_CODEGUIDE  = EVAL_DIR / "responses_codeguide.jsonl"
RESP_BASELINE   = EVAL_DIR / "responses_baseline.jsonl"
JUDGE_RESULTS   = EVAL_DIR / "judge_results.jsonl"
REPORT_PATH     = EVAL_DIR / "eval_report.md"


# ════════════════════════════════════════════════════════════════
# Phase 1 — 测试集准备
# ════════════════════════════════════════════════════════════════

def _load_train_ids(sft_path: str) -> set[str]:
    """从 sft_train.jsonl 读取所有题目 ID，用于排除训练集。"""
    ids: set[str] = set()
    p = Path(sft_path)
    if not p.exists():
        return ids
    with open(p, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip()) if line.strip() else {}
            if "id" in obj:
                ids.add(obj["id"])
    logger.info("训练集 ID 数量：%d", len(ids))
    return ids


def prepare_test_set(
    sft_train_path: str = "data/sft_train.jsonl",
    n: int = 100,
    seed: int = 42,
) -> None:
    """
    从 code_contests valid split + TACO 中采样 100 道未见题目。

    策略：
    - deepmind/code_contests 的 valid split 与训练集天然隔离
    - BAAI/TACO 没有独立验证集，通过 ID 排重
    - 过滤 easy/medium，去重，随机采样 n 道
    """
    if TEST_SET_PATH.exists():
        lines = TEST_SET_PATH.read_text().strip().splitlines()
        if len(lines) >= n:
            logger.info("test_100.jsonl 已存在（%d 条），跳过准备阶段", len(lines))
            return

    from src.data.loader import load_problems

    train_ids = _load_train_ids(sft_train_path)

    # code_contests valid split（验证集未参与训练）
    logger.info("加载 code_contests valid split …")
    valid_problems = load_problems(
        source="code_contests",
        split="valid",
        max_items=2000,
        deduplicate=True,
    )
    valid_problems = [p for p in valid_problems if p.id not in train_ids]
    logger.info("code_contests valid（过滤后）：%d 条", len(valid_problems))

    # TACO（按 ID 排重）
    logger.info("加载 BAAI/TACO（排除训练集 ID）…")
    taco_problems = load_problems(
        source="taco",
        split="train",
        max_items=5000,
        deduplicate=True,
    )
    taco_problems = [p for p in taco_problems if p.id not in train_ids]
    logger.info("TACO（过滤后）：%d 条", len(taco_problems))

    pool = valid_problems + taco_problems
    if len(pool) < n:
        logger.warning("可用题目不足 %d 道（%d 道），全部使用", n, len(pool))
        n = len(pool)

    random.seed(seed)
    sampled = random.sample(pool, n)
    diff_dist = {d: sum(1 for p in sampled if p.difficulty == d) for d in ("easy", "medium")}
    logger.info("采样完成：%s", diff_dist)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(TEST_SET_PATH, "w", encoding="utf-8") as f:
        for p in sampled:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    logger.info("测试集已保存：%s", TEST_SET_PATH)


# ════════════════════════════════════════════════════════════════
# Phase 2 — 模型生成
# ════════════════════════════════════════════════════════════════

_EVAL_SYSTEM = (
    "你是 CodeGuide，一位专为 OI/ACM 初学者设计的算法教学助手。"
    "当用户提出一道算法题时，你会逐步讲解解题思路：先理解题意，"
    "再从暴力解出发推导最优解，最后给出带详细注释的完整 Python 代码。"
    "讲解应当通俗易懂，适合初学者。"
)


def _load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["id"])
    return done


def _generate_responses(
    model_path: str,
    model_label: str,
    out_path: Path,
    max_new_tokens: int = 1024,
) -> None:
    """
    加载模型，对 test_100.jsonl 中每道题生成一个教学回答。

    使用 temperature=0（确定性贪心解码）保证评测可复现。
    逐条写入 out_path，支持断点续传（跳过已有 ID）。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not TEST_SET_PATH.exists():
        raise FileNotFoundError(f"测试集不存在：{TEST_SET_PATH}，请先运行 --phase prepare")

    problems = [json.loads(l) for l in TEST_SET_PATH.read_text().splitlines() if l.strip()]
    done_ids = _load_done_ids(out_path)
    pending  = [p for p in problems if p["id"] not in done_ids]

    if not pending:
        logger.info("[%s] 所有回答已生成，跳过", model_label)
        return

    logger.info("[%s] 加载模型：%s", model_label, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for i, prob in enumerate(pending):
            logger.info("[%s] 生成 %d/%d — %s", model_label, i + 1, len(pending), prob["id"])
            messages = [
                {"role": "system", "content": _EVAL_SYSTEM},
                {"role": "user",   "content": prob["description"]},
            ]
            text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536).to(model.device)

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens = max_new_tokens,
                    do_sample      = False,     # temperature=0，确定性解码
                    pad_token_id   = tokenizer.pad_token_id,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            response   = tokenizer.decode(new_tokens, skip_special_tokens=True)

            record = {"id": prob["id"], "response": response, "model": model_label}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    # 显式释放 VRAM，供下一个模型使用
    logger.info("[%s] 生成完毕，释放 VRAM…", model_label)
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def generate_responses(
    codeguide_model: str = "models/codeguide_llm_merged",
    baseline_model: str  = "Qwen/Qwen2.5-Coder-7B-Instruct",
    max_new_tokens: int  = 1024,
) -> None:
    """依次生成 CodeGuide 和 Baseline 的回答（串行加载，避免 VRAM 超限）。"""
    _generate_responses(codeguide_model, "codeguide", RESP_CODEGUIDE, max_new_tokens)
    _generate_responses(baseline_model,  "baseline",  RESP_BASELINE,  max_new_tokens)


# ════════════════════════════════════════════════════════════════
# Phase 3 — GPT-4o 盲测评审
# ════════════════════════════════════════════════════════════════

_JUDGE_SYSTEM = (
    "你是一位严格、公正的算法教学评委，专门评估面向 OI/ACM 初学者的讲解质量。"
    "你不知道两个回答分别来自哪个模型，评分应完全基于内容质量。"
)

_JUDGE_USER_TMPL = textwrap.dedent("""\
你是一位严格的算法教学评委。以下是两个助手对同一道题的讲解回答，
请从以下三个维度各打 1-5 分，并选出整体更优的一个（A 或 B），给出理由。

【评分维度说明】
- 讲解易懂性（clarity）：语言是否清晰、表达是否准确、例子是否有助理解
- 思路连贯性（coherence）：从问题到解法的推导是否流畅、逻辑是否前后一致
- 初学者友好度（beginner_friendly）：是否使用生活化比喻、是否避免过专业术语、
  是否步步铺垫让零基础读者也能跟上（1分=完全看不懂，5分=零基础也能理解）

题目：
{problem}

回答A：
{response_1}

回答B：
{response_2}

请严格按如下 JSON 格式返回，不要输出其他内容：
{{"clarity_1": <int 1-5>, "clarity_2": <int 1-5>,
 "coherence_1": <int 1-5>, "coherence_2": <int 1-5>,
 "beginner_friendly_1": <int 1-5>, "beginner_friendly_2": <int 1-5>,
 "winner": "<A 或 B>", "reason": "<简短评价，100字以内>"}}\
""")


def _parse_judge_json(raw: str) -> dict | None:
    """
    从 GPT-4o 输出中提取 JSON。
    处理两种常见格式：
      1. 纯 JSON
      2. ```json ... ``` 包裹的 JSON
    """
    # 优先提取 markdown 代码块内的 JSON
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)

    # 提取最外层的 {...}
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        raw = brace.group(0)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def _judge_one(
    client,
    problem_id: str,
    problem_desc: str,
    resp_codeguide: str,
    resp_baseline: str,
    flip: bool,                # True → codeguide 作为回答B展示
    semaphore: asyncio.Semaphore,
    retry: int = 3,
) -> dict | None:
    """
    对一道题调用 GPT-4o 进行盲测评审。

    flip=False → 回答A=codeguide, 回答B=baseline
    flip=True  → 回答A=baseline,  回答B=codeguide
    """
    r1 = resp_baseline  if flip else resp_codeguide
    r2 = resp_codeguide if flip else resp_baseline

    # 截断超长回答（避免超出 context）
    r1 = r1[:3000] if len(r1) > 3000 else r1
    r2 = r2[:3000] if len(r2) > 3000 else r2
    desc = problem_desc[:1000] if len(problem_desc) > 1000 else problem_desc

    user_content = _JUDGE_USER_TMPL.format(
        problem    = desc,
        response_1 = r1,
        response_2 = r2,
    )

    async with semaphore:
        for attempt in range(retry):
            try:
                resp = await client.chat.completions.create(
                    model       = "gpt-4o",
                    temperature = 0.0,
                    messages    = [
                        {"role": "system", "content": _JUDGE_SYSTEM},
                        {"role": "user",   "content": user_content},
                    ],
                )
                raw = resp.choices[0].message.content
                parsed = _parse_judge_json(raw)
                if parsed is None:
                    raise ValueError(f"JSON 解析失败：{raw[:200]}")

                # 将展示标签（A/B）映射到真实模型
                display_winner = str(parsed.get("winner", "")).strip().upper()
                if flip:
                    # A=baseline, B=codeguide
                    codeguide_won = (display_winner == "B")
                else:
                    # A=codeguide, B=baseline
                    codeguide_won = (display_winner == "A")

                return {
                    "id":                         problem_id,
                    "flip":                       flip,
                    "clarity_codeguide":          parsed["clarity_2"] if flip else parsed["clarity_1"],
                    "clarity_baseline":           parsed["clarity_1"] if flip else parsed["clarity_2"],
                    "coherence_codeguide":        parsed["coherence_2"] if flip else parsed["coherence_1"],
                    "coherence_baseline":         parsed["coherence_1"] if flip else parsed["coherence_2"],
                    "beginner_friendly_codeguide": parsed.get("beginner_friendly_2" if flip else "beginner_friendly_1", 3),
                    "beginner_friendly_baseline":  parsed.get("beginner_friendly_1" if flip else "beginner_friendly_2", 3),
                    "judge_display_winner":       display_winner,
                    "codeguide_won":              codeguide_won,
                    "reason":                     parsed.get("reason", ""),
                    "raw_judge":                  raw,
                }
            except Exception as e:
                import asyncio as _asyncio
                wait = 2 ** attempt
                logger.warning("[%s] 评审失败（%s），%.0fs 后重试…", problem_id, e, wait)
                await _asyncio.sleep(wait)

    logger.error("[%s] 评审放弃", problem_id)
    return None


async def _run_judge_async(concurrency: int = 5) -> None:
    """异步并发调用 GPT-4o 完成所有题目的盲测评审。"""
    from openai import AsyncOpenAI
    from tqdm.asyncio import tqdm as async_tqdm

    if not RESP_CODEGUIDE.exists() or not RESP_BASELINE.exists():
        raise FileNotFoundError("回答文件不存在，请先运行 --phase generate")

    # 加载两个模型的回答
    cg_map = {r["id"]: r["response"]
              for r in (json.loads(l) for l in RESP_CODEGUIDE.read_text().splitlines() if l.strip())}
    bl_map = {r["id"]: r["response"]
              for r in (json.loads(l) for l in RESP_BASELINE.read_text().splitlines() if l.strip())}
    problems = [json.loads(l) for l in TEST_SET_PATH.read_text().splitlines() if l.strip()]

    # 已完成的 ID（断点续传）
    done_ids: set[str] = set()
    if JUDGE_RESULTS.exists():
        for l in JUDGE_RESULTS.read_text().splitlines():
            if l.strip():
                done_ids.add(json.loads(l)["id"])
    logger.info("评审已完成：%d / %d", len(done_ids), len(problems))

    pending = [p for p in problems if p["id"] not in done_ids
               and p["id"] in cg_map and p["id"] in bl_map]
    if not pending:
        logger.info("所有评审已完成，跳过")
        return

    # 为每道题确定展示顺序（固定 seed 保证可复现）
    rng = random.Random(42)
    flips = {p["id"]: rng.choice([True, False]) for p in pending}

    client    = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        _judge_one(
            client, p["id"], p["description"],
            cg_map[p["id"]], bl_map[p["id"]],
            flips[p["id"]], semaphore,
        )
        for p in pending
    ]

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    lock = asyncio.Lock()
    with open(JUDGE_RESULTS, "a", encoding="utf-8") as f:
        for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="GPT-4o 评审"):
            result = await coro
            if result:
                async with lock:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()


def run_judge(concurrency: int = 5) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("请先 export OPENAI_API_KEY=sk-...")
    asyncio.run(_run_judge_async(concurrency))


# ════════════════════════════════════════════════════════════════
# Phase 4 — 统计与报告生成
# ════════════════════════════════════════════════════════════════

@dataclass
class EvalStats:
    total:               int   = 0
    codeguide_wins:      int   = 0
    baseline_wins:       int   = 0
    ties:                int   = 0           # 分数相等时视为平局

    clarity_codeguide:            list  = field(default_factory=list)
    clarity_baseline:             list  = field(default_factory=list)
    coherence_codeguide:          list  = field(default_factory=list)
    coherence_baseline:           list  = field(default_factory=list)
    beginner_friendly_codeguide:  list  = field(default_factory=list)
    beginner_friendly_baseline:   list  = field(default_factory=list)

    # Pass@1（只统计有 public_tests 的题目）
    pass1_codeguide:     list  = field(default_factory=list)  # float per problem
    pass1_baseline:      list  = field(default_factory=list)
    pass1_n:             int   = 0            # 参与 Pass@1 统计的题目数

    # 案例（用于报告示例）
    win_cases:  list = field(default_factory=list)   # codeguide 明显胜出
    loss_cases: list = field(default_factory=list)   # codeguide 明显落败

    @property
    def win_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.codeguide_wins / self.total

    @property
    def avg_clarity_cg(self) -> float:
        return mean(self.clarity_codeguide) if self.clarity_codeguide else 0.0

    @property
    def avg_clarity_bl(self) -> float:
        return mean(self.clarity_baseline) if self.clarity_baseline else 0.0

    @property
    def avg_coherence_cg(self) -> float:
        return mean(self.coherence_codeguide) if self.coherence_codeguide else 0.0

    @property
    def avg_coherence_bl(self) -> float:
        return mean(self.coherence_baseline) if self.coherence_baseline else 0.0

    @property
    def avg_beginner_cg(self) -> float:
        return mean(self.beginner_friendly_codeguide) if self.beginner_friendly_codeguide else 0.0

    @property
    def avg_beginner_bl(self) -> float:
        return mean(self.beginner_friendly_baseline) if self.beginner_friendly_baseline else 0.0

    @property
    def pass1_cg(self) -> float | str:
        return mean(self.pass1_codeguide) if self.pass1_codeguide else "N/A"

    @property
    def pass1_bl(self) -> float | str:
        return mean(self.pass1_baseline) if self.pass1_baseline else "N/A"


def _compute_pass1(resp_map: dict[str, str], problems: list[dict]) -> dict[str, float]:
    """
    对有 public_tests 的题目计算 Pass@1。
    返回 {problem_id: pass_rate}。
    """
    from src.data.code_validator import extract_code
    from src.reward_functions import accuracy_reward

    results: dict[str, float] = {}
    for prob in problems:
        tc = prob.get("public_tests", [])
        if not tc:
            continue
        pid = prob["id"]
        if pid not in resp_map:
            continue
        code = extract_code(resp_map[pid]) or ""
        results[pid] = accuracy_reward(code, tc, timeout=5.0)
    return results


def _bootstrap_ci(
    data: list[float],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    非参数 Bootstrap 置信区间（均值统计量）。

    返回 (lower, upper) 构成的双侧 CI。

    设计选择：
    - 使用百分位法（percentile bootstrap），无正态假设
    - n_bootstrap=2000 在 100 条样本下标准误 < 0.001
    - 固定 seed 保证报告可复现

    用法：
        lower, upper = _bootstrap_ci(scores, ci=0.95)
        print(f"{mean(scores):.3f} [{lower:.3f}, {upper:.3f}]")
    """
    import random as _random

    if not data:
        return (0.0, 0.0)
    rng = _random.Random(seed)
    n = len(data)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(data) for _ in range(n)]
        boot_means.append(mean(sample))

    boot_means.sort()
    alpha = 1.0 - ci
    lo_idx = int(alpha / 2 * n_bootstrap)
    hi_idx = int((1 - alpha / 2) * n_bootstrap) - 1
    return (round(boot_means[lo_idx], 4), round(boot_means[hi_idx], 4))


def _bootstrap_win_rate_ci(
    wins: list[int],     # 1=codeguide won, 0=baseline won
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Win-rate 的 bootstrap 置信区间（比例统计量）。"""
    import random as _random

    if not wins:
        return (0.0, 0.0)
    rng = _random.Random(seed)
    n = len(wins)
    boot_rates = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(wins) for _ in range(n)]
        boot_rates.append(sum(sample) / n)

    boot_rates.sort()
    alpha = 1.0 - ci
    lo_idx = int(alpha / 2 * n_bootstrap)
    hi_idx = int((1 - alpha / 2) * n_bootstrap) - 1
    return (round(boot_rates[lo_idx], 4), round(boot_rates[hi_idx], 4))


def _significance_test(
    cg_scores: list[float],
    bl_scores: list[float],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, bool]:
    """
    配对 Bootstrap 显著性检验（双侧，H0: mean(cg) == mean(bl)）。

    返回 (p_value, is_significant_at_05)。

    方法：Paired bootstrap permutation 风格
    - 计算观测差值 delta = mean(cg) - mean(bl)
    - Bootstrap 抽样重估零分布
    - p_value = fraction of bootstraps where |boot_delta| >= |delta|
    """
    import random as _random

    if not cg_scores or not bl_scores or len(cg_scores) != len(bl_scores):
        return (1.0, False)

    rng = _random.Random(seed)
    n = len(cg_scores)
    obs_delta = mean(cg_scores) - mean(bl_scores)

    # 在 H0 下，配对差值以零为中心 → 随机翻转符号
    diffs = [c - b for c, b in zip(cg_scores, bl_scores)]
    extreme = 0
    for _ in range(n_bootstrap):
        boot_delta = mean([rng.choice([-1, 1]) * d for d in diffs])
        if abs(boot_delta) >= abs(obs_delta):
            extreme += 1

    p_value = extreme / n_bootstrap
    return (round(p_value, 4), p_value < 0.05)


def _ascii_bar(value: float, width: int = 20, char: str = "█") -> str:
    filled = round(value * width)
    return char * filled + "░" * (width - filled)


def generate_report() -> None:
    """汇总评审结果，输出 eval_report.md。"""
    if not JUDGE_RESULTS.exists():
        raise FileNotFoundError("评审结果不存在，请先运行 --phase judge")

    problems   = {p["id"]: p for p in
                  (json.loads(l) for l in TEST_SET_PATH.read_text().splitlines() if l.strip())}
    cg_map     = {r["id"]: r["response"]
                  for r in (json.loads(l) for l in RESP_CODEGUIDE.read_text().splitlines() if l.strip())}
    bl_map     = {r["id"]: r["response"]
                  for r in (json.loads(l) for l in RESP_BASELINE.read_text().splitlines() if l.strip())}
    judge_rows = [json.loads(l) for l in JUDGE_RESULTS.read_text().splitlines() if l.strip()]

    # ── Pass@1 ────────────────────────────────────────────────
    logger.info("计算 Pass@1（有测试用例的题目）…")
    pass1_cg = _compute_pass1(cg_map, list(problems.values()))
    pass1_bl = _compute_pass1(bl_map, list(problems.values()))

    # ── 聚合统计 ──────────────────────────────────────────────
    stats = EvalStats()
    win_indicators: list[int] = []  # 1=codeguide won, 0=baseline won（用于 win-rate CI）
    for row in judge_rows:
        stats.total += 1
        won = row["codeguide_won"]

        # 平局判断：双方分数完全相同
        cg_sum = row["clarity_codeguide"]  + row["coherence_codeguide"]
        bl_sum = row["clarity_baseline"]   + row["coherence_baseline"]
        if cg_sum > bl_sum:
            stats.codeguide_wins += 1
            win_indicators.append(1)
        elif cg_sum < bl_sum:
            stats.baseline_wins += 1
            win_indicators.append(0)
        else:
            # 分数相同时以 judge_winner 决定
            if won:
                stats.codeguide_wins += 1
                win_indicators.append(1)
            else:
                stats.baseline_wins += 1
                win_indicators.append(0)

        stats.clarity_codeguide.append(row["clarity_codeguide"])
        stats.clarity_baseline.append(row["clarity_baseline"])
        stats.coherence_codeguide.append(row["coherence_codeguide"])
        stats.coherence_baseline.append(row["coherence_baseline"])
        # beginner_friendly（兼容旧格式：字段缺失时默认 3 分）
        stats.beginner_friendly_codeguide.append(
            row.get("beginner_friendly_codeguide", 3)
        )
        stats.beginner_friendly_baseline.append(
            row.get("beginner_friendly_baseline", 3)
        )

        pid = row["id"]
        if pid in pass1_cg:
            stats.pass1_codeguide.append(pass1_cg[pid])
            stats.pass1_baseline.append(pass1_bl.get(pid, 0.0))
            stats.pass1_n += 1

        # 收集案例（按分差排序）
        margin = (row["clarity_codeguide"] + row["coherence_codeguide"]
                  - row["clarity_baseline"]  - row["coherence_baseline"])
        case = {
            "id": pid,
            "margin": margin,
            "reason": row.get("reason", ""),
            "problem_snippet": problems.get(pid, {}).get("description", "")[:200],
        }
        if margin >= 2:
            stats.win_cases.append(case)
        elif margin <= -2:
            stats.loss_cases.append(case)

    stats.win_cases  = sorted(stats.win_cases,  key=lambda x: -x["margin"])[:3]
    stats.loss_cases = sorted(stats.loss_cases, key=lambda x:  x["margin"])[:3]

    # ── 分难度分析 ────────────────────────────────────────────
    by_diff: dict[str, dict] = {"easy": {"wins": 0, "total": 0}, "medium": {"wins": 0, "total": 0}}
    for row in judge_rows:
        pid  = row["id"]
        diff = problems.get(pid, {}).get("difficulty", "unknown")
        if diff in by_diff:
            by_diff[diff]["total"] += 1
            if row["codeguide_won"]:
                by_diff[diff]["wins"] += 1

    # ── Bootstrap 置信区间计算 ────────────────────────────────
    logger.info("计算 Bootstrap 95%% 置信区间…")
    wr_lo, wr_hi = _bootstrap_win_rate_ci(win_indicators)
    cl_cg_lo, cl_cg_hi = _bootstrap_ci(stats.clarity_codeguide)
    cl_bl_lo, cl_bl_hi = _bootstrap_ci(stats.clarity_baseline)
    co_cg_lo, co_cg_hi = _bootstrap_ci(stats.coherence_codeguide)
    co_bl_lo, co_bl_hi = _bootstrap_ci(stats.coherence_baseline)
    bg_cg_lo, bg_cg_hi = _bootstrap_ci(stats.beginner_friendly_codeguide)
    bg_bl_lo, bg_bl_hi = _bootstrap_ci(stats.beginner_friendly_baseline)

    # ── 配对显著性检验 ────────────────────────────────────────
    cl_pval,  cl_sig  = _significance_test(stats.clarity_codeguide,           stats.clarity_baseline)
    co_pval,  co_sig  = _significance_test(stats.coherence_codeguide,         stats.coherence_baseline)
    bg_pval,  bg_sig  = _significance_test(stats.beginner_friendly_codeguide, stats.beginner_friendly_baseline)

    def _sig_mark(is_sig: bool) -> str:
        return "✅ p<0.05（显著）" if is_sig else "❌ p≥0.05（不显著）"

    # ── 报告渲染 ──────────────────────────────────────────────
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    p1_cg_str = f"{stats.pass1_cg:.1%}" if isinstance(stats.pass1_cg, float) else stats.pass1_cg
    p1_bl_str = f"{stats.pass1_bl:.1%}" if isinstance(stats.pass1_bl, float) else stats.pass1_bl

    report = f"""\
# CodeGuide-LLM 双盲评测报告

> 生成时间：{now}
> 评委模型：GPT-4o（temperature=0）
> 解码策略：Greedy（temperature=0，确保可复现）

---

## 一、汇总结果

| 指标 | CodeGuide-LLM | 95% CI | Qwen2.5-Coder-7B（基座） | 95% CI |
|:-----|:---:|:---:|:---:|:---:|
| **Win-rate** | **{stats.win_rate:.1%}** | [{wr_lo:.1%}, {wr_hi:.1%}] | {1 - stats.win_rate:.1%} | — |
| 讲解易懂性（1-5） | **{stats.avg_clarity_cg:.2f}** | [{cl_cg_lo:.2f}, {cl_cg_hi:.2f}] | {stats.avg_clarity_bl:.2f} | [{cl_bl_lo:.2f}, {cl_bl_hi:.2f}] |
| 思路连贯性（1-5） | **{stats.avg_coherence_cg:.2f}** | [{co_cg_lo:.2f}, {co_cg_hi:.2f}] | {stats.avg_coherence_bl:.2f} | [{co_bl_lo:.2f}, {co_bl_hi:.2f}] |
| 初学者友好度（1-5） | **{stats.avg_beginner_cg:.2f}** | [{bg_cg_lo:.2f}, {bg_cg_hi:.2f}] | {stats.avg_beginner_bl:.2f} | [{bg_bl_lo:.2f}, {bg_bl_hi:.2f}] |
| Pass@1（代码正确性） | {p1_cg_str} | — | {p1_bl_str} | — |
| 评审题目总数 | {stats.total} | — | — | — |
| Pass@1 参与题目数 | {stats.pass1_n} | — | — | — |

### Win-rate 可视化

```
CodeGuide    {_ascii_bar(stats.win_rate)} {stats.win_rate:.1%}  ({stats.codeguide_wins}题)
Baseline     {_ascii_bar(1-stats.win_rate)} {1-stats.win_rate:.1%}  ({stats.baseline_wins}题)
```

---

## 二、按难度分析

| 难度 | CodeGuide 胜 | 总计 | Win-rate |
|:-----|:---:|:---:|:---:|
| easy   | {by_diff['easy']['wins']} | {by_diff['easy']['total']} | {by_diff['easy']['wins']/max(by_diff['easy']['total'],1):.1%} |
| medium | {by_diff['medium']['wins']} | {by_diff['medium']['total']} | {by_diff['medium']['wins']/max(by_diff['medium']['total'],1):.1%} |

---

## 三、评分分布

### 讲解易懂性（Clarity）

```
CodeGuide  均值 {stats.avg_clarity_cg:.2f}  {_ascii_bar(stats.avg_clarity_cg / 5)}  95% CI [{cl_cg_lo:.2f}, {cl_cg_hi:.2f}]
Baseline   均值 {stats.avg_clarity_bl:.2f}  {_ascii_bar(stats.avg_clarity_bl / 5)}  95% CI [{cl_bl_lo:.2f}, {cl_bl_hi:.2f}]
```

### 思路连贯性（Coherence）

```
CodeGuide  均值 {stats.avg_coherence_cg:.2f}  {_ascii_bar(stats.avg_coherence_cg / 5)}  95% CI [{co_cg_lo:.2f}, {co_cg_hi:.2f}]
Baseline   均值 {stats.avg_coherence_bl:.2f}  {_ascii_bar(stats.avg_coherence_bl / 5)}  95% CI [{co_bl_lo:.2f}, {co_bl_hi:.2f}]
```

### 初学者友好度（Beginner-Friendly）

> 新增维度：评估讲解是否使用生活化比喻、避免过专业术语、步步铺垫让零基础读者跟上。

```
CodeGuide  均值 {stats.avg_beginner_cg:.2f}  {_ascii_bar(stats.avg_beginner_cg / 5)}  95% CI [{bg_cg_lo:.2f}, {bg_cg_hi:.2f}]
Baseline   均值 {stats.avg_beginner_bl:.2f}  {_ascii_bar(stats.avg_beginner_bl / 5)}  95% CI [{bg_bl_lo:.2f}, {bg_bl_hi:.2f}]
```

---

## 四、Pass@1 代码正确性

> 仅统计含公开测试用例的 {stats.pass1_n} 道题目（stdin/stdout 风格执行验证）

| 模型 | Pass@1 |
|:-----|:---:|
| CodeGuide-LLM | {p1_cg_str} |
| Qwen2.5-Coder-7B | {p1_bl_str} |

---

## 五、典型案例

### CodeGuide 明显优于基座的案例（Top 3）

{_render_cases(stats.win_cases, "CodeGuide 胜")}

### CodeGuide 明显弱于基座的案例（Top 3）

{_render_cases(stats.loss_cases, "基座胜")}

---

## 六、📊 统计显著性检验

> 使用配对 Bootstrap 显著性检验（n_bootstrap=2000，双侧，H0: 两模型均值相同）。
> p < 0.05 表示差异在 95% 置信水平下统计显著。

| 评分维度 | CodeGuide 均值 | Baseline 均值 | 差值 | p-value | 是否显著 |
|:--------|:---:|:---:|:---:|:---:|:---:|
| 讲解易懂性（Clarity） | {stats.avg_clarity_cg:.3f} | {stats.avg_clarity_bl:.3f} | {stats.avg_clarity_cg - stats.avg_clarity_bl:+.3f} | {cl_pval:.4f} | {_sig_mark(cl_sig)} |
| 思路连贯性（Coherence） | {stats.avg_coherence_cg:.3f} | {stats.avg_coherence_bl:.3f} | {stats.avg_coherence_cg - stats.avg_coherence_bl:+.3f} | {co_pval:.4f} | {_sig_mark(co_sig)} |
| 初学者友好度（Beginner） | {stats.avg_beginner_cg:.3f} | {stats.avg_beginner_bl:.3f} | {stats.avg_beginner_cg - stats.avg_beginner_bl:+.3f} | {bg_pval:.4f} | {_sig_mark(bg_sig)} |

**综合结论**：
- {"CodeGuide-LLM 在 Clarity 上显著优于基座（p<0.05）" if cl_sig else "Clarity 差异不显著（p≥0.05）"}
- {"CodeGuide-LLM 在 Coherence 上显著优于基座（p<0.05）" if co_sig else "Coherence 差异不显著（p≥0.05）"}
- {"CodeGuide-LLM 在初学者友好度上显著优于基座（p<0.05）" if bg_sig else "初学者友好度差异不显著（p≥0.05）"}

---

## 七、评测方法说明

1. **双盲设计**：对每道题，随机决定哪个模型的回答作为"回答A"展示，
   GPT-4o 裁判不知道两个回答的来源，消除位置偏差。

2. **解码策略**：两个模型均使用 Greedy decoding（temperature=0），
   保证评测结果可复现。

3. **评分维度**（v2 新增初学者友好度）：
   - 讲解易懂性（Clarity）：语言清晰度与举例质量
   - 思路连贯性（Coherence）：推导逻辑前后一致性
   - 初学者友好度（Beginner-Friendly）：是否用生活化比喻、避免专业术语
   - 代码正确性（Pass@1）：本地沙箱执行 + stdin/stdout 比对

4. **置信区间**：使用非参数百分位 Bootstrap 法（n=2000），无正态假设。

5. **显著性检验**：使用配对 Bootstrap 检验（双侧），不假设任何分布形式，
   鲁棒性高于配对 t 检验。

6. **测试集说明**：100 道题均从验证集/未见过的题目中采样，
   已通过 ID 排重确保与训练集无重叠。
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    logger.info("报告已生成：%s", REPORT_PATH)

    # 终端摘要
    logger.info(
        "\n  Win-rate: %.1f%% [%.1f%%, %.1f%%]"
        "  |  Clarity: %.2f vs %.2f (p=%.4f%s)"
        "  |  Coherence: %.2f vs %.2f (p=%.4f%s)"
        "  |  Beginner: %.2f vs %.2f (p=%.4f%s)"
        "  |  Pass@1: %s vs %s",
        stats.win_rate * 100, wr_lo * 100, wr_hi * 100,
        stats.avg_clarity_cg, stats.avg_clarity_bl, cl_pval, " *" if cl_sig else "",
        stats.avg_coherence_cg, stats.avg_coherence_bl, co_pval, " *" if co_sig else "",
        stats.avg_beginner_cg, stats.avg_beginner_bl, bg_pval, " *" if bg_sig else "",
        p1_cg_str, p1_bl_str,
    )


def _render_cases(cases: list[dict], label: str) -> str:
    if not cases:
        return f"_（无满足条件的 {label} 案例）_\n"
    lines = []
    for i, c in enumerate(cases, 1):
        snippet = c["problem_snippet"].replace("\n", " ").strip()
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        lines.append(
            f"**案例 {i}**（分差 {c['margin']:+d}）\n"
            f"- 题目：{snippet}\n"
            f"- 裁判评语：{c['reason']}\n"
        )
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CodeGuide-LLM 双盲评测流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=["prepare", "generate", "judge", "report", "all"],
        default="all",
        help="运行指定阶段（默认 all）",
    )
    parser.add_argument("--codeguide_model", default="models/codeguide_llm_merged")
    parser.add_argument("--baseline_model",  default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--sft_train_path",  default="data/sft_train.jsonl")
    parser.add_argument("--n_test",          type=int, default=100)
    parser.add_argument("--max_new_tokens",  type=int, default=1024)
    parser.add_argument("--judge_concurrency", type=int, default=5)
    args = parser.parse_args()

    phases = (
        ["prepare", "generate", "judge", "report"]
        if args.phase == "all"
        else [args.phase]
    )

    for phase in phases:
        logger.info("══ Phase: %s ══", phase.upper())
        if phase == "prepare":
            prepare_test_set(
                sft_train_path=args.sft_train_path,
                n=args.n_test,
            )
        elif phase == "generate":
            generate_responses(
                codeguide_model=args.codeguide_model,
                baseline_model =args.baseline_model,
                max_new_tokens =args.max_new_tokens,
            )
        elif phase == "judge":
            run_judge(concurrency=args.judge_concurrency)
        elif phase == "report":
            generate_report()

    logger.info("══ 评测完成 ══")


if __name__ == "__main__":
    main()
