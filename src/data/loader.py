"""
数据集加载与归一化

支持两个数据源，对外暴露统一的 Problem 结构：
  - deepmind/code_contests
  - BAAI/TACO

用法：
    from src.data.loader import load_problems
    problems = load_problems(source="taco", split="train", max_items=10000)
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)

# ── 难度常量 ────────────────────────────────────────────────

# code_contests 的 difficulty 字段是 int（枚举值）
# 文档：1=UNKNOWN, 2=EASY, 3=MEDIUM, 4=HARD, 5=HARDER, 6=HARDEST
_CC_EASY_MEDIUM = {2, 3}  # EASY / MEDIUM

# TACO 的 difficulty 字段是字符串
_TACO_EASY_MEDIUM = {"EASY", "MEDIUM"}


# ── 数据结构 ─────────────────────────────────────────────────

@dataclass
class Problem:
    id: str                          # 全局唯一 ID（由来源+题目 hash 生成）
    source: str                      # "code_contests" | "taco"
    description: str                 # 题目描述（原始文本，已去 HTML）
    difficulty: str                  # "easy" | "medium"
    tags: List[str] = field(default_factory=list)
    # 参考答案（来自数据集自带，用于代码校验时构建测试输入）
    reference_solution: Optional[str] = None
    # 原始测试用例（input/output 字符串对）
    public_tests: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "description": self.description,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "reference_solution": self.reference_solution,
            "public_tests": self.public_tests,
        }


# ── 工具函数 ─────────────────────────────────────────────────

def _make_id(source: str, text: str) -> str:
    h = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:10]
    return f"{source}_{h}"


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&nbsp;", " ", text)
    return text.strip()


def _pick_python_solution(solutions: dict | list) -> Optional[str]:
    """
    从 code_contests 的 solutions 字段取一段 Python 参考实现。
    solutions 格式为 {"language": [3, 3], "solution": ["code1", "code2"]}
    language 枚举：3 = PYTHON3
    """
    if isinstance(solutions, dict):
        langs = solutions.get("language", [])
        codes = solutions.get("solution", [])
        for lang, code in zip(langs, codes):
            if lang == 3:  # PYTHON3
                return code
    return None


# ── code_contests 加载器 ─────────────────────────────────────

def _load_code_contests(split: str, max_items: int) -> List[Problem]:
    logger.info("加载 deepmind/code_contests (%s)…", split)
    try:
        ds = load_dataset("deepmind/code_contests", split=split, trust_remote_code=True)
    except Exception as e:
        logger.error("code_contests 加载失败：%s", e)
        return []

    problems: List[Problem] = []
    for row in ds:
        if len(problems) >= max_items:
            break

        diff_int = row.get("difficulty", 0)
        if diff_int not in _CC_EASY_MEDIUM:
            continue

        desc = _strip_html(row.get("description", ""))
        if len(desc) < 80:  # 过短说明数据有问题
            continue

        ref = _pick_python_solution(row.get("solutions", {}))

        # 公开测试用例
        pub = row.get("public_tests", {})
        tests = [
            {"input": inp, "output": out}
            for inp, out in zip(
                pub.get("input", []), pub.get("output", [])
            )
        ]

        diff_str = "easy" if diff_int == 2 else "medium"
        problems.append(Problem(
            id=_make_id("cc", desc),
            source="code_contests",
            description=desc,
            difficulty=diff_str,
            tags=[],
            reference_solution=ref,
            public_tests=tests,
        ))

    logger.info("code_contests 筛选后：%d 条", len(problems))
    return problems


# ── TACO 加载器 ──────────────────────────────────────────────

def _load_taco(split: str, max_items: int) -> List[Problem]:
    logger.info("加载 BAAI/TACO (%s)…", split)
    try:
        # TACO 只有 train split
        ds = load_dataset("BAAI/TACO", split="train", trust_remote_code=True)
    except Exception as e:
        logger.error("TACO 加载失败：%s", e)
        return []

    problems: List[Problem] = []
    for row in ds:
        if len(problems) >= max_items:
            break

        diff = (row.get("difficulty") or "").upper()
        if diff not in _TACO_EASY_MEDIUM:
            continue

        desc = _strip_html(row.get("question", ""))
        if len(desc) < 80:
            continue

        # TACO solutions 是 JSON 字符串列表
        ref = None
        raw_solutions = row.get("solutions", "[]")
        try:
            import json
            sols = json.loads(raw_solutions) if isinstance(raw_solutions, str) else raw_solutions
            if sols:
                ref = sols[0]
        except Exception:
            pass

        # TACO 没有结构化的公开测试用例，跳过
        tags_raw = row.get("tags") or []
        tags = tags_raw if isinstance(tags_raw, list) else [tags_raw]

        problems.append(Problem(
            id=_make_id("taco", desc),
            source="taco",
            description=desc,
            difficulty=diff.lower(),
            tags=tags,
            reference_solution=ref,
            public_tests=[],
        ))

    logger.info("TACO 筛选后：%d 条", len(problems))
    return problems


# ── 公共接口 ─────────────────────────────────────────────────

def load_problems(
    source: Literal["code_contests", "taco", "both"] = "both",
    split: str = "train",
    max_items: int = 10_000,
    deduplicate: bool = True,
) -> List[Problem]:
    """
    加载并归一化题目数据集。

    Args:
        source:      数据来源，"both" 时合并两个数据集
        split:       HF split 名称（code_contests 有 train/valid/test；TACO 只有 train）
        max_items:   每个来源最多加载多少条（过滤后）
        deduplicate: 按题目描述前 200 字去重
    Returns:
        List[Problem]，已按难度排序（easy 优先）
    """
    problems: List[Problem] = []

    if source in ("code_contests", "both"):
        problems += _load_code_contests(split, max_items)

    if source in ("taco", "both"):
        remaining = max(0, max_items - len(problems))
        if remaining > 0:
            problems += _load_taco(split, remaining)

    if deduplicate:
        seen: set[str] = set()
        deduped: List[Problem] = []
        for p in problems:
            key = p.description[:200]
            if key not in seen:
                seen.add(key)
                deduped.append(p)
        logger.info("去重后：%d → %d 条", len(problems), len(deduped))
        problems = deduped

    # easy 排前面，方便 seed 采样时优先选简单题
    problems.sort(key=lambda p: 0 if p.difficulty == "easy" else 1)
    return problems[:max_items]
