"""
FormatComplianceReward: 基于规则的格式合规性检查
检查步进式讲解是否包含必要的结构标记
"""
import re
from typing import List

from omegaconf import DictConfig

# 至少要出现其中一种步骤标题形式
_STEP_PATTERNS = [
    re.compile(r"第[一二三四五六七八九十\d]+步"),
    re.compile(r"\*\*Step \d+"),
    re.compile(r"#{1,3} "),           # Markdown 标题
    re.compile(r"\d+\.\s+\*\*"),      # 有序列表加粗标题
]
_CODE_BLOCK = re.compile(r"```[\w]*\n.*?```", re.DOTALL)
_COMPLEXITY = re.compile(r"O\([^)]+\)|时间复杂度|空间复杂度|Time complexity")


def _score_format(text: str) -> float:
    score = 0.0
    # 1. 包含步骤结构 (+0.4)
    if any(p.search(text) for p in _STEP_PATTERNS):
        score += 0.4
    # 2. 包含代码块 (+0.3)
    if _CODE_BLOCK.search(text):
        score += 0.3
    # 3. 包含复杂度分析 (+0.2)
    if _COMPLEXITY.search(text):
        score += 0.2
    # 4. 长度合理（200-2000 字符）(+0.1)
    if 200 <= len(text) <= 2000:
        score += 0.1
    return min(score, 1.0)


class FormatComplianceReward:
    def __init__(self, cfg: DictConfig):
        pass

    def __call__(self, prompts: List[str], completions: List[str]) -> List[float]:
        return [_score_format(c) for c in completions]
