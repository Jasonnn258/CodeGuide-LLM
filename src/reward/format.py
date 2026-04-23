"""
FormatComplianceReward: 基于规则的格式合规性检查

改进版（v2）：
- 修复 Reward Hacking：步骤标题必须有实质内容（≥50字符）才计分
- 步骤间内容多样性检测：避免模型复制粘贴同一段文字
- 改为连续评分（比例制），而非固定 threshold 跳变
- 代码块合法性验证：必须有实际代码行，不能是空块

背景：原版只检查是否"存在"步骤标题/代码块，模型可以只写空洞
      的格式符号（第一步：\n[任意文字]）拿到满分。
"""
import re
from typing import List

from omegaconf import DictConfig

# 步骤标题 pattern（匹配后紧跟的内容才是重点）
_STEP_PATTERNS = [
    re.compile(r"(第[一二三四五六七八九十\d]+步[：:])(.+?)(?=第[一二三四五六七八九十\d]+步|##|$)", re.DOTALL),
    re.compile(r"(\*\*Step \d+[：:]\*\*)(.+?)(?=\*\*Step \d+|##|$)", re.DOTALL),
    re.compile(r"(#{1,3} .+?\n)(.+?)(?=#{1,3} |$)", re.DOTALL),
    re.compile(r"(\d+\.\s+\*\*.+?\*\*)(.+?)(?=\d+\.\s+\*\*|$)", re.DOTALL),
]

# 代码块：要求至少有一行非空代码（不只是注释或空行）
_CODE_BLOCK = re.compile(r"```[\w]*\n(.*?)```", re.DOTALL)
_COMPLEXITY = re.compile(r"O\([^)]+\)|时间复杂度|空间复杂度|Time complexity")

# 教学类关键词（体现讲解性）
_TEACHING_WORDS = re.compile(
    r"因为|所以|因此|注意|关键|核心思路|观察到|我们需要|可以发现|"
    r"暴力|优化|考虑|首先|然后|最后|总结|举例"
)

_MIN_STEP_CONTENT = 50   # 步骤正文至少 50 字符才计为有效步骤
_MIN_SIMILARITY_THRESHOLD = 0.7  # 步骤间 Jaccard 相似度超过此值视为重复


def _jaccard_similarity(a: str, b: str) -> float:
    """字符级 Jaccard 相似度，用于检测步骤间重复。"""
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _score_steps(text: str) -> float:
    """
    步骤结构得分 [0, 1]。

    只有步骤标题后紧跟的正文 >= 50 字符，才视为"有效步骤"。
    最终分数 = min(有效步骤数 / 3, 1.0) * 0.4，上限 0.4。
    额外惩罚：若步骤间 Jaccard 相似度过高，视为重复扣分。
    """
    valid_steps = []
    for pattern in _STEP_PATTERNS:
        for match in pattern.finditer(text):
            content = match.group(2).strip() if len(match.groups()) >= 2 else ""
            if len(content) >= _MIN_STEP_CONTENT:
                valid_steps.append(content)

    if not valid_steps:
        return 0.0

    # 去重检测：相邻步骤过于相似则只计一个
    unique_steps = [valid_steps[0]]
    for step in valid_steps[1:]:
        if _jaccard_similarity(step, unique_steps[-1]) < _MIN_SIMILARITY_THRESHOLD:
            unique_steps.append(step)

    # 至少 3 个不同步骤才得满分
    ratio = min(len(unique_steps) / 3.0, 1.0)
    return ratio * 0.4


def _score_code_block(text: str) -> float:
    """
    代码块得分 [0, 0.3]。

    代码块必须包含至少 3 行实质代码（非空行、非纯注释行）。
    """
    blocks = _CODE_BLOCK.findall(text)
    if not blocks:
        return 0.0

    best_real_lines = 0
    for block_content in blocks:
        lines = block_content.strip().splitlines()
        real_lines = sum(
            1 for l in lines
            if l.strip() and not l.strip().startswith("#")
        )
        best_real_lines = max(best_real_lines, real_lines)

    if best_real_lines < 3:
        return 0.1  # 有代码块但内容不足，给少量分
    return 0.3


def _score_complexity(text: str) -> float:
    """复杂度分析得分 [0, 0.2]。"""
    return 0.2 if _COMPLEXITY.search(text) else 0.0


def _score_length(text: str) -> float:
    """
    长度得分 [0, 0.05]。

    改为宽松区间（150-3000），避免因讲解详细而扣分。
    """
    return 0.05 if 150 <= len(text) <= 3000 else 0.0


def _score_teaching_words(text: str) -> float:
    """
    教学词汇得分 [0, 0.05]。

    新增：检测是否出现教学导向词汇，鼓励讲解性表达。
    """
    count = len(_TEACHING_WORDS.findall(text))
    return min(count / 5.0, 1.0) * 0.05


def _score_format(text: str) -> float:
    """
    综合格式得分 [0, 1.0]，各子分项：
      步骤结构          0.40（含有效内容检测 + 多样性检测）
      代码块合法性      0.30（要求有实质代码行）
      复杂度分析        0.20
      长度合理          0.05
      教学词汇          0.05（新增）
    """
    score = (
        _score_steps(text)
        + _score_code_block(text)
        + _score_complexity(text)
        + _score_length(text)
        + _score_teaching_words(text)
    )
    return min(score, 1.0)


class FormatComplianceReward:
    def __init__(self, cfg: DictConfig):
        pass

    def __call__(self, prompts: List[str], completions: List[str]) -> List[float]:
        return [_score_format(c) for c in completions]
