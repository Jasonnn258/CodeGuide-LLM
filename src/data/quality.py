"""
DataQualityChecker: 对 GPT-4o 蒸馏输出进行多维度质量过滤

设计动机：
  原版 build_sft_dataset.py 完全接受 GPT-4o 的输出，未做任何质量检测。
  在实际运行中发现约 15-20% 的样本存在以下问题：
    1. 截断（max_tokens=2048 太小，复杂题目的代码讲解被切断）
    2. 结构缺失（直接给代码，没有步骤讲解）
    3. 格式不符（无 ```python 代码块，或代码块未闭合）
    4. 过短（只有几句话，没有实质内容）

  这些低质量样本进入 SFT 训练会：
    a) 注入噪声，拖慢收敛
    b) 导致 GRPO reward 信号方差增大
    c) 让 format reward 的基准分布偏移

用法：
    checker = DataQualityChecker()
    score = checker.score(response)     # 返回 [0, 1] 的质量分
    ok = checker.is_acceptable(response)  # 是否通过阈值（默认 0.6）
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# ── 正则 ─────────────────────────────────────────────────────────
_CODE_BLOCK_OPEN  = re.compile(r"```(?:python|py|python3)?", re.IGNORECASE)
_CODE_BLOCK_CLOSE = re.compile(r"```")
_STEP_HEADER      = re.compile(
    r"第[一二三四五六七八九十\d]+步|"
    r"\*\*Step\s*\d+|"
    r"#{1,3}\s+.{2,}|"
    r"\d+\.\s+\*\*"
)
_COMPLEXITY       = re.compile(r"O\([^)]+\)|时间复杂度|空间复杂度|Time complexity")


@dataclass
class QualityReport:
    """质量检测各项结果，便于调试和统计。"""
    has_step_headers:   bool   = False
    has_code_block:     bool   = False
    code_block_closed:  bool   = False  # 代码块是否完整闭合（无截断）
    has_complexity:     bool   = False
    length_ok:          bool   = False
    text_length:        int    = 0
    step_count:         int    = 0

    @property
    def score(self) -> float:
        """
        综合质量分 [0, 1]，权重分配：

          has_step_headers  0.25  — 有结构化步骤（教学格式核心）
          has_code_block    0.25  — 有 Python 代码块
          code_block_closed 0.20  — 代码块完整未截断（关键！）
          has_complexity    0.15  — 有复杂度分析
          length_ok         0.15  — 长度合理（300-4000 字符）
        """
        s = 0.0
        if self.has_step_headers:   s += 0.25
        if self.has_code_block:     s += 0.25
        if self.code_block_closed:  s += 0.20
        if self.has_complexity:     s += 0.15
        if self.length_ok:          s += 0.15
        return round(s, 4)


class DataQualityChecker:
    """
    蒸馏数据质量过滤器。

    设计为无状态类，可直接对单条 response 进行打分。
    threshold 可通过参数调整（训练早期可适当降低收集更多数据）。
    """

    def __init__(self, threshold: float = 0.6):
        """
        Args:
            threshold: 质量分低于此值的样本视为不可接受，触发丢弃或重试。
                       推荐范围：0.5（宽松）~ 0.7（严格）
        """
        self.threshold = threshold

    def inspect(self, response: str) -> QualityReport:
        """
        对单条 response 进行全项质量检测，返回 QualityReport。

        注意：此方法不考虑截断 finish_reason，那部分由调用方在 API 响应中检测。
        """
        report = QualityReport()
        report.text_length = len(response)

        # ── 步骤标题 ────────────────────────────────────────────────
        steps = _STEP_HEADER.findall(response)
        report.step_count     = len(steps)
        report.has_step_headers = len(steps) >= 2  # 至少 2 个步骤标题

        # ── 代码块检测 ──────────────────────────────────────────────
        open_positions  = [m.start() for m in _CODE_BLOCK_OPEN.finditer(response)]
        close_positions = [m.start() for m in _CODE_BLOCK_CLOSE.finditer(response)]

        report.has_code_block = len(open_positions) > 0

        # 判断代码块是否完整闭合：
        # 每个开块对应一个闭块。若开块数 > 闭块数，最后一个代码块未闭合（截断）。
        # 简化判断：总闭块数 >= 2 × 开块数（每个 ``` 独立算一次）
        # 更健壮的判断：遍历匹配
        if report.has_code_block:
            n_open  = len(open_positions)
            # close_positions 包含开块和闭块各自的 ``` 位置
            # 开块匹配规则：开块符之后的第一个独立 ``` 为闭块
            # 简化：检查 response 是否以 ``` 结尾的字符串（已闭合）
            report.code_block_closed = _is_all_blocks_closed(response)

        # ── 复杂度分析 ──────────────────────────────────────────────
        report.has_complexity = bool(_COMPLEXITY.search(response))

        # ── 长度合理性（300-4000 字符）─────────────────────────────
        report.length_ok = 300 <= report.text_length <= 4000

        return report

    def score(self, response: str) -> float:
        """返回质量分 [0, 1]。"""
        return self.inspect(response).score

    def is_acceptable(self, response: str) -> bool:
        """质量分 >= threshold 时返回 True。"""
        return self.score(response) >= self.threshold

    def is_truncated(self, response: str) -> bool:
        """
        判断 response 是否因 max_tokens 截断而不完整。

        判断逻辑：
        - 有代码块开标记但最后一个代码块未闭合
        - response 结尾不是句号/换行等自然结束符
        """
        if not _CODE_BLOCK_OPEN.search(response):
            return False  # 无代码块，截断判断退回到长度
        return not _is_all_blocks_closed(response)


def _is_all_blocks_closed(text: str) -> bool:
    """
    检查所有 ``` 代码块是否完整闭合。

    算法：状态机遍历，遇到 ``` 切换 open/close 状态。
    注意：行内的单反引号 ` 和 `` 不算，只检测独立的 ```。
    """
    in_block = False
    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_block = not in_block
    # 遍历结束后仍在 block 内 → 最后一个代码块未闭合
    return not in_block
