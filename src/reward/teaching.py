"""
Teaching Reward 模块

提供两种实现，通过 configs/train_config.yaml 的 reward.teaching_mode 字段切换：

  local  (默认) — LocalTeachingReward
    无 API 调用，基于启发式规则的连续评分 [0, 1]。
    延迟：~1ms/样本（vs GPT-4o-mini 的 2-5s）。
    适合：GRPO 训练循环中每 step 调用。

  api           — TeachingCompletenessReward
    调用 GPT-4o-mini 作 LLM-as-Judge，返回 0-10 离散整数 → 归一化。
    适合：离线评估、ablation 对比。

设计背景：
  原版仅有 TeachingCompletenessReward（API 版）。在 GRPO 训练中，
  每个 batch 需对 4×batch_size 个生成调用 GPT-4o-mini，导致：
    1. 每 step 延迟 8-20 秒，严重拖慢训练速度
    2. 离散的 10 档整数评分导致策略梯度信号稀疏（相邻生成的奖励差仅 0.1）
    3. 网络异常统一返回 0.0，注入伪负样本

  LocalTeachingReward 解决上述三点，同时保留完整的 API 版作为对照基线。
"""
import re
from typing import List

from omegaconf import DictConfig


# ════════════════════════════════════════════════════════════════
# 本地启发式 Teaching Reward
# ════════════════════════════════════════════════════════════════

# 结构检测：三阶段标志词
_BRUTE_FORCE_RE = re.compile(
    r"暴力|枚举|O\(n[²2]|O\(n\^2|朴素|最简单|直接遍历|穷举"
)
_OPTIMIZE_RE = re.compile(
    r"优化|更优|复杂度更低|O\(n log|O\(log|最优|更快|贪心|动态规划|dp|二分"
)
_CODE_SECTION_RE = re.compile(r"```(?:python|py)?", re.IGNORECASE)

# 教学导向词（体现因果推导、读者引导）
_REASONING_WORDS_RE = re.compile(
    r"因为|所以|因此|注意|关键|核心|观察|我们|可以发现|需要|"
    r"首先|然后|接下来|最后|总结|举例|比如|例如|换句话说"
)

# 注释行检测（Python 风格）
_COMMENT_LINE_RE = re.compile(r"^\s*#.+", re.MULTILINE)

# 代码注释密度（衡量"带注释的代码"质量）
_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _ttr(text: str) -> float:
    """
    Type-Token Ratio：唯一词 / 总词数，衡量词汇多样性。
    防止模型生成大量重复句式（如"第一步：……""第二步：……"内容完全一致）。
    """
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text)
    if not tokens:
        return 0.0
    return min(len(set(tokens)) / len(tokens), 1.0)


def _code_comment_ratio(text: str) -> float:
    """
    代码块内的注释行比例（注释行 / 非空行），上限截断为 0.5。
    初学者友好的代码应有较高注释率（建议 20-50%）。
    """
    blocks = _CODE_BLOCK_RE.findall(text)
    if not blocks:
        return 0.0
    total_lines = 0
    comment_lines = 0
    for block in blocks:
        for line in block.splitlines():
            stripped = line.strip()
            if stripped:
                total_lines += 1
                if stripped.startswith("#"):
                    comment_lines += 1
    if total_lines == 0:
        return 0.0
    return min(comment_lines / total_lines, 0.5)  # 最高贡献 0.5，归一后满分


def _score_local(problem: str, completion: str) -> float:
    """
    本地 Teaching 质量评分，返回 [0, 1] 连续分数。

    评分维度（满分 1.0）：
      +0.25  结构完整性：暴力解 → 优化解 → 代码三阶段全覆盖
             （每缺一阶段 -0.10，全有 +0.25）
      +0.25  词汇多样性：TTR > 0.5 得满分，避免模板化重复
      +0.30  代码注释率：注释行 / 非空代码行，鼓励教学友好的代码
      +0.20  教学推导词：出现 ≥5 个推导/引导词得满分

    设计理由：
    - 不依赖外部 API，毫秒级计算
    - 连续分数（非离散）提供更平滑的梯度信号
    - 每个维度有独立的物理含义，便于 ablation 分析
    """
    score = 0.0

    # ── 维度 1：三阶段结构完整性 (0.0-0.25) ─────────────────────
    has_brute = bool(_BRUTE_FORCE_RE.search(completion))
    has_opt   = bool(_OPTIMIZE_RE.search(completion))
    has_code  = bool(_CODE_SECTION_RE.search(completion))
    stage_count = sum([has_brute, has_opt, has_code])
    score += stage_count / 3.0 * 0.25

    # ── 维度 2：词汇多样性 (0.0-0.25) ─────────────────────────
    # 去掉代码块再算 TTR，避免代码关键词干扰
    text_only = _CODE_BLOCK_RE.sub("", completion)
    ttr_score = _ttr(text_only)
    score += ttr_score * 0.25

    # ── 维度 3：代码注释密度 (0.0-0.30) ───────────────────────
    comment_ratio = _code_comment_ratio(completion)
    # comment_ratio 在 [0, 0.5] 范围内，归一化到 [0, 1] 再乘权重
    score += (comment_ratio / 0.5) * 0.30

    # ── 维度 4：教学推导词 (0.0-0.20) ─────────────────────────
    reasoning_count = len(_REASONING_WORDS_RE.findall(completion))
    reasoning_score = min(reasoning_count / 5.0, 1.0)
    score += reasoning_score * 0.20

    return round(min(score, 1.0), 4)


class LocalTeachingReward:
    """
    本地启发式 Teaching Reward（无 API 调用）。

    训练默认使用此版本（reward.teaching_mode: local）。
    相比 API 版的优势：
    - 速度：~1ms/样本 vs 2-5s/样本（约 2000x 加速）
    - 信号：连续分 [0,1] vs 离散 10 档，梯度更平滑
    - 可靠：无网络依赖，无伪负样本注入风险
    """
    def __init__(self, cfg: DictConfig):
        pass  # 无需初始化 API client

    def __call__(self, prompts: List[str], completions: List[str]) -> List[float]:
        return [_score_local(p, c) for p, c in zip(prompts, completions)]


# ════════════════════════════════════════════════════════════════
# API 版 Teaching Reward（保留作为可选基线）
# ════════════════════════════════════════════════════════════════

JUDGE_SYSTEM = """你是一位 OI/ACM 算法教学质量评审员。
请根据以下标准，对给定的「算法讲解」打分（0-10 分整数）：
- 10分：完整覆盖题意分析、思路推导（含多种方案对比）、复杂度分析、代码实现四阶段，语言清晰适合初学者
- 7-9分：覆盖主要阶段，少量细节缺失
- 4-6分：仅有部分阶段，缺少关键推导过程
- 1-3分：几乎直接给代码，无讲解
- 0分：内容无关或有害

只输出一个整数分数，不要有任何其他文字。"""

JUDGE_USER_TMPL = """题目：
{problem}

模型讲解：
{completion}

评分："""


class TeachingCompletenessReward:
    """
    API 版 Teaching Reward（LLM-as-Judge，调用 GPT-4o-mini）。

    仅用于离线评估和 ablation 对比，不在 GRPO 训练循环中使用。
    通过 reward.teaching_mode: api 启用。

    已知缺陷（保留原始实现，供对比研究）：
    - 每 step 需调用 API（延迟 2-5s × batch_size × 4 gen）
    - 离散整数评分（0.0/0.1/.../1.0），梯度信号稀疏
    - 网络异常统一返回 0.0，注入伪负样本
    """
    def __init__(self, cfg: DictConfig):
        self.model = cfg.reward.judge_model
        self.temperature = cfg.reward.judge_temperature
        # 延迟导入，避免在 local 模式下强制要求 openai 包
        from openai import OpenAI
        self.client = OpenAI()

    def __call__(self, prompts: List[str], completions: List[str]) -> List[float]:
        scores = []
        for prompt, completion in zip(prompts, completions):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": JUDGE_USER_TMPL.format(
                            problem=prompt, completion=completion
                        )},
                    ],
                )
                raw = resp.choices[0].message.content.strip()
                score = int(raw) / 10.0
            except Exception:
                score = 0.0
            scores.append(score)
        return scores


# ════════════════════════════════════════════════════════════════
# 工厂函数
# ════════════════════════════════════════════════════════════════

def build_teaching_reward(cfg: DictConfig):
    """
    根据 reward.teaching_mode 配置选择 Teaching Reward 实现。

    用法（在 composite.py 或 train_grpo.py 中）：
        reward = build_teaching_reward(cfg)
        scores = reward(prompts, completions)
    """
    mode = getattr(cfg.reward, "teaching_mode", "local")
    if mode == "api":
        return TeachingCompletenessReward(cfg)
    return LocalTeachingReward(cfg)
