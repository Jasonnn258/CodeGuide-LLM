"""
TeachingCompletenessReward: 用 LLM-as-Judge 评估讲解步骤完整性
- 检查是否覆盖「题意分析 → 思路推导 → 复杂度分析 → 代码实现」四阶段
- 默认使用 gpt-4o-mini，也可换为本地 Qwen2.5-72B
"""
from typing import List

from omegaconf import DictConfig
from openai import OpenAI

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
    def __init__(self, cfg: DictConfig):
        self.model = cfg.reward.judge_model
        self.temperature = cfg.reward.judge_temperature
        self.client = OpenAI()  # 从环境变量 OPENAI_API_KEY 读取

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
                score = int(raw) / 10.0  # 归一化到 [0, 1]
            except Exception:
                score = 0.0
            scores.append(score)
        return scores
