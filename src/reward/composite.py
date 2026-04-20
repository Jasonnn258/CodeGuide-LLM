"""
复合奖励函数：加权合并 teaching / correctness / format 三路奖励信号
"""
from dataclasses import dataclass
from typing import List

from omegaconf import DictConfig


@dataclass
class RewardOutput:
    total: float
    teaching: float
    correctness: float
    format: float


class CompositeReward:
    def __init__(self, cfg: DictConfig):
        self.w_teaching = cfg.reward.teaching_completeness
        self.w_correctness = cfg.reward.code_correctness
        self.w_format = cfg.reward.format_compliance

        from .teaching import TeachingCompletenessReward
        from .correctness import CodeCorrectnessReward
        from .format import FormatComplianceReward

        self.teaching_fn = TeachingCompletenessReward(cfg)
        self.correctness_fn = CodeCorrectnessReward(cfg)
        self.format_fn = FormatComplianceReward(cfg)

    def __call__(self, prompts: List[str], completions: List[str]) -> List[RewardOutput]:
        teaching_scores = self.teaching_fn(prompts, completions)
        correctness_scores = self.correctness_fn(prompts, completions)
        format_scores = self.format_fn(prompts, completions)

        results = []
        for t, c, f in zip(teaching_scores, correctness_scores, format_scores):
            total = self.w_teaching * t + self.w_correctness * c + self.w_format * f
            results.append(RewardOutput(total=total, teaching=t, correctness=c, format=f))
        return results
