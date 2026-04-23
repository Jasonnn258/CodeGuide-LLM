"""
CodeCorrectnessReward: 提取模型输出中的代码块，在沙箱中运行测试用例
返回 pass 率作为奖励 [0.0, 1.0]

改进（v2）：
- 无测试用例时改为调用 _estimate_code_quality 做静态分析
  而非固定返回 0.5，保持与 reward_functions.accuracy_reward 一致
"""
import re
import subprocess
import tempfile
import textwrap
from typing import List, Optional

from omegaconf import DictConfig

from src.reward_functions import _estimate_code_quality

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> Optional[str]:
    """提取最后一个代码块（通常是最终实现）"""
    matches = _CODE_BLOCK_RE.findall(text)
    return matches[-1].strip() if matches else None


def _run_tests(code: str, test_cases: List[dict], timeout: int = 5) -> float:
    """在子进程中运行测试用例，返回通过率"""
    if not test_cases:
        # 原实现：固定返回 0.5（中性分），无法区分代码质量好坏。
        # 改为：静态分析估分，保留梯度信号。
        return _estimate_code_quality(code)

    passed = 0
    for tc in test_cases:
        harness = textwrap.dedent(f"""
{code}

_input = {tc['input']!r}
_expected = {tc['output']!r}
try:
    _result = solution(*_input) if isinstance(_input, (list, tuple)) else solution(_input)
    assert _result == _expected, f"got {{_result}}, expected {{_expected}}"
    print("PASS")
except Exception as e:
    print(f"FAIL: {{e}}")
""")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(harness)
            fname = f.name
        try:
            result = subprocess.run(
                ["python", fname],
                capture_output=True, text=True, timeout=timeout
            )
            if "PASS" in result.stdout:
                passed += 1
        except subprocess.TimeoutExpired:
            pass
    return passed / len(test_cases)


class CodeCorrectnessReward:
    def __init__(self, cfg: DictConfig):
        self.timeout = 5

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        test_cases_batch: Optional[List[List[dict]]] = None,
    ) -> List[float]:
        scores = []
        for i, completion in enumerate(completions):
            code = _extract_code(completion)
            if code is None:
                scores.append(0.0)
                continue
            tc = test_cases_batch[i] if test_cases_batch else []
            scores.append(_run_tests(code, tc, self.timeout))
        return scores
