"""
tests/test_rewards.py

三个端到端场景验证奖励函数的正确性：

场景 A — 正确代码 + 规范教学格式
    accuracy_reward  → 1.0
    format_reward    → 1.0
    combined_reward  → 1.0

场景 B — 错误代码 + 规范教学格式
    accuracy_reward  → 0.0
    format_reward    → 1.0
    combined_reward  → 0.4  (= 0.6×0 + 0.4×1)

场景 C — 正确代码 + 无教学格式（直接给代码）
    accuracy_reward  → 1.0
    format_reward    → 低分（无步骤、无中文解释）
    combined_reward  → 介于 0.6 和 1.0 之间

另附：
  - 安全扫描单元测试（os.system / open / subprocess 等）
  - FormatWeights 权重校验异常测试
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 确保项目根目录在 sys.path（无论从哪里运行 pytest）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reward_functions import (
    FormatWeights,
    _scan_security,
    accuracy_reward,
    combined_reward,
    format_reward,
)


# ════════════════════════════════════════════════════════════════
# 公共测试数据
# ════════════════════════════════════════════════════════════════

# ── 两数之和：a + b ───────────────────────────────────────────
_ADD_CASES = [
    {"input": "3 5\n",    "output": "8"},
    {"input": "1 2\n",    "output": "3"},    # 改掉 0 0→0（减法也成立的边界）
    {"input": "-1 7\n",   "output": "6"},
    {"input": "100 200\n", "output": "300"},
]

# 正确实现（stdin 读入，stdout 输出）
_ADD_CODE_CORRECT = """\
a, b = map(int, input().split())
print(a + b)
"""

# 错误实现（输出 a - b，结果不对）
_ADD_CODE_WRONG = """\
a, b = map(int, input().split())
print(a - b)
"""

# 规范教学格式回复（含步骤标题 + 代码块 + 充分中文解释）
_GOOD_FORMAT_RESPONSE = """\
**第一步：理解题意**

题目要求读入两个整数 a 和 b，输出它们的和。
输入格式：一行，两个整数用空格分隔。
输出格式：一行，输出两数之和。

举例：输入 "3 5"，则输出 "8"。

**第二步：分析暴力解法**

由于只有两个数，直接相加即可，时间复杂度 O(1)，无需优化。

**第三步：设计最优解**

这道题本身就是最优解，没有更复杂的算法需求。
核心思路：读取 → 相加 → 输出，一行代码即可完成。

**第四步：完整 Python 实现**

```python
a, b = map(int, input().split())
print(a + b)
```

**复杂度分析**
- 时间复杂度：O(1)，只做一次加法运算。
- 空间复杂度：O(1)，只用了两个变量。
"""

# 无格式回复（直接给代码，无步骤标题，无中文解释）
_BARE_CODE_RESPONSE = """\
```python
a, b = map(int, input().split())
print(a + b)
```
"""


# ════════════════════════════════════════════════════════════════
# 场景 A：正确代码 + 规范格式
# ════════════════════════════════════════════════════════════════

class TestScenarioA:
    """正确解法 + 完整教学格式：三个奖励函数均应给高分。"""

    def test_accuracy_is_full(self):
        score = accuracy_reward(_ADD_CODE_CORRECT, _ADD_CASES)
        assert score == pytest.approx(1.0), (
            f"正确代码通过所有测试用例，期望 1.0，实际 {score}"
        )

    def test_format_is_full(self):
        score = format_reward(_GOOD_FORMAT_RESPONSE)
        assert score == pytest.approx(1.0), (
            f"规范格式应得满分，实际 {score}"
        )

    def test_combined_is_full(self):
        score = combined_reward(_ADD_CODE_CORRECT, _ADD_CASES, _GOOD_FORMAT_RESPONSE)
        assert score == pytest.approx(1.0), (
            f"双满分组合应得 1.0，实际 {score}"
        )


# ════════════════════════════════════════════════════════════════
# 场景 B：错误代码 + 规范格式
# ════════════════════════════════════════════════════════════════

class TestScenarioB:
    """错误解法（减法而非加法）+ 完整教学格式：accuracy=0，format=1，combined=0.4。"""

    def test_accuracy_is_zero(self):
        score = accuracy_reward(_ADD_CODE_WRONG, _ADD_CASES)
        assert score == pytest.approx(0.0), (
            f"错误代码应全部失败，期望 0.0，实际 {score}"
        )

    def test_format_still_full(self):
        """格式得分与代码正确性无关。"""
        score = format_reward(_GOOD_FORMAT_RESPONSE)
        assert score == pytest.approx(1.0)

    def test_combined_equals_format_weight(self):
        """combined = 0.6×0 + 0.4×1 = 0.4"""
        score = combined_reward(_ADD_CODE_WRONG, _ADD_CASES, _GOOD_FORMAT_RESPONSE)
        assert score == pytest.approx(0.4, abs=1e-4), (
            f"期望 0.4，实际 {score}"
        )

    def test_partial_pass_rate(self):
        """
        构造只有一半测试用例能通过的情况，验证 pass_rate 按比例返回。
        案例：a==b 时 a-b==0 与 a+b==0 相同，其余不同。
        """
        mixed_cases = [
            {"input": "0 0\n",  "output": "0"},   # PASS：0-0 == 0+0
            {"input": "3 5\n",  "output": "8"},   # FAIL：3-5 != 3+5
        ]
        score = accuracy_reward(_ADD_CODE_WRONG, mixed_cases)
        assert score == pytest.approx(0.5, abs=1e-4)


# ════════════════════════════════════════════════════════════════
# 场景 C：正确代码 + 无教学格式
# ════════════════════════════════════════════════════════════════

class TestScenarioC:
    """正确解法 + 仅有代码块无步骤讲解：accuracy=1，format 较低。"""

    def test_accuracy_is_full(self):
        score = accuracy_reward(_ADD_CODE_CORRECT, _ADD_CASES)
        assert score == pytest.approx(1.0)

    def test_format_is_low(self):
        """无步骤标题且无中文解释，得分应低于 0.5。"""
        score = format_reward(_BARE_CODE_RESPONSE)
        # 只有代码块（+0.35），其余不满足，最多 0.35 + 0.1(长度) = 0.45
        assert score < 0.5, f"无教学格式应低分，实际 {score}"

    def test_combined_between_accuracy_and_one(self):
        """combined = 0.6×1 + 0.4×low，应在 [0.6, 1.0) 区间。"""
        score = combined_reward(_ADD_CODE_CORRECT, _ADD_CASES, _BARE_CODE_RESPONSE)
        assert 0.6 <= score < 1.0, (
            f"组合分数应在 [0.6, 1.0)，实际 {score}"
        )

    def test_custom_alpha(self):
        """alpha=1.0 时 combined 完全等于 accuracy_reward。"""
        score = combined_reward(
            _ADD_CODE_CORRECT, _ADD_CASES, _BARE_CODE_RESPONSE, alpha=1.0
        )
        assert score == pytest.approx(1.0)


# ════════════════════════════════════════════════════════════════
# 安全扫描单元测试
# ════════════════════════════════════════════════════════════════

class TestSecurityScanner:
    """验证 AST 安全扫描器能正确识别危险代码。"""

    def test_os_system_blocked(self):
        code = "import os\nos.system('rm -rf /')"
        violations = _scan_security(code)
        assert any("os" in v for v in violations), violations

    def test_subprocess_blocked(self):
        code = "import subprocess\nsubprocess.run(['ls'])"
        violations = _scan_security(code)
        assert violations, "subprocess 应被阻止"

    def test_open_blocked(self):
        code = "open('/etc/passwd').read()"
        violations = _scan_security(code)
        assert any("open" in v for v in violations), violations

    def test_socket_import_blocked(self):
        code = "import socket\ns = socket.socket()"
        violations = _scan_security(code)
        assert violations, "socket 应被阻止"

    def test_dunder_globals_blocked(self):
        code = "().__class__.__bases__[0].__subclasses__()"
        violations = _scan_security(code)
        assert violations, "dunder 属性访问应被阻止"

    def test_safe_code_passes(self):
        """普通算法竞赛代码不应触发安全扫描。"""
        code = """\
import sys
from collections import defaultdict

def solve():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    graph = defaultdict(list)
    for _ in range(n - 1):
        u, v = map(int, sys.stdin.readline().split())
        graph[u].append(v)
        graph[v].append(u)
    print(sum(a))

solve()
"""
        violations = _scan_security(code)
        assert not violations, f"安全代码不应有违规：{violations}"

    def test_accuracy_returns_zero_on_unsafe_code(self):
        """不安全代码传入 accuracy_reward，应直接返回 0.0 而不执行。"""
        unsafe_code = "import os\nos.system('echo hacked')"
        score = accuracy_reward(unsafe_code, [{"input": "", "output": ""}])
        assert score == 0.0


# ════════════════════════════════════════════════════════════════
# FormatWeights 配置测试
# ════════════════════════════════════════════════════════════════

class TestFormatWeights:
    """验证 FormatWeights 权重配置与校验逻辑。"""

    def test_default_weights_sum_to_one(self):
        w = FormatWeights()
        total = w.step_headings + w.code_block + w.chinese_explanation + w.length_sanity
        assert abs(total - 1.0) < 1e-9

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError, match="1.0"):
            FormatWeights(step_headings=0.5, code_block=0.5,
                          chinese_explanation=0.5, length_sanity=0.5)

    def test_custom_weights_applied(self):
        """将 code_block 权重设为 1.0，其余全为 0，则只要有代码块就满分。"""
        w = FormatWeights(step_headings=0.0, code_block=1.0,
                          chinese_explanation=0.0, length_sanity=0.0)
        score = format_reward(_BARE_CODE_RESPONSE, weights=w)
        assert score == pytest.approx(1.0)

    def test_all_zero_except_step(self):
        """只看步骤标题时，无格式回复应得 0。"""
        w = FormatWeights(step_headings=1.0, code_block=0.0,
                          chinese_explanation=0.0, length_sanity=0.0)
        score = format_reward(_BARE_CODE_RESPONSE, weights=w)
        assert score == pytest.approx(0.0)


# ════════════════════════════════════════════════════════════════
# 边界条件测试
# ════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """各种边界情况：空输入、超时、无测试用例等。"""

    def test_empty_code_returns_zero(self):
        assert accuracy_reward("", _ADD_CASES) == 0.0

    def test_syntax_error_returns_zero(self):
        assert accuracy_reward("def f(:\n    pass", _ADD_CASES) == 0.0

    def test_no_test_cases_syntax_ok(self):
        """无测试用例时，语法合法的代码给 1.0。"""
        assert accuracy_reward(_ADD_CODE_CORRECT, []) == pytest.approx(1.0)

    def test_timeout_returns_zero(self):
        """死循环代码应超时并返回 0.0。"""
        infinite_loop = "while True: pass"
        score = accuracy_reward(infinite_loop, _ADD_CASES, timeout=1.0)
        assert score == 0.0

    def test_runtime_error_returns_zero(self):
        """运行时抛出异常的代码（除零）应返回 0.0。"""
        crash_code = "print(1 / 0)"
        score = accuracy_reward(crash_code, [{"input": "", "output": "0"}])
        assert score == 0.0

    def test_empty_response_format(self):
        assert format_reward("") == pytest.approx(0.0)

    def test_combined_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="alpha"):
            combined_reward("", [], "", alpha=1.5)
