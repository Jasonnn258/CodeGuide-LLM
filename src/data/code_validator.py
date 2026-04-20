"""
代码提取与校验

功能：
1. extract_code()       — 从 LLM 输出中抽取最后一个 ```python ... ``` 块
2. validate_syntax()    — ast.parse() 语法检查（快，无副作用）
3. validate_execution() — 沙箱子进程执行 + 测试用例比对（可选）
4. validate()           — 组合校验，返回 ValidationResult
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# 匹配 ```python ... ``` 或 ``` ... ```（宽松）
_CODE_FENCE_RE = re.compile(
    r"```(?:python3?|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# 执行超时（秒）
_EXEC_TIMEOUT = 8


# ── 数据结构 ─────────────────────────────────────────────────

@dataclass
class ValidationResult:
    ok: bool
    code: Optional[str]      # 提取到的代码，None 表示未找到代码块
    error: Optional[str]     # 失败原因（供调试）
    pass_rate: float = 0.0   # 测试用例通过率（0.0 ~ 1.0）


# ── 代码提取 ─────────────────────────────────────────────────

def extract_code(text: str) -> Optional[str]:
    """
    提取 LLM 输出中最后一个代码块。
    优先找 ```python```，其次找任意 ``` 块。
    返回代码字符串（已去首尾空行），找不到返回 None。
    """
    matches = _CODE_FENCE_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


# ── 语法校验 ─────────────────────────────────────────────────

def validate_syntax(code: str) -> tuple[bool, Optional[str]]:
    """
    用 ast.parse 检查 Python 语法。
    返回 (ok, error_msg)。
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"


# ── 沙箱执行校验 ─────────────────────────────────────────────

def _build_harness(code: str, test_cases: List[dict]) -> str:
    """
    构造测试驱动脚本。
    test_cases 格式：[{"input": "1 2\n", "output": "3\n"}, ...]
    代码必须从 stdin 读取输入、向 stdout 输出结果。
    """
    cases_repr = repr(test_cases)
    harness = textwrap.dedent(f"""\
import sys, io

_user_code = {repr(code)}
_test_cases = {cases_repr}

passed = 0
for tc in _test_cases:
    _in  = tc.get("input", "")
    _exp = tc.get("output", "").strip()
    sys.stdin  = io.StringIO(_in)
    sys.stdout = io.StringIO()
    try:
        exec(compile(_user_code, "<solution>", "exec"), {{}})
        got = sys.stdout.getvalue().strip()
        if got == _exp:
            passed += 1
    except Exception as exc:
        pass  # 运行时错误算失败

sys.stdout = sys.__stdout__
print(f"{{passed}}/{{len(_test_cases)}}")
""")
    return harness


def validate_execution(
    code: str,
    test_cases: List[dict],
    timeout: int = _EXEC_TIMEOUT,
) -> tuple[bool, float, Optional[str]]:
    """
    在独立子进程中运行代码，比对测试用例输出。

    Args:
        code:       待测代码字符串
        test_cases: [{"input": str, "output": str}, ...]
        timeout:    子进程超时秒数

    Returns:
        (ok, pass_rate, error_msg)
        ok = pass_rate >= 0.5（至少通过一半测试）
    """
    if not test_cases:
        # 无测试用例时只做语法检查
        ok, err = validate_syntax(code)
        return ok, (1.0 if ok else 0.0), err

    harness = _build_harness(code, test_cases)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(harness)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout.strip()
        # 解析 "passed/total" 格式
        if "/" in out:
            num, denom = out.split("/")
            pass_rate = int(num) / max(int(denom), 1)
        else:
            pass_rate = 0.0
        ok = pass_rate >= 0.5
        err = result.stderr.strip() or None
        return ok, pass_rate, err

    except subprocess.TimeoutExpired:
        return False, 0.0, f"执行超时（>{timeout}s）"
    except Exception as e:
        return False, 0.0, str(e)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── 组合校验入口 ─────────────────────────────────────────────

def validate(
    llm_output: str,
    test_cases: Optional[List[dict]] = None,
    run_code: bool = False,
) -> ValidationResult:
    """
    完整校验流程：提取 → 语法 → （可选）执行。

    Args:
        llm_output:  LLM 的完整输出文本
        test_cases:  用于执行校验的测试用例，为空列表或 None 时跳过执行
        run_code:    是否启用沙箱执行（False 时只做语法检查）

    Returns:
        ValidationResult
    """
    # Step 1: 提取代码块
    code = extract_code(llm_output)
    if code is None:
        return ValidationResult(ok=False, code=None, error="未找到代码块")

    # Step 2: 语法检查
    syntax_ok, syntax_err = validate_syntax(code)
    if not syntax_ok:
        return ValidationResult(ok=False, code=code, error=syntax_err)

    # Step 3: 执行校验（可选）
    if run_code and test_cases:
        exec_ok, pass_rate, exec_err = validate_execution(code, test_cases)
        return ValidationResult(
            ok=exec_ok,
            code=code,
            error=exec_err,
            pass_rate=pass_rate,
        )

    return ValidationResult(ok=True, code=code, error=None, pass_rate=1.0)
