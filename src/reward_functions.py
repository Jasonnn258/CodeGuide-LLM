"""
GRPO 训练奖励函数

公开接口
--------
accuracy_reward(solution_code, test_cases)  -> float [0.0, 1.0]
format_reward(response, weights=None)        -> float [0.0, 1.0]
combined_reward(solution_code, test_cases, response, *, alpha=0.6) -> float [0.0, 1.0]

测试用例格式（stdin/stdout 风格，与 OJ 保持一致）
-------------------------------------------------
test_cases = [
    {"input": "3 5\n",  "output": "8\n"},
    {"input": "0 0\n",  "output": "0\n"},
]
"""

from __future__ import annotations

import ast
import re
import sys
import tempfile
import textwrap
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ════════════════════════════════════════════════════════════════
# 1. 安全扫描器
# ════════════════════════════════════════════════════════════════

# 禁止导入的模块（完整名匹配或前缀匹配）
_BLOCKED_MODULES: frozenset[str] = frozenset({
    "os", "subprocess", "socket", "shutil", "pty",
    "ctypes", "multiprocessing", "threading",
    "urllib", "http", "ftplib", "smtplib",
    "importlib", "pkgutil",
})

# 禁止调用的内置函数名
_BLOCKED_BUILTINS: frozenset[str] = frozenset({
    "open", "exec", "eval", "__import__", "compile",
    "breakpoint", "memoryview",
})

# os 模块中允许的安全属性（白名单）
_OS_SAFE_ATTRS: frozenset[str] = frozenset({
    "path", "sep", "linesep", "getcwd",
})


class _SecurityVisitor(ast.NodeVisitor):
    """
    AST 访问器，收集所有安全违规点。
    违规返回 violations 列表（非空即不安全）。
    """

    def __init__(self) -> None:
        self.violations: list[str] = []

    # ── Import / ImportFrom ──────────────────────────────────
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _BLOCKED_MODULES:
                self.violations.append(f"禁止导入模块 '{alias.name}'（第 {node.lineno} 行）")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root = node.module.split(".")[0]
            if root in _BLOCKED_MODULES:
                self.violations.append(
                    f"禁止 from-import 模块 '{node.module}'（第 {node.lineno} 行）"
                )
        self.generic_visit(node)

    # ── 危险内置调用 ─────────────────────────────────────────
    def visit_Call(self, node: ast.Call) -> None:
        func = node.func

        # 直接调用：open(...), exec(...), eval(...) 等
        if isinstance(func, ast.Name) and func.id in _BLOCKED_BUILTINS:
            self.violations.append(
                f"禁止调用内置函数 '{func.id}'（第 {node.lineno} 行）"
            )

        # 属性调用：os.system(...), os.popen(...) 等
        elif isinstance(func, ast.Attribute):
            obj = func.value
            attr = func.attr
            # os.xxx 中非白名单属性
            if isinstance(obj, ast.Name) and obj.id == "os":
                if attr not in _OS_SAFE_ATTRS:
                    self.violations.append(
                        f"禁止调用 os.{attr}（第 {node.lineno} 行）"
                    )

        self.generic_visit(node)

    # ── 危险 __dunder__ 属性访问 ─────────────────────────────
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in ("__class__", "__bases__", "__subclasses__",
                         "__globals__", "__builtins__"):
            self.violations.append(
                f"禁止访问属性 '{node.attr}'（第 {node.lineno} 行）"
            )
        self.generic_visit(node)


def _scan_security(code: str) -> list[str]:
    """
    对代码做 AST 安全扫描。
    返回违规描述列表（空列表 = 通过）。
    若代码无法解析，返回语法错误描述。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"语法错误：第 {e.lineno} 行 — {e.msg}"]
    visitor = _SecurityVisitor()
    visitor.visit(tree)
    return visitor.violations


# ════════════════════════════════════════════════════════════════
# 2. 静态代码质量估算（无测试用例时的代理信号）
# ════════════════════════════════════════════════════════════════

def _estimate_code_quality(code: str) -> float:
    """
    当题目没有 public_tests 时，用 AST 静态分析估计代码质量。

    设计动机：
    - 原实现：无测试直接 return 1.0，约 40% 的 TACO 样本受影响，
      导致模型学会输出"能过安全扫描的任意代码"即可得满分。
    - 改为：用 5 个维度打分，返回 [0, 1] 的连续分数，保留梯度信号。

    评分维度（满分 1.0）：
      +0.30  AST 解析成功（语法正确）
      +0.20  至少定义了一个函数（有逻辑结构，非单纯脚本）
      +0.20  有注释行（体现教学要求的可读性）
      +0.20  代码行数在合理范围 [10, 100]（太短=不完整，太长=乱填充）
      +0.10  无空实现（no `pass` as sole function body）

    注意：此函数只在无测试用例时调用，有测试用例时仍走沙箱执行。
    """
    score = 0.0

    # 维度 1：语法合法性（AST 解析成功）
    try:
        tree = ast.parse(code)
        score += 0.30
    except SyntaxError:
        return 0.0  # 语法错误直接 0 分，无需继续

    # 维度 2：至少有一个函数定义
    has_func = any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   for node in ast.walk(tree))
    if has_func:
        score += 0.20

    # 维度 3：有注释行（# 开头的行或 docstring）
    lines = code.splitlines()
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    has_docstring = any(
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for node in ast.walk(tree)
    )
    if comment_lines >= 2 or has_docstring:
        score += 0.20

    # 维度 4：代码行数合理（10~100 行）
    non_empty_lines = sum(1 for l in lines if l.strip())
    if 10 <= non_empty_lines <= 100:
        score += 0.20

    # 维度 5：无空 pass 实现（函数体不能只有 pass）
    has_pass_only = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                has_pass_only = True
                break
    if not has_pass_only:
        score += 0.10

    return round(min(score, 1.0), 4)


# ════════════════════════════════════════════════════════════════
# 3. accuracy_reward
# ════════════════════════════════════════════════════════════════

_SENTINEL = "__CODEGUIDE_PASS__"   # 用于标记单条测试通过的哨兵串

def _build_exec_harness(code: str, test_cases: list[dict]) -> str:
    """
    构造一次性执行脚本。
    每个 test case 重置 stdin / stdout，exec 用户代码，比对输出。
    结果行格式：PASS 或 FAIL:<reason>
    """
    return textwrap.dedent(f"""\
import sys, io, traceback

_code = {repr(code)}
_cases = {repr(test_cases)}
_sentinel = {repr(_SENTINEL)}

for _tc in _cases:
    _inp = _tc.get("input", "")
    _exp = _tc.get("output", "").rstrip("\\n")
    _stdin_bak, _stdout_bak = sys.stdin, sys.stdout
    sys.stdin  = io.StringIO(_inp)
    sys.stdout = io.StringIO()
    try:
        exec(compile(_code, "<solution>", "exec"), {{}})
        _got = sys.stdout.getvalue().rstrip("\\n")
        _ok  = (_got == _exp)
    except Exception:
        _ok = False
    finally:
        sys.stdin  = _stdin_bak
        sys.stdout = _stdout_bak

    print(_sentinel if _ok else "FAIL")
""")


def accuracy_reward(
    solution_code: str,
    test_cases: list[dict],
    *,
    timeout: float = 5.0,
) -> float:
    """
    在沙箱子进程中执行 solution_code，按 test_cases 计算通过率。

    Args:
        solution_code: 待执行的 Python 代码字符串（已提取，非含 ``` 的全文）
        test_cases:    [{"input": str, "output": str}, ...]，stdin/stdout 风格
        timeout:       单次执行超时秒数（默认 5s）

    Returns:
        float in [0.0, 1.0]：通过的测试用例比例
        - 0.0：安全扫描失败 / 全部超时 / 全部运行时错误
        - 1.0：全部通过
    """
    if not solution_code or not solution_code.strip():
        return 0.0

    # ── Step 1: AST 安全扫描 ──────────────────────────────────
    violations = _scan_security(solution_code)
    if violations:
        return 0.0

    # ── Step 2: 无测试用例时用静态分析估分 ──────────────────────
    # 原实现：无测试直接给 1.0，导致 reward hacking（模型只需输出
    # 能通过安全扫描的任意代码即可得满分，约 40% TACO 题受影响）。
    # 改为：使用 AST 静态质量分析作为代理信号，强制保留梯度。
    if not test_cases:
        return _estimate_code_quality(solution_code)

    # ── Step 3: 沙箱执行 ─────────────────────────────────────
    harness = _build_exec_harness(solution_code, test_cases)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(harness)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            return 0.0
        passed = sum(1 for l in lines if l == _SENTINEL)
        return passed / len(test_cases)

    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════
# 3. format_reward
# ════════════════════════════════════════════════════════════════

@dataclass
class FormatWeights:
    """
    format_reward 各维度权重（必须加和为 1.0）。

    Attributes:
        step_headings:        分步骤标题（第X步 / Step X / Markdown 标题等）
        code_block:           包含 ```python … ``` 代码块
        chinese_explanation:  代码块外存在中文解释文字（≥ 20 汉字）
        length_sanity:        回复长度合理（100 ~ 3000 字符）
    """
    step_headings:        float = 0.35
    code_block:           float = 0.35
    chinese_explanation:  float = 0.20
    length_sanity:        float = 0.10

    def __post_init__(self) -> None:
        total = (
            self.step_headings + self.code_block
            + self.chinese_explanation + self.length_sanity
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"FormatWeights 各项之和必须为 1.0，当前为 {total:.4f}")


# 步骤标题正则（至少命中一个）
_STEP_RE: list[re.Pattern] = [
    re.compile(r"第\s*[一二三四五六七八九十百\d]+\s*步"),   # 第一步 / 第1步
    re.compile(r"\*{1,2}Step\s*\d+", re.IGNORECASE),        # **Step 1** / Step 1
    re.compile(r"^#{1,3}\s+.{2,}", re.MULTILINE),           # ## 标题
    re.compile(r"^\d+\.\s+\*{1,2}\S", re.MULTILINE),        # 1. **xxx
    re.compile(r"【第?\s*\d+\s*步】"),                        # 【第1步】
]

# 代码块正则
_CODE_BLOCK_RE = re.compile(r"```(?:python3?|py)?\s*\n.+?```", re.DOTALL | re.IGNORECASE)

# 汉字字符范围
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

# 合理长度区间
_LEN_MIN, _LEN_MAX = 100, 3000


def _strip_code_blocks(text: str) -> str:
    """从文本中移除所有代码块，保留解释性文字。"""
    return _CODE_BLOCK_RE.sub("", text)


def format_reward(
    response: str,
    weights: Optional[FormatWeights] = None,
) -> float:
    """
    检查模型回复是否符合教学格式，返回加权综合得分。

    Args:
        response: 模型的完整回复文本
        weights:  各维度权重（None 时使用默认值）

    Returns:
        float in [0.0, 1.0]
    """
    if weights is None:
        weights = FormatWeights()

    score = 0.0

    # ── 维度 1：分步骤标题 ────────────────────────────────────
    if any(p.search(response) for p in _STEP_RE):
        score += weights.step_headings

    # ── 维度 2：代码块 ────────────────────────────────────────
    if _CODE_BLOCK_RE.search(response):
        score += weights.code_block

    # ── 维度 3：代码块外的中文解释 ────────────────────────────
    explanation_text = _strip_code_blocks(response)
    chinese_chars = len(_CHINESE_RE.findall(explanation_text))
    if chinese_chars >= 20:
        score += weights.chinese_explanation

    # ── 维度 4：长度合理性 ────────────────────────────────────
    if _LEN_MIN <= len(response) <= _LEN_MAX:
        score += weights.length_sanity

    return round(min(score, 1.0), 4)


# ════════════════════════════════════════════════════════════════
# 4. combined_reward
# ════════════════════════════════════════════════════════════════

def combined_reward(
    solution_code: str,
    test_cases: list[dict],
    response: str,
    *,
    alpha: float = 0.6,
    timeout: float = 5.0,
    format_weights: Optional[FormatWeights] = None,
) -> float:
    """
    组合奖励 = alpha * accuracy_reward + (1-alpha) * format_reward

    默认 alpha=0.6，即：
        combined = 0.6 × accuracy + 0.4 × format

    Args:
        solution_code:   已提取的 Python 代码（可传空串，此时 accuracy=0）
        test_cases:      stdin/stdout 格式测试用例列表
        response:        模型完整回复（含讲解 + 代码块）
        alpha:           accuracy 的权重，取值 [0, 1]
        timeout:         accuracy 子进程超时秒数
        format_weights:  format_reward 各维度权重

    Returns:
        float in [0.0, 1.0]
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha 必须在 [0, 1] 范围内，当前为 {alpha}")

    acc = accuracy_reward(solution_code, test_cases, timeout=timeout)
    fmt = format_reward(response, weights=format_weights)
    return round(alpha * acc + (1.0 - alpha) * fmt, 4)
