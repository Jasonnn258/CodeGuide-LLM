#!/usr/bin/env python3
"""
CodeGuide-LLM 命令行交互式推理 Demo

功能：
  - 4bit NF4 量化加载（单卡 ~5-6 GB VRAM）
  - TextIteratorStreamer 流式输出，字符逐步显示
  - 多轮对话（模型记住上下文）
  - 内置命令：/clear（重置对话）、/exit（退出）、/help（帮助）

用法：
    python scripts/inference_demo.py
    python scripts/inference_demo.py --model models/codeguide_llm_merged
    python scripts/inference_demo.py --model models/codeguide_llm_merged --no_4bit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from threading import Thread

# 确保项目根在 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── 系统提示（与训练时保持一致）────────────────────────────────
SYSTEM_PROMPT = (
    "你是 CodeGuide，一位专为 OI/ACM 初学者设计的算法教学助手。\n"
    "当用户提出一道算法题时，你会：\n"
    "1. 先用简单的语言理解题意，举一个小例子走一遍；\n"
    "2. 从暴力解出发，逐步推导出最优解；\n"
    "3. 每一步都解释"为什么这样想"，而不只是"怎么做"；\n"
    "4. 最后给出带详细注释的完整 Python 代码。\n"
    "讲解应通俗易懂，适合刚开始学习算法竞赛的初学者。"
)

# ── 终端颜色（自动检测是否支持）────────────────────────────────
_USE_COLOR = sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

CYAN   = lambda t: _c(t, "36")
GREEN  = lambda t: _c(t, "32")
YELLOW = lambda t: _c(t, "33")
BOLD   = lambda t: _c(t, "1")
DIM    = lambda t: _c(t, "2")


# ── 模型加载 ─────────────────────────────────────────────────────

def load_model(model_path: str, use_4bit: bool = True):
    """
    加载模型与 tokenizer。

    use_4bit=True（默认）：NF4 量化，约 5-6 GB VRAM，推荐
    use_4bit=False：bf16 全精度，约 14 GB VRAM
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    print(DIM(f"  加载 tokenizer：{model_path}"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_quant_type       = "nf4",
            bnb_4bit_compute_dtype    = torch.bfloat16,
            bnb_4bit_use_double_quant = True,
        )
        print(DIM("  量化策略：NF4 4bit（双重量化）"))
    else:
        print(DIM("  量化策略：bf16 全精度"))

    print(DIM("  加载模型权重…"))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config = quant_cfg,
        torch_dtype         = torch.bfloat16 if not use_4bit else None,
        device_map          = "auto",
        trust_remote_code   = True,
    )
    model.eval()

    vram = _vram_used_gb()
    print(DIM(f"  模型就绪  VRAM 使用：{vram}"))
    return model, tokenizer


def _vram_used_gb() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            mb = torch.cuda.memory_allocated() / 1024 ** 2
            return f"{mb / 1024:.1f} GB"
    except Exception:
        pass
    return "未知"


# ── 流式生成 ─────────────────────────────────────────────────────

def stream_generate(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    使用 TextIteratorStreamer 流式生成，边生成边打印。
    返回完整生成文本（用于追加到对话历史）。
    """
    import torch
    from transformers import TextIteratorStreamer

    # apply_chat_template → 单条 token 序列
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        text,
        return_tensors  = "pt",
        truncation      = True,
        max_length      = 3072,
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt         = True,
        skip_special_tokens = True,
    )

    gen_kwargs = dict(
        **inputs,
        streamer         = streamer,
        max_new_tokens   = max_new_tokens,
        temperature      = temperature,
        top_p            = top_p,
        do_sample        = temperature > 0,
        repetition_penalty = 1.1,
        pad_token_id     = tokenizer.pad_token_id,
    )

    # 在后台线程中执行生成，主线程消费 streamer
    thread = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
    thread.start()

    print()
    print(CYAN("CodeGuide："))
    full_text = ""
    for chunk in streamer:
        print(chunk, end="", flush=True)
        full_text += chunk
    print("\n")           # 生成结束后换行

    thread.join()
    return full_text


# ── 帮助文本 ─────────────────────────────────────────────────────

_HELP_TEXT = """\
可用命令：
  /clear    重置对话历史，开始新话题
  /exit     退出程序
  /help     显示此帮助

直接输入算法题目描述，模型将逐步讲解解题思路。

示例题目：
  给定一个整数数组 nums 和一个目标值 target，
  在数组中找出和为 target 的两个整数的下标。
"""

_BANNER = r"""
  ____          _      ____       _     _        _     _     __  __
 / ___|___   __| | ___|  _ \ _  _(_) __| | ___  | |   | |   |  \/  |
| |   / _ \ / _` |/ _ \ | | | | | | |/ _` |/ _ \ | |   | |   | |\/| |
| |__| (_) | (_| |  __/ |_| | |_| | | (_| |  __/ | |___| |___| |  | |
 \____\___/ \__,_|\___|____/ \__,_|_|\__,_|\___| |_____|_____|_|  |_|

  OI/ACM 算法教学助手  ·  步进式讲解  ·  输入 /help 查看命令
"""


# ── 主循环 ───────────────────────────────────────────────────────

def chat_loop(model, tokenizer, args: argparse.Namespace) -> None:
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(GREEN(_BANNER))
    print(DIM(f"模型路径：{args.model}"))
    print(DIM(f"量化模式：{'4bit NF4' if args.use_4bit else 'bf16 全精度'}"))
    print(DIM(f"max_new_tokens={args.max_new_tokens}  temperature={args.temperature}  top_p={args.top_p}"))
    print()

    while True:
        try:
            user_input = input(BOLD("你：")).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n" + DIM("（Ctrl-C 退出）"))
            break

        if not user_input:
            continue

        # ── 内置命令 ──────────────────────────────────────────
        if user_input.lower() == "/exit":
            print(DIM("再见！"))
            break
        if user_input.lower() == "/clear":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print(YELLOW("  对话历史已清空，开始新话题\n"))
            continue
        if user_input.lower() == "/help":
            print(YELLOW(_HELP_TEXT))
            continue

        # ── 正常对话轮次 ──────────────────────────────────────
        history.append({"role": "user", "content": user_input})

        response = stream_generate(
            model, tokenizer, history,
            max_new_tokens = args.max_new_tokens,
            temperature    = args.temperature,
            top_p          = args.top_p,
        )
        history.append({"role": "assistant", "content": response})

        # 防止历史过长（保留 system + 最近 6 轮）
        if len(history) > 14:
            history = [history[0]] + history[-12:]


# ── 入口 ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CodeGuide-LLM 命令行推理 Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="models/codeguide_llm_merged",
        help="模型路径（merged 全量模型）",
    )
    parser.add_argument(
        "--no_4bit", dest="use_4bit", action="store_false",
        help="禁用 4bit 量化（需要约 14 GB VRAM）",
    )
    parser.add_argument("--max_new_tokens", type=int,   default=1024)
    parser.add_argument("--temperature",    type=float, default=0.7)
    parser.add_argument("--top_p",          type=float, default=0.9)
    parser.set_defaults(use_4bit=True)
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"错误：模型路径不存在：{args.model}", file=sys.stderr)
        print("请先完成训练并合并模型：python scripts/train_grpo.py", file=sys.stderr)
        sys.exit(1)

    print(DIM("\n正在加载模型，请稍候…"))
    model, tokenizer = load_model(args.model, use_4bit=args.use_4bit)
    chat_loop(model, tokenizer, args)


if __name__ == "__main__":
    main()
