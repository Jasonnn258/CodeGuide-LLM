#!/usr/bin/env python3
"""
CodeGuide-LLM Gradio Web UI Demo

布局：
  ┌────────────────────┬─────────────────────────────────────┐
  │  📝 题目描述（左侧）  │  💡 教学步骤（右侧，流式输出）           │
  │  [Textbox]         │  [Markdown 实时更新]                  │
  │  [提交] [清空]       │                                      │
  │  示例题目快捷按钮     │  对话历史（Chatbot 组件）               │
  └────────────────────┴─────────────────────────────────────┘

流式推理：TextIteratorStreamer + Thread，每产生一个 token 立即更新界面。

用法：
    python scripts/gradio_demo.py
    python scripts/gradio_demo.py --model models/codeguide_llm_merged
    python scripts/gradio_demo.py --share          # 生成公开分享链接
    python scripts/gradio_demo.py --no_4bit        # bf16 全精度（需 14 GB VRAM）
    python scripts/gradio_demo.py --port 8080
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from threading import Thread
from typing import Generator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── 系统提示（与训练保持一致）────────────────────────────────────
SYSTEM_PROMPT = (
    "你是 CodeGuide，一位专为 OI/ACM 初学者设计的算法教学助手。\n"
    "当用户提出一道算法题时，你会：\n"
    "1. 先用简单的语言理解题意，举一个小例子走一遍；\n"
    "2. 从暴力解出发，逐步推导出最优解；\n"
    "3. 每一步都解释"为什么这样想"，而不只是"怎么做"；\n"
    "4. 最后给出带详细注释的完整 Python 代码。\n"
    "讲解应通俗易懂，适合刚开始学习算法竞赛的初学者。"
)

# ── 示例题目 ─────────────────────────────────────────────────────
EXAMPLE_PROBLEMS = [
    (
        "两数之和",
        "给定一个整数数组 nums 和一个目标值 target，请在数组中找出和为 target 的两个整数，"
        "并返回它们的下标。\n\n"
        "示例：nums = [2, 7, 11, 15]，target = 9，输出 [0, 1]（因为 nums[0]+nums[1]=9）",
    ),
    (
        "最长公共子序列",
        "给两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。\n\n"
        "示例：text1='abcde', text2='ace'，最长公共子序列为 'ace'，长度为 3",
    ),
    (
        "接雨水",
        "给定 n 个非负整数表示每个宽度为 1 的柱子高度图，计算按此排列的柱子，下雨之后能接多少雨水。\n\n"
        "示例：height = [0,1,0,2,1,0,1,3,2,1,2,1]，输出 6",
    ),
    (
        "二叉树的最大路径和",
        "给你一个二叉树的根节点 root，返回其最大路径和。路径中每个节点只能出现一次，"
        "且不一定经过根节点。\n\n"
        "示例：root = [-10,9,20,null,null,15,7]，输出 42（路径：15 → 20 → 7）",
    ),
]

# ── 全局模型（启动时加载一次）────────────────────────────────────
_model = None
_tokenizer = None
_args: argparse.Namespace | None = None


def _load_model_once(model_path: str, use_4bit: bool) -> None:
    global _model, _tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"[CodeGuide] 加载模型：{model_path}")
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    quant_cfg = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_quant_type       = "nf4",
            bnb_4bit_compute_dtype    = torch.bfloat16,
            bnb_4bit_use_double_quant = True,
        )

    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config = quant_cfg,
        torch_dtype         = torch.bfloat16 if not use_4bit else None,
        device_map          = "auto",
        trust_remote_code   = True,
    )
    _model.eval()
    print("[CodeGuide] 模型就绪 ✓")


# ── 流式生成核心逻辑 ─────────────────────────────────────────────

def _build_messages(
    history: list[list[str | None]],
    new_problem: str,
) -> list[dict]:
    """
    将 Gradio Chatbot 的 history（[[user,assistant], ...]）
    + 当前新输入 组装为 ChatML messages 列表。
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": new_problem})
    return messages


def stream_response(
    problem: str,
    history: list[list[str | None]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Generator[tuple[list, str], None, None]:
    """
    Gradio 生成器函数：
    - 每产生一个 token 块，yield (更新后的history, 空字符串) 给界面
    - 最终 yield 包含完整回复的 history
    """
    if _model is None or _tokenizer is None:
        yield history + [[problem, "⚠️ 模型尚未加载，请稍候…"]], ""
        return

    if not problem.strip():
        yield history, ""
        return

    import torch
    from transformers import TextIteratorStreamer

    messages = _build_messages(history, problem)
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(
        text, return_tensors="pt", truncation=True, max_length=3072
    ).to(_model.device)

    streamer = TextIteratorStreamer(
        _tokenizer,
        skip_prompt         = True,
        skip_special_tokens = True,
    )
    gen_kwargs = dict(
        **inputs,
        streamer           = streamer,
        max_new_tokens     = max_new_tokens,
        temperature        = max(temperature, 1e-4),
        top_p              = top_p,
        do_sample          = temperature > 0.05,
        repetition_penalty = 1.1,
        pad_token_id       = _tokenizer.pad_token_id,
    )

    thread = Thread(target=_model.generate, kwargs=gen_kwargs, daemon=True)
    thread.start()

    # 逐 token 更新 Chatbot 组件
    partial = ""
    new_history = history + [[problem, ""]]
    for chunk in streamer:
        partial += chunk
        new_history[-1][1] = partial
        yield new_history, ""

    thread.join()
    # 最终状态：清空输入框
    yield new_history, ""


# ── Gradio UI 定义 ───────────────────────────────────────────────

def build_ui(args: argparse.Namespace):
    import gradio as gr

    # ── 主题与 CSS ───────────────────────────────────────────────
    custom_css = """
    .title-area { text-align: center; padding: 8px 0 4px 0; }
    .title-area h1 { font-size: 1.6rem; margin: 0; }
    .title-area p  { color: #666; margin: 2px 0 0 0; font-size: 0.9rem; }
    .example-btn { font-size: 0.82rem !important; }
    #submit-btn  { background: #2563eb !important; color: white !important; }
    #submit-btn:hover { background: #1d4ed8 !important; }
    """

    with gr.Blocks(
        title  = "CodeGuide-LLM",
        theme  = gr.themes.Soft(primary_hue="blue"),
        css    = custom_css,
    ) as demo:

        # ── 页头 ─────────────────────────────────────────────────
        with gr.Row(elem_classes="title-area"):
            gr.HTML("""
            <div class="title-area">
              <h1>🧑‍🏫 CodeGuide-LLM</h1>
              <p>OI/ACM 算法教学助手 · 步进式讲解 · 由 Qwen2.5-Coder-7B + GRPO 微调</p>
            </div>
            """)

        # ── 主体：左右双栏 ────────────────────────────────────────
        with gr.Row(equal_height=False):

            # ── 左栏：输入区 ──────────────────────────────────────
            with gr.Column(scale=4, min_width=300):
                gr.Markdown("### 📝 题目描述")
                problem_input = gr.Textbox(
                    label       = "",
                    placeholder = "在这里输入算法题目描述，例如：\n\n给定一个整数数组 nums…",
                    lines       = 10,
                    max_lines   = 20,
                    show_label  = False,
                )

                with gr.Row():
                    submit_btn = gr.Button(
                        "▶ 开始讲解", variant="primary",
                        elem_id="submit-btn", scale=3,
                    )
                    clear_btn = gr.Button("🗑 清空对话", scale=1)

                # ── 示例快捷按钮 ──────────────────────────────────
                gr.Markdown("##### 示例题目（点击填入）")
                with gr.Row(wrap=True):
                    for title, _ in EXAMPLE_PROBLEMS:
                        gr.Button(
                            title,
                            size           = "sm",
                            elem_classes   = "example-btn",
                        )

                # ── 生成参数（折叠） ──────────────────────────────
                with gr.Accordion("⚙️ 生成参数", open=False):
                    max_new_tokens_slider = gr.Slider(
                        256, 2048, value=1024, step=64,
                        label="最大生成 token 数",
                    )
                    temperature_slider = gr.Slider(
                        0.0, 1.5, value=0.7, step=0.05,
                        label="Temperature（0 = 确定性输出）",
                    )
                    top_p_slider = gr.Slider(
                        0.5, 1.0, value=0.9, step=0.05,
                        label="Top-p",
                    )

            # ── 右栏：输出区 ──────────────────────────────────────
            with gr.Column(scale=6, min_width=400):
                gr.Markdown("### 💡 教学步骤（流式输出）")
                chatbot = gr.Chatbot(
                    label          = "",
                    bubble_full_width = False,
                    height         = 580,
                    show_label     = False,
                    render_markdown= True,
                    avatar_images  = (None, "https://i.imgur.com/UezaxjX.png"),
                )

        # ── 底部提示 ──────────────────────────────────────────────
        gr.Markdown(
            "<center style='color:#999;font-size:0.8rem'>"
            "CodeGuide-LLM · 基于 Qwen2.5-Coder-7B-Instruct + SFT + GRPO 微调 · "
            "仅供学习交流使用"
            "</center>"
        )

        # ── 事件绑定 ──────────────────────────────────────────────

        def _submit(problem, history, max_tok, temp, top_p):
            """提交按钮：触发流式生成。"""
            yield from stream_response(problem, history, max_tok, temp, top_p)

        def _clear():
            return [], ""

        # 提交按钮 & 回车触发（Textbox 的 submit 事件）
        gen_event = submit_btn.click(
            fn      = _submit,
            inputs  = [problem_input, chatbot,
                       max_new_tokens_slider, temperature_slider, top_p_slider],
            outputs = [chatbot, problem_input],
        )
        problem_input.submit(
            fn      = _submit,
            inputs  = [problem_input, chatbot,
                       max_new_tokens_slider, temperature_slider, top_p_slider],
            outputs = [chatbot, problem_input],
        )

        # 清空按钮
        clear_btn.click(fn=_clear, outputs=[chatbot, problem_input])

        # 示例按钮（逐个绑定，点击后填入输入框）
        example_buttons = [
            btn for btn in demo.blocks.values()
            if hasattr(btn, "elem_classes")
            and isinstance(getattr(btn, "elem_classes", None), list)
            and "example-btn" in getattr(btn, "elem_classes", [])
        ]
        for btn, (_, prob_text) in zip(example_buttons, EXAMPLE_PROBLEMS):
            btn.click(
                fn      = lambda t=prob_text: t,
                outputs = problem_input,
            )

    return demo


# ── 入口 ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CodeGuide-LLM Gradio Web Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="models/codeguide_llm_merged",
        help="模型路径",
    )
    parser.add_argument(
        "--no_4bit", dest="use_4bit", action="store_false",
        help="禁用 4bit 量化（需约 14 GB VRAM）",
    )
    parser.add_argument("--port",  type=int,  default=7860)
    parser.add_argument("--share", action="store_true",
                        help="生成 Gradio 公开分享链接")
    parser.set_defaults(use_4bit=True)
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"错误：模型路径不存在：{args.model}", file=sys.stderr)
        print("请先完成训练并合并模型：python scripts/train_grpo.py", file=sys.stderr)
        sys.exit(1)

    try:
        import gradio as gr  # noqa: F401
    except ImportError:
        print("错误：未安装 gradio，请运行：pip install gradio>=4.0", file=sys.stderr)
        sys.exit(1)

    # 加载模型（全局单次）
    _load_model_once(args.model, args.use_4bit)

    demo = build_ui(args)
    print(f"\n[CodeGuide] 启动 Web UI → http://localhost:{args.port}")
    if args.share:
        print("[CodeGuide] 正在生成公开分享链接…")

    demo.queue(max_size=4).launch(
        server_name = "0.0.0.0",
        server_port = args.port,
        share       = args.share,
        show_error  = True,
        quiet       = False,
    )


if __name__ == "__main__":
    main()
