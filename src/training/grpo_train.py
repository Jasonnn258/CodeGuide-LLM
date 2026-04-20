#!/usr/bin/env python3
"""
CodeGuide-LLM GRPO 后训练主入口

技术要点：
  - 从 SFT adapter（models/sft_adapter/）热启动，无 adapter 则从 base 冷启动
  - TRL GRPOTrainer：在线采样 + Group Relative Policy Optimization
  - 奖励信号：combined_reward = alpha×accuracy + (1-alpha)×format（本地，无 API）
  - 数据：sft_train.jsonl 的 prompt 部分（只传题目，不传参考答案）
  - 产物：LoRA adapter（models/grpo_final/）+ 合并模型（models/codeguide_llm_merged/）

运行：
    python src/training/grpo_train.py
    python src/training/grpo_train.py --config configs/train_config.yaml
    python src/training/grpo_train.py --resume_from_checkpoint models/grpo_final/checkpoint-200

显存估算（RTX 4090, 24 GB）：
    NF4 base model      ~4  GB
    LoRA adapter        ~0.3 GB
    4 × rollout buffer  ~3  GB  (4 generations × 1024 tokens)
    Optimizer (8bit)    ~1.5 GB
    Activations (gc)    ~3  GB
    ──────────────────────────
    合计                ~12 GB  （余量充足）
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

# 项目根目录加入 sys.path（src/training/ → 上两级为项目根）
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("grpo_train")


# ════════════════════════════════════════════════════════════════
# 1. 模型加载（SFT adapter 热启动）
# ════════════════════════════════════════════════════════════════

def load_policy_model(cfg):
    """
    优先从 SFT adapter 热启动；若 adapter 不存在则从 base model 冷启动。

    热启动步骤：
      1. Unsloth 检测到 adapter_config.json 后自动加载 base + adapter
      2. from_pretrained 默认冻结所有参数；手动恢复 LoRA 参数的 requires_grad
    冷启动步骤：
      1. 直接加载 base model
      2. 附加新的 LoRA adapter

    注：use_gradient_checkpointing 使用 True（标准 HF 实现），
    而非 Unsloth 专属的 "unsloth" 模式，因为 GRPOTrainer 的 online rollout
    阶段需要在 inference/training 模式间切换，Unsloth 实现兼容性不稳定。
    """
    from unsloth import FastLanguageModel

    grpo = cfg.grpo
    adapter_path = Path(grpo.sft_adapter_path)
    has_adapter = (adapter_path / "adapter_config.json").exists()

    if has_adapter:
        logger.info("热启动：从 SFT adapter 加载 → %s", adapter_path)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name    = str(adapter_path),
            max_seq_length= grpo.max_seq_length,
            dtype         = None,
            load_in_4bit  = True,
        )
        # 恢复 LoRA 参数为可训练状态（base model 权重保持冻结）
        trainable = 0
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)
                trainable += param.numel()
        logger.info("已恢复 %s 个 LoRA 参数为可训练状态", f"{trainable:,}")

    else:
        logger.warning(
            "未找到 SFT adapter（%s），从 base model 冷启动。"
            "建议先运行 python scripts/train_sft.py", adapter_path
        )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name    = cfg.model.local_path or cfg.model.name,
            max_seq_length= grpo.max_seq_length,
            dtype         = None,
            load_in_4bit  = True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r              = cfg.sft.lora_r,
            lora_alpha     = cfg.sft.lora_alpha,
            lora_dropout   = cfg.sft.lora_dropout,
            target_modules = list(cfg.lora.target_modules),
            bias           = "none",
            use_gradient_checkpointing = True,
            random_state   = grpo.seed,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # GRPO 生成阶段需要 left padding

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(
        "可训练参数：%s / %s（%.3f%%）",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )
    return model, tokenizer


# ════════════════════════════════════════════════════════════════
# 2. 数据集准备
# ════════════════════════════════════════════════════════════════

_GRPO_SYSTEM = (
    "你是 CodeGuide，一位专为 OI/ACM 初学者设计的算法教学助手。"
    "当用户提出一道算法题时，你会："
    "1. 先用简单的语言理解题意，举例说明；"
    "2. 从暴力解出发，逐步优化到最优解；"
    "3. 每一步都解释"为什么这样想"，而不只是"怎么做"；"
    "4. 最后给出带详细注释的完整 Python 代码。"
    "你的讲解应当通俗易懂，适合刚开始学习算法竞赛的初学者。"
)


def prepare_dataset(cfg, tokenizer, Dataset):
    """
    从 sft_train.jsonl 构建 GRPO 数据集。

    输出列：
      - "prompt"      : ChatML 格式字符串（system+user，末尾含 generation prompt）
      - "test_cases"  : JSON 字符串，反序列化后为 List[dict]（透传给 reward_fn）

    不传入 assistant 内容，让模型自由生成后由 reward_fn 打分。
    """
    data_path = Path(cfg.grpo.train_data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"训练数据不存在：{data_path}\n"
            "请先运行：python scripts/build_sft_dataset.py"
        )

    logger.info("加载训练数据：%s", data_path)
    records, skipped = [], 0

    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            messages = obj.get("messages", [])

            # 提取 user 消息内容
            user_content = next(
                (m["content"] for m in messages if m.get("role") == "user"),
                None,
            )
            if not user_content:
                skipped += 1
                continue

            # 只保留 system + user（不含 assistant），末尾追加 generation prompt
            prompt_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": _GRPO_SYSTEM},
                    {"role": "user",   "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            meta = obj.get("metadata", {})
            tc_raw = meta.get("test_cases", [])
            records.append({
                "prompt":     prompt_text,
                "test_cases": json.dumps(tc_raw, ensure_ascii=False),
            })

    logger.info("数据集：%d 条（跳过 %d 条缺少 user 消息）", len(records), skipped)
    if not records:
        raise ValueError("数据集为空，请检查 sft_train.jsonl 格式")

    return Dataset.from_list(records)


# ════════════════════════════════════════════════════════════════
# 3. 奖励函数
# ════════════════════════════════════════════════════════════════

def make_reward_fn(alpha: float, exec_timeout: float):
    """
    返回符合 GRPOTrainer 接口的奖励函数（本地执行，无 API 调用）。

    GRPOTrainer 调用签名（trl >= 0.8）：
        reward_fn(prompts, completions, **extra_dataset_columns)

    奖励公式：
        combined = alpha × accuracy_reward(code, test_cases)
                 + (1-alpha) × format_reward(completion)
    """
    from src.data.code_validator import extract_code
    from src.reward_functions import combined_reward

    def reward_fn(
        prompts:     list[str],
        completions: list[str],
        test_cases:  list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        rewards: list[float] = []
        for idx, completion in enumerate(completions):
            code = extract_code(completion) or ""

            tc: list[dict] = []
            if test_cases is not None and idx < len(test_cases):
                try:
                    parsed = json.loads(test_cases[idx])
                    if isinstance(parsed, list):
                        tc = parsed
                except (json.JSONDecodeError, TypeError):
                    pass

            score = combined_reward(
                solution_code = code,
                test_cases    = tc,
                response      = completion,
                alpha         = alpha,
                timeout       = exec_timeout,
            )
            rewards.append(score)
        return rewards

    return reward_fn


# ════════════════════════════════════════════════════════════════
# 4. 训练摘要
# ════════════════════════════════════════════════════════════════

def _print_summary(cfg, n_samples: int) -> None:
    grpo = cfg.grpo
    steps_per_epoch = math.ceil(
        n_samples
        / (grpo.per_device_train_batch_size * grpo.gradient_accumulation_steps)
    )
    total_steps     = steps_per_epoch * grpo.num_train_epochs
    rollouts_per_step = grpo.per_device_train_batch_size * grpo.num_generations
    logger.info(
        "\n"
        "╔══════════════════════════════════════════════╗\n"
        "║       CodeGuide-LLM  GRPO  Training           ║\n"
        "╠══════════════════════════════════════════════╣\n"
        "║  SFT adapter : %-29s║\n"
        "║  num_gen     : %-3d  (rollouts/step: %-10d)║\n"
        "║  Batch       : %d × accumulate %d steps        ║\n"
        "║  LR          : %-29s║\n"
        "║  KL coef     : %-29s║\n"
        "║  Epochs      : %-3d  steps/epoch: %-13d║\n"
        "║  Total steps : %-29d║\n"
        "╚══════════════════════════════════════════════╝",
        grpo.sft_adapter_path,
        grpo.num_generations, rollouts_per_step,
        grpo.per_device_train_batch_size, grpo.gradient_accumulation_steps,
        f"{grpo.learning_rate:.1e}",
        str(grpo.kl_coef),
        grpo.num_train_epochs, steps_per_epoch,
        total_steps,
    )


# ════════════════════════════════════════════════════════════════
# 5. 主流程
# ════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CodeGuide-LLM GRPO 后训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/train_config.yaml",
                        help="OmegaConf YAML 配置文件路径")
    parser.add_argument("--resume_from_checkpoint", default=None,
                        help="从指定 checkpoint 恢复，例如 models/grpo_final/checkpoint-200")
    cli = parser.parse_args()

    # ── 依赖检查 ──────────────────────────────────────────────────
    try:
        from unsloth import FastLanguageModel  # noqa: F401
    except ImportError:
        logger.error("未找到 unsloth，请安装：pip install unsloth[cu121-ampere-torch240]")
        sys.exit(1)

    try:
        import wandb
        from datasets import Dataset
        from omegaconf import OmegaConf
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        logger.error("缺少依赖：%s\n请运行 pip install -r requirements.txt", e)
        sys.exit(1)

    # ── 配置 ──────────────────────────────────────────────────────
    cfg_path = Path(cli.config)
    if not cfg_path.exists():
        logger.error("配置文件不存在：%s", cfg_path)
        sys.exit(1)
    cfg  = OmegaConf.load(cfg_path)
    grpo = cfg.grpo

    for d in (grpo.output_dir, grpo.merged_dir, grpo.logging_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── W&B ───────────────────────────────────────────────────────
    wandb.init(
        project = "codeguide-grpo",
        name    = f"grpo-qwen2.5-coder-7b-kl{grpo.kl_coef}",
        tags    = ["grpo", "qlora", "qwen2.5", "oi-teaching"],
        config  = OmegaConf.to_container(grpo, resolve=True),
    )

    # ── 模型加载 ──────────────────────────────────────────────────
    model, tokenizer = load_policy_model(cfg)

    # ── 数据集 ────────────────────────────────────────────────────
    dataset = prepare_dataset(cfg, tokenizer, Dataset)
    _print_summary(cfg, len(dataset))

    # ── 奖励函数 ──────────────────────────────────────────────────
    reward_fn = make_reward_fn(
        alpha        = float(cfg.reward.alpha),
        exec_timeout = float(cfg.reward.exec_timeout),
    )

    # ── GRPOConfig ────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        # GRPO 专属
        num_generations   = grpo.num_generations,
        max_new_tokens    = grpo.max_new_tokens,
        temperature       = grpo.temperature,
        top_p             = grpo.top_p,
        kl_coef           = grpo.kl_coef,        # 注：TRL 用 kl_coef（非 kl_coeff）
        max_prompt_length = grpo.max_prompt_length,

        # 优化器 / 调度器
        learning_rate            = grpo.learning_rate,
        lr_scheduler_type        = grpo.lr_scheduler_type,
        warmup_ratio             = grpo.warmup_ratio,
        weight_decay             = grpo.weight_decay,
        max_grad_norm            = grpo.max_grad_norm,

        # 批次
        per_device_train_batch_size = grpo.per_device_train_batch_size,
        gradient_accumulation_steps = grpo.gradient_accumulation_steps,
        num_train_epochs            = grpo.num_train_epochs,

        # 精度 & 显存
        bf16                   = grpo.bf16,
        fp16                   = grpo.fp16,
        gradient_checkpointing = grpo.gradient_checkpointing,

        # 保存 & 日志
        output_dir       = grpo.output_dir,
        logging_dir      = grpo.logging_dir,
        logging_steps    = grpo.logging_steps,
        save_strategy    = "steps",
        save_steps       = grpo.save_steps,
        save_total_limit = 3,

        # 其他
        seed                   = grpo.seed,
        report_to              = "wandb",
        run_name               = wandb.run.name,
        dataloader_num_workers = 2,
        remove_unused_columns  = False,  # 必须 False，否则 test_cases 列被删除
    )

    # ── GRPOTrainer ───────────────────────────────────────────────
    trainer = GRPOTrainer(
        model         = model,
        args          = grpo_config,
        train_dataset = dataset,
        reward_funcs  = reward_fn,
        tokenizer     = tokenizer,
    )

    # ── 训练 ──────────────────────────────────────────────────────
    logger.info("开始 GRPO 训练…")
    train_result = trainer.train(
        resume_from_checkpoint=cli.resume_from_checkpoint
    )

    # ── 保存 LoRA adapter ─────────────────────────────────────────
    logger.info("保存 GRPO LoRA adapter → %s", grpo.output_dir)
    model.save_pretrained(grpo.output_dir)
    tokenizer.save_pretrained(grpo.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    # ── 合并 LoRA → 完整模型（bf16，约 14 GB）─────────────────────
    logger.info("合并 LoRA → 完整模型 → %s", grpo.merged_dir)
    model.save_pretrained_merged(
        grpo.merged_dir,
        tokenizer,
        save_method = "merged_16bit",
    )

    wandb.finish()
    logger.info(
        "\n训练完成！\n"
        "  LoRA adapter  : %s\n"
        "  合并模型       : %s\n"
        "\n推理：\n"
        "  python scripts/inference_demo.py --model %s",
        grpo.output_dir, grpo.merged_dir, grpo.merged_dir,
    )


if __name__ == "__main__":
    main()
