#!/usr/bin/env python3
"""
CodeGuide-LLM SFT 训练脚本

技术要点：
  - Unsloth FastLanguageModel 加载 Qwen2.5-Coder-7B-Instruct（NF4 QLoRA）
  - TRL SFTTrainer + train_on_responses_only：只对 assistant token 计算 loss
  - ChatML 格式：system / user / assistant 三角色
  - 梯度检查点 + bf16，适配单卡 RTX 4090 (24GB)
  - wandb 实验追踪，每 50 step 打印 loss

用法：
    python scripts/train_sft.py
    python scripts/train_sft.py --config configs/train_config.yaml
    python scripts/train_sft.py --resume_from_checkpoint models/sft_checkpoints/checkpoint-400
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_sft")

# ── 延迟导入（unsloth 必须最先导入，否则可能与 torch 冲突）────────
def _import_heavy():
    """集中管理重量级导入，便于环境未安装时给出友好报错。"""
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import train_on_responses_only
    except ImportError:
        logger.error(
            "未找到 unsloth，请先安装：\n"
            "  pip install unsloth[cu121-ampere-torch240]"
        )
        sys.exit(1)

    try:
        import wandb
        from datasets import Dataset
        from omegaconf import OmegaConf
        from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        logger.error("缺少依赖：%s\n请运行 pip install -r requirements.txt", e)
        sys.exit(1)

    return (
        FastLanguageModel,
        train_on_responses_only,
        wandb,
        Dataset,
        OmegaConf,
        TrainerCallback,
        TrainerState,
        TrainerControl,
        TrainingArguments,
        SFTTrainer,
        SFTConfig,
    )


# ── 自定义 Callback：每 N step 打印格式化 loss ────────────────────

def make_print_loss_callback(log_every: int):
    """
    工厂函数，返回每 log_every step 打印 loss 的 Callback 类。
    使用工厂函数避免循环导入 TrainerCallback 基类。
    """
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

    class PrintLossCallback(TrainerCallback):
        """
        在 on_log 中捕获 loss，每 log_every 步打印一行：
            Step  150/1200 | loss=0.8421 | lr=1.23e-04 | grad_norm=0.42 | elapsed=12m30s
        """

        def __init__(self):
            self._start_time: float | None = None

        def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ) -> None:
            import time
            self._start_time = time.time()

        def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs: dict | None = None,
            **kwargs,
        ) -> None:
            import time

            if logs is None or state.global_step % log_every != 0:
                return

            loss     = logs.get("loss", logs.get("train_loss"))
            lr       = logs.get("learning_rate")
            gnorm    = logs.get("grad_norm")
            max_step = state.max_steps or "?"

            elapsed_str = ""
            if self._start_time is not None:
                elapsed_s   = time.time() - self._start_time
                elapsed_str = f" | elapsed={int(elapsed_s//60)}m{int(elapsed_s%60):02d}s"

            parts = [f"Step {state.global_step:>6}/{max_step}"]
            if loss  is not None: parts.append(f"loss={loss:.4f}")
            if lr    is not None: parts.append(f"lr={lr:.3e}")
            if gnorm is not None: parts.append(f"grad_norm={gnorm:.3f}")

            logger.info(" | ".join(parts) + elapsed_str)

    return PrintLossCallback


# ── 模型 & Tokenizer ──────────────────────────────────────────────

def load_model_and_tokenizer(cfg, FastLanguageModel):
    """
    使用 Unsloth 加载 QLoRA 量化模型并附加 LoRA adapter。

    Unsloth 在 from_pretrained 阶段就完成 4bit 量化加载，
    get_peft_model 则在量化权重之上插入可训练的 LoRA 层。
    gradient_checkpointing="unsloth" 使用 Unsloth 自定义实现，
    比 HF 原生省约 30% 显存且支持长上下文。
    """
    sft = cfg.sft
    model_name = cfg.model.local_path or cfg.model.name

    logger.info("加载模型：%s（NF4 QLoRA）", model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = model_name,
        max_seq_length= sft.max_seq_length,
        dtype         = None,          # 自动选择（4090 上为 bf16）
        load_in_4bit  = True,
    )

    logger.info("附加 LoRA adapter（r=%d, alpha=%d）", sft.lora_r, sft.lora_alpha)
    model = FastLanguageModel.get_peft_model(
        model,
        r              = sft.lora_r,
        lora_alpha     = sft.lora_alpha,
        lora_dropout   = sft.lora_dropout,
        target_modules = list(cfg.lora.target_modules),  # 复用 GRPO 中定义的模块列表
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = sft.seed,
    )

    # Qwen2.5 tokenizer 已内置 ChatML 模板，此处确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # SFT 阶段右 padding，避免 attention mask 问题

    # 打印可训练参数量
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(
        "可训练参数：%s / %s（%.2f%%）",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )
    return model, tokenizer


# ── 数据集加载 & ChatML 格式化 ────────────────────────────────────

def load_sft_dataset(cfg, tokenizer, Dataset):
    """
    从 sft_train.jsonl 加载数据，切分 train/eval，
    并将 messages 列表格式化为带 chat template 的字符串。

    只保留 messages 字段，其余 metadata 在训练中不使用。

    ChatML 格式（Qwen2.5 原生支持）：
        <|im_start|>system\n...<|im_end|>
        <|im_start|>user\n...<|im_end|>
        <|im_start|>assistant\n...<|im_end|>
    """
    import json
    import random

    data_path = Path(cfg.sft.train_data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"训练数据不存在：{data_path}\n"
            "请先运行：python scripts/build_sft_dataset.py"
        )

    logger.info("加载数据集：%s", data_path)
    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # 只保留 messages 字段
            if "messages" in obj:
                records.append({"messages": obj["messages"]})

    logger.info("原始样本数：%d", len(records))
    if not records:
        raise ValueError("数据集为空，请检查 sft_train.jsonl 格式")

    # 切分 train/eval
    random.seed(cfg.sft.seed)
    random.shuffle(records)
    eval_size  = max(1, int(len(records) * cfg.sft.eval_split_ratio))
    train_recs = records[eval_size:]
    eval_recs  = records[:eval_size]
    logger.info("train: %d 条 | eval: %d 条", len(train_recs), len(eval_recs))

    def fmt(batch):
        """
        将 messages 列表转为 chat template 字符串。
        add_generation_prompt=False：训练时不在末尾追加 <|im_start|>assistant\n。
        tokenize=False：返回字符串，让 SFTTrainer 自行 tokenize。
        """
        return {
            "text": [
                tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for msgs in batch["messages"]
            ]
        }

    train_ds = Dataset.from_list(train_recs).map(fmt, batched=True, remove_columns=["messages"])
    eval_ds  = Dataset.from_list(eval_recs ).map(fmt, batched=True, remove_columns=["messages"])

    return train_ds, eval_ds


# ── 训练入口 ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CodeGuide-LLM SFT 训练")
    parser.add_argument("--config", default="configs/train_config.yaml", help="配置文件路径")
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="从指定 checkpoint 继续训练，如 models/sft_checkpoints/checkpoint-400",
    )
    args = parser.parse_args()

    # ── 导入 ──────────────────────────────────────────────────────
    (
        FastLanguageModel,
        train_on_responses_only,
        wandb,
        Dataset,
        OmegaConf,
        TrainerCallback,
        TrainerState,
        TrainerControl,
        TrainingArguments,
        SFTTrainer,
        SFTConfig,
    ) = _import_heavy()

    # ── 配置 ──────────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    sft = cfg.sft

    # 确保输出目录存在
    Path(sft.output_dir).mkdir(parents=True, exist_ok=True)
    Path(sft.adapter_dir).mkdir(parents=True, exist_ok=True)
    Path(sft.logging_dir).mkdir(parents=True, exist_ok=True)

    # ── W&B ───────────────────────────────────────────────────────
    wandb.init(
        project = "codeguide-sft",                      # 用户指定的 project name
        name    = cfg.wandb.sft_run_name,
        tags    = list(cfg.wandb.sft_tags),
        config  = OmegaConf.to_container(cfg.sft, resolve=True),
    )

    # ── 模型 ──────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg, FastLanguageModel)

    # ── 数据 ──────────────────────────────────────────────────────
    train_ds, eval_ds = load_sft_dataset(cfg, tokenizer, Dataset)

    # ── SFTConfig（等同于 TrainingArguments，但含 SFT 专属参数）──────
    training_cfg = SFTConfig(
        # 路径
        output_dir          = sft.output_dir,
        logging_dir         = sft.logging_dir,

        # 训练超参
        num_train_epochs    = sft.num_train_epochs,
        per_device_train_batch_size = sft.per_device_train_batch_size,
        gradient_accumulation_steps = sft.gradient_accumulation_steps,
        learning_rate       = sft.learning_rate,
        lr_scheduler_type   = sft.lr_scheduler_type,
        warmup_ratio        = sft.warmup_ratio,
        weight_decay        = sft.weight_decay,
        max_grad_norm       = 1.0,

        # 序列长度（SFTConfig 专属，截断超长样本）
        max_seq_length      = sft.max_seq_length,

        # 精度 & 显存
        bf16                = sft.bf16,
        fp16                = sft.fp16,
        gradient_checkpointing = sft.gradient_checkpointing,
        dataloader_num_workers = sft.dataloader_num_workers,

        # 日志 & 评估
        logging_steps       = sft.logging_steps,
        save_strategy       = "steps",
        save_steps          = sft.save_steps,
        eval_strategy       = "steps",
        eval_steps          = sft.eval_steps,
        load_best_model_at_end = True,
        metric_for_best_model  = "eval_loss",
        greater_is_better      = False,

        # SFT 专属
        dataset_text_field  = "text",   # 使用我们 fmt() 生成的 text 列
        packing             = False,    # ChatML 不做 sequence packing，保留对话完整性

        # 其他
        seed                = sft.seed,
        report_to           = "wandb",
        run_name            = cfg.wandb.sft_run_name,
        save_total_limit    = 3,        # 最多保留 3 个 checkpoint
    )

    # ── SFTTrainer ────────────────────────────────────────────────
    trainer = SFTTrainer(
        model         = model,
        tokenizer     = tokenizer,
        args          = training_cfg,
        train_dataset = train_ds,
        eval_dataset  = eval_ds,
    )

    # ── 只对 assistant 回复计算 loss ─────────────────────────────
    # Qwen2.5 ChatML 中：
    #   user    部分以 <|im_start|>user\n 开头
    #   assistant 部分以 <|im_start|>assistant\n 开头
    # train_on_responses_only 会将 system + user token 的 label 置为 -100，
    # 使其不参与 cross-entropy loss 计算。
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part    = "<|im_start|>assistant\n",
    )

    # ── 注册 Callback ─────────────────────────────────────────────
    PrintLossCallback = make_print_loss_callback(sft.logging_steps)
    trainer.add_callback(PrintLossCallback())

    # ── 打印训练概要 ──────────────────────────────────────────────
    steps_per_epoch = math.ceil(
        len(train_ds) / (sft.per_device_train_batch_size * sft.gradient_accumulation_steps)
    )
    total_steps = steps_per_epoch * sft.num_train_epochs
    logger.info(
        "\n"
        "╔══════════════════════════════════════════╗\n"
        "║         CodeGuide-LLM  SFT  Training      ║\n"
        "╠══════════════════════════════════════════╣\n"
        "║  Backbone : %-29s║\n"
        "║  LoRA r   : %-3d  alpha : %-21d║\n"
        "║  Batch    : %d×%d = %-25d║\n"
        "║  LR       : %-29s║\n"
        "║  Epochs   : %-3d  Steps/epoch : %-12d║\n"
        "║  Total    : %-6d steps                  ║\n"
        "╚══════════════════════════════════════════╝",
        cfg.model.name.split("/")[-1],
        sft.lora_r, sft.lora_alpha,
        sft.per_device_train_batch_size, sft.gradient_accumulation_steps,
        sft.per_device_train_batch_size * sft.gradient_accumulation_steps,
        f"{sft.learning_rate:.1e}",
        sft.num_train_epochs, steps_per_epoch,
        total_steps,
    )

    # ── 训练 ──────────────────────────────────────────────────────
    logger.info("开始训练…")
    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    # ── 保存 LoRA Adapter ─────────────────────────────────────────
    # 只保存 LoRA 差量权重（~几十 MB），不保存全量模型
    logger.info("保存 LoRA adapter → %s", sft.adapter_dir)
    model.save_pretrained(sft.adapter_dir)
    tokenizer.save_pretrained(sft.adapter_dir)

    # ── 训练指标 ──────────────────────────────────────────────────
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    wandb.finish()
    logger.info(
        "训练完成！\n"
        "  LoRA adapter : %s\n"
        "  Checkpoints  : %s\n"
        "  合并全量权重（可选）：\n"
        "    python -c \"\n"
        "    from unsloth import FastLanguageModel\n"
        "    model, tok = FastLanguageModel.from_pretrained('%s')\n"
        "    model.save_pretrained_merged('models/sft_merged', tok, save_method='merged_16bit')\n"
        "    \"",
        sft.adapter_dir,
        sft.output_dir,
        sft.adapter_dir,
    )


if __name__ == "__main__":
    main()
