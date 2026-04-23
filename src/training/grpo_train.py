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
    "3. 每一步都解释「为什么这样想」，而不只是「怎么做」；"
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
# 2b. Curriculum 数据集构建
# ════════════════════════════════════════════════════════════════

def build_curriculum_dataset(cfg, tokenizer, Dataset, difficulty: str):
    """
    从 sft_train.jsonl 中按难度过滤，构建 Curriculum 某阶段的数据集。

    设计动机（v3 新增）：
    GRPO 早期直接遇到 hard 题，Pass@1≈0，所有 reward≈0，梯度消失。
    课程学习按 easy→medium→hard 分阶段，让模型先建立基础能力再接受挑战。

    面试叙事："我观察到训练初期 accuracy_reward 几乎全为零，模型学不到任何东西；
    引入 Curriculum Learning 后，easy 阶段结束时 Pass@1 从 0 提升到 0.3，
    medium 阶段 Pass@1 进一步到 0.5，最后 hard 阶段仍有梯度可学。"

    Args:
        difficulty: "easy" | "medium" | "hard"（匹配 metadata.difficulty 字段）
    """
    data_path = Path(cfg.grpo.train_data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"训练数据不存在：{data_path}\n"
            "请先运行：python scripts/build_sft_dataset.py"
        )

    logger.info("Curriculum 数据集（difficulty=%s）：%s", difficulty, data_path)
    records, skipped, filtered = [], 0, 0

    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta = obj.get("metadata", {})

            # 难度过滤
            if meta.get("difficulty", "").lower() != difficulty.lower():
                filtered += 1
                continue

            messages     = obj.get("messages", [])
            user_content = next(
                (m["content"] for m in messages if m.get("role") == "user"), None
            )
            if not user_content:
                skipped += 1
                continue

            prompt_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": _GRPO_SYSTEM},
                    {"role": "user",   "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            tc_raw = meta.get("test_cases", [])
            records.append({
                "prompt":     prompt_text,
                "test_cases": json.dumps(tc_raw, ensure_ascii=False),
            })

    logger.info(
        "Curriculum [%s]：%d 条（过滤其他难度 %d 条，跳过格式异常 %d 条）",
        difficulty, len(records), filtered, skipped,
    )
    if not records:
        logger.warning(
            "Curriculum [%s] 数据集为空！sft_train.jsonl 中可能缺少 difficulty 字段。",
            difficulty,
        )
    return Dataset.from_list(records)


# ════════════════════════════════════════════════════════════════
# 3. 奖励函数与可观测性
# ════════════════════════════════════════════════════════════════

# 全局 reward 缓冲区，由 reward_fn 写入，由 RewardLoggingCallback 读取并上报 WandB
# 使用列表而非 dict，便于线程安全追加（GRPO 在 trl 内部串行调用 reward_fn）
_reward_buffer: list[dict] = []


def make_reward_fn(alpha: float, exec_timeout: float):
    """
    返回符合 GRPOTrainer 接口的奖励函数（本地执行，无 API 调用）。

    改进（v2）：
    - 拆分三路奖励：accuracy / format / teaching 各自记录到 _reward_buffer
    - WandB 通过 RewardLoggingCallback 读取缓冲区，分别上报 reward/accuracy 等指标
    - 同时记录 reward_std（批内奖励标准差），监控奖励坍缩

    GRPOTrainer 调用签名（trl >= 0.8）：
        reward_fn(prompts, completions, **extra_dataset_columns)

    奖励公式：
        combined = alpha × accuracy_reward(code, test_cases)
                 + (1-alpha) × format_reward(completion)
        （teaching 单独记录用于监控，不进入 GRPO 梯度，可通过 composite.py 接入）
    """
    import statistics
    from src.data.code_validator import extract_code
    from src.reward_functions import accuracy_reward, format_reward, combined_reward
    from src.reward.teaching import build_teaching_reward

    # 注意：teaching reward 需要 cfg，这里用简单的 lambda 占位
    # 实际使用时通过 make_reward_fn_with_cfg 传入
    _teaching_fn = None

    def reward_fn(
        prompts:     list[str],
        completions: list[str],
        test_cases:  list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        acc_scores:  list[float] = []
        fmt_scores:  list[float] = []
        combined:    list[float] = []

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

            acc = accuracy_reward(code, tc, timeout=exec_timeout)
            fmt = format_reward(completion)
            score = alpha * acc + (1.0 - alpha) * fmt

            acc_scores.append(acc)
            fmt_scores.append(fmt)
            combined.append(score)

        # 记录到缓冲区（批级别统计）
        n = len(combined)
        batch_stats = {
            "reward/accuracy":   sum(acc_scores) / n,
            "reward/format":     sum(fmt_scores) / n,
            "reward/total":      sum(combined) / n,
            "reward/std":        statistics.stdev(combined) if n > 1 else 0.0,
            "reward/min":        min(combined),
            "reward/max":        max(combined),
        }
        # 计算 teaching 分数（仅监控，不进梯度）
        if _teaching_fn is not None:
            try:
                teaching_scores = _teaching_fn(prompts, completions)
                batch_stats["reward/teaching"] = sum(teaching_scores) / len(teaching_scores)
            except Exception:
                pass

        _reward_buffer.append(batch_stats)
        return combined

    return reward_fn


def make_reward_fn_with_cfg(cfg):
    """
    带完整配置的 reward_fn 工厂，支持 teaching reward 监控。

    改进（v3）：
    - Batch 内 Z-score 归一化（normalize_rewards: true 时启用）
      三路 reward 方差各异，直接加权后高方差路径会隐式主导梯度信号。
      Batch 内归一化使各路径梯度贡献与 alpha 配置真正对齐。
      公式：r_norm = (r - mean(batch)) / (std(batch) + 1e-8)
    - 每组 num_generations 条 completion 计算组内 reward 方差（Generation Collapse 检测）
      组内方差趋零说明模型坍缩为重复输出，GRPO 梯度消失，需要调高 temperature。
    """
    import statistics
    from src.data.code_validator import extract_code
    from src.reward_functions import accuracy_reward, format_reward
    from src.reward.teaching import build_teaching_reward

    alpha           = float(cfg.reward.alpha)
    exec_timeout    = float(cfg.reward.exec_timeout)
    teaching_fn     = build_teaching_reward(cfg)
    num_gen         = int(cfg.grpo.num_generations)
    normalize       = bool(getattr(cfg.grpo, "normalize_rewards", False))

    def reward_fn(
        prompts:     list[str],
        completions: list[str],
        test_cases:  list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        acc_scores: list[float] = []
        fmt_scores: list[float] = []
        raw_combined: list[float] = []

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

            acc = accuracy_reward(code, tc, timeout=exec_timeout)
            fmt = format_reward(completion)
            acc_scores.append(acc)
            fmt_scores.append(fmt)
            raw_combined.append(alpha * acc + (1.0 - alpha) * fmt)

        # Teaching reward（本地模式毫秒级，无 API 调用）
        try:
            teaching_scores = teaching_fn(prompts, completions)
            avg_teaching = sum(teaching_scores) / len(teaching_scores)
        except Exception:
            avg_teaching = 0.0

        n = len(raw_combined)

        # ── 每组（prompt）内 reward 方差（Generation Collapse 检测） ──
        # GRPO 每个 prompt 采样 num_gen 条 completion，组内方差趋零 → 坍缩。
        group_vars: list[float] = []
        if num_gen > 1 and n >= num_gen:
            for i in range(0, n, num_gen):
                grp = raw_combined[i : i + num_gen]
                if len(grp) > 1:
                    grp_mean = sum(grp) / len(grp)
                    grp_var  = sum((r - grp_mean) ** 2 for r in grp) / len(grp)
                    group_vars.append(grp_var)

        # ── Batch 内 Z-score 归一化（可选）──────────────────────────
        # 归一化后送入 GRPO；日志记录原始 raw_combined，保留可解释性。
        if normalize and n > 1:
            mean_r = sum(raw_combined) / n
            std_r  = statistics.stdev(raw_combined)
            if std_r > 1e-8:
                combined_out = [(r - mean_r) / std_r for r in raw_combined]
            else:
                combined_out = [0.0] * n          # 方差为零时全部置零（坍缩态）
        else:
            combined_out = raw_combined

        _reward_buffer.append({
            "reward/accuracy":   sum(acc_scores) / n,
            "reward/format":     sum(fmt_scores) / n,
            "reward/teaching":   avg_teaching,
            "reward/total":      sum(raw_combined) / n,   # 记录原始分，非归一化
            "reward/std":        statistics.stdev(raw_combined) if n > 1 else 0.0,
            "reward/min":        min(raw_combined),
            "reward/max":        max(raw_combined),
            "_group_vars":       group_vars,              # 供 collapse 检测使用
        })
        return combined_out

    return reward_fn


# ════════════════════════════════════════════════════════════════
# 3b. 可观测性 Callback
# ════════════════════════════════════════════════════════════════

class RewardLoggingCallback:
    """
    自定义 WandB 回调（v2）：

    1. 分路 reward 上报（accuracy / format / teaching / total / std）
    2. Generation Collapse 检测（v2 新增）：
       - 每 batch 统计组内（num_gen 条 completion 为一组）reward 方差
       - 记录 train/generation_reward_var（组内方差均值）
       - 记录 train/collapse_ratio（组内方差 < 阈值的 prompt 占比）
       - 连续 collapse_warn_steps 步 collapse_ratio > 0.5 则发出 WARNING
    3. 定期样本示例记录

    设计动机：
    - Generation Collapse 比 KL 爆炸更隐蔽：loss 正常，但 reward 停止提升。
      根本原因是 4 条 completion 趋同后 GRPO 梯度接近零（组内 reward 方差 → 0）。
    - collapse_ratio 提供了早期预警，让用户能在梯度完全消失前调整 temperature。
    """

    def __init__(
        self,
        log_example_every_n_steps: int = 100,
        collapse_var_threshold: float   = 0.01,
        collapse_warn_steps: int        = 20,
    ):
        self.log_example_every_n_steps = log_example_every_n_steps
        self.collapse_var_threshold    = collapse_var_threshold
        self.collapse_warn_steps       = collapse_warn_steps
        self._step                     = 0
        self._consecutive_collapse     = 0   # 连续 collapse 步数计数
        self._example_table            = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        import wandb
        self._step += 1

        if not _reward_buffer:
            return
        latest = _reward_buffer[-1]

        # ── 分路 reward 上报（过滤内部字段）──────────────────────
        public_metrics = {k: v for k, v in latest.items() if not k.startswith("_")}
        wandb.log(public_metrics, step=state.global_step)

        # ── Generation Collapse 检测 ──────────────────────────────
        group_vars: list[float] = latest.get("_group_vars", [])
        if group_vars:
            avg_var       = sum(group_vars) / len(group_vars)
            collapse_ratio = sum(1 for v in group_vars if v < self.collapse_var_threshold) / len(group_vars)
            wandb.log({
                "train/generation_reward_var": avg_var,
                "train/collapse_ratio":        collapse_ratio,
            }, step=state.global_step)

            if collapse_ratio > 0.5:
                self._consecutive_collapse += 1
            else:
                self._consecutive_collapse = 0

            if self._consecutive_collapse >= self.collapse_warn_steps:
                logger.warning(
                    "⚠️  [Generation Collapse] 连续 %d 步 collapse_ratio=%.2f > 0.5。"
                    "组内 reward 方差持续趋零，GRPO 梯度即将消失。"
                    "建议：提高 temperature（当前 %.2f）或降低 kl_coef。",
                    self._consecutive_collapse, collapse_ratio,
                    getattr(args, "temperature", float("nan")),
                )

        # ── 定期记录样本示例 ──────────────────────────────────────
        if (self._step % self.log_example_every_n_steps == 0
                and "completions" in (logs or {})):
            completions = logs.get("completions", [])
            prompts     = logs.get("prompts", [])
            if completions and prompts:
                if self._example_table is None:
                    self._example_table = wandb.Table(
                        columns=["step", "prompt_excerpt", "completion_excerpt",
                                 "reward_total"]
                    )
                self._example_table.add_data(
                    state.global_step,
                    str(prompts[0])[:300],
                    str(completions[0])[:500],
                    latest.get("reward/total", 0.0),
                )
                wandb.log({"examples/completions": self._example_table},
                          step=state.global_step)


# ════════════════════════════════════════════════════════════════
# 3c. 验证集 Pass@1 最优 Checkpoint 选择
# ════════════════════════════════════════════════════════════════

class BestCheckpointCallback:
    """
    每 eval_steps 步在有测试用例的验证题上计算 Pass@1，
    当 Pass@1 超越历史最高时自动保存最优 checkpoint。

    设计动机（v3 新增）：
    GRPO 后期 training reward 虚高（模型在"讨好" reward 函数），
    但验证集 Pass@1 实际已开始下降（reward overfit）。
    按最后一个 epoch 保存等于随机选 checkpoint。
    用验证集 Pass@1 选优，能保证实际部署的模型质量。

    面试叙事："我在 GRPO 训练中发现 training reward 在 300 步后持续上升，
    但手动测试 Pass@1 却在下降——这是 reward overfit 的典型症状。
    因此我加了 BestCheckpointCallback，每 100 步用验证集 Pass@1 做 checkpoint 选择。"
    """

    def __init__(self, model, tokenizer, eval_problems: list[dict], cfg):
        from src.data.code_validator import extract_code
        from src.reward_functions import accuracy_reward

        self._extract_code    = extract_code
        self._accuracy_reward = accuracy_reward
        self.model        = model
        self.tokenizer    = tokenizer
        # 只保留有测试用例的题目
        self.eval_problems = [p for p in eval_problems if p.get("public_tests")]
        self.cfg          = cfg
        self.best_pass1   = -1.0
        self.best_step    = -1
        best_dir = getattr(cfg.grpo, "best_model_dir", "models/grpo_best")
        self.best_dir     = Path(best_dir)
        self.eval_steps   = int(getattr(cfg.grpo, "eval_steps", 100))
        logger.info(
            "[BestCheckpoint] 初始化：%d 道有测试用例的验证题，每 %d 步评估一次",
            len(self.eval_problems), self.eval_steps,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return
        if not self.eval_problems:
            return

        import torch
        import wandb

        logger.info("[BestCheckpoint] step=%d 开始验证集 Pass@1 评估…", state.global_step)
        self.model.eval()
        pass1_scores: list[float] = []

        with torch.inference_mode():
            for prob in self.eval_problems[:20]:   # 限制 20 题，避免评估耗时过长
                messages = [
                    {"role": "system", "content": _GRPO_SYSTEM},
                    {"role": "user",   "content": prob.get("description", "")},
                ]
                text   = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=1024
                ).to(self.model.device)
                out = self.model.generate(
                    **inputs,
                    max_new_tokens = 512,
                    do_sample      = False,
                    pad_token_id   = self.tokenizer.pad_token_id,
                )
                new_tokens = out[0][inputs["input_ids"].shape[1]:]
                completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                code = self._extract_code(completion) or ""
                score = self._accuracy_reward(
                    code, prob.get("public_tests", []), timeout=5.0
                )
                pass1_scores.append(score)

        self.model.train()
        avg_pass1 = sum(pass1_scores) / len(pass1_scores) if pass1_scores else 0.0
        is_best   = avg_pass1 > self.best_pass1

        wandb.log({
            "eval/pass1":    avg_pass1,
            "eval/is_best":  float(is_best),
        }, step=state.global_step)

        logger.info(
            "[BestCheckpoint] step=%d  Pass@1=%.3f  历史最优=%.3f  %s",
            state.global_step, avg_pass1, max(self.best_pass1, 0.0),
            "✅ 更新最优" if is_best else "—",
        )

        if is_best:
            self.best_pass1 = avg_pass1
            self.best_step  = state.global_step
            self.best_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(self.best_dir))
            self.tokenizer.save_pretrained(str(self.best_dir))


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

    # ── 奖励函数（带三路拆分监控 + 归一化）──────────────────────
    reward_fn = make_reward_fn_with_cfg(cfg)

    # ── GRPOConfig 工厂（Curriculum 各阶段可复用，仅 max_new_tokens 不同）──
    def _make_grpo_config(max_new_tokens: int, num_epochs: int = 1) -> "GRPOConfig":
        return GRPOConfig(
            # GRPO 专属
            num_generations   = grpo.num_generations,
            max_new_tokens    = max_new_tokens,
            temperature       = grpo.temperature,
            top_p             = grpo.top_p,
            kl_coef           = grpo.kl_coef,
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
            num_train_epochs            = num_epochs,
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
            remove_unused_columns  = False,
        )

    # ── Callback 构建 ─────────────────────────────────────────────
    from transformers import TrainerCallback

    collapse_threshold = float(getattr(cfg.grpo, "collapse_var_threshold", 0.01))
    collapse_warn_steps = int(getattr(cfg.grpo, "collapse_warn_steps", 20))
    reward_cb = RewardLoggingCallback(
        log_example_every_n_steps = 100,
        collapse_var_threshold    = collapse_threshold,
        collapse_warn_steps       = collapse_warn_steps,
    )

    class _RewardCallbackAdapter(TrainerCallback):
        """将 RewardLoggingCallback 适配到 HuggingFace TrainerCallback 接口。"""
        def on_log(self, args, state, control, logs=None, **kwargs):
            reward_cb.on_log(args, state, control, logs=logs, **kwargs)

    callbacks: list = [_RewardCallbackAdapter()]

    # ── BestCheckpointCallback（需要验证题目）────────────────────
    save_best = bool(getattr(cfg.grpo, "save_best", False))
    best_ckpt_cb: BestCheckpointCallback | None = None
    if save_best:
        # 从 sft_train.jsonl 中读取有测试用例的题目作为验证集
        eval_problems: list[dict] = []
        try:
            with open(Path(cfg.grpo.train_data), encoding="utf-8") as _ef:
                for _line in _ef:
                    _line = _line.strip()
                    if not _line:
                        continue
                    _obj = json.loads(_line)
                    meta = _obj.get("metadata", {})
                    if meta.get("test_cases"):
                        eval_problems.append({
                            "description": next(
                                (m["content"] for m in _obj.get("messages", [])
                                 if m.get("role") == "user"), ""
                            ),
                            "public_tests": meta.get("test_cases", []),
                        })
            import random as _rand
            _rand.seed(42)
            eval_problems = _rand.sample(eval_problems, min(50, len(eval_problems)))
        except Exception as _e:
            logger.warning("BestCheckpoint：无法加载验证集（%s），跳过", _e)

        if eval_problems:
            best_ckpt_cb = BestCheckpointCallback(model, tokenizer, eval_problems, cfg)

            class _BestCkptAdapter(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    best_ckpt_cb.on_step_end(args, state, control, **kwargs)
            callbacks.append(_BestCkptAdapter())
            logger.info("BestCheckpointCallback 已启用，验证题目数：%d", len(eval_problems))

    # ── 训练（支持 Curriculum Learning）──────────────────────────
    curriculum_cfg = getattr(cfg, "curriculum", None)
    use_curriculum = curriculum_cfg is not None and bool(getattr(curriculum_cfg, "enabled", False))

    if use_curriculum:
        stages = list(curriculum_cfg.stages)
        logger.info("Curriculum Learning 已启用，共 %d 阶段：%s",
                    len(stages), [s.difficulty for s in stages])

        for stage_idx, stage in enumerate(stages):
            difficulty    = stage.difficulty
            stage_epochs  = int(getattr(stage, "epochs", 1))
            stage_tokens  = int(getattr(stage, "max_new_tokens", grpo.max_new_tokens))

            logger.info(
                "═══ Curriculum Stage %d/%d (%s) — epochs=%d, max_new_tokens=%d ═══",
                stage_idx + 1, len(stages), difficulty, stage_epochs, stage_tokens,
            )
            wandb.log({"train/curriculum_stage": stage_idx + 1})

            stage_dataset = build_curriculum_dataset(cfg, tokenizer, Dataset, difficulty)
            if len(stage_dataset) == 0:
                logger.warning("Stage [%s] 数据集为空，跳过", difficulty)
                continue

            _print_summary(cfg, len(stage_dataset))
            grpo_config = _make_grpo_config(stage_tokens, stage_epochs)

            trainer = GRPOTrainer(
                model         = model,
                args          = grpo_config,
                train_dataset = stage_dataset,
                reward_funcs  = reward_fn,
                tokenizer     = tokenizer,
                callbacks     = callbacks,
            )
            # 仅第一阶段支持 resume；后续阶段从当前权重继续
            resume = cli.resume_from_checkpoint if stage_idx == 0 else None
            trainer.train(resume_from_checkpoint=resume)

            # 阶段结束保存中间 checkpoint
            stage_dir = Path(grpo.output_dir) / f"stage_{stage_idx + 1}_{difficulty}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(stage_dir))
            tokenizer.save_pretrained(str(stage_dir))
            logger.info("Stage [%s] 中间 checkpoint 已保存 → %s", difficulty, stage_dir)

    else:
        # 普通训练（无 Curriculum）
        dataset = prepare_dataset(cfg, tokenizer, Dataset)
        _print_summary(cfg, len(dataset))
        grpo_config = _make_grpo_config(grpo.max_new_tokens, grpo.num_train_epochs)

        trainer = GRPOTrainer(
            model         = model,
            args          = grpo_config,
            train_dataset = dataset,
            reward_funcs  = reward_fn,
            tokenizer     = tokenizer,
            callbacks     = callbacks,
        )
        logger.info("开始 GRPO 训练…")
        train_result = trainer.train(resume_from_checkpoint=cli.resume_from_checkpoint)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

    # ── 保存最终 LoRA adapter ────────────────────────────────────
    logger.info("保存 GRPO LoRA adapter → %s", grpo.output_dir)
    model.save_pretrained(grpo.output_dir)
    tokenizer.save_pretrained(grpo.output_dir)

    # ── 合并 LoRA → 完整模型（bf16，约 14 GB）─────────────────────
    logger.info("合并 LoRA → 完整模型 → %s", grpo.merged_dir)
    model.save_pretrained_merged(
        grpo.merged_dir,
        tokenizer,
        save_method = "merged_16bit",
    )

    if save_best and best_ckpt_cb and best_ckpt_cb.best_step >= 0:
        logger.info(
            "最优 Checkpoint（Pass@1=%.3f）已保存 → %s（step %d）",
            best_ckpt_cb.best_pass1, best_ckpt_cb.best_dir, best_ckpt_cb.best_step,
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
