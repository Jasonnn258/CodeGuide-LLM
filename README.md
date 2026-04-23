# CodeGuide-LLM

> 用强化学习让代码大模型学会"讲解"，而非"直接给答案"

## 项目背景

OI/ACM 初学者的核心痛点不是缺少题解代码，而是缺少**逐步推理的过程讲解**。现有代码 LLM（如 GPT-4o、DeepSeek-Coder）在被问到算法题时，往往直接输出完整代码，跳过了"为什么这样想"的关键步骤。

CodeGuide-LLM 的目标是：通过 **GRPO 强化学习 + QLoRA 参数高效微调**，使开源代码模型具备"步进式算法教学"能力——像一位有耐心的 OI 教练，先分析题意，再引导思路，最后才给出代码。

本项目在标准 GRPO 训练流程的基础上，针对实际训练中发现的 **12 项具体问题**进行了有针对性的迭代改进，形成了一套更鲁棒的强化学习教学微调方案。

---

## 技术路线

```
原始题目（deepmind/code_contests + BAAI/TACO）
        │
        ▼
GPT-4o 蒸馏 → 步进式讲解数据（DataQualityChecker 过滤低质量样本）
        │
        ▼
监督微调 SFT（Qwen2.5-Coder-7B-Instruct + QLoRA，warm-start）
        │
        ▼
GRPO 强化学习微调
  ├── Reward 1: 代码正确性（本地沙箱执行 / AST 静态分析）
  ├── Reward 2: 讲解格式规范（内容感知连续评分，防 hacking）
  ├── Reward 3: 教学质量（LocalTeachingReward，毫秒级启发式）
  └── Batch 内 Z-score 归一化 + Generation Collapse 检测
        │
        ▼
最优 Checkpoint 选择（验证集 Pass@1，非最后一个 epoch）
        │
        ▼
双盲评测（GPT-4o Judge，3 维度 + Bootstrap 95% CI + 显著性检验）
```

---

## Backbone：Qwen2.5-Coder-7B-Instruct

| 维度 | 理由 |
|------|------|
| 代码能力 | HumanEval 88.4%，EvoEval / LiveCodeBench 均优于同参数竞品 |
| 中文友好 | 原生中英双语预训练，讲解中文无需额外对齐 |
| Unsloth 兼容 | 官方支持 Qwen2.5 系列，2× 训练加速 |
| 单卡可训 | 7B + NF4 QLoRA：约 10-12 GB，RTX 4090 (24 GB) 绰绰有余 |
| 指令跟随 | Instruct 版已做 RLHF，对 GRPO 进一步对齐更友好 |

---

## 目录结构

```
CodeGuide-LLM/
├── configs/
│   └── train_config.yaml        # 全量训练配置（含 curriculum / normalize_rewards 等）
├── data/
│   └── sft_train.jsonl          # GPT-4o 蒸馏数据（ChatML 格式）
├── evals/
│   ├── blind_eval.py            # 双盲评测流水线（4 phase）
│   ├── ablation.py              # 控制变量 Ablation 实验
│   ├── benchmarks/              # 评测集
│   └── results/                 # 评测报告输出
├── models/
│   ├── sft_adapter/             # SFT LoRA adapter
│   ├── grpo_final/              # GRPO 最终 adapter
│   ├── grpo_best/               # 验证集 Pass@1 最优 checkpoint（自动保存）
│   └── codeguide_llm_merged/    # 合并后完整模型（推理用）
├── scripts/
│   ├── build_sft_dataset.py     # 异步并发蒸馏 + 质量过滤
│   ├── train_sft.py             # SFT 训练入口
│   ├── inference_demo.py        # CLI 推理 Demo
│   └── gradio_demo.py           # Web UI
├── src/
│   ├── data/
│   │   ├── loader.py            # 题库加载（code_contests / TACO）
│   │   ├── code_validator.py    # 代码提取 + 沙箱执行
│   │   └── quality.py          # DataQualityChecker（蒸馏质量过滤）
│   ├── reward/
│   │   ├── format.py            # FormatComplianceReward（内容感知，防 hacking）
│   │   ├── correctness.py       # CodeCorrectnessReward（含 AST 静态分析）
│   │   ├── teaching.py          # LocalTeachingReward + API 版（可配置切换）
│   │   └── composite.py        # 三路复合奖励
│   ├── reward_functions.py      # accuracy_reward / format_reward（GRPO 接口）
│   └── training/
│       └── grpo_train.py        # GRPO 训练主入口（含 Curriculum / Best Ckpt）
└── tests/
    ├── test_rewards.py           # 三路 reward 单元测试
    └── test_teaching_alignment.py # Local vs API Teaching Reward Spearman 对齐验证
```

---

## 奖励函数设计

### 三路奖励（GRPO 训练使用）

| 奖励信号 | 权重 | 实现方式 | 关键改进 |
|---------|------|---------|---------|
| 代码正确性（Accuracy） | 0.6 | 本地沙箱执行 + stdin/stdout 比对 | 无测试用例时改为 AST 静态分析（原版直接给 1.0 导致 hacking） |
| 格式规范性（Format） | 0.4 | 正则 + 内容感知评分 | 步骤标题必须有 ≥50 字符实质内容 + Jaccard 去重，防止空洞结构得满分 |
| 教学质量（Teaching） | 监控 | LocalTeachingReward（4 维度启发式） | 替换 GPT-4o-mini API（2-5s → <1ms，离散 10 档 → 连续分） |

### Batch 内 Z-score 归一化

三路 reward 方差各异，直接加权会导致高方差路径隐式主导梯度。启用 `normalize_rewards: true` 后，每个 batch 内对 combined reward 做归一化，使梯度贡献与 alpha 配置真正对齐：

```python
rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

### Generation Collapse 检测

GRPO 每个 prompt 采样 `num_generations=4` 条 completion。若 4 条趋同，组内 reward 方差趋零，梯度消失但 loss 不报警。训练器实时监控：

- `train/generation_reward_var`：组内方差均值
- `train/collapse_ratio`：方差 < 0.01 的 prompt 占比
- 连续 20 步 collapse_ratio > 0.5 自动 WARNING

---

## 训练特性

### Curriculum Learning（课程学习）

在 `configs/train_config.yaml` 中启用 `curriculum.enabled: true`，按难度分阶段训练：

```
阶段 1（easy）   → 建立基础能力，max_new_tokens=512
阶段 2（medium） → 中等难度泛化，max_new_tokens=768
阶段 3（hard）   → 挑战难题，max_new_tokens=1024
```

设计动机：GRPO 早期直接遇到 hard 题时 Pass@1≈0，reward≈0，梯度完全消失。

### 最优 Checkpoint 选择

启用 `save_best: true` 后，`BestCheckpointCallback` 每 `eval_steps` 步在验证集（有测试用例的题目）上计算 Pass@1，Pass@1 超越历史最高时自动保存到 `models/grpo_best/`。

设计动机：GRPO 后期 training reward 虚高（reward overfit），实际 Pass@1 在下降；按最后一个 epoch 保存等于随机选 checkpoint。

---

## 快速复现

### 1. 环境安装

```bash
# Python 3.11 + CUDA 12.1
conda create -n codeguide python=3.11 -y
conda activate codeguide
pip install -r requirements.txt
```

> unsloth 需与 CUDA 版本匹配，详见 [unsloth 安装指南](https://github.com/unslothai/unsloth)

### 2. 数据蒸馏

```bash
export OPENAI_API_KEY=sk-...

# 异步并发蒸馏（含质量过滤，建议 quality_threshold=0.6）
python scripts/build_sft_dataset.py \
    --max_items 10000 \
    --concurrency 10 \
    --quality_threshold 0.6 \
    --out data/sft_train.jsonl
```

质量过滤会丢弃无代码块、步骤截断、内容过短的样本（约过滤 15-20%）。

### 3. SFT 监督微调（可选热启动）

```bash
python scripts/train_sft.py --config configs/train_config.yaml
```

### 4. GRPO 强化学习训练

```bash
# 标准训练
python src/training/grpo_train.py --config configs/train_config.yaml

# 启用 Curriculum Learning
# 在 configs/train_config.yaml 中设置 curriculum.enabled: true，然后：
python src/training/grpo_train.py --config configs/train_config.yaml
```

训练过程中 WandB 记录（需 `wandb login`）：

| 指标 | 含义 |
|------|------|
| `reward/accuracy` | 代码正确性均值 |
| `reward/format` | 格式规范性均值 |
| `reward/teaching` | 教学质量均值 |
| `reward/total` | 加权总分（原始，未归一化） |
| `train/generation_reward_var` | 组内 reward 方差（collapse 检测） |
| `train/collapse_ratio` | 坍缩 prompt 占比 |
| `eval/pass1` | 验证集 Pass@1（save_best 模式） |
| `train/curriculum_stage` | 当前课程阶段（curriculum 模式） |

### 5. 推理 Demo

```bash
# CLI
python scripts/inference_demo.py --model models/codeguide_llm_merged/

# Web UI
python scripts/gradio_demo.py --model models/codeguide_llm_merged/
```

---

## 评测流水线

### 双盲评测（`evals/blind_eval.py`）

```bash
# 一键运行完整评测（约 2-4 小时）
export OPENAI_API_KEY=sk-...
python evals/blind_eval.py --phase all

# 仅重跑报告
python evals/blind_eval.py --phase report
```

评测包含 3 个维度（GPT-4o 盲测，temperature=0）：

| 维度 | 说明 |
|------|------|
| 讲解易懂性（Clarity） | 语言是否清晰、举例是否有助理解 |
| 思路连贯性（Coherence） | 推导逻辑是否前后一致 |
| 初学者友好度（Beginner-Friendly） | 是否使用生活化比喻、避免过专业术语 |

报告包含 95% Bootstrap 置信区间和配对 Bootstrap 显著性检验（p<0.05）。

### Ablation 实验（`evals/ablation.py`）

```bash
# 离线运行，无需 GPU，约 2 分钟
python evals/ablation.py --n 50 --out evals/ablation_report.md
```

对 4 种配置（baseline / +format_fix / +acc_fix / +local_teaching）打分，量化每个改动的独立增益。

### Teaching Reward 对齐验证（`tests/test_teaching_alignment.py`）

```bash
# 仅本地版（无需 API key）
python tests/test_teaching_alignment.py --local_only

# 完整对齐分析（需 OPENAI_API_KEY，约 200 × 2s）
export OPENAI_API_KEY=sk-...
python tests/test_teaching_alignment.py --n 200
```

计算 LocalTeachingReward 与 GPT-4o-mini 评分的 Spearman ρ（目标 ρ > 0.6）。

---

## 数据格式（ChatML）

```json
{
  "id": "code_contests_12345",
  "messages": [
    {"role": "system",    "content": "你是 CodeGuide，一位专为 OI/ACM 初学者设计的算法教学助手…"},
    {"role": "user",      "content": "请按上述格式讲解以下算法题：\n【题目描述】…"},
    {"role": "assistant", "content": "**第一步：理解题意**\n…\n```python\n…\n```"}
  ],
  "metadata": {
    "source": "code_contests",
    "difficulty": "medium",
    "tags": ["heap", "sorting"],
    "pass_rate": 0.823
  }
}
```

---

## 关键设计决策（面试叙事）

| 问题 | 现象 | 解决方案 |
|------|------|---------|
| Format Reward Hacking | 模型学会写空洞步骤标题，reward 高但内容空洞（Goodhart's Law） | 步骤内容长度约束（≥50字符）+ Jaccard 去重 |
| Accuracy Reward 退化 | 40% TACO 样本无测试用例，全给 1.0 → 模型绕过代码正确性检测 | 无测试时改为 AST 静态分析估分 |
| Teaching Reward 瓶颈 | GPT-4o-mini 每 step 调用耗时 8-20s；10 档离散分梯度稀疏 | LocalTeachingReward：<1ms，连续分 [0,1] |
| 梯度方差不对齐 | 三路 reward 方差各异，高方差路径隐式主导梯度，alpha 权重失效 | Batch 内 Z-score 归一化 |
| Generation Collapse | 4 条 completion 趋同，组内 reward 方差→0，梯度消失但 loss 正常 | WandB 实时监控 collapse_ratio，连续超阈值预警 |
| Reward Overfit | GRPO 后期 training reward 虚高，实际 Pass@1 下降 | BestCheckpointCallback：验证集 Pass@1 选优 |
| 早期梯度消失 | 直接遇到 hard 题 Pass@1≈0，reward≈0，无法学习 | Curriculum Learning：easy→medium→hard |
| 评测置信度不足 | 100 题无 CI，无法判断改进是否统计显著 | Bootstrap 95% CI + 配对显著性检验 |

---

## 参考资料

- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: 用 RL 激发推理能力](https://arxiv.org/abs/2501.12948)
- [Unsloth 官方文档](https://github.com/unslothai/unsloth)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
- [Qwen2.5-Coder 技术报告](https://arxiv.org/abs/2409.12186)
