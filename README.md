# CodeGuide-LLM

> 用强化学习让代码大模型学会"讲解"，而非"直接给答案"

## 项目背景

OI/ACM 初学者的核心痛点不是缺少题解代码，而是缺少**逐步推理的过程讲解**。现有代码 LLM（如 GPT-4o、DeepSeek-Coder）在被问到算法题时，往往直接输出完整代码，跳过了"为什么这样想"的关键步骤。

CodeGuide-LLM 的目标是：通过 **GRPO 强化学习 + QLoRA 参数高效微调**，使开源代码模型具备"步进式算法教学"能力——像一位有耐心的 OI 教练，先分析题意，再引导思路，最后才给出代码。

## 技术路线

```
原始数据 (LeetCode / Codeforces)
        │
        ▼
GPT-4o 蒸馏生成"步进式讲解"数据
        │
        ▼
监督微调 SFT（warm-start，可选）
        │
        ▼
GRPO 强化学习微调
  ├── Reward Model 1: 讲解步骤完整性（LLM-as-Judge）
  ├── Reward Model 2: 最终代码正确性（在线判题）
  └── Reward Model 3: 教学格式规范（规则打分）
        │
        ▼
Qwen2.5-Coder-7B-Instruct（QLoRA NF4，单卡 4090）
        │
        ▼
评测：教学质量 / Pass@k / 用户满意度
```

## Backbone 选型：Qwen2.5-Coder-7B-Instruct

| 维度 | 理由 |
|------|------|
| 代码能力 | HumanEval 88.4%，EvoEval、LiveCodeBench 均优于同参数规模竞品 |
| 中文友好 | 原生中英双语预训练，讲解中文无需额外对齐 |
| Unsloth 兼容 | unsloth 官方支持 Qwen2.5 系列，可获 2x 训练加速 |
| 单卡可训 | 7B + NF4 QLoRA：显存占用约 10-12 GB，4090 (24 GB) 绰绰有余 |
| 指令跟随 | Instruct 版已做 RLHF，对 GRPO 进一步对齐更友好 |

## 目录结构

```
CodeGuide-LLM/
├── data/
│   ├── raw/            # 原始题目数据（LeetCode JSON / Codeforces CF）
│   ├── processed/      # 清洗后的 prompt-response pair
│   └── distilled/      # GPT-4o 蒸馏生成的步进式讲解数据
├── models/
│   ├── checkpoints/    # 训练中间 checkpoint
│   └── final/          # 合并 LoRA 后的完整模型
├── src/
│   ├── data/           # 数据处理、蒸馏、格式化
│   ├── training/       # GRPO 训练主逻辑
│   ├── reward/         # 奖励函数实现
│   └── inference/      # 推理与 demo
├── scripts/
│   ├── prepare_data.sh
│   ├── run_distill.sh
│   └── run_train.sh
├── configs/
│   └── train_config.yaml
├── evals/
│   ├── benchmarks/     # 评测集
│   └── results/        # 评测结果
├── notebooks/          # 探索性实验
├── requirements.txt
└── README.md
```

## 快速复现

### 1. 环境安装

```bash
# 推荐 Python 3.11 + CUDA 12.1
conda create -n codeguide python=3.11 -y
conda activate codeguide

pip install -r requirements.txt
```

> unsloth 需要与 CUDA 版本匹配，详见 [unsloth 安装指南](https://github.com/unslothai/unsloth)

### 2. 数据准备

```bash
# 下载并清洗原始题目
bash scripts/prepare_data.sh

# 调用 GPT-4o 生成步进式讲解（需配置 OPENAI_API_KEY）
export OPENAI_API_KEY=sk-...
bash scripts/run_distill.sh
```

### 3. GRPO 训练

```bash
bash scripts/run_train.sh
# 或直接：
python src/training/grpo_train.py --config configs/train_config.yaml
```

### 4. 推理 Demo

```bash
python src/inference/chat.py --model models/final/codeguide-7b
```

## 数据格式

蒸馏数据的标准格式（存储为 JSONL）：

```json
{
  "problem": "给定一个整数数组，找出其中第 k 大的元素...",
  "difficulty": "medium",
  "tags": ["heap", "quickselect"],
  "teaching_steps": [
    "**第一步：理解题意**\n我们要找第 k 大，注意是排序后第 k 个...",
    "**第二步：分析暴力解**\n最直接的做法是排序，时间复杂度 O(n log n)...",
    "**第三步：思考优化**\n能否不全排序？注意到只需要前 k 大...",
    "**第四步：选择数据结构**\n维护一个大小为 k 的最小堆..."
  ],
  "final_code": "import heapq\ndef findKthLargest(nums, k):\n    return heapq.nlargest(k, nums)[-1]"
}
```

## 奖励函数设计

| 奖励信号 | 权重 | 实现方式 |
|---------|------|---------|
| 步骤完整性 | 0.4 | GPT-4o / Qwen2.5-72B 作为 Judge，检查是否覆盖"分析→思路→代码"三阶段 |
| 代码正确性 | 0.4 | 提取 code block 在本地执行测试用例，pass 率作为奖励 |
| 格式规范性 | 0.2 | 正则检查 Markdown 格式、步骤标题、代码块等 |

## 评测指标

- **教学质量**：LLM-as-Judge（1-10 分），评估讲解清晰度、递进性、适合初学者程度
- **代码正确性**：Pass@1 / Pass@5（在 LeetCode 样例上）
- **步骤覆盖率**：自动检测"题意分析 / 思路推导 / 复杂度分析 / 代码实现"四阶段的出现率

## 快速开始

> 前提：已完成训练并生成合并模型 `models/codeguide_llm_merged/`（见"快速复现"章节）。
> 若只想体验 Demo，可替换为任意兼容 Qwen2.5-Coder-7B-Instruct 的模型路径。

### 1. 安装依赖

```bash
# 推荐 Python 3.11 + CUDA 12.1
conda create -n codeguide python=3.11 -y
conda activate codeguide

pip install -r requirements.txt
pip install gradio>=4.0            # Web UI 额外依赖
```

> **注意**：`unsloth` 需与 CUDA 版本匹配，详见 [unsloth 安装指南](https://github.com/unslothai/unsloth)

---

### 2. 命令行交互 Demo

```bash
python scripts/inference_demo.py --model models/codeguide_llm_merged/
```

**可选参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `models/codeguide_llm_merged` | 模型路径 |
| `--no_4bit` | — | 禁用 4bit 量化（需约 14 GB VRAM） |
| `--max_new_tokens` | `1024` | 最大生成长度 |
| `--temperature` | `0.7` | 采样温度 |

**示例输出：**

```
你：给定一个整数数组，找出其中和为 target 的两个整数下标

CodeGuide：

**第一步：理解题意**
我们需要在数组中找到两个数，使得它们的和恰好等于 target，
并返回这两个数的下标。举例：
  nums = [2, 7, 11, 15]，target = 9
  → 因为 nums[0] + nums[1] = 2 + 7 = 9，返回 [0, 1]

**第二步：分析暴力解法**
最直接的做法是双重循环枚举所有下标对 (i, j)：
  for i in range(n):
      for j in range(i+1, n):
          if nums[i] + nums[j] == target: return [i, j]
时间复杂度 O(n²)，当 n 很大时会超时。

**第三步：寻找优化方向**
注意：当我们遍历到 nums[i] 时，
只需要快速查找数组中是否存在 target - nums[i]。
这正是哈希表的强项！

**第四步：哈希表解法**
用字典记录"已见过的数 → 其下标"：
  seen = {}
  for i, num in enumerate(nums):
      complement = target - num
      if complement in seen:
          return [seen[complement], i]
      seen[num] = i
时间 O(n)，空间 O(n)。

**第五步：完整 Python 实现**

\`\`\`python
def twoSum(nums: list[int], target: int) -> list[int]:
    seen = {}          # 存储 {值: 下标}
    for i, num in enumerate(nums):
        complement = target - num      # 需要找的另一个数
        if complement in seen:
            return [seen[complement], i]   # 找到，返回两个下标
        seen[num] = i  # 还没找到，记录当前数
    return []          # 题目保证有解，不会走到这里

# 验证
print(twoSum([2, 7, 11, 15], 9))   # [0, 1]
print(twoSum([3, 2, 4], 6))         # [1, 2]
\`\`\`

**复杂度分析**
- 时间复杂度：O(n)，只需遍历一次数组
- 空间复杂度：O(n)，哈希表最多存 n 个元素
```

**内置命令：**
- `/clear` — 重置对话历史，开始新话题
- `/exit`  — 退出程序
- `/help`  — 显示帮助信息

---

### 3. Gradio Web UI

```bash
python scripts/gradio_demo.py
# 默认在 http://localhost:7860 打开
```

**可选参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `models/codeguide_llm_merged` | 模型路径 |
| `--port` | `7860` | 监听端口 |
| `--share` | — | 生成 Gradio 公开分享链接 |
| `--no_4bit` | — | 禁用 4bit 量化 |

**界面布局：**

```
┌─────────────────────┬─────────────────────────────────────┐
│  📝 题目描述（左侧）  │  💡 教学步骤——流式输出（右侧）          │
│                     │                                     │
│  [输入框]            │  ┌─ 对话历史 ────────────────────┐  │
│                     │  │  你：题目描述…                  │  │
│  [▶ 开始讲解] [清空]  │  │  CodeGuide：第一步：理解题意…  │  │
│                     │  │  （实时流式更新）               │  │
│  示例题目快捷按钮:    │  └───────────────────────────────┘  │
│  两数之和 | 接雨水…  │                                     │
│                     │  ⚙️ 生成参数（可折叠展开）             │
└─────────────────────┴─────────────────────────────────────┘
```

**显存占用参考（RTX 4090）：**

| 模式 | 显存占用 | 启动命令 |
|------|---------|---------|
| 4bit NF4（默认） | ~5-6 GB | `python scripts/gradio_demo.py` |
| bf16 全精度 | ~14 GB | `python scripts/gradio_demo.py --no_4bit` |

---

## 参考资料

- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- [Unsloth 官方文档](https://github.com/unslothai/unsloth)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
- [Qwen2.5-Coder 技术报告](https://arxiv.org/abs/2409.12186)
- [DeepSeek-R1: 用 RL 激发推理能力](https://arxiv.org/abs/2501.12948)
# CodeGuide-LLM
