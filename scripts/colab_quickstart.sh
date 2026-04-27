#!/bin/bash
# CodeGuide-LLM - Colab 快速启动脚本
# 适用于 Google Colab 环境

echo "=========================================="
echo " CodeGuide-LLM - Colab 快速启动脚本"
echo "=========================================="

# 检查是否在 Colab 环境
if [ ! -d "/content" ]; then
    echo "❌ 错误：此脚本仅适用于 Google Colab 环境"
    exit 1
fi

# 步骤 1：检查 GPU
echo "步骤 1：检查 GPU 状态..."
nvidia-smi

# 步骤 2：安装依赖
echo "步骤 2：安装依赖..."
pip install -q ipywidgets==8.1.5
pip install -q torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -q unsloth[cu121-ampere-torch240]>=2024.12.0
pip install -q trl>=0.12.0
pip install -q transformers>=4.47.0
pip install -q datasets>=3.0.0
pip install -q peft>=0.14.0
pip install -q accelerate>=1.1.0
pip install -q bitsandbytes>=0.45.0
pip install -q flash-attn>=2.7.0
pip install -q einops>=0.8.0
pip install -q openai>=1.55.0
pip install -q wandb>=0.18.0
pip install -q numpy>=1.26.0
pip install -q pandas>=2.2.0
pip install -q tqdm>=4.66.0
pip install -q jsonlines>=4.0.0
pip install -q pytest>=8.0.0
pip install -q rich>=13.9.0
pip install -q omegaconf>=2.3.0

echo "✅ 依赖安装完成！"

# 步骤 3：挂载 Google Drive（可选）
echo "步骤 3：挂载 Google Drive..."
if command -v python3 &> /dev/null; then
    python3 -c "from google.colab import drive; drive.mount('/content/drive')"
fi

# 步骤 4：创建工作目录
echo "步骤 4：创建工作目录..."
WORK_DIR="/content/CodeGuide-LLM"
mkdir -p $WORK_DIR
cd $WORK_DIR

# 步骤 5：克隆项目（如果还没有）
if [ ! -f "README.md" ]; then
    echo "步骤 5：克隆项目..."
    git clone https://github.com/你的用户名/CodeGuide-LLM.git .
fi

# 步骤 6：创建 Colab 配置文件
echo "步骤 6：创建 Colab 配置文件..."
cat > configs/colab_config.yaml << 'EOF'
model:
  base_model: Qwen/Qwen2.5-Coder-7B-Instruct
  max_seq_length: 4096
  quantization: nf4
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05

training:
  output_dir: models/grpo_final
  num_train_epochs: 3
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-05
  warmup_ratio: 0.03
  max_grad_norm: 0.3
  save_steps: 100
  eval_steps: 100
  logging_steps: 10
  fp16: true
  bf16: true
  save_best: true
  best_model_dir: models/grpo_best

grpo:
  num_generations: 4
  max_new_tokens: 1024
  reward_weights:
    accuracy: 0.6
    format: 0.4
    teaching: 0.0
  normalize_rewards: true

curriculum:
  enabled: false
  stages:
  - name: easy
    difficulty: easy
    max_new_tokens: 512
  - name: medium
    difficulty: medium
    max_new_tokens: 768
  - name: hard
    difficulty: hard
    max_new_tokens: 1024

data:
  train_file: data/sft_train.jsonl
  eval_file: data/eval.jsonl
  test_size: 0.1

wandb:
  project: codeguide-llm
  name: colab-run-1
  log_model: true
EOF

# 步骤 7：提示用户配置 API Keys
echo ""
echo "=========================================="
echo " 接下来请手动配置："
echo "=========================================="
echo ""
echo "1. 配置 OpenAI API Key："
echo "   export OPENAI_API_KEY=你的API密钥"
echo ""
echo "2. 配置 WandB："
echo "   wandb login"
echo ""
echo "3. 准备训练数据（如果需要）："
echo "   python scripts/build_sft_dataset.py --help"
echo ""
echo "4. 开始训练："
echo "   python scripts/train_sft.py --config configs/colab_config.yaml"
echo "   python src/training/grpo_train.py --config configs/colab_config.yaml"
echo ""
echo "✅ 初始化完成！"
