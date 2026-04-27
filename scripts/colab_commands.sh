#!/bin/bash
# Colab 常用命令

# 启动 Colab 代码服务器
# colabcode --port 10000

# 查看 GPU 状态
# !nvidia-smi

# 监控 GPU 使用
# !nvidia-smi -l 1

# 查看系统资源
# !cat /proc/meminfo | grep MemTotal
# !cat /proc/cpuinfo | grep "model name"

# 挂载 Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# 安装依赖
# !pip install -r requirements.txt

# 运行训练
# !python scripts/train_sft.py --config configs/train_config.yaml
