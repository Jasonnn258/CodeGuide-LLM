# 百度 AI Studio 初始化脚本
# 在 AI Studio 笔记本中运行此代码

import os
import sys

# 检查环境
print("检查 AI Studio 环境...")

# 安装 PyTorch（如果需要）
print("安装 PyTorch...")
!pip install torch torchvision -i https://mirror.baidu.com/pypi/simple

# 安装项目依赖
print("安装项目依赖...")
!pip install -q -r requirements.txt -i https://mirror.baidu.com/pypi/simple

# 检查 GPU
print("检查 GPU 状态...")
!nvidia-smi

# 验证 PyTorch
print("验证 PyTorch...")
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"GPU 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("\n✅ AI Studio 环境配置完成！")
print("项目目录:", os.getcwd())
