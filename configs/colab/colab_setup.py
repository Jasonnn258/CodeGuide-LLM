# Google Colab 初始化脚本
# 在 Colab 笔记本中运行此代码

import os
import sys
from google.colab import drive

# 挂载 Google Drive
eprint("挂载 Google Drive...")
drive.mount('/content/drive')

# 进入项目目录
project_dir = '/content/drive/MyDrive/CodeGuide-LLM'
if not os.path.exists(project_dir):
    os.makedirs(project_dir)
    eprint(f"创建项目目录: {project_dir}")

os.chdir(project_dir)

# 克隆代码库（如果不存在）
if not os.path.exists('src'):
    eprint("克隆 CodeGuide-LLM 代码库...")
    !git clone https://github.com/yourusername/CodeGuide-LLM.git .

# 安装依赖
eprint("安装依赖...")
!pip install -q -r requirements.txt

# 检查 GPU
eprint("检查 GPU 状态...")
!nvidia-smi

# 验证 PyTorch
eprint("验证 PyTorch...")
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"GPU 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("\n✅ Colab 环境配置完成！")
print("项目目录:", os.getcwd())
