#!/bin/bash
# Google Colab 环境配置脚本
# 用于本地准备和 Colab 环境配置

echo "=========================================="
echo " Google Colab 环境配置"
echo "=========================================="

# 1. 检查 Python 版本
echo "步骤 1：检查 Python 版本..."
python3 --version

# 2. 安装 Google Colab 本地工具
echo ""
echo "步骤 2：安装 Colab 本地工具..."
pip install colabcode

# 3. 创建 Colab 配置文件
echo ""
echo "步骤 3：创建 Colab 配置文件..."
mkdir -p configs/colab

cat > configs/colab/colab_setup.py << 'EOF'
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
EOF

# 4. 创建 Colab 快捷命令脚本
echo ""
echo "步骤 4：创建 Colab 快捷命令..."
cat > scripts/colab_commands.sh << 'EOF'
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
EOF

chmod +x scripts/colab_commands.sh

echo ""
echo "=========================================="
echo " ✅ Colab 环境配置完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 访问 Google Colab: https://colab.research.google.com"
echo "2. 创建新的笔记本"
echo "3. 在 '运行时' -> '更改运行时类型' 中选择 'GPU'"
echo "4. 复制 configs/colab/colab_setup.py 中的代码到 Colab 运行"
echo ""
echo "Google Colab 免费 GPU 资源："
echo "  - Tesla T4 (16GB) 或 P100 (16GB)"
echo "  - 单次会话最长 12 小时"
echo "  - 每天约 24 小时使用限制"
echo ""
