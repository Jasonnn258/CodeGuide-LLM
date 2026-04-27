#!/bin/bash
# 百度 AI Studio 星河社区环境配置脚本
# 用于本地准备和 AI Studio 环境配置

echo "=========================================="
echo " 百度 AI Studio 星河社区环境配置"
echo "=========================================="

# 1. 检查 Python 版本
echo "步骤 1：检查 Python 版本..."
python3 --version

# 2. 安装必要的工具
echo ""
echo "步骤 2：安装必要的工具..."
pip install requests

# 3. 创建 AI Studio 配置文件
echo ""
echo "步骤 3：创建 AI Studio 配置文件..."
mkdir -p configs/aistudio

cat > configs/aistudio/aistudio_setup.py << 'EOF'
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
EOF

# 4. 创建 AI Studio 快捷命令脚本
echo ""
echo "步骤 4：创建 AI Studio 快捷命令..."
cat > scripts/aistudio_commands.sh << 'EOF'
#!/bin/bash
# AI Studio 常用命令

# 安装 PyTorch
# !pip install torch torchvision -i https://mirror.baidu.com/pypi/simple

# 安装依赖
# !pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple

# 查看 GPU 状态
# !nvidia-smi

# 运行训练
# !python scripts/train_sft.py --config configs/train_config.yaml

# 上传文件到 AI Studio
# 在 AI Studio 界面中使用 "上传文件" 功能

# 下载结果
# 在 AI Studio 界面中使用 "下载" 功能
EOF

chmod +x scripts/aistudio_commands.sh

echo ""
echo "=========================================="
echo " ✅ AI Studio 环境配置完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 访问百度 AI Studio 星河社区: https://aistudio.baidu.com"
echo "2. 登录百度账号"
echo "3. 创建新的 Notebook 项目"
echo "4. 选择免费 GPU 资源（V100）"
echo "5. 复制 configs/aistudio/aistudio_setup.py 中的代码到 AI Studio 运行"
echo ""
echo "百度 AI Studio 免费 GPU 资源："
echo "  - Tesla V100 (16GB)"
echo "  - 每天 8-12 小时免费使用时间"
echo "  - 需实名认证"
echo ""
