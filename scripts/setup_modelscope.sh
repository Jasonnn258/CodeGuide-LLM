#!/bin/bash
# ModelScope 环境配置脚本
# 用于本地开发环境配置

echo "=========================================="
echo " ModelScope 本地开发环境配置"
echo "=========================================="

# 1. 检查 Python 版本
echo "步骤 1：检查 Python 版本..."
python3 --version

# 2. 创建虚拟环境（推荐）
echo ""
echo "步骤 2：创建虚拟环境..."
if [ ! -d "modelscope_env" ]; then
    python3 -m venv modelscope_env
    echo "✅ 虚拟环境创建成功！"
else
    echo "⚠️ 虚拟环境已存在"
fi

# 激活虚拟环境
source modelscope_env/bin/activate

# 3. 安装 PyTorch（根据你的 CUDA 版本选择）
echo ""
echo "步骤 3：安装 PyTorch..."
echo "请选择你的 CUDA 版本："
echo "  1) CUDA 11.8"
echo "  2) CUDA 12.1"
echo "  3) CPU only"
read -p "请输入选项 [1-3]: " cuda_version

case $cuda_version in
    1)
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        ;;
    2)
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        ;;
    3)
        pip install torch torchvision
        ;;
    *)
        echo "❌ 无效选项，使用 CPU 版本"
        pip install torch torchvision
        ;;
esac

# 4. 安装 ModelScope SDK
echo ""
echo "步骤 4：安装 ModelScope SDK..."
pip install modelscope

# 5. 安装常用依赖
echo ""
echo "步骤 5：安装常用依赖..."
pip install \
    numpy>=1.26.0 \
    pandas>=2.2.0 \
    tqdm>=4.66.0 \
    transformers>=4.47.0 \
    datasets>=3.0.0 \
    peft>=0.14.0 \
    accelerate>=1.1.0 \
    bitsandbytes>=0.45.0 \
    einops>=0.8.0 \
    openai>=1.55.0 \
    wandb>=0.18.0

echo ""
echo "=========================================="
echo " ✅ 环境配置完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 激活虚拟环境：source modelscope_env/bin/activate"
echo "2. 登录 ModelScope：modelscope login"
echo "3. 或访问 ModelScope Notebook 使用免费 GPU："
echo "   https://www.modelscope.cn/my/notebook"
echo ""
