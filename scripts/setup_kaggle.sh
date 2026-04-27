#!/bin/bash
# Kaggle 环境配置脚本
# 用于本地开发环境配置 Kaggle API 和准备上传代码

echo "=========================================="
echo " Kaggle 本地开发环境配置"
echo "=========================================="

# 1. 检查 Python 版本
echo "步骤 1：检查 Python 版本..."
python3 --version

# 2. 安装 Kaggle API
echo ""
echo "步骤 2：安装 Kaggle API..."
pip install kaggle

# 3. 创建 Kaggle 配置目录
echo ""
echo "步骤 3：创建 Kaggle 配置目录..."
mkdir -p ~/.kaggle

# 4. 获取 Kaggle API Token
echo ""
echo "=========================================="
echo " ⚠️ 重要：获取 Kaggle API Token"
echo "=========================================="
echo ""
echo "请按照以下步骤操作："
echo ""
echo "1. 访问 https://www.kaggle.com"
echo "2. 登录你的 Kaggle 账号"
echo "3. 点击右上角头像 → 'Account'"
echo "4. 滚动到 'API' 部分"
echo "5. 点击 'Create New API Token'"
echo "6. 下载 kaggle.json 文件"
echo "7. 将 kaggle.json 移动到 ~/.kaggle/ 目录"
echo ""
echo "命令："
echo "  mv ~/Downloads/kaggle.json ~/.kaggle/"
echo "  chmod 600 ~/.kaggle/kaggle.json"
echo ""

# 5. 测试 Kaggle API
echo "步骤 5：测试 Kaggle API..."
if [ -f ~/.kaggle/kaggle.json ]; then
    echo "✅ Kaggle API Token 已配置"
    chmod 600 ~/.kaggle/kaggle.json

    # 测试 API 连接
    echo ""
    echo "测试 API 连接..."
    kaggle datasets list -s "machine learning" --max-size 10 2>/dev/null | head -5

    echo ""
    echo "=========================================="
    echo " ✅ Kaggle API 配置成功！"
    echo "=========================================="
else
    echo "❌ Kaggle API Token 未找到"
    echo "请先下载 kaggle.json 文件"
fi

# 6. 创建 Kaggle 快捷命令脚本
echo ""
echo "步骤 6：创建常用 Kaggle 命令快捷脚本..."
cat > scripts/kaggle_commands.sh << 'EOF'
#!/bin/bash
# Kaggle 常用命令

# 下载数据集
# kaggle datasets download -d username/dataset-name -p ./data

# 下载竞赛数据
# kaggle competitions download -c competition-name -p ./data

# 提交竞赛结果
# kaggle competitions submit -c competition-name -f submission.csv -m "message"

# 查看竞赛信息
# kaggle competitions info -c competition-name

# 列出已下载的数据集
kaggle datasets list

# 搜索数据集
# kaggle datasets search "keyword"
EOF

chmod +x scripts/kaggle_commands.sh

echo ""
echo "=========================================="
echo " ✅ Kaggle 环境配置完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 下载 kaggle.json 到 ~/.kaggle/"
echo "2. 运行 'chmod 600 ~/.kaggle/kaggle.json'"
echo "3. 使用 Kaggle Notebook 免费 GPU："
echo "   https://www.kaggle.com/code"
echo ""
echo "Kaggle 免费 GPU 资源："
echo "  - Tesla T4 (16GB) 或 P100 (16GB)"
echo "  - 每周 30 小时 GPU 时间"
echo "  - 单次会话最长 9 小时"
echo ""
