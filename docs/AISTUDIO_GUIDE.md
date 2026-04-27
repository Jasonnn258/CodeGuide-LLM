# 百度 AI Studio 星河社区免费 GPU 使用指南

本文档介绍如何使用百度 AI Studio 星河社区的免费 GPU 算力进行 CodeGuide-LLM 项目开发。

## 目录

- [免费 GPU 资源介绍](#免费-gpu-资源介绍)
- [快速开始](#快速开始)
- [环境配置](#环境配置)
- [项目部署](#项目部署)
- [使用技巧](#使用技巧)
- [常见问题](#常见问题)

---

## 免费 GPU 资源介绍

### AI Studio 提供的免费资源

| 资源类型 | 说明 | 限制 |
|---------|------|------|
| **GPU 算力** | Tesla V100 (16GB) | 每天 8-12 小时 |
| **CPU 算力** | 4 vCPU | 无特殊限制 |
| **内存** | 16 GB | 无特殊限制 |
| **存储空间** | 临时存储 + 数据集存储 | 临时存储会话结束后清除 |

### 限制说明

- **使用时间**：每天 8-12 小时免费 GPU 时间
- **资源分配**：需要实名认证
- **申请难度**：白天可能需要排队，晚上更容易获得
- **环境重置**：每次启动环境会重置，需要重新安装依赖

---

## 快速开始

### 步骤 1：访问 AI Studio

访问 https://aistudio.baidu.com

### 步骤 2：登录账号

1. 使用百度账号登录
2. 完成实名认证（需要身份证信息）

### 步骤 3：创建项目

1. 点击 **"创建项目"**
2. 选择 **"Notebook 项目"**
3. 填写项目信息：
   - 项目名称：CodeGuide-LLM
   - 项目描述：代码指南大模型训练
   - 运行环境：选择 **PaddlePaddle 2.5+**
   - 资源配置：**GPU V100**（免费）

### 步骤 4：启动环境

1. 点击 **"启动环境"**
2. 等待环境初始化（约 1-2 分钟）
3. 进入 Notebook 界面

---

## 环境配置

### 1. 安装依赖

在 Notebook 中运行：

```python
# 安装 PyTorch
!pip install torch torchvision -i https://mirror.baidu.com/pypi/simple

# 安装项目依赖
!pip install -q -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```

### 2. 验证环境

```python
# 检查 GPU
!nvidia-smi

# 检查 PyTorch
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"GPU 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 检查 Python 版本
!python --version
```

### 3. 上传项目文件

1. 在 AI Studio 界面左侧点击 **"文件"**
2. 点击 **"上传文件"**
3. 选择本地的 `CodeGuide-LLM` 项目文件
4. 等待上传完成

---

## 项目部署

### 1. 克隆项目（可选）

如果没有上传文件，可以直接克隆：

```python
# 克隆代码库
!git clone https://github.com/yourusername/CodeGuide-LLM.git

# 进入项目目录
%cd CodeGuide-LLM
```

### 2. 配置训练参数

编辑 `configs/train_config.yaml` 文件：

```yaml
training:
  batch_size: 8
  epochs: 10
  learning_rate: 1e-5
  gradient_accumulation_steps: 4
```

### 3. 运行训练

```python
# 运行 SFT 训练
!python scripts/train_sft.py --config configs/train_config.yaml

# 或运行 GRPO 训练
!python scripts/train_grpo.py --config configs/train_config.yaml
```

### 4. 监控训练

```python
# 查看 GPU 使用情况
!nvidia-smi

# 查看训练日志
!tail -n 100 logs/training.log
```

---

## 使用技巧

### 1. 资源监控

```python
# 实时监控 GPU
!nvidia-smi -l 1

# 查看内存使用
!free -h

# 查看 CPU 信息
!cat /proc/cpuinfo | grep "model name"
```

### 2. 性能优化

- **批量大小调整**：从 8 开始，逐步增加直到 GPU 内存占用 80% 左右
- **混合精度训练**：使用 FP16 加速
- **梯度累积**：设置 `gradient_accumulation_steps` 为 4-8
- **数据加载优化**：使用 `num_workers` > 0

### 3. 防止环境重置

- 定期保存 checkpoint 到数据集存储
- 使用 `!cp` 命令将重要文件复制到数据集目录
- 训练完成后及时下载结果

### 4. 模型保存

```python
# 保存模型到数据集目录
!mkdir -p /home/aistudio/data/checkpoints
!cp -r checkpoints/* /home/aistudio/data/checkpoints/

# 下载模型到本地
# 在 AI Studio 界面中，右键点击文件 → 下载
```

---

## 常见问题

### Q1: GPU 申请不到怎么办？

**解决方案：**
1. 尝试在晚上或凌晨申请
2. 提前预约 GPU 资源
3. 检查是否已完成实名认证

### Q2: 环境重置后依赖丢失怎么办？

**解决方案：**
1. 创建一个 `requirements.txt` 文件
2. 使用百度源加速安装：`!pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple`
3. 考虑使用持久化数据集存储依赖包

### Q3: 上传文件大小限制怎么办？

**解决方案：**
1. 分卷压缩文件
2. 使用百度云盘挂载
3. 直接克隆 GitHub 仓库

### Q4: 训练中断怎么办？

**解决方案：**
1. 检查是否达到每日 GPU 时间限制
2. 从上次保存的 checkpoint 继续训练
3. 调整训练参数减少内存使用

### Q5: 如何持久化数据？

**解决方案：**
1. 使用 `数据集` 选项卡上传和管理数据
2. 将重要文件保存到 `/home/aistudio/data/` 目录
3. 使用百度云盘进行长期存储

---

## 最佳实践

### 1. 工作流建议

1. **本地开发**：在本地编写和测试代码
2. **上传到 AI Studio**：将代码上传到 AI Studio
3. **AI Studio 训练**：使用免费 GPU 进行训练
4. **下载结果**：将训练结果下载到本地

### 2. 代码示例

```python
# 完整的 AI Studio 训练脚本

# 1. 检查环境
!nvidia-smi

# 2. 安装依赖
!pip install torch torchvision -i https://mirror.baidu.com/pypi/simple
!pip install -q -r requirements.txt -i https://mirror.baidu.com/pypi/simple

# 3. 检查 PyTorch
import torch
print(f"GPU 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")

# 4. 运行训练
!python scripts/train_sft.py --config configs/train_config.yaml

# 5. 保存结果
!mkdir -p /home/aistudio/data/results
!cp -r checkpoints/* /home/aistudio/data/results/

print("训练完成！")
```

---

## 高级技巧

### 1. 使用百度源加速

```python
# 使用百度源安装包
!pip install package -i https://mirror.baidu.com/pypi/simple

# 使用百度源安装 PyTorch
!pip install torch torchvision -i https://mirror.baidu.com/pypi/simple
```

### 2. 持久化环境

创建一个 `setup_env.sh` 脚本：

```bash
#!/bin/bash
# 环境设置脚本

# 安装依赖
pip install torch torchvision -i https://mirror.baidu.com/pypi/simple
pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple

# 检查环境
nvidia-smi
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

### 3. 数据集管理

- **上传数据集**：在 "数据集" 选项卡上传
- **挂载数据集**：在 Notebook 中直接访问 `/home/aistudio/data/`
- **使用公开数据集**：AI Studio 提供了丰富的公开数据集

---

## 参考链接

- [百度 AI Studio 官网](https://aistudio.baidu.com)
- [AI Studio 文档中心](https://aistudio.baidu.com/docs)
- [飞桨 PaddlePaddle 文档](https://www.paddlepaddle.org.cn/documentation)

---

## 附录：AI Studio 命令速查表

| 命令 | 功能 |
|------|------|
| `!nvidia-smi` | 查看 GPU 信息 |
| `!pip install package` | 安装包 |
| `%cd path` | 切换目录 |
| `%pwd` | 显示当前目录 |
| `!ls -la` | 列出文件 |
| `!cat file` | 查看文件内容 |
| `!cp src dst` | 复制文件 |
| `!mkdir -p dir` | 创建目录 |
| `!git clone repo` | 克隆仓库 |

---

## 故障排除

### 常见错误及解决方案

| 错误信息 | 可能原因 | 解决方案 |
|---------|---------|---------|
| `CUDA out of memory` | 内存不足 | 减小 batch size |
| `No module named 'torch'` | 依赖未安装 | 运行 `!pip install torch` |
| `Permission denied` | 权限不足 | 使用 `sudo` 或检查文件权限 |
| `Connection timeout` | 网络问题 | 使用百度源或检查网络连接 |

---

现在你已经掌握了百度 AI Studio 星河社区免费 GPU 的使用方法，可以开始进行 CodeGuide-LLM 项目的训练了！
