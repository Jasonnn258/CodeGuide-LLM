# Google Colab 免费 GPU 使用指南

本文档介绍如何使用 Google Colab 的免费 GPU 算力进行 CodeGuide-LLM 项目开发。

## 目录

- [免费 GPU 资源介绍](#免费-gpu-资源介绍)
- [快速开始](#快速开始)
- [环境配置](#环境配置)
- [项目部署](#项目部署)
- [使用技巧](#使用技巧)
- [常见问题](#常见问题)

---

## 免费 GPU 资源介绍

### Colab 提供的免费资源

| 资源类型 | 说明 | 限制 |
|---------|------|------|
| **GPU 算力** | Tesla T4 (16GB) 或 P100 (16GB) | 单次会话最长 12 小时 |
| **CPU 算力** | 2 vCPU | 无特殊限制 |
| **内存** | 12-16 GB | 无特殊限制 |
| **存储空间** | 临时存储 + Google Drive | 临时存储会话结束后清除 |

### 限制说明

- **会话时长**：单次最长 12 小时
- **每日限额**：约 24 小时（动态调整）
- **闲置回收**：闲置 30 分钟自动断开
- **资源分配**：新用户更容易获得 GPU

---

## 快速开始

### 步骤 1：访问 Colab

访问 https://colab.research.google.com

### 步骤 2：创建新笔记本

1. 点击 **"新建笔记本"**
2. 选择 **Python 3** 作为运行时类型

### 步骤 3：启用 GPU

1. 点击顶部菜单：**运行时** → **更改运行时类型**
2. 在 **硬件加速器** 下拉菜单中选择 **GPU**
3. 点击 **保存**

### 步骤 4：验证 GPU

在笔记本中运行：

```python
import torch

print(f"GPU 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

---

## 环境配置

### 1. 挂载 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. 安装依赖

```python
# 进入项目目录
%cd /content/drive/MyDrive/CodeGuide-LLM

# 安装依赖
!pip install -q -r requirements.txt
```

### 3. 验证环境

```python
# 检查 GPU
!nvidia-smi

# 检查 PyTorch
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"GPU 可用: {torch.cuda.is_available()}")

# 检查 Python 版本
!python --version
```

---

## 项目部署

### 1. 克隆项目（首次使用）

```python
# 克隆代码库
!git clone https://github.com/yourusername/CodeGuide-LLM.git /content/drive/MyDrive/CodeGuide-LLM

# 进入项目目录
%cd /content/drive/MyDrive/CodeGuide-LLM
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
!cat /proc/meminfo | grep MemTotal

# 查看 CPU 信息
!cat /proc/cpuinfo | grep "model name"
```

### 2. 性能优化

- **批量大小调整**：从 8 开始，逐步增加直到 GPU 内存占用 80% 左右
- **混合精度训练**：使用 FP16 加速
- **梯度累积**：设置 `gradient_accumulation_steps` 为 4-8
- **数据加载优化**：使用 `num_workers` > 0

### 3. 防止断开连接

- 定期保存 checkpoint
- 使用 `drive.mount` 持久化数据
- 避免长时间闲置

### 4. 模型保存

```python
# 保存模型到 Google Drive
!cp -r checkpoints/* /content/drive/MyDrive/checkpoints/

# 下载模型到本地
from google.colab import files
files.download('checkpoints/model.pt')
```

---

## 常见问题

### Q1: GPU 不可用怎么办？

**解决方案：**
1. 检查是否已启用 GPU（运行时 → 更改运行时类型）
2. 重启运行时（运行时 → 重启运行时）
3. 等待一段时间后再尝试

### Q2: 会话断开怎么办？

**解决方案：**
1. 定期保存 checkpoint
2. 使用 `drive.mount` 持久化数据
3. 重新连接并从上次保存的 checkpoint 继续训练

### Q3: 内存不足怎么办？

**解决方案：**
1. 减小 batch size
2. 使用更小的模型
3. 清理未使用的变量
4. 使用梯度累积

### Q4: 如何上传本地文件？

**解决方案：**
```python
from google.colab import files
uploaded = files.upload()
```

### Q5: 如何下载文件？

**解决方案：**
```python
from google.colab import files
files.download('file.txt')
```

---

## 最佳实践

### 1. 工作流建议

1. **本地开发**：在本地编写和测试代码
2. **上传到 Drive**：将代码上传到 Google Drive
3. **Colab 训练**：在 Colab 中使用 GPU 进行训练
4. **下载结果**：将训练结果下载到本地

### 2. 代码示例

```python
# 完整的 Colab 训练脚本

# 1. 挂载 Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. 进入项目目录
%cd /content/drive/MyDrive/CodeGuide-LLM

# 3. 安装依赖
!pip install -q -r requirements.txt

# 4. 检查 GPU
import torch
print(f"GPU 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")

# 5. 运行训练
!python scripts/train_sft.py --config configs/train_config.yaml

# 6. 保存结果
!mkdir -p /content/drive/MyDrive/results
!cp -r checkpoints/* /content/drive/MyDrive/results/

print("训练完成！")
```

---

## 高级技巧

### 1. 使用 TPU（可选）

如果你的模型支持 TPU，可以在运行时类型中选择 TPU 获得更快的训练速度。

### 2. 持久化环境

创建一个 `requirements.txt` 文件，包含所有依赖：

```
torch>=2.0.0
transformers>=4.40.0
datasets>=2.0.0
peft>=0.10.0
accelerate>=0.20.0
bitsandbytes>=0.40.0
```

### 3. 自动重连脚本

在浏览器控制台中运行以下代码，防止 Colab 断开：

```javascript
function keepAlive() {
  console.log("保持连接...");
  document.querySelector("colab-toolbar-button#connect").click();
}
setInterval(keepAlive, 60000);
```

---

## 参考链接

- [Google Colab 官网](https://colab.research.google.com)
- [Colab 文档](https://colab.research.google.com/notebooks/intro.ipynb)
- [PyTorch 文档](https://pytorch.org/docs/stable/index.html)
- [Hugging Face 文档](https://huggingface.co/docs)

---

## 附录：Colab 命令速查表

| 命令 | 功能 |
|------|------|
| `!nvidia-smi` | 查看 GPU 信息 |
| `!pip install package` | 安装包 |
| `%cd path` | 切换目录 |
| `%pwd` | 显示当前目录 |
| `!ls -la` | 列出文件 |
| `!cat file` | 查看文件内容 |
| `from google.colab import drive; drive.mount('/content/drive')` | 挂载 Drive |
| `from google.colab import files; files.download('file')` | 下载文件 |
| `from google.colab import files; uploaded = files.upload()` | 上传文件 |

---

## 故障排除

### 常见错误及解决方案

| 错误信息 | 可能原因 | 解决方案 |
|---------|---------|---------|
| `CUDA out of memory` | 内存不足 | 减小 batch size |
| `Session crashed for unknown reasons` | 资源耗尽 | 重启运行时 |
| `Drive not mounted` | 未授权 | 重新运行 drive.mount() |
| `Module not found` | 依赖未安装 | 运行 pip install |

---

现在你已经掌握了 Google Colab 免费 GPU 的使用方法，可以开始进行 CodeGuide-LLM 项目的训练了！
