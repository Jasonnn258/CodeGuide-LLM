# ModelScope 免费算力使用指南

本文档介绍如何使用 ModelScope 的免费 GPU 算力。

## 目录

- [免费算力介绍](#免费算力介绍)
- [快速开始](#快速开始)
- [使用方法](#使用方法)
- [ModelScope SDK 使用](#modelscope-sdk-使用)
- [常见问题](#常见问题)

---

## 免费算力介绍

### ModelScope 提供的免费资源

| 资源类型 | 说明 | 限制 |
|---------|------|------|
| **GPU 算力** | A10 24GB 显存 | 新用户赠送一定额度 |
| **CPU 算力** | 不限时长使用 | 无限制 |
| **存储空间** | 临时存储 | 实例关闭后清除 |

### 计费规则

- GPU 按小时计费，从赠送额度中扣除
- 单次实例最长运行 10 小时
- 空闲 1 小时自动关闭
- 建议不使用时手动关闭实例

---

## 快速开始

### 步骤 1：注册账号

1. 访问 [ModelScope](https://www.modelscope.cn)
2. 注册并登录账号
3. 完成阿里云授权（获取免费 GPU 时长）

### 步骤 2：启动 Notebook

1. 登录 ModelScope
2. 点击右上角 "Notebook快速开发"
3. 选择 "PAI-DSW" 或 "阿里云弹性加速计算EAIS"
4. 选择 GPU 环境（A10 推荐）
5. 点击启动

### 步骤 3：开始使用

- Notebook 环境已预装 ModelScope SDK
- 可直接运行代码
- 支持 Jupyter Notebook 和 Terminal

---

## 使用方法

### 方式 1：使用预置模型

ModelScope Notebook 已预装常用环境，可直接使用：

```python
from modelscope import snapshot_download

# 下载模型
model_dir = snapshot_download('Qwen/Qwen2.5-Coder-7B-Instruct')
```

### 方式 2：训练自定义模型

参考 CodeGuide-LLM 项目：

```python
# 安装项目依赖
!pip install -q -r requirements.txt

# 配置 API Keys
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

# 运行训练
!python scripts/train_sft.py --config configs/train_config.yaml
```

### 方式 3：使用免费 GPU 运行推理

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 检查 GPU
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# 加载模型
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 推理
inputs = tokenizer("def quick_sort(", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## ModelScope SDK 使用

### 安装 SDK

```bash
pip install modelscope
```

### 登录账号

```python
import modelscope
modelscope.login()
```

### 下载模型

```python
from modelscope import snapshot_download

# 下载到默认目录
model_dir = snapshot_download('Qwen/Qwen2.5-Coder-7B-Instruct')

# 下载到指定目录
model_dir = snapshot_download(
    'Qwen/Qwen2.5-Coder-7B-Instruct',
    cache_dir='./models'
)
```

### 下载数据集

```python
from modelscope.msdatasets import MsDataset

# 加载数据集
ds = MsDataset.load('code_contest')
print(ds)
```

### 模型推理

```python
from modelscope import pipeline

# 创建推理管道
p = pipeline('text-generation', 'Qwen/Qwen2.5-Coder-7B-Instruct')

# 推理
result = p({'text': 'def hello_world():'})
print(result)
```

---

## 常见问题

### Q1: 如何获取免费 GPU 算力？

**答：**
1. 登录 ModelScope 账号
2. 完成阿里云授权
3. 即可获得免费 GPU 时长
4. 在 "我的Notebook" 中查看剩余额度

### Q2: 空闲自动关闭怎么办？

**答：**
1. 避免长时间空闲
2. 可以在代码中添加保持存活的逻辑
3. 及时手动关闭不需要的实例
4. 在 "我的Notebook" 中管理运行中的实例

### Q3: 如何上传本地代码到 Notebook？

**答：**
1. **方式 1**：在 Notebook 中直接创建文件
2. **方式 2**：使用 SCP 上传
   ```bash
   scp local_file root@instance:/workspace/
   ```
3. **方式 3**：使用 Git 克隆
   ```bash
   git clone https://github.com/your/repo.git
   ```

### Q4: 显存不足怎么办？

**答：**
1. 使用量化模型（4-bit/8-bit）
2. 减小 batch size
3. 使用更小的模型（如 7B 而非 72B）
4. 使用梯度累积

### Q5: 如何保存训练结果？

**答：**
1. **及时保存**：定期保存 checkpoint
2. **下载到本地**：训练完成后下载文件
3. **上传到云存储**：保存到阿里云 OSS

```python
# 在训练代码中添加保存逻辑
import shutil
shutil.copy('checkpoint.pt', '/workspace/checkpoint.pt')
```

### Q6: 关闭实例后还能恢复吗？

**答：**
- 关闭后实例状态会清除
- 建议在关闭前保存所有重要数据
- 重新启动会创建新实例

---

## 最佳实践

### 1. 资源管理

```python
# 定期保存 checkpoint
if step % 100 == 0:
    save_checkpoint(model, optimizer, step)

# 监控 GPU 使用
import torch
print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
```

### 2. 代码管理

```bash
# 在 Notebook 中克隆项目
!git clone https://github.com/your/CodeGuide-LLM.git
%cd CodeGuide-LLM

# 安装依赖
!pip install -q -r requirements.txt
```

### 3. 数据管理

```python
# 下载数据集到工作目录
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('code_contest', cache_dir='/workspace/data')
```

---

## 参考链接

- [ModelScope 官网](https://www.modelscope.cn)
- [ModelScope 文档](https://www.modelscope.cn/docs)
- [ModelScope Notebook 介绍](https://www.modelscope.cn/docs/notebook/intro)
- [ModelScope SDK](https://www.modelscope.cn/docs/%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B)

---

## 附录：ModelScope CLI 命令

```bash
# 登录
modelscope login

# 查看帮助
modelscope --help

# 下载模型
modelscope download --model Qwen/Qwen2.5-Coder-7B-Instruct

# 上传模型
modelscope upload --path ./your_model --model-id your/model-name
```
