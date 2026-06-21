# 项目构建计划

> 本项目从零开始完整训练一个小型 LLM，面向人工智能学生入门 LLM 的项目。

## 已完成

### 第 1 阶段：项目骨架搭建
- [x] 创建 `framework/` 目录结构
- [x] 创建 `src/` 源码目录（`__init__.py`）
- [x] 创建 `config.yaml` 配置文件（模型/训练/推理/路径）
- [x] 创建数据目录 `data/data_train/` 和 `data/data_test/`
- [x] 创建 `checkpoints/`、`runs/`、`logs/`、`output/` 目录

### 第 2 阶段：工具模块 (`src/utils.py`)
- [x] `CharTokenizer`：字符级分词器，支持中文+英文
  - [x] 词表构建（特殊 token + 频次过滤）
  - [x] encode/decode
  - [x] 词表保存/加载
- [x] `compute_perplexity()`：困惑度计算
- [x] `count_parameters()`：参数量统计
- [x] `seed_everything()`：随机种子设置

### 第 3 阶段：模型模块 (`src/models.py`)
- [x] `SinusoidalPositionalEmbedding`：正弦位置编码
- [x] `MultiHeadCausalAttention`：多头因果自注意力
- [x] `FeedForward`：SwiGLU 激活的前馈网络
- [x] `TransformerBlock`：Pre-LayerNorm Transformer 块
- [x] `DecoderOnlyTransformer`：主体模型（GPT-2 风格）
  - [x] 权重绑定（Embedding + LM Head）
  - [x] 前向传播 + 损失计算
  - [x] `generate()`：自回归生成（贪心/温度/Top-k/Top-p）

### 第 4 阶段：数据模块 (`src/datasets.py`)
- [x] `TextDataset`：文本分块数据集
- [x] `load_texts_from_directory()`：从目录加载 .txt 文件
- [x] `create_dataloaders()`：创建训练/验证 DataLoader

### 第 5 阶段：训练模块 (`src/train.py`)
- [x] `get_cosine_schedule_with_warmup()`：带预热的余弦退火 LR
- [x] `Trainer`：训练器
  - [x] 训练循环（损失/困惑度/学习率日志）
  - [x] 验证循环
  - [x] 梯度裁剪
  - [x] 检查点保存/加载
  - [x] TensorBoard 日志
  - [x] 最佳模型自动保存

### 第 6 阶段：推理模块 (`src/infer.py`)
- [x] `generate_text()`：文本生成函数
- [x] `batch_inference()`：批量推理 + CSV 保存

### 第 7 阶段：CLI 主入口 (`main.py`)
- [x] 训练命令：`--train --data data/data_train --epoch 10`
- [x] 恢复训练：`--train --resume checkpoints/model_best.pt`
- [x] 推理命令：`--infer --checkpoint ... --output_path ...`
- [x] 配置查看：`--config`

### 第 8 阶段：训练数据集（双语 TinyStories）
- [x] **英文**: [roneneldan/TinyStories](https://hf-mirror.com/datasets/roneneldan/TinyStories) — GPT 生成的儿童故事
- [x] **中文**: [adam89/TinyStoriesChinese](https://hf-mirror.com/datasets/adam89/TinyStoriesChinese) — 中文翻译版
- [x] 数据下载脚本 `data/download_datasets.py`（通过 hf-mirror.com 下载）
- [x] 自动划分训练/验证集

### 第 9 阶段：安装与验证
- [x] 依赖安装（torch, numpy, pandas, yaml, tensorboard, datasets）
- [x] 模块导入测试
- [x] 端到端流水线测试（分词→数据→模型→训练→推理）

## 数据统计

| 文件 | 大小 | 故事数 |
|------|------|--------|
| `tinystories_en.txt` | 3.84 MB | 4,732 |
| `tinystories_zh.txt` | 3.53 MB | 4,768 |
| `valid.txt` | 0.39 MB | 500 |
| **总计** | **7.76 MB** | **10,000** |

- 总 token 数（字符级）：约 562 万
- 词表大小：~2809（中英文混合）

## 模型配置

| 参数 | 值 | 说明 |
|------|-----|------|
| vocab_size | ~2800（自动构建） | 字符级分词 |
| hidden_dim | 256 | 嵌入维度 |
| num_layers | 6 | Transformer 层数 |
| num_heads | 8 | 注意力头数 |
| max_seq_len | 256 | 最大序列长度 |
| dropout | 0.1 | Dropout 概率 |
| 参数量 | ~6.5M | 百万级参数 |

## 使用方式

```bash
cd framework

# 完整训练（10 epoch）
python main.py --train --data data/data_train --epoch 10

# 推理
python main.py --infer --checkpoint checkpoints/model_best.pt --prompt "Once upon a time"

# 恢复训练
python main.py --train --resume checkpoints/model_best.pt

# 重新下载数据
cd data && python download_datasets.py --max_stories 10000
```

## 硬件要求

- **推荐**: NVIDIA GPU 6GB+ VRAM（如 RTX 3060 Laptop）
- **最低**: CPU（训练较慢，可减小模型）
- **存储**: ~1GB
