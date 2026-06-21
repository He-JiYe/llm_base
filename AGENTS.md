# 训练 LLM - AGENTS.md

> 本文件面向 AI Coding Agent 读者。项目所有注释和文档均使用中文，Agent 在处理代码时应保持中文注释风格。

---

## 项目概述

本项目从零开始完整训练一个小型 LLM，面向人工智能学生入门 LLM 的项目。整个项目包括：数据收集 -> 数据预处理 -> 模型建立 -> 训练监控 -> Post Traing 的完整过程。

### 目录结构

```
 ├── framework/              # 项目主代码
 │   ├── data/               # 训练数据集（小型数据集）
 │   │   ├── data1/          # 数据集 1
 │   │   ├── data2/          # 数据集 2
 │   │   └── ...
 │   ├── src/                # 自主生成的模型与训练代码
 │   │   ├── models.py       # Transformer Decoder Only Models
 │   │   ├── datasets.py     # 数据集和数据加载器
 │   │   ├── train.py        # 训练入口
 │   │   ├── infer.py        # 推理入口
 │   │   └── utils.py        # 工具函数：评估指标
 │   ├── checkpoints/        # 模型参数（加载点）
 │   ├── runs/               # Tensorboard 文件
 │   ├── logs/               # 训练日志
 │   ├── output/             # 测试结果
 │   ├── main.py             # CLI主入口
 │   └── config.yaml         # YAML配置文件示例
 ├── .gitignore              # Git 文件
 ├── .python-version         # python 3.11
 ├── .venv/                  # Python虚拟环境（已初始化）
 ├── pyproject.toml          # 项目环境配置文件
 └── plan.md                 # 项目构建计划
```

## 环境

- **编程语言**: Python 3.11
- **深度学习框架**: PyTorch >= 2.0.0
- **科学计算**: numpy >= 1.24.0, scipy >= 1.10.0, pandas >= 2.0.0
- **配置**: PyYAML >= 6.0
- **环境管理**: 使用 uv 工具管理 `.venv/` 目录下的虚拟环境

---

## 构建与运行命令

### 环境准备

虚拟环境已存在于 `.venv/` 目录下。激活方式：

```bash
source .venv/Scripts/activate
```

依赖安装（如需要）：
```bash
uv add -r framework/requirements.txt
```

### 运行主程序

```bash
cd framework

# 指定数据集完整训练
python main.py --train --data data/data_train --epoch 10

# 恢复之前运行
python main.py --train --resume checkpoints/ckpt_path 

# 推理
python main.py --infer --data data/data_test --checkpoint checkpoints/ckpt_path --output_path output/ckpt_path.csv
```

---

## 代码风格指南

### 注释语言
- **所有注释和文档字符串必须使用中文**。这是项目统一规范，不可改为英文。
- 代码中的字符串输出（如日志、用户提示）也使用中文。

### 命名规范
- 类名：`PascalCase`
- 函数/变量：`snake_case`
- 常量：`UPPER_SNAKE_CASE`
- 私有方法：以 `_` 开头

### 文档字符串风格
- 使用 `"""三重双引号"""` 包裹模块和类文档字符串
- 函数文档字符串包含：功能描述、Args、Returns
- 示例：
  ```python
  def normalize_adj(adj):
      """对称归一化邻接矩阵: D^(-1/2) * (A + I) * D^(-1/2)

      Args:
          adj: 邻接矩阵, scipy稀疏矩阵或numpy数组或torch.Tensor

      Returns:
          归一化后的邻接矩阵, torch.Tensor (N, N)
      """
  ```

### 代码组织
- 每个 `.py` 文件顶部有模块级文档字符串，说明职责
- 使用 `===== 章节标题 =====` 风格的注释分隔不同功能区块
- 导入顺序：标准库 → 第三方库 → 项目内部模块
- 避免循环导入，必要时使用延迟导入（函数内 `import`）

### 日志规范
- 使用 `logging` 模块，不使用 `print`
- 训练/推理代码中使用 `logging.info()` / `logging.warning()`
- 日志格式：`[%(asctime)s] %(levelname)s - %(message)s`

---

## 关键提示

1. **所有修改保持中文注释**。不要英文化现有注释。
2. **配置驱动**: 多数超参数可通过 `config.yaml` 或命令行参数调整，优先尝试配置修改而非代码重写。
3. **使用适当大小的数据集和参数了**: 训练设备为搭载有 3060 Laptop GPU 的笔记本，避免选取过大的数据集和模型参数。
4. **教学为主**：保证整个项目的过程完整，并辅佐对应的 MarkDown 文件对模型、功能进行解释，并展示实验结果。 