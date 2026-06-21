"""推理模块：文本生成与评估。

支持多种解码策略：
- 贪心解码 (greedy)
- 温度采样 (temperature)
- Top-k 采样 (topk)
- Top-p Nucleus 采样 (topp)

支持批量推理，结果保存为 CSV 格式。
"""

import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils import BaseTokenizer

logger = logging.getLogger(__name__)


def generate_text(
    model: nn.Module,
    tokenizer: BaseTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    strategy: str = "topp",
    device: torch.device = None,
) -> str:
    """根据 prompt 生成文本。

    Args:
        model: 训练好的模型
        tokenizer: 分词器
        prompt: 提示文本
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度
        top_k: Top-k 采样的 k 值
        top_p: Top-p 采样的 p 值
        strategy: 解码策略 (greedy/temperature/topk/topp)
        device: 计算设备

    Returns:
        生成的完整文本
    """
    if device is None:
        device = next(model.parameters()).device

    # 编码 prompt
    prompt_ids = tokenizer.encode(prompt, add_special=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # 根据策略设置生成参数
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": tokenizer.eos_id,
        "pad_token_id": tokenizer.pad_id,
    }

    if strategy == "greedy":
        # 贪心：temperature -> 0，无采样
        gen_kwargs["temperature"] = 1.0
        gen_kwargs["top_k"] = 1
        gen_kwargs["top_p"] = None
        logger.debug("解码策略: 贪心")
    elif strategy == "temperature":
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_k"] = None
        gen_kwargs["top_p"] = None
        logger.debug(f"解码策略: 温度采样 (T={temperature})")
    elif strategy == "topk":
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_k"] = top_k
        gen_kwargs["top_p"] = None
        logger.debug(f"解码策略: Top-k 采样 (k={top_k}, T={temperature})")
    elif strategy == "topp":
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_k"] = top_k
        gen_kwargs["top_p"] = top_p
        logger.debug(f"解码策略: Top-p 采样 (p={top_p}, k={top_k}, T={temperature})")
    else:
        raise ValueError(f"未知的解码策略: {strategy}，支持: greedy/temperature/topk/topp")

    # 生成
    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kwargs)

    # 解码
    generated_ids = output_ids[0].tolist()
    full_text = tokenizer.decode(generated_ids, skip_special=True)

    return full_text


def batch_inference(
    model: nn.Module,
    tokenizer: BaseTokenizer,
    prompts: List[str],
    config: dict,
    device: torch.device,
    output_path: str = "output/results.csv",
) -> List[Dict[str, str]]:
    """批量推理：对多个 prompt 生成文本。

    Args:
        model: 训练好的模型
        tokenizer: 分词器
        prompts: 提示文本列表
        config: 推理配置字典
        device: 计算设备
        output_path: 结果保存路径

    Returns:
        结果字典列表 [{"prompt": ..., "generated": ...}, ...]
    """
    infer_config = config.get("infer", {})

    results = []
    pbar = tqdm(prompts, desc="推理", unit="prompt")
    for prompt in pbar:
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=infer_config.get("max_new_tokens", 100),
            temperature=infer_config.get("temperature", 0.8),
            top_k=infer_config.get("top_k", 40),
            top_p=infer_config.get("top_p", 0.9),
            strategy=infer_config.get("strategy", "topp"),
            device=device,
        )
        results.append({"prompt": prompt, "generated": generated})

        logger.info(f"Prompt: {prompt}")
        logger.info(f"生成: {generated}")
        logger.info("-" * 40)
    pbar.close()

    # 保存结果
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "generated"])
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"推理结果已保存至: {output_path}")
    return results
