# -*- coding: utf-8 -*-
"""CLI 主入口：训练、恢复训练、推理。

用法：
    # 训练（英文数据集）
    python main.py --train --data data/tinystories_en --epoch 10

    # 训练（中文数据集）
    python main.py --train --data data/tinystories_zh --epoch 10

    # 恢复训练
    python main.py --train --resume checkpoints/ckpt_path

    # 推理
    python main.py --infer --checkpoint checkpoints/model_best.pt --prompt "Once upon a time"

    # 查看配置
    python main.py --config
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

# 确保项目根目录在 sys.path 中
_framework_dir = Path(__file__).parent
if str(_framework_dir) not in sys.path:
    sys.path.insert(0, str(_framework_dir))

from src.datasets import create_dataloaders
from src.infer import batch_inference, generate_text
from src.models import create_model
from src.train import Trainer
from src.utils import (
    create_tokenizer,
    create_tokenizer_from_file,
    seed_everything,
    setup_logging,
    strip_compile_prefix,
)


# ===== 配置加载 =====

def load_config(config_path: str = "config.yaml") -> dict:
    """加载 YAML 配置文件。

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logging.info(f"配置文件已加载: {config_path}")
    return config


# ===== 设备检测 =====

def get_device() -> torch.device:
    """检测可用的计算设备。

    Returns:
        CUDA 设备（如可用），否则 CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"使用 GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        logging.info("使用 CPU（建议使用 GPU 加速训练）")
    return device


# ===== 主函数 =====

def main():
    """主入口：解析命令行参数并执行训练或推理。"""
    parser = argparse.ArgumentParser(description="Mini LLM - 从零训练语言模型")

    # 模式选择
    parser.add_argument("--train", action="store_true", help="训练模式")
    parser.add_argument("--infer", action="store_true", help="推理模式")
    parser.add_argument("--config", action="store_true", help="显示配置文件")

    # 数据参数
    parser.add_argument("--data", type=str, default=None, help="数据目录路径")
    parser.add_argument("--epoch", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小")

    # 模型参数
    parser.add_argument("--checkpoint", type=str, default=None, help="检查点路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")

    # 推理参数
    parser.add_argument("--prompt", type=str, default=None, help="推理用的提示文本")
    parser.add_argument("--output_path", type=str, default=None, help="推理结果输出路径")

    # 配置
    parser.add_argument("--config_path", type=str, default="configs/ts_zh.yaml", help="配置文件路径")

    args = parser.parse_args()
    config_stem = Path(args.config_path).stem  # e.g., configs/ts_zh.yaml → ts_zh

    # ===== 再加载完整配置 =====
    config = load_config(args.config_path)

    # 命令行参数覆盖配置
    if args.epoch is not None:
        config["train"]["max_epochs"] = args.epoch
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.output_path is not None:
        config["infer"]["output_path"] = args.output_path

    # ===== 用 config_stem 填充未指定的路径 =====
    config.setdefault("paths", {})
    config["paths"].setdefault("log_dir", f"logs/{config_stem}")
    config["paths"].setdefault("run_dir", f"runs/{config_stem}")
    config["paths"].setdefault("checkpoint_dir", f"checkpoints/{config_stem}")
    config["paths"].setdefault("output_dir", f"output/{config_stem}")

    # ===== 显示配置 =====
    if args.config:
        print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
        return

    # ===== 设置随机种子 =====
    seed = config["train"].get("seed", 42)
    seed_everything(seed)

    log_dir = config["paths"]["log_dir"]
    log_file = os.path.join(log_dir, "training.log" if args.train else ("infer.log" if args.infer else None))
    setup_logging(log_file=log_file)

    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("Mini LLM - 从零训练语言模型")
    logger.info("=" * 50)

    # ===== 检测设备 =====
    device = get_device()

    # ===== 训练模式 =====
    if args.train:
        data_dir = args.data or config["paths"]["data_dir"]
        checkpoint_dir = config["paths"]["checkpoint_dir"]

        logger.info(f"数据目录: {data_dir}")

        # 根据配置选择分词器类型
        data_cfg = config.get("data", {})
        tokenizer_type = data_cfg.get("tokenizer_type", "char")
        logger.info(f"分词器类型: {tokenizer_type}")
        tokenizer = create_tokenizer(tokenizer_type)

        # 创建数据加载器（会自动保存词表到 data_dir/tokenizer.json）
        dataloaders = create_dataloaders(
            data_dir=data_dir,
            tokenizer=tokenizer,
            max_seq_len=config["model"]["max_seq_len"],
            min_seq_len=data_cfg.get("min_seq_len"),
            batch_size=config["train"]["batch_size"],
            num_workers=config["train"]["num_workers"],
            seed=seed,
            max_vocab_size=data_cfg.get("max_vocab_size", 5000),
            min_freq=data_cfg.get("min_freq", 1),
        )

        # 更新配置中的词表大小
        config["model"]["vocab_size"] = tokenizer.vocab_size

        # 创建模型
        model = create_model(config)
        model = model.to(device)

        # 创建训练器
        trainer = Trainer(
            model=model,
            train_loader=dataloaders["train"],
            valid_loader=dataloaders["valid"],
            config=config,
            device=device,
            run_dir=config["paths"]["run_dir"],
            checkpoint_dir=checkpoint_dir,
        )

        # 恢复训练（复用前面已预加载的检查点数据）
        if args.resume:
            success = trainer.load_checkpoint(args.resume)
            if not success:
                logger.warning("检查点加载失败，从头开始训练")

        # 开始训练
        trainer.train()

        logger.info("训练完成！")

    # ===== 推理模式 =====
    if args.infer:
        checkpoint_path = args.checkpoint or os.path.join(config["paths"]["checkpoint_dir"], "model_best.pt")
        if not os.path.exists(checkpoint_path):
            logger.error("未指定检查点路径，且默认路径不存在")
            logger.error("请使用 --checkpoint 参数指定检查点路径")
            return

        logger.info(f"加载检查点: {checkpoint_path}")

        # 加载检查点获取配置
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_config = checkpoint.get("config", config)

        # 创建模型
        model = create_model(saved_config)
        state_dict = strip_compile_prefix(checkpoint["model_state_dict"])
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        # 加载分词器（从数据目录）
        data_dir = args.data or config["paths"]["data_dir"]
        tp = os.path.join(data_dir, "tokenizer.json")

        tokenizer = None
        if os.path.exists(tp):
            tokenizer = create_tokenizer_from_file(tp)

        if tokenizer is None:
            logger.error("未找到分词器文件（tokenizer.json），请确保训练时已保存")
            return

        # 准备 prompt
        prompts = []

        if args.prompt:
            # 直接从命令行参数获取 prompt
            prompts = [args.prompt]
            logger.info(f"推理 prompt: {args.prompt}")
        elif data_dir:
            # 从测试数据目录加载
            from src.datasets import load_texts_from_directory
            test_texts = load_texts_from_directory(data_dir)
            # 取每个文本的前 20 个字符作为 prompt
            prompts = [text[:min(20, len(text))] for text in test_texts]
            logger.info(f"从 {data_dir} 加载了 {len(prompts)} 个测试 prompt")
        else:
            # 使用默认 prompt
            prompts = ["从前有座山，", "人工智能是一种", "深度学习是"]
            logger.info("使用默认 prompt")

        # 批量推理
        output_path = args.output_path or os.path.join(
            config["paths"]["output_dir"], "results.csv"
        )
        results = batch_inference(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            config=config,
            device=device,
            output_path=output_path,
        )

        # 打印结果
        print("\n" + "=" * 60)
        print("推理结果")
        print("=" * 60)
        for r in results:
            print(f"Prompt:   {r['prompt']}")
            print(f"生成:     {r['generated']}")
            print("-" * 60)

        logger.info(f"推理完成！结果已保存至: {output_path}")

    # 如果没有指定模式，显示帮助
    if not args.train and not args.infer and not args.config:
        parser.print_help()


if __name__ == "__main__":
    main()
