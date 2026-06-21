#!/usr/bin/env python3
"""中文数据预处理脚本。

对中文数据集做繁→简转换 + NFKC 标准化，
将清理后的数据保存到新目录，避免训练时每次重复预处理。

用法:
    # 对中文 TinyStories 进行预处理
    python prepare.py data/tinystories_zh

    # 指定输出目录
    python prepare.py data/tinystories_zh --output_dir data/tinystories_zh_clean

    # 原地覆盖（不保留原始文件）
    python prepare.py data/tinystories_zh --inplace

    # 试运行（不实际写入）
    python prepare.py data/tinystories_zh --dry_run
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加上级目录到路径，使 src 可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm
from src.utils import ChineseTextPreprocessor

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 需要处理的文件名
DATA_FILES = ["train.txt", "valid.txt", "test.txt"]


def preprocess_file(src_path: Path, dst_path: Path, desc: str) -> int:
    """对单个文件做中文预处理并保存。

    Args:
        src_path: 源文件路径
        dst_path: 目标文件路径
        desc: 日志描述

    Returns:
        处理的字符数
    """
    raw = src_path.read_text(encoding="utf-8")
    logger.info(f"{desc}: 读取 {src_path.name} ({len(raw):,} 字符)")

    # 分块预处理（避免超大文件一次性处理）
    chunk_size = 1_000_000
    total_chars = 0
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dst_path, "w", encoding="utf-8") as fout:
        with tqdm(
            total=len(raw),
            desc=f"预处理 {src_path.name}",
            unit="字符",
            unit_scale=True,
        ) as pbar:
            for i in range(0, len(raw), chunk_size):
                chunk = raw[i:i + chunk_size]
                cleaned = ChineseTextPreprocessor.preprocess(chunk)
                fout.write(cleaned)
                total_chars += len(chunk)
                pbar.update(len(chunk))

    src_size = src_path.stat().st_size
    dst_size = dst_path.stat().st_size
    logger.info(f"  → 已保存: {dst_path} ({dst_size / 1024 / 1024:.1f} MB)")
    logger.info(f"  → 压缩率: {src_size}/{dst_size} = {dst_size / src_size:.2%}" if src_size > 0 else "")
    return total_chars


def main():
    parser = argparse.ArgumentParser(description="中文数据预处理：繁转简 + NFKC 标准化")
    parser.add_argument("data_dir", type=str, help="中文数据目录路径")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录（默认: data_dir_prepared）"
    )
    parser.add_argument(
        "--inplace", action="store_true",
        help="原地覆盖，不保留原始文件"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="试运行，仅显示将如何处理"
    )
    parser.add_argument(
        "--files", type=str, nargs="*",
        default=DATA_FILES,
        help=f"要处理的文件名（默认: {' '.join(DATA_FILES)}）"
    )
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists():
        logger.error(f"数据目录不存在: {data_path}")
        sys.exit(1)

    # 确定输出目录
    if args.inplace:
        output_path = data_path
        logger.info("模式: 原地覆盖（不保留原始文件）")
    elif args.output_dir:
        output_path = Path(args.output_dir)
        logger.info(f"模式: 输出到指定目录: {output_path}")
    else:
        # 默认: data_dir -> data_dir_prepared
        output_path = data_path.parent / (data_path.name + "_prepared")
        logger.info(f"模式: 输出到: {output_path}")

    # 收集需要处理的文件
    files_to_process = []
    for fname in args.files:
        src = data_path / fname
        if src.exists():
            dst = output_path / fname
            files_to_process.append((src, dst))
        else:
            logger.warning(f"文件不存在，跳过: {src}")

    if not files_to_process:
        logger.error(f"在 {data_path} 中未找到可处理的文件: {args.files}")
        sys.exit(1)

    # 显示概览
    logger.info(f"数据目录: {data_path}")
    logger.info(f"输出目录: {output_path}")
    logger.info(f"待处理文件:")
    for src, dst in files_to_process:
        size_mb = src.stat().st_size / 1024 / 1024
        logger.info(f"  {src.name} ({size_mb:.1f} MB)")

    if args.dry_run:
        logger.info("试运行模式，不执行写入。使用以下命令实际执行:")
        logger.info(f"  python prepare.py {args.data_dir} " +
                    f"{'--inplace' if args.inplace else f'--output_dir {output_path}'}")
        return

    # 执行预处理
    total_files = 0
    total_chars = 0
    for src, dst in files_to_process:
        chars = preprocess_file(src, dst, f"[{total_files + 1}/{len(files_to_process)}]")
        total_files += 1
        total_chars += chars

    logger.info("=" * 50)
    logger.info(f"预处理完成!")
    logger.info(f"  处理文件: {total_files}")
    logger.info(f"  总字符数: {total_chars:,}")
    logger.info(f"  输出目录: {output_path}")
    logger.info("=" * 50)
    logger.info("提示: 训练时可在 config.yaml 中设置 chinese_normalize: false")
    logger.info(f"       python main.py --train --data {output_path}")


if __name__ == "__main__":
    main()
