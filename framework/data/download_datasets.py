#!/usr/bin/env python3
"""数据集下载与处理脚本。

下载完整的 TinyStories（英文）和 TinyStoriesChinese（中文翻译），
按以下目录结构组织：

data/
├── tinystories_en/
│   ├── train.txt      # 训练集（80%）
│   ├── test.txt       # 测试集（10%）
│   └── valid.txt      # 验证集（10%）
├── tinystories_zh/
│   ├── train.txt      # 训练集（80%）
│   ├── test.txt       # 测试集（10%）
│   └── valid.txt      # 验证集（10%）
└── download_datasets.py

用法的：
    python download_datasets.py                          # 下载所有数据
    python download_datasets.py --max_stories 10000       # 仅下载前 N 条（快速测试）
    python download_datasets.py --english_only            # 仅英文
    python download_datasets.py --chinese_only            # 仅中文
    python download_datasets.py --dry_run 1000            # 先试下载 1000 条看看
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ===== 输出目录 =====
BASE_DIR = Path(__file__).parent


# ===== 辅助函数 =====

def write_stories_to_file(stories: list, filepath: str, mode: str = "a"):
    """将故事列表追加写入 .txt 文件，用 <|endoftext|> 分隔。

    Args:
        stories: 故事文本列表
        filepath: 输出文件路径
        mode: 写入模式（'w' 覆盖, 'a' 追加）
    """
    os.makedirs(Path(filepath).parent, exist_ok=True)
    with open(filepath, mode, encoding="utf-8") as f:
        for i, story in enumerate(stories):
            if i > 0:
                f.write("\n<|endoftext|>\n")
            f.write(story.strip())


def save_split(stories: list, output_dir: Path, split_name: str):
    """将一批故事保存到对应分片文件。

    Args:
        stories: 故事列表
        output_dir: 输出目录（如 data/tinystories_en）
        split_name: 分片名（train / test / valid）
    """
    filepath = output_dir / f"{split_name}.txt"
    write_stories_to_file(stories, filepath, mode="a")
    logger.info(f"  → 追加 {len(stories)} 个故事至 {filepath}")


def verify_split(output_dir: Path):
    """验证并打印数据分片统计。"""
    for split in ["train", "test", "valid"]:
        fp = output_dir / f"{split}.txt"
        if fp.exists():
            with open(fp, encoding="utf-8") as f:
                content = f.read()
            stories = content.count("<|endoftext|>") + 1 if content.strip() else 0
            size_mb = fp.stat().st_size / 1024 / 1024
            logger.info(f"  {split}: {fp.stat().st_size / 1024 / 1024:.1f} MB, {stories} 故事")


# ===== 英文数据集 =====

def download_tinystories_en(max_stories: int = None) -> dict:
    """下载英文 TinyStories，返回划分后的故事字典。

    Args:
        max_stories: 最多故事数（None=全部）

    Returns:
        {"train": [...], "test": [...], "valid": [...]}
    """
    logger.info("=" * 50)
    logger.info("下载英文 TinyStories（约 210 万故事）")

    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    # 第一遍：收集故事
    all_stories = []
    for i, sample in enumerate(ds):
        text = sample.get("text", "").strip()
        if text:
            all_stories.append(text)
            if max_stories and len(all_stories) >= max_stories:
                break

    logger.info(f"共获取 {len(all_stories)} 个英文故事")

    # 打乱并划分
    random.shuffle(all_stories)
    total = len(all_stories)
    train_end = int(total * 0.8)
    test_end = int(total * 0.9)

    result = {
        "train": all_stories[:train_end],
        "test": all_stories[train_end:test_end],
        "valid": all_stories[test_end:],
    }
    logger.info(
        f"划分: 训练 {len(result['train'])} / "
        f"测试 {len(result['test'])} / "
        f"验证 {len(result['valid'])}"
    )
    return result


# ===== 中文数据集 =====

def download_tinystories_zh(max_stories: int = None) -> dict:
    """下载中文 TinyStoriesChinese，返回划分后的故事字典。

    Args:
        max_stories: 最多故事数（None=全部）

    Returns:
        {"train": [...], "test": [...], "valid": [...]}
    """
    logger.info("=" * 50)
    logger.info("下载中文 TinyStoriesChinese（约 210 万故事）")
    logger.info("此过程需要较长时间，请耐心等待...")

    ds = load_dataset("adam89/TinyStoriesChinese", split="train", streaming=True)

    all_stories = []
    for sample in ds:
        text = sample["jsonl"].decode("utf-8")
        for line in text.strip().split("\n"):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                zh = record.get("story_zh", "").strip()
                if zh:
                    all_stories.append(zh)
                    if max_stories and len(all_stories) >= max_stories:
                        break
            except json.JSONDecodeError:
                continue
        if max_stories and len(all_stories) >= max_stories:
            break

    logger.info(f"共获取 {len(all_stories)} 个中文故事")

    # 打乱并划分
    random.shuffle(all_stories)
    total = len(all_stories)
    train_end = int(total * 0.8)
    test_end = int(total * 0.9)

    result = {
        "train": all_stories[:train_end],
        "test": all_stories[train_end:test_end],
        "valid": all_stories[test_end:],
    }
    logger.info(
        f"划分: 训练 {len(result['train'])} / "
        f"测试 {len(result['test'])} / "
        f"验证 {len(result['valid'])}"
    )
    return result


# ===== 保存数据 =====

def save_dataset(stories_dict: dict, output_dir: Path):
    """将划分后的故事字典保存到目录下的 train/test/valid.txt。

    Args:
        stories_dict: {"train": [...], "test": [...], "valid": [...]}
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"保存至: {output_dir}")

    for split_name, stories in stories_dict.items():
        filepath = output_dir / f"{split_name}.txt"
        # 覆盖写入
        write_stories_to_file(stories, filepath, mode="w")
        size_mb = filepath.stat().st_size / 1024 / 1024
        logger.info(f"  {split_name}: {len(stories)} 故事, {size_mb:.1f} MB → {filepath}")


# ===== 主流程 =====

def main():
    parser = argparse.ArgumentParser(description="下载完整的 TinyStories 数据集")
    parser.add_argument(
        "--max_stories", type=int, default=None,
        help="每语种最多故事数（默认 None=全部）"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认 42）"
    )
    parser.add_argument(
        "--english_only", action="store_true",
        help="仅下载英文数据"
    )
    parser.add_argument(
        "--chinese_only", action="store_true",
        help="仅下载中文数据"
    )
    parser.add_argument(
        "--dry_run", type=int, default=0,
        help="试运行模式，下载 N 条数据预览"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # 试运行模式
    max_n = args.dry_run if args.dry_run > 0 else args.max_stories

    # ===== 下载英文 =====
    if not args.chinese_only:
        en_data = download_tinystories_en(max_n)
        save_dataset(en_data, BASE_DIR / "tinystories_en")
        if not args.dry_run:
            verify_split(BASE_DIR / "tinystories_en")

    # ===== 下载中文 =====
    if not args.english_only:
        zh_data = download_tinystories_zh(max_n)
        save_dataset(zh_data, BASE_DIR / "tinystories_zh")
        if not args.dry_run:
            verify_split(BASE_DIR / "tinystories_zh")

    logger.info("=" * 50)
    logger.info("全部完成！")
    if args.dry_run:
        logger.info(f"试运行完成，下载了 {args.dry_run} 条数据。去掉 --dry_run 下载全部。")
    else:
        logger.info("目录结构:")
        logger.info("  data/tinystories_en/{train,test,valid}.txt")
        logger.info("  data/tinystories_zh/{train,test,valid}.txt")
        logger.info("训练时使用:")
        logger.info("  python main.py --train --data data/tinystories_en")
        logger.info("  python main.py --train --data data/tinystories_zh")


if __name__ == "__main__":
    main()
