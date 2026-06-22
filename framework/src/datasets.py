"""数据集模块：数据加载、预处理和 DataLoader。

词表管理：
- 从训练集构建词表后自动保存到 data_dir/tokenizer.json
- 测试/验证加载时自动复用训练集词表
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from src.utils import BaseTokenizer

logger = logging.getLogger(__name__)


# ===== 分词器保存路径 =====
TOKENIZER_FILENAME = "tokenizer.json"


class TextDataset(Dataset):
    """文本数据集（变长序列）。

    从 token id 连续序列中采样变长片段（长度在 [min_seq_len, max_seq_len] 间随机），
    单个样本不定长，由 collate_fn 统一填充。

    每个 index 映射到 token 流中的一个固定锚点位置，
    `__getitem__` 从此位置开始取一段随机长度的序列。
    """

    def __init__(
        self,
        token_ids: List[int],
        max_seq_len: int = 512,
        min_seq_len: Optional[int] = None,
    ):
        """初始化数据集。

        Args:
            token_ids: 全部 token id 序列
            max_seq_len: 最大序列长度（也是每个样本的锚点间距）
            min_seq_len: 最小序列长度，默认 max_seq_len // 2
        """
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len or (max_seq_len // 2)
        assert self.min_seq_len <= self.max_seq_len, "min_seq_len 不能大于 max_seq_len"

        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        total = len(self.token_ids)
        # 用锚点间距 = max_seq_len 来估算样本数，保证覆盖
        self.num_samples = max(1, total // max_seq_len)
        # 记录最后一个有效起始位置，防止取越界
        self._max_start = max(0, total - self.min_seq_len)

        logger.info(
            f"数据集创建完成 | 总token数: {total:,} | "
            f"样本数: {self.num_samples:,} | "
            f"长度范围: [{self.min_seq_len}, {self.max_seq_len}]"
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 锚点均匀分布在整个 token 流上
        stride = max(1, self._max_start // max(1, self.num_samples))
        start = (idx * stride) % max(1, self._max_start + 1)

        # 用 idx 做种子，保证同一样本每次返回相同长度（可复现）
        rng = torch.Generator()
        rng.manual_seed(idx)
        length = int(torch.randint(self.min_seq_len, self.max_seq_len + 1, (1,), generator=rng).item())

        segment = self.token_ids[start:start + length]
        # 尾部不足时回退取最后一段
        if len(segment) < self.min_seq_len:
            start = max(0, len(self.token_ids) - length)
            segment = self.token_ids[start:start + length]
        return {"input_ids": segment, "labels": segment.clone()}


# ===== 变长 batch 的 collate 函数 =====

PAD_ID = 0  # 与 utils.py 中 tokenizer.pad_id 保持一致

# 预定义的 bucket 边界（对齐到这些长度，减少 CUDAGraph 重录次数）
# 值根据 max_seq_len 选择，步长 64 在降低重录次数和减少浪费 padding 间取得平衡
_BUCKET_SIZES = [256, 384, 512]


def _ceil_to_bucket(length: int, buckets: list = None) -> int:
    """将长度上取整到最近的 bucket。"""
    if buckets is None:
        buckets = _BUCKET_SIZES
    for b in buckets:
        if b >= length:
            return b
    return buckets[-1]


def collate_varlen_batch(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = PAD_ID,
    buckets: list = None,
) -> Dict[str, torch.Tensor]:
    """将变长样本列表填充为等长 batch（带 bucket 对齐）。

    先按样本实际最大长度上取整到最近的 bucket 值，
    再将所有样本统一 pad 到此长度。
    这样不同 batch 更容易落到相同的 bucket，大幅减少
    torch.compile CUDAGraph 的重新录制次数。

    1. input_ids 尾部补 pad_token_id
    2. labels 先补 pad_token_id，再替换为 -100（让 loss 忽略）
    3. 生成 attention_mask（1=真实token, 0=padding）

    Args:
        batch: __getitem__ 返回的样本列表
        pad_token_id: padding token id
        buckets: bucket 列表，默认 _BUCKET_SIZES

    Returns:
        {
            "input_ids": (batch_size, bucket_size),
            "labels": (batch_size, bucket_size)，padding 位置为 -100,
            "attention_mask": (batch_size, bucket_size),
        }
    """
    input_ids_list = [item["input_ids"] for item in batch]
    labels_list = [item["labels"] for item in batch]

    # 实际最大长度 → 对齐到 bucket
    actual_max = max(len(x) for x in input_ids_list)
    bucket_size = _ceil_to_bucket(actual_max, buckets)

    # 1. pad input_ids 到 bucket_size
    padded_input_ids = pad_sequence(
        input_ids_list, batch_first=True, padding_value=pad_token_id
    )
    # 若不足 bucket_size，继续补全
    if padded_input_ids.size(1) < bucket_size:
        pad_len = bucket_size - padded_input_ids.size(1)
        padded_input_ids = torch.nn.functional.pad(
            padded_input_ids, (0, pad_len), value=pad_token_id
        )

    # 2. pad labels，再替换为 -100
    padded_labels = pad_sequence(
        labels_list, batch_first=True, padding_value=pad_token_id
    )
    if padded_labels.size(1) < bucket_size:
        pad_len = bucket_size - padded_labels.size(1)
        padded_labels = torch.nn.functional.pad(
            padded_labels, (0, pad_len), value=pad_token_id
        )
    padded_labels[padded_labels == pad_token_id] = -100

    # 3. attention_mask：1 为有效 token，0 为 padding
    attention_mask = (padded_input_ids != pad_token_id).long()

    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "attention_mask": attention_mask,
    }


# ===== 文件加载 =====
def load_single_file(filepath: str) -> str:
    """加载单个文本文件内容。"""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    text = path.read_text(encoding="utf-8").strip()
    logger.info(f"从 {filepath} 加载文本 ({len(text):,} 字符)")
    return text


def load_texts_from_directory(data_dir: str) -> List[str]:
    """从数据目录加载测试文本（每行为一个样本）。

    Args:
        data_dir: 数据目录路径

    Returns:
        文本列表
    """
    test_file = Path(data_dir) / "test.txt"
    if not test_file.exists():
        logger.warning(f"测试文件不存在: {test_file}")
        return []
    text = test_file.read_text(encoding="utf-8").strip()
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    logger.info(f"从 {test_file} 加载了 {len(lines)} 条测试文本")
    return lines

# ===== 加载文本并构建/复用词表 =====

# 数据集中的特殊标记（如故事分隔符），在 tokenization 之前剥离
# 避免模型学习无语义的标记，也避免污染词表
_DATA_SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_sep|>"]

# 构建词表时的文本块大小（字符数），避免一次载入全部文本
_VOCAB_CHUNK_SIZE = 500_000


def _clean_special_tokens(text: str) -> str:
    """移除数据集中的特殊标记，防止污染词表和模型输出。"""
    for t in _DATA_SPECIAL_TOKENS:
        text = text.replace(t, "")
    return text.strip()


def _text_chunker(text: str, chunk_size: int = _VOCAB_CHUNK_SIZE):
    """将大文本拆分为块，逐块产出用于增量构建词表。

    Args:
        text: 完整文本
        chunk_size: 每块字符数

    Yields:
        文本块
    """
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


def _tokenize_text(
    text: str,
    tokenizer: BaseTokenizer,
    data_dir: str,
    is_train: bool,
    max_vocab_size: int = 5000,
    min_freq: int = 1,
) -> List[int]:
    """对文本编码，若为训练集则先构建/保存词表。

    Args:
        text: 文本内容
        tokenizer: 分词器
        data_dir: 数据目录
        is_train: 是否为训练集
        max_vocab_size: 最大词表大小
        min_freq: 构建词表的最低字符频次（>=2 可过滤罕见字）

    Returns:
        token id 列表
    """
    # 清洗数据集特殊标记，避免污染词表和模型输出
    text = _clean_special_tokens(text)

    # 先加载已有词表
    if tokenizer.vocab_size == 0:
        path = Path(data_dir) / TOKENIZER_FILENAME
        if path.exists():
            # 将已加载的词表复制到当前 tokenizer
            tokenizer.load(path)

    # 加载后仍为空 -> 构建词表
    if tokenizer.vocab_size == 0:
        if is_train:
            logger.info("从训练集构建词表（增量模式，每块最大 500K 字符）...")
            tokenizer.build_vocab(
                _text_chunker(text),
                max_vocab_size=max_vocab_size,
                min_freq=min_freq,
            )
            tokenizer.save(str(path))
        else:
            raise RuntimeError(
                f"分词器词表为空且未找到已保存的词表文件。"
                f"请先运行训练集（train.txt）构建词表。"
            )

    return tokenizer.encode(text)


# ===== 数据加载器创建 =====
def create_dataloaders(
    data_dir: str,
    tokenizer: BaseTokenizer,
    max_seq_len: int = 512,
    min_seq_len: Optional[int] = None,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42,
    max_vocab_size: int = 5000,
    min_freq: int = 1,
) -> Dict[str, DataLoader]:
    """创建训练和验证 DataLoader。

    训练集词表会自动保存到 data_dir/tokenizer.json。
    验证集自动使用训练集构建的词表。

    Args:
        data_dir: 数据目录路径
        tokenizer: 分词器实例
        max_seq_len: 最大序列长度
        min_seq_len: 最小序列长度，默认 max_seq_len // 2
        batch_size: 批次大小
        num_workers: 数据加载线程数
        seed: 随机种子
        max_vocab_size: 最大词表大小
        min_freq: 构建词表的最低字符频次（>=2 可过滤罕见字）

    Returns:
        {"train": train_loader, "valid": valid_loader, "test": test_loader}
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    train_file = data_path / "train.txt"
    valid_file = data_path / "valid.txt"
    test_file = data_path / "test.txt"

    # 训练集：构建并保存词表
    train_text = load_single_file(str(train_file))
    train_ids = _tokenize_text(
        train_text, tokenizer, data_dir, is_train=True,
        max_vocab_size=max_vocab_size, min_freq=min_freq,
    )
    logger.info(f"训练集 token 数: {len(train_ids):,}")

    # 验证集：复用训练集词表
    valid_text = load_single_file(str(valid_file))
    valid_ids = _tokenize_text(valid_text, tokenizer, data_dir, is_train=False)
    logger.info(f"验证集 token 数: {len(valid_ids):,}")

    # 测试集
    test_text = load_single_file(str(test_file))
    test_ids = _tokenize_text(test_text, tokenizer, data_dir, is_train=False)
    logger.info(f"测试集 token 数: {len(test_ids):,}")

    train_dataset = TextDataset(train_ids, max_seq_len=max_seq_len, min_seq_len=min_seq_len)
    valid_dataset = TextDataset(valid_ids, max_seq_len=max_seq_len, min_seq_len=min_seq_len)
    test_dataset = TextDataset(test_ids, max_seq_len=max_seq_len, min_seq_len=min_seq_len)

    # ===== 创建 DataLoader（使用自定义 collate_fn） =====
    dataloader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_varlen_batch,
    )

    # 固定种子保证 DataLoader shuffle 可复现
    _generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=_generator,
        **dataloader_kwargs,
    )
    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        drop_last=False,
        **dataloader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **dataloader_kwargs,
    )

    logger.info(
        f"DataLoader 创建完成 | "
        f"训练: {len(train_dataset):,} 样本, {len(train_loader):,} 批次 | "
        f"验证: {len(valid_dataset):,} 样本, {len(valid_loader):,} 批次 | "
        f"测试: {len(test_dataset):,} 样本, {len(test_loader):,} 批次 | "
        f"collate_fn: 变长 padding"
    )
    return {"train": train_loader, "valid": valid_loader, "test": test_loader}
