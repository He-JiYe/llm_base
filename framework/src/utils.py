"""工具函数模块：分词器（Char + BPE）、评估指标、辅助函数"""

import json
import logging
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ===== 全局日志配置（只初始化一次） =====

_logging_configured = False


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO):
    """统一配置日志系统：同时输出到文件和终端。

    幂等设计，只初始化一次。所有模块共用同一套配置。

    Args:
        log_dir: 日志文件目录
        log_level: 日志级别
    """
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    import os as _os
    import sys as _sys

    _os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(log_level)

    # 清除根 logger 已有 handler（某些库在 import 时可能添加了默认 handler）
    for h in root.handlers[:]:
        root.removeHandler(h)

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")

    # 文件处理器
    file_handler = logging.FileHandler(
        _os.path.join(log_dir, "training.log"), encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # 控制台处理器
    console_handler = logging.StreamHandler(_sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    logger.info(f"日志系统已初始化 | 级别: {logging.getLevelName(log_level)} | 文件: {log_dir}/training.log")


# ===== 中文文本预处理 =====

class ChineseTextPreprocessor:
    """中文文本预处理：繁转简 + Unicode 标准化。

    减少非常用字/繁体字对词表的污染，使 CharTokenizer 的词表更紧凑。
    使用 opencc 库进行繁→简转换，回退到内置映射。
    """

    _opencc_available = False
    _converter = None

    @classmethod
    def _ensure_opencc(cls):
        """尝试初始化 opencc（仅一次）。"""
        if cls._converter is None:
            try:
                from opencc import OpenCC
                cls._converter = OpenCC("t2s")
                cls._opencc_available = True
                logger.info("opencc 繁→简转换器已初始化")
            except ImportError:
                logger.warning(
                    "opencc 未安装，繁→简转换不可用。"
                    "安装: uv pip install opencc-python-reimplemented"
                )
                cls._opencc_available = False

    @classmethod
    def traditional_to_simplified(cls, text: str) -> str:
        """繁体中文转简体中文。

        Args:
            text: 输入文本

        Returns:
            简体中文文本
        """
        cls._ensure_opencc()
        if cls._opencc_available and cls._converter is not None:
            return cls._converter.convert(text)
        return text

    @staticmethod
    def unicode_normalize(text: str) -> str:
        """Unicode 标准化（NFKC），统一异体字和全半角。

        NFKC 会将：
        - 全角字母数字 → 半角（Ａ→A）
        - 兼容性字符 → 标准形式
        - 部分异体字 → 正体

        Args:
            text: 输入文本

        Returns:
            标准化后的文本
        """
        return unicodedata.normalize("NFKC", text)

    @classmethod
    def preprocess(cls, text: str) -> str:
        """完整预处理流水线：NFKC 标准化 → 繁转简。

        Args:
            text: 输入文本

        Returns:
            预处理后的文本
        """
        text = cls.unicode_normalize(text)
        text = cls.traditional_to_simplified(text)
        return text


# ===== 分词器基类 / 通用接口 =====

TOKENIZER_TYPE_KEY = "_tokenizer_type"  # 保存时标记类型的 key


class BaseTokenizer:
    """分词器基类，统一接口。"""

    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"]

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}

        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.mask_id = 4

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: str):
        """保存词表到文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            TOKENIZER_TYPE_KEY: self._tokenizer_type(),
            "vocab": self.vocab,
        }
        data.update(self._extra_save_data())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"{self._tokenizer_type()} 已保存至: {path}")

    def load(self, path: str):
        """从文件加载词表。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.id2token = {v: k for k, v in self.vocab.items()}
        self._extra_load_data(data)
        logger.info(f"{self._tokenizer_type()} 已加载 | 大小: {len(self.vocab)} | 来源: {path}")

    def _tokenizer_type(self) -> str:
        raise NotImplementedError

    def _extra_save_data(self) -> dict:
        return {}

    def _extra_load_data(self, data: dict):
        pass


def create_tokenizer_from_file(path: str) -> "BaseTokenizer":
    """从文件加载 tokenizer（自动检测类型）。

    Args:
        path: tokenizer 保存路径

    Returns:
        CharTokenizer 或 BPETokenizer 实例
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ttype = data.get(TOKENIZER_TYPE_KEY, "char")
    if ttype == "bpe":
        tok = BPETokenizer()
    else:
        tok = CharTokenizer()
    tok.load(path)
    return tok


# ===== 字符级分词器（中文） =====

class CharTokenizer(BaseTokenizer):
    """字符级分词器，适合中文。

    以单个字符为 token，自动从训练文本构建词表。
    """

    def __init__(self):
        super().__init__()

    def _tokenizer_type(self) -> str:
        return "char"

    def build_vocab(
        self,
        texts,
        min_freq: int = 1,
        max_vocab_size: int = 5000,
    ):
        """从文本流构建词表。

        Args:
            texts: 文本可迭代对象（list / generator）
            min_freq: 最小出现频率
            max_vocab_size: 最大词表大小（含特殊 token）
        """
        char_freq = {}
        text_count = 0
        for text in tqdm(texts, desc="统计字符频率", unit="块"):
            text_count += 1
            for char in text:
                char_freq[char] = char_freq.get(char, 0) + 1

        sorted_chars = sorted(char_freq.items(), key=lambda x: -x[1])
        filtered_chars = [c for c, f in sorted_chars if f >= min_freq]

        all_tokens = self.SPECIAL_TOKENS + filtered_chars
        if len(all_tokens) > max_vocab_size:
            all_tokens = all_tokens[:max_vocab_size]

        self.vocab = {t: i for i, t in enumerate(all_tokens)}
        self.id2token = {i: t for t, i in self.vocab.items()}
        logger.info(
            f"CharTokenizer 词表构建完成 | 大小: {len(self.vocab)} | "
            f"文本块数: {text_count} | 最低频次: {min_freq}"
        )

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """将文本转为 token id 序列。"""
        ids = []
        if add_special:
            ids.append(self.bos_id)
        for char in text:
            ids.append(self.vocab.get(char, self.unk_id))
        if add_special:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """将 token id 序列转回文本。"""
        if skip_special:
            special_set = {self.pad_id, self.bos_id, self.eos_id, self.mask_id}
            chars = [self.id2token.get(i, "<unk>") for i in ids if i not in special_set]
        else:
            chars = [self.id2token.get(i, "<unk>") for i in ids]
        return "".join(chars)


# ===== BPE 分词器（英文） =====

# 预分词正则：拆分单词、标点和空白
# 匹配: 字母数字序列, 标点符号, 空白
_PRETOKENIZE_PAT = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']+|\s+")


def _pretokenize(text: str) -> List[str]:
    """预分词：按 GPT-2 风格拆分为单词/符号片段。"""
    return _PRETOKENIZE_PAT.findall(text)


def _merge_pair(word: str, pair: Tuple[str, str], new_token: str) -> str:
    """将 word 中所有相邻的 pair 替换为 new_token。"""
    first, second = pair
    result = []
    symbols = word.split()
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
            result.append(new_token)
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return " ".join(result)


class BPETokenizer(BaseTokenizer):
    """BPE（Byte-Pair Encoding）分词器，适合英文。

    通过迭代合并最频繁的字符合并来构建子词词表。
    """

    def _tokenizer_type(self) -> str:
        return "bpe"

    def __init__(self):
        super().__init__()
        # BPE 合并规则列表：[(token_a, token_b), ...]
        self.merges: List[Tuple[str, str]] = []

    def build_vocab(
        self,
        texts,
        max_vocab_size: int = 5000,
        min_freq: int = 2,
    ):
        """从文本流学习 BPE 合并规则并构建词表。

        Args:
            texts: 文本可迭代对象（list / generator)
            max_vocab_size: 目标词表大小（含特殊 token）
            min_freq: 合并的最小频次
        """
        # 第一遍：增量统计单词频次（不载入全部文本）
        word_counts = defaultdict(int)
        char_set = set()
        block_count = 0
        for text in tqdm(texts, desc="统计单词频次", unit="块"):
            block_count += 1
            for word in _pretokenize(text):
                word_counts[word] += 1
                for ch in word:
                    char_set.add(ch)
        logger.info(f"文本处理完成 | 块数: {block_count} | 独立单词: {len(word_counts)}")

        # 初始词表：特殊 token + 所有字符（去重后排序）
        initial_chars = sorted(char_set)
        vocab = {t: i for i, t in enumerate(self.SPECIAL_TOKENS + initial_chars)}
        logger.info(f"初始字符词表大小: {len(vocab)}")
        # 将每个单词表示为字符序列（用空格分隔）
        word_to_symbols = {}
        for word in tqdm(word_counts, desc="初始化单词表示", unit="词"):
            word_to_symbols[word] = " ".join(list(word))

        # 当前 token 数量
        current_vocab_size = len(vocab)
        target_size = max_vocab_size
        self.merges = []

        total_merges = target_size - current_vocab_size
        if total_merges <= 0:
            logger.info("词表已达到目标大小，无需合并")
        else:
            # 迭代合并最频繁的 token 对
            pbar = tqdm(total=total_merges, desc="BPE 合并", unit="合并")
            while current_vocab_size < target_size:
                # 从更新后的符号表示统计 pair 频次
                pair_counts = defaultdict(int)
                for word in word_counts:
                    freq = word_counts[word]
                    symbols = word_to_symbols[word].split()
                    for i in range(len(symbols) - 1):
                        pair_counts[(symbols[i], symbols[i + 1])] += freq

                if not pair_counts:
                    break

                # 过滤低频合并
                if min_freq > 1:
                    pair_counts = {p: c for p, c in pair_counts.items() if c >= min_freq}
                    if not pair_counts:
                        break

                # 找到最频繁的 pair
                best_pair = max(pair_counts, key=pair_counts.get)
                best_freq = pair_counts[best_pair]

                # 创建新 token
                new_token = best_pair[0] + best_pair[1]
                self.merges.append(best_pair)

                # 在词表中添加新 token
                vocab[new_token] = current_vocab_size
                current_vocab_size += 1

                # 更新所有单词的符号表示
                for word in word_counts:
                    word_to_symbols[word] = _merge_pair(
                        word_to_symbols[word], best_pair, new_token
                    )

                pbar.update(1)

            pbar.close()

        self.vocab = vocab
        self.id2token = {i: t for t, i in vocab.items()}
        logger.info(
            f"BPETokenizer 训练完成 | 词表大小: {len(self.vocab)} | "
            f"合并规则数: {len(self.merges)}"
        )

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """使用 BPE 规则将文本编码为 token id 序列。

        Args:
            text: 输入文本
            add_special: 是否添加 <bos> 和 <eos>

        Returns:
            token id 列表
        """
        words = _pretokenize(text)
        ids = []
        if add_special:
            ids.append(self.bos_id)

        for word in words:
            # 将单词拆分为字符
            symbols = list(word)
            # 贪心应用合并规则
            merged = True
            while merged:
                merged = False
                # 尝试按顺序应用每个合并规则
                for pair in self.merges:
                    pair_str = f"{pair[0]} {pair[1]}"
                    new_str = pair[0] + pair[1]
                    # 将 symbols 拼成字符串检查
                    symbol_str = " ".join(symbols)
                    if pair_str in symbol_str:
                        # 找到第一个匹配位置
                        idx = symbol_str.find(pair_str)
                        # 重新构建 symbols
                        new_symbols = []
                        i = 0
                        syms = symbol_str.split()
                        while i < len(syms):
                            if i < len(syms) - 1 and syms[i] == pair[0] and syms[i + 1] == pair[1]:
                                new_symbols.append(new_str)
                                i += 2
                                merged = True
                            else:
                                new_symbols.append(syms[i])
                                i += 1
                        symbols = new_symbols
                        break  # 每次只应用一个规则

            # 将单词的 token id 加入结果
            for sym in symbols:
                ids.append(self.vocab.get(sym, self.unk_id))

        if add_special:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """将 token id 序列解码为文本。

        BPE 解码需要将子词拼接回单词。
        """
        if skip_special:
            special_set = {self.pad_id, self.bos_id, self.eos_id, self.mask_id}
            tokens = [self.id2token.get(i, "<unk>") for i in ids if i not in special_set]
        else:
            tokens = [self.id2token.get(i, "<unk>") for i in ids]

        # BPE 解码：合并子词（以 </w> 或 ## 等标记结尾的表示词边界）
        # 这里使用更简单的规则：如果 token 以空格开头则保留
        text = ""
        for t in tokens:
            if t.startswith(" "):
                text += t
            else:
                text += t
        return text.strip()

    def _extra_save_data(self) -> dict:
        return {"merges": self.merges}

    def _extra_load_data(self, data: dict):
        self.merges = [tuple(m) for m in data.get("merges", [])]


# ===== 分词器工厂 =====

def create_tokenizer(
    tokenizer_type: str = "char",
    chinese_normalize: bool = False,
) -> BaseTokenizer:
    """创建指定类型的分词器。

    Args:
        tokenizer_type: "char" 或 "bpe"
        chinese_normalize: 是否开启中文预处理（仅对 char 有效）

    Returns:
        分词器实例
    """
    if tokenizer_type == "bpe":
        return BPETokenizer()
    return CharTokenizer()


# ===== 评估指标 =====

def compute_perplexity(loss: float) -> float:
    """根据交叉熵损失计算困惑度 (Perplexity)。

    Args:
        loss: 交叉熵损失值

    Returns:
        困惑度值: exp(loss)
    """
    return float(np.exp(loss))


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型可训练参数数量。

    Args:
        model: PyTorch 模型

    Returns:
        可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_mask_and_position_ids(
    input_ids: torch.Tensor,
    pad_token_id: int = 0,
) -> tuple:
    """生成 attention mask 和位置编码 id。

    Args:
        input_ids: 输入 token id 张量 (batch_size, seq_len)
        pad_token_id: padding token 的 id

    Returns:
        (attention_mask, position_ids) 元组
    """
    attention_mask = (input_ids != pad_token_id).long()
    position_ids = torch.cumsum(attention_mask, dim=1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    return attention_mask, position_ids


def seed_everything(seed: int):
    """设置全局随机种子，保证可复现性。"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"随机种子已设置: {seed}")
