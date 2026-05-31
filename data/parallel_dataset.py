"""
平行语料数据集模块：支持行对齐的双语文本（如 .de / .en 句对）。

本模块实现了机器翻译任务的数据加载流程，核心组件包括：

  1. TranslationBatch — 一个批次的翻译训练数据容器（dataclass）
  2. ParallelTextDataset — PyTorch Dataset 子类，逐句对加载并编码
  3. pad_2d — 将变长序列补齐为等长 batch 张量
  4. collate_translation — DataLoader 的 collate_fn，组 batch + 构造掩码
  5. iter_tokenized_lines — 文件逐行 tokenize 的迭代器（用于构建词表）

数据流总览（端到端）：
  文本文件(.de/.en)
    → ParallelTextDataset（读取 + tokenize + encode + BOS/EOS）
      → DataLoader + collate_translation（pad 对齐 + 掩码构造）
        → TranslationBatch（src, tgt_in, tgt_out, src_mask, tgt_mask）
          → Transformer 模型

Teacher Forcing 机制：
  训练时不使用模型自己的预测作为下一步输入，而是直接使用真实目标序列。
  tgt_in  = 目标序列去掉最后一个 token（输入解码器）
  tgt_out = 目标序列去掉第一个 token（作为预测标签）
  例如目标 "BOS A B C EOS"：
    tgt_in:  "BOS A B C"    → Decoder 输入
    tgt_out: "A B C EOS"    → 与 Generator 输出比较计算 loss
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch.utils.data import Dataset

from data.masks import make_src_mask, make_tgt_mask
from data.vocab import Vocab


# ============================================================================
# TranslationBatch — 批次数据容器
# ============================================================================
@dataclass
class TranslationBatch:
    """
    一个批次的翻译训练数据（dataclass，自动生成 __init__）。

    字段说明：
      - src:     源语言 token id 序列，形状 (batch, src_len)，已 padding
      - tgt_in:  目标语言解码器输入，形状 (batch, tgt_len-1)，去掉最后一位
      - tgt_out: 目标语言预测标签，形状 (batch, tgt_len-1)，去掉第一位
      - src_mask: 源序列 padding 掩码，形状 (batch, 1, 1, src_len)
      - tgt_mask: 目标序列联合掩码（padding + causal），形状 (batch, 1, tgt_len-1, tgt_len-1)
    """

    src: torch.Tensor
    tgt_in: torch.Tensor
    tgt_out: torch.Tensor
    src_mask: torch.Tensor
    tgt_mask: torch.Tensor


# ============================================================================
# ParallelTextDataset — 平行语料 PyTorch Dataset
# ============================================================================
class ParallelTextDataset(Dataset):
    """
    行对齐的双语文本数据集。

    前提假设：
      - 源语言文件与目标语言文件行数相同，第 i 行互为翻译对
      - 每行是一个完整的句子
      - 文件中的句子已经过分词处理（以空格分隔）

    典型使用：
      >>> ds = ParallelTextDataset(
      ...     Path("data/train.de"), Path("data/train.en"),
      ...     src_vocab, tgt_vocab,
      ...     whitespace_tokenizer, whitespace_tokenizer,
      ...     max_len=64
      ... )
      >>> src_ids, tgt_ids = ds[0]
    """

    def __init__(
        self,
        src_path: Path,
        tgt_path: Path,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        src_tokenizer: Callable[[str], list[str]],
        tgt_tokenizer: Callable[[str], list[str]],
        max_len: int,
    ) -> None:
        """
        Args:
            src_path:       源语言文本文件路径（每行一句）
            tgt_path:       目标语言文本文件路径（每行一句）
            src_vocab:      源语言词表
            tgt_vocab:      目标语言词表
            src_tokenizer:  源语言分词函数（如 whitespace_tokenizer）
            tgt_tokenizer:  目标语言分词函数（如 whitespace_tokenizer）
            max_len:        单句最大 token 数（此为 BOS/EOS 前的主体长度限制，
                           加上 BOS/EOS 后实际长度 ≤ max_len）

        Raises:
            ValueError: 源文件和目标文件行数不一致时抛出
        """
        super().__init__()
        # 一次性读取全部行到内存（对于 Multi30k 这种规模完全可行）
        self.src_lines = src_path.read_text(encoding="utf-8").splitlines()
        self.tgt_lines = tgt_path.read_text(encoding="utf-8").splitlines()

        if len(self.src_lines) != len(self.tgt_lines):
            raise ValueError(
                f"源/目标文件行数不一致: {len(self.src_lines)} vs {len(self.tgt_lines)}"
            )

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """返回样本数（句对数）。"""
        return len(self.src_lines)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        """
        取第 idx 个句对，返回编码后的 id 序列。

        Tokenize → Encode → 加 BOS/EOS：
          - 原始句子: "the cat runs"
          - max_len=64, max_len-2=62（为主体的最大长度）
          - encode 截断到 62 个 token
          - 加首尾：[BOS, ...token_ids..., EOS]

        返回的 tgt 序列是完整序列（含 BOS/EOS），
        由 collate_fn 拆分为 tgt_in（去尾）和 tgt_out（去头）。

        Args:
            idx: 样本索引

        Returns:
            (源 id 序列, 目标 id 序列)，均含 BOS/EOS
        """
        s = self.src_tokenizer(self.src_lines[idx])
        t = self.tgt_tokenizer(self.tgt_lines[idx])

        # max_len - 2: 预留 BOS 和 EOS 的位置
        src_ids = self.src_vocab.encode(s, self.max_len - 2)
        tgt_ids = self.tgt_vocab.encode(t, self.max_len - 2)

        # 在首尾添加 BOS 和 EOS
        src_seq = [Vocab.BOS] + src_ids + [Vocab.EOS]
        tgt_seq = [Vocab.BOS] + tgt_ids + [Vocab.EOS]
        return src_seq, tgt_seq


# ============================================================================
# pad_2d — 变长序列 Padding
# ============================================================================
def pad_2d(sequences: list[list[int]], pad_idx: int) -> torch.Tensor:
    """
    将不等长的 id 序列列表 padding 为等长二维张量。

    这是 NLP batch 处理的经典操作：batch 内各句子长度不同，
    需要用 <pad> (id=0) 填充到该 batch 内最长句子的长度。

    示例：
      sequences = [[2, 5, 8, 3], [2, 7, 3]]  (BOS...EOS)
      pad_2d(sequences, pad_idx=0)
      → tensor([[2, 5, 8, 3],
                [2, 7, 3, 0]])   # 第二句末尾补 0 (PAD)

    注意：
      本实现先创建全 pad 矩阵，再将每个序列填入前几列，
      而不是使用 torch.nn.utils.rnn.pad_sequence，目的是保持代码自包含。

    Args:
        sequences: 不等长的 id 列表，每个为 list[int]
        pad_idx:   填充值（通常为 Vocab.PAD = 0）

    Returns:
        形状 (batch_size, max_seq_len) 的 LongTensor
    """
    max_len = max(len(s) for s in sequences)
    out = torch.full((len(sequences), max_len), pad_idx, dtype=torch.long)
    for i, s in enumerate(sequences):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


# ============================================================================
# collate_translation — 批处理组装函数（DataLoader 的 collate_fn）
# ============================================================================
def collate_translation(
    batch: list[tuple[list[int], list[int]]], pad_idx: int = 0
) -> TranslationBatch:
    """
    DataLoader 的 collate_fn：将多个样本组装为一个 TranslationBatch。

    组装流程：
      1. 分离 src 和 tgt 序列
      2. padding 对齐 → src 和 tgt_full 张量
      3. Teacher Forcing 拆分：
         tgt_full = [BOS, w1, w2, ..., wn, EOS]
         tgt_in   = [BOS, w1, w2, ..., wn]      ← 去尾，作为 Decoder 输入
         tgt_out  = [w1, w2, ..., wn, EOS]      ← 去头，作为预测目标
      4. 构造掩码：
         src_mask → padding 掩码
         tgt_mask → padding + causal 掩码

    为什么 tgt_in 和 tgt_out 要错开一位？
      这是 Teacher Forcing 的标准做法：
        - Decoder 第 0 步输入 BOS，预测第 1 个词；对比第 1 个词的真实值
        - Decoder 第 1 步输入 [BOS, w1]，预测第 2 个词；对比第 2 个词的真实值
        - ...
      错位确保了模型每一步都见到"正确答案的历史前缀"。

    Args:
        batch:   Dataset.__getitem__ 返回的 (src_ids, tgt_ids) 列表
        pad_idx: padding 符号 id

    Returns:
        组装好的 TranslationBatch
    """
    # Step 1: 分离 src 和 tgt
    src_seqs, tgt_seqs = zip(*batch)

    # Step 2: padding
    src = pad_2d(list(src_seqs), pad_idx)
    tgt_full = pad_2d(list(tgt_seqs), pad_idx)

    # Step 3: Teacher Forcing 拆分
    # tgt_full[:, :-1]: 保留除最后一列外的所有列 → tgt_in
    # tgt_full[:, 1:]:  保留除第一列外的所有列 → tgt_out
    tgt_in = tgt_full[:, :-1].contiguous()
    tgt_out = tgt_full[:, 1:].contiguous()

    # Step 4: 构造掩码
    src_mask = make_src_mask(src, pad_idx)
    tgt_mask = make_tgt_mask(tgt_in, pad_idx)

    return TranslationBatch(
        src=src, tgt_in=tgt_in, tgt_out=tgt_out,
        src_mask=src_mask, tgt_mask=tgt_mask,
    )


# ============================================================================
# iter_tokenized_lines — 文件 tokenize 迭代器
# ============================================================================
def iter_tokenized_lines(
    path: Path, tokenizer: Callable[[str], list[str]]
) -> Iterator[list[str]]:
    """
    读取文件每一行并用 tokenizer 分词，用于构建词表。

    与 yield_tokens 类似，但直接接受 Path 而非 Iterator[str]，
    在 parallel_dataset 模块内提供更便捷的接口。

    Args:
        path:      文本文件路径
        tokenizer: 分词函数

    Yields:
        每行分词后的 token 列表
    """
    for line in path.read_text(encoding="utf-8").splitlines():
        yield tokenizer(line)