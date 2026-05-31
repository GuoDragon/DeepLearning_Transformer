"""
合成序列反转任务数据集：用于快速验证 Transformer 实现的正确性。

序列反转任务（Reverse Task）是一个经典的"玩具任务"：
  - 输入：随机整数序列
  - 输出：该序列的反转

例如：
  输入: [BOS, 5, 12, 7, 9, EOS]
  输出: [BOS, 9, 7, 12, 5, EOS]

为什么需要这个玩具任务？
  1. 快速自检：反转任务比翻译简单得多（词汇小、语法简单），
     可以快速判断 Transformer 实现是否正确。
  2. 验证注意力与掩码：反转要求模型记住整个序列再倒序输出，
     对因果掩码和跨位置依赖是很好的测试。
  3. 训练极快：使用合成数据，无需下载语料，几分钟内即可收敛。

模块结构：
  - ReverseBatch               — 反转任务批次数据容器
  - ReverseSequenceDataset     — 随机生成序列及其反转的 Dataset
  - pad_2d                     — 变长序列 padding（与翻译任务共用逻辑）
  - collate_reverse            — 反转任务批处理组装函数
  - infinite_loader            — DataLoader 无限循环迭代器
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset

from data.masks import make_src_mask, make_tgt_mask
from data.vocab import Vocab


# 内容词的最低 id：id 0~3 固定为 PAD/UNK/BOS/EOS，内容词从 4 开始
FIRST_CONTENT_ID = 4


# ============================================================================
# ReverseBatch — 反转任务批次容器
# ============================================================================
@dataclass
class ReverseBatch:
    """
    序列反转任务的一个批次。

    字段说明：
      - src:     源整数序列（已 padding），形状 (batch, src_len)
      - tgt_in:  目标序列解码器输入（去掉最后一位），形状 (batch, tgt_len-1)
      - tgt_out: 目标序列预测标签（去掉第一位），形状 (batch, tgt_len-1)
      - src_mask: 源序列 padding 掩码，形状 (batch, 1, 1, src_len)
      - tgt_mask: 目标序列联合掩码，形状 (batch, 1, tgt_len-1, tgt_len-1)
    """

    src: torch.Tensor
    tgt_in: torch.Tensor
    tgt_out: torch.Tensor
    src_mask: torch.Tensor
    tgt_mask: torch.Tensor


# ============================================================================
# ReverseSequenceDataset — 合成反转数据集
# ============================================================================
class ReverseSequenceDataset(Dataset):
    """
    随机生成整数序列及其反转，用于验证 Transformer 实现的正确性。

    生成规则：
      - 序列长度在 [min_len, max_len] 内随机采样
      - 每个位置的 token id 从 [FIRST_CONTENT_ID, vocab_size-1] 中随机选取
      - 目标序列 = 源序列的反转
      - 源/目标序列均添加 BOS 前缀和 EOS 后缀

    为什么 token id 从 FIRST_CONTENT_ID 开始？
      保留 id 0~3 给特殊符号（与 Vocab 类约定一致），
      PAD/UNK/BOS/EOS 不会出现在内容序列中，避免歧义。

    固定随机种子：
      构造函数中传入 seed，确保每次运行生成相同的序列，
      便于结果对比和调试。
    """

    def __init__(
        self,
        num_samples: int,
        vocab_size: int,
        min_len: int = 4,
        max_len: int = 12,
        seed: int = 42,
    ) -> None:
        """
        Args:
            num_samples: 生成的样本总数
            vocab_size:  词表大小（必须 > FIRST_CONTENT_ID + 1，即至少 6）
                         —— 预留空间给 4 个特殊符号 + 至少 1 个内容词
            min_len:     生成序列的最短长度（不含 BOS/EOS）
            max_len:     生成序列的最长长度（不含 BOS/EOS）
            seed:        随机种子（保证可复现）

        Raises:
            AssertionError: vocab_size 过小或长度范围不合法
        """
        super().__init__()
        assert vocab_size > FIRST_CONTENT_ID + 1, (
            f"vocab_size ({vocab_size}) 必须 > {FIRST_CONTENT_ID + 1}"
        )
        assert min_len >= 2 and max_len >= min_len, (
            f"序列长度范围不合法: min_len={min_len}, max_len={max_len}"
        )

        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.min_len = min_len
        self.max_len = max_len

        # 使用 Python 的 random.Random 而非全局 random，避免干扰其他模块
        self._rng = random.Random(seed)

        # 预生成所有序列并缓存（合成数据量小，内存完全够用）
        self._cache: list[list[int]] = []
        for _ in range(num_samples):
            L = self._rng.randint(min_len, max_len)
            # 从 [FIRST_CONTENT_ID, vocab_size-1] 中随机选 L 个数
            seq = [
                self._rng.randint(FIRST_CONTENT_ID, vocab_size - 1)
                for _ in range(L)
            ]
            self._cache.append(seq)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        """
        返回 (源序列 id, 反转后的目标序列 id)，均含 BOS 和 EOS。

        示例（min_len=2, max_len=4）：
          缓存序列: [7, 12, 5]
          反转:     [5, 12, 7]
          返回:
            src = [BOS, 7, 12, 5, EOS] = [2, 7, 12, 5, 3]
            tgt = [BOS, 5, 12, 7, EOS] = [2, 5, 12, 7, 3]

        Args:
            idx: 样本索引

        Returns:
            (源 id 序列, 目标 id 序列)
        """
        seq = self._cache[idx]
        rev = list(reversed(seq))
        src = [Vocab.BOS] + seq + [Vocab.EOS]
        tgt = [Vocab.BOS] + rev + [Vocab.EOS]
        return src, tgt


# ============================================================================
# pad_2d — 变长序列 padding（与翻译任务共用相同逻辑）
# ============================================================================
def pad_2d(sequences: list[list[int]], pad_idx: int = 0) -> torch.Tensor:
    """
    将不等长的 id 序列列表 padding 为等长二维张量 (batch, max_len)。

    反转任务中序列长度在 [min_len+2, max_len+2] 之间变化
    （加 2 是因为 BOS 和 EOS），因此 batch 内句子长度可能不同。

    Args:
        sequences: 不等长的 id 列表
        pad_idx:   填充值（默认 Vocab.PAD = 0）

    Returns:
        (batch_size, max_seq_len) 的 LongTensor
    """
    max_len = max(len(s) for s in sequences)
    out = torch.full((len(sequences), max_len), pad_idx, dtype=torch.long)
    for i, s in enumerate(sequences):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


# ============================================================================
# collate_reverse — 反转任务批处理组装函数
# ============================================================================
def collate_reverse(
    batch: list[tuple[list[int], list[int]]]
) -> ReverseBatch:
    """
    反转任务的 collate_fn：padding + Teacher Forcing 拆分 + 掩码构造。

    逻辑与 collate_translation 完全相同，只是 batch 类型为 ReverseBatch
    而非 TranslationBatch。这是因为反转任务不需要记录 src_vocab/tgt_vocab，
    两个 batch 类型在字段上完全一致但语义上分离，便于代码审计。

    Args:
        batch: (src_ids, tgt_ids) 元组列表

    Returns:
        组装好的 ReverseBatch
    """
    src_seqs, tgt_seqs = zip(*batch)

    # Step 1: padding
    src = pad_2d(list(src_seqs), Vocab.PAD)
    tgt_full = pad_2d(list(tgt_seqs), Vocab.PAD)

    # Step 2: Teacher Forcing 拆分
    tgt_in = tgt_full[:, :-1].contiguous()
    tgt_out = tgt_full[:, 1:].contiguous()

    # Step 3: 构造掩码
    src_mask = make_src_mask(src, Vocab.PAD)
    tgt_mask = make_tgt_mask(tgt_in, Vocab.PAD)

    return ReverseBatch(
        src=src, tgt_in=tgt_in, tgt_out=tgt_out,
        src_mask=src_mask, tgt_mask=tgt_mask,
    )


# ============================================================================
# infinite_loader — 无限循环 DataLoader
# ============================================================================
def infinite_loader(loader: DataLoader) -> Iterator:
    """
    将有限 DataLoader 包装为无限循环的迭代器。

    使用场景：
      反转任务使用合成数据，数据集理论上可以无限生成。
      但我们不希望真的生成无限数据，而是用有限样本反复训练。
      此时设定固定的 steps_per_epoch，通过 infinite_loader 循环读取。

    注意此函数会无限循环，调用方需要在合适的地方 break。

    Args:
        loader: PyTorch DataLoader 实例

    Yields:
        无限循环产生的 batch（永远不会 StopIteration）
    """
    while True:
        for b in loader:
            yield b