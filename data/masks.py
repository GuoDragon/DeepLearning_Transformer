"""
掩码构造模块：为 Transformer 的 Encoder 和 Decoder 生成注意力掩码。

在 Transformer 中，掩码（mask）是控制注意力范围的关键机制，共有两类：

  1. Padding Mask（填充掩码）  —— 屏蔽填充符号 <pad>，使其不参与注意力计算
  2. Causal Mask（因果掩码）   —— 屏蔽未来 token，保证自回归解码的正确性

两类掩码的组合使用方式：

    位置            Encoder           Decoder 自注意力      Decoder 交叉注意力
    ─────────────────────────────────────────────────────────────────────
    padding mask     ✓（屏蔽 <pad>）   ✓（屏蔽 <pad>）        ✓（屏蔽源语言 <pad>）
    causal mask      ✗                ✓（只看已生成 token）   ✗

掩码形状说明：
  - Encoder 用 (batch, 1, 1, seq_len)：广播到 Q/K 的 (batch, heads, seq, d_k)
    只按 key 维度屏蔽
  - Decoder 用 (batch, 1, tgt_len, tgt_len)：按矩阵的两个维度同时屏蔽

掩码语义：True = 可关注（保留），False = 屏蔽（softmax 前置为 -inf）
"""

from __future__ import annotations

import torch


# ============================================================================
# 因果掩码（subsequent mask）—— 保证解码器自回归生成的因果性
# ============================================================================
def subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    """
    构造因果掩码（下三角矩阵），确保解码时位置 i 只能关注 <= i 的位置。

    掩码形式（以 size=4 为例）：

        [[ True, False, False, False ],
         [ True,  True, False, False ],
         [ True,  True,  True, False ],
         [ True,  True,  True,  True ]]

    对角线及下方为 True（可见），上方为 False（不可见）。

    数学原理：
      自回归生成时，第 i 步只能依赖已生成的 0..i 位置的 token。
      若不加因果掩码，模型会"作弊"直接看目标答案，无法学会真正的生成能力。

    实现技巧：
      使用 torch.triu（上三角矩阵）再取反，简洁高效。

    Args:
        size:   目标序列长度
        device: 张量所在设备（CPU / CUDA）

    Returns:
        形状 (1, size, size) 的布尔张量
        —— 第 0 维为 1，用于 batch 维度的广播
    """
    # 构造上三角全 True 矩阵（对角线偏移 1，不含对角线本身）
    ahead = torch.triu(
        torch.ones((1, size, size), device=device, dtype=torch.bool), diagonal=1
    )
    # 取反 → 下三角为 True（可见），上三角为 False（不可见）
    return ~ahead


# ============================================================================
# 源序列 padding 掩码 —— 用于 Encoder 自注意力 和 交叉注意力
# ============================================================================
def make_src_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    构造源序列的 padding 掩码，屏蔽 <pad> token 位置。

    用途：
      - Encoder 的每一层自注意力中，防止 <pad> token 被关注
      - Decoder 的交叉注意力中，防止关注源语言的 <pad> 位置

    形状变换：
      (batch, src_len) → unsqueeze(1) → unsqueeze(2) → (batch, 1, 1, src_len)

    为什么要 unsqueeze 到 (batch, 1, 1, src_len)？
      注意力 scores 的形状是 (batch, heads, seq_q, seq_k)。
      掩码需要能广播到这个形状：
        - seq_q 维度（第 3 维）为 1，广播到所有 query 位置
        - seq_k 维度（第 4 维）为 src_len，逐 key 位置屏蔽
        - heads 维度（第 2 维）为 1，广播到所有注意力头

    Args:
        src:     源序列 token id 张量，形状 (batch, src_len)
        pad_idx: padding 符号的 id（通常为 0）

    Returns:
        形状 (batch, 1, 1, src_len) 的布尔张量，True = 非 padding
    """
    # src != pad_idx: (batch, src_len)，非 pad 位置为 True
    # unsqueeze(1):   (batch, 1, src_len)
    # unsqueeze(2):   (batch, 1, 1, src_len)
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


# ============================================================================
# 目标序列掩码 —— padding 掩码 + 因果掩码的组合
# ============================================================================
def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    构造目标序列的联合掩码：padding 掩码与因果掩码的逐元素逻辑与。

    为什么需要两种掩码的组合？
      1. Padding 掩码：屏蔽 <pad> token，因为 batch 内句子长度不一致
      2. Causal 掩码： 屏蔽未来 token（位置 i 不能看 > i 的位置）

    两种掩码分别作用于不同维度：
      - pad_keys 形状: (batch, 1, 1, tgt_len)，沿 key 维度屏蔽 <pad>
      - causal 形状:   (1, tgt_len, tgt_len)，下三角为 True
      → 两者做 broadcast &（逻辑与），得到最终掩码

    实际效果（假设 tgt="A B C <pad>"，tgt_len=4）：
      pad_keys 使第 4 列全为 False
      causal 使上三角为 False
      组合后：
        [[T, F, F, F],
         [T, T, F, F],
         [T, T, T, F],
         [F, F, F, F]]   ← 第 4 行因 padding 全 False

    Args:
        tgt:     目标序列 token id 张量，形状 (batch, tgt_len)
                 —— 注意：传入的是 tgt_in（去掉最后一位），而非完整的 tgt_full
        pad_idx: padding 符号 id（通常为 0）

    Returns:
        形状 (batch, 1, tgt_len, tgt_len) 的布尔张量，True = 可关注
    """
    # Step 1: 构造 padding 掩码（key 维度）
    # (batch, tgt_len) → (batch, 1, 1, tgt_len)
    pad_keys = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)

    # Step 2: 构造因果掩码
    # (1, tgt_len, tgt_len) → 增加 batch 维度 → (1, 1, tgt_len, tgt_len)
    causal = subsequent_mask(tgt.size(1), tgt.device).unsqueeze(0)

    # Step 3: 按位与（逻辑与）组合两种掩码
    # pad_keys(batch, 1, 1, tgt_len) & causal(1, 1, tgt_len, tgt_len)
    # → 广播 → (batch, 1, tgt_len, tgt_len)
    return pad_keys & causal