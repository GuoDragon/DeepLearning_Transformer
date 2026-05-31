"""
Transformer (Encoder–Decoder) 从零实现，对齐 Vaswani et al., Attention Is All You Need (2017)。

本文件按 Transformer 模型的数据流顺序组织各模块，从输入嵌入到最终输出，逐层构建完整模型：

  (1)  Embeddings              — 词嵌入：将 token id 映射为稠密向量，并乘以 sqrt(d_model) 缩放
  (2)  PositionalEncoding      — 位置编码：使用正弦/余弦函数为序列注入位置信息
  (3)  scaled_dot_product_attention — 缩放点积注意力：Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
  (4)  MultiHeadAttention      — 多头注意力：并行计算多组注意力，拼接后投影
  (5)  PositionwiseFeedForward — 逐位置前馈网络：两层全连接 + ReLU 激活
  (6)  SublayerConnection      — 子层连接包装器：LayerNorm → 子层 → Dropout → 残差相加
  (7)  EncoderLayer            — 编码器单层：自注意力子层 + 前馈子层
  (8)  DecoderLayer            — 解码器单层：掩码自注意力 + 交叉注意力 + 前馈
  (9)  Encoder                 — 编码器：堆叠 N 个 EncoderLayer + 最终 LayerNorm
  (10) Decoder                 — 解码器：堆叠 N 个 DecoderLayer + 最终 LayerNorm
  (11) Generator               — 输出生成器：线性投影 + log_softmax，映射到目标词表
  (12) Transformer             — 完整模型：Encoder + Decoder + Embeddings + Generator 的组合
  ── 工具函数 ──
  (13) clones                  — 深拷贝模块，用于构造多头注意力和子层连接
  (14) subsequent_mask         — 构造因果掩码（下三角矩阵），保证解码时只能看到已生成的位置
  (15) build_model             — 工厂函数：按超参数组装并初始化完整 Transformer
"""

from __future__ import annotations

import math
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Embeddings — 词嵌入层
# ============================================================================
class Embeddings(nn.Module):
    """
    词嵌入层：将离散的 token id 映射为连续的稠密向量表示。

    与标准 nn.Embedding 不同的是，嵌入结果会乘以 sqrt(d_model) 进行缩放。
    这一缩放操作使得嵌入向量的方差与位置编码的幅度处于同一数量级，
    防止位置编码在相加后主导或淹没嵌入信息。

    数学形式：
        output = Embedding(token_ids) * sqrt(d_model)

    数据流位置：
        src / tgt token ids → Embeddings → 缩放后的向量 → PositionalEncoding
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        """
        Args:
            vocab_size: 词表大小（源语言或目标语言的 token 种类数）
            d_model:   模型嵌入维度（所有子层统一的表示维度）
        """
        super().__init__()
        # 查找表（lookup table）：nn.Embedding 本质上是一个可学习的权重矩阵，
        # 形状为 (vocab_size, d_model)，按索引取值实现"查表"操作
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token id 张量，形状 (batch_size, seq_len)，int64 类型

        Returns:
            嵌入向量，形状 (batch_size, seq_len, d_model)
            —— 每个 token id 被映射为一个 d_model 维向量，并乘以 sqrt(d_model) 缩放
        """
        return self.lut(x) * math.sqrt(self.d_model)


# ============================================================================
# 2. PositionalEncoding — 正弦/余弦位置编码
# ============================================================================
class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码：为序列中的每个位置生成唯一的向量表示。

    Transformer 的注意力机制本身不具备序列顺序感知能力（置换不变性），
    因此需要显式地为每个位置注入位置信息。

    采用 Vaswani et al. (2017) 提出的固定正弦/余弦编码方案：

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    其中 pos 是位置索引，i 是维度索引。偶数维度用 sin，奇数维度用 cos。
    这种方案的优势：
      - 无需额外学习参数
      - 能够外推到训练时未见过的序列长度
      - 相邻位置编码的线性关系便于模型学习相对位置

    数据流位置：
        Embeddings 输出 → + PositionalEncoding → Dropout → Encoder / Decoder 输入
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        """
        Args:
            d_model: 嵌入维度（必须与 Embeddings 的 d_model 一致，才能相加）
            入 + 位置dropout: 作用于（嵌编码）结果后的 dropout 概率
            max_len: 预计算位置编码的最大序列长度（超出此长度需重新计算）
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算位置编码矩阵 pe，形状 (1, max_len, d_model)
        # batch 维度为 1，便于后续广播加法
        pe = torch.zeros(max_len, d_model)

        # position: 形状 (max_len, 1)，即 [0, 1, 2, ..., max_len-1]^T
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: 形状 (d_model/2,)，即 [1/10000^(0/d_model), 1/10000^(2/d_model), ...]
        # 使用 exp(log(...)) 是为了数值稳定性，等价于 1 / 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        # 偶数维度（0, 2, 4, ...）填充 sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维度（1, 3, 5, ...）填充 cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度：(max_len, d_model) → (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # register_buffer：将 pe 注册为模型 buffer（非参数但随模型保存/移动设备）
        # persistent=False 表示不写入 state_dict（可按需重新计算）
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状 (batch_size, seq_len, d_model)——来自 Embeddings

        Returns:
            加上位置编码并经过 dropout 的张量，形状 (batch_size, seq_len, d_model)
        """
        # 截取与当前序列长度匹配的位置编码，直接相加
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ============================================================================
# 3. scaled_dot_product_attention — 缩放点积注意力（核心计算单元）
# ============================================================================
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    缩放点积注意力 —— Transformer 注意力机制的核心计算。

    数学公式：
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    计算步骤：
        1. 计算 Q 与 K 的点积（衡量 query 与各 key 的匹配程度）
        2. 除以 sqrt(d_k) 进行缩放，防止点积过大导致 softmax 梯度消失
        3. （可选）用 mask 屏蔽非法位置（padding token / 因果位置），置为 -inf
        4. 对最后一维做 softmax，得到注意力权重（概率分布）
        5. （可选）对注意力权重施加 dropout 正则化
        6. 加权求和 V，得到注意力输出

    为什么需要缩放因子 sqrt(d_k)？
      假设 Q、K 各分量独立同分布，均值为 0、方差为 1，
      则 Q·K 的方差为 d_k。当 d_k 较大时，点积值落入 softmax 饱和区，
      梯度极小，训练困难。除以 sqrt(d_k) 将方差控制为 1，缓解此问题。

    Args:
        query: 查询张量，形状 (batch, heads, seq_q, d_k)
        key:   键张量，  形状 (batch, heads, seq_k, d_k)
        value: 值张量，  形状 (batch, heads, seq_k, d_k)（seq_v = seq_k）
        mask:  可选的布尔掩码，形状需能与 scores 广播，核心功能是屏蔽某些位置的注意力。
               True 表示保留该位置，False 表示屏蔽（softmax 前置为 -inf）
        dropout: 可选的 dropout 层，作用于注意力权重矩阵

    Returns:
        (注意力输出, 注意力权重)：
          - 输出形状 (batch, heads, seq_q, d_k)
          - 注意力权重形状 (batch, heads, seq_q, seq_k)
    """
    d_k = query.size(-1)

    # Step 1 & 2: 计算缩放后的注意力分数
    # scores 形状: (batch, heads, seq_q, seq_k)
    # key.transpose(-2, -1) 交换最后两维，使 Q 与 K^T 做矩阵乘法
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 3: 掩码屏蔽 —— mask 中 False 的位置置为 -inf，softmax 后概率为 0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Step 4: softmax 归一化，dim=-1 表示在所有 key 位置上形成概率分布
    p_attn = F.softmax(scores, dim=-1)

    # Step 5: 对注意力权重施加 dropout（训练时随机丢弃部分连接）
    if dropout is not None:
        p_attn = dropout(p_attn)

    # Step 6: 加权求和 V，得到注意力输出
    return torch.matmul(p_attn, value), p_attn


# ============================================================================
# 4. MultiHeadAttention — 多头注意力
# ============================================================================
class MultiHeadAttention(nn.Module):
    """
    多头注意力：将 Q、K、V 分别投影到 h 个不同的低维子空间（头），
    在每个子空间内独立计算缩放点积注意力，最后拼接结果并线性投影。

    数学形式：
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
        head_i = Attention(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)

    多头的意义：
      - 不同头可以关注不同子空间的信息（语法、语义、位置等）
      - 类似于 CNN 中多个卷积核提取不同特征
      - 每个头的维度 d_k = d_model / num_heads，总计算量与单头大维度相近

    实现中，4 个线性投影 (W_Q, W_K, W_V, W_O) 各为一个 nn.Linear(d_model, d_model)，
    存储在 self.linears 中（索引 0~2 为 Q/K/V 投影，索引 3 为输出投影）。

    数据流：
        输入 (batch, seq, d_model)
        → 线性投影 + reshape 为 (batch, heads, seq, d_k)
        → scaled_dot_product_attention
        → reshape 回 (batch, seq, d_model)
        → 输出线性投影
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        """
        Args:
            d_model:   模型总维度，必须能被 num_heads 整除
            num_heads: 注意力头数（论文中 base 模型用 8）
            dropout:   注意力权重的 dropout 概率
        """
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 4 个线性层：第 0 个投影 Q，第 1 个投影 K，第 2 个投影 V，第 3 个投影输出
        # 每个都是 d_model → d_model 的线性变换
        self.linears = clones(nn.Linear(d_model, d_model), 4)

        self.dropout = nn.Dropout(dropout)

        # 保存最近一次前向的注意力权重，便于可视化和分析
        self.attn: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: 形状 (batch_size, seq_len_q, d_model)
            key:   形状 (batch_size, seq_len_k, d_model)
            value: 形状 (batch_size, seq_len_v, d_model)，通常 seq_len_v == seq_len_k
            mask:  注意力掩码，形状需与多头注意力 scores 广播匹配

        Returns:
            多头注意力输出，形状 (batch_size, seq_len_q, d_model)

        内部流程说明：
            投影分头 → 每头独立计算注意力 → 拼接多头 → 最终投影
            d_model → (heads × d_k) → d_model
        """
        nbatches = query.size(0)

        def proj(x: torch.Tensor, i: int) -> torch.Tensor:
            """
            第 i 个线性层投影，并 reshape/transpose 以适应多头计算。

            变换过程：
                (batch, seq, d_model)
                → Linear → (batch, seq, d_model)
                → view  → (batch, seq, heads, d_k)
                → transpose → (batch, heads, seq, d_k)

            转置后 heads 维度移到第 2 位，便于批量矩阵乘法中
            各头独立参与计算（PyTorch 会自动广播 batch 和 heads 维度）。
            """
            return (
                self.linears[i](x)
                .view(nbatches, -1, self.num_heads, self.d_k)
                .transpose(1, 2)
            )

        # Step 1: 分别投影 Q、K、V 并分头
        q = proj(query, 0)  # (batch, heads, seq_q, d_k)
        k = proj(key, 1)    # (batch, heads, seq_k, d_k)
        v = proj(value, 2)  # (batch, heads, seq_v, d_k)

        # Step 2: 对每个头计算缩放点积注意力
        # x: (batch, heads, seq_q, d_k), p_attn: (batch, heads, seq_q, seq_k)
        x, p_attn = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        self.attn = p_attn  # 保存注意力权重，供外部访问

        # Step 3: 拼接所有头的输出
        # (batch, heads, seq_q, d_k) → transpose → (batch, seq_q, heads, d_k)
        # → contiguous + view → (batch, seq_q, heads * d_k) = (batch, seq_q, d_model)
        x = (
            x.transpose(1, 2)
            .contiguous()   # 确保张量在内存中是连续存储
            .view(nbatches, -1, self.num_heads * self.d_k)
        )

        # Step 4: 最终输出线性投影 W_O
        return self.linears[-1](x)


# ============================================================================
# 5. PositionwiseFeedForward — 逐位置前馈网络
# ============================================================================
class PositionwiseFeedForward(nn.Module):
    """
    逐位置前馈网络（Position-wise Feed-Forward Network, FFN）。

    对序列中每个位置的表示独立地做相同的两层全连接变换（同一套参数），
    等价于对每个 token 的 d_model 维向量做逐点 MLP。

    结构：
        FFN(x) = Linear_2(Dropout(ReLU(Linear_1(x))))

    数学形式：
        FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2

    其中 W_1 形状为 (d_model, d_ff)，W_2 形状为 (d_ff, d_model)。
    d_ff 通常为 d_model 的 4 倍（论文 base 模型：512 → 2048）。

    为什么需要 FFN？
      - 注意力机制本质上是线性加权组合，缺乏非线性变换能力
      - FFN 为每个位置引入非线性激活，增强模型的表达能力
      - 两层结构先升维再降维，在更高维度空间中做特征变换

    "逐位置" 的含义：
      对 seq_len 个位置，用同一组 W_1、W_2 独立计算（类似 1×1 卷积），
      不同位置之间没有信息交互（信息交互仅发生在注意力层）。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        """
        Args:
            d_model: 输入和输出的维度（模型统一维度）
            d_ff:    中间隐藏层维度（论文中通常 d_ff = 4 * d_model）
            dropout: 第一个 Linear 之后、ReLU 之前的 dropout 概率
        """
        super().__init__()
        # W_1: 升维映射 d_model → d_ff
        self.w_1 = nn.Linear(d_model, d_ff)
        # W_2: 降维映射 d_ff → d_model（恢复原始维度以进行残差连接）
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 形状 (batch_size, seq_len, d_model)

        Returns:
            形状 (batch_size, seq_len, d_model)

        计算流程：
            Linear(d_model→d_ff) → ReLU → Dropout → Linear(d_ff→d_model)
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# ============================================================================
# 6. SublayerConnection — 子层连接包装器（残差 + LayerNorm）
# ============================================================================
class SublayerConnection(nn.Module):
    """
    子层连接包装器：实现 Transformer 中的残差连接 + LayerNorm 模式。

    论文中的两种子层连接方案（我们采用 Post-LN，即论文原始方案）：
      Post-LN:  LayerNorm(x + Dropout(Sublayer(x)))   ← 本实现采用
      Pre-LN:   x + Dropout(Sublayer(LayerNorm(x)))    ← 后续工作的改进

    本实现采用 Pre-LN 方案（LayerNorm 在子层之前），
    实际输出 = x + Dropout(Sublayer(LayerNorm(x)))。

    为什么需要残差连接？
      - 缓解深层网络的梯度消失问题，使梯度能通过恒等路径直接回传
      - 让模型更容易学习"恒等映射"，即使子层学不好也不至于退化

    为什么需要 LayerNorm？
      - 对每个样本的特征维度做归一化（区别于 BatchNorm 对 batch 维度归一化）
      - 在 NLP 中 batch 内各序列长度不一，LayerNorm 更稳定
      - 将输入标准化为零均值、单位方差，加速训练收敛
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        """
        Args:
            d_model: 特征维度（LayerNorm 将在此维度上归一化）
            dropout: 子层输出上的 dropout 概率
        """
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """
        Args:
            x:        子层输入，形状 (batch, seq_len, d_model)
            sublayer: 可调用对象（如 MultiHeadAttention 或 PositionwiseFeedForward），
                     接收 LayerNorm(x) 后的张量作为输入

        Returns:
            残差连接后的输出，形状与 x 相同

        计算公式：
            output = x + Dropout(sublayer(LayerNorm(x)))

        这里 sublayer 是一个 lambda 或 callable，在调用处构造，
        例如：sublayer(x_norm) = self.self_attn(x_norm, x_norm, x_norm, mask)
        """
        # LayerNorm → 子层 → Dropout → 残差相加
        return x + self.dropout(sublayer(self.norm(x)))


# ============================================================================
# 7. EncoderLayer — 编码器单层
# ============================================================================
class EncoderLayer(nn.Module):
    """
    编码器单层（Transformer Encoder 的一个重复单元）。

    结构（按数据流顺序）：
        (1) 多头自注意力子层：   x → LayerNorm → MultiHeadAttention(x, x, x) → Dropout → + x
        (2) 逐位置前馈子层：     x → LayerNorm → FFN → Dropout → + x

    每个子层都配有独立的残差连接和 LayerNorm（通过 SublayerConnection 实现）。

    自注意力（Self-Attention）：
      Q、K、V 全部来自同一个输入序列 x，让每个 token 都能关注序列中所有 token
      （包括自身），从而捕获全局依赖关系。

    mask 的作用：
      传入的 mask 通常为 padding mask，屏蔽 <pad> token 对应的位置，
      使其注意力权重为 0，防止填充符号影响语义表示。
    """

    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ) -> None:
        """
        Args:
            d_model:      模型维度
            self_attn:    多头自注意力模块
            feed_forward: 逐位置前馈网络模块
            dropout:      子层 dropout 概率
        """
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 两个子层连接：第一个包装自注意力，第二个包装前馈网络
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x:    输入序列表示，形状 (batch, src_len, d_model)
            mask: padding 掩码，True=保留，False=屏蔽

        Returns:
            编码后的序列表示，形状 (batch, src_len, d_model)

        计算流程：
            x = SublayerConnection[0](x, lambda: SelfAttention(x, x, x, mask))
            x = SublayerConnection[1](x, FFN)
        """
        # 子层 1: 多头自注意力
        # Q=K=V=x，让每个位置关注所有位置
        x = self.sublayer[0](x, lambda x_: self.self_attn(x_, x_, x_, mask))
        # 子层 2: 逐位置前馈网络
        return self.sublayer[1](x, self.feed_forward)


# ============================================================================
# 8. DecoderLayer — 解码器单层
# ============================================================================
class DecoderLayer(nn.Module):
    """
    解码器单层（Transformer Decoder 的一个重复单元）。

    结构与编码器单层不同，解码器包含三个子层（编码器只有两个）：

        (1) 掩码多头自注意力子层：
            x → LayerNorm → Masked MultiHeadAttention(x, x, x, tgt_mask) → Dropout → + x
            使用因果掩码（causal mask）保证位置 i 只能关注 ≤ i 的位置

        (2) 编码器-解码器交叉注意力子层：
            x → LayerNorm → MultiHeadAttention(x, memory, memory, src_mask) → Dropout → + x
            Q 来自解码器当前层输出，K 和 V 来自编码器输出 (memory)

        (3) 逐位置前馈子层（同编码器）：
            x → LayerNorm → FFN → Dropout → + x

    关键设计：
      - 掩码自注意力：防止解码时"偷看"未来 token，保证自回归生成的正确性
      - 交叉注意力：解码器通过 Q 查询编码器的 K/V，将源语言信息融入目标语言生成
      - 三个子层各有独立的残差连接和 LayerNorm，总共 3 个 SublayerConnection
    """

    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadAttention,
        src_attn: MultiHeadAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ) -> None:
        """
        Args:
            d_model:      模型维度
            self_attn:    解码器掩码自注意力模块
            src_attn:     编码器-解码器交叉注意力模块
            feed_forward: 逐位置前馈网络模块
            dropout:      子层 dropout 概率
        """
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 三个子层连接：自注意力 → 交叉注意力 → 前馈
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)
        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            x:        解码器输入，形状 (batch, tgt_len, d_model)
            memory:   编码器输出（源语言编码记忆），形状 (batch, src_len, d_model)
            src_mask: 源序列 padding 掩码（屏蔽源语言的 <pad> 位置）
            tgt_mask: 目标序列掩码（padding + 因果掩码），
                     确保解码位置 i 只能看到已生成的 0..i 位置

        Returns:
            解码后的隐状态，形状 (batch, tgt_len, d_model)
        """
        # 子层 1: 掩码自注意力 —— Q=K=V=x，使用 tgt_mask 实现因果遮罩
        x = self.sublayer[0](x, lambda x_: self.self_attn(x_, x_, x_, tgt_mask))

        # 子层 2: 交叉注意力 —— Q=x（解码器）, K=V=memory（编码器输出）
        # 解码器通过注意力查询编码器中的相关信息
        x = self.sublayer[1](x, lambda x_: self.src_attn(x_, memory, memory, src_mask))

        # 子层 3: 逐位置前馈网络
        return self.sublayer[2](x, self.feed_forward)


# ============================================================================
# 9. Encoder — 编码器（堆叠 EncoderLayer）
# ============================================================================
class Encoder(nn.Module):
    """
    编码器：堆叠 N 个完全相同的 EncoderLayer，最后过一个 LayerNorm。

    结构：
        Encoder(x) = LayerNorm(EncoderLayer_N ∘ ... ∘ EncoderLayer_1(x))

    输入依次经过每一层编码器层，每层的输出作为下一层的输入。
    最后通过一个 LayerNorm 将输出归一化，得到编码器记忆 (memory)。

    所有层的 mask 相同（padding mask），确保各层都不关注 <pad> token。
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        """
        Args:
            layers: EncoderLayer 的 ModuleList，长度为 n_layers
        """
        super().__init__()
        self.layers = layers
        # 最终 LayerNorm，将堆叠后的输出归一化
        self.norm = nn.LayerNorm(layers[0].d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x:    输入序列表示（Embedding + PositionalEncoding 后的结果）
                 形状 (batch, src_len, d_model)
            mask: padding 掩码

        Returns:
            编码器记忆 (memory)，形状 (batch, src_len, d_model)
            —— 该输出将作为 Decoder 交叉注意力的 K 和 V
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ============================================================================
# 10. Decoder — 解码器（堆叠 DecoderLayer）
# ============================================================================
class Decoder(nn.Module):
    """
    解码器：堆叠 N 个完全相同的 DecoderLayer，最后过一个 LayerNorm。

    结构：
        Decoder(x, memory) = LayerNorm(DecoderLayer_N ∘ ... ∘ DecoderLayer_1(x, memory))

    与编码器的区别：
      - 解码器每层额外接收 encoder memory 和 src_mask，用于交叉注意力
      - 解码器的自注意力使用 tgt_mask（因果掩码 + padding 掩码）

    自回归生成时的行为：
      训练时：一次输入完整的目标序列，用 tgt_mask 保证因果关系
      推理时：逐 token 生成，每次生成后将新 token 拼入序列继续解码
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        """
        Args:
            layers: DecoderLayer 的 ModuleList，长度为 n_layers
        """
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(layers[0].d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            x:        解码器输入表示（目标语言 Embedding + PositionalEncoding）
                     形状 (batch, tgt_len, d_model)
            memory:   编码器输出，形状 (batch, src_len, d_model)
            src_mask: 源序列 padding 掩码
            tgt_mask: 目标序列掩码（padding + causal）

        Returns:
            解码器隐状态，形状 (batch, tgt_len, d_model)
            —— 该输出将传入 Generator 生成词表概率分布
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ============================================================================
# 11. Generator — 输出生成器
# ============================================================================
class Generator(nn.Module):
    """
    输出生成器：将解码器的 d_model 维隐状态映射为目标词表上的 log 概率分布。

    结构：
        Linear(d_model → tgt_vocab) → log_softmax

    数学形式：
        output = log_softmax(x @ W_proj + b)

    其中 W_proj 形状为 (d_model, tgt_vocab)。注意论文中 Generator 的权重矩阵
    与目标语言 Embedding 的权重矩阵是共享的（weight tying），但本实现暂未强制共享。

    为什么使用 log_softmax 而非 softmax？
      - 配合 nn.NLLLoss 使用，等价于 CrossEntropyLoss
      - log 空间计算更数值稳定，避免 softmax 的指数溢出问题
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Args:
            d_model:    解码器输出维度（模型维度）
            vocab_size: 目标语言词表大小
        """
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 解码器输出隐状态，形状 (..., d_model)

        Returns:
            log 概率分布，形状 (..., vocab_size)
            —— 对最后一维做 log_softmax
        """
        return F.log_softmax(self.proj(x), dim=-1)


# ============================================================================
# 12. Transformer — 完整 Encoder–Decoder 模型
# ============================================================================
class Transformer(nn.Module):
    """
    完整的 Transformer Encoder–Decoder 模型，将前述所有模块组合为一个端到端系统。

    组件组合：
        Transformer:
          ├── src_embed:  nn.Sequential(Embeddings, PositionalEncoding)  —— 源语言端
          ├── tgt_embed:  nn.Sequential(Embeddings, PositionalEncoding)  —— 目标语言端
          ├── encoder:    Encoder(EncoderLayer × N)                       —— 编码
          ├── decoder:    Decoder(DecoderLayer × N)                       —— 解码
          └── generator:  Generator                                       —— 输出

    训练时数据流：
        src → src_embed → Encoder → memory  ─────────────────┐
        tgt → tgt_embed → Decoder(memory) → hidden → Generator → log_probs
                                                              │
                                               loss = NLLLoss(log_probs, tgt_labels)

    推理时数据流（自回归解码）：
        src → src_embed → Encoder → memory
        tgt 从 <bos> 开始，每步：
          tgt → tgt_embed → Decoder(memory) → Generator → 采样/greedy → 下一个 token
          将新 token 拼入 tgt，继续直到 <eos> 或达到最大长度
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Module,
        tgt_embed: nn.Module,
        generator: Generator,
    ) -> None:
        """
        Args:
            encoder:    编码器实例
            decoder:    解码器实例
            src_embed:  源语言嵌入模块（Embeddings + PositionalEncoding 的 Sequential）
            tgt_embed:  目标语言嵌入模块（Embeddings + PositionalEncoding 的 Sequential）
            generator:  输出生成器
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        仅运行编码器：源语言嵌入 + 编码器前向。

        Args:
            src:      源语言 token id 序列，形状 (batch, src_len)
            src_mask: 源序列 padding 掩码

        Returns:
            编码器记忆 (memory)，形状 (batch, src_len, d_model)
        """
        # src_embed 内部包含 Embeddings（token→向量） + PositionalEncoding（+位置信息）
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        在给定编码器记忆的条件下运行解码器。

        Args:
            memory:   编码器输出，形状 (batch, src_len, d_model)
            tgt:      目标语言 token id 序列，形状 (batch, tgt_len)
            src_mask: 源序列 padding 掩码
            tgt_mask: 目标序列掩码

        Returns:
            解码器隐状态，形状 (batch, tgt_len, d_model)
            —— 可继续传给 Generator 获得词表 log 概率
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        训练时前向传播（Teacher Forcing 模式）。

        一次性将整个源序列和目标序列输入模型，得到解码器隐状态。
        外部需再经过 Generator 得到 log 概率，并与真实标签计算 loss。

        Args:
            src:      源语言 token id 序列，形状 (batch, src_len)
            tgt:      目标语言 token id 序列，形状 (batch, tgt_len)
            src_mask: 源序列 padding 掩码
            tgt_mask: 目标序列掩码（padding + causal）

        Returns:
            解码器隐状态，形状 (batch, tgt_len, d_model)

        数据流：
            src → src_embed → Encoder ─┐
            tgt → tgt_embed ───────────┤→ Decoder → hidden
        """
        return self.decode(self.encode(src, src_mask), tgt, src_mask, tgt_mask)


# ============================================================================
# 工具函数
# ============================================================================

# ---------------------------------------------------------------------------
# 13. clones — 模块克隆工具
# ---------------------------------------------------------------------------
def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    深拷贝同一模块 n 份，组成 nn.ModuleList。

    用途：
      - MultiHeadAttention 中：克隆 4 个 Linear(d_model, d_model)（Q、K、V、O 投影）
      - EncoderLayer 中：克隆 2 个 SublayerConnection（自注意力 + 前馈）
      - DecoderLayer 中：克隆 3 个 SublayerConnection（自注意力 + 交叉注意力 + 前馈）

    注意：深拷贝意味着 n 个模块是独立的（不共享参数），各自有自己的权重。

    Args:
        module: 要克隆的模块实例
        n:      克隆份数

    Returns:
        包含 n 个独立模块副本的 ModuleList
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


# ---------------------------------------------------------------------------
# 14. subsequent_mask — 因果掩码构造
# ---------------------------------------------------------------------------
def subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    """
    构造解码器因果掩码（下三角矩阵）。

    掩码形式（以 size=4 为例）：
        [[True,  False, False, False],
         [True,  True,  False, False],
         [True,  True,  True,  False],
         [True,  True,  True,  True ]]

    含义：
      - 位置 i 只能关注 ≤ i 的位置（即已生成的历史 token）
      - 对角线及其下方为 True（可见），上方为 False（屏蔽）
      - 保证自回归解码时不会"偷看"未来的 token

    实现方式：
      构造上三角全 1 矩阵 → 取反 → 得到下三角 True 矩阵
      等价于 torch.tril(torch.ones(...))，此处用 ~triu 更直观地表达"不允许看到未来"

    Args:
        size:   目标序列长度
        device: 张量所在设备（CPU / CUDA）

    Returns:
        形状 (1, size, size) 的布尔张量，True=可见，False=屏蔽
        —— 第 0 维为 1 用于广播到 batch 维度
    """
    attn_shape = (1, size, size)
    # triu: 上三角（不含对角线偏移 1）为 True，其余 False
    ahead = torch.triu(
        torch.ones(attn_shape, device=device, dtype=torch.bool), diagonal=1
    )
    # 取反：下三角（含对角线）为 True = 可见；上三角为 False = 屏蔽
    return ~ahead


# ---------------------------------------------------------------------------
# 15. build_model — 模型组装工厂
# ---------------------------------------------------------------------------
def build_model(
    src_vocab: int,
    tgt_vocab: int,
    n_layers: int = 3,
    d_model: int = 128,
    d_ff: int = 512,
    num_heads: int = 8,
    dropout: float = 0.1,
    max_len: int = 5000,
) -> Transformer:
    """
    按超参数组装完整 Transformer 模型，并对权重做 Xavier 初始化。

    组装过程：
        1. 创建 n_layers 个 EncoderLayer（每层含独立的自注意力 + FFN）
        2. 创建 n_layers 个 DecoderLayer（每层含独立的自注意力 + 交叉注意力 + FFN）
        3. 分别创建源语言和目标语言的 Embeddings + PositionalEncoding
        4. 创建 Generator 输出层
        5. 将所有组件传入 Transformer 构造函数，组装为完整模型
        6. 对所有权重矩阵（dim > 1）做 Xavier/Glorot 均匀初始化

    Xavier 初始化：
      对于 Linear 层的权重矩阵 W ∈ R^(fan_in × fan_out)：
        W ~ U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
      有助于保持前向传播和反向传播中各层的方差稳定，加速收敛。

    Args:
        src_vocab:  源语言词表大小
        tgt_vocab:  目标语言词表大小
        n_layers:   编码器 / 解码器的层数（两层对称，层数相同）
        d_model:    模型统一维度（所有子层输入的输出维度）
        d_ff:       前馈网络隐藏层维度（通常 = 4 * d_model）
        num_heads:  多头注意力的头数（d_model 必须能被 num_heads 整除）
        dropout:    dropout 比例（应用于注意力权重、FFN、嵌入后的位置编码）
        max_len:    预计算位置编码的最大序列长度

    Returns:
        已初始化的 Transformer 实例，可直接用于训练或推理
    """
    enc_layers: list[EncoderLayer] = []
    dec_layers: list[DecoderLayer] = []

    for _ in range(n_layers):
        # 编码器层：自注意力 + 前馈
        enc_layers.append(
            EncoderLayer(
                d_model,
                MultiHeadAttention(d_model, num_heads, dropout),
                PositionwiseFeedForward(d_model, d_ff, dropout),
                dropout,
            )
        )
        # 解码器层：掩码自注意力 + 交叉注意力 + 前馈
        dec_layers.append(
            DecoderLayer(
                d_model,
                MultiHeadAttention(d_model, num_heads, dropout),  # 掩码自注意力
                MultiHeadAttention(d_model, num_heads, dropout),  # 交叉注意力
                PositionwiseFeedForward(d_model, d_ff, dropout),
                dropout,
            )
        )

    # 源语言和目标语言各自独立的 PositionalEncoding（不共享参数）
    src_pe = PositionalEncoding(d_model, dropout, max_len=max_len)
    tgt_pe = PositionalEncoding(d_model, dropout, max_len=max_len)

    # 组装完整模型
    model = Transformer(
        Encoder(nn.ModuleList(enc_layers)),
        Decoder(nn.ModuleList(dec_layers)),
        # nn.Sequential 将 Embeddings 和 PositionalEncoding 串联：
        # token ids → Embeddings → + PositionalEncoding → 编码器/解码器输入
        nn.Sequential(Embeddings(src_vocab, d_model), src_pe),
        nn.Sequential(Embeddings(tgt_vocab, d_model), tgt_pe),
        Generator(d_model, tgt_vocab),
    )

    # Xavier 初始化：对所有二维及以上的参数（即权重矩阵）做均匀分布初始化
    # 一维参数（如 bias、LayerNorm 的 scale/shift）保持默认初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model