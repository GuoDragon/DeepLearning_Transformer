"""
Transformer 推理模块：实现贪心解码算法进行自回归序列生成。

贪心解码是最简单的序列生成策略，每一步选择概率最大的 token。
虽然不如 Beam Search 等方法生成质量高，但实现简单、速度快，适合作为基础推理方法。

解码流程：
  ┌───────────────────────────────────────────────────────────────────┐
  │                        贪心解码流程                               │
  ├───────────────────────────────────────────────────────────────────┤
  │  1. Encoder 编码源序列 → memory (batch, src_len, d_model)        │
  │  2. 初始化 ys = [BOS] (batch, 1)                                 │
  │  3. for t in 1..max_len:                                        │
  │       ├─ 创建 tgt_mask（Causal Mask + Padding Mask）             │
  │       ├─ Decoder 解码 → out (batch, t, d_model)                 │
  │       ├─ Generator 投影 → log_probs (batch, vocab_size)          │
  │       ├─ argmax → nxt_token (batch, 1)                          │
  │       └─ ys = concat(ys, nxt_token)                             │
  │  4. 返回 ys (batch, gen_len)                                     │
  └───────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import torch

from data.masks import make_src_mask, make_tgt_mask


@torch.no_grad()
def greedy_decode(
    model: torch.nn.Module,
    src: torch.Tensor,
    pad_idx: int,
    bos_idx: int,
    eos_idx: int,
    max_len: int,
) -> torch.Tensor:
    """
    贪心解码：自回归生成目标序列，每一步选择概率最大的 token。

    算法流程：
      1. 首先对源序列进行编码，得到 memory 表示
      2. 初始化生成序列为 [BOS]（句子开始符号）
      3. 循环生成下一个 token：
         - 对当前生成的序列添加 Causal Mask（防止看到未来 token）
         - 通过 Decoder 解码
         - 通过 Generator 投影到词表空间
         - 取 argmax 得到概率最大的 token
         - 追加到生成序列
      4. 当所有样本都生成 EOS（句子结束符号）或达到最大长度时停止

    Args:
        model:   Transformer 模型实例，必须包含 encode、decode、generator 方法
        src:     源序列 token id，shape (batch_size, src_len)
        pad_idx: padding 符号的 id（用于构建 padding mask）
        bos_idx: 句子开始符号（Begin Of Sentence）的 id
        eos_idx: 句子结束符号（End Of Sentence）的 id
        max_len: 生成序列的最大长度（包含 BOS）

    Returns:
        生成的目标序列 token id，shape (batch_size, gen_len)，以 BOS 开头

    Example:
        >>> pred = greedy_decode(model, src_tokens, pad_idx=0, bos_idx=2, eos_idx=3, max_len=50)
        >>> # pred[:, 0] == BOS, pred 可能在某个位置出现 EOS 后停止
    """
    # 获取计算设备（与输入 tensor 相同）
    device = src.device

    # Step 1: 对源序列进行编码，得到 memory 表示
    # src_mask: (batch, 1, 1, src_len)，mask掉 padding 位置
    src_mask = make_src_mask(src, pad_idx)
    memory = model.encode(src, src_mask)

    # Step 2: 初始化生成序列，以 BOS 开头
    # ys: (batch_size, 1)，所有样本的生成序列初始化为 [BOS]
    ys = torch.full((src.size(0), 1), bos_idx, dtype=torch.long, device=device)

    # Step 3: 自回归生成循环
    # 最多生成 max_len - 1 个 token（因为已经包含 BOS）
    for _ in range(max_len - 1):
        # 创建目标序列的 mask（Causal Mask + Padding Mask）
        # tgt_mask: (batch, 1, t, t)，确保每个位置只能看到前面的 token
        tgt_mask = make_tgt_mask(ys, pad_idx)

        # Decoder 解码：结合 memory 和已生成的序列
        # out: (batch_size, current_len, d_model)
        out = model.decode(memory, ys, src_mask, tgt_mask)

        # Generator 投影：将最后一个位置的隐藏状态投影到词表
        # 只取最后一个时间步：out[:, -1] -> (batch, d_model)
        # generator 输出：(batch, vocab_size)
        # argmax(dim=-1)：选择概率最大的 token
        nxt = model.generator(out[:, -1]).argmax(dim=-1, keepdim=True)

        # 将新生成的 token 追加到序列末尾
        # ys: (batch_size, current_len + 1)
        ys = torch.cat([ys, nxt], dim=1)

        # 提前终止条件：如果所有样本都生成了 EOS，停止解码
        # squeeze(-1): (batch, 1) -> (batch,)
        if (nxt.squeeze(-1) == eos_idx).all():
            break

    # Step 4: 返回生成的序列
    return ys
