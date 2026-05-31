"""
词表（Vocabulary）模块：实现 token 字符串与整数 id 之间的双向映射。

词表是 NLP 模型的基础组件，负责将人类可读的文本转换为模型可处理的数字序列。

核心功能：
  1. 双向映射：stoi（string → id）和 itos（id → string）
  2. 特殊符号：PAD(0) 填充、UNK(1) 未知词、BOS(2) 句首、EOS(3) 句尾
  3. 词表构建：从语料中统计词频，自动生成词表
  4. 序列化：支持词表的保存和恢复（配合 checkpoint）

特殊符号说明：
  - <pad> (PAD, id=0): 填充符号，将不等长序列补齐到相同长度
  - <unk> (UNK, id=1): 未知词，训练时未出现的词在推理时映射到此
  - <bos> (BOS, id=2): 句子开始标记，解码器以此开始自回归生成
  - <eos> (EOS, id=3): 句子结束标记，解码器生成此标记时停止
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, Iterator, Sequence


class Vocab:
    """
    词表类：字符串与整数 id 的双向映射，内置四个特殊符号。

    属性：
      - stoi: dict[str, int]，词 → id 映射
      - itos: list[str]，id → 词映射（下标即 id）

    使用示例：
      >>> v = Vocab({"<pad>":0, "hello":4, "world":5}, ["<pad>", "<unk>", "<bos>", "<eos>", "hello", "world"])
      >>> v.encode(["hello", "world"])        # [4, 5]
      >>> v.decode([2, 4, 5, 3])              # "hello world"
    """

    # 类常量：特殊符号的固定 id
    PAD, UNK, BOS, EOS = 0, 1, 2, 3

    def __init__(self, stoi: dict[str, int], itos: list[str]) -> None:
        """
        Args:
            stoi: 词 → id 的映射字典
            itos: id → 词的列表（下标即 id）
                  —— 由于 itos[i] 就是 id=i 对应的词，stoi 其实可以从中自动生成
        """
        self.stoi = stoi
        self.itos = itos

    def __len__(self) -> int:
        """返回词表大小（含特殊符号）。"""
        return len(self.itos)

    def encode(self, tokens: Sequence[str], max_len: int | None = None) -> list[int]:
        """
        将 token 字符串列表编码为 id 列表。

        编码规则：
          - 词表中有该词 → 使用对应 id
          - 词表中没有该词 → 映射为 UNK(id=1)，避免 OOV（Out-of-Vocabulary）错误
          - 如果指定了 max_len，截断多余 token

        Args:
            tokens:  分词后的 token 字符串列表（如 ["the", "cat", "runs"]）
            max_len: 可选，最大保留的 token 数（不含 BOS/EOS），超出则截断

        Returns:
            token id 列表

        示例：
            {"the": 5, "cat": 8}.encode(["the", "cat"]) → [5, 8]
            {}.encode(["xzzz"]) → [1]  (UNK)
        """
        ids = [self.stoi.get(t, self.UNK) for t in tokens]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        """
        将 token id 列表还原为可读的自然语言字符串（空格分隔）。

        解码规则：
          - PAD(id=0) 和 BOS(id=2)：始终跳过
          - EOS(id=3)：遇到则立即停止解码（即使后面还有内容）
          - UNK(id=1)：如果 skip_special=True 则跳过

        使用场景：
          - 测试时，将模型生成的 id 序列还原为人类可读的译文
          - BLEU 评测时，将模型输出转为字符串与参考译文比较

        Args:
            ids:          token id 序列
            skip_special: 是否跳过 UNK 和特殊符号（默认 True）

        Returns:
            空格分隔的自然语言字符串
        """
        out: list[str] = []
        for i in ids:
            # PAD 和 BOS 始终跳过（不影响语义）
            if i in (self.PAD, self.BOS):
                continue
            # 遇到 EOS 表明句子结束，停止解码
            if i == self.EOS:
                break
            # UNK 在 skip_special 模式下跳过
            if i == self.UNK and skip_special:
                continue
            # 合法 id → 取对应字符串
            if 0 <= i < len(self.itos):
                out.append(self.itos[i])
        return " ".join(out)


# ============================================================================
# 词表构建函数
# ============================================================================
def build_vocab_from_tokens(
    token_stream: Iterable[list[str]],
    min_freq: int = 2,
    max_size: int = 8000,
    specials_first: tuple[str, ...] = ("<pad>", "<unk>", "<bos>", "<eos>"),
) -> Vocab:
    """
    从 token 流中统计词频，按频率排序构建词表。

    构建流程：
      1. 遍历所有 token 流，统计每个词的出现次数（Counter）
      2. 将特殊符号固定放在词表最前面（id 0~3）
      3. 按词频从高到低，依次收录到词表
      4. 跳过词频低于 min_freq 的词（视为罕见/噪声词）
      5. 词表大小达到 max_size 时停止收录

    为什么限制词表大小？
      - 控制模型参数量（Embedding 矩阵大小为 vocab_size × d_model）
      - 低频词对模型贡献小，收录反而增加噪声
      - 被排除的词在 encode 时会自动映射为 UNK

    Args:
        token_stream:   可迭代的 token 列表（每个元素是一句话的分词结果）
                       如：[["the", "cat"], ["a", "dog"], ...]
        min_freq:       最低词频阈值，低于此值的词不收录
        max_size:       词表上限（含特殊符号，通常 8000~32000）
        specials_first: 固定排在词表最开头的特殊符号元组

    Returns:
        构建完成的 Vocab 实例
    """
    counter: Counter[str] = Counter()
    # 统计词频
    for toks in token_stream:
        counter.update(toks)

    # 特殊符号优先，占据 id 0~3
    itos: list[str] = list(specials_first)

    # 按词频降序遍历，收录高频词
    for w, c in counter.most_common():
        if w in itos:       # 跳过已在词表中的特殊符号
            continue
        if c < min_freq:    # 词频不足，跳过
            continue
        itos.append(w)
        if max_size and len(itos) >= max_size:  # 达到上限
            break

    # 构建双向映射
    stoi = {s: i for i, s in enumerate(itos)}
    return Vocab(stoi, itos)


# ============================================================================
# 分词器
# ============================================================================
def whitespace_tokenizer(s: str) -> list[str]:
    """
    按空白符分词：将句子按空格/换行等切分为 token 列表。

    这是最简单的分词方式，适用于：
      - 英文/德文等以空格分隔的语言
      - 已经过预分词的语料（如 Multi30k）

    局限性：
      - 无法处理标点符号粘在词尾的情况（如 "cat." → ["cat."]）
      - 不适用于中文/日文等无空格分隔的语言

    Args:
        s: 输入句子字符串

    Returns:
        token 字符串列表
    """
    return s.strip().split()


# ============================================================================
# 词表序列化 / 反序列化（配合 checkpoint）
# ============================================================================
def vocab_to_dict(v: Vocab) -> dict:
    """
    将词表序列化为纯 Python 字典，便于存入 checkpoint。

    只保存 itos 列表（因为 stoi 可从 itos 重建），
    减小 checkpoint 体积。

    Args:
        v: Vocab 实例

    Returns:
        含 "itos" 键的字典：{"itos": ["<pad>", "<unk>", ..., "hello", ...]}
    """
    return {"itos": v.itos}


def vocab_from_dict(d: dict) -> Vocab:
    """
    从 checkpoint 字典中恢复词表实例。

    Args:
        d: vocab_to_dict 产出的字典

    Returns:
        重建的 Vocab 实例（stoi 自动从 itos 计算）
    """
    itos = list(d["itos"])
    stoi = {s: i for i, s in enumerate(itos)}
    return Vocab(stoi, itos)


# ============================================================================
# 辅助函数
# ============================================================================
def yield_tokens(
    lines: Iterator[str], tokenizer: Callable[[str], list[str]]
) -> Iterator[list[str]]:
    """
    逐行读取文本并用 tokenizer 转为 token 列表。

    这是 build_vocab_from_tokens 的输入准备函数，
    避免一次性将所有文本加载到内存。

    使用示例：
        with open("train.en") as f:
            token_stream = yield_tokens(f, whitespace_tokenizer)
            vocab = build_vocab_from_tokens(token_stream)

    Args:
        lines:     文本行的迭代器（如文件对象的逐行读取）
        tokenizer: 分词函数（如 whitespace_tokenizer）

    Yields:
        每行分词后的 token 字符串列表
    """
    for line in lines:
        yield tokenizer(line)