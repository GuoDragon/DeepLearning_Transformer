"""
生成小型德-英平行语料（行对齐），用于无外网或官方 Multi30k 下载失败时的本地训练验收。

生成策略：
  使用模板句结构随机组合词汇，生成语法正确的德英平行句对：
  - 德语模板：{冠词} {名词} {动词} und ist {形容词} .
  - 英语模板：the {名词} {动词} and is {形容词} .

数据集划分比例：
  - 训练集：85%（约 11900 句）
  - 验证集：10%（约 1400 句）
  - 测试集：5%（约 700 句）

输出文件格式与 Multi30k 完全一致，可直接替换使用。
"""

from __future__ import annotations

import random
from pathlib import Path


def main() -> None:
    """
    随机组合德/英模板句，划分 train/val/test 写入 data/corpus/。

    执行流程：
      1. 定义德语和英语的词汇表（冠词、名词、动词、形容词）
      2. 随机组合生成 14000 对平行句
      3. 按 85%/10%/5% 比例划分训练/验证/测试集
      4. 写入 data/corpus/{train,val,test}.{de,en}
    """
    # 设置随机种子，确保结果可复现
    random.seed(42)

    # 词汇表定义（德英对齐）
    articles = ["der", "die", "das"]  # 德语冠词（阳性/阴性/中性）
    subs_de = ["katz", "hund", "mann", "frau", "kind", "lehrer", "student", "vogel", "fisch", "baum", "auto", "haus"]  # 德语名词
    subs_en = ["cat", "dog", "man", "woman", "child", "teacher", "student", "bird", "fish", "tree", "car", "house"]  # 英语名词
    v_de = ["rennt", "geht", "isst", "schlaft", "liest", "schreibt", "spielt", "schwimmt", "singt", "springt", "arbeitet", "lacht"]  # 德语动词
    v_en = ["runs", "walks", "eats", "sleeps", "reads", "writes", "plays", "swims", "sings", "jumps", "works", "laughs"]  # 英语动词
    a_de = ["klein", "gross", "schnell", "langsam", "gluecklich", "muede", "jung", "alt", "still", "laut", "muede", "nett"]  # 德语形容词
    a_en = ["small", "big", "fast", "slow", "happy", "tired", "young", "old", "quiet", "loud", "tired", "nice"]  # 英语形容词

    # 生成平行句对
    pairs: list[tuple[str, str]] = []
    for _ in range(14000):
        # 随机选择名词、动词、形容词的索引（确保德英对应）
        i, j, k = random.randrange(len(subs_en)), random.randrange(len(v_en)), random.randrange(len(a_en))
        art = random.choice(articles)  # 随机选择冠词
        # 构建德语和英语句子
        de = f"{art} {subs_de[i]} {v_de[j]} und ist {a_de[k]} ."
        en = f"the {subs_en[i]} {v_en[j]} and is {a_en[k]} ."
        pairs.append((de, en))

    # 划分数据集
    n = len(pairs)
    i_train = int(n * 0.85)  # 训练集结束位置（85%）
    i_val = int(n * 0.95)    # 验证集结束位置（95%）

    # 创建输出目录
    corpus_dir = Path("data/corpus")
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # 定义写入函数：将切片内的句对写入文件
    def dump(name: str, sl: slice) -> None:
        """
        将指定切片范围内的平行句对写入文件。

        Args:
            name: 数据集名称（train/val/test）
            sl:   切片对象，指定要写入的句对范围
        """
        chunk = pairs[sl]
        # 写入德语文件
        (corpus_dir / f"{name}.de").write_text("\n".join(p[0] for p in chunk) + "\n", encoding="utf-8")
        # 写入英语文件
        (corpus_dir / f"{name}.en").write_text("\n".join(p[1] for p in chunk) + "\n", encoding="utf-8")

    # 写入三个数据集
    dump("train", slice(0, i_train))      # 训练集：0 ~ 85%
    dump("val", slice(i_train, i_val))    # 验证集：85% ~ 95%
    dump("test", slice(i_val, n))         # 测试集：95% ~ 100%

    print("written:", corpus_dir)


if __name__ == "__main__":
    main()
