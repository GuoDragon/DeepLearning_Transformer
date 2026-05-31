"""
数据与掩码工具包。

提供 Transformer 模型所需的数据层支持，包括：
  - vocab.py:              词表（双向映射、词频统计、序列化）
  - masks.py:              掩码构造（padding mask + causal mask）
  - parallel_dataset.py:   平行语料数据集（德英翻译）
  - reverse_dataset.py:    合成序列反转数据集（玩具任务）
"""