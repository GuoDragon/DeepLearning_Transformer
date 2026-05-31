# GPT Image 2 海报生成 Prompt（参考 README.zh-CN.md 与课程 PDF）

```text
# ROLE
你是一个严谨的学术海报设计师和信息可视化设计师。请生成一张《深度学习》课程 Project 的 A1 竖版学术海报。海报必须像真实课程答辩海报，而不是通用 AI 艺术图。

# OUTPUT_SCHEMA
format: A1 vertical academic poster
orientation: portrait
resolution: 4K or higher, print-ready
language: Chinese main text, keep necessary English technical terms
style_category: Charts & Infographics + Posters & Typography
layout: structured infographic, clear academic hierarchy, readable from distance
must_be_legible: true

# PROJECT_IDENTITY
main_title: Transformer 论文阅读、代码复现与实验分析
subtitle: Attention Is All You Need / Encoder-Decoder Transformer from Scratch
course: 深度学习课程 Project
repository: github.com/GuoDragon/DeepLearning_Transformer
team_members: 刘承龙、韩玉轩、潘旭
work_division:
  - 刘承龙：项目汇总、代码整理、PPT 讲解与整体统筹
  - 韩玉轩：代码编写与运行
  - 潘旭：PPT 制作与展示材料排版

# COURSE_REQUIREMENT_ALIGNMENT
海报必须体现以下内容，缺一不可：
1. 项目标题
2. 小组成员
3. Transformer 简介
4. 模型结构图
5. 核心模块说明
6. 代码复现流程
7. 数据集介绍
8. 实验设置
9. 训练结果
10. 结论与收获
11. GitHub 项目链接或二维码占位
12. 小组自己的实验结果，不能只做论文简介

# CANVAS_LAYOUT
Use a three-zone A1 poster grid:
- Top header band: title, subtitle, course label, team members, GitHub link.
- Main body: three vertical columns.
  - Left column: paper background, Transformer motivation, comparison with RNN/LSTM/GRU.
  - Center column: large architecture blueprint of Encoder-Decoder Transformer.
  - Right column: code reproduction workflow, dataset, experiment setup.
- Bottom dashboard band: training results, parameter statistics, conclusion, GitHub QR placeholder.

# CENTRAL_VISUAL
Place a large, clean, vector-like Transformer architecture diagram in the center:
Input Tokens -> Input Embedding -> Positional Encoding -> Encoder Stack -> Decoder Stack -> Generator -> Translation Output.
Inside Encoder Stack: Multi-Head Self-Attention, Add & Norm, Feed Forward.
Inside Decoder Stack: Masked Multi-Head Attention, Encoder-Decoder Attention, Feed Forward, Add & Norm.
Show Q, K, V as three colored streams feeding Scaled Dot-Product Attention.
Show padding mask and subsequent mask as small matrix panels.
Use arrows, layered blocks, and attention-head nodes; keep labels readable.

# CONTENT_BLOCKS
block_1_paper_background:
  title: 论文背景
  text: Transformer 用完全基于注意力机制的结构替代循环结构，提升并行计算能力，是现代 NLP 和大语言模型的重要基础。

block_2_core_modules:
  title: 核心模块
  bullets:
    - Input Embedding
    - Positional Encoding
    - Scaled Dot-Product Attention
    - Multi-Head Attention
    - Position-wise Feed Forward Network
    - Encoder Layer / Decoder Layer
    - Padding Mask / Subsequent Mask

block_3_code_reproduction:
  title: 代码复现流程
  pipeline: data preprocessing -> vocabulary/tokenizer -> build_model -> train.py -> test.py -> inference.py -> results/figures
  key_files: model.py, train.py, test.py, inference.py, data/, configs/, results/
  note: PyTorch from scratch implementation; Adam optimizer; NLLLoss with log_softmax.

block_4_dataset_experiment:
  title: 数据集与实验设置
  bullets:
    - Dataset: Multi30k German-English translation
    - Train: 29,000 sentence pairs
    - Validation: 1,014 sentence pairs
    - Test: 1,000 sentence pairs
    - Tokenizer: whitespace tokenizer
    - Task: German -> English translation
    - Training: 8 epochs on CPU

block_5_results:
  title: 训练结果
  metric_cards:
    - Train loss: 4.5257 -> 1.8617
    - Validation loss: 3.3645 -> 2.0334
    - Test BLEU: 76.52
    - Reverse toy validation accuracy: 0.9258
  include_small_chart: draw a simple downward loss curve card labeled train/val loss.

block_6_parameters:
  title: 参数量统计
  metric_cards:
    - Total trainable parameters: 11,664,156
    - Source Embedding: 2,048,000
    - Target Embedding: 2,038,784
    - Single Multi-Head Attention: 263,168
    - Single FFN: 525,568
    - Encoder stack: 2,369,792
    - Decoder stack: 3,160,832
    - Generator: 2,046,748

block_7_conclusion:
  title: 结论与收获
  text: 项目完成了 Encoder-Decoder Transformer 的从零实现、真实数据训练、测试、Loss 可视化、BLEU 评估和参数分析。重点是理解 Transformer 的模型结构、代码实现与训练流程。

block_8_github:
  title: GitHub 项目
  text: github.com/GuoDragon/DeepLearning_Transformer
  qr_placeholder: Draw a clean square placeholder labeled GitHub QR. Do not invent an inaccurate QR code.

# VISUAL_LANGUAGE
palette:
  - deep academic navy background
  - cyan and electric blue technical lines
  - graphite gray panels
  - warm orange highlights for metrics
texture: subtle grid, faint attention matrix pattern, light blueprint glow
shape_language: rounded technical cards, thin connector lines, vector arrows, transparent model blocks
composition: balanced, clean, dense but not crowded
font_direction: bold Chinese title, readable sans-serif body, monospaced code labels

# TEXT_RULES
- Keep all Chinese text sharp and readable.
- Use short labels and metric cards instead of long paragraphs.
- Preserve all numbers exactly.
- Do not add extra team members.
- Do not change the GitHub URL.
- Do not invent datasets, metrics, institutions, or results.
- If any text is too small, simplify wording rather than making unreadable microtext.

# NEGATIVE_PROMPT
No cartoon style, no unrelated robots, no generic stock photo people, no messy cyberpunk, no excessive neon, no distorted Chinese characters, no unreadable pseudo-text, no fake QR code, no wrong formulas, no extra author names, no irrelevant icons, no cluttered layout, no thesis-only poster that ignores code reproduction and experiments.
```
