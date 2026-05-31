# Transformer 课程项目（对齐 `transformer_project.pdf`）

从零实现的 **Encoder–Decoder Transformer**（PyTorch），满足课程对 **模块拆解、掩码、Tokenizer、翻译任务、超参、优化器与损失、参数量统计、目录结构** 的要求。仓库内已在 **`docs/transformer_project.pdf`** 附带作业说明，便于对照。

## 小组成员与分工

| 成员 | 主要分工 |
|------|----------|
| 刘承龙 | 项目汇总、代码整理、PPT 讲解，以及除组内其他成员负责内容外的整体统筹。 |
| 韩玉轩 | 代码编写与运行，配合完成训练、测试和结果检查。 |
| 潘旭 | PPT 制作与展示材料排版。 |

---

## 1. 与 PDF 要求的对应关系

### 4.2「Transformer from scratch README」模块清单

| 序号 | 要求 | 实现位置 |
|------|------|----------|
| 1 | Input Embedding | `model.py` → `Embeddings` |
| 2 | Positional Encoding | `model.py` → `PositionalEncoding` |
| 3 | Scaled Dot-Product Attention | `model.py` → `scaled_dot_product_attention` |
| 4 | Multi-Head Attention | `model.py` → `MultiHeadAttention` |
| 5 | Position-wise Feed Forward | `model.py` → `PositionwiseFeedForward` |
| 6 | Encoder Layer | `model.py` → `EncoderLayer` |
| 7 | Decoder Layer | `model.py` → `DecoderLayer` |
| 8 | Transformer Encoder | `model.py` → `Encoder` |
| 9 | Transformer Decoder | `model.py` → `Decoder` |
| 10 | 完整 Transformer | `model.py` → `Transformer`、`build_model` |
| 11 | Mask（padding + subsequent） | `data/masks.py`；批数据在 `data/parallel_dataset.py` / `data/reverse_dataset.py` |

### 4.3 PPT / 讲解要点（代码侧支撑）

- **Q、K、V**：见 `MultiHeadAttention` 与 `scaled_dot_product_attention`。
- **Multi-Head**：多头拆分与拼接在同一模块内完成。
- **Encoder / Decoder**：`Encoder` / `Decoder` 堆叠，Decoder 含 **self-attn + cross-attn**。
- **Mask**：`make_src_mask`（padding）、`make_tgt_mask`（padding ∧ 因果下三角）。
- **Loss / Optimizer**：`train.py` 使用 `NLLLoss`（配合 `Generator` 的 `log_softmax`）与 **Adam**；支持梯度裁剪。

### 5.1–5.2 数据集与流程

- 课程推荐 **Multi30k** 或 **IWSLT**。
  - **默认（真实 Multi30k）**：运行 `python scripts/download_multi30k.py`，从官方仓库 [multi30k/dataset](https://github.com/multi30k/dataset) 拉取德英语料到 **`data/multi30k/`**（`configs/default.yaml` 已指向该目录）。
  - **备选**：`scripts/build_sample_corpus.py` 生成 **`data/corpus/`** 小型合成语料，用于离线快速调试；使用时把 `configs/default.yaml` 的 `paths` 改回 `data/corpus/` 即可。

### 5 节超参数

均在 **`configs/default.yaml`**（或 `configs/quick.yaml`、`configs/reverse.yaml`）中配置，包括：`d_model`、`num_heads`、`num_layers`、`d_ff`、`batch_size`、`lr`、`dropout`、`epochs` 等。

### 参数量统计（PDF 第 6 节）

训练开始时 `train.py` 会调用 `print_param_count()` 打印参数量；已保存 checkpoint 的实测统计如下：

| 配置 / checkpoint | 词表与规模 | 总参数量 | 可训练参数量 |
|---|---|---:|---:|
| `checkpoints/best_translation.pt` | Multi30k，`d_model=256`，3 层 Encoder + 3 层 Decoder，源词表 8000、目标词表 7964 | 11,664,156 | 11,664,156 |
| `checkpoints/best_corpus_quick.pt` | 合成语料 quick，`d_model=128`，2 层 Encoder + 2 层 Decoder，源词表 45、目标词表 43 | 943,019 | 943,019 |
| `checkpoints/best_reverse.pt` | 序列反转任务，`d_model=128`，3 层 Encoder + 3 层 Decoder，词表 64 | 1,413,696 | 1,413,696 |

Multi30k 默认模型的主要参数组成：

| 组成 | 参数量 | 说明 |
|---|---:|---|
| Source Embedding | 2,048,000 | `8000 × 256` |
| Target Embedding | 2,038,784 | `7964 × 256` |
| 单个 Multi-Head Attention | 263,168 | Q/K/V/输出 4 个 `256→256` Linear，含 bias |
| 单个 FFN | 525,568 | `256→1024→256`，含 bias |
| Encoder 堆叠 | 2,369,792 | 3 层 Encoder + 最终 LayerNorm |
| Decoder 堆叠 | 3,160,832 | 3 层 Decoder，每层含 self-attn、cross-attn、FFN |
| Generator | 2,046,748 | `256→7964` 输出投影，含 bias |

参数量统计代码位置：`train.py` -> `print_param_count()`；参数分析文字见 `report/code_explanation.md` 和 `report/report_draft.md`。

---

## 2. 目录结构（对齐 PDF 第 7 节）

```text
project/
├── README.md
├── requirements.txt
├── docs/                    # 作业 PDF 与论文材料
├── train.py
├── test.py
├── model.py
├── inference.py
├── configs/
│   ├── default.yaml
│   ├── quick.yaml          # 小模型、少 epoch，CPU 冒烟
│   └── reverse.yaml        # 序列反转玩具任务
├── data/
│   ├── __init__.py
│   ├── masks.py
│   ├── vocab.py
│   ├── parallel_dataset.py
│   ├── reverse_dataset.py
│   ├── corpus/             # 示例平行语料（脚本生成）
│   └── multi30k/           # 真实 Multi30k 文本
├── scripts/
│   ├── download_multi30k.py
│   ├── build_sample_corpus.py
│   ├── plot_loss.py
│   └── run_all.py
├── checkpoints/            # best.pt、best_translation.pt、best_reverse.pt、best_corpus_quick.pt
├── results/                # translation/、reverse/、corpus_quick/、run_summary.json
├── figures/                # Loss 曲线、architecture/ 架构图、project_brief/ 作业要求截图
├── PPT/                    # Transformer_Project_PPT.pptx
├── poster/                 # poster.png / poster.pptx / poster_A1.pdf / QR / poster_prompt.md
└── report/                 # Word 报告、代码讲解、自查表
```

---

## 3. 环境

```bash
pip install -r requirements.txt
```

- **Python**：建议 3.10+；本项目依赖 `torch`、`PyYAML`、`sacrebleu`、`matplotlib`，未强制依赖 `torchtext` 或 `spaCy`。
- **Tokenizer**：默认 **空格分词**（`data/vocab.py`），便于验收；若需 **spaCy** 词级分词，可自行替换 `ParallelTextDataset` 的 tokenizer 回调并在 README 中记录。

---

## 4. 数据准备

### 4.1 下载真实 Multi30k（默认）

```bash
python scripts/download_multi30k.py
```

会写入 `data/multi30k/`：

| 文件 | 说明 | 行数 |
|------|------|------|
| `train.de` / `train.en` | 训练集 | 29,000 |
| `val.de` / `val.en` | 验证集 | 1,014 |
| `test.de` / `test.en` | 测试集（2016 Flickr） | 1,000 |

数据来源：`https://github.com/multi30k/dataset`（Task 1 raw，德→英）。

### 4.2 使用内置示例语料（可选）

```bash
python scripts/build_sample_corpus.py
```

会写入 `data/corpus/train.* / val.* / test.*`；训练时在 `configs/default.yaml` 中将 `paths` 改为 `data/corpus/` 对应文件。

### 4.3 IWSLT

流程相同：准备行对齐双语文件，改 `paths` 即可（模型与训练脚本与语料路径解耦）。

---

## 5. 一键本地运行（推荐）

安装依赖后执行，会自动完成数据检查、Multi30k 训练/评测、反转任务、示例语料 quick 训练，并把结果写入对应目录。请在 `transformer/` 目录下运行下面的命令：

```bash
python scripts/run_all.py
```

约 40 分钟（CPU，Multi30k 8 epoch）。完成后查看：

| 目录 | 内容 |
|------|------|
| `checkpoints/` | `best_translation.pt`、`best_reverse.pt`、`best_corpus_quick.pt` |
| `results/translation/` | `train_log.csv`、`test_bleu.json`、`predictions_sample.txt` |
| `results/reverse/` | `train_log.csv`、`test_metrics.json` |
| `results/corpus_quick/` | quick 语料训练日志与 BLEU |
| `figures/` | `translation_loss.png`、`reverse_loss.png` |
| `results/run_summary.json` | 运行汇总 |

## 6. 训练

下面命令均默认在 `transformer/` 目录下执行。

**翻译任务（默认 `task: translation`）**

```bash
python train.py --config configs/default.yaml
# 或快速冒烟：
python train.py --config configs/quick.yaml
```

**序列反转玩具任务（`configs/reverse.yaml`）**

```bash
python train.py --config configs/reverse.yaml
```

常用覆盖参数：

```bash
python train.py --config configs/default.yaml --epochs 30 --device cuda
```

成功后会保存：

- `checkpoints/best_translation.pt`（Multi30k 翻译）或 `checkpoints/best_reverse.pt`（反转）
- `checkpoints/best_corpus_quick.pt`（`configs/quick.yaml` 的合成语料 quick 训练）
- 同时写入 `checkpoints/best.pt` 为最近一次「当前任务」的最优权重，便于快速覆盖调试。

---

## 7. 测试与 BLEU

翻译任务（默认读取 `checkpoints/best_translation.pt`）：

```bash
python test.py --checkpoint checkpoints/best_translation.pt --device auto
```

反转任务：

```bash
python test.py --checkpoint checkpoints/best_reverse.pt --device auto
```

说明：本文件名为 `test.py`，请使用 **`python test.py`** 运行；若使用 **pytest**，需在配置中排除本文件以免被当作单元测试收集。

翻译评测使用 **sacrebleu** 的 `corpus_bleu`，默认结果会写入 `results/test_bleu.json`；本项目最终汇总结果位于 `results/translation/test_bleu.json`。

若需要重新生成预测样例，可执行：

```bash
python test.py --checkpoint checkpoints/best_translation.pt --device cpu --output-json results/translation/test_bleu.json --output-samples results/translation/predictions_sample.txt
```

---

## 8. 真实训练记录与实验结果

| 内容 | 文件位置 | 当前结果 |
|------|----------|----------|
| Multi30k 真实训练日志 | `results/translation/train_log.csv` | 8 epoch；train loss 4.5257 → 1.8617，val loss 3.3645 → 2.0334 |
| Multi30k 最佳验证结果 | `results/translation/best_val_loss.json` | best val loss = 2.0334，epoch = 8 |
| Multi30k 测试 BLEU | `results/translation/test_bleu.json` | BLEU = 76.52 |
| Multi30k 预测样例 | `results/translation/predictions_sample.txt` | 已用 `best_translation.pt` 重新生成前 50 条 REF/HYP 对照 |
| Multi30k Loss 曲线 | `figures/translation_loss.png` | 由 `scripts/plot_loss.py` 根据训练日志生成 |
| 反转任务日志 | `results/reverse/train_log.csv` | val accuracy 0.000 → 0.926 |
| 反转任务指标 | `results/reverse/test_metrics.json` | val accuracy = 0.9258 |
| quick 合成语料日志 | `results/corpus_quick/train_log.csv` | 2 epoch 冒烟训练 |
| 一键运行汇总 | `results/run_summary.json` | 记录各任务输出路径和总耗时 2391.3 秒 |

说明：`results/translation/` 对应 `configs/default.yaml` 指向的真实 Multi30k 训练；`results/corpus_quick/` 和 `data/corpus/` 只作为离线冒烟/调试对照，不作为主要真实训练结论。

---

## 9. 任务类型配置

在 YAML 顶层设置：

- `task: translation` — 德→英平行文件（`paths` 指向 `.de` / `.en`）。
- `task: reverse` — 合成序列反转，用于快速检验注意力与掩码（见 `configs/reverse.yaml`）。

---

## 10. 参考实现链接（PDF 4.1）

课程列出的参考仓库可作为对照阅读，本实现代码为独立撰写，结构上与 Harvard 注释版 Transformer 教学思路一致，但文件划分按本作业仓库组织。

- `https://github.com/jadore801120/attention-is-all-you-need-pytorch`
- `https://github.com/aladdinpersson/Machine-Learning-Collection`
- 论文：Vaswani et al., *Attention Is All You Need*, NeurIPS 2017。

---

## 11. 最终交付物

| 类型 | 路径 | 说明 |
|------|------|------|
| Project 报告 | `report/Transformer_Project_Report.docx` | 统一参数量为 11,664,156，并补充小组分工与 GitHub 信息。 |
| 答辩 PPT | `PPT/Transformer_Project_PPT.pptx` | 用于课堂展示。 |
| 海报 | `poster/poster.png`, `poster/github_repository_QR.png` | 已生成海报图片与 GitHub 二维码。 |
| 要求自查表 | `report/requirements_checklist.md` | 逐项对照 `transformer_project.pdf` 的完成情况。 |
| 作业说明与论文 | `docs/` | 包含 Project PDF、英文论文与中文参考材料。 |
| 实验结果 | `results/`, `figures/` | 保存 loss、BLEU、预测样例、曲线图和一键运行汇总。 |
| GitHub 链接/二维码 | [GuoDragon/DeepLearning_Transformer](https://github.com/GuoDragon/DeepLearning_Transformer) | 海报使用该链接，并提供 `poster/github_repository_QR.png`。 |

建议以当前 `transformer/` 目录作为 GitHub 仓库根目录提交。

---

## 许可

代码用于课程学习；算法出处见 Vaswani et al., *Attention Is All You Need* (NeurIPS 2017)。
