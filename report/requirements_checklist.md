# Transformer Project 要求完成度自查表

对照 `docs/transformer_project.pdf` 整理，便于答辩、提交与 GitHub 检查。

## 1. 论文阅读与原理说明

| PDF 要求 | 完成情况 | 对应文件 |
|---|---:|---|
| 阅读 Attention Is All You Need | 已完成 | `docs/Attention_Is_All_You_Need.pdf`, `docs/Attention_Is_All_You_Need_zh_cn.pdf`, `PPT/Transformer_Project_PPT.pptx` |
| 说明 Transformer 解决什么问题 | 已完成 | `report/Transformer_Project_Report.docx`, `PPT/Transformer_Project_PPT.pptx` |
| 对比 Transformer 与 RNN/LSTM/GRU | 已完成 | `report/Transformer_Project_Report.docx` |
| 说明 Encoder、Decoder、Attention、FFN、Residual、LayerNorm | 已完成 | `model.py`, `report/code_explanation.md`, `report/Transformer_Project_Report.docx` |
| 说明 Scaled Dot-Product Attention 与 Multi-Head Attention | 已完成 | `model.py`, `figures/architecture/attention_qkv.svg`, `figures/architecture/multi_head.svg` |
| 说明 Positional Encoding 作用 | 已完成 | `model.py`, `figures/architecture/positional_encoding.svg` |

## 2. 代码实现要求

| PDF 要求模块 | 完成情况 | 对应实现 |
|---|---:|---|
| Input Embedding | 已完成 | `model.py` -> `Embeddings` |
| Positional Encoding | 已完成 | `model.py` -> `PositionalEncoding` |
| Scaled Dot-Product Attention | 已完成 | `model.py` -> `scaled_dot_product_attention` |
| Multi-Head Attention | 已完成 | `model.py` -> `MultiHeadAttention` |
| Position-wise Feed Forward Network | 已完成 | `model.py` -> `PositionwiseFeedForward` |
| Encoder Layer | 已完成 | `model.py` -> `EncoderLayer` |
| Decoder Layer | 已完成 | `model.py` -> `DecoderLayer` |
| Transformer Encoder | 已完成 | `model.py` -> `Encoder` |
| Transformer Decoder | 已完成 | `model.py` -> `Decoder` |
| 完整 Transformer 模型 | 已完成 | `model.py` -> `Transformer`, `build_model` |
| Padding mask 与 subsequent mask | 已完成 | `data/masks.py`, `data/parallel_dataset.py`, `data/reverse_dataset.py` |
| 训练程序 | 已完成 | `train.py` |
| 测试程序 | 已完成 | `test.py` |
| 推理程序 | 已完成 | `inference.py` |
| README 说明参考来源 | 已完成 | `README.md` |

## 3. 代码讲解要求

| PDF 要求 | 完成情况 | 对应文件 |
|---|---:|---|
| 目录结构 | 已完成 | `README.md`, `report/code_explanation.md` |
| 主要 `.py` 文件作用 | 已完成 | `README.md`, `report/code_explanation.md` |
| Transformer 核心类/函数 | 已完成 | `report/code_explanation.md` |
| 数据如何处理 | 已完成 | `data/`, `report/code_explanation.md` |
| 输入输出张量维度 | 已完成 | `report/code_explanation.md`, `report/Transformer_Project_Report.docx` |
| Attention 中 Q/K/V | 已完成 | `model.py`, `report/code_explanation.md` |
| Multi-Head Attention 实现 | 已完成 | `model.py`, `report/code_explanation.md` |
| Encoder 和 Decoder | 已完成 | `model.py`, `report/code_explanation.md` |
| Mask 作用 | 已完成 | `data/masks.py`, `report/code_explanation.md` |
| Loss function 和 optimizer | 已完成 | `train.py` |
| 训练如何执行 | 已完成 | `README.md`, `scripts/run_all.py` |
| 结果如何保存和分析 | 已完成 | `results/`, `figures/`, `report/Transformer_Project_Report.docx` |

## 4. 数据与实验要求

| PDF 要求 | 完成情况 | 对应文件 |
|---|---:|---|
| 数据下载或准备 | 已完成 | `scripts/download_multi30k.py`, `data/multi30k/`, `data/corpus/` |
| Tokenizer 或词表构建 | 已完成 | `data/vocab.py` |
| 模型训练 | 已完成 | `train.py`, `checkpoints/`, `results/translation/train_log.csv` |
| 测试 | 已完成 | `test.py`, `results/translation/test_bleu.json` |
| Loss 曲线 | 已完成 | `figures/translation_loss.png`, `figures/reverse_loss.png` |
| 预测样例 | 已完成 | `results/translation/predictions_sample.txt` 已用 `best_translation.pt` 重新生成 Multi30k 测试集前 50 条 REF/HYP 对照 |
| 结果分析 | 已完成 | `report/Transformer_Project_Report.docx` |
| 模型参数量 | 已完成 | `train.py` 打印；`README.md` 已列出 checkpoint 实测参数量与模块拆分 |
| 超参数说明 | 已完成 | `configs/default.yaml`, `configs/reverse.yaml`, `configs/quick.yaml` |

## 5. 交付物要求

| PDF 要求 | 完成情况 | 对应文件/目录 |
|---|---:|---|
| PPT | 已完成 | `PPT/Transformer_Project_PPT.pptx` |
| 海报 | 已补齐 | `poster/poster.png`, `poster/poster.pptx`, `poster/poster_A1.pdf`, `poster/github_repository_QR.png` |
| GitHub 项目 | 已整理 | 当前 `transformer/` 目录 |
| Project 报告 | 已完成 | `report/Transformer_Project_Report.docx` |
| 代码 | 已完成 | `model.py`, `train.py`, `test.py`, `inference.py`, `data/`, `scripts/` |
| 日志和结果 | 已完成 | `results/`, `figures/` |
| 运行说明 | 已完成 | `README.md` |
| 小组分工说明 | 已完成 | `README.md` |
| GitHub 仓库链接 | 已完成 | [GuoDragon/DeepLearning_Transformer](https://github.com/GuoDragon/DeepLearning_Transformer) |

## 6. 当前仍需提交前核对的问题

| 问题 | 当前状态 | 建议处理 |
|---|---|---|
| GitHub 链接/二维码 | 已补充仓库链接和二维码 | [GuoDragon/DeepLearning_Transformer](https://github.com/GuoDragon/DeepLearning_Transformer)；二维码文件为 `poster/github_repository_QR.png` |
| Word 报告参数量 | 已完成 | `report/Transformer_Project_Report_final.docx` 已统一为 11,664,156，并补充小组分工与 GitHub 信息；原 `Transformer_Project_Report.docx` 保留为旧版备份参考 |
| Markdown 报告参数量 | 已同步修正 | `report/report_draft.md`、`report/code_explanation.md` 已统一为 11,664,156 |
| PPT 文件状态 | 已完成 | `PPT/Transformer_Project_PPT.pptx` 已补充成员姓名和 GitHub URL |
| 成员参与要求 | PDF 写明答辩时每位成员都需要参与讲解或回答问题 | 即使 README 已写分工，课堂展示也应准备每位成员的发言/问答点 |

## 7. 当前建议提交目录

建议提交或上传 GitHub 时，以 `transformer/` 目录作为项目根目录。根目录外的 `figures_tmp/`、旧版草稿和模板属于工作过程文件，不需要作为最终项目内容提交。
