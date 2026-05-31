"""
Transformer 模型评估模块：支持翻译任务 BLEU 评分和序列反转任务准确率计算。

本模块是训练后的模型评估入口，负责：
  1. 加载训练好的模型 checkpoint（包含模型权重、配置、词表）
  2. 根据任务类型选择评估流程：
     - 翻译任务：使用 sacrebleu 计算 corpus-level BLEU 分数
     - 反转任务：计算序列级完全匹配准确率
  3. 执行贪心解码生成预测结果
  4. 保存评估指标到 JSON 文件
  5. 可选保存样例预测结果（翻译任务）

评估流程总览：
  ┌─────────────────────────────────────────────────────────────┐
  │                    模型评估流程                             │
  ├─────────────────────────────────────────────────────────────┤
  │  1. 加载 checkpoint（模型权重 + 配置 + 词表）               │
  │  2. 构建模型并加载权重                                     │
  │  3. 创建测试数据集和 DataLoader                            │
  │  4. model.eval() + torch.no_grad()                        │
  │  5. 贪心解码生成预测序列                                   │
  │  6. 计算评估指标（BLEU / Accuracy）                        │
  │  7. 保存结果到 JSON 文件                                   │
  └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from sacrebleu import corpus_bleu
from torch.utils.data import DataLoader

from data.parallel_dataset import ParallelTextDataset, collate_translation
from data.reverse_dataset import ReverseSequenceDataset, collate_reverse
from data.vocab import Vocab, vocab_from_dict, whitespace_tokenizer
from inference import greedy_decode
from model import build_model


# ============================================================================
# 辅助工具函数
# ============================================================================
def _torch_load(path: Path) -> dict[str, Any]:
    """
    加载 PyTorch checkpoint 文件，兼容不同版本的 weights_only 参数。

    PyTorch 2.0+ 引入了 weights_only 参数，为兼容旧版本，采用 try-except 处理。

    Args:
        path: checkpoint 文件路径

    Returns:
        checkpoint 字典，包含模型权重、配置、词表等
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_yaml(path: Path) -> dict[str, Any]:
    """
    从 YAML 文件加载配置字典。

    Args:
        path: YAML 配置文件路径

    Returns:
        配置字典
    """
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(name: str) -> torch.device:
    """
    解析设备字符串，支持自动检测 CUDA。

    Args:
        name: 设备名，'auto' 表示自动检测（有 GPU 用 GPU，否则 CPU）

    Returns:
        torch.device 实例
    """
    if str(name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ============================================================================
# 翻译任务评估
# ============================================================================
def eval_translation(
    ckpt: dict[str, Any],
    cfg: dict[str, Any],
    device: torch.device,
    out_json: Path,
    out_samples: Path | None = None,
) -> dict[str, Any]:
    """
    在测试集上执行贪心解码并计算 corpus BLEU 分数。

    执行流程：
      1. 从 checkpoint 恢复源/目标词表
      2. 加载测试数据集（德英平行语料）
      3. 构建模型并加载训练权重
      4. 对测试集每个样本执行贪心解码
      5. 使用 sacrebleu.corpus_bleu 计算 BLEU-4 分数
      6. 保存结果到 JSON 文件和可选的样例文件

    BLEU 指标说明：
      - score:      BLEU-4 综合分数（0-100）
      - precisions: 1-gram 到 4-gram 的精确率
      - bp:         简短惩罚（Brevity Penalty），惩罚过短的翻译

    Args:
        ckpt:         checkpoint 字典（含模型权重、词表、配置）
        cfg:          配置字典
        device:       计算设备
        out_json:     评估指标输出路径（JSON）
        out_samples:  样例译文输出路径（可选，默认保存前50条）

    Returns:
        包含 BLEU 指标的字典
    """
    # 提取配置参数
    paths = cfg["paths"]
    m = cfg["model"]

    # 加载测试数据路径和参考译文
    test_de = Path(paths["test_src"])  # 德语测试集
    test_en = Path(paths["test_tgt"])  # 英语参考译文
    ref_lines = test_en.read_text(encoding="utf-8").splitlines()

    # 从 checkpoint 恢复词表
    src_vocab = vocab_from_dict(ckpt["src_vocab"])  # 德语词表
    tgt_vocab = vocab_from_dict(ckpt["tgt_vocab"])  # 英语词表

    # 创建测试数据集和 DataLoader
    max_len = int(m["max_len"])
    ds = ParallelTextDataset(
        test_de, test_en, src_vocab, tgt_vocab,
        whitespace_tokenizer, whitespace_tokenizer, max_len
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_translation,
    )

    # 构建模型并加载权重
    model = build_model(
        len(src_vocab),
        len(tgt_vocab),
        n_layers=int(m["num_layers"]),
        d_model=int(m["d_model"]),
        d_ff=int(m["d_ff"]),
        num_heads=int(m["num_heads"]),
        dropout=float(m["dropout"]),
        max_len=max_len + 8,  # 预留余量
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()  # 设置为评估模式（关闭 dropout/batchnorm）

    # 贪心解码生成预测译文
    hyps: list[str] = []       # 模型预测的译文列表
    refs_wrap: list[list[str]] = []  # 参考译文列表（sacrebleu 要求的格式）
    idx = 0  # 当前处理的样本索引

    with torch.no_grad():  # 禁用梯度计算，节省内存
        for batch in loader:
            b = batch.src.size(0)
            src = batch.src.to(device)
            # 贪心解码生成目标序列
            pred = greedy_decode(model, src, Vocab.PAD, Vocab.BOS, Vocab.EOS, max_len + 8)
            # 将预测结果解码为文本
            for i in range(b):
                hyps.append(tgt_vocab.decode(pred[i].tolist()))
                refs_wrap.append([ref_lines[idx + i]])
            idx += b

    # 计算 BLEU 分数
    bleu = corpus_bleu(hyps, refs_wrap)
    print(bleu.format())  # 打印格式化的 BLEU 结果

    # 整理指标字典
    metrics = {
        "score": float(bleu.score),        # BLEU-4 分数
        "precisions": [float(x) for x in bleu.precisions],  # 1-4 gram 精确率
        "bp": float(bleu.bp),              # 简短惩罚因子
    }

    # 保存指标到 JSON 文件
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("saved:", out_json)

    # 可选：保存样例译文对比（前50条）
    if out_samples is not None:
        lines = []
        for h, r in zip(hyps[:50], [x[0] for x in refs_wrap[:50]]):
            lines.append(f"REF: {r}\nHYP: {h}\n")
        out_samples.write_text("\n".join(lines), encoding="utf-8")
        print("saved:", out_samples)

    return metrics


# ============================================================================
# 序列反转任务评估
# ============================================================================
def eval_reverse(ckpt: dict[str, Any], cfg: dict[str, Any], device: torch.device, out_json: Path) -> dict[str, Any]:
    """
    在序列反转任务验证集上计算序列级完全匹配准确率。

    序列准确率定义：模型生成的序列（去除 BOS/EOS/PAD）与目标序列完全一致即为正确。

    Args:
        ckpt:      checkpoint 字典
        cfg:       配置字典
        device:    计算设备
        out_json:  评估指标输出路径（JSON）

    Returns:
        包含 val_accuracy 的字典
    """
    # 提取配置参数
    rt = cfg["reverse_toy"]
    m = cfg["model"]
    vocab_size = int(rt["vocab_size"])

    # 创建验证数据集（使用与训练相同的随机种子 + 1，确保数据不重叠）
    val_ds = ReverseSequenceDataset(
        int(rt["val_samples"]), vocab_size,
        seed=int(cfg.get("seed", 42)) + 1
    )
    loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_reverse,
    )

    # 构建模型并加载权重
    model = build_model(
        vocab_size,
        vocab_size,  # 源/目标词表相同
        n_layers=int(m["num_layers"]),
        d_model=int(m["d_model"]),
        d_ff=int(m["d_ff"]),
        num_heads=int(m["num_heads"]),
        dropout=float(m["dropout"]),
        max_len=512,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 计算准确率
    accs: list[float] = []
    pad, eos = Vocab.PAD, Vocab.EOS

    with torch.no_grad():
        for batch in loader:
            src = batch.src.to(device)
            # 贪心解码生成反转序列
            pred = greedy_decode(
                model, src, pad, Vocab.BOS, eos,
                batch.tgt_out.size(1) + 8  # 最大生成长度
            )
            tgt = batch.tgt_out

            # 逐样本计算完全匹配准确率
            for i in range(tgt.size(0)):
                # 处理目标序列：去除 PAD，去掉结尾的 EOS
                t = tgt[i]
                t_list = t[t != pad].tolist()
                if t_list and t_list[-1] == eos:
                    t_list = t_list[:-1]

                # 处理预测序列：跳过 BOS，遇到 EOS 停止
                pl: list[int] = []
                for x in pred[i, 1:].tolist():  # pred[:, 0] 是 BOS
                    if x == eos:
                        break
                    pl.append(x)

                # 判断是否完全匹配
                accs.append(float(t_list == pl))

    # 计算平均准确率
    acc = sum(accs) / max(len(accs), 1)
    print("reverse val acc:", acc)

    # 保存结果
    metrics = {"val_accuracy": acc}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("saved:", out_json)

    return metrics


# ============================================================================
# 主入口函数
# ============================================================================
def main() -> None:
    """
    命令行入口：解析参数并执行模型评估。

    命令行参数：
      --checkpoint:   checkpoint 文件路径（默认 checkpoints/best_translation.pt）
      --config:       YAML 配置文件路径（可选，优先使用 checkpoint 中的 config）
      --device:       计算设备（cpu/cuda/auto）
      --output-json:  评估指标输出路径（JSON）
      --output-samples: 样例译文输出路径（翻译任务专用）

    根据 checkpoint 中的 task 字段自动选择评估任务：
      - task == "reverse": 序列反转准确率评估
      - 其他: 翻译 BLEU 评估
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_translation.pt")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-samples", type=str, default=None)
    args = parser.parse_args()

    # 加载 checkpoint
    ckpt_path = Path(args.checkpoint)
    ckpt = _torch_load(ckpt_path)

    # 获取配置（优先使用命令行指定的 config，否则用 checkpoint 中的 config）
    cfg = ckpt.get("config") or {}
    if args.config is not None:
        cfg = load_yaml(Path(args.config))

    # 解析设备
    device = resolve_device(args.device)

    # 根据任务类型选择评估流程
    task = str(ckpt.get("task") or cfg.get("task", "translation")).lower().strip()
    if task == "reverse":
        out = Path(args.output_json or "results/reverse/test_metrics.json")
        eval_reverse(ckpt, cfg, device, out)
    else:
        out = Path(args.output_json or "results/translation/test_bleu.json")
        samples = Path(args.output_samples) if args.output_samples else Path("results/translation/predictions_sample.txt")
        eval_translation(ckpt, cfg, device, out, samples)


if __name__ == "__main__":
    main()
