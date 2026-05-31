"""
Transformer 训练入口模块：支持德英平行语料翻译任务和序列反转玩具任务。

本模块是整个训练流程的主控入口，负责：
  1. 解析命令行参数和 YAML 配置文件
  2. 根据任务类型（translation/reverse）选择训练流程
  3. 构建数据集、词表、模型、优化器
  4. 执行训练循环（前向传播、反向传播、梯度裁剪、参数更新）
  5. 验证集评估与最优模型保存
  6. 训练日志记录（CSV 格式）

训练流程总览：
  ┌─────────────────────────────────────────────────────────────────────┐
  │                        训练循环（每 epoch）                          │
  ├─────────────────────────────────────────────────────────────────────┤
  │  训练阶段                                                          │
  │    for batch in train_loader:                                      │
  │      ├─ 数据迁移到设备                                             │
  │      ├─ 清空梯度 (opt.zero_grad)                                   │
  │      ├─ 前向传播 (model → generator → loss)                        │
  │      ├─ 反向传播 (loss.backward)                                   │
  │      ├─ 梯度裁剪 (clip_grad_norm_)                                 │
  │      └─ 参数更新 (opt.step)                                        │
  ├─────────────────────────────────────────────────────────────────────┤
  │  验证阶段                                                          │
  │    model.eval()                                                    │
  │    with torch.no_grad():                                           │
  │      for batch in val_loader:                                      │
  │        └─ 计算 val_loss（翻译）或 val_acc（反转）                    │
  ├─────────────────────────────────────────────────────────────────────┤
  │  模型保存                                                          │
  │    if val_loss < best:                                             │
  │      └─ 保存 checkpoint（模型权重、配置、词表等）                    │
  └─────────────────────────────────────────────────────────────────────┘

配置文件说明：
  支持 configs/*.yaml 文件，包含：
    - model:    模型超参数（num_layers, d_model, d_ff, num_heads, dropout）
    - vocab:    词表参数（min_freq, max_size）
    - train:    训练参数（batch_size, lr, epochs, grad_clip）
    - paths:    数据文件路径（train_src, train_tgt, val_src, val_tgt）
    - reverse_toy: 反转任务专用参数（vocab_size, train_samples, steps_per_epoch）
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader

from data.parallel_dataset import ParallelTextDataset, collate_translation
from data.reverse_dataset import ReverseSequenceDataset, collate_reverse, infinite_loader
from data.vocab import Vocab, build_vocab_from_tokens, vocab_to_dict, whitespace_tokenizer
from inference import greedy_decode
from model import build_model


# ============================================================================
# 辅助工具函数
# ============================================================================
def load_yaml(path: Path) -> dict[str, Any]:
    """
    从 YAML 文件加载配置字典。

    YAML 格式示例：
      model:
        num_layers: 3
        d_model: 128
      train:
        batch_size: 32
        lr: 0.0001

    Args:
        path: YAML 文件路径

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


def print_param_count(model: nn.Module) -> None:
    """
    打印模型总参数量和可训练参数量（用于作业报告）。

    Args:
        model: PyTorch 模型实例
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters:", total)
    print("Trainable parameters:", trainable)


def checkpoint_basename(results_dir: Path) -> str:
    """
    根据结果子目录名返回对应的 checkpoint 文件名。

    映射规则：
      - results/translation → best_translation.pt
      - results/reverse → best_reverse.pt
      - results/corpus_quick → best_corpus_quick.pt
      - 其他 → best.pt

    Args:
        results_dir: 结果输出目录

    Returns:
        checkpoint 文件名
    """
    mapping = {
        "translation": "best_translation.pt",
        "reverse": "best_reverse.pt",
        "corpus_quick": "best_corpus_quick.pt",
    }
    return mapping.get(results_dir.name, "best.pt")


# ============================================================================
# 核心训练函数
# ============================================================================
def run_step_nll(model: nn.Module, criterion: nn.Module, batch) -> torch.Tensor:
    """
    单步前向传播 + NLLLoss 计算。

    执行流程：
      1. 模型前向 → decoder 输出 (batch, tgt_len, d_model)
      2. Generator 投影 → log_probs (batch, tgt_len, vocab_size)
      3. reshape → (batch*tgt_len, vocab_size)
      4. 与 tgt_out (batch*tgt_len) 计算 NLLLoss

    Args:
        model:      Transformer 模型
        criterion:  损失函数（nn.NLLLoss，ignore_index=PAD）
        batch:      TranslationBatch 或 ReverseBatch

    Returns:
        标量损失张量
    """
    out = model(batch.src, batch.tgt_in, batch.src_mask, batch.tgt_mask)
    logits = model.generator(out)
    # reshape 以适应 NLLLoss 的输入要求
    return criterion(logits.reshape(-1, logits.size(-1)), batch.tgt_out.reshape(-1))


# ============================================================================
# 翻译任务训练
# ============================================================================
def train_translation(
    cfg: dict[str, Any],
    ckpt_dir: Path,
    results_dir: Path,
) -> None:
    """
    德英平行语料翻译训练流程。

    完整流程：
      1. 加载配置和数据路径
      2. 从训练语料构建源/目标词表
      3. 创建 Dataset 和 DataLoader
      4. 构建 Transformer 模型并初始化权重
      5. 配置 Adam 优化器和 NLLLoss
      6. 训练循环（训练 + 验证 + 日志记录）
      7. 保存最优模型到 checkpoint

    Args:
        cfg:          配置字典
        ckpt_dir:     checkpoint 保存目录
        results_dir:  训练日志和指标输出目录
    """
    # 从配置中提取各部分参数
    paths = cfg["paths"]
    m = cfg["model"]
    vc = cfg["vocab"]
    tr = cfg["train"]
    device = resolve_device(cfg.get("device", "auto"))
    torch.manual_seed(int(cfg.get("seed", 42)))

    # 解析数据文件路径
    train_de = Path(paths["train_src"])
    train_en = Path(paths["train_tgt"])
    val_de = Path(paths["val_src"])
    val_en = Path(paths["val_tgt"])

    # Step 1: 构建词表（仅用训练数据）
    # 源语言词表（德语）
    src_vocab = build_vocab_from_tokens(
        (whitespace_tokenizer(line) for line in train_de.read_text(encoding="utf-8").splitlines()),
        min_freq=int(vc["min_freq"]),
        max_size=int(vc["max_size"]),
    )
    # 目标语言词表（英语）
    tgt_vocab = build_vocab_from_tokens(
        (whitespace_tokenizer(line) for line in train_en.read_text(encoding="utf-8").splitlines()),
        min_freq=int(vc["min_freq"]),
        max_size=int(vc["max_size"]),
    )

    # Step 2: 创建数据集
    tok = whitespace_tokenizer
    max_len = int(m["max_len"])
    train_ds = ParallelTextDataset(train_de, train_en, src_vocab, tgt_vocab, tok, tok, max_len)
    val_ds = ParallelTextDataset(val_de, val_en, src_vocab, tgt_vocab, tok, tok, max_len)

    # Step 3: 创建 DataLoader
    bs = int(tr["batch_size"])
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=int(tr.get("num_workers", 0)),
        collate_fn=collate_translation,
        drop_last=True,  # 丢弃最后一个不完整的 batch
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=int(tr.get("num_workers", 0)),
        collate_fn=collate_translation,
    )

    # Step 4: 构建模型
    model = build_model(
        len(src_vocab),
        len(tgt_vocab),
        n_layers=int(m["num_layers"]),
        d_model=int(m["d_model"]),
        d_ff=int(m["d_ff"]),
        num_heads=int(m["num_heads"]),
        dropout=float(m["dropout"]),
        max_len=max_len + 8,  # 预留一点余量
    ).to(device)
    print_param_count(model)

    # Step 5: 配置优化器和损失函数
    # Adam 优化器使用论文推荐的超参数：betas=(0.9, 0.98), eps=1e-9
    opt = Adam(model.parameters(), lr=float(tr["lr"]), betas=(0.9, 0.98), eps=1e-9)
    # NLLLoss：配合 Generator 的 log_softmax，ignore_index 忽略 PAD
    crit = nn.NLLLoss(ignore_index=Vocab.PAD)

    # Step 6: 初始化训练状态
    best_val_loss = float("inf")
    log_path = results_dir / "train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建训练日志文件（写入表头）
    with log_path.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["epoch", "train_loss", "val_loss", "seconds"])

    epochs = int(tr["epochs"])
    clip = float(tr["grad_clip"])

    # Step 7: 训练循环
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_losses: list[float] = []

        # 训练一个 epoch
        for batch in train_loader:
            # 数据迁移到设备
            batch.src = batch.src.to(device)
            batch.tgt_in = batch.tgt_in.to(device)
            batch.tgt_out = batch.tgt_out.to(device)
            batch.src_mask = batch.src_mask.to(device)
            batch.tgt_mask = batch.tgt_mask.to(device)

            # 清空梯度
            opt.zero_grad(set_to_none=True)

            # 前向传播 + 损失计算
            loss = run_step_nll(model, crit, batch)

            # 反向传播
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

            # 参数更新
            opt.step()

            train_losses.append(loss.item())

        # 计算训练集平均损失
        train_loss = sum(train_losses) / max(len(train_losses), 1)

        # Step 8: 验证阶段
        model.eval()
        val_losses: list[float] = []

        with torch.no_grad():  # 禁用梯度计算，节省内存
            for vb in val_loader:
                vb.src = vb.src.to(device)
                vb.tgt_in = vb.tgt_in.to(device)
                vb.tgt_out = vb.tgt_out.to(device)
                vb.src_mask = vb.src_mask.to(device)
                vb.tgt_mask = vb.tgt_mask.to(device)
                val_losses.append(run_step_nll(model, crit, vb).item())

        val_loss = sum(val_losses) / max(len(val_losses), 1)
        elapsed = time.time() - t0

        # Step 9: 记录日志
        with log_path.open("a", newline="", encoding="utf-8") as fcsv:
            csv.writer(fcsv).writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{elapsed:.1f}"])
        print(f"epoch {epoch:03d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  {elapsed:.1f}s")

        # Step 10: 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "model": model.state_dict(),
                "config": cfg,
                "src_vocab": vocab_to_dict(src_vocab),
                "tgt_vocab": vocab_to_dict(tgt_vocab),
                "task": "translation",
            }
            ck_name = checkpoint_basename(results_dir)
            torch.save(ckpt, ckpt_dir / ck_name)
            torch.save(ckpt, ckpt_dir / "best.pt")  # 同时保存到通用路径

            # 记录最佳验证 loss
            with (results_dir / "best_val_loss.json").open("w", encoding="utf-8") as fjs:
                json.dump({"best_val_loss": best_val_loss, "epoch": epoch}, fjs, indent=2)


# ============================================================================
# 序列反转任务训练
# ============================================================================
def train_reverse(cfg: dict[str, Any], ckpt_dir: Path, results_dir: Path) -> None:
    """
    序列反转玩具任务训练流程。

    与翻译任务的主要区别：
      1. 使用合成数据（ReverseSequenceDataset），无需加载真实语料
      2. 使用 steps_per_epoch 控制每 epoch 的迭代次数（而非遍历整个数据集）
      3. 验证指标使用序列级准确率（而非 loss）

    Args:
        cfg:          配置字典
        ckpt_dir:     checkpoint 保存目录
        results_dir:  训练日志和指标输出目录
    """
    rt = cfg["reverse_toy"]
    m = cfg["model"]
    tr = cfg["train"]
    device = resolve_device(cfg.get("device", "auto"))
    torch.manual_seed(int(cfg.get("seed", 42)))

    # Step 1: 创建合成数据集
    vocab_size = int(rt["vocab_size"])
    train_ds = ReverseSequenceDataset(int(rt["train_samples"]), vocab_size, seed=int(cfg.get("seed", 42)))
    val_ds = ReverseSequenceDataset(int(rt["val_samples"]), vocab_size, seed=int(cfg.get("seed", 42)) + 1)

    # Step 2: 创建 DataLoader
    bs = int(tr["batch_size"])
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        collate_fn=collate_reverse,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_reverse)

    # Step 3: 构建模型
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
    print_param_count(model)

    # Step 4: 配置优化器和损失函数
    opt = Adam(model.parameters(), lr=float(tr["lr"]), betas=(0.9, 0.98), eps=1e-9)
    crit = nn.NLLLoss(ignore_index=Vocab.PAD)

    # Step 5: 初始化训练状态
    steps_per_epoch = int(rt["steps_per_epoch"])
    train_iter = infinite_loader(train_loader)  # 无限循环迭代器
    best_acc = -1.0  # 序列准确率，初始为 -1

    # 定义准确率计算函数
    def acc_batch(batch) -> float:
        """计算当前 batch 的序列完全匹配准确率。"""
        pred = greedy_decode(
            model,
            batch.src.to(device),
            Vocab.PAD,
            Vocab.BOS,
            Vocab.EOS,
            batch.tgt_out.size(1) + 8,
        )
        tgt = batch.tgt_out
        match = 0
        for i in range(tgt.size(0)):
            # 去除 padding 和 EOS 后的真实序列
            t_list = tgt[i][tgt[i] != Vocab.PAD].tolist()
            if t_list and t_list[-1] == Vocab.EOS:
                t_list = t_list[:-1]
            # 模型预测序列（去掉 BOS，遇到 EOS 停止）
            pl: list[int] = []
            for x in pred[i, 1:].tolist():
                if x == Vocab.EOS:
                    break
                pl.append(x)
            match += int(t_list == pl)
        return match / tgt.size(0)

    # 创建训练日志
    log_path = results_dir / "train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as fcsv:
        csv.writer(fcsv).writerow(["epoch", "train_loss", "val_loss", "val_acc", "seconds"])

    epochs = int(tr["epochs"])
    clip = float(tr["grad_clip"])

    # Step 6: 训练循环
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_losses: list[float] = []

        # 每个 epoch 固定 steps_per_epoch 步
        for _ in range(steps_per_epoch):
            batch = next(train_iter)
            batch.src = batch.src.to(device)
            batch.tgt_in = batch.tgt_in.to(device)
            batch.tgt_out = batch.tgt_out.to(device)
            batch.src_mask = batch.src_mask.to(device)
            batch.tgt_mask = batch.tgt_mask.to(device)

            opt.zero_grad(set_to_none=True)
            loss = run_step_nll(model, crit, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            opt.step()
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)

        # Step 7: 验证阶段（同时计算 loss 和 accuracy）
        model.eval()
        val_losses: list[float] = []
        val_accs: list[float] = []

        with torch.no_grad():
            for vb in val_loader:
                vb.src = vb.src.to(device)
                vb.tgt_in = vb.tgt_in.to(device)
                vb.tgt_out = vb.tgt_out.to(device)
                vb.src_mask = vb.src_mask.to(device)
                vb.tgt_mask = vb.tgt_mask.to(device)
                val_losses.append(run_step_nll(model, crit, vb).item())
                val_accs.append(acc_batch(vb))

        val_loss = sum(val_losses) / len(val_losses)
        val_acc = sum(val_accs) / len(val_accs)
        elapsed = time.time() - t0

        # 记录日志
        with log_path.open("a", newline="", encoding="utf-8") as fcsv:
            csv.writer(fcsv).writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_acc:.3f}", f"{elapsed:.1f}"])
        print(f"epoch {epoch:03d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  {elapsed:.1f}s")

        # 保存最优模型（基于准确率）
        if val_acc > best_acc:
            best_acc = val_acc
            ck = {"model": model.state_dict(), "config": cfg, "task": "reverse"}
            ck_name = checkpoint_basename(results_dir)
            torch.save(ck, ckpt_dir / ck_name)
            torch.save(ck, ckpt_dir / "best.pt")


# ============================================================================
# 主入口函数
# ============================================================================
def main() -> None:
    """
    命令行入口：解析参数并启动训练。

    命令行参数：
      --config:      YAML 配置文件路径（默认 configs/default.yaml）
      --epochs:      训练轮数（覆盖配置文件）
      --device:      训练设备（cpu/cuda/auto，覆盖配置文件）
      --results-dir: 结果输出目录（默认 results）
      --checkpoints-dir: checkpoint 保存目录（默认 checkpoints）

    根据配置中的 task 字段选择训练任务：
      - task == "reverse": 序列反转任务
      - 其他: 翻译任务
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None, help="训练日志与指标输出目录")
    parser.add_argument("--checkpoints-dir", type=str, default=None)
    args = parser.parse_args()

    # 加载配置
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)

    # 命令行参数覆盖配置文件
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = int(args.epochs)
    if args.device is not None:
        cfg["device"] = args.device

    # 创建必要的目录
    ckpt_dir = Path(args.checkpoints_dir or "checkpoints")
    results_dir = Path(args.results_dir or "results")
    figures_dir = Path("figures")
    for d in (ckpt_dir, results_dir, figures_dir, Path("data/multi30k"), Path("PPT"), Path("poster"), Path("report")):
        d.mkdir(parents=True, exist_ok=True)

    # 根据 task 字段选择训练流程
    task = str(cfg.get("task", "translation")).lower().strip()
    if task == "reverse":
        train_reverse(cfg, ckpt_dir, results_dir)
    else:
        train_translation(cfg, ckpt_dir, results_dir)


if __name__ == "__main__":
    main()