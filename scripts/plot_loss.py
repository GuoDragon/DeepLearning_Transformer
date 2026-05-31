"""
训练日志可视化模块：根据 train_log.csv 绘制 loss 曲线。

功能特点：
  1. 自动检测 CSV 列，支持 loss 曲线和准确率曲线的叠加显示
  2. 使用 Agg 后端，支持无图形界面环境运行
  3. 生成高质量 PNG 图片（150 dpi）

CSV 格式要求：
  必须包含列：epoch, train_loss, val_loss
  可选包含列：val_acc（序列反转任务使用）

输出图片内容：
  - 主坐标轴：训练 loss（蓝色圆点）和验证 loss（橙色方块）
  - 次坐标轴（如有 val_acc）：验证准确率（绿色虚线三角）
  - 网格线、图例、标题等
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def plot_csv(csv_path: Path, out_png: Path, title: str) -> None:
    """
    读取训练日志 CSV 文件，绘制 loss 曲线（和可选的准确率曲线）。

    Args:
        csv_path: 训练日志 CSV 文件路径，必须包含 epoch、train_loss、val_loss 列
        out_png:  输出 PNG 图片路径
        title:    图表标题
    """
    # 使用 Agg 后端，避免图形界面依赖
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 数据容器初始化
    epochs: list[int] = []
    train_loss: list[float] = []
    val_loss: list[float] = []
    val_acc: list[float] = []

    # 读取 CSV 文件
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # 检查是否包含 val_acc 列（序列反转任务特有）
        has_acc = "val_acc" in (reader.fieldnames or [])

        # 逐行读取数据
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            if has_acc:
                val_acc.append(float(row["val_acc"]))

    # 创建画布和坐标轴
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # 绘制训练 loss（蓝色圆点）
    ax1.plot(epochs, train_loss, label="train_loss", marker="o", color="#1f77b4")
    # 绘制验证 loss（橙色方块）
    ax1.plot(epochs, val_loss, label="val_loss", marker="s", color="#ff7f0e")

    # 设置主坐标轴属性
    ax1.set_xlabel("epoch", fontsize=12)
    ax1.set_ylabel("loss", fontsize=12)
    ax1.set_title(title, fontsize=14, pad=15)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0, right=epochs[-1] + 1)

    # 如果有准确率数据，添加次坐标轴
    if val_acc:
        ax2 = ax1.twinx()  # 创建共享 x 轴的次坐标轴
        # 绘制验证准确率（绿色虚线三角）
        ax2.plot(epochs, val_acc, color="#2ca02c", label="val_acc", marker="^", linestyle="--")
        ax2.set_ylabel("accuracy", fontsize=12)
        ax2.set_ylim(bottom=0, top=1.0)

        # 合并两个坐标轴的图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # 确保输出目录存在
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # 调整布局并保存
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("saved:", out_png)


def main() -> None:
    """
    命令行入口：解析参数并绘制 loss 曲线。

    命令行参数：
      --csv:   训练日志 CSV 文件路径（必需）
      --out:   输出 PNG 图片路径（必需）
      --title: 图表标题（可选，默认为 "Training"）
    """
    parser = argparse.ArgumentParser(description="绘制训练 loss 曲线")
    parser.add_argument("--csv", type=str, required=True, help="训练日志 CSV 文件路径")
    parser.add_argument("--out", type=str, required=True, help="输出 PNG 图片路径")
    parser.add_argument("--title", type=str, default="Training", help="图表标题")
    args = parser.parse_args()

    # 调用绘图函数
    plot_csv(Path(args.csv), Path(args.out), args.title)


if __name__ == "__main__":
    main()
