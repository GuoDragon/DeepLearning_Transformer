"""
一键运行脚本：自动执行数据准备、训练、评测全流程。

本脚本是项目的一站式运行入口，按顺序执行以下步骤：
  1. 创建必要目录
  2. 数据准备：检查/下载 Multi30k + 生成示例语料
  3. Multi30k 德英翻译任务训练与评测
  4. 序列反转玩具任务训练与评测
  5. 示例语料 quick 训练与评测
  6. 生成汇总报告

输出目录结构：
  - checkpoints/          # 模型权重文件
    - best_translation.pt
    - best_reverse.pt
    - best_corpus_quick.pt
  - results/              # 训练日志和评估结果
    - translation/        # 翻译任务结果
    - reverse/            # 反转任务结果
    - corpus_quick/       # 示例语料结果
    - run_summary.json    # 汇总报告
  - figures/              # 可视化图表
    - translation_loss.png
    - reverse_loss.png

使用方式：
  python scripts/run_all.py

注意：建议在 CPU 模式下运行（默认），避免 GPU 环境差异问题。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

# 项目根目录路径（scripts/run_all.py 的父目录的父目录）
ROOT = Path(__file__).parent.parent


def run(cmd: list[str], desc: str) -> None:
    """
    在项目根目录执行子进程命令。

    Args:
        cmd:  命令行参数列表，包含 python 解释器和脚本名
        desc: 步骤说明，用于日志输出

    Raises:
        subprocess.CalledProcessError: 子进程非零退出时抛出异常
    """
    print(f"\n{'=' * 60}\n{desc}\n{'=' * 60}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    """
    主执行函数：顺序执行数据准备、训练、评测全流程。

    执行步骤：
      1. 创建所有必要目录
      2. 数据准备阶段：
         - 检查 Multi30k 是否存在，不存在则下载
         - 生成示例语料 data/corpus/
      3. Multi30k 翻译任务：
         - 训练（8 epochs）
         - 绘制 loss 曲线
         - 测试集 BLEU 评估
      4. 序列反转任务：
         - 训练
         - 绘制曲线
         - 验证准确率评估
      5. 示例语料 quick 任务：
         - 训练
         - BLEU 评估
      6. 生成汇总报告 run_summary.json
    """
    # 获取当前 Python 解释器路径（确保使用同一环境）
    py = sys.executable
    # 记录开始时间，用于计算总耗时
    t0 = time.time()
    # 汇总字典，记录各任务的结果路径
    summary: dict = {"steps": []}

    # Step 1: 创建所有必要目录
    for d in (
        "checkpoints",          # 模型 checkpoint 保存目录
        "results/translation",  # 翻译任务结果目录
        "results/reverse",      # 反转任务结果目录
        "results/corpus_quick", # 示例语料结果目录
        "figures",              # 可视化图表目录
        "data/multi30k",        # Multi30k 数据集目录
        "data/corpus",          # 示例语料目录
    ):
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    # Step 2: 数据准备
    # 检查 Multi30k 是否已存在
    if not (ROOT / "data/multi30k/train.de").exists():
        run([py, "scripts/download_multi30k.py"], "下载 Multi30k")
    else:
        print("Multi30k 已存在，跳过下载")
    # 生成示例语料（用于离线环境测试）
    run([py, "scripts/build_sample_corpus.py"], "生成示例语料 data/corpus/")

    # Step 3: Multi30k 德英翻译任务
    trans_epochs = 8
    # 训练
    run(
        [
            py,
            "train.py",
            "--config", "configs/default.yaml",
            "--epochs", str(trans_epochs),
            "--device", "cpu",
            "--results-dir", "results/translation",
        ],
        f"Multi30k 翻译训练 ({trans_epochs} epochs)",
    )
    # 绘制 loss 曲线
    run(
        [
            py,
            "scripts/plot_loss.py",
            "--csv", "results/translation/train_log.csv",
            "--out", "figures/translation_loss.png",
            "--title", "Multi30k DE->EN",
        ],
        "绘制翻译 loss 曲线",
    )
    # 测试集 BLEU 评估
    run(
        [
            py,
            "test.py",
            "--checkpoint", "checkpoints/best_translation.pt",
            "--device", "cpu",
            "--output-json", "results/translation/test_bleu.json",
            "--output-samples", "results/translation/predictions_sample.txt",
        ],
        "翻译测试集 BLEU",
    )
    # 记录翻译任务结果路径
    summary["translation"] = {
        "epochs": trans_epochs,
        "checkpoint": "checkpoints/best_translation.pt",
        "results": "results/translation/",
        "figure": "figures/translation_loss.png",
    }

    # Step 4: 序列反转玩具任务
    # 训练
    run(
        [
            py,
            "train.py",
            "--config", "configs/reverse.yaml",
            "--device", "cpu",
            "--results-dir", "results/reverse",
        ],
        "序列反转训练",
    )
    # 绘制曲线
    run(
        [
            py,
            "scripts/plot_loss.py",
            "--csv", "results/reverse/train_log.csv",
            "--out", "figures/reverse_loss.png",
            "--title", "Reverse Toy Task",
        ],
        "绘制反转任务曲线",
    )
    # 验证准确率评估
    run(
        [
            py,
            "test.py",
            "--checkpoint", "checkpoints/best_reverse.pt",
            "--device", "cpu",
            "--output-json", "results/reverse/test_metrics.json",
        ],
        "反转任务验证准确率",
    )
    # 记录反转任务结果路径
    summary["reverse"] = {
        "checkpoint": "checkpoints/best_reverse.pt",
        "results": "results/reverse/",
        "figure": "figures/reverse_loss.png",
    }

    # Step 5: 示例语料 quick 训练（快速验证）
    # 训练
    run(
        [
            py,
            "train.py",
            "--config", "configs/quick.yaml",
            "--device", "cpu",
            "--results-dir", "results/corpus_quick",
        ],
        "示例语料 quick 训练",
    )
    # BLEU 评估
    run(
        [
            py,
            "test.py",
            "--checkpoint", "checkpoints/best_corpus_quick.pt",
            "--config", "configs/quick.yaml",
            "--device", "cpu",
            "--output-json", "results/corpus_quick/test_bleu.json",
        ],
        "示例语料 quick BLEU",
    )
    # 记录 quick 任务结果路径
    summary["corpus_quick"] = {"results": "results/corpus_quick/"}

    # Step 6: 生成汇总报告
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    out = ROOT / "results" / "run_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # 输出完成信息
    print(f"\n全部完成，耗时 {summary['elapsed_seconds']}s")
    print("汇总:", out)


if __name__ == "__main__":
    main()
