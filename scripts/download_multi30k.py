"""
从 Multi30k 官方 GitHub 仓库下载德英平行语料并解压到 data/multi30k/。

Multi30k 数据集简介：
  - 来源：multi30k/dataset GitHub 仓库
  - 语言对：德语 → 英语 (DE → EN)
  - 规模：训练集 ~29,000 句，验证集 ~1,000 句，测试集 ~1,000 句
  - 领域：Flickr 图片描述（图像标题生成任务的副产品）

文件映射关系：
  - train.de.gz    → train.de    (德语训练集)
  - train.en.gz    → train.en    (英语训练集)
  - val.de.gz      → val.de      (德语验证集)
  - val.en.gz      → val.en      (英语验证集)
  - test_2016_flickr.de.gz → test.de  (德语测试集)
  - test_2016_flickr.en.gz → test.en  (英语测试集)

注意：测试集使用 2016 Flickr 版本，这是 Multi30k 任务的标准测试集。
"""

from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

# Multi30k 数据集官方 GitHub 原始数据地址
BASE = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw"

# 远程 gzip 文件名 → 本地解压后的文件名
# 映射关系与 configs/default.yaml 中的 paths 配置一致
FILES = {
    "train.de.gz": "train.de",           # 德语训练集
    "train.en.gz": "train.en",           # 英语训练集
    "val.de.gz": "val.de",               # 德语验证集
    "val.en.gz": "val.en",               # 英语验证集
    "test_2016_flickr.de.gz": "test.de", # 德语测试集（2016 Flickr 版本）
    "test_2016_flickr.en.gz": "test.en", # 英语测试集（2016 Flickr 版本）
}


def download_file(url: str, dest: Path) -> None:
    """
    从指定 URL 下载文件到本地路径。

    Args:
        url:  远程文件下载地址
        dest: 本地保存路径（包含文件名）
    """
    # 确保父目录存在
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载: {url}")
    # 使用 urllib 下载文件
    urllib.request.urlretrieve(url, dest)


def gunzip_to(src_gz: Path, dest_txt: Path) -> None:
    """
    将 gzip 压缩文件解压为纯文本文件。

    Args:
        src_gz:   源压缩文件路径（.gz）
        dest_txt: 解压后的文本文件路径
    """
    # 读取压缩数据
    with gzip.open(src_gz, "rb") as f_in:
        data = f_in.read()
    # 写入解压后的文本文件
    dest_txt.write_bytes(data)
    print(f"解压: {dest_txt.name} ({len(data)} bytes)")


def main() -> None:
    """
    下载并解压 Multi30k 数据集的六个文件到 data/multi30k/。

    执行流程：
      1. 创建输出目录 data/multi30k/
      2. 遍历 FILES 字典，逐个处理文件
      3. 如果目标文本文件已存在且非空，跳过下载
      4. 否则下载 .gz 文件，解压后删除压缩包
      5. 输出最终文件列表和行数统计
    """
    out_dir = Path("data/multi30k")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 遍历所有需要下载的文件
    for remote, local_name in FILES.items():
        url = f"{BASE}/{remote}"
        gz_path = out_dir / remote   # 临时压缩包路径
        txt_path = out_dir / local_name  # 最终文本文件路径

        # 如果目标文件已存在且非空，跳过下载
        if txt_path.exists() and txt_path.stat().st_size > 0:
            print(f"跳过（已存在）: {local_name}")
            continue

        # 下载压缩包
        download_file(url, gz_path)
        # 解压到目标路径
        gunzip_to(gz_path, txt_path)
        # 删除临时压缩包
        gz_path.unlink(missing_ok=True)

    # 输出下载完成信息
    print("\n完成。文件列表:")
    # 按语言分组输出：德语文件在前，英语文件在后
    for p in sorted(out_dir.glob("*.de")) + sorted(out_dir.glob("*.en")):
        # 统计文件行数
        n = sum(1 for _ in p.open(encoding="utf-8"))
        print(f"  {p.name}: {n} 行")


if __name__ == "__main__":
    main()
