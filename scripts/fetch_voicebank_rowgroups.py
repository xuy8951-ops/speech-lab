#!/usr/bin/env python3
"""按 Parquet row group 拉取真实训练子集，用于 CPU pilot，不触碰测试集。

默认拉取 p226（验证候选 100 条）与 p228（训练候选 200 条）：
    python scripts/fetch_voicebank_rowgroups.py --shard 0 --row-groups 0,8,9
依赖：pyarrow、fsspec、aiohttp、requests。
"""
import argparse
from pathlib import Path
import re

import fsspec
import pyarrow.parquet as pq

BASE = "https://hf-mirror.com/datasets/JacobLinCool/VoiceBank-DEMAND-16k/resolve/main/data/"
DIRECTORIES = ("clean_trainset_28spk_wav", "noisy_trainset_28spk_wav")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/voicebank_demand"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--row-groups", default="0,8,9")
    args = parser.parse_args()
    groups = [int(value) for value in args.row_groups.split(",")]
    name = f"train-{args.shard:05d}-of-00005.parquet"
    directories = [args.root / value for value in DIRECTORIES]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    remote = fsspec.filesystem("http").open(BASE + name, "rb", block_size=65536)
    with pq.ParquetFile(remote) as parquet:
        count = 0
        for group in groups:
            table = parquet.read_row_group(group)
            for row in table.to_pylist():
                sample_id = row["id"]
                if not re.fullmatch(r"p\d+_\d+", sample_id):
                    raise ValueError(f"异常样本 ID: {sample_id}")
                for kind, directory in zip(("clean", "noisy"), directories):
                    path = directory / f"{sample_id}.wav"
                    if not path.exists():
                        path.write_bytes(row[kind]["bytes"])
                count += 1
            print(f"row group {group}: 累计解包 {count} 条", flush=True)
    print(f"训练子集就绪: {args.root.resolve()}；请用 prepare_voicebank.py --only train --allow-partial")


if __name__ == "__main__":
    main()
