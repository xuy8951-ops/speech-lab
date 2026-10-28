#!/usr/bin/env python3
"""从公开镜像下载 VoiceBank-DEMAND 16 kHz Parquet 并解包为 WAV。

官方条件日志仍用于噪声/SNR 标签。依赖：pyarrow。
    python scripts/download_voicebank_16k.py --only all
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import urllib.request

import pyarrow.parquet as pq

from download_voicebank import download as download_official

BASE = "https://hf-mirror.com/datasets/JacobLinCool/VoiceBank-DEMAND-16k/resolve/main/data/"
FILES = {
    "test": ["test-00000-of-00001.parquet"],
    "train": [f"train-{index:05d}-of-00005.parquet" for index in range(5)],
}
EXPECTED_COUNTS = {"train": 11572, "test": 824}
DIRECTORIES = {
    "test": ("clean_testset_wav", "noisy_testset_wav"),
    "train": ("clean_trainset_28spk_wav", "noisy_trainset_28spk_wav"),
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(name, root):
    destination = root / name
    partial = root / (name + ".part")
    if destination.exists():
        try:
            with pq.ParquetFile(destination) as parquet:
                if parquet.metadata.num_rows > 0:
                    print(f"跳过已完成下载: {name}", flush=True)
                    return destination
        except Exception:
            pass
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(BASE + name, headers={"Range": f"bytes={offset}-"} if offset else {})
    print(f"下载 {name}" + (f"，从 {offset / 2**20:.1f} MiB 续传" if offset else ""), flush=True)
    with urllib.request.urlopen(request, timeout=90) as response:
        mode = "ab" if offset and response.status == 206 else "wb"
        with open(partial, mode) as f:
            shutil.copyfileobj(response, f, length=8 * 1024 * 1024)
    with pq.ParquetFile(partial) as parquet:
        if parquet.metadata.num_rows <= 0:
            raise RuntimeError(f"Parquet 无有效样本: {partial}")
    os.replace(partial, destination)
    return destination


def _unpack(split, root):
    clean_dir, noisy_dir = (root / name for name in DIRECTORIES[split])
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for name in FILES[split]:
        with pq.ParquetFile(root / "hf_parquet" / name) as parquet:
            for batch in parquet.iter_batches(batch_size=32):
                for row in batch.to_pylist():
                    sample_id = row["id"]
                    if not re.fullmatch(r"p\d+_\d+", sample_id):
                        raise ValueError(f"异常样本 ID: {sample_id}")
                    for kind, directory in (("clean", clean_dir), ("noisy", noisy_dir)):
                        path = directory / f"{sample_id}.wav"
                        if not path.exists():
                            path.write_bytes(row[kind]["bytes"])
                    count += 1
        print(f"已解包 {split}: {count}/{EXPECTED_COUNTS[split]}", flush=True)
    if count != EXPECTED_COUNTS[split]:
        raise RuntimeError(f"{split} 样本数 {count}，预期 {EXPECTED_COUNTS[split]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/voicebank_demand"))
    parser.add_argument("--only", choices=["train", "test", "all"], default="all")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    parquet_root = args.root / "hf_parquet"
    parquet_root.mkdir(exist_ok=True)
    if not (args.root / "logfiles.zip").exists():
        download_official("logfiles.zip", args.root)

    splits = ("train", "test") if args.only == "all" else (args.only,)
    checksum_path = args.root / "hf_checksums.json"
    checksums = json.loads(checksum_path.read_text()) if checksum_path.exists() else {}
    for split in splits:
        for name in FILES[split]:
            path = _download(name, parquet_root)
            checksums[name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        _unpack(split, args.root)
    checksum_path.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
    print(f"16 kHz WAV 就绪: {args.root.resolve()}")


if __name__ == "__main__":
    main()
