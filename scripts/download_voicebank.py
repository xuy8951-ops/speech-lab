#!/usr/bin/env python3
"""下载 VoiceBank-DEMAND 28 说话人官方数据，支持断点续传与 ZIP CRC 校验。

用法（仓库根目录）：
    python scripts/download_voicebank.py --only all
    python scripts/download_voicebank.py --only test  # 仅测试集与条件日志
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile

API = "https://datashare.ed.ac.uk/server/api/core/bitstreams/{}/content"
FILES = {
    "clean_testset_wav.zip": "dec213d3-bf57-4777-9663-c24bdce92d5e",
    "noisy_testset_wav.zip": "13c1bfbf-14a6-41db-9b41-8f7310f01ad5",
    "clean_trainset_28spk_wav.zip": "245452b6-6235-44b6-a6f9-e7eb19797769",
    "noisy_trainset_28spk_wav.zip": "ecb5a102-bb00-46d3-8af5-40c79823b837",
    "logfiles.zip": "11185dc8-9cf1-405b-b858-35bd6a04aedd",
}
GROUPS = {
    "test": ["clean_testset_wav.zip", "noisy_testset_wav.zip", "logfiles.zip"],
    "train": ["clean_trainset_28spk_wav.zip", "noisy_trainset_28spk_wav.zip", "logfiles.zip"],
}
GROUPS["all"] = list(FILES)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_zip(path):
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"下载结果不是 ZIP（可能是网页错误响应）: {path}")
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
    if bad is not None:
        raise RuntimeError(f"ZIP CRC 校验失败: {path}，首个损坏成员: {bad}")


def download(name, root):
    destination = root / name
    partial = root / (name + ".part")
    if destination.exists():
        print(f"校验已有文件: {name}", flush=True)
        verify_zip(destination)
        print(f"跳过已完成下载: {name}", flush=True)
        return destination

    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(API.format(FILES[name]), headers=headers)
    mode = "ab"
    print(f"下载 {name}" + (f"，从 {offset / 2**20:.1f} MiB 续传" if offset else ""), flush=True)
    with urllib.request.urlopen(request, timeout=90) as response:
        # 服务端忽略 Range 时必须从头写，否则会拼出损坏文件。
        if offset and response.status != 206:
            mode = "wb"
            offset = 0
        with open(partial, mode) as f:
            shutil.copyfileobj(response, f, length=8 * 1024 * 1024)
    print(f"下载完成，执行 ZIP CRC 校验: {name}", flush=True)
    verify_zip(partial)
    os.replace(partial, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/voicebank_demand"))
    parser.add_argument("--only", choices=GROUPS, default="all")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    checksums_path = args.root / "checksums.json"
    checksums = json.loads(checksums_path.read_text()) if checksums_path.exists() else {}
    for name in GROUPS[args.only]:
        path = download(name, args.root)
        checksums[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        checksums_path.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
        print(f"SHA256: {checksums[name]['sha256']}", flush=True)
    print(f"数据就绪: {args.root.resolve()}")


if __name__ == "__main__":
    main()
