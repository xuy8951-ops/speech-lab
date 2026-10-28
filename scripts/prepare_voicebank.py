#!/usr/bin/env python3
"""清洗 VoiceBank-DEMAND 并生成 train/val/test manifest（test 冻结）。"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speechlab.data.voicebank import prepare_voicebank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/voicebank_demand"))
    parser.add_argument("--out-root", type=Path, default=Path("data/processed/voicebank_demand"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests/voicebank_demand"))
    parser.add_argument("--val-speakers", default="p226,p227", help="逗号分隔；默认两个独立验证说话人")
    parser.add_argument("--only", choices=["train", "test", "all"], default="all")
    parser.add_argument("--allow-partial", action="store_true",
                        help="仅用于真实训练子集 pilot；禁止用于 test")
    args = parser.parse_args()
    splits = ("train", "test") if args.only == "all" else (args.only,)
    counts = prepare_voicebank(
        args.raw_root,
        args.out_root,
        args.manifest_dir,
        val_speakers=tuple(x.strip() for x in args.val_speakers.split(",") if x.strip()),
        splits=splits,
        allow_partial=args.allow_partial,
    )
    print("manifest 样本数: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    print(f"目录: {args.manifest_dir.resolve()}")


if __name__ == "__main__":
    main()
