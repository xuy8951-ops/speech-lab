#!/usr/bin/env python3
"""S0 垂直切片入口：
    python eval/run.py configs/exp/000_baseline.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from speechlab.data.datasets import load_cases
from speechlab.engine.registry import ENHANCERS
from speechlab.eval.runner import EvalRunner
import speechlab.baselines.spectral_gate  # noqa: F401 触发注册
import speechlab.models.tiny_gru  # noqa: F401 触发注册


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/exp/000_baseline.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test",
                        help="默认评测冻结测试集；val 用于模型选择且写入独立目录")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data"].setdefault("seed", cfg.get("seed", 42))

    enhancer = ENHANCERS.get(cfg["model"]["name"])(**cfg["model"].get("args", {}))
    if cfg["model"].get("checkpoint"):
        enhancer.load(cfg["model"]["checkpoint"])
        print(f"权重: {cfg['model']['checkpoint']}")
    output = cfg["output"] if args.split == "test" else f"{cfg['output']}/validation"
    runner = EvalRunner(enhancer, cfg["eval"]["metrics"], output)
    cases = load_cases(cfg["data"], args.split)
    label = "测试" if args.split == "test" else "验证"
    print(f"算法: {enhancer.name}  {label}样本: {len(cases)}")
    runner.run(cases)


if __name__ == "__main__":
    main()
