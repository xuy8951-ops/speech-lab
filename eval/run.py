#!/usr/bin/env python3
"""S0 垂直切片入口：
    python eval/run.py configs/exp/000_baseline.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from speechlab.data.synthesize import frozen_test_set
from speechlab.engine.registry import ENHANCERS
from speechlab.eval.runner import EvalRunner
import speechlab.baselines.spectral_gate  # noqa: F401 触发注册


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/exp/000_baseline.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    enhancer = ENHANCERS.get(cfg["model"]["name"])(**cfg["model"].get("args", {}))
    runner = EvalRunner(enhancer, cfg["eval"]["metrics"], cfg["output"])
    cases = frozen_test_set(n_per_case=cfg["data"].get("n_per_case", 2))
    print(f"算法: {enhancer.name}  测试样本: {len(cases)}")
    runner.run(cases)


if __name__ == "__main__":
    main()
