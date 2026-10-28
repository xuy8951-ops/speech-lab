#!/usr/bin/env python3
"""训练入口：
    python train/run.py configs/exp/001_gru_overfit.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from speechlab.data.datasets import load_cases
from speechlab.engine.registry import ENHANCERS
from speechlab.engine.trainer import Trainer
import speechlab.models.tiny_gru  # noqa: F401 触发注册


def main():
    cfg_path = sys.argv[1]
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    cfg["data"].setdefault("seed", cfg["seed"])

    model = ENHANCERS.get(cfg["model"]["name"])(**cfg["model"].get("args", {}))
    print(f"模型: {model.name}  参数量: {model.num_params():,}")

    train_cases = load_cases(cfg["data"], "train")
    val_cases = load_cases(cfg["data"], "val") if cfg["data"].get("validation", False) else None
    print(f"训练样本: {len(train_cases)}  验证样本: {len(val_cases) if val_cases is not None else '训练集'}")
    trainer = Trainer(model, cfg, cfg["output"])
    trainer.fit(train_cases, val_cases)


if __name__ == "__main__":
    main()
