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

from speechlab.data.synthesize import make_train_cases
from speechlab.engine.registry import ENHANCERS
from speechlab.engine.trainer import Trainer
import speechlab.models.tiny_gru  # noqa: F401 触发注册


def main():
    cfg_path = sys.argv[1]
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    model = ENHANCERS.get(cfg["model"]["name"])(**cfg["model"].get("args", {}))
    print(f"模型: {model.name}  参数量: {model.num_params():,}")

    cases = make_train_cases(n=cfg["data"]["n_train"],
                             duration=cfg["data"].get("duration", 2.0),
                             seed=cfg["seed"])
    trainer = Trainer(model, cfg, cfg["output"])
    trainer.fit(cases)  # S1：纯过拟合，验证集=训练集


if __name__ == "__main__":
    main()
