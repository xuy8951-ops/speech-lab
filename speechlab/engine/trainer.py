"""通用训练引擎：给定(模型, 样本列表, 配置)执行训练，不出现任何具体模型/数据名。

特性：Adam + 余弦退火、梯度裁剪、最佳/最近断点、逐epoch日志、history.json。
S1 目标：小数据过拟合（loss→接近0）以验证训练链路无 bug。
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from speechlab.dsp.stft import stft


def cases_to_tensors(cases):
    """预计算幅度谱，返回 (noisy_mag, clean_mag) 两个 (N,T,257) 张量。"""
    noisy_mags, clean_mags = [], []
    for c in cases:
        noisy_mags.append(np.abs(stft(c["noisy"])))
        clean_mags.append(np.abs(stft(c["clean"])))
    return (torch.from_numpy(np.stack(noisy_mags)).float(),
            torch.from_numpy(np.stack(clean_mags)).float())


class Trainer:
    def __init__(self, model, cfg, out_dir):
        self.model = model
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        t = cfg["train"]
        self.opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
        self.epochs = t["epochs"]
        self.batch = t["batch"]
        self.clip = t.get("grad_clip", 5.0)
        self.loss_fn = torch.nn.L1Loss()
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.epochs)

    def _epoch_loss(self, noisy_mag, clean_mag, train=True):
        n = len(noisy_mag)
        idx = np.random.permutation(n) if train else np.arange(n)
        total, count = 0.0, 0
        for i in range(0, n, self.batch):
            j = idx[i:i + self.batch]
            nb, cb = noisy_mag[j], clean_mag[j]
            if train:
                mask = self.model(nb)
                loss = self.loss_fn(mask * nb, cb)
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.opt.step()
            else:
                with torch.no_grad():
                    mask = self.model(nb)
                    loss = self.loss_fn(mask * nb, cb)
            total += loss.item() * len(j)
            count += len(j)
        return total / count

    def fit(self, train_cases, val_cases=None):
        tr_noisy, tr_clean = cases_to_tensors(train_cases)
        if val_cases is None:  # 纯过拟合模式：训练集即验证集
            va_noisy, va_clean = tr_noisy, tr_clean
        else:
            va_noisy, va_clean = cases_to_tensors(val_cases)

        best = float("inf")
        history = []
        for ep in range(1, self.epochs + 1):
            t0 = time.time()
            self.model.train()
            tr_loss = self._epoch_loss(tr_noisy, tr_clean, train=True)
            self.model.eval()
            va_loss = self._epoch_loss(va_noisy, va_clean, train=False)
            self.sched.step()
            history.append(dict(epoch=ep, train_loss=tr_loss, val_loss=va_loss))

            torch.save(self.model.state_dict(), self.out_dir / "last.pt")
            if va_loss < best:
                best = va_loss
                torch.save(self.model.state_dict(), self.out_dir / "best.pt")
            if ep % self.cfg["train"].get("log_every", 5) == 0 or ep == 1:
                print(f"epoch {ep:3d}/{self.epochs}  train={tr_loss:.5f}  "
                      f"val={va_loss:.5f}  ({time.time()-t0:.1f}s)")

        with open(self.out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"训练完成，最佳验证 loss={best:.5f}，权重: {self.out_dir/'best.pt'}")
        return history
