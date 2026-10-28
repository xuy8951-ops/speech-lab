"""通用训练引擎：给定(模型, 样本列表, 配置)执行训练，不出现具体模型/数据名。

特性：可变长度 batch、Adam + 余弦退火、梯度裁剪、验证早停、最佳/最近断点、
逐 epoch JSONL 日志与 history.json。S1 的过拟合模式（验证集=训练集）保持可用。
"""
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from speechlab.dsp.stft import stft


def cases_to_tensors(cases):
    """预计算幅度谱；保留列表以支持不同样本具有不同帧数。"""
    noisy_mags, clean_mags = [], []
    for c in cases:
        noisy_mags.append(torch.from_numpy(np.abs(stft(c["noisy"]))).float())
        clean_mags.append(torch.from_numpy(np.abs(stft(c["clean"]))).float())
    return noisy_mags, clean_mags


def _pad_batch(noisy_mag, clean_mag, indices):
    """按当前 batch 最大帧数补齐，并返回 (B,T,1) 的有效帧掩码。"""
    max_frames = max(len(noisy_mag[index]) for index in indices)
    freq_bins = noisy_mag[indices[0]].shape[-1]
    shape = (len(indices), max_frames, freq_bins)
    noisy = torch.zeros(shape, dtype=torch.float32)
    clean = torch.zeros(shape, dtype=torch.float32)
    valid = torch.zeros((len(indices), max_frames, 1), dtype=torch.float32)
    for row, index in enumerate(indices):
        frames = len(noisy_mag[index])
        noisy[row, :frames] = noisy_mag[index]
        clean[row, :frames] = clean_mag[index]
        valid[row, :frames] = 1.0
    return noisy, clean, valid


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
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.epochs)
        early = t.get("early_stop", {})
        self.monitor = early.get("metric", "val_loss")
        self.mode = early.get("mode", "min")
        self.patience = early.get("patience")
        self.min_delta = early.get("min_delta", 0.0)
        if self.mode not in {"min", "max"}:
            raise ValueError("early_stop.mode 只能是 min 或 max")
        with open(self.out_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        environment = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
        }
        with open(self.out_dir / "environment.json", "w", encoding="utf-8") as f:
            json.dump(environment, f, indent=2)

    def _loss_terms(self, mask, noisy, clean, valid):
        """默认保持旧实验的幅度 L1；可选对数幅度项或直接 IRM 掩蔽监督。"""
        enhanced = mask * noisy
        mag_l1 = self._masked_l1(enhanced, clean, valid)
        loss_cfg = self.cfg["train"].get("loss", {})
        name = loss_cfg.get("name", "mag_l1")
        if name == "mag_l1":
            return {"loss": mag_l1, "mag_l1": mag_l1}
        if name == "mag_log_l1":
            log_l1 = self._masked_l1(torch.log1p(enhanced), torch.log1p(clean), valid)
            weight = float(loss_cfg.get("log_weight", 1.0))
            if weight < 0:
                raise ValueError("train.loss.log_weight 不能为负数")
            return {
                "loss": mag_l1 + weight * log_l1,
                "mag_l1": mag_l1,
                "log_l1": log_l1,
            }
        if name in {"irm_l1", "mag_irm_l1"}:
            ideal_mask = (clean / (noisy + 1e-8)).clamp(0.0, 1.0)
            irm_l1 = self._masked_l1(mask, ideal_mask, valid)
            if name == "irm_l1":
                return {"loss": irm_l1, "irm_l1": irm_l1, "mag_l1": mag_l1}
            weight = float(loss_cfg.get("irm_weight", 1.0))
            if weight < 0:
                raise ValueError("train.loss.irm_weight 不能为负数")
            return {
                "loss": mag_l1 + weight * irm_l1,
                "mag_l1": mag_l1,
                "irm_l1": irm_l1,
            }
        raise ValueError(f"未知训练损失: {name}")

    def _epoch_loss(self, noisy_mag, clean_mag, train=True):
        n = len(noisy_mag)
        indices = np.random.permutation(n) if train else np.arange(n)
        totals, count = {}, 0
        for i in range(0, n, self.batch):
            batch_indices = indices[i:i + self.batch].tolist()
            noisy, clean, valid = _pad_batch(noisy_mag, clean_mag, batch_indices)
            if train:
                mask = self.model(noisy)
                terms = self._loss_terms(mask, noisy, clean, valid)
                self.opt.zero_grad()
                terms["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.opt.step()
            else:
                with torch.no_grad():
                    mask = self.model(noisy)
                    terms = self._loss_terms(mask, noisy, clean, valid)
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + value.item() * len(batch_indices)
            count += len(batch_indices)
        return {key: value / count for key, value in totals.items()}

    @staticmethod
    def _masked_l1(enhanced, clean, valid):
        error = (enhanced - clean).abs() * valid
        return error.sum() / (valid.sum() * clean.shape[-1]).clamp_min(1.0)

    def _is_improvement(self, value, best):
        if self.mode == "min":
            return value < best - self.min_delta
        return value > best + self.min_delta

    def fit(self, train_cases, val_cases=None):
        tr_noisy, tr_clean = cases_to_tensors(train_cases)
        if val_cases is None:  # S1 纯过拟合模式：训练集即验证集
            va_noisy, va_clean = tr_noisy, tr_clean
        else:
            va_noisy, va_clean = cases_to_tensors(val_cases)
        if not tr_noisy or not va_noisy:
            raise ValueError("训练集和验证集都不能为空")

        best = float("inf") if self.mode == "min" else float("-inf")
        best_epoch, stale_epochs, stopped_early = 0, 0, False
        history, start_epoch = [], 1
        state_path = self.out_dir / "trainer_state.pt"
        if self.cfg["train"].get("resume", False) and state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state["model"])
            self.opt.load_state_dict(state["optimizer"])
            self.sched.load_state_dict(state["scheduler"])
            best = state["best"]
            best_epoch = state["best_epoch"]
            stale_epochs = state["stale_epochs"]
            history = state["history"]
            start_epoch = state["epoch"] + 1
            np.random.set_state(state["numpy_rng"])
            torch.set_rng_state(state["torch_rng"])
            print(f"从 epoch {start_epoch} 断点续训；历史最佳 {self.monitor}={best:.5f}。", flush=True)

        log_mode = "a" if start_epoch > 1 else "w"
        with open(self.out_dir / "history.jsonl", log_mode, encoding="utf-8") as log_file:
            for ep in range(start_epoch, self.epochs + 1):
                t0 = time.time()
                self.model.train()
                tr_terms = self._epoch_loss(tr_noisy, tr_clean, train=True)
                self.model.eval()
                va_terms = self._epoch_loss(va_noisy, va_clean, train=False)
                row = {"epoch": ep}
                for split, terms in (("train", tr_terms), ("val", va_terms)):
                    for key, value in terms.items():
                        row[f"{split}_{key}"] = value
                row.update(
                    lr=self.opt.param_groups[0]["lr"],
                    seconds=round(time.time() - t0, 3),
                )
                history.append(row)
                log_file.write(json.dumps(row) + "\n")
                log_file.flush()
                self.sched.step()

                torch.save(self.model.state_dict(), self.out_dir / "last.pt")
                monitored = row.get(self.monitor)
                if monitored is None:
                    raise KeyError(f"early_stop.metric 不存在于日志: {self.monitor}")
                if self._is_improvement(monitored, best):
                    best, best_epoch, stale_epochs = monitored, ep, 0
                    torch.save(self.model.state_dict(), self.out_dir / "best.pt")
                else:
                    stale_epochs += 1
                torch.save({
                    "epoch": ep,
                    "model": self.model.state_dict(),
                    "optimizer": self.opt.state_dict(),
                    "scheduler": self.sched.state_dict(),
                    "best": best,
                    "best_epoch": best_epoch,
                    "stale_epochs": stale_epochs,
                    "history": history,
                    "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                }, state_path)
                if ep % self.cfg["train"].get("log_every", 5) == 0 or ep == 1:
                    print(f"epoch {ep:3d}/{self.epochs}  train={row['train_loss']:.5f}  "
                          f"val={row['val_loss']:.5f}  ({row['seconds']:.1f}s)", flush=True)
                if self.patience is not None and stale_epochs >= self.patience:
                    stopped_early = True
                    print(f"验证指标连续 {self.patience} 轮未改善，提前停止。", flush=True)
                    break

        with open(self.out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        summary = {
            "best_epoch": best_epoch,
            "best_metric": self.monitor,
            "best_value": best,
            "epochs_run": len(history),
            "stopped_early": stopped_early,
        }
        with open(self.out_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        self.model.load_state_dict(torch.load(self.out_dir / "best.pt", map_location="cpu", weights_only=True))
        print(f"训练完成，最佳 {self.monitor}={best:.5f}（epoch {best_epoch}），"
              f"已恢复最佳权重: {self.out_dir/'best.pt'}", flush=True)
        return history
