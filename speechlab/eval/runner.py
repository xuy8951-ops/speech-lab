"""EvalRunner：逐样本评分 → scores.jsonl → 按[语言×噪声×SNR]透视汇总。

报告里的每个数字都能溯源到 scores.jsonl 的具体行。
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from speechlab.engine.registry import METRICS
import speechlab.eval.metrics  # noqa: F401 触发指标注册


class EvalRunner:
    def __init__(self, enhancer, metric_names, out_dir):
        self.enhancer = enhancer
        self.metric_names = metric_names
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def run(self, cases):
        rows = []
        for case in cases:
            noisy = torch.from_numpy(case["noisy"])[None]
            enhanced = self.enhancer.enhance(noisy)[0].cpu().numpy()
            clean = case["clean"]
            row = dict(case["meta"])
            row["enhancer"] = self.enhancer.name
            for m in self.metric_names:
                fn = METRICS.get(m)
                row[f"noisy_{m}"] = fn(clean, case["noisy"])
                row[f"enhanced_{m}"] = fn(clean, enhanced)
            rows.append(row)
            print(f"  {row['sample_id']:24s} "
                  + " ".join(f"{m}:{row[f'enhanced_{m}']:.2f}" for m in self.metric_names))

        scores_path = self.out_dir / "scores.jsonl"
        with open(scores_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        self._summarize(rows)
        print(f"\n明细: {scores_path}\n汇总: {self.out_dir / 'summary.csv'}")
        return rows

    def _summarize(self, rows):
        groups = {}
        for r in rows:
            key = (r["language"], r["noise_type"], r["snr_db"])
            groups.setdefault(key, []).append(r)
        summary_path = self.out_dir / "summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["language", "noise_type", "snr_db", "n"]
            for m in self.metric_names:
                header += [f"noisy_{m}", f"enhanced_{m}", f"delta_{m}"]
            writer.writerow(header)
            for (lang, kind, snr_db), rs in sorted(groups.items()):
                line = [lang, kind, snr_db, len(rs)]
                for m in self.metric_names:
                    n = np.nanmean([r[f"noisy_{m}"] for r in rs])
                    e = np.nanmean([r[f"enhanced_{m}"] for r in rs])
                    line += [round(n, 3), round(e, 3), round(e - n, 3)]
                writer.writerow(line)
