#!/usr/bin/env python3
"""诊断带噪相位瓶颈；oracle 使用测试 clean，仅用于分析，不是模型成绩。"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speechlab.data.datasets import load_cases
from speechlab.dsp.stft import stft, istft
from speechlab.eval.metrics import snr, si_sdr, stoi


def summarize(rows, metric_names):
    result = {}
    for metric in metric_names:
        noisy = float(np.mean([row[f"noisy_{metric}"] for row in rows]))
        enhanced = float(np.mean([row[f"enhanced_{metric}"] for row in rows]))
        result[metric] = {
            "noisy": round(noisy, 6),
            "enhanced": round(enhanced, 6),
            "delta": round(enhanced - noisy, 6),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/exp/003_gru_voicebank_multispk.yaml")
    parser.add_argument("--output", default="runs/analysis/phase_oracle")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data"].setdefault("seed", cfg["seed"])
    cases = load_cases(cfg["data"], "test")
    metric_fns = {"snr": snr, "si_sdr": si_sdr, "stoi": stoi}
    variants = ("clean_mag_noisy_phase", "ideal_mask_noisy_phase")
    rows = {name: [] for name in variants}

    for index, case in enumerate(cases, 1):
        clean, noisy = case["clean"], case["noisy"]
        clean_spec, noisy_spec = stft(clean), stft(noisy)
        noisy_mag = np.abs(noisy_spec)
        clean_mag = np.abs(clean_spec)
        ideal_mask = np.clip(clean_mag / (noisy_mag + 1e-8), 0.0, 1.0)
        specs = {
            "clean_mag_noisy_phase": clean_mag * np.exp(1j * np.angle(noisy_spec)),
            "ideal_mask_noisy_phase": ideal_mask * noisy_spec,
        }
        for name, spec in specs.items():
            enhanced = istft(spec, length=len(noisy))
            row = dict(case["meta"])
            for metric, fn in metric_fns.items():
                row[f"noisy_{metric}"] = fn(clean, noisy)
                row[f"enhanced_{metric}"] = fn(clean, enhanced)
            rows[name].append(row)
        if index % 100 == 0 or index == len(cases):
            print(f"已分析 {index}/{len(cases)}", flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "warning": "Oracle 使用干净测试音频，仅用于定位误差，不是可部署模型成绩。",
        "n": len(cases),
        "overall": {name: summarize(variant_rows, metric_fns)
                    for name, variant_rows in rows.items()},
        "by_snr": {},
    }
    for snr_db in sorted({row["snr_db"] for row in rows[variants[0]]}):
        report["by_snr"][str(snr_db)] = {
            name: summarize([row for row in variant_rows if row["snr_db"] == snr_db], metric_fns)
            for name, variant_rows in rows.items()
        }
    with open(output / "summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    for name, variant_rows in rows.items():
        with open(output / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in variant_rows:
                f.write(json.dumps(row) + "\n")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    print(f"诊断报告: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
