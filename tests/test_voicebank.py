import json
import zipfile

import numpy as np
import pytest
from scipy.io import wavfile

from speechlab.data.voicebank import prepare_voicebank, verify_frozen_manifest


def _write_wav(path, sr=48000, frequency=220.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(sr // 10) / sr
    wavfile.write(path, sr, (0.1 * np.sin(2 * np.pi * frequency * t)).astype(np.float32))


def _fixture(root):
    root.mkdir(parents=True)
    entries = {
        "train": [("p226_001", "bus", 5), ("p228_001", "car", 10)],
        "test": [("p257_001", "cafe", 2.5)],
    }
    with zipfile.ZipFile(root / "logfiles.zip", "w") as zf:
        for split, rows in entries.items():
            log_name = "log_trainset_28spk.txt" if split == "train" else "log_testset.txt"
            zf.writestr(log_name, "".join(f"{sid} {noise} {snr}\n" for sid, noise, snr in rows))
        for split, rows in entries.items():
            clean_dir = "clean_trainset_28spk_wav" if split == "train" else "clean_testset_wav"
            noisy_dir = "noisy_trainset_28spk_wav" if split == "train" else "noisy_testset_wav"
            for sid, _, _ in rows:
                _write_wav(root / clean_dir / f"{sid}.wav")
                _write_wav(root / noisy_dir / f"{sid}.wav", frequency=440.0)
    return entries


def test_prepare_load_and_freeze(tmp_path):
    raw = tmp_path / "raw"
    _fixture(raw)
    manifests = tmp_path / "manifests"
    counts = prepare_voicebank(raw, tmp_path / "processed", manifests, val_speakers=("p226",))
    assert counts == {"train": 1, "val": 1, "test": 1}
    verify_frozen_manifest(manifests)

    rows = [json.loads(line) for line in (manifests / "test.jsonl").read_text().splitlines()]
    assert rows[0]["noise_type"] == "cafe"
    assert rows[0]["snr_db"] == 2.5
    assert rows[0]["sample_rate"] == 16000
    sr, _ = wavfile.read(tmp_path / "processed" / "clean_testset_wav_16k" / "p257_001.wav")
    assert sr == 16000

    original = (manifests / "test.jsonl").read_bytes()
    (manifests / "test.jsonl").write_bytes(original + b"{}\n")
    with pytest.raises(RuntimeError, match="校验失败"):
        prepare_voicebank(raw, tmp_path / "processed", manifests, splits=("test",))
