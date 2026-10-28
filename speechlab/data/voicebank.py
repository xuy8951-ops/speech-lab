"""VoiceBank-DEMAND 准备、冻结 manifest 与波形加载。

原始 48 kHz 成对音频统一转为单声道 float32/16 kHz；测试 manifest 一次写入后
只可校验，不能由准备脚本自动覆盖。所有路径均默认相对于仓库根目录。
"""
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 16000
ARCHIVES = {
    "train": ("clean_trainset_28spk_wav", "noisy_trainset_28spk_wav"),
    "test": ("clean_testset_wav", "noisy_testset_wav"),
}
LOGS = {"train": "log_trainset_28spk.txt", "test": "log_testset.txt"}


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _portable_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_wav(path):
    sr, data = wavfile.read(path)
    if data.ndim != 1:
        raise ValueError(f"只接受单声道音频: {path}，实际形状 {data.shape}")
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = float(max(abs(info.min), info.max))
        data = data.astype(np.float32) / scale
    else:
        data = data.astype(np.float32)
    if len(data) == 0 or not np.isfinite(data).all():
        raise ValueError(f"音频为空或含 NaN/Inf: {path}")
    return sr, data


def _safe_extract(zip_path, raw_root):
    with zipfile.ZipFile(zip_path) as zf:
        root = raw_root.resolve()
        for member in zf.namelist():
            target = (raw_root / member).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"ZIP 含越界路径，拒绝解压: {member}")
        zf.extractall(raw_root)


def _ensure_extracted(raw_root, split):
    _safe_extract(raw_root / "logfiles.zip", raw_root)
    for archive in ARCHIVES[split]:
        directory = raw_root / archive
        if directory.exists() and any(directory.glob("*.wav")):
            continue
        _safe_extract(raw_root / (archive + ".zip"), raw_root)


def _conditions(raw_root, split):
    with zipfile.ZipFile(raw_root / "logfiles.zip") as zf:
        lines = zf.read(LOGS[split]).decode("utf-8").splitlines()
    rows = []
    for line in lines:
        if not line.strip():
            continue
        sample_id, noise_type, snr_db = line.split()
        rows.append({
            "sample_id": sample_id,
            "speaker": sample_id.split("_")[0],
            "noise_type": noise_type,
            "snr_db": float(snr_db),
        })
    return sorted(rows, key=lambda row: row["sample_id"])


def _process_pair(clean_src, noisy_src, clean_dst, noisy_dst):
    if clean_dst.exists() and noisy_dst.exists():
        clean_sr, clean = _read_wav(clean_dst)
        noisy_sr, noisy = _read_wav(noisy_dst)
        if clean_sr != SAMPLE_RATE or noisy_sr != SAMPLE_RATE:
            raise ValueError(f"已处理音频采样率异常: {clean_dst} / {noisy_dst}")
    else:
        clean_sr, clean = _read_wav(clean_src)
        noisy_sr, noisy = _read_wav(noisy_src)
        if clean_sr != noisy_sr or len(clean) != len(noisy):
            raise ValueError(f"成对音频采样率或长度不一致: {clean_src} / {noisy_src}")
        if clean_sr != SAMPLE_RATE:
            divisor = np.gcd(clean_sr, SAMPLE_RATE)
            clean = resample_poly(clean, SAMPLE_RATE // divisor, clean_sr // divisor)
            noisy = resample_poly(noisy, SAMPLE_RATE // divisor, noisy_sr // divisor)
        clean = np.asarray(clean, dtype=np.float32)
        noisy = np.asarray(noisy, dtype=np.float32)

    n = min(len(clean), len(noisy))
    clean, noisy = clean[:n], noisy[:n]
    if n == 0 or not np.isfinite(clean).all() or not np.isfinite(noisy).all():
        raise ValueError(f"清洗后音频无效: {clean_dst} / {noisy_dst}")
    if not clean_dst.exists() or not noisy_dst.exists():
        clean_dst.parent.mkdir(parents=True, exist_ok=True)
        noisy_dst.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(clean_dst, SAMPLE_RATE, clean)
        wavfile.write(noisy_dst, SAMPLE_RATE, noisy)
    return n


def _manifest_row(condition, split, raw_root, out_root, with_hashes=False):
    clean_dir, noisy_dir = ARCHIVES[split]
    filename = condition["sample_id"] + ".wav"
    clean_dst = out_root / (clean_dir + "_16k") / filename
    noisy_dst = out_root / (noisy_dir + "_16k") / filename
    n = _process_pair(
        raw_root / clean_dir / filename,
        raw_root / noisy_dir / filename,
        clean_dst,
        noisy_dst,
    )
    row = dict(condition)
    row.update({
        "clean": _portable_path(clean_dst),
        "noisy": _portable_path(noisy_dst),
        "split": split,
        "language": "en",
        "sample_rate": SAMPLE_RATE,
        "num_samples": n,
        "duration": round(n / SAMPLE_RATE, 6),
    })
    if with_hashes:
        row["clean_sha256"] = _sha256(clean_dst)
        row["noisy_sha256"] = _sha256(noisy_dst)
    return row


def _encode(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def _write_replaceable(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_encode(rows))
    os.replace(temporary, path)


def _write_frozen(path, rows):
    """测试清单写一次；内容变化时拒绝覆盖，保护历史指标可比性。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _encode(rows)
    digest = hashlib.sha256(payload).hexdigest()
    checksum_path = path.with_suffix(".sha256")
    if path.exists():
        verify_frozen_manifest(path.parent)
        if path.read_bytes() != payload:
            raise RuntimeError(
                f"冻结测试 manifest 内容发生变化，已拒绝覆盖: {path}。"
                "若确为数据处理错误，请人工审查并新建数据集版本，不要原地改写。"
            )
    else:
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    expected = f"{digest}  {path.name}\n"
    if checksum_path.exists() and checksum_path.read_text() != expected:
        raise RuntimeError(f"冻结校验和异常: {checksum_path}")
    checksum_path.write_text(expected)


def verify_frozen_manifest(manifest_dir):
    path = _resolve(manifest_dir) / "test.jsonl"
    checksum_path = path.with_suffix(".sha256")
    if not path.exists() or not checksum_path.exists():
        raise FileNotFoundError(f"冻结测试 manifest 或校验和缺失: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = checksum_path.read_text().split()[0]
    if actual != expected:
        raise RuntimeError(f"冻结测试 manifest 校验失败: {path}")


def prepare_voicebank(raw_root, out_root, manifest_dir, val_speakers=("p226", "p227"),
                      splits=("train", "test"), allow_partial=False):
    if allow_partial and "test" in splits:
        raise ValueError("冻结测试集不允许使用部分数据模式")
    raw_root, out_root, manifest_dir = map(_resolve, (raw_root, out_root, manifest_dir))
    all_rows = {"train": [], "val": [], "test": []}
    for split in splits:
        _ensure_extracted(raw_root, split)
        conditions = _conditions(raw_root, split)
        rows = []
        for index, condition in enumerate(conditions, 1):
            filename = condition["sample_id"] + ".wav"
            raw_pair = [raw_root / directory / filename for directory in ARCHIVES[split]]
            if allow_partial and not all(path.exists() for path in raw_pair):
                continue
            rows.append(_manifest_row(condition, split, raw_root, out_root, with_hashes=split == "test"))
            if index % 250 == 0 or index == len(conditions):
                print(f"{split}: 已扫描 {index}/{len(conditions)}，保留 {len(rows)}", flush=True)
        if not rows:
            raise ValueError(f"{split} 没有可用音频")
        if split == "test":
            all_rows["test"] = rows
            _write_frozen(manifest_dir / "test.jsonl", rows)
        else:
            val_speakers = set(val_speakers)
            speakers = {row["speaker"] for row in rows}
            missing = val_speakers - speakers
            if missing and not allow_partial:
                raise ValueError(f"验证说话人不在训练数据中: {sorted(missing)}")
            all_rows["val"] = [row for row in rows if row["speaker"] in val_speakers]
            all_rows["train"] = [row for row in rows if row["speaker"] not in val_speakers]
            for row in all_rows["val"]:
                row["split"] = "val"
            _write_replaceable(manifest_dir / "train.jsonl", all_rows["train"])
            _write_replaceable(manifest_dir / "val.jsonl", all_rows["val"])
    return {split: len(rows) for split, rows in all_rows.items() if rows or split in splits}


def _subsample(rows, limit, seed):
    if limit is None or limit >= len(rows):
        return rows
    if limit <= 0:
        raise ValueError("数据子集大小必须为正整数")
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(rows), size=limit, replace=False).tolist())
    return [rows[index] for index in indices]


def load_voicebank_cases(cfg, split):
    if split not in {"train", "val", "test"}:
        raise ValueError(f"VoiceBank 不支持的数据划分: {split}")
    default_manifest_dir = cfg.get("manifest_dir", "data/manifests/voicebank_demand")
    manifest_key = f"{split}_manifest_dir" if split in {"val", "test"} else "manifest_dir"
    manifest_dir = _resolve(cfg.get(manifest_key, default_manifest_dir))
    if split == "test":
        verify_frozen_manifest(manifest_dir)
    path = manifest_dir / f"{split}.jsonl"
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    import zlib
    seed = int(cfg.get("seed", 42)) + zlib.crc32(split.encode("utf-8"))
    rows = _subsample(rows, cfg.get(f"max_{split}"), seed)

    cases = []
    verify_audio = split == "test" and cfg.get("verify_audio_hash", True)
    for row in rows:
        clean_path, noisy_path = _resolve(row["clean"]), _resolve(row["noisy"])
        clean_sr, clean = _read_wav(clean_path)
        noisy_sr, noisy = _read_wav(noisy_path)
        if clean_sr != SAMPLE_RATE or noisy_sr != SAMPLE_RATE or len(clean) != len(noisy):
            raise ValueError(f"manifest 音频不符合 16 kHz 成对约定: {row['sample_id']}")
        if verify_audio and (
            _sha256(clean_path) != row["clean_sha256"]
            or _sha256(noisy_path) != row["noisy_sha256"]
        ):
            raise RuntimeError(f"冻结测试音频校验失败: {row['sample_id']}")
        meta = {key: value for key, value in row.items() if key not in {"clean", "noisy"}}
        cases.append({"clean": clean, "noisy": noisy, "meta": meta})
    return cases
