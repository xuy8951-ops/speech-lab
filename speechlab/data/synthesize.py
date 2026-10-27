"""数据合成：SNR 混合 + 教学用合成语音/噪声（S0 最小数据引擎）。

真实数据集（VoiceBank-DEMAND / AISHELL）在 S2 接入；
本模块的 mix() 与 manifest 思想保持不变。
"""
import numpy as np

SR = 16000


def mix(clean, noise, snr_db, seed=0):
    """按指定 SNR 混合；单测要求实测 SNR 与设定误差 <0.1dB。"""
    rng = np.random.default_rng(seed)
    noise = np.asarray(noise, dtype=np.float32)
    if len(noise) < len(clean):  # 噪声循环拼接
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)
    start = rng.integers(0, len(noise) - len(clean) + 1) if len(noise) > len(clean) else 0
    noise = noise[start:start + len(clean)]

    p_clean, p_noise = np.mean(clean ** 2), np.mean(noise ** 2)
    noise = noise * np.sqrt(p_clean / (p_noise * 10 ** (snr_db / 10)))
    return (clean + noise).astype(np.float32), noise


def synth_speech_like(duration=2.0, sr=SR, seed=0):
    """合成类语音（谐波+共振峰包络+清音段），仅用于教学/冒烟测试。
    基频/调制/包络参数随 seed 变化，避免不同样本之间相关。"""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    f0_base = 90 + seed % 140            # 不同"说话人"基频不同
    mod_rate = 0.5 + (seed % 5) * 0.15
    env_rate = 1.2 + (seed % 7) * 0.25
    f0 = f0_base + 0.3 * f0_base * np.sin(2 * np.pi * mod_rate * t)
    sig = np.zeros_like(t)
    for h in range(1, 12):
        amp = np.exp(-h / 6) * (1 + 1.5 * np.exp(-((h * f0.mean() - 700) ** 2) / 300 ** 2)
                                + 1.2 * np.exp(-((h * f0.mean() - 2200) ** 2) / 500 ** 2))
        phase = np.cumsum(2 * np.pi * h * f0 / sr)
        sig += amp * np.sin(phase)
    burst = rng.standard_normal(len(t)) * 0.15
    gate = (np.sin(2 * np.pi * 1.3 * t) > 0.6).astype(float)
    sig += burst * gate
    env = 0.5 + 0.5 * np.sin(2 * np.pi * env_rate * t) ** 2
    sig *= env
    return (0.9 * sig / np.max(np.abs(sig))).astype(np.float32)


def synth_noise(kind, n, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "white":
        return rng.standard_normal(n).astype(np.float32) * 0.1
    if kind == "pink":
        # 简易粉噪：白噪频域加权 1/sqrt(f)
        x = rng.standard_normal(n)
        X = np.fft.rfft(x)
        f = np.maximum(np.fft.rfftfreq(n), 1e-3)
        X /= np.sqrt(f / f[1])
        y = np.fft.irfft(X, n)
        return (y / (np.std(y) + 1e-9) * 0.1).astype(np.float32)
    if kind == "babble":
        # 多"说话人"类语音叠加模拟 babble
        out = np.zeros(n, dtype=np.float32)
        for k in range(4):
            s = synth_speech_like(duration=n / SR, seed=100 + k)
            out += s[:n] * 0.25
        return out
    raise ValueError(f"未知噪声类型: {kind}")


def make_train_cases(n=6, duration=2.0, snr_range=(0, 10), seed=1234):
    """训练/验证用合成样本（S1 过拟合验证；S2 由真实数据集替代）。"""
    rng = np.random.default_rng(seed)
    kinds = ["white", "pink", "babble"]
    cases = []
    for i in range(n):
        clean = synth_speech_like(duration, seed=seed + i)
        kind = kinds[i % len(kinds)]
        noise = synth_noise(kind, len(clean), seed=seed + 1000 + i)
        snr_db = float(rng.uniform(*snr_range))
        noisy, _ = mix(clean, noise, snr_db, seed=seed + 2000 + i)
        cases.append(dict(clean=clean, noisy=noisy,
                          meta=dict(snr_db=snr_db, noise_type=kind,
                                    language="synth", sample_id=f"train_{i}")))
    return cases


def frozen_test_set(n_per_case=2, duration=2.0):
    """冻结的合成测试集：固定 seed，永不重生成（指标可比性的前提）。

    返回 list[dict]: clean/noisy/noise/meta{snr_db, noise_type, language, sample_id}
    """
    cases = []
    import zlib
    for snr in [0.0, 5.0, 10.0]:
        for kind in ["white", "pink", "babble"]:
            for i in range(n_per_case):
                # 用 crc32 代替 hash()：跨进程/跨机器结果恒定
                seed = zlib.crc32(f"{snr}|{kind}|{i}".encode()) % (2 ** 31)
                clean = synth_speech_like(duration, seed=seed)
                noise = synth_noise(kind, len(clean), seed=seed + 1)
                noisy, _ = mix(clean, noise, snr, seed=seed + 2)
                cases.append(dict(
                    clean=clean, noisy=noisy,
                    meta=dict(snr_db=snr, noise_type=kind, language="synth",
                              sample_id=f"{kind}_{snr}_{i}"),
                ))
    return cases
