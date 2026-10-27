"""谱门控基线：迁移自 speech-denoiser-mini 的已验证实现（合成数据 SNR 5->13dB）。

算法：噪声底估计（中位数谱低分位数）→ 维纳式软掩蔽 → 时间平滑 → ISTFT。
"""
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from speechlab.dsp.stft import stft, istft
from speechlab.engine.registry import ENHANCERS
from speechlab.models.base import Enhancer


def spectral_gate(noisy: np.ndarray, strength: float = 1.0, floor: float = 0.02):
    spec = stft(noisy)
    mag, phase = np.abs(spec), np.angle(spec)

    # 噪声估计：假设背景噪声频谱平坦（白噪/风扇类）
    med = np.median(mag, axis=0)
    floor_est = np.percentile(med, 15)
    thresh = (1.0 + 0.5 * strength) * floor_est

    # 维纳式软掩蔽 + 时间平滑（抑制音乐噪声）
    snr_prior = np.maximum((mag / thresh) ** 2 - 1, 0)
    mask = np.maximum(snr_prior / (snr_prior + 1), floor)
    mask = uniform_filter1d(mask, size=5, axis=0)

    enhanced = mag * mask * np.exp(1j * phase)
    return istft(enhanced, length=len(noisy))


@ENHANCERS.register("spectral_gate")
class SpectralGate(Enhancer):
    name = "spectral_gate"

    def __init__(self, strength: float = 1.0):
        self.strength = strength

    def enhance(self, noisy: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        out = [spectral_gate(x.cpu().numpy(), self.strength) for x in noisy]
        return torch.from_numpy(np.stack(out)).to(noisy.device)
