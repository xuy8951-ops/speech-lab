"""TinyGRUEnhancer：谱掩蔽 GRU（v1 模型）。

训练走 forward(mag)->mask（可微），推理解 Enhancer 波形接口。
结构：log1p幅度谱 → 双向GRU → FC+sigmoid → 时频掩蔽，乘回带噪谱保留相位。
"""
import numpy as np
import torch
import torch.nn as nn

from speechlab.dsp.stft import stft, istft, N_FFT
from speechlab.engine.registry import ENHANCERS
from speechlab.models.base import Enhancer


@ENHANCERS.register("tiny_gru")
class TinyGRUEnhancer(Enhancer, nn.Module):
    name = "tiny_gru"

    def __init__(self, hidden: int = 128, layers: int = 2):
        nn.Module.__init__(self)
        self.freq_bins = N_FFT // 2 + 1
        self.gru = nn.GRU(self.freq_bins, hidden, layers,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, self.freq_bins)

    def forward(self, mag: torch.Tensor) -> torch.Tensor:
        """mag: (B, T, 257) 幅度谱 → mask (B, T, 257)，取值(0,1)。"""
        h, _ = self.gru(torch.log1p(mag))
        return torch.sigmoid(self.fc(h))

    def load(self, ckpt_path: str) -> None:
        self.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def enhance(self, noisy: torch.Tensor, sr: int = 16000) -> torch.Tensor:
        self.eval()
        outs = []
        for x in noisy.cpu().numpy():
            spec = stft(x)
            mag, phase = np.abs(spec), np.angle(spec)
            m = torch.from_numpy(mag).float()[None]
            mask = self.forward(m)[0].numpy()
            enhanced = mag * mask * np.exp(1j * phase)
            outs.append(istft(enhanced, length=len(x)))
        return torch.from_numpy(np.stack(outs)).to(noisy.device)
