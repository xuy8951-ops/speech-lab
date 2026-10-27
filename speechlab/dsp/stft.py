"""全局唯一 STFT 入口。参数约定见 configs/const.yaml，禁止各处自定义。

COLA 可逆性由 tests/test_stft.py 保证（此前边界失真的教训固化为测试）。
"""
import numpy as np

N_FFT = 512
HOP = 128
SR = 16000


def stft(sig, n_fft=N_FFT, hop=HOP):
    """(T,) -> (n_frames, 257) 复数谱；两端 reflect 填充保证可逆。"""
    sig = np.asarray(sig, dtype=np.float32)
    pad = n_fft // 2
    sig = np.pad(sig, pad, mode="reflect")
    win = np.hanning(n_fft).astype(np.float32)
    frames = [np.fft.rfft(sig[i:i + n_fft] * win)
              for i in range(0, len(sig) - n_fft + 1, hop)]
    return np.array(frames)


def istft(spec, n_fft=N_FFT, hop=HOP, length=None):
    """(n_frames, 257) -> (T,) overlap-add 重建，自动去除填充。"""
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = len(spec)
    out = np.zeros(n_fft + hop * (n_frames - 1), dtype=np.float32)
    wsum = np.zeros_like(out)
    for t, frame in enumerate(spec):
        seg = np.fft.irfft(frame, n_fft)
        s = t * hop
        out[s:s + n_fft] += seg * win
        wsum[s:s + n_fft] += win ** 2
    wsum[wsum < 1e-8] = 1e-8
    out /= wsum
    pad = n_fft // 2
    out = out[pad:]
    if length is not None:
        out = out[:length]
    return out
