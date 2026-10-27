import numpy as np

from speechlab.dsp.stft import stft, istft


def test_stft_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000).astype(np.float32)
    rec = istft(stft(x), length=len(x))
    assert np.max(np.abs(x - rec)) < 1e-5
