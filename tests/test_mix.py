import numpy as np

from speechlab.data.synthesize import mix, synth_speech_like, synth_noise


def test_mix_snr_accuracy():
    clean = synth_speech_like(2.0, seed=1)
    noise = synth_noise("white", len(clean), seed=2)
    for target in [0.0, 5.0, 10.0]:
        noisy, scaled_noise = mix(clean, noise, target, seed=3)
        actual = 10 * np.log10(np.mean(clean ** 2) / np.mean(scaled_noise ** 2))
        assert abs(actual - target) < 0.1
