"""指标：S0 内置 SNR / SI-SDR（无外部依赖）；PESQ/STOI 为可选适配器。"""
import numpy as np

from speechlab.engine.registry import METRICS


def _align(ref, est):
    n = min(len(ref), len(est))
    return ref[:n], est[:n]


@METRICS.register("snr")
def snr(clean, est, **kw):
    ref, est = _align(np.asarray(clean), np.asarray(est))
    err = ref - est
    return float(10 * np.log10(np.mean(ref ** 2) / (np.mean(err ** 2) + 1e-12)))


@METRICS.register("si_sdr")
def si_sdr(clean, est, **kw):
    ref, est = _align(np.asarray(clean), np.asarray(est))
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10(np.sum(target ** 2) / (np.sum(noise ** 2) + 1e-12)))


@METRICS.register("pesq")
def pesq(clean, est, **kw):
    try:
        from pesq import pesq as _pesq
    except ImportError:
        return float("nan")  # 可选依赖，缺失时记 NaN 而不是中断评测
    ref, est = _align(np.asarray(clean), np.asarray(est))
    return float(_pesq(16000, ref, est, "wb"))
