import numpy as np
from scipy.signal import welch


def compute_psd(data, meta, cfg):
    sfreq = meta["sfreq"]
    nperseg = min(cfg["features"]["welch_nperseg"], data.shape[1])
    freqs, pxx = welch(data, fs=sfreq, nperseg=nperseg, axis=1)
    return {
        "freqs": freqs.tolist(),
        "psd": pxx.tolist(),
        "config": {"nperseg": nperseg, "sfreq": sfreq},
    }
