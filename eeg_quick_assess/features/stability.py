import numpy as np
from scipy.signal import welch

from eeg_quick_assess.features.bandpower import BANDS


def _band_power(freqs, psd, band):
    low, high = band
    idx = (freqs >= low) & (freqs <= high)
    return np.trapz(psd[..., idx], freqs[idx], axis=-1)


def compute_stability(windows, meta, cfg):
    sfreq = meta["sfreq"]
    if windows.size == 0:
        return {"bandpower_cv": {}, "ratio_cv": {}, "bandpower_drift": {}, "ratio_drift": {}}

    band_series = {name: [] for name in BANDS}
    ratio_series = {"theta_beta": [], "beta_alpha": [], "delta_theta_alpha": []}

    for win in windows:
        f, pxx = welch(win, fs=sfreq, nperseg=min(512, win.shape[1]), axis=1)
        for name, band in BANDS.items():
            power = _band_power(f, pxx, band)
            band_series[name].append(np.mean(power))
        total = sum(band_series[name][-1] for name in BANDS) + 1e-12
        rel = {name: band_series[name][-1] / total for name in BANDS}
        ratio_series["theta_beta"].append(rel["theta"] / (rel["beta"] + 1e-12))
        ratio_series["beta_alpha"].append(rel["beta"] / (rel["alpha"] + 1e-12))
        ratio_series["delta_theta_alpha"].append((rel["theta"] + rel["delta"]) / (rel["alpha"] + 1e-12))

    def _cv(x):
        arr = np.array(x, dtype=float)
        return float(np.std(arr) / (np.mean(arr) + 1e-12))

    def _drift(x):
        arr = np.array(x, dtype=float)
        idx = np.arange(len(arr))
        slope = np.polyfit(idx, arr, 1)[0]
        return float(slope)

    return {
        "bandpower_cv": {k: _cv(v) for k, v in band_series.items()},
        "ratio_cv": {k: _cv(v) for k, v in ratio_series.items()},
        "bandpower_drift": {k: _drift(v) for k, v in band_series.items()},
        "ratio_drift": {k: _drift(v) for k, v in ratio_series.items()},
    }
