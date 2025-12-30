import numpy as np
from scipy.signal import welch

from eeg_quick_assess.features.bandpower import BANDS

PAIR_MAP = [("F3", "F4"), ("C3", "C4"), ("P3", "P4"), ("O1", "O2")]


def _band_power(freqs, psd, band):
    low, high = band
    idx = (freqs >= low) & (freqs <= high)
    return np.trapz(psd[..., idx], freqs[idx], axis=-1)


def compute_laterality(data, meta, cfg):
    ch_names = meta.get("ch_names", [])
    sfreq = meta["sfreq"]
    nperseg = min(cfg["features"]["welch_nperseg"], data.shape[1])
    f, pxx = welch(data, fs=sfreq, nperseg=nperseg, axis=1)

    laterality = {}
    for left, right in PAIR_MAP:
        if left in ch_names and right in ch_names:
            li = ch_names.index(left)
            ri = ch_names.index(right)
            alpha_left = _band_power(f, pxx[li], BANDS["alpha"])
            alpha_right = _band_power(f, pxx[ri], BANDS["alpha"])
            diff = float(alpha_right - alpha_left)
            laterality[f"{left}_{right}"] = {
                "alpha_diff": diff,
                "direction": "right" if diff > 0 else "left" if diff < 0 else "balanced",
            }

    faa = None
    if "F3" in ch_names and "F4" in ch_names:
        li = ch_names.index("F3")
        ri = ch_names.index("F4")
        alpha_left = _band_power(f, pxx[li], BANDS["alpha"])
        alpha_right = _band_power(f, pxx[ri], BANDS["alpha"])
        faa = float(np.log(alpha_right + 1e-12) - np.log(alpha_left + 1e-12))

    return {
        "laterality_pairs": laterality,
        "faa": faa,
        "config": {"pairs": PAIR_MAP},
    }
