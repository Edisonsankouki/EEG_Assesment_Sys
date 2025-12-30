import numpy as np


BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def _band_power(freqs, psd, band):
    low, high = band
    idx = (freqs >= low) & (freqs <= high)
    return np.trapz(psd[..., idx], freqs[idx], axis=-1)


def compute_bandpower(psd_dict, meta, cfg):
    freqs = np.array(psd_dict["freqs"])
    psd = np.array(psd_dict["psd"])

    band_abs = {}
    for name, band in BANDS.items():
        power = _band_power(freqs, psd, band)
        band_abs[name] = float(np.mean(power))

    total_power = sum(band_abs.values()) + 1e-12
    band_rel = {k: float(v / total_power) for k, v in band_abs.items()}

    iaf = None
    alpha_idx = (freqs >= BANDS["alpha"][0]) & (freqs <= BANDS["alpha"][1])
    if alpha_idx.any():
        alpha_psd = np.mean(psd[:, alpha_idx], axis=0)
        iaf = float(freqs[alpha_idx][np.argmax(alpha_psd)])

    return {
        "absolute": band_abs,
        "relative": band_rel,
        "iaf_hz": iaf,
        "config": {"bands": BANDS},
    }
