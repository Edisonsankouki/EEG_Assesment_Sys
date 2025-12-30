import numpy as np
from scipy.signal import welch

from eeg_quick_assess.qc.artifact_flags import compute_window_artifacts


def compute_qc(data, windows, meta, cfg):
    sfreq = meta["sfreq"]
    duration = data.shape[1] / sfreq

    f, pxx = welch(data, fs=sfreq, nperseg=min(2048, data.shape[1]), axis=1)
    total = np.trapz(pxx[:, (f >= 1) & (f <= 45)], f[(f >= 1) & (f <= 45)])
    line_freq = meta.get("linefreq", cfg["qc"]["line_freq"])
    line_band = cfg["qc"]["line_band"]
    emg_band = cfg["qc"]["emg_band"]

    line = np.trapz(pxx[:, (f >= line_freq - line_band) & (f <= line_freq + line_band)], f[(f >= line_freq - line_band) & (f <= line_freq + line_band)])
    emg = np.trapz(pxx[:, (f >= emg_band[0]) & (f <= emg_band[1])], f[(f >= emg_band[0]) & (f <= emg_band[1])])

    line_noise_ratio = float(np.mean(line / (total + 1e-12)))
    emg_ratio = float(np.mean(emg / (total + 1e-12)))
    max_abs = float(np.max(np.abs(data)))

    flags = compute_window_artifacts(windows, sfreq, cfg)
    artifact_windows = sum(
        1 for f in flags if f["amp_exceed"] or f["grad_exceed"] or f["line_exceed"] or f["emg_exceed"]
    )
    artifact_ratio = artifact_windows / max(len(flags), 1)

    return {
        "valid_duration_sec": duration,
        "artifact_window_ratio": artifact_ratio,
        "line_noise_ratio": line_noise_ratio,
        "emg_ratio": emg_ratio,
        "max_abs_uV": max_abs,
        "artifact_flags": flags,
    }
