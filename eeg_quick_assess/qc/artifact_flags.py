import numpy as np
from scipy.signal import welch


def compute_window_artifacts(windows, sfreq, cfg):
    amp_th = cfg["qc"]["amplitude_uV"]
    grad_th = cfg["qc"]["gradient_uV"]
    line_freq = cfg["qc"]["line_freq"]
    emg_band = cfg["qc"]["emg_band"]
    line_band = cfg["qc"]["line_band"]

    flags = []
    for win in windows:
        max_abs = np.max(np.abs(win))
        grad = np.max(np.abs(np.diff(win, axis=1)))
        f, pxx = welch(win, fs=sfreq, nperseg=min(512, win.shape[1]), axis=1)
        total = np.trapz(pxx[:, (f >= 1) & (f <= 45)], f[(f >= 1) & (f <= 45)])
        line = np.trapz(pxx[:, (f >= line_freq - line_band) & (f <= line_freq + line_band)], f[(f >= line_freq - line_band) & (f <= line_freq + line_band)])
        emg = np.trapz(pxx[:, (f >= emg_band[0]) & (f <= emg_band[1])], f[(f >= emg_band[0]) & (f <= emg_band[1])])
        line_ratio = float(np.mean(line / (total + 1e-12)))
        emg_ratio = float(np.mean(emg / (total + 1e-12)))
        flag = {
            "amp_exceed": bool(max_abs > amp_th),
            "grad_exceed": bool(grad > grad_th),
            "line_exceed": bool(line_ratio > cfg["qc"]["line_ratio_th"]),
            "emg_exceed": bool(emg_ratio > cfg["qc"]["emg_ratio_th"]),
        }
        flags.append(flag)
    return flags
