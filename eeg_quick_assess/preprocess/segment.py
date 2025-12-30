import numpy as np


def segment_windows(data, meta, cfg):
    sfreq = meta["sfreq"]
    win_sec = cfg["windowing"]["window_sec"]
    overlap = cfg["windowing"]["overlap"]
    win_len = int(win_sec * sfreq)
    step = int(win_len * (1 - overlap))
    if step <= 0:
        raise ValueError("Overlap too large, step <= 0")

    windows = []
    for start in range(0, data.shape[1] - win_len + 1, step):
        windows.append(data[:, start : start + win_len])
    return np.stack(windows, axis=0) if windows else np.empty((0, data.shape[0], win_len))
