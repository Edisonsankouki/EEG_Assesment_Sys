import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def apply_filters(data, meta, cfg):
    sfreq = meta["sfreq"]
    filt_cfg = cfg["preprocess"]
    low, high = filt_cfg["bandpass"]["low"], filt_cfg["bandpass"]["high"]
    b, a = butter(4, [low / (sfreq / 2), high / (sfreq / 2)], btype="band")
    filtered = filtfilt(b, a, data, axis=1)

    notch_freq = meta.get("linefreq", filt_cfg["notch_freq"])
    if notch_freq:
        bn, an = iirnotch(notch_freq / (sfreq / 2), Q=30)
        filtered = filtfilt(bn, an, filtered, axis=1)

    return filtered
