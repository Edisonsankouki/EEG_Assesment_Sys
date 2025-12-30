from pathlib import Path

import numpy as np


def read_numpy(path: Path, sfreq: float, ch_names=None):
    data = np.load(path)
    if data.ndim != 2:
        raise ValueError("numpy input must have shape [n_channels, n_samples]")
    if ch_names is None:
        ch_names = [f"Ch{i+1}" for i in range(data.shape[0])]
    meta = {
        "sfreq": float(sfreq),
        "ch_names": list(ch_names),
        "reference": None,
        "linefreq": 50.0,
        "input_type": "numpy",
    }
    return data, meta
