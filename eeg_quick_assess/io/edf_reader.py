from pathlib import Path

import mne


def read_edf(path: Path):
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    data = raw.get_data()
    meta = {
        "sfreq": float(raw.info["sfreq"]),
        "ch_names": list(raw.ch_names),
        "reference": None,
        "linefreq": 50.0,
        "input_type": "edf",
    }
    return data, meta
