import numpy as np

from eeg_quick_assess.features.psd import compute_psd
from eeg_quick_assess.features.bandpower import compute_bandpower


def test_bandpower_and_iaf():
    sfreq = 100
    t = np.arange(0, 10, 1 / sfreq)
    signal = np.sin(2 * np.pi * 10 * t)
    data = np.stack([signal, signal])
    meta = {"sfreq": sfreq}
    cfg = {"features": {"welch_nperseg": 256}}

    psd = compute_psd(data, meta, cfg)
    bandpower = compute_bandpower(psd, meta, cfg)

    iaf = bandpower["iaf_hz"]
    assert iaf is not None
    assert 9.0 <= iaf <= 11.0
    assert bandpower["relative"]["alpha"] > bandpower["relative"]["theta"]
