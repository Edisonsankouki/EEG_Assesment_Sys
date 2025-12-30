import json
from pathlib import Path

import numpy as np

from eeg_quick_assess.pipeline import run_assessment


def main():
    sfreq = 250
    duration_sec = 6 * 60
    n_samples = sfreq * duration_sec
    ch_names = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    rng = np.random.default_rng(42)
    data = rng.normal(0, 20, size=(len(ch_names), n_samples))

    meta = {
        "sfreq": sfreq,
        "ch_names": ch_names,
        "reference": "average",
        "linefreq": 50.0,
        "input_type": "numpy",
    }

    result = run_assessment(data, meta, use_llm=False)
    out = Path(__file__).parent / "sample_output.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved sample output to {out}")


if __name__ == "__main__":
    main()
