import argparse
import json
from pathlib import Path

from eeg_quick_assess.io.edf_reader import read_edf
from eeg_quick_assess.pipeline import run_assessment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf", required=True, help="Path to EDF file")
    args = parser.parse_args()

    data, meta = read_edf(Path(args.edf))
    result = run_assessment(data, meta, use_llm=False)
    out = Path(__file__).parent / "sample_output.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved sample output to {out}")


if __name__ == "__main__":
    main()
