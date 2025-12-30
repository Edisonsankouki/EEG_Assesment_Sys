import argparse
import json
from pathlib import Path

from eeg_quick_assess.io.edf_reader import read_edf
from eeg_quick_assess.io.numpy_reader import read_numpy
from eeg_quick_assess.pipeline import run_assessment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EEG Quick Assess CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run EEG quick assessment")
    run.add_argument("--in", dest="input_path", required=True, help="Input EDF or .npy file")
    run.add_argument("--sfreq", type=float, default=None, help="Sampling frequency (Hz)")
    run.add_argument("--linefreq", type=float, default=50.0, help="Line noise frequency (Hz)")
    run.add_argument("--out", dest="out_dir", required=True, help="Output directory")
    run.add_argument("--ch-names", nargs="*", default=None, help="Channel names for numpy input")
    run.add_argument("--reference", default=None, help="Reference mode: none/average/linked_ears")
    run.add_argument("--disable-llm", action="store_true", help="Disable LLM summarizer")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() == ".edf":
        data, meta = read_edf(input_path)
    elif input_path.suffix.lower() == ".npy":
        if args.sfreq is None:
            raise SystemExit("--sfreq is required for numpy input")
        data, meta = read_numpy(input_path, args.sfreq, args.ch_names)
    else:
        raise SystemExit("Unsupported input. Provide .edf or .npy")

    if args.reference is not None:
        meta["reference"] = args.reference
    meta["linefreq"] = args.linefreq

    result = run_assessment(data, meta, use_llm=not args.disable_llm)

    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    for module_key in ["psych", "physio", "cognitive", "risk"]:
        (out_dir / f"{module_key}.txt").write_text(result["modules"][module_key]["module_text_rule"], encoding="utf-8")

    summary_text = result["final"].get("final_report_text_llm") or result["final"].get("fallback_text")
    (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")


if __name__ == "__main__":
    main()
