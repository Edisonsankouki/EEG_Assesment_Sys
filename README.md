# EEG Quick Assess

Deterministic, rule-based EEG assessment for a 6-minute recording. The system computes shared preprocessing, QC, and features, then produces four module reports plus a final summary. All scoring is **traditional algorithms + fixed rules**. An optional small LLM can **only** reformat the final summary; it may **not** change any scores, add dimensions, or make diagnoses.

> **Compliance**: This project never outputs diagnosis, treatment, prescriptions, or medical advice. Risk outputs are **screening hints only**. Physiological items like “brain oxygenation” are flagged as **EEG proxy indices**.

## Features
- EDF input support (MNE), plus numpy array input.
- Configurable preprocessing: bandpass, notch, re-reference, windowing.
- QC metrics with artifact windows and contamination ratios.
- Feature extraction: bandpower, ratios, IAF, spectral entropy, stability (CV + drift), laterality + FAA, permutation entropy.
- Deterministic scoring with fixed weights and thresholds in YAML configs.
- Module texts + LLM summary with strict JSON validation + fallback.

## Installation
```bash
pip install -e .
```

Optional LLM dependencies:
```bash
pip install -e ".[llm]"
```

## CLI Usage
```bash
eeg-qa run --in demo.edf --sfreq 250 --linefreq 50 --out ./out
```
Outputs:
```
./out/result.json
./out/psych.txt
./out/physio.txt
./out/cognitive.txt
./out/risk.txt
./out/summary.txt
```

## Python Examples
```bash
python examples/demo_run_numpy.py
python examples/demo_run_edf.py --edf ./path/to/file.edf
```

## Configuration
- `eeg_quick_assess/configs/runtime.yaml`: preprocessing, windowing, thresholds.
- `eeg_quick_assess/configs/scoring_rules.yaml`: weights, normalization ranges, score thresholds.
- `eeg_quick_assess/configs/sentence_bank.yaml`: fixed sentence library for each module/dimension/level, plus default LLM model.

## JSON Output Schema (Summary)
Top-level fields:
- `version`, `timestamp`, `meta`, `qc`
- `modules`: `psych`, `physio`, `cognitive`, `risk`
- `final`: `final_report_text_llm` or `fallback_text`, `llm_used`, `disclaimer`

Each module contains:
- `module_name`
- `dimensions`: `{dimension_name: {score, level, evidence[], fixed_text, missing_features?, confidence?}}`
- `module_text_rule`

See `examples/sample_output.json` for a concrete instance.

## Algorithms (Deterministic)
- PSD via Welch; bandpowers: delta/theta/alpha/beta/gamma (absolute + relative).
- Ratios: theta/beta, (theta+delta)/alpha, beta/alpha.
- IAF: peak in 8–13 Hz if available.
- Spectral entropy from PSD distribution.
- Stability: CV and drift (slope) over windowed features.
- Laterality: L/R pairs (F3/F4, C3/C4, P3/P4, O1/O2), plus FAA.
- Permutation entropy (simple) for added nonlinearity.

## Compliance & Disclaimers
- No diagnosis, treatment, prescriptions, or medical advice.
- Risk module outputs **screening hints only**.
- “Brain oxygenation / blood oxygen metabolism” are **EEG proxy indices** only.

## FAQ
**Q: Missing channels or different ordering?**
A: The pipeline tolerates missing channels; missing features are flagged and confidence is reduced.

**Q: Poor EEG quality?**
A: QC outputs artifact ratios, line noise, EMG ratios, and max amplitude. Reports include QC hints.

**Q: No task paradigm?**
A: Behavior response is labeled as **resting-state proxy assessment**.

## Running Tests
```bash
pytest -q
```
