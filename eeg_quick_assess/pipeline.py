import json
from datetime import datetime
from pathlib import Path

import yaml

from eeg_quick_assess.preprocess.filters import apply_filters
from eeg_quick_assess.preprocess.reref import rereference
from eeg_quick_assess.preprocess.segment import segment_windows
from eeg_quick_assess.qc.qc_metrics import compute_qc
from eeg_quick_assess.features.bandpower import compute_bandpower
from eeg_quick_assess.features.ratios import compute_ratios
from eeg_quick_assess.features.psd import compute_psd
from eeg_quick_assess.features.entropy import compute_spectral_entropy, compute_permutation_entropy
from eeg_quick_assess.features.stability import compute_stability
from eeg_quick_assess.features.laterality import compute_laterality
from eeg_quick_assess.scoring.score_engine import score_modules
from eeg_quick_assess.report.templates import render_module_texts
from eeg_quick_assess.report.summarizer import summarize_reports


CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_assessment(data, meta, use_llm: bool = True) -> dict:
    runtime_cfg = load_config("runtime.yaml")
    scoring_cfg = load_config("scoring_rules.yaml")
    sentence_cfg = load_config("sentence_bank.yaml")

    filtered = apply_filters(data, meta, runtime_cfg)
    reref = rereference(filtered, meta, runtime_cfg)
    windows = segment_windows(reref, meta, runtime_cfg)

    qc = compute_qc(reref, windows, meta, runtime_cfg)

    psd = compute_psd(reref, meta, runtime_cfg)
    bandpower = compute_bandpower(psd, meta, runtime_cfg)
    ratios = compute_ratios(bandpower)
    spectral_entropy = compute_spectral_entropy(psd)
    perm_entropy = compute_permutation_entropy(windows)
    stability = compute_stability(windows, meta, runtime_cfg)
    laterality = compute_laterality(reref, meta, runtime_cfg)

    features = {
        "bandpower": bandpower,
        "ratios": ratios,
        "spectral_entropy": spectral_entropy,
        "permutation_entropy": perm_entropy,
        "stability": stability,
        "laterality": laterality,
    }

    modules = score_modules(features, qc, scoring_cfg, sentence_cfg)
    module_texts = render_module_texts(modules, qc)
    for key, text in module_texts.items():
        modules[key]["module_text_rule"] = text

    final = summarize_reports(modules, sentence_cfg, use_llm=use_llm)

    return {
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "meta": meta,
        "qc": qc,
        "features": json.loads(json.dumps(features)),
        "modules": modules,
        "final": final,
    }
