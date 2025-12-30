import yaml

from eeg_quick_assess.scoring.score_engine import score_modules


def test_level_thresholds():
    scoring_cfg = yaml.safe_load(open("eeg_quick_assess/configs/scoring_rules.yaml", "r", encoding="utf-8"))
    sentence_cfg = yaml.safe_load(open("eeg_quick_assess/configs/sentence_bank.yaml", "r", encoding="utf-8"))

    features = {
        "bandpower": {"relative": {"alpha": 0.3, "beta": 0.3, "theta": 0.3, "delta": 0.1}, "absolute": {}},
        "ratios": {"theta_beta": 1.0, "beta_alpha": 1.0, "delta_theta_alpha": 1.0},
        "spectral_entropy": {"spectral_entropy": 0.8},
        "permutation_entropy": {"permutation_entropy": 0.8},
        "stability": {
            "bandpower_cv": {"alpha": 0.1, "beta": 0.1},
            "ratio_cv": {"theta_beta": 0.1, "beta_alpha": 0.1},
            "bandpower_drift": {"alpha": 0.0},
            "ratio_drift": {"theta_beta": 0.0},
        },
        "laterality": {"faa": 0.0, "laterality_pairs": {}},
    }
    qc = {"artifact_window_ratio": 0.0, "line_noise_ratio": 0.0}

    modules = score_modules(features, qc, scoring_cfg, sentence_cfg)
    assert modules["psych"]["dimensions"]["安全感"]["level"] in {"低", "中", "高"}
