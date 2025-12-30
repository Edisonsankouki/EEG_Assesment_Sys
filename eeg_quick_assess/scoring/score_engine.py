from eeg_quick_assess.scoring.normalize import min_max_norm


def _get_feature(features, path):
    ref = features
    for part in path.split("."):
        if isinstance(ref, dict) and part in ref:
            ref = ref[part]
        else:
            return None
    return ref


def _dimension_score(dimension_cfg, features):
    weights = dimension_cfg["weights"]
    norms = dimension_cfg["norms"]
    scores = []
    missing = []
    for feat_name, weight in weights.items():
        value = _get_feature(features, feat_name)
        if value is None:
            missing.append(feat_name)
            continue
        norm_cfg = norms[feat_name]
        norm = min_max_norm(value, norm_cfg["min"], norm_cfg["max"])
        scores.append(weight * norm)
    if not scores:
        return 0.0, missing, 0.0
    score = 100 * sum(scores)
    confidence = max(0.1, 1.0 - 0.1 * len(missing))
    return score, missing, confidence


def _level_from_score(score, thresholds):
    if score < thresholds["low"]:
        return "低"
    if score < thresholds["high"]:
        return "中"
    return "高"


def score_modules(features, qc, scoring_cfg, sentence_cfg):
    modules_out = {}
    for module_key, module_cfg in scoring_cfg["modules"].items():
        module_out = {"module_name": module_cfg["module_name"], "dimensions": {}}
        for dim_key, dim_cfg in module_cfg["dimensions"].items():
            score, missing, confidence = _dimension_score(dim_cfg, features)
            thresholds = scoring_cfg["levels"]
            level = _level_from_score(score, thresholds)
            evidence = list(dim_cfg["weights"].keys())

            sentence = sentence_cfg["modules"][module_key][dim_key][level]
            fixed_text = sentence.format(score=f"{score:.1f}", qc_hint=_qc_hint(qc), direction=_laterality_hint(features))

            dim_out = {
                "score": round(score, 2),
                "level": level,
                "evidence": evidence,
                "fixed_text": fixed_text,
                "missing_features": missing if missing else None,
                "confidence": round(confidence, 2),
            }
            module_out["dimensions"][dim_key] = dim_out
        modules_out[module_key] = module_out
    return modules_out


def _qc_hint(qc):
    if qc["artifact_window_ratio"] > 0.3:
        return "数据伪迹较多，分数需谨慎解读"
    if qc["line_noise_ratio"] > 0.2:
        return "工频污染偏高，建议复核"
    return "数据质量可接受"


def _laterality_hint(features):
    lat = features.get("laterality", {}).get("laterality_pairs", {})
    if not lat:
        return "左右偏侧信息不足"
    directions = [v["direction"] for v in lat.values()]
    if directions.count("right") > directions.count("left"):
        return "右侧偏强"
    if directions.count("left") > directions.count("right"):
        return "左侧偏强"
    return "左右均衡"
