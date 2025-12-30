import json
import re

from eeg_quick_assess.report.llm_backend import run_llama_cpp, run_transformers
from eeg_quick_assess.report.llm_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def summarize_reports(modules, sentence_cfg, use_llm: bool = True):
    payload = {
        "modules": modules,
    }
    prompt = SYSTEM_PROMPT + "\n" + USER_PROMPT_TEMPLATE.format(payload=json.dumps(payload, ensure_ascii=False))

    if use_llm:
        for backend in _preferred_backends(sentence_cfg):
            try:
                response = backend(prompt)
                parsed = _validate_llm_output(response.text, payload)
                if parsed:
                    return {
                        "final_report_text_llm": parsed["summary_report"],
                        "llm_used": True,
                        "disclaimer": parsed.get("disclaimer", sentence_cfg["disclaimer"]),
                        "module_reports": parsed["module_reports"],
                    }
            except RuntimeError:
                continue

    fallback = _fallback_text(modules, sentence_cfg)
    return {
        "fallback_text": fallback,
        "llm_used": False,
        "disclaimer": sentence_cfg["disclaimer"],
    }


def _preferred_backends(sentence_cfg):
    backends = []
    llm_cfg = sentence_cfg.get("llm", {})
    if llm_cfg.get("transformers_model"):
        backends.append(lambda prompt: run_transformers(prompt, llm_cfg["transformers_model"]))
    if llm_cfg.get("llama_cpp_model"):
        backends.append(lambda prompt: run_llama_cpp(prompt, llm_cfg["llama_cpp_model"]))
    return backends


def _fallback_text(modules, sentence_cfg):
    lines = ["EEG 评估汇总（规则拼接）"]
    for key in ["psych", "physio", "cognitive", "risk"]:
        module = modules[key]
        lines.append(f"[{module['module_name']}]")
        for dim_key, dim in module["dimensions"].items():
            lines.append(f"- {dim_key}: {dim['fixed_text']}")
    lines.append(sentence_cfg["disclaimer"])
    return "\n".join(lines)


def _validate_llm_output(text, payload):
    try:
        json_text = _extract_json(text)
        output = json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        return None

    required_keys = {"summary_report", "module_reports", "disclaimer"}
    if set(output.keys()) != required_keys:
        return None
    if set(output["module_reports"].keys()) != {"psych", "physio", "cognitive", "risk"}:
        return None

    allowed_numbers = _collect_scores(payload)
    output_numbers = _extract_numbers(json_text)
    if not output_numbers.issubset(allowed_numbers):
        return None

    return output


def _extract_json(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("No JSON found")
    return match.group(0)


def _collect_scores(payload):
    scores = set()
    for module in payload["modules"].values():
        for dim in module["dimensions"].values():
            scores.add(round(float(dim["score"]), 1))
            scores.add(round(float(dim["score"]), 2))
    return scores


def _extract_numbers(text):
    numbers = set()
    for num in re.findall(r"\d+\.\d+|\d+", text):
        numbers.add(round(float(num), 2))
    return numbers
