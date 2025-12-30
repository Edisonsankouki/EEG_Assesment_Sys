import yaml

from eeg_quick_assess.report.summarizer import summarize_reports


def test_llm_fallback():
    sentence_cfg = yaml.safe_load(open("eeg_quick_assess/configs/sentence_bank.yaml", "r", encoding="utf-8"))
    modules = {
        "psych": {"module_name": "心理健康模块", "dimensions": {"安全感": {"score": 50.0, "level": "中", "evidence": [], "fixed_text": "A"}}},
        "physio": {"module_name": "脑生理监测模块", "dimensions": {"脑耗氧": {"score": 50.0, "level": "中", "evidence": [], "fixed_text": "B"}}},
        "cognitive": {"module_name": "认知功能评估模块", "dimensions": {"脑稳定度": {"score": 50.0, "level": "中", "evidence": [], "fixed_text": "C"}}},
        "risk": {"module_name": "风险预测/评估模块", "dimensions": {"焦虑风险": {"score": 50.0, "level": "中", "evidence": [], "fixed_text": "D"}}},
    }
    out = summarize_reports(modules, sentence_cfg, use_llm=False)
    assert out["llm_used"] is False
    assert "fallback_text" in out
