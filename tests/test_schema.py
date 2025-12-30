from eeg_quick_assess.scoring.schema import DimensionResult, ModuleResult


def test_schema_models():
    dim = DimensionResult(
        score=55.0,
        level="中",
        evidence=["bandpower.relative.alpha"],
        fixed_text="测试文本",
        missing_features=None,
        confidence=0.9,
    )
    module = ModuleResult(module_name="心理健康模块", dimensions={"安全感": dim}, module_text_rule="")
    assert module.dimensions["安全感"].score == 55.0
