
def render_module_texts(modules, qc):
    texts = {}
    for key, module in modules.items():
        lines = [f"模块：{module['module_name']}", f"QC提示：{_qc_hint(qc)}"]
        for dim_key, dim in module["dimensions"].items():
            lines.append(f"- {dim_key}: 分数 {dim['score']:.1f}，等级 {dim['level']}。{dim['fixed_text']}")
        texts[key] = "\n".join(lines)
    return texts


def _qc_hint(qc):
    if qc["artifact_window_ratio"] > 0.3:
        return "伪迹窗占比较高，请谨慎解读"
    if qc["line_noise_ratio"] > 0.2:
        return "工频污染较高，建议复核"
    return "数据质量可接受"
