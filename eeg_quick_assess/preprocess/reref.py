import numpy as np


def rereference(data, meta, cfg):
    mode = meta.get("reference") or cfg["preprocess"]["reference"]
    if mode == "none":
        return data
    if mode == "average":
        return data - data.mean(axis=0, keepdims=True)
    if mode == "linked_ears":
        ch_names = meta.get("ch_names", [])
        if "A1" in ch_names and "A2" in ch_names:
            idx1, idx2 = ch_names.index("A1"), ch_names.index("A2")
            ref = (data[idx1] + data[idx2]) / 2
            return data - ref
    return data
