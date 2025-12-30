import numpy as np


def compute_spectral_entropy(psd_dict):
    psd = np.array(psd_dict["psd"])
    psd = psd / (np.sum(psd, axis=1, keepdims=True) + 1e-12)
    entropy = -np.sum(psd * np.log(psd + 1e-12), axis=1)
    entropy_norm = entropy / np.log(psd.shape[1] + 1e-12)
    return {
        "spectral_entropy": float(np.mean(entropy_norm)),
        "config": {"normalized": True},
    }


def compute_permutation_entropy(windows, order: int = 3, delay: int = 1):
    if windows.size == 0:
        return {"permutation_entropy": None, "config": {"order": order, "delay": delay}}
    x = windows.reshape(-1)
    n = len(x) - delay * (order - 1)
    if n <= 0:
        return {"permutation_entropy": None, "config": {"order": order, "delay": delay}}
    patterns = {}
    for i in range(n):
        window = x[i : i + delay * order : delay]
        key = tuple(np.argsort(window))
        patterns[key] = patterns.get(key, 0) + 1
    counts = np.array(list(patterns.values()), dtype=float)
    probs = counts / counts.sum()
    pe = -np.sum(probs * np.log2(probs + 1e-12))
    pe_norm = pe / np.log2(np.math.factorial(order))
    return {"permutation_entropy": float(pe_norm), "config": {"order": order, "delay": delay}}
