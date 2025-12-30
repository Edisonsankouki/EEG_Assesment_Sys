import numpy as np


def min_max_norm(value, lower, upper):
    if value is None:
        return None
    if upper == lower:
        return 0.0
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))
