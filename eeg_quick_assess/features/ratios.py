
def compute_ratios(bandpower):
    rel = bandpower["relative"]
    theta = rel.get("theta", 0.0)
    beta = rel.get("beta", 0.0)
    delta = rel.get("delta", 0.0)
    alpha = rel.get("alpha", 0.0)

    return {
        "theta_beta": theta / (beta + 1e-12),
        "delta_theta_alpha": (theta + delta) / (alpha + 1e-12),
        "beta_alpha": beta / (alpha + 1e-12),
    }
