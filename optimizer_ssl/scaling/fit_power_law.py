import numpy as np


def fit_power_law(x, y):
    """Fit y = c * x^beta using ordinary least squares in log-log space."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        raise ValueError("Need at least two positive points for a power-law fit")
    lx, ly = np.log(x[mask]), np.log(y[mask])
    beta, log_c = np.polyfit(lx, ly, deg=1)
    pred = beta * lx + log_c
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"beta": float(beta), "intercept": float(log_c), "r2": r2}
