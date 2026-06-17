"""Backward-compatible power-law fit helper.

Paper-facing scaling fits live in :mod:`optimizer_ssl.analysis.scaling_fits`.
This module keeps the small historical ``fit_power_law`` API while delegating
all numerical work to the shared log-log/t-interval implementation.
"""

from __future__ import annotations

from typing import Any, Iterable

from optimizer_ssl.analysis.scaling_fits import fit_power_law_with_ci


def fit_power_law(x: Iterable[Any], y: Iterable[Any]) -> dict[str, float]:
    """Fit ``y = c * x^beta`` in log-log space.

    This compatibility wrapper returns the compact keys used by older tests and
    examples: ``beta``, ``intercept``, and ``r2``. Use
    ``optimizer_ssl.analysis.scaling_fits.fit_power_law_with_ci`` for the full
    paper-facing fit result with confidence intervals.
    """
    fit = fit_power_law_with_ci(x, y, min_points=2)
    if not fit.get("valid"):
        raise ValueError("Need at least two positive points for a power-law fit")
    return {
        "beta": float(fit["beta"]),
        "intercept": float(fit["intercept"]),
        "r2": float(fit["r_squared"]),
    }
