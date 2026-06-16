"""Power-law fitting utilities for processed rank-scaling points."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

BETA_COLUMNS = [
    "paper_experiment",
    "model_scale",
    "bucket",
    "metric",
    "optimizer",
    "optimizer_folder",
    "optimizer_variant",
    "optimizer_display_name",
    "dion_rank_fraction",
    "beta",
    "intercept",
    "r_squared",
    "beta_lower",
    "beta_upper",
    "n_widths",
    "ci_method",
]

_T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    25: 2.060,
    30: 2.042,
}


def _to_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _t_critical_975(dof: int) -> float:
    if dof <= 0:
        return float("inf")
    if dof in _T_CRITICAL_975:
        return _T_CRITICAL_975[dof]
    if dof < 30:
        larger = min(k for k in _T_CRITICAL_975 if k > dof)
        return _T_CRITICAL_975[larger]
    return 1.96


def fit_power_law_with_ci(
    d_values: Iterable[Any],
    metric_values: Iterable[Any],
    min_points: int = 3,
) -> dict[str, Any]:
    """Fit ``metric = A * D^beta`` in log-log space.

    The confidence interval follows the same ordinary least-squares/t-interval
    convention used in the original plotting scripts, but avoids requiring scipy
    for the lightweight analysis tests.
    """
    x = np.asarray([_to_float(v) for v in d_values], dtype=float)
    y = np.asarray([_to_float(v) for v in metric_values], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if int(mask.sum()) < min_points:
        return {"valid": False}

    lx = np.log(x[mask])
    ly = np.log(y[mask])
    n = int(mask.sum())
    slope, intercept = np.polyfit(lx, ly, deg=1)
    pred = slope * lx + intercept
    residuals = ly - pred
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((ly - float(np.mean(ly))) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    dof = n - 2
    if dof > 0:
        s_err = float(np.sqrt(ss_res / dof))
        sxx = float(np.sum((lx - float(np.mean(lx))) ** 2))
        std_err = s_err / np.sqrt(sxx) if sxx > 0 else float("inf")
        ci_half = _t_critical_975(dof) * std_err
    else:
        ci_half = float("inf")

    return {
        "valid": True,
        "beta": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "beta_lower": float(slope - ci_half),
        "beta_upper": float(slope + ci_half),
        "n_widths": n,
        "ci_method": "ols_loglog_t_interval",
    }


def fit_rank_scaling_points(points: Iterable[dict[str, Any]], min_points: int = 3) -> list[dict[str, Any]]:
    """Fit beta for each model/bucket/metric/optimizer group."""
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        key = (
            str(point.get("model_scale", "")),
            str(point.get("bucket", "")),
            str(point.get("metric", "")),
            str(point.get("optimizer_folder", point.get("optimizer", ""))),
            str(point.get("optimizer_variant", "")),
        )
        groups[key].append(point)

    fits: list[dict[str, Any]] = []
    for key, group_points in sorted(groups.items()):
        model_scale, bucket, metric, optimizer_folder, optimizer_variant = key
        d_values = [p.get("ffn_hidden_dim") for p in group_points]
        values = [p.get("value") for p in group_points]
        fit = fit_power_law_with_ci(d_values, values, min_points=min_points)
        if not fit.get("valid"):
            continue
        fits.append(
            {
                "paper_experiment": group_points[0].get("paper_experiment", ""),
                "model_scale": model_scale,
                "bucket": bucket,
                "metric": metric,
                "optimizer": group_points[0].get("optimizer", optimizer_folder),
                "optimizer_folder": optimizer_folder,
                "optimizer_variant": optimizer_variant,
                "optimizer_display_name": group_points[0].get("optimizer_display_name", group_points[0].get("optimizer", optimizer_folder)),
                "dion_rank_fraction": group_points[0].get("dion_rank_fraction", ""),
                "beta": fit["beta"],
                "intercept": fit["intercept"],
                "r_squared": fit["r_squared"],
                "beta_lower": fit["beta_lower"],
                "beta_upper": fit["beta_upper"],
                "n_widths": fit["n_widths"],
                "ci_method": fit["ci_method"],
            }
        )
    return fits
