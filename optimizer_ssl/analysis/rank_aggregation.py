"""Aggregation utilities for spectral-rank scaling analyses."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

POINT_COLUMNS = [
    "run_id",
    "paper_experiment",
    "model_scale",
    "model_name",
    "optimizer",
    "optimizer_folder",
    "optimizer_variant",
    "optimizer_display_name",
    "dion_rank_fraction",
    "width_multiplier",
    "ffn_hidden_dim",
    "bucket",
    "metric",
    "value",
    "err_low",
    "err_high",
    "n_layers",
    "n_layers_used",
    "final_samples",
    "aggregation",
    "frequency_bucket_reduction",
    "seed",
]


def _to_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _median_with_sd(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(arr))
    if arr.size > 1:
        sd = float(np.std(arr, ddof=1))
    else:
        sd = 0.0
    return med, med - sd, med + sd


def aggregate_final_window_points(
    rows: Iterable[dict[str, Any]],
    final_samples: int = 5,
    metrics: tuple[str, ...] = ("soft_rank", "hard_rank"),
    site: str = "post",
) -> list[dict[str, Any]]:
    """Aggregate layer-step rows into one scaling point per run/bucket/metric.

    This mirrors the submitted plotting scripts:

    1. choose the last ``final_samples`` checkpoints for each layer;
    2. take the median over those checkpoints for each layer;
    3. take the median over layers;
    4. use one standard deviation over layer medians as the error band.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("run_id", "")), str(row.get("bucket", "")))].append(row)

    output: list[dict[str, Any]] = []
    for (run_id, bucket), group_rows in sorted(groups.items()):
        if not run_id or not bucket:
            continue
        # Use metadata from the first row in this run/bucket.
        meta = group_rows[0]
        by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            try:
                layer = int(row.get("layer", -1))
            except (TypeError, ValueError):
                continue
            if layer >= 0:
                by_layer[layer].append(row)

        for metric in metrics:
            metric_key = f"{metric}_{site}"
            layer_values: list[float] = []
            for layer, layer_rows in by_layer.items():
                sorted_rows = sorted(layer_rows, key=lambda r: _to_float(r.get("step")))
                vals = [_to_float(r.get(metric_key)) for r in sorted_rows]
                vals = [v for v in vals if np.isfinite(v)]
                if not vals:
                    continue
                layer_values.append(float(np.median(vals[-final_samples:])))

            value, lo, hi = _median_with_sd(layer_values)
            if not np.isfinite(value):
                continue
            output.append(
                {
                    "run_id": run_id,
                    "paper_experiment": meta.get("paper_experiment", ""),
                    "model_scale": meta.get("model_scale", ""),
                    "model_name": meta.get("model_name", ""),
                    "optimizer": meta.get("optimizer", ""),
                    "optimizer_folder": meta.get("optimizer_folder", meta.get("optimizer", "")),
                    "optimizer_variant": meta.get("optimizer_variant", ""),
                    "optimizer_display_name": meta.get("optimizer_display_name", meta.get("optimizer", "")),
                    "dion_rank_fraction": meta.get("dion_rank_fraction", ""),
                    "width_multiplier": meta.get("width_multiplier", ""),
                    "ffn_hidden_dim": meta.get("ffn_hidden_dim", ""),
                    "bucket": bucket,
                    "metric": metric,
                    "value": value,
                    "err_low": lo,
                    "err_high": hi,
                    "n_layers": len(layer_values),
                    "n_layers_used": len(layer_values),
                    "final_samples": final_samples,
                    "aggregation": f"median_final{final_samples}_then_layer_median",
                    "frequency_bucket_reduction": meta.get("frequency_bucket_reduction", ""),
                    "seed": meta.get("seed", ""),
                }
            )
    return output


def split_global_and_frequency_points(points: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    global_points: list[dict[str, Any]] = []
    bucket_points: list[dict[str, Any]] = []
    for point in points:
        if point.get("bucket") == "global":
            global_points.append(point)
        else:
            bucket_points.append(point)
    return global_points, bucket_points
