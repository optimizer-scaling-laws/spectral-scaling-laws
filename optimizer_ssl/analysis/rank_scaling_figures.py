"""Figure generation for processed rank-scaling CSVs.

These plotting utilities intentionally consume processed CSVs rather than raw
training logs. Raw logs are first normalized by the analysis parser into the
paper-facing metric vocabulary; figure generation is therefore deterministic and
independent of legacy log formatting.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

from optimizer_ssl.analysis.figure_specs import (
    BUCKET_DISPLAY_NAMES,
    BUCKET_ORDER,
    DION_RANK_SWEEP_ORDER,
    METRIC_DISPLAY_NAMES,
    OPTIMIZER_ORDER,
    TAIL_350M_OPTIMIZER_ORDER,
    dion_rank_sweep_color,
    dion_rank_sweep_linestyle,
    display_optimizer,
    optimizer_color,
    optimizer_marker,
    tail_350m_color,
)

DEFAULT_DPI = 300


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _lighten_color(hex_color: str, amount: float = 0.65) -> tuple[float, float, float]:
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def setup_matplotlib_style() -> None:
    """Apply the paper figure style used by the original plotting scripts."""
    mpl.rc_file_defaults()
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif", "serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def _available_optimizers(rows: Iterable[dict[str, Any]]) -> list[str]:
    present = {str(row.get("optimizer_folder", "")) for row in rows}
    ordered = [opt for opt in OPTIMIZER_ORDER if opt in present]
    extras = sorted(present.difference(ordered).difference({""}))
    return ordered + extras


def _filter_rows(
    rows: Iterable[dict[str, Any]],
    *,
    metric: str,
    bucket: str | None = None,
    model_scale: str | None = None,
    optimizer_folders: set[str] | None = None,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if str(row.get("metric", "")) != metric:
            continue
        if bucket is not None and str(row.get("bucket", "")) != bucket:
            continue
        if model_scale is not None and str(row.get("model_scale", "")) != model_scale:
            continue
        if optimizer_folders is not None and str(row.get("optimizer_folder", "")) not in optimizer_folders:
            continue
        out.append(row)
    return out


def _index_beta_rows(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Index beta rows by (optimizer_folder, metric, bucket)."""
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("optimizer_folder", "")),
            str(row.get("metric", "")),
            str(row.get("bucket", "global") or "global"),
        )
        out[key] = row
    return out


def _group_points_by_optimizer(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("optimizer_folder", ""))].append(row)
    for group in groups.values():
        group.sort(key=lambda r: _to_float(r.get("ffn_hidden_dim")))
    return groups


def _setup_axes(ax: plt.Axes, d_values: list[float], y_values: list[float], *, grid: bool) -> None:
    if not d_values:
        return
    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    base = min(d_values)
    max_d = max(d_values)
    candidate_multipliers = [1, 2, 4, 8] if not grid else [1, 8]
    ticks = []
    for multiplier in candidate_multipliers:
        tick = base * multiplier
        if tick <= max_d * 1.05:
            ticks.append(tick)
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(t)}" for t in ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())

    finite_y = [y for y in y_values if math.isfinite(y) and y > 0]
    if finite_y:
        y_min, y_max = min(finite_y), max(finite_y)
        start = 2 ** math.floor(math.log2(y_min * 0.8))
        ticks_y = []
        val = start
        while val <= y_max * 1.3:
            ticks_y.append(val)
            val *= 2
        ax.set_yticks(ticks_y)
        ax.set_yticklabels([f"{int(t)}" if float(t).is_integer() else f"{t:.1f}" for t in ticks_y])
    ax.yaxis.set_minor_locator(plt.NullLocator())


def _draw_optimizer_points_and_fit(
    ax: plt.Axes,
    points: list[dict[str, Any]],
    beta_index: dict[tuple[str, str, str], dict[str, Any]],
    *,
    metric: str,
    bucket: str,
) -> tuple[list[Line2D], list[float], list[float]]:
    handles: list[Line2D] = []
    all_d: list[float] = []
    all_y: list[float] = []
    grouped = _group_points_by_optimizer(points)

    for opt in _available_optimizers(points):
        group = grouped.get(opt, [])
        if not group:
            continue
        color = optimizer_color(opt)
        light = _lighten_color(color)
        marker = optimizer_marker(opt)
        label_name = display_optimizer(opt, group[0].get("optimizer_display_name") or opt)
        xs = [_to_float(row.get("ffn_hidden_dim")) for row in group]
        ys = [_to_float(row.get("value")) for row in group]
        lows = [_to_float(row.get("err_low")) for row in group]
        highs = [_to_float(row.get("err_high")) for row in group]

        for x, y, lo, hi in zip(xs, ys, lows, highs):
            if not (math.isfinite(x) and math.isfinite(y) and y > 0):
                continue
            err_low = max(y - lo, 0.0) if math.isfinite(lo) else 0.0
            err_high = max(hi - y, 0.0) if math.isfinite(hi) else 0.0
            min_pos = y * 0.9
            err_low = min(err_low, y - min_pos)
            ax.errorbar(
                x,
                y,
                yerr=[[err_low], [err_high]],
                fmt=marker,
                markersize=6,
                capsize=2,
                capthick=1.0,
                elinewidth=1.0,
                color=color,
                markeredgecolor=color,
                ecolor=light,
                zorder=5,
            )
            all_d.append(x)
            all_y.append(y)

        beta_row = beta_index.get((opt, metric, bucket)) or beta_index.get((opt, metric, "global"))
        beta = _to_float(beta_row.get("beta")) if beta_row else float("nan")
        intercept = _to_float(beta_row.get("intercept")) if beta_row else float("nan")
        if math.isfinite(beta) and math.isfinite(intercept) and xs:
            finite_xs = [x for x in xs if math.isfinite(x) and x > 0]
            if finite_xs:
                d_range = np.logspace(np.log10(min(finite_xs) * 0.8), np.log10(max(finite_xs) * 1.2), 100)
                fit_line = np.exp(intercept) * d_range**beta
                ax.plot(d_range, fit_line, "-", linewidth=2.0, color=color, alpha=0.85, zorder=4)
                label = f"{label_name}:  $\\beta={beta:.2f}$"
            else:
                label = label_name
        else:
            label = label_name

        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=6,
                linewidth=2.0,
                linestyle="-",
                label=label,
            )
        )
    return handles, all_d, all_y


def _compute_beta_deltas(beta_rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    by_opt_metric: dict[tuple[str, str], float] = {}
    for row in beta_rows:
        bucket = str(row.get("bucket", "global") or "global")
        if bucket != "global":
            continue
        beta = _to_float(row.get("beta"))
        if math.isfinite(beta):
            by_opt_metric[(str(row.get("optimizer_folder", "")), str(row.get("metric", "")))] = beta
    deltas = {}
    for opt, metric in list(by_opt_metric):
        if metric != "soft_rank":
            continue
        hard = by_opt_metric.get((opt, "hard_rank"))
        soft = by_opt_metric.get((opt, "soft_rank"))
        if hard is not None and soft is not None:
            deltas[opt] = soft - hard
    return deltas


def _render_delta_inset(ax: plt.Axes, beta_rows: Iterable[dict[str, Any]]) -> None:
    deltas = _compute_beta_deltas(beta_rows)
    if not deltas:
        return
    ordered = [(opt, deltas[opt]) for opt in OPTIMIZER_ORDER if opt in deltas]
    ordered += sorted((opt, val) for opt, val in deltas.items() if opt not in OPTIMIZER_ORDER)
    if not ordered:
        return
    max_name = max(len(display_optimizer(opt)) for opt, _ in ordered)
    body = "\n".join(f"{display_optimizer(opt):<{max_name}}   {delta:+.2f}" for opt, delta in ordered)
    header = TextArea(
        r"$\Delta_{1,2} = \beta_{\mathrm{soft}} - \beta_{\mathrm{hard}}$",
        textprops={"fontsize": 14},
    )
    body_area = TextArea(body, textprops={"fontsize": 11, "family": "monospace"})
    packed = VPacker(children=[header, body_area], align="left", pad=0, sep=4)
    anchored = AnchoredOffsetbox(
        loc="upper left",
        child=packed,
        pad=0.45,
        frameon=True,
        bbox_to_anchor=(0.03, 0.97),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    anchored.patch.set_boxstyle("round,pad=0.45")
    anchored.patch.set_edgecolor("gray")
    anchored.patch.set_facecolor("white")
    anchored.patch.set_alpha(0.92)
    anchored.patch.set_linewidth(0.8)
    anchored.set_zorder(10)
    ax.add_artist(anchored)


def save_global_rank_figures(
    points_csv: str | Path,
    beta_csv: str | Path,
    output_dir: str | Path,
    *,
    model_scale: str = "160m",
    formats: tuple[str, ...] = ("pdf",),
    optimizer_folders: Iterable[str] | None = None,
) -> list[Path]:
    """Create global soft/hard rank scaling figures from processed CSVs."""
    setup_matplotlib_style()
    points = _read_csv(points_csv)
    betas = _read_csv(beta_csv)
    if optimizer_folders is not None:
        optimizer_set = set(optimizer_folders)
        points = [row for row in points if str(row.get("optimizer_folder", "")) in optimizer_set]
        betas = [row for row in betas if str(row.get("optimizer_folder", "")) in optimizer_set]
    beta_index = _index_beta_rows([{**row, "bucket": "global"} for row in betas])
    output_paths: list[Path] = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in ("hard_rank", "soft_rank"):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        subset = _filter_rows(points, metric=metric, bucket="global", model_scale=model_scale)
        handles, d_values, y_values = _draw_optimizer_points_and_fit(
            ax, subset, beta_index, metric=metric, bucket="global"
        )
        _setup_axes(ax, d_values, y_values, grid=False)
        ax.set_xlabel("FFN Hidden Dimension ($D$)", fontsize=18)
        ax.set_ylabel(METRIC_DISPLAY_NAMES[metric], fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=16)
        if handles:
            ax.legend(handles=handles, fontsize=15, framealpha=0.9, loc="lower right")
        if metric == "hard_rank":
            _render_delta_inset(ax, betas)
        fig.tight_layout()
        stem = "global_hard_rank_scaling" if metric == "hard_rank" else "global_soft_rank_scaling"
        for fmt in formats:
            path = output_dir / f"{stem}.{fmt}"
            fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight", format=fmt)
            output_paths.append(path)
        plt.close(fig)
    return output_paths


def save_frequency_bucket_grid(
    points_csv: str | Path,
    beta_csv: str | Path,
    output_dir: str | Path,
    *,
    model_scale: str = "160m",
    formats: tuple[str, ...] = ("pdf",),
    optimizer_folders: Iterable[str] | None = None,
) -> list[Path]:
    """Create the 3x2 HEAD/MID/TAIL soft/hard rank scaling grid."""
    setup_matplotlib_style()
    points = _read_csv(points_csv)
    betas = _read_csv(beta_csv)
    if optimizer_folders is not None:
        optimizer_set = set(optimizer_folders)
        points = [row for row in points if str(row.get("optimizer_folder", "")) in optimizer_set]
        betas = [row for row in betas if str(row.get("optimizer_folder", "")) in optimizer_set]
    beta_index = _index_beta_rows(betas)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 8.4),
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )

    for row_idx, metric in enumerate(("soft_rank", "hard_rank")):
        row_points = _filter_rows(points, metric=metric, model_scale=model_scale)
        row_d = [_to_float(row.get("ffn_hidden_dim")) for row in row_points]
        row_y = [_to_float(row.get("value")) for row in row_points]
        for col_idx, bucket in enumerate(BUCKET_ORDER):
            ax = axes[row_idx, col_idx]
            ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
            subset = _filter_rows(points, metric=metric, bucket=bucket, model_scale=model_scale)
            handles, _, _ = _draw_optimizer_points_and_fit(
                ax, subset, beta_index, metric=metric, bucket=bucket
            )
            _setup_axes(ax, row_d, row_y, grid=True)
            ax.tick_params(axis="both", which="major", labelsize=14)
            if row_idx == 0:
                ax.set_title(BUCKET_DISPLAY_NAMES[bucket], fontsize=22, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(METRIC_DISPLAY_NAMES[metric], fontsize=18)
            if handles:
                loc = "upper left" if col_idx == 0 else "lower right"
                ax.legend(
                    handles=handles,
                    fontsize=12,
                    framealpha=0.9,
                    loc=loc,
                    handlelength=1.5,
                    handletextpad=0.4,
                    borderpad=0.35,
                    labelspacing=0.25,
                    borderaxespad=0.4,
                )
    fig.supxlabel("FFN Hidden Dimension ($D$)", fontsize=18)

    output_paths: list[Path] = []
    for fmt in formats:
        path = output_dir / f"frequency_bucket_rank_grid.{fmt}"
        fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight", format=fmt)
        output_paths.append(path)
    plt.close(fig)
    return output_paths



def save_dion_rank_sweep_figures(
    points_csv: str | Path,
    beta_csv: str | Path,
    output_dir: str | Path,
    *,
    model_scale: str = "160m",
    formats: tuple[str, ...] = ("pdf",),
    optimizer_folders: Iterable[str] | None = None,
) -> list[Path]:
    """Create TAIL-token Dion rank-sweep soft/hard rank scaling figures.

    The submitted paper's Dion-rank sweep figure compares AdamW as a dashed
    baseline against Dion rank fractions r=1/2, 1/4, 1/8, and 1/16. The figure
    uses only TAIL-token bucket metrics, with the same final-window/layer-median
    aggregation used by the main rank-scaling figures.
    """
    setup_matplotlib_style()
    points = _read_csv(points_csv)
    betas = _read_csv(beta_csv)

    if optimizer_folders is None:
        optimizer_order = list(DION_RANK_SWEEP_ORDER)
    else:
        optimizer_order = list(optimizer_folders)
        optimizer_set = set(optimizer_order)
        points = [row for row in points if str(row.get("optimizer_folder", "")) in optimizer_set]
        betas = [row for row in betas if str(row.get("optimizer_folder", "")) in optimizer_set]

    beta_index = _index_beta_rows(betas)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for metric in ("hard_rank", "soft_rank"):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        subset = _filter_rows(points, metric=metric, bucket="tail", model_scale=model_scale)
        grouped = _group_points_by_optimizer(subset)
        all_d: list[float] = []
        all_y: list[float] = []
        handles: list[Line2D] = []

        for opt in optimizer_order:
            group = grouped.get(opt, [])
            if not group:
                continue
            group.sort(key=lambda r: _to_float(r.get("ffn_hidden_dim")))
            color = dion_rank_sweep_color(opt)
            light = _lighten_color(color)
            marker = optimizer_marker(opt)
            linestyle = dion_rank_sweep_linestyle(opt)
            label_name = display_optimizer(opt, group[0].get("optimizer_display_name") or opt)

            xs = [_to_float(row.get("ffn_hidden_dim")) for row in group]
            ys = [_to_float(row.get("value")) for row in group]
            lows = [_to_float(row.get("err_low")) for row in group]
            highs = [_to_float(row.get("err_high")) for row in group]

            for x, y, lo, hi in zip(xs, ys, lows, highs):
                if not (math.isfinite(x) and math.isfinite(y) and y > 0):
                    continue
                err_low = max(y - lo, 0.0) if math.isfinite(lo) else 0.0
                err_high = max(hi - y, 0.0) if math.isfinite(hi) else 0.0
                min_pos = y * 0.9
                err_low = min(err_low, y - min_pos)
                ax.errorbar(
                    x,
                    y,
                    yerr=[[err_low], [err_high]],
                    fmt=marker,
                    markersize=6,
                    capsize=2,
                    capthick=1.0,
                    elinewidth=1.0,
                    color=color,
                    markeredgecolor=color,
                    ecolor=light,
                    zorder=5,
                )
                all_d.append(x)
                all_y.append(y)

            beta_row = beta_index.get((opt, metric, "tail"))
            beta = _to_float(beta_row.get("beta")) if beta_row else float("nan")
            intercept = _to_float(beta_row.get("intercept")) if beta_row else float("nan")
            r_squared = _to_float(beta_row.get("r_squared")) if beta_row else float("nan")
            finite_xs = [x for x in xs if math.isfinite(x) and x > 0]
            if math.isfinite(beta) and math.isfinite(intercept) and finite_xs:
                d_range = np.logspace(
                    np.log10(min(finite_xs) * 0.8),
                    np.log10(max(finite_xs) * 1.2),
                    100,
                )
                fit_line = np.exp(intercept) * d_range**beta
                ax.plot(
                    d_range,
                    fit_line,
                    linestyle,
                    linewidth=2.0,
                    color=color,
                    alpha=0.85,
                    zorder=4,
                )
                if math.isfinite(r_squared):
                    label = f"{label_name}:  $\\beta={beta:.2f}$  ($R^2={r_squared:.2f}$)"
                else:
                    label = f"{label_name}:  $\\beta={beta:.2f}$"
            else:
                label = label_name

            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=6,
                    linewidth=2.0,
                    linestyle=linestyle,
                    label=label,
                )
            )

        _setup_axes(ax, all_d, all_y, grid=False)
        ax.set_xlabel("FFN Hidden Dimension ($D$)", fontsize=18)
        ylabel = f"{METRIC_DISPLAY_NAMES[metric]} (TAIL tokens)"
        ax.set_ylabel(ylabel, fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=16)
        if handles:
            loc = "lower right" if metric == "hard_rank" else "upper left"
            ax.legend(handles=handles, fontsize=12, framealpha=0.9, loc=loc)
        fig.tight_layout()

        stem = "dion_tail_hard_rank_sweep" if metric == "hard_rank" else "dion_tail_soft_rank_sweep"
        for fmt in formats:
            path = output_dir / f"{stem}.{fmt}"
            fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight", format=fmt)
            output_paths.append(path)
        plt.close(fig)

    return output_paths



def _setup_tail_350m_axes(ax: plt.Axes, d_values: list[float], y_values: list[float]) -> None:
    """Axis setup matching the submitted GPT2-350M TAIL plotting script."""
    if not d_values:
        return
    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    ticks = sorted({int(x) for x in d_values if math.isfinite(x) and x > 0})
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())

    finite_y = [y for y in y_values if math.isfinite(y) and y > 0]
    if finite_y:
        y_min, y_max = min(finite_y), max(finite_y)
        start = 2 ** math.floor(math.log2(y_min * 0.8))
        ticks_y: list[float] = []
        val = start
        while val <= y_max * 1.3:
            ticks_y.append(val)
            val *= 2
        ax.set_yticks(ticks_y)
        ax.set_yticklabels([f"{int(t)}" if float(t).is_integer() else f"{t:.1f}" for t in ticks_y])
    ax.yaxis.set_minor_locator(plt.NullLocator())


def save_tail_350m_rank_figures(
    points_csv: str | Path,
    beta_csv: str | Path,
    output_dir: str | Path,
    *,
    model_scale: str = "350m",
    formats: tuple[str, ...] = ("pdf",),
    optimizer_folders: Iterable[str] | None = None,
) -> list[Path]:
    """Create GPT2-350M TAIL-token hard/soft rank scaling figures.

    The submitted paper's GPT2-350M confirmation figure uses only TAIL-token
    frequency-bucket metrics, four width points (1x--4x), and four optimizers:
    AdamW, Muon, NorMuon, and Dion r=1/16. The aggregation and beta fitting are
    the same final-window/layer-median pipeline used by the 160M figures.
    """
    setup_matplotlib_style()
    points = _read_csv(points_csv)
    betas = _read_csv(beta_csv)

    if optimizer_folders is None:
        optimizer_order = list(TAIL_350M_OPTIMIZER_ORDER)
    else:
        optimizer_order = list(optimizer_folders)
        optimizer_set = set(optimizer_order)
        points = [row for row in points if str(row.get("optimizer_folder", "")) in optimizer_set]
        betas = [row for row in betas if str(row.get("optimizer_folder", "")) in optimizer_set]

    beta_index = _index_beta_rows(betas)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for metric in ("hard_rank", "soft_rank"):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        subset = _filter_rows(points, metric=metric, bucket="tail", model_scale=model_scale)
        grouped = _group_points_by_optimizer(subset)
        all_d: list[float] = []
        all_y: list[float] = []
        handles: list[Line2D] = []

        for opt in optimizer_order:
            group = grouped.get(opt, [])
            if not group:
                continue
            group.sort(key=lambda r: _to_float(r.get("ffn_hidden_dim")))
            color = tail_350m_color(opt)
            light = _lighten_color(color)
            marker = optimizer_marker(opt)
            label_name = display_optimizer(opt, group[0].get("optimizer_display_name") or opt)

            xs = [_to_float(row.get("ffn_hidden_dim")) for row in group]
            ys = [_to_float(row.get("value")) for row in group]
            lows = [_to_float(row.get("err_low")) for row in group]
            highs = [_to_float(row.get("err_high")) for row in group]

            for x, y, lo, hi in zip(xs, ys, lows, highs):
                if not (math.isfinite(x) and math.isfinite(y) and y > 0):
                    continue
                err_low = max(y - lo, 0.0) if math.isfinite(lo) else 0.0
                err_high = max(hi - y, 0.0) if math.isfinite(hi) else 0.0
                min_pos = y * 0.9
                err_low = min(err_low, y - min_pos)
                ax.errorbar(
                    x,
                    y,
                    yerr=[[err_low], [err_high]],
                    fmt=marker,
                    markersize=6,
                    capsize=2,
                    capthick=1.0,
                    elinewidth=1.0,
                    color=color,
                    markeredgecolor=color,
                    ecolor=light,
                    zorder=5,
                )
                all_d.append(x)
                all_y.append(y)

            beta_row = beta_index.get((opt, metric, "tail"))
            beta = _to_float(beta_row.get("beta")) if beta_row else float("nan")
            intercept = _to_float(beta_row.get("intercept")) if beta_row else float("nan")
            r_squared = _to_float(beta_row.get("r_squared")) if beta_row else float("nan")
            finite_xs = [x for x in xs if math.isfinite(x) and x > 0]
            if math.isfinite(beta) and math.isfinite(intercept) and finite_xs:
                d_range = np.logspace(
                    np.log10(min(finite_xs) * 0.8),
                    np.log10(max(finite_xs) * 1.2),
                    100,
                )
                fit_line = np.exp(intercept) * d_range**beta
                ax.plot(d_range, fit_line, "-", linewidth=2.0, color=color, alpha=0.85, zorder=4)
                if math.isfinite(r_squared):
                    label = f"{label_name}:  $\\beta={beta:.2f}$  ($R^2={r_squared:.2f}$)"
                else:
                    label = f"{label_name}:  $\\beta={beta:.2f}$"
            else:
                label = label_name

            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=6,
                    linewidth=2.0,
                    linestyle="-",
                    label=label,
                )
            )

        _setup_tail_350m_axes(ax, all_d, all_y)
        ax.set_xlabel("FFN Hidden Dimension ($D$)", fontsize=18)
        ax.set_ylabel(METRIC_DISPLAY_NAMES[metric], fontsize=18)
        ax.tick_params(axis="both", which="major", labelsize=16)
        if handles:
            ax.legend(handles=handles, fontsize=12, framealpha=0.9, loc="lower right")
        fig.tight_layout()

        stem = "tail_350m_hard_rank_scaling" if metric == "hard_rank" else "tail_350m_soft_rank_scaling"
        for fmt in formats:
            path = output_dir / f"{stem}.{fmt}"
            fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight", format=fmt)
            output_paths.append(path)
        plt.close(fig)

    return output_paths
