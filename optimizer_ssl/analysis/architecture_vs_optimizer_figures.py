"""Architecture-vs-optimizer comparison figure from processed beta values."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np


BUCKET_ORDER = ("head", "mid", "tail")
BUCKET_DISPLAY = {"head": "HEAD", "mid": "MID", "tail": "TAIL"}
METRIC_ORDER = ("soft_rank", "hard_rank")
METRIC_SHORT = {"soft_rank": "soft", "hard_rank": "hard"}
OPTIMIZER_ORDER = ("AdamW", "Muon", "NorMuon", "Dion (1/2)", "Dion (1/16)")

COLOR_BAR = "#D5A8E6"
COLOR_EDGE = "#5A0E72"
COLOR_LINE = "#C03221"
DEFAULT_DPI = 300


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def setup_matplotlib_style() -> None:
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
        }
    )


def _comparison_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["bucket"], row["metric"], row["optimizer_display_name"])
        out[key] = row
    return out


def save_architecture_vs_optimizer_figure(
    comparison_csv: str | Path,
    out_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf",),
) -> list[Path]:
    """Save the 2x3 architecture-vs-optimizer comparison figure.

    The input CSV should contain one row per (bucket, metric, optimizer), with
    `architecture_shift_abs` as the bar height and `optimizer_gain_star` as the
    dashed horizontal reference line.
    """
    rows = _read_csv(comparison_csv)
    index = _comparison_index(rows)

    setup_matplotlib_style()

    # Preserve the original paper-run plot's row order: soft on top, hard below.
    row_values: dict[str, list[float]] = {metric: [] for metric in METRIC_ORDER}
    for metric in METRIC_ORDER:
        for bucket in BUCKET_ORDER:
            for optimizer in OPTIMIZER_ORDER:
                row = index[(bucket, metric, optimizer)]
                row_values[metric].append(_to_float(row["architecture_shift_abs"]))
            row_values[metric].append(_to_float(index[(bucket, metric, OPTIMIZER_ORDER[0])]["optimizer_gain_star"]))

    y_max_by_metric: dict[str, float] = {}
    for metric in METRIC_ORDER:
        if metric == "soft_rank":
            y_max_by_metric[metric] = 0.70
        else:
            y_max_by_metric[metric] = max(row_values[metric]) * 1.22

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(11, 5.4),
        sharex=True,
        sharey="row",
        gridspec_kw={"height_ratios": [y_max_by_metric[m] for m in METRIC_ORDER]},
    )

    x = np.arange(len(OPTIMIZER_ORDER))

    for ri, metric in enumerate(METRIC_ORDER):
        y_max = y_max_by_metric[metric]
        for ci, bucket in enumerate(BUCKET_ORDER):
            ax = axes[ri, ci]
            panel_rows = [index[(bucket, metric, opt)] for opt in OPTIMIZER_ORDER]
            arch_values = [_to_float(row["architecture_shift_abs"]) for row in panel_rows]
            opt_star = _to_float(panel_rows[0]["optimizer_gain_star"])
            all_below = all(row["all_architecture_shifts_below_optimizer_gain"].lower() == "true" for row in panel_rows)

            ax.bar(
                x,
                arch_values,
                width=0.7,
                color=COLOR_BAR,
                edgecolor=COLOR_EDGE,
                linewidth=0.8,
                zorder=3,
            )

            for xi, value in zip(x, arch_values):
                if value >= 0.03:
                    ax.text(
                        xi,
                        value + y_max * 0.012,
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        color="black",
                        zorder=5,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                    )

            ax.axhline(
                opt_star,
                color=COLOR_LINE,
                linestyle="--",
                linewidth=1.9,
                alpha=0.95,
                zorder=4,
                dashes=(5, 3),
            )

            prefix = r"$\bigstar\ $" if all_below else ""
            line_label = prefix + rf"$\Delta\beta^{{\star}}_{{\mathrm{{opt}}}}={opt_star:.2f}$"
            label_y = min(opt_star / y_max + 0.022, 0.93)
            ax.text(
                0.975,
                label_y,
                line_label,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=13,
                color=COLOR_LINE,
                fontweight="bold" if all_below else "normal",
                path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
                zorder=6,
            )

            if ri == 0:
                ax.set_title(BUCKET_DISPLAY[bucket], fontsize=16, pad=8, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(OPTIMIZER_ORDER, fontsize=13, rotation=30, ha="right", rotation_mode="anchor")
            ax.set_xlim(-0.6, len(OPTIMIZER_ORDER) - 0.4)
            ax.set_ylim(0, y_max)
            ax.tick_params(axis="y", labelsize=13)

            if ci == 0:
                metric_short = METRIC_SHORT[metric]
                ax.set_ylabel(rf"$|\Delta\beta_{{\mathrm{{{metric_short}}}}}|$", fontsize=22)

    plt.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        path = out_dir / f"architecture_vs_optimizer.{fmt}"
        fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight", format=fmt)
        paths.append(path)
    plt.close(fig)
    return paths
