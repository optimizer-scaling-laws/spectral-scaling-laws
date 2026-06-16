"""Figure generation for matched-loss / extended-AdamW diagnostics.

The matched-loss figure family explains why a simple width-scaling law can
break during extended AdamW training. It consumes processed CSVs generated from
frequency-bucket logs, not raw training logs.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from optimizer_ssl.analysis.figure_specs import BUCKET_DISPLAY_NAMES, BUCKET_ORDER

DEFAULT_DPI = 300
HARD_COLOR = "#1740CC"
SOFT_COLOR = "#E8371C"
WIDTH_COLORMAP = "viridis"


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
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def _k_formatter(x: float, _pos: int) -> str:
    if x == 0:
        return "0"
    if abs(x) >= 1000:
        if x % 1000 == 0:
            return f"{int(x / 1000)}K"
        return f"{x / 1000:.1f}K"
    return f"{int(x)}"


def _width_colors(widths: Iterable[int]) -> dict[int, tuple[float, float, float, float]]:
    widths_sorted = sorted(set(widths))
    cmap = mpl.colormaps[WIDTH_COLORMAP]
    if not widths_sorted:
        return {}
    if len(widths_sorted) == 1:
        return {widths_sorted[0]: cmap(0.5)}
    lo, hi = 0.10, 0.90
    return {
        width: cmap(lo + (hi - lo) * idx / (len(widths_sorted) - 1))
        for idx, width in enumerate(widths_sorted)
    }


def _group_by_bucket(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("bucket", "")).lower()].append(row)
    for group in grouped.values():
        group.sort(key=lambda r: _to_float(r.get("step")))
    return grouped


def _group_pr_by_bucket_width(rows: Iterable[dict[str, str]]) -> dict[tuple[str, int], list[dict[str, str]]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        width = int(float(row.get("width_multiplier", "nan")))
        grouped[(str(row.get("bucket", "")).lower(), width)].append(row)
    for group in grouped.values():
        group.sort(key=lambda r: _to_float(r.get("step")))
    return grouped


def save_matched_loss_beta_dynamics_figure(
    beta_dynamics_csv: str | Path,
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf",),
) -> list[Path]:
    """Create the three-panel beta-dynamics figure from processed CSVs."""
    setup_matplotlib_style()
    rows = _read_csv(beta_dynamics_csv)
    grouped = _group_by_bucket(rows)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    legend_drawn = False

    for idx, (ax, bucket) in enumerate(zip(axes, BUCKET_ORDER)):
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4, zorder=0)
        ax.tick_params(axis="both", which="major", labelsize=16)

        bucket_rows = grouped.get(bucket, [])
        steps = [_to_float(row.get("step")) for row in bucket_rows]
        beta_hard = [_to_float(row.get("beta_hard")) for row in bucket_rows]
        beta_soft = [_to_float(row.get("beta_soft")) for row in bucket_rows]
        if bucket_rows:
            ax.plot(steps, beta_hard, linestyle="-", color=HARD_COLOR, linewidth=2.0, label=r"$\beta_{\mathrm{hard}}$")
            ax.plot(steps, beta_soft, linestyle="-", color=SOFT_COLOR, linewidth=2.0, label=r"$\beta_{\mathrm{soft}}$")

        ax.set_title(BUCKET_DISPLAY_NAMES[bucket], fontsize=18)
        ax.xaxis.set_major_formatter(FuncFormatter(_k_formatter))
        if idx == 1:
            ax.set_xlabel("Training Steps", fontsize=18)
        if idx == 0:
            ax.set_ylabel(r"Spectral scaling exponent ($\beta$)", fontsize=18)
        if bucket_rows and not legend_drawn:
            ax.legend(fontsize=24, framealpha=0.9, loc="best")
            legend_drawn = True

    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for fmt in formats:
        path = output_dir / f"matched_loss_beta_dynamics.{fmt}"
        fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight", format=fmt)
        out_paths.append(path)
    plt.close(fig)
    return out_paths


def save_matched_loss_pr_trajectories_figure(
    pr_trajectories_csv: str | Path,
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf",),
) -> list[Path]:
    """Create the three-panel hard-rank trajectory figure from processed CSVs."""
    setup_matplotlib_style()
    rows = _read_csv(pr_trajectories_csv)
    grouped = _group_pr_by_bucket_width(rows)
    widths = sorted({int(float(row["width_multiplier"])) for row in rows})
    colors = _width_colors(widths)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for idx, (ax, bucket) in enumerate(zip(axes, BUCKET_ORDER)):
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        ax.tick_params(axis="both", which="major", labelsize=16)

        for width in widths:
            group = grouped.get((bucket, width), [])
            if not group:
                continue
            steps = [_to_float(row.get("step")) for row in group]
            hard_rank = [_to_float(row.get("hard_rank_smoothed") or row.get("hard_rank")) for row in group]
            ax.plot(steps, hard_rank, linestyle="-", color=colors[width], linewidth=2.0, label=fr"${width}\times$")

        ax.set_title(BUCKET_DISPLAY_NAMES[bucket], fontsize=18)
        ax.xaxis.set_major_formatter(FuncFormatter(_k_formatter))
        if idx == 1:
            ax.set_xlabel("Training Steps", fontsize=18)
        if idx == 0:
            ax.set_ylabel(r"Hard Spectral Rank ($PR_{\mathrm{post}}$)", fontsize=18)
        if idx == 1:
            ax.legend(fontsize=18, framealpha=0.9, loc="lower right", title="Width", title_fontsize=18, ncol=2)

    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for fmt in formats:
        path = output_dir / f"matched_loss_pr_trajectories_by_width.{fmt}"
        fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight", format=fmt)
        out_paths.append(path)
    plt.close(fig)
    return out_paths


def save_matched_loss_breakdown_figure(
    beta_dynamics_csv: str | Path,
    pr_trajectories_csv: str | Path,
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf",),
) -> list[Path]:
    """Create the combined two-row matched-loss breakdown figure.

    Top row: beta dynamics for HEAD/MID/TAIL.
    Bottom row: hard-rank width trajectories for HEAD/MID/TAIL.
    """
    setup_matplotlib_style()
    beta_rows = _read_csv(beta_dynamics_csv)
    pr_rows = _read_csv(pr_trajectories_csv)
    beta_by_bucket = _group_by_bucket(beta_rows)
    pr_by_bucket_width = _group_pr_by_bucket_width(pr_rows)
    widths = sorted({int(float(row["width_multiplier"])) for row in pr_rows})
    colors = _width_colors(widths)

    fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=False)
    for col, bucket in enumerate(BUCKET_ORDER):
        # Top row: beta dynamics.
        ax = axes[0, col]
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4, zorder=0)
        ax.tick_params(axis="both", which="major", labelsize=14)
        bucket_rows = beta_by_bucket.get(bucket, [])
        steps = [_to_float(row.get("step")) for row in bucket_rows]
        beta_hard = [_to_float(row.get("beta_hard")) for row in bucket_rows]
        beta_soft = [_to_float(row.get("beta_soft")) for row in bucket_rows]
        if bucket_rows:
            ax.plot(steps, beta_hard, linestyle="-", color=HARD_COLOR, linewidth=2.0, label=r"$\beta_{\mathrm{hard}}$")
            ax.plot(steps, beta_soft, linestyle="-", color=SOFT_COLOR, linewidth=2.0, label=r"$\beta_{\mathrm{soft}}$")
        ax.set_title(BUCKET_DISPLAY_NAMES[bucket], fontsize=18)
        ax.xaxis.set_major_formatter(FuncFormatter(_k_formatter))
        if col == 0:
            ax.set_ylabel(r"Spectral scaling exponent ($\beta$)", fontsize=16)
            ax.legend(fontsize=20, framealpha=0.9, loc="best")
        if col == 1:
            ax.set_xlabel("Training Steps", fontsize=16)

        # Bottom row: width trajectories.
        ax2 = axes[1, col]
        ax2.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        ax2.tick_params(axis="both", which="major", labelsize=14)
        for width in widths:
            group = pr_by_bucket_width.get((bucket, width), [])
            if not group:
                continue
            steps2 = [_to_float(row.get("step")) for row in group]
            values = [_to_float(row.get("hard_rank_smoothed") or row.get("hard_rank")) for row in group]
            ax2.plot(steps2, values, linestyle="-", color=colors[width], linewidth=2.0, label=fr"${width}\times$")
        ax2.set_title(BUCKET_DISPLAY_NAMES[bucket], fontsize=18)
        ax2.xaxis.set_major_formatter(FuncFormatter(_k_formatter))
        if col == 0:
            ax2.set_ylabel(r"Hard Spectral Rank ($PR_{\mathrm{post}}$)", fontsize=16)
        if col == 1:
            ax2.set_xlabel("Training Steps", fontsize=16)
            ax2.legend(fontsize=16, framealpha=0.9, loc="lower right", title="Width", title_fontsize=16, ncol=2)

    fig.text(0.012, 0.96, "A. Scaling exponents over training", fontsize=14, weight="bold")
    fig.text(0.012, 0.49, "B. Width-capacity ordering breaks", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.95))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for fmt in formats:
        path = output_dir / f"matched_loss_scaling_breakdown.{fmt}"
        fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight", format=fmt)
        out_paths.append(path)
    plt.close(fig)
    return out_paths


def save_matched_loss_figures(
    beta_dynamics_csv: str | Path,
    pr_trajectories_csv: str | Path,
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("pdf",),
) -> list[Path]:
    """Generate all matched-loss figure artifacts from processed CSVs."""
    paths: list[Path] = []
    paths.extend(save_matched_loss_breakdown_figure(beta_dynamics_csv, pr_trajectories_csv, output_dir, formats=formats))
    paths.extend(save_matched_loss_beta_dynamics_figure(beta_dynamics_csv, output_dir, formats=formats))
    paths.extend(save_matched_loss_pr_trajectories_figure(pr_trajectories_csv, output_dir, formats=formats))
    return paths
