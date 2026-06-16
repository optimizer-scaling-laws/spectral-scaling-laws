"""Figure styling and display metadata for rank-scaling plots."""

from __future__ import annotations

# Public processed artifacts and launch configs use one canonical Dion naming
# scheme: dion_r1_2, dion_r1_4, dion_r1_8, dion_r1_16. Legacy submitted-log
# folder aliases are intentionally not part of the public figure orders.
OPTIMIZER_DISPLAY_NAMES: dict[str, str] = {
    "adamw": "AdamW",
    "muon": "Muon",
    "normuon": "NorMuon",
    "dion_r1_2": "Dion (1/2)",
    "dion_r1_4": "Dion (1/4)",
    "dion_r1_8": "Dion (1/8)",
    "dion_r1_16": "Dion (1/16)",
}

OPTIMIZER_ORDER: tuple[str, ...] = (
    "adamw",
    "dion_r1_2",
    "dion_r1_16",
    "muon",
    "normuon",
)

DION_RANK_SWEEP_ORDER: tuple[str, ...] = (
    "adamw",
    "dion_r1_2",
    "dion_r1_4",
    "dion_r1_8",
    "dion_r1_16",
)

OPTIMIZER_COLORS: dict[str, str] = {
    "adamw": "#2ca02c",
    "muon": "#d62728",
    "normuon": "#1f77b4",
    "dion_r1_2": "#5A0E72",
    "dion_r1_4": "#8B1DAF",
    "dion_r1_8": "#B465D1",
    "dion_r1_16": "#D5A8E6",
}

# The Dion rank-sweep paper figure uses AdamW as a dashed black baseline,
# rather than the green AdamW color used in the main optimizer comparison plots.
DION_RANK_SWEEP_COLORS: dict[str, str] = {
    "adamw": "#000000",
    "dion_r1_2": "#5A0E72",
    "dion_r1_4": "#8B1DAF",
    "dion_r1_8": "#B465D1",
    "dion_r1_16": "#D5A8E6",
}

# The GPT2-350M TAIL-only paper figure uses a compact four-optimizer
# palette from the submitted plotting script.
TAIL_350M_OPTIMIZER_ORDER: tuple[str, ...] = (
    "adamw",
    "muon",
    "normuon",
    "dion_r1_16",
)

TAIL_350M_COLORS: dict[str, str] = {
    "adamw": "#1740CC",
    "muon": "#8B1DAF",
    "normuon": "#E8371C",
    "dion_r1_16": "#F58C00",
}

OPTIMIZER_MARKERS: dict[str, str] = {
    "adamw": "o",
    "muon": "o",
    "normuon": "o",
    "dion_r1_2": "o",
    "dion_r1_4": "o",
    "dion_r1_8": "o",
    "dion_r1_16": "o",
}

BUCKET_ORDER: tuple[str, ...] = ("head", "mid", "tail")
BUCKET_DISPLAY_NAMES: dict[str, str] = {
    "global": "Global",
    "head": "HEAD",
    "mid": "MID",
    "tail": "TAIL",
}

METRIC_ORDER: tuple[str, ...] = ("soft_rank", "hard_rank")
METRIC_DISPLAY_NAMES: dict[str, str] = {
    "soft_rank": "Soft Spectral Rank",
    "hard_rank": "Hard Spectral Rank",
}


def display_optimizer(optimizer_folder: str, fallback: str | None = None) -> str:
    """Return the paper-facing display name for an optimizer folder/key."""
    return OPTIMIZER_DISPLAY_NAMES.get(optimizer_folder, fallback or optimizer_folder)


def optimizer_color(optimizer_folder: str) -> str:
    """Return the figure color for an optimizer folder/key."""
    return OPTIMIZER_COLORS.get(optimizer_folder, "#333333")


def dion_rank_sweep_color(optimizer_folder: str) -> str:
    """Return the Dion-rank-sweep color for an optimizer folder/key."""
    return DION_RANK_SWEEP_COLORS.get(optimizer_folder, optimizer_color(optimizer_folder))


def tail_350m_color(optimizer_folder: str) -> str:
    """Return the GPT2-350M TAIL-scaling color for an optimizer folder/key."""
    return TAIL_350M_COLORS.get(optimizer_folder, optimizer_color(optimizer_folder))


def optimizer_marker(optimizer_folder: str) -> str:
    """Return the marker for an optimizer folder/key."""
    return OPTIMIZER_MARKERS.get(optimizer_folder, "o")


def dion_rank_sweep_linestyle(optimizer_folder: str) -> str:
    """Return paper-style line style for the Dion rank-sweep figure."""
    return "--" if optimizer_folder == "adamw" else "-"
