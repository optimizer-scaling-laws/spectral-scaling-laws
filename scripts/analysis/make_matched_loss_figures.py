#!/usr/bin/env python3
"""Generate matched-loss / extended-AdamW figures from processed CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optimizer_ssl.analysis.matched_loss_figures import save_matched_loss_figures  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--beta-dynamics",
        type=Path,
        default=Path("results/processed/matched_loss_beta_dynamics.csv"),
        help="Processed beta-dynamics CSV for extended AdamW training.",
    )
    parser.add_argument(
        "--pr-trajectories",
        type=Path,
        default=Path("results/processed/matched_loss_pr_trajectories.csv"),
        help="Processed hard-rank trajectory CSV for selected widths.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/figures"),
        help="Output figure directory.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf"],
        help="Output formats supported by matplotlib, e.g. pdf.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = save_matched_loss_figures(
        args.beta_dynamics,
        args.pr_trajectories,
        args.out_dir,
        formats=tuple(args.formats),
    )
    print("Generated matched-loss figures:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
