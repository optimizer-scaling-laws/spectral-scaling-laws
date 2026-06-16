#!/usr/bin/env python3
"""Generate GPT2-350M TAIL-token soft/hard rank figures from processed CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optimizer_ssl.analysis.rank_scaling_figures import save_tail_350m_rank_figures  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--points",
        type=Path,
        default=Path("results/processed/tail_350m_rank_scaling_points.csv"),
        help="Processed GPT2-350M TAIL rank scaling points CSV.",
    )
    parser.add_argument(
        "--betas",
        type=Path,
        default=Path("results/processed/tail_350m_beta_table.csv"),
        help="GPT2-350M TAIL beta table CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/figures"),
        help="Output figure directory.",
    )
    parser.add_argument("--model-scale", default="350m")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf"],
        help="Output formats supported by matplotlib, e.g. pdf.",
    )
    parser.add_argument(
        "--optimizers",
        nargs="*",
        default=None,
        help="Optional optimizer_folder subset, e.g. adamw muon normuon dion_r1_16.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = save_tail_350m_rank_figures(
        args.points,
        args.betas,
        args.out_dir,
        model_scale=args.model_scale,
        formats=tuple(args.formats),
        optimizer_folders=args.optimizers,
    )
    print("Generated GPT2-350M TAIL figures:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
