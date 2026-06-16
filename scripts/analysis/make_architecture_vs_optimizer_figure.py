#!/usr/bin/env python3
"""Generate the architecture-vs-optimizer comparison figure from processed CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optimizer_ssl.analysis.architecture_vs_optimizer_figures import save_architecture_vs_optimizer_figure  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("results/processed/architecture_vs_optimizer_comparison.csv"),
        help="Processed architecture-vs-optimizer comparison CSV.",
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
    paths = save_architecture_vs_optimizer_figure(
        args.comparison,
        args.out_dir,
        formats=tuple(args.formats),
    )
    print("Generated architecture-vs-optimizer figure:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
