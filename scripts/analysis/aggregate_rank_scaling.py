#!/usr/bin/env python3
"""Aggregate normalized layer metrics into rank-scaling points and beta tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optimizer_ssl.analysis.rank_aggregation import (  # noqa: E402
    POINT_COLUMNS,
    aggregate_final_window_points,
    split_global_and_frequency_points,
)
from optimizer_ssl.analysis.scaling_fits import BETA_COLUMNS, fit_rank_scaling_points  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--final-samples", type=int, default=5)
    parser.add_argument("--min-fit-points", type=int, default=3)
    args = parser.parse_args()

    layer_rows = _read_csv(args.layer_metrics)
    points = aggregate_final_window_points(layer_rows, final_samples=args.final_samples)
    global_points, bucket_points = split_global_and_frequency_points(points)

    _write_csv(args.output_dir / "global_rank_scaling_points.csv", global_points, POINT_COLUMNS)
    _write_csv(
        args.output_dir / "frequency_bucket_rank_scaling_points.csv", bucket_points, POINT_COLUMNS
    )

    global_betas = fit_rank_scaling_points(global_points, min_points=args.min_fit_points)
    bucket_betas = fit_rank_scaling_points(bucket_points, min_points=args.min_fit_points)
    _write_csv(args.output_dir / "main_beta_table.csv", global_betas, BETA_COLUMNS)
    _write_csv(args.output_dir / "frequency_bucket_beta_table.csv", bucket_betas, BETA_COLUMNS)


if __name__ == "__main__":
    main()
