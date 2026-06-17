#!/usr/bin/env python3
"""Parse raw eigen telemetry logs into a normalized layer-metrics CSV.

Input is a run manifest CSV. Each row describes one training run and points to
its ``eigen_metrics_logs`` directory. The parser supports both paper-run
legacy logs (SE/PR names) and logs emitted by the cleaned public tracker.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optimizer_ssl.analysis.eigen_log_parser import (  # noqa: E402
    LAYER_METRIC_COLUMNS,
    attach_metadata,
    parse_run_log_dir,
)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    fieldnames = set(reader.fieldnames or [])
    required = {"run_id", "log_dir"}
    missing = required - fieldnames
    if missing:
        raise ValueError(f"Manifest {path} is missing required columns: {sorted(missing)}")
    if "n_layer" not in fieldnames and "num_layers" not in fieldnames:
        raise ValueError(f"Manifest {path} must include either 'n_layer' or 'num_layers'")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Run manifest CSV")
    parser.add_argument("--output", required=True, type=Path, help="Output layer_metrics.csv")
    parser.add_argument(
        "--require-all-buckets-per-step",
        action="store_true",
        help="Match the legacy plotting scripts by keeping a frequency step only if HEAD/MID/TAIL all appear.",
    )
    args = parser.parse_args()

    manifest = _read_manifest(args.manifest)
    all_rows: list[dict[str, object]] = []
    for run in manifest:
        run_id = run["run_id"]
        log_dir = Path(run["log_dir"])
        if not log_dir.is_absolute():
            log_dir = (args.manifest.parent / log_dir).resolve()
        n_layer = int(run.get("n_layer") or run.get("num_layers"))
        parsed = parse_run_log_dir(
            log_dir=log_dir,
            run_id=run_id,
            num_layers=n_layer,
            require_all_buckets_per_step=args.require_all_buckets_per_step,
        )
        parsed = attach_metadata(parsed, run)
        all_rows.extend(parsed)
        print(f"{run_id}: parsed {len(parsed)} rows from {log_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LAYER_METRIC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
