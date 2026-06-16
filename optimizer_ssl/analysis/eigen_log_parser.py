"""Parsers for submitted-run and released spectral telemetry logs.

The public processed-data pipeline converts text logs into a normalized CSV
schema before any aggregation or plotting.  This file handles both old
submitted-run logs (``SE_post``/``PR_post``) and logs emitted by the cleaned
public tracker (``soft_rank_post``/``hard_rank_post``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from optimizer_ssl.analysis.log_schema import parse_layer_metric_line, parse_metric_pairs

GLOBAL_LOG_PATTERN = "layer_{layer}_eigen.txt"
FREQUENCY_LOG_PATTERN = "frequency_tertiles/layer_{layer}_eigen_freq.txt"

_BUCKET_RE = re.compile(
    r"(?P<bucket>HEAD|MID|TAIL)\s*\(n=(?P<n>\d+)\)\s*:\s*(?P<body>.*)"
)
_STEP_RE = re.compile(r"Step\s+(?P<step>\d+)\s*:")
_TII_RE = re.compile(r"Step\s+(?P<step>\d+)\s*:\s*TII=(?P<tii>[-+0-9.eE]+)")

BUCKET_NAME_MAP = {"HEAD": "head", "MID": "mid", "TAIL": "tail"}

BASE_LAYER_COLUMNS = [
    "run_id",
    "step",
    "layer",
    "bucket",
    "n_tokens",
    "spectral_entropy_pre",
    "spectral_entropy_post",
    "soft_rank_pre",
    "soft_rank_post",
    "hard_rank_pre",
    "hard_rank_post",
    "source_log_schema",
    "source_path",
]

METADATA_COLUMNS = [
    "model_scale",
    "model_name",
    "model_dim",
    "n_layer",
    "num_layers",
    "base_ffn_dim",
    "width_multiplier",
    "ffn_hidden_dim",
    "optimizer",
    "optimizer_folder",
    "optimizer_variant",
    "optimizer_display_name",
    "dion_rank_fraction",
    "num_gpus",
    "seed",
    "frequency_bucket_reduction",
    "config_path",
    "paper_experiment",
    "notes",
]

LAYER_METRIC_COLUMNS = BASE_LAYER_COLUMNS + METADATA_COLUMNS


def _schema_from_metrics(row: dict[str, Any]) -> str:
    if "spectral_entropy_post" in row and "soft_rank_post" in row:
        # Current logs may contain both too, but legacy rows only have entropy in
        # text and get soft_rank filled by normalization. Keep the coarse marker.
        return "normalized"
    return "unknown"


def parse_global_layer_file(path: str | Path, layer: int, run_id: str = "") -> list[dict[str, Any]]:
    """Parse one ``layer_<n>_eigen.txt`` file.

    Returns rows with ``bucket='global'`` and normalized metric names.
    """
    path = Path(path)
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        parsed = parse_layer_metric_line(line)
        if parsed is None:
            continue
        parsed.update(
            {
                "run_id": run_id,
                "layer": layer,
                "bucket": "global",
                "n_tokens": "",
                "source_log_schema": _schema_from_metrics(parsed),
                "source_path": str(path),
            }
        )
        rows.append(parsed)
    return rows


def parse_frequency_layer_file(
    path: str | Path,
    layer: int,
    run_id: str = "",
    require_all_buckets_per_step: bool = False,
) -> list[dict[str, Any]]:
    """Parse one ``layer_<n>_eigen_freq.txt`` frequency-bucket log file.

    The submitted plotting scripts only accepted a step if HEAD/MID/TAIL were all
    present.  The default here keeps every parsed bucket row; callers that need
    exact submitted-script behavior can set ``require_all_buckets_per_step=True``.
    """
    path = Path(path)
    if not path.exists():
        return []

    rows_by_step: dict[int, list[dict[str, Any]]] = {}
    current_step: int | None = None
    for line in path.read_text().splitlines():
        step_match = _STEP_RE.search(line)
        if step_match:
            current_step = int(step_match.group("step"))
            rows_by_step.setdefault(current_step, [])
            continue
        if current_step is None:
            continue
        bucket_match = _BUCKET_RE.search(line)
        if not bucket_match:
            continue
        bucket = BUCKET_NAME_MAP[bucket_match.group("bucket")]
        row: dict[str, Any] = {
            "run_id": run_id,
            "step": current_step,
            "layer": layer,
            "bucket": bucket,
            "n_tokens": int(bucket_match.group("n")),
            "source_path": str(path),
        }
        row.update(parse_metric_pairs(bucket_match.group("body")))
        row["source_log_schema"] = _schema_from_metrics(row)
        rows_by_step[current_step].append(row)

    rows: list[dict[str, Any]] = []
    for step in sorted(rows_by_step):
        step_rows = rows_by_step[step]
        if require_all_buckets_per_step:
            seen = {str(r["bucket"]) for r in step_rows}
            if seen != {"head", "mid", "tail"}:
                continue
        rows.extend(step_rows)
    return rows


def parse_tail_integrity_file(path: str | Path, run_id: str = "") -> list[dict[str, Any]]:
    """Parse optional tail-integrity-index logs."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        match = _TII_RE.search(line)
        if match:
            rows.append(
                {
                    "run_id": run_id,
                    "step": int(match.group("step")),
                    "tail_integrity_index": float(match.group("tii")),
                    "source_path": str(path),
                }
            )
    return rows


def parse_run_log_dir(
    log_dir: str | Path,
    run_id: str,
    num_layers: int,
    include_frequency: bool = True,
    include_global: bool = True,
    require_all_buckets_per_step: bool = False,
) -> list[dict[str, Any]]:
    """Parse all global and frequency-bucket layer logs for one run."""
    log_dir = Path(log_dir)
    rows: list[dict[str, Any]] = []
    for layer in range(int(num_layers)):
        if include_global:
            rows.extend(parse_global_layer_file(log_dir / GLOBAL_LOG_PATTERN.format(layer=layer), layer, run_id))
        if include_frequency:
            rows.extend(
                parse_frequency_layer_file(
                    log_dir / FREQUENCY_LOG_PATTERN.format(layer=layer),
                    layer,
                    run_id,
                    require_all_buckets_per_step=require_all_buckets_per_step,
                )
            )
    return rows


def attach_metadata(rows: Iterable[dict[str, Any]], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Append run metadata to parsed metric rows."""
    output: list[dict[str, Any]] = []
    for row in rows:
        merged = dict(row)
        for col in METADATA_COLUMNS:
            merged[col] = metadata.get(col, "")
        output.append(merged)
    return output
