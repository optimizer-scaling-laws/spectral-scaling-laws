"""Log-schema normalization helpers for submitted and released telemetry logs.

The submitted-run logs used short legacy names such as ``SE_post`` and
``PR_post`` and also contained legacy diagnostics (``EEE``/``JS``) that are not
part of the public metric vocabulary.  The released code uses explicit names:
``spectral_entropy_*``, ``soft_rank_*``, and ``hard_rank_*``.

This module is intentionally torch-free so raw-log parsing works in the
lightweight analysis environment.
"""

from __future__ import annotations

import math
import re
from typing import Any

_STEP_RE = re.compile(r"Step\s+(?P<step>\d+)\s*:")
_PAIR_RE = re.compile(r"(?P<key>[A-Za-z_]+)=(?P<value>[-+0-9.eE]+)")

LEGACY_KEY_MAP = {
    "SE_pre": "spectral_entropy_pre",
    "SE_post": "spectral_entropy_post",
    "PR_pre": "hard_rank_pre",
    "PR_post": "hard_rank_post",
}
IGNORED_LEGACY_KEYS = {"EEE_pre", "EEE_post", "JS"}


def parse_metric_pairs(text: str) -> dict[str, float]:
    """Parse and normalize metric ``key=value`` pairs from a log line.

    The returned dictionary may contain both pre- and post-activation metrics.
    Legacy spectral entropy values are converted to the public soft-rank fields.
    Legacy EEE/JS fields are ignored.
    """
    row: dict[str, float] = {}
    for match in _PAIR_RE.finditer(text):
        key = match.group("key")
        if key in IGNORED_LEGACY_KEYS:
            continue
        normalized = LEGACY_KEY_MAP.get(key, key)
        row[normalized] = float(match.group("value"))

    if "soft_rank_pre" not in row and "spectral_entropy_pre" in row:
        row["soft_rank_pre"] = math.exp(float(row["spectral_entropy_pre"]))
    if "soft_rank_post" not in row and "spectral_entropy_post" in row:
        row["soft_rank_post"] = math.exp(float(row["spectral_entropy_post"]))
    return row


def parse_layer_metric_line(line: str) -> dict[str, Any] | None:
    """Parse one global layer-metric log line and normalize metric names.

    Supports both the released schema (``soft_rank_pre``, ``hard_rank_pre``, ...)
    and the submitted-run legacy schema (``SE_pre``, ``PR_pre``, plus EEE/JS).
    """
    step_match = _STEP_RE.search(line)
    if not step_match:
        return None
    row: dict[str, Any] = {"step": int(step_match.group("step"))}
    row.update(parse_metric_pairs(line))
    return row
