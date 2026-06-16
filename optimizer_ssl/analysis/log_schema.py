"""Log-schema normalization helpers for submitted and released telemetry logs.

This is a lightweight pre-analysis utility, not the full paper analysis pipeline.
It converts historical text logs into the clean metric vocabulary used by the
public codebase.
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


def parse_layer_metric_line(line: str) -> dict[str, Any] | None:
    """Parse one layer-metric log line and normalize metric names.

    Supports both the released schema (`soft_rank_pre`, `hard_rank_pre`, ...)
    and the submitted-run legacy schema (`SE_pre`, `PR_pre`, plus EEE/JS fields).
    Legacy entropy fields are preserved as `spectral_entropy_*` and converted to
    `soft_rank_* = exp(spectral_entropy_*)`.
    """
    step_match = _STEP_RE.search(line)
    if not step_match:
        return None
    row: dict[str, Any] = {"step": int(step_match.group("step"))}
    for match in _PAIR_RE.finditer(line):
        key = match.group("key")
        value = float(match.group("value"))
        if key in IGNORED_LEGACY_KEYS:
            continue
        normalized = LEGACY_KEY_MAP.get(key, key)
        row[normalized] = value

    if "soft_rank_pre" not in row and "spectral_entropy_pre" in row:
        row["soft_rank_pre"] = math.exp(float(row["spectral_entropy_pre"]))
    if "soft_rank_post" not in row and "spectral_entropy_post" in row:
        row["soft_rank_post"] = math.exp(float(row["spectral_entropy_post"]))
    return row
