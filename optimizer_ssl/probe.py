"""Model-agnostic spectral-rank diagnostics.

This module is the lightweight adoption API: pass activation tensors from any
model and get the paper-facing spectral telemetry without launching the full
Optimizer-SSL training stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from optimizer_ssl.spectra.covariance import compute_covariance, compute_sorted_eigs, normalize_eigs
from optimizer_ssl.spectra.effective_rank import (
    compute_participation_ratio,
    compute_soft_rank,
    compute_spectral_entropy,
)
from optimizer_ssl.spectra.frequency_metrics import TokenFrequencyTable

CaptureKind = Literal["input", "output"]


def _flatten_activations(activations: torch.Tensor) -> torch.Tensor:
    """Flatten activations to ``[num_samples, hidden_dim]``."""
    if activations.dim() < 2:
        raise ValueError(f"Expected at least 2D activations, got shape {tuple(activations.shape)}")
    if activations.dtype in (torch.float16, torch.bfloat16):
        activations = activations.float()
    return activations.reshape(-1, activations.shape[-1])


def _rank_metrics_from_2d(x2d: torch.Tensor) -> dict[str, float | int]:
    if x2d.dim() != 2:
        raise ValueError(f"Expected 2D activations, got shape {tuple(x2d.shape)}")
    if x2d.shape[0] < 2:
        raise ValueError("At least two activation samples are required")
    cov = compute_covariance(x2d.float())
    eigvals = compute_sorted_eigs(cov)
    normalized = normalize_eigs(eigvals)
    spectral_entropy = float(compute_spectral_entropy(normalized).item())
    return {
        "soft_rank": float(compute_soft_rank(normalized).item()),
        "hard_rank": float(compute_participation_ratio(eigvals).item()),
        "spectral_entropy": spectral_entropy,
        "n_samples": int(x2d.shape[0]),
        "hidden_dim": int(x2d.shape[1]),
    }


def spectral_rank(
    activations: torch.Tensor,
    *,
    token_ids: torch.Tensor | None = None,
    token_freq_file: str | Path | None = None,
    token_freq_table: TokenFrequencyTable | None = None,
    min_samples_per_bucket: int = 2,
) -> dict:
    """Compute soft/hard spectral-rank diagnostics for arbitrary activations.

    Args:
        activations: Tensor with hidden dimension last, e.g. ``[batch, seq, dim]``
            or ``[tokens, dim]``.
        token_ids: Optional token IDs aligned with the activation samples. If
            provided with a token-frequency table, HEAD/MID/TAIL bucket metrics
            are also returned.
        token_freq_file: Optional ``.npy``, ``.npz``, or ``.pt`` frequency table.
        token_freq_table: Preloaded frequency table.
        min_samples_per_bucket: Minimum samples required before a bucket metric
            is computed.
    """
    x2d = _flatten_activations(activations.detach().cpu())
    out = _rank_metrics_from_2d(x2d)

    if token_ids is None and token_freq_file is None and token_freq_table is None:
        return out
    if token_ids is None:
        raise ValueError("token_ids are required for frequency-bucketed metrics")

    if token_freq_table is None:
        if token_freq_file is None:
            raise ValueError("token_freq_file or token_freq_table is required for bucket metrics")
        token_freq_table = TokenFrequencyTable()
        token_freq_table.load_from_file(token_freq_file)

    flat_tokens = token_ids.detach().cpu().reshape(-1)
    if flat_tokens.numel() != x2d.shape[0]:
        raise ValueError(
            f"token_ids has {flat_tokens.numel()} entries but activations have {x2d.shape[0]} samples"
        )

    buckets = token_freq_table.get_tertile(flat_tokens)
    bucket_metrics = {}
    for bucket_idx, bucket_name in enumerate(TokenFrequencyTable.TERTILE_NAMES):
        mask = buckets == bucket_idx
        n = int(mask.sum().item())
        if n < min_samples_per_bucket:
            bucket_metrics[bucket_name] = {"status": "insufficient_samples", "n_samples": n}
        else:
            bucket_metrics[bucket_name] = _rank_metrics_from_2d(x2d[mask])
    out["frequency_buckets"] = bucket_metrics
    return out


@dataclass
class SpectralHook:
    """Attach to any ``nn.Module`` and record input or output activations."""

    module: torch.nn.Module
    capture: CaptureKind = "output"
    activations: list[torch.Tensor] = field(default_factory=list)

    def __post_init__(self):
        if self.capture not in {"input", "output"}:
            raise ValueError("capture must be 'input' or 'output'")
        if self.capture == "output":
            self.handle = self.module.register_forward_hook(self._hook_output)
        else:
            self.handle = self.module.register_forward_pre_hook(self._hook_input)

    def _hook_output(self, module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        self.activations.append(output.detach().cpu())

    def _hook_input(self, module, inputs):
        self.activations.append(inputs[0].detach().cpu())

    def clear(self) -> None:
        self.activations.clear()

    def close(self) -> None:
        self.handle.remove()

    def compute(self, **kwargs) -> dict:
        if not self.activations:
            raise RuntimeError("No activations recorded. Run a forward pass first.")
        return spectral_rank(torch.cat([_flatten_activations(a) for a in self.activations], dim=0), **kwargs)


def attach_spectral_probe(module: torch.nn.Module, *, capture: CaptureKind = "output") -> SpectralHook:
    """Attach a spectral probe to any PyTorch module."""
    return SpectralHook(module=module, capture=capture)
