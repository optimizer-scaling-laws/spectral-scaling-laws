"""Backward-compatible imports for GPT spectral telemetry.

New code should import from ``optimizer_ssl.spectra.tracker`` or directly from the
smaller metric modules. This shim preserves older imports used by earlier
training scripts.
"""

from optimizer_ssl.spectra.covariance import compute_covariance, compute_sorted_eigs, normalize_eigs
from optimizer_ssl.spectra.effective_rank import (
    compute_effective_rank_metrics,
    compute_participation_ratio,
    compute_soft_rank,
    compute_spectral_entropy,
)
from optimizer_ssl.spectra.frequency_metrics import TokenFrequencyTable, tail_integrity_index
from optimizer_ssl.spectra.tracker import GPTEigenMetricsTracker

__all__ = [
    "GPTEigenMetricsTracker",
    "TokenFrequencyTable",
    "tail_integrity_index",
    "compute_covariance",
    "compute_sorted_eigs",
    "normalize_eigs",
    "compute_spectral_entropy",
    "compute_soft_rank",
    "compute_participation_ratio",
    "compute_effective_rank_metrics",
]
