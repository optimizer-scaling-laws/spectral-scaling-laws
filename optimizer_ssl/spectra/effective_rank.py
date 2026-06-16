"""Paper-facing effective-rank metrics for covariance spectra."""

from __future__ import annotations

import torch

from optimizer_ssl.spectra.covariance import normalize_eigs


def compute_spectral_entropy(lam_norm: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of a normalized eigenvalue spectrum."""
    eps = 1e-12
    return -torch.sum(lam_norm * torch.log(lam_norm + eps))


def compute_soft_rank(lam_norm: torch.Tensor) -> torch.Tensor:
    """Shannon effective rank: ``exp(spectral_entropy)``."""
    return torch.exp(compute_spectral_entropy(lam_norm))


def compute_participation_ratio(lam: torch.Tensor) -> torch.Tensor:
    """Hard effective rank, implemented as participation ratio."""
    sum_ = torch.sum(lam) + 1e-12
    sum_sq = torch.sum(lam**2) + 1e-12
    return (sum_**2) / sum_sq


def compute_effective_rank_metrics(eigs: torch.Tensor) -> dict[str, float]:
    """Compute the paper-facing spectral metrics from covariance eigenvalues."""
    lam_norm = normalize_eigs(eigs)
    spectral_entropy = compute_spectral_entropy(lam_norm).item()
    soft_rank = compute_soft_rank(lam_norm).item()
    hard_rank = compute_participation_ratio(eigs).item()
    return {
        "spectral_entropy": spectral_entropy,
        "soft_rank": soft_rank,
        "hard_rank": hard_rank,
    }


# Short aliases for interactive use and backward compatibility.
def spectral_entropy(eigenvalues: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute entropy from raw eigenvalues."""
    lam = torch.clamp(eigenvalues.float(), min=eps)
    return compute_spectral_entropy(lam / (lam.sum() + eps))


def soft_rank(eigenvalues: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute Shannon effective rank from raw eigenvalues."""
    lam = torch.clamp(eigenvalues.float(), min=eps)
    return torch.exp(spectral_entropy(lam, eps=eps))


def participation_ratio(eigenvalues: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute participation ratio from raw eigenvalues."""
    lam = torch.clamp(eigenvalues.float(), min=eps)
    return (lam.sum() ** 2) / (torch.sum(lam**2) + eps)


hard_rank = participation_ratio
