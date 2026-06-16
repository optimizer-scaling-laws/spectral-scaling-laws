"""Covariance and spectrum utilities for FFN activation telemetry."""

import torch


def compute_covariance(x2d: torch.Tensor) -> torch.Tensor:
    """Return the sample covariance matrix for flattened activations.

    Args:
        x2d: ``[N, D]`` tensor of flattened token activations.

    Returns:
        ``[D, D]`` covariance matrix.
    """
    if x2d.dtype in (torch.float16, torch.bfloat16):
        x2d = x2d.float()

    mean = x2d.mean(dim=0, keepdim=True)
    centered = x2d - mean
    n_samples = x2d.size(0)
    return (centered.t() @ centered) / (n_samples - 1 + 1e-12)


def compute_sorted_eigs(cov: torch.Tensor) -> torch.Tensor:
    """Return covariance eigenvalues sorted in descending order."""
    if cov.dtype in (torch.float16, torch.bfloat16):
        cov = cov.float()

    values = torch.linalg.eigvalsh(cov)
    return torch.flip(torch.clamp(values, min=1e-12), dims=[0])


def normalize_eigs(eigs: torch.Tensor) -> torch.Tensor:
    """Normalize eigenvalues into a probability distribution."""
    return eigs / (torch.sum(eigs) + 1e-12)


# Backward-compatible aliases used by earlier tests/docs.
covariance = compute_covariance
sorted_eigenvalues = compute_sorted_eigs
normalize_spectrum = normalize_eigs
