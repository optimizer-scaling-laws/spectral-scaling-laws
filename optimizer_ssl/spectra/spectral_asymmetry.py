"""Spectral asymmetry utilities."""


def spectral_asymmetry(soft_rank: float, hard_rank: float) -> float:
    """Soft-minus-hard rank gap for a single metric point."""
    return float(soft_rank) - float(hard_rank)


def scaling_asymmetry(beta_soft: float, beta_hard: float) -> float:
    """Soft-minus-hard scaling exponent gap."""
    return float(beta_soft) - float(beta_hard)
