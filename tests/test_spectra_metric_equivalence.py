import torch

from optimizer_ssl.spectra import GPTEigenMetricsTracker
from optimizer_ssl.spectra.covariance import compute_covariance, compute_sorted_eigs, normalize_eigs
from optimizer_ssl.spectra.effective_rank import (
    compute_effective_rank_metrics,
    compute_participation_ratio,
    compute_soft_rank,
    compute_spectral_entropy,
)
from optimizer_ssl.spectra.eigen_metrics_gpt import GPTEigenMetricsTracker as ShimmedTracker
from optimizer_ssl.spectra.frequency_metrics import TokenFrequencyTable, tail_integrity_index


def test_effective_rank_metric_relationships():
    eigs = torch.tensor([4.0, 2.0, 1.0, 0.5])
    lam_norm = normalize_eigs(eigs)
    entropy = compute_spectral_entropy(lam_norm)
    assert torch.isclose(compute_soft_rank(lam_norm), torch.exp(entropy))
    assert torch.isclose(
        compute_participation_ratio(eigs),
        (eigs.sum() ** 2) / (torch.sum(eigs**2) + 1e-12),
    )

    metrics = compute_effective_rank_metrics(eigs)
    assert set(metrics) == {"spectral_entropy", "soft_rank", "hard_rank"}
    assert metrics["soft_rank"] > metrics["hard_rank"]


def test_covariance_to_metrics_pipeline_smoke():
    torch.manual_seed(0)
    activations = torch.randn(16, 8)
    cov = compute_covariance(activations)
    eigs = compute_sorted_eigs(cov)
    metrics = compute_effective_rank_metrics(eigs)
    assert cov.shape == (8, 8)
    assert eigs.shape == (8,)
    assert metrics["soft_rank"] >= 1.0
    assert metrics["hard_rank"] >= 1.0


def test_frequency_table_and_tail_integrity_index():
    freq = torch.tensor([100, 80, 40, 20, 10, 5, 1, 0], dtype=torch.long)
    table = TokenFrequencyTable(vocab_size=8)
    table.load_from_tensor(freq)
    buckets = table.get_tertile(torch.arange(8))
    assert set(buckets.tolist()).issubset({0, 1, 2})
    assert table.TERTILE_NAMES == ["head", "mid", "tail"]
    assert tail_integrity_index([10.0, 12.0], [20.0, 18.0]) == -8.0


def test_eigen_metrics_shim_preserves_tracker_import():
    assert ShimmedTracker is GPTEigenMetricsTracker
