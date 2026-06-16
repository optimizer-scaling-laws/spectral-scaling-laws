import pytest
import torch

from optimizer_ssl.spectra.covariance import compute_covariance
from optimizer_ssl.spectra.frequency_metrics import (
    covariance_stats_from_2d,
    merge_covariance_statistics,
)
from optimizer_ssl.spectra.tracker import GPTEigenMetricsTracker
from optimizer_ssl.train.config import Hyperparameters, validate_hyperparameters


def test_covariance_stats_merge_matches_concatenated_covariance():
    torch.manual_seed(0)
    shard0 = torch.randn(17, 5) + 1.5
    shard1 = torch.randn(23, 5) - 0.7
    shard2 = torch.empty(0, 5)
    shard3 = torch.randn(1, 5) + 4.0

    stats = [covariance_stats_from_2d(x) for x in [shard0, shard1, shard2, shard3]]
    merged_cov, total_n = merge_covariance_statistics(stats)

    full = torch.cat([shard0, shard1, shard3], dim=0)
    expected_cov = compute_covariance(full)

    assert total_n == full.shape[0]
    assert torch.allclose(merged_cov, expected_cov, atol=1e-5, rtol=1e-5)


def test_covariance_stats_handles_empty_and_singleton_shards():
    stats = [
        covariance_stats_from_2d(torch.empty(0, 3)),
        covariance_stats_from_2d(torch.tensor([[1.0, 2.0, 3.0]])),
    ]
    cov, n = merge_covariance_statistics(stats)
    assert n == 1
    assert cov.shape == (3, 3)
    assert torch.allclose(cov, torch.zeros(3, 3))


def test_frequency_bucket_reduction_config_validation():
    hp = Hyperparameters(track_by_frequency=True, token_freq_file="results/processed/token_frequencies.npy")
    validate_hyperparameters(hp)

    hp_bad = Hyperparameters(frequency_bucket_reduction="rank0")
    with pytest.raises(ValueError, match="frequency_bucket_reduction"):
        validate_hyperparameters(hp_bad)

    hp_missing_freq = Hyperparameters(track_by_frequency=True, token_freq_file=None)
    with pytest.raises(ValueError, match="token_freq_file"):
        validate_hyperparameters(hp_missing_freq)


def test_tracker_rejects_invalid_frequency_bucket_reduction_without_model_introspection():
    # Invalid reduction mode is rejected before model hook registration, so this
    # does not require constructing a GPT model.
    with pytest.raises(ValueError, match="frequency_bucket_reduction"):
        GPTEigenMetricsTracker(
            model=object(),
            num_layers=1,
            track_by_frequency=False,
            frequency_bucket_reduction="rank0",
        )


def test_tracker_rejects_invalid_error_policy_without_model_introspection():
    with pytest.raises(ValueError, match="error_policy"):
        GPTEigenMetricsTracker(
            model=object(),
            num_layers=1,
            track_by_frequency=False,
            error_policy="silent",
        )
