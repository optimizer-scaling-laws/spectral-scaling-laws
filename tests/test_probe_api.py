import torch
import torch.nn as nn

from optimizer_ssl.probe import attach_spectral_probe, spectral_rank


def test_spectral_rank_cpu_tensor_api():
    torch.manual_seed(0)
    x = torch.randn(64, 16)
    metrics = spectral_rank(x)
    assert metrics["n_samples"] == 64
    assert metrics["hidden_dim"] == 16
    assert metrics["soft_rank"] > 0
    assert metrics["hard_rank"] > 0
    assert metrics["soft_rank"] >= metrics["hard_rank"] - 1e-5


def test_attach_spectral_probe_to_arbitrary_module():
    torch.manual_seed(0)
    layer = nn.Linear(8, 32)
    probe = attach_spectral_probe(layer, capture="output")
    _ = layer(torch.randn(20, 8))
    metrics = probe.compute()
    probe.close()
    assert metrics["n_samples"] == 20
    assert metrics["hidden_dim"] == 32
    assert metrics["soft_rank"] > 0
