import torch
from optimizer_ssl.spectra.effective_rank import soft_rank, participation_ratio


def test_uniform_spectrum_has_full_effective_rank():
    lam = torch.ones(8)
    assert torch.isclose(soft_rank(lam), torch.tensor(8.0), atol=1e-4)
    assert torch.isclose(participation_ratio(lam), torch.tensor(8.0), atol=1e-4)


def test_rank_one_spectrum_has_low_effective_rank():
    lam = torch.tensor([1.0, 1e-12, 1e-12, 1e-12])
    assert participation_ratio(lam) < 1.01


def test_soft_rank_rank_one():
    lam = torch.tensor([10.0, 0.0, 0.0, 0.0])
    assert torch.isclose(soft_rank(lam), torch.tensor(1.0), atol=1e-4)
