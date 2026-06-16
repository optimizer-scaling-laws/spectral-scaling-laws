from pathlib import Path

import numpy as np
import torch
import yaml

from optimizer_ssl.models.gpt_utils import DistributedDataLoader
from optimizer_ssl.spectra.frequency_metrics import TokenFrequencyTable, load_frequency_vector
from optimizer_ssl.train import Hyperparameters
from optimizer_ssl.train.config import validate_hyperparameters
from optimizer_ssl.utils.seed import seed_everything


def test_seed_everything_is_reproducible():
    seed_everything(123)
    a = torch.randn(4)
    seed_everything(123)
    b = torch.randn(4)
    assert torch.equal(a, b)


def test_hyperparameters_include_seed_and_error_policy():
    hp = Hyperparameters(seed=1337, spectral_error_policy="warn")
    validate_hyperparameters(hp)
    hp_bad = Hyperparameters(spectral_error_policy="silent")
    try:
        validate_hyperparameters(hp_bad)
    except ValueError as exc:
        assert "spectral_error_policy" in str(exc)
    else:
        raise AssertionError("invalid spectral_error_policy should fail validation")


def test_all_configs_include_seed_and_spectral_error_policy():
    for path in Path("configs").rglob("*.yaml"):
        cfg = yaml.safe_load(path.read_text()) or {}
        assert cfg.get("seed") == 1337, path
        assert cfg.get("spectral_error_policy") in {"warn", "raise", "nan"}, path


def test_token_frequency_public_npy_artifact_loads():
    npy_path = Path("results/processed/token_frequencies.npy")
    assert npy_path.exists()
    freq = load_frequency_vector(npy_path, vocab_size=50304)
    assert int(freq.sum().item()) == 10255376574
    table = TokenFrequencyTable()
    table.load_from_file(npy_path)
    assert table.total_tokens == 10255376574


def test_distributed_dataloader_can_return_cpu_batches(tmp_path):
    path = tmp_path / "fineweb_train_000001.bin"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    tokens = np.arange(129, dtype=np.uint16)
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())

    loader = DistributedDataLoader(str(tmp_path / "fineweb_train_*.bin"), B=2, T=8, dp_rank=0, dp_world_size=1, device="cpu")
    x, y = loader.next_batch()
    assert x.device.type == "cpu"
    assert y.device.type == "cpu"
    assert x.shape == (2, 8)
    assert y.shape == (2, 8)
