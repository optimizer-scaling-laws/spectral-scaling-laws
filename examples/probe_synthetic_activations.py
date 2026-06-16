#!/usr/bin/env python3
"""CPU-safe spectral-rank diagnostic on synthetic activations."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from optimizer_ssl.probe import spectral_rank


def main():
    torch.manual_seed(1337)
    activations = torch.randn(256, 128)
    metrics = spectral_rank(activations)
    print("Synthetic activation spectral diagnostics")
    for key in ["soft_rank", "hard_rank", "spectral_entropy", "n_samples", "hidden_dim"]:
        value = metrics[key]
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
