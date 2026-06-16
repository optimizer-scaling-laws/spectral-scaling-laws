#!/usr/bin/env python3
"""Example: attach a spectral probe to any PyTorch module.

Replace the tiny MLP below with your own model and module of interest.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

from optimizer_ssl.probe import attach_spectral_probe


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 128)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(128, 32)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def main():
    torch.manual_seed(1337)
    model = TinyMLP()
    probe = attach_spectral_probe(model.fc1, capture="output")
    _ = model(torch.randn(64, 32))
    metrics = probe.compute()
    probe.close()
    print("Module spectral diagnostics")
    for key in ["soft_rank", "hard_rank", "spectral_entropy", "n_samples", "hidden_dim"]:
        value = metrics[key]
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
