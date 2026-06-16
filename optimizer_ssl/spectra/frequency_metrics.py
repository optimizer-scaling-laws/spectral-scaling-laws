"""HEAD/MID/TAIL frequency-bucket utilities for spectral telemetry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_frequency_vector(filepath: str | Path, *, vocab_size: int | None = None) -> torch.Tensor:
    """Load a token-frequency vector from ``.npy``, ``.npz``, or ``.pt``.

    Use this helper for public artifacts so pickle-backed ``.pt`` files are
    loaded with ``weights_only=True`` when available.
    """
    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        freq = torch.from_numpy(np.load(path)).to(torch.long)
    elif suffix == ".npz":
        archive = np.load(path)
        key = "frequencies" if "frequencies" in archive else archive.files[0]
        freq = torch.from_numpy(archive[key]).to(torch.long)
    elif suffix == ".pt":
        try:
            freq = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            freq = torch.load(path, map_location="cpu")
        if not isinstance(freq, torch.Tensor):
            raise TypeError(f"Expected tensor in {path}, found {type(freq)!r}")
        freq = freq.to(torch.long).cpu()
    else:
        raise ValueError(f"Unsupported token-frequency file format: {path.suffix}")
    freq = freq.flatten()
    if vocab_size is not None and len(freq) != vocab_size:
        raise ValueError(f"Frequency table size {len(freq)} != vocab_size {vocab_size}")
    return freq


class TokenFrequencyTable:
    """Token frequency statistics and occurrence-balanced HEAD/MID/TAIL buckets."""

    TERTILE_NAMES = ["head", "mid", "tail"]

    def __init__(self, vocab_size: int = 50304):
        self.vocab_size = vocab_size
        self.frequencies = None
        self.total_tokens = 0
        self.head_min_freq = None
        self.mid_min_freq = None
        self.tokens_per_tertile = None
        self.occurrences_per_tertile = None

    def load_from_file(self, filepath: str | Path):
        """Load a saved ``[vocab_size]`` token-frequency vector."""
        self.frequencies = load_frequency_vector(filepath, vocab_size=self.vocab_size)
        self.total_tokens = int(self.frequencies.sum().item())
        self._compute_tertile_boundaries()
        self._compute_statistics()

    def load_from_tensor(self, frequencies: torch.Tensor):
        """Load token frequencies directly from a tensor."""
        self.frequencies = frequencies.cpu()
        self.vocab_size = len(frequencies)
        self.total_tokens = self.frequencies.sum().item()
        self._compute_tertile_boundaries()
        self._compute_statistics()

    def compute_from_data_files(self, data_files: list, save_path: str | None = None):
        """Compute frequencies from raw uint16 token files.

        This method is kept for compatibility with older scripts. Public FineWeb
        preprocessing uses ``scripts/preprocess/compute_token_frequencies.py``,
        which understands the repository's headered shard format.
        """
        self.frequencies = torch.zeros(self.vocab_size, dtype=torch.long)
        for filepath in data_files:
            tokens = np.memmap(filepath, dtype=np.uint16, mode="r")
            for token_id in tokens:
                if token_id < self.vocab_size:
                    self.frequencies[token_id] += 1

        self.total_tokens = self.frequencies.sum().item()
        self._compute_tertile_boundaries()
        self._compute_statistics()

        if save_path:
            torch.save(self.frequencies, save_path)
            print(f"Saved token frequencies to {save_path}")

    def _compute_tertile_boundaries(self):
        """Compute occurrence-balanced HEAD/MID/TAIL boundaries."""
        sorted_freqs, _ = self.frequencies.sort(descending=True)
        cumsum = sorted_freqs.cumsum(0).float()
        total = cumsum[-1].item()

        head_cutoff_idx = (cumsum <= total * 0.33).sum().item()
        mid_cutoff_idx = (cumsum <= total * 0.67).sum().item()

        head_cutoff_idx = max(1, min(head_cutoff_idx, len(sorted_freqs) - 1))
        mid_cutoff_idx = max(head_cutoff_idx + 1, min(mid_cutoff_idx, len(sorted_freqs) - 1))

        self.head_min_freq = sorted_freqs[head_cutoff_idx - 1].item()
        self.mid_min_freq = sorted_freqs[mid_cutoff_idx - 1].item()

    def _compute_statistics(self):
        """Compute bucket sizes and occurrence counts."""
        self.tokens_per_tertile = {
            "head": (self.frequencies >= self.head_min_freq).sum().item(),
            "mid": ((self.frequencies >= self.mid_min_freq) & (self.frequencies < self.head_min_freq)).sum().item(),
            "tail": (self.frequencies < self.mid_min_freq).sum().item(),
        }
        self.occurrences_per_tertile = {
            "head": self.frequencies[self.frequencies >= self.head_min_freq].sum().item(),
            "mid": self.frequencies[
                (self.frequencies >= self.mid_min_freq) & (self.frequencies < self.head_min_freq)
            ].sum().item(),
            "tail": self.frequencies[self.frequencies < self.mid_min_freq].sum().item(),
        }

    def get_tertile(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Assign token IDs to 0=HEAD, 1=MID, 2=TAIL buckets."""
        freqs = self.frequencies[token_ids.cpu()].to(token_ids.device)
        tertiles = torch.full_like(token_ids, 2)
        tertiles[freqs >= self.mid_min_freq] = 1
        tertiles[freqs >= self.head_min_freq] = 0
        return tertiles

    def get_frequency(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return corpus frequency for each token ID."""
        return self.frequencies[token_ids.cpu()].to(token_ids.device)

    def summary(self) -> str:
        """Human-readable summary of the frequency distribution."""
        lines = [
            "Token Frequency Table Summary",
            "=" * 40,
            f"Vocab size: {self.vocab_size:,}",
            f"Total tokens: {self.total_tokens:,}",
            "",
            "Tertile Boundaries:",
            f"  HEAD: freq >= {self.head_min_freq:,}",
            f"  MID:  {self.mid_min_freq:,} <= freq < {self.head_min_freq:,}",
            f"  TAIL: freq < {self.mid_min_freq:,}",
            "",
            "Tokens per Tertile:",
            f"  HEAD: {self.tokens_per_tertile['head']:,} tokens ({100*self.tokens_per_tertile['head']/self.vocab_size:.2f}% of vocab)",
            f"  MID:  {self.tokens_per_tertile['mid']:,} tokens ({100*self.tokens_per_tertile['mid']/self.vocab_size:.2f}% of vocab)",
            f"  TAIL: {self.tokens_per_tertile['tail']:,} tokens ({100*self.tokens_per_tertile['tail']/self.vocab_size:.2f}% of vocab)",
            "",
            "Occurrences per Tertile:",
            f"  HEAD: {self.occurrences_per_tertile['head']:,} ({100*self.occurrences_per_tertile['head']/self.total_tokens:.2f}% of occurrences)",
            f"  MID:  {self.occurrences_per_tertile['mid']:,} ({100*self.occurrences_per_tertile['mid']/self.total_tokens:.2f}% of occurrences)",
            f"  TAIL: {self.occurrences_per_tertile['tail']:,} ({100*self.occurrences_per_tertile['tail']/self.total_tokens:.2f}% of occurrences)",
        ]
        return "\n".join(lines)



def covariance_stats_from_2d(x2d: torch.Tensor) -> dict[str, torch.Tensor | int]:
    """Return sufficient statistics for an unbiased sample covariance.

    The returned ``scatter`` is the unnormalized centered second moment,
    ``(X - mean).T @ (X - mean)``. These statistics can be merged exactly
    across data-parallel ranks without gathering raw activations.
    """
    if x2d.dim() != 2:
        raise ValueError(f"Expected a 2D tensor, got shape {tuple(x2d.shape)}")
    if x2d.dtype in (torch.float16, torch.bfloat16):
        x2d = x2d.float()

    n_samples = int(x2d.shape[0])
    dim = int(x2d.shape[1])
    if n_samples == 0:
        return {
            "n": 0,
            "mean": torch.zeros(dim, dtype=x2d.dtype, device=x2d.device),
            "scatter": torch.zeros(dim, dim, dtype=x2d.dtype, device=x2d.device),
        }

    mean = x2d.mean(dim=0)
    if n_samples == 1:
        scatter = torch.zeros(dim, dim, dtype=x2d.dtype, device=x2d.device)
    else:
        centered = x2d - mean
        scatter = centered.t() @ centered
    return {"n": n_samples, "mean": mean, "scatter": scatter}


def merge_covariance_statistics(stats: list[dict[str, torch.Tensor | int]]) -> tuple[torch.Tensor, int]:
    """Merge per-shard covariance sufficient statistics.

    Args:
        stats: list of dictionaries returned by :func:`covariance_stats_from_2d`.

    Returns:
        ``(covariance, total_samples)`` where covariance is the unbiased sample
        covariance over the concatenation of all shards. If fewer than two
        total samples are present, the covariance is a zero matrix with the
        correct shape.
    """
    if not stats:
        raise ValueError("Cannot merge an empty list of covariance statistics")

    first_mean = stats[0]["mean"]
    assert isinstance(first_mean, torch.Tensor)
    dim = first_mean.numel()
    device = first_mean.device
    dtype = first_mean.dtype

    total_n = int(sum(int(s["n"]) for s in stats))
    if total_n == 0:
        return torch.zeros(dim, dim, dtype=dtype, device=device), 0

    global_mean = torch.zeros(dim, dtype=dtype, device=device)
    for s in stats:
        n = int(s["n"])
        if n == 0:
            continue
        mean = s["mean"]
        assert isinstance(mean, torch.Tensor)
        global_mean += mean.to(device=device, dtype=dtype) * (n / total_n)

    merged_scatter = torch.zeros(dim, dim, dtype=dtype, device=device)
    for s in stats:
        n = int(s["n"])
        scatter = s["scatter"]
        mean = s["mean"]
        assert isinstance(scatter, torch.Tensor)
        assert isinstance(mean, torch.Tensor)
        merged_scatter += scatter.to(device=device, dtype=dtype)
        if n > 0:
            diff = mean.to(device=device, dtype=dtype) - global_mean
            merged_scatter += torch.outer(diff, diff) * n

    if total_n < 2:
        return torch.zeros(dim, dim, dtype=dtype, device=device), total_n
    return merged_scatter / (total_n - 1), total_n

def tail_integrity_index(tail_hard_ranks: list[float], head_hard_ranks: list[float]) -> float:
    """Average tail-minus-head hard-rank gap across layers."""
    if len(tail_hard_ranks) != len(head_hard_ranks):
        raise ValueError("tail_hard_ranks and head_hard_ranks must have equal length")
    if not tail_hard_ranks:
        raise ValueError("Cannot compute Tail Integrity Index from an empty list")
    return float(sum(t - h for t, h in zip(tail_hard_ranks, head_hard_ranks)) / len(tail_hard_ranks))
