import torch


def compute_bucket_stats(freq: torch.Tensor) -> dict:
    freq = freq.to(torch.long).cpu().flatten()
    sorted_freqs, _ = freq.sort(descending=True)
    cumsum = sorted_freqs.cumsum(0).float()
    total = cumsum[-1].item()
    head_cutoff_idx = (cumsum <= total * 0.33).sum().item()
    mid_cutoff_idx = (cumsum <= total * 0.67).sum().item()
    head_cutoff_idx = max(1, min(head_cutoff_idx, len(sorted_freqs) - 1))
    mid_cutoff_idx = max(head_cutoff_idx + 1, min(mid_cutoff_idx, len(sorted_freqs) - 1))
    head_min_freq = int(sorted_freqs[head_cutoff_idx - 1].item())
    mid_min_freq = int(sorted_freqs[mid_cutoff_idx - 1].item())
    head = freq >= head_min_freq
    mid = (freq >= mid_min_freq) & (freq < head_min_freq)
    tail = freq < mid_min_freq
    return {
        "vocab_size": int(len(freq)),
        "total_tokens": int(freq.sum().item()),
        "head_min_frequency": head_min_freq,
        "mid_min_frequency": mid_min_freq,
        "zero_frequency_tokens": int((freq == 0).sum().item()),
        "buckets": {
            "head": {"num_tokens": int(head.sum().item()), "num_occurrences": int(freq[head].sum().item())},
            "mid": {"num_tokens": int(mid.sum().item()), "num_occurrences": int(freq[mid].sum().item())},
            "tail": {"num_tokens": int(tail.sum().item()), "num_occurrences": int(freq[tail].sum().item())},
        },
    }


def assign_frequency_buckets(token_ids: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
    stats = compute_bucket_stats(freq)
    token_freqs = freq[token_ids.cpu()].to(token_ids.device)
    buckets = torch.full_like(token_ids, 2)
    buckets[token_freqs >= stats["mid_min_frequency"]] = 1
    buckets[token_freqs >= stats["head_min_frequency"]] = 0
    return buckets
