# Metrics

The repository tracks spectral capacity through covariance spectra of FFN activations.

## Core metrics

- **Soft rank**: `exp(H(p))`, where `p` is the normalized eigenvalue spectrum and `H(p) = -sum_i p_i log p_i`.
- **Hard rank / participation ratio**: `(sum_i lambda_i)^2 / sum_i lambda_i^2`.
- **Spectral asymmetry**: the gap between soft-rank and hard-rank scaling behavior.
- **HEAD/MID/TAIL buckets**: token-frequency buckets formed by occurrence-mass tertiles using `results/processed/token_frequencies.npy`.
- **Tail Integrity Index**: a paper-level summary of how strongly low-frequency-token representations preserve spectral capacity under width scaling.

## Log vocabulary

Training logs use paper-facing names:

```text
soft_rank_pre
soft_rank_post
hard_rank_pre
hard_rank_post
spectral_entropy_pre
spectral_entropy_post
```

`spectral_entropy_*` is retained as a diagnostic because soft rank is computed from it. The headline effective-rank metrics are `soft_rank_*` and `hard_rank_*`.

## Frequency-bucket reduction modes

Pooled all-token spectral metrics are computed from globally reduced covariance statistics across data-parallel ranks when distributed training is enabled. Frequency-bucketed HEAD/MID/TAIL metrics support two explicit modes:

```yaml
frequency_bucket_reduction: rank0_local
```

This is the submitted-paper compatibility mode. It computes bucketed metrics from the rank-0 local shard at each logging step and is used by all frozen paper-run configs. This preserves exact compatibility with the submitted experiments.

```yaml
frequency_bucket_reduction: distributed_covariance
```

This mode computes bucket-specific covariance sufficient statistics on every rank, reduces those statistics globally, and then computes soft/hard ranks on rank 0. It is the recommended mode for new multi-GPU experiments where globally reduced bucket telemetry is desired.

Both modes use the same public log vocabulary. The difference is only how the bucket-specific covariance is estimated before rank metrics are computed.


## Code organization

The metric implementation is split into small modules:

```text
optimizer_ssl/spectra/covariance.py          # covariance and eigenvalue utilities
optimizer_ssl/spectra/effective_rank.py      # spectral entropy, soft rank, hard rank
optimizer_ssl/spectra/frequency_metrics.py   # HEAD/MID/TAIL buckets and TII helpers
optimizer_ssl/spectra/spectral_asymmetry.py  # asymmetry helpers
optimizer_ssl/spectra/tracker.py             # training-time hooks and logging
```

`optimizer_ssl/spectra/eigen_metrics_gpt.py` is retained only as a compatibility shim. New code should import from `optimizer_ssl.spectra` or the smaller modules directly.

## Standalone diagnostic API

For model-agnostic use, the metric code can be called directly:

```python
from optimizer_ssl.probe import spectral_rank
metrics = spectral_rank(activations)
```

This path is CPU-safe and does not require the full training stack or Triton. See
`docs/diagnostic_api.md`.
