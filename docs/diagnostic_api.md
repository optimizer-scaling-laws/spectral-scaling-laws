# Standalone spectral diagnostic API

The training tracker is useful for paper-scale runs, but the spectral-rank
metrics can also be used independently on activations from any model.

Install the lightweight metrics path:

```bash
pip install -e ".[metrics]"
```

Compute metrics from a tensor:

```python
import torch
from optimizer_ssl.probe import spectral_rank

acts = torch.randn(256, 128)       # [num_samples, hidden_dim]
metrics = spectral_rank(acts)
print(metrics["soft_rank"], metrics["hard_rank"])
```

Attach to an arbitrary PyTorch module:

```python
from optimizer_ssl.probe import attach_spectral_probe

probe = attach_spectral_probe(model.some_module, capture="output")
_ = model(batch)
metrics = probe.compute()
probe.close()
```

CPU-safe examples:

```bash
python examples/probe_synthetic_activations.py
python examples/probe_your_model.py
```

Frequency-bucketed diagnostics are also supported when activation samples align
with token IDs and a token-frequency table is supplied:

```python
metrics = spectral_rank(
    activations,
    token_ids=input_ids,
    token_freq_file="results/processed/token_frequencies.npy",
)
```

For public downloads, `token_frequencies.npy` is preferred. The `.pt` artifact is
retained for compatibility with earlier training scripts.
