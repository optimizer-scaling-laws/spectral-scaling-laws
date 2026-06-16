# Troubleshooting

## `Could not find any files matching data/fineweb10B/fineweb_train_*.bin`

Download shards first:

```bash
bash scripts/preprocess/prepare_fineweb10b_token_buckets.sh
```

For a smoke test, use:

```bash
bash scripts/train/train_tiny_debug.sh
```

## CUDA out of memory

Try reducing `device_batch_size`, running a smaller width, or setting:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The launch scripts already set this by default unless you override it.

## WandB prompts or authentication errors

Public configs set:

```yaml
no_wandb: true
```

If you enable WandB manually, set `WANDB_API_KEY` and `WANDB_HOST` as needed.

## Triton compilation delay

Muon-family optimizers may compile Triton kernels on first use. For debugging, use AdamW or set `no_triton: true` in a debug config.

## NCCL or distributed initialization errors

Make sure you launch through `torchrun` or the provided scripts, not plain `python optimizer_ssl/train/train_lm.py`. Also check that `NPROC_PER_NODE` matches the number of visible GPUs.

## Token-frequency audit mismatch

Check that the frequency table was computed from the same shard set you are auditing. The released `token_frequencies.npy` corresponds to the FineWeb10B tokenized shard set used in the paper experiments.

## Installing only the metrics

If `pip install -e .` fails on a CPU-only or macOS machine because of training
extras such as Triton, install the lightweight metrics path instead:

```bash
pip install -e ".[metrics]"
python examples/probe_synthetic_activations.py
```

Full multi-GPU training still requires the training extras and a CUDA-capable
PyTorch environment.
