# Reproducibility notes

## Seeds

Released configs include an explicit seed:

```yaml
seed: 1337
```

The training entrypoint calls `seed_everything(seed)` before model initialization, dataloader construction, and optimizer construction. The submitted paper used one training run per configuration rather than a multi-seed sweep. Therefore, analysis outputs should not describe confidence intervals as seed confidence intervals unless future multi-seed runs are added.

For historical submitted logs where the seed was not recorded explicitly, the analysis metadata should use `seed=not_recorded` rather than fabricating a value.

## Frequency-bucket reduction modes

Pooled all-token spectra are computed from covariance statistics reduced across all data-parallel ranks. Frequency-bucketed HEAD/MID/TAIL telemetry has two modes:

- `rank0_local`: rank-0 local bucket metrics, matching the submitted experiments.
- `distributed_covariance`: globally reduced bucket covariance statistics for new runs.

Paper configs use `rank0_local` to preserve submitted-run compatibility.

## Token-frequency artifact format

The preferred public token-frequency artifact is:

```text
results/processed/token_frequencies.npy
```

The older PyTorch `.pt` token-frequency artifact is intentionally not committed in this repository. Compatibility loading for `.pt` files remains in the code for users who already have older local artifacts; new public reproduction should use `.npy`.

## Figure and table intervals

The reported beta intervals in the processed beta tables are ordinary least-squares log-log fit intervals over width points. They are not multi-seed confidence intervals.

## External artifacts

Full submitted raw logs, checkpoints, and large intermediate per-layer/per-step CSVs are external artifacts. The committed repository is designed to reproduce figures from compact processed CSVs, and to parse raw logs when the external raw-log bundles are supplied.
