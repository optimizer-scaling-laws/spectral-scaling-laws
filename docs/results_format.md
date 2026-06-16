# Results format

Processed paper CSVs should use long-form tables with columns such as:

```text
model_scale, width_multiplier, optimizer_name, optimizer_variant, token_bucket, layer, metric, beta, intercept, r2, ci_lower, ci_upper, seed, num_points
```

Metric names should use the public vocabulary:

```text
soft_rank
hard_rank
spectral_asymmetry
tail_integrity_index
scaling_exponent_beta
```

The analysis integration pass will finalize exact CSV schemas for paper figures and tables.

## Seed metadata

Processed run metadata should include a `seed` field. For newly launched runs
from this release, configs default to `seed=1337`. For historical submitted logs
where the seed was not recorded, use `seed=not_recorded`; do not infer or
fabricate a seed value.
