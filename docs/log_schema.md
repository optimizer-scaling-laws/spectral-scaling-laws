# Spectral telemetry log schema

The public codebase logs paper-facing names:

```text
soft_rank_pre
soft_rank_post
hard_rank_pre
hard_rank_post
spectral_entropy_pre
spectral_entropy_post
```

Historical submitted-run logs may use the older schema inherited from earlier
projects:

```text
SE_pre
SE_post
PR_pre
PR_post
EEE_pre
EEE_post
JS
```

Analysis code must normalize these fields before fitting scaling laws:

| Legacy field | Normalized field |
| --- | --- |
| `SE_pre` | `spectral_entropy_pre` |
| `SE_post` | `spectral_entropy_post` |
| `exp(SE_pre)` | `soft_rank_pre` |
| `exp(SE_post)` | `soft_rank_post` |
| `PR_pre` | `hard_rank_pre` |
| `PR_post` | `hard_rank_post` |
| `EEE_*`, `JS` | ignored for this paper repo |

The helper `optimizer_ssl.analysis.log_schema.parse_layer_metric_line` performs
this normalization. Processed CSVs should use only the normalized vocabulary.
