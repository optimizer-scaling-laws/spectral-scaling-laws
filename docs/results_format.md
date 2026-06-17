# Results format

The repository separates raw logs, large intermediate tables, compact processed artifacts, and final figures.

## Raw logs

Raw eigen telemetry logs are text files produced during training. The paper's legacy logs use names such as `SE_post` and `PR_post`; released-code logs use the cleaner `spectral_entropy`, `soft_rank`, and `hard_rank` vocabulary in normalized CSVs. Raw logs are parsed through `optimizer_ssl.analysis.log_schema` before analysis.

Full raw logs can be large and should normally be distributed as external artifacts. The repository includes only small examples under `results/sample_logs/`.

## Run manifest

`results/processed/run_metadata.csv` describes the processed main-sweep runs and supplies metadata needed for raw-log parsing when external logs are available. A template for rebuilding from raw logs is provided at `results/processed/run_metadata_template.csv`.

Important columns include:

```text
run_id, paper_experiment, model_scale, model_name, num_layers, base_ffn_dim,
width_multiplier, ffn_hidden_dim, optimizer, optimizer_folder,
optimizer_variant, dion_rank_fraction, num_gpus, seed,
frequency_bucket_reduction, log_dir
```

For the paper's original logs whose seed was not recorded, use `seed=not_recorded`. For frequency-bucket logs from the paper, use `frequency_bucket_reduction=rank0_local`.

In the public processed artifact, `log_dir` uses an `external://...` placeholder rather than a local machine path.

## Normalized layer metrics

`layer_metrics.csv` has one row per run, layer, step, site, and bucket. It is a large intermediate artifact and is not committed by default. It can be regenerated from external raw logs for the main parser path.

Important columns:

```text
run_id, step, layer, bucket, site, n_tokens,
spectral_entropy, soft_rank, hard_rank,
source_log_schema, source_path,
<run metadata columns>
```

`bucket=global` corresponds to pooled all-token logs. `bucket=head`, `mid`, and `tail` correspond to frequency-bucketed telemetry.

## Scaling points

`global_rank_scaling_points.csv` and `frequency_bucket_rank_scaling_points.csv` contain one row per run/bucket/metric after the paper's aggregation rule:

1. take the final five checkpoints per layer;
2. take the median over those checkpoints for each layer;
3. take the median over layers;
4. use one standard deviation over layer medians as the error band.

Important columns:

```text
run_id, model_scale, optimizer, optimizer_folder, optimizer_variant,
width_multiplier, ffn_hidden_dim, bucket, metric, value, err_low, err_high,
n_layers_used, final_samples, aggregation, frequency_bucket_reduction
```

## Beta tables

`main_beta_table.csv` and `frequency_bucket_beta_table.csv` fit `metric = A * D^beta` in log-log space over FFN hidden dimension `D`.

Important columns:

```text
model_scale, bucket, metric, optimizer, optimizer_folder, optimizer_variant,
beta, intercept, r_squared, beta_lower, beta_upper, n_widths, ci_method
```

The confidence interval is an ordinary least-squares log-log t-interval over width points. It is **not** a multi-seed confidence interval unless the input manifest explicitly contains multiple seeds and the analysis is extended to model that source of variation.

## Figure manifest

`results/figure_manifest.csv` lists every committed PDF figure, its processed inputs, reproduction command, raw-log coverage level, config coverage, and paper role.

Use this manifest to audit whether a result is:

- reproducible directly from committed processed CSVs,
- rebuildable from raw logs using the main parser path,
- or released with committed processed data and full launch configs while the paper's raw logs remain external.

## Figures

Only publication-quality PDFs are committed:

```text
results/figures/global_hard_rank_scaling.pdf
results/figures/global_soft_rank_scaling.pdf
results/figures/frequency_bucket_rank_grid.pdf
results/figures/dion_tail_hard_rank_sweep.pdf
results/figures/dion_tail_soft_rank_sweep.pdf
results/figures/matched_loss_scaling_breakdown.pdf
results/figures/matched_loss_beta_dynamics.pdf
results/figures/matched_loss_pr_trajectories_by_width.pdf
results/figures/tail_350m_hard_rank_scaling.pdf
results/figures/tail_350m_soft_rank_scaling.pdf
results/figures/architecture_vs_optimizer.pdf
```

PNG previews are intentionally not tracked.

## Dion rank-sweep processed tables

The Dion TAIL-token rank-sweep figure uses:

```text
results/processed/dion_tail_rank_sweep_points.csv
results/processed/dion_tail_rank_sweep_beta_table.csv
```

These files use the same point/beta schema as the main rank-scaling tables, with additional Dion-rank metadata columns: `dion_rank_fraction`, `rank_fraction_float`, and `line_style`. AdamW is retained as the dashed baseline reference.

## Matched-loss processed CSVs

The full matched-loss launch-config grid is committed under `configs/paper_runs/matched_loss/160m/`. The processed CSVs below remain the source of truth for committed figure regeneration.

`matched_loss_beta_dynamics.csv` contains one row per bucket and logged step for the extended AdamW run, with `beta_hard` and `beta_soft` computed from the cross-width scaling fit at that step.

`matched_loss_pr_trajectories.csv` contains hard-rank trajectories for selected widths used to visualize where width-capacity ordering breaks.

`matched_loss_terminal_beta_table.csv` contains terminal beta fits for the matched-loss comparison runs. These are fit/regression intervals, not seed intervals.

## GPT2-350M TAIL-token processed CSVs

The 350M confirmation plots use only TAIL-token bucket metrics and four width points. The processed files are:

```text
results/processed/tail_350m_rank_scaling_points.csv
results/processed/tail_350m_beta_table.csv
results/processed/tail_350m_run_metadata.csv
results/processed/tail_350m_summary.json
```

These files follow the same point/beta schema as the Dion TAIL rank-sweep artifacts, but with `model_scale=350m`, `num_layers=24`, `base_ffn_dim=1024`, and width multipliers 1--4.

## Architecture-vs-optimizer processed CSVs

The architecture-vs-optimizer figure is regenerated from processed beta values, and the corresponding full 12-head/6-head launch-config grid is committed under `configs/paper_runs/architecture_vs_optimizer/160m/`.

```text
results/processed/architecture_vs_optimizer_beta_values.csv
results/processed/architecture_vs_optimizer_comparison.csv
```

`architecture_vs_optimizer_beta_values.csv` contains one row per head count, bucket, rank metric, and optimizer. `architecture_vs_optimizer_comparison.csv` contains the derived quantities plotted in the figure: the absolute architectural beta shift `|beta_6h - beta_12h|` and the optimizer-induced reference gain `max_o beta_{o,12h} - beta_{AdamW,12h}`.
