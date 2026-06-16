# Figures

Generated paper-style figures from processed CSV artifacts.

Regenerate all committed PDFs from processed CSVs with:

```bash
make figures
```

or directly:

```bash
bash scripts/reproduce/reproduce_main_results_from_processed.sh \
  results/processed \
  results/figures
```

Current outputs:

```text
global_hard_rank_scaling.pdf
global_soft_rank_scaling.pdf
frequency_bucket_rank_grid.pdf
dion_tail_hard_rank_sweep.pdf
dion_tail_soft_rank_sweep.pdf
matched_loss_scaling_breakdown.pdf
matched_loss_beta_dynamics.pdf
matched_loss_pr_trajectories_by_width.pdf
tail_350m_hard_rank_scaling.pdf
tail_350m_soft_rank_scaling.pdf
architecture_vs_optimizer.pdf
```

Only publication-quality PDF figures are tracked in this repository. PNG previews are intentionally omitted.

For provenance and reproduction-status details for each PDF, see:

```text
results/figure_manifest.csv
```

Notes:

- The matched-loss figure family explains why extended AdamW training can improve loss while degrading hard-rank width scaling.
- The GPT2-350M figures reproduce the main-paper TAIL-token hard/soft rank scaling confirmation with four width points and the optimizer subset used in that experiment.
- The architecture-vs-optimizer figure compares processed beta values from 12-head and 6-head runs; raw ablation logs are not bundled in the repository.
