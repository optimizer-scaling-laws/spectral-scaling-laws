# Results artifacts

This directory contains the lightweight artifacts needed to audit and regenerate the released figures. Large raw logs, checkpoints, and layer-level intermediate CSVs are intentionally external.

## Source of truth

- `figure_manifest.csv` is the per-figure provenance table. It records the committed PDF, processed inputs, reproduction command, launch-config coverage, and raw-log status.
- `processed/` contains compact CSV/NumPy artifacts used by the figure scripts. See [`docs/results_format.md`](../docs/results_format.md) for schemas.
- `figures/` contains the committed publication-quality PDF figures. PNG previews are intentionally omitted.
- `external_artifacts.md` is the place to record external raw-log or checkpoint bundles if they are released later.

## Reproduce committed figures

```bash
make figures
```

This regenerates all committed PDFs from the committed processed artifacts. The main 160M global and HEAD/MID/TAIL processed CSVs can also be rebuilt from raw logs if external raw logs are supplied. Dion rank-sweep, matched-loss, GPT2-350M TAIL, and architecture-vs-optimizer have full launch-config grid coverage, but the paper's raw logs remain external unless supplied separately.

## Large artifacts not included

The following are not committed: `layer_metrics.csv`, `matched_loss_layer_metrics.csv`, `tail_350m_layer_metrics.csv`, the paper's full raw logs, and checkpoints. They are large and can be regenerated or distributed as external artifacts.
