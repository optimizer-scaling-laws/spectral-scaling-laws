# External artifacts

Full raw logs, checkpoints, and large intermediate CSVs are not committed to this repository. The repo includes compact processed CSVs sufficient to regenerate all committed PDF figures.

Recommended external artifacts:

| Artifact | Contents | Needed for |
|---|---|---|
| main_160m_raw_eigen_logs | Full `eigen_metrics_logs/` directories for main 160M width-sweep runs | Rebuilding `layer_metrics.csv`, main scaling points, and main beta tables from raw logs |
| dion_rank_sweep_logs | Full `eigen_metrics_logs/` directories for AdamW and Dion rank fractions | Rebuilding Dion TAIL rank-sweep processed CSVs from raw logs |
| matched_loss_logs | Full `eigen_metrics_logs/` directories for AdamW 6K/12K and Dion comparison runs | Rebuilding matched-loss processed CSVs from raw logs |
| 350m_tail_logs | Full `eigen_metrics_logs/` directories for GPT2-350M TAIL-token runs | Rebuilding 350M TAIL processed CSVs from raw logs |
| architecture_ablation_logs | Full 12-head and 6-head raw logs across optimizers | Rebuilding architecture-vs-optimizer beta values from raw logs |
| layer_metrics.csv.gz | Normalized per-layer/per-step metrics for main parser path | Recomputing scaling-point CSVs without reparsing raw logs |

When hosting externally, record the URL, file size, checksum, and date here.

## Future external-hosting record

External raw-log bundles are not currently hosted in this repository. If they are released later, record the URL, file size, SHA256 checksum, and release date in this file.
