# External artifacts

Full raw logs, checkpoints, and large intermediate CSVs are not committed to this repository. The repo includes compact processed CSVs sufficient to regenerate all committed PDF figures.

Recommended external artifacts:

| Artifact | Contents | Needed for |
|---|---|---|
| submitted_main_160m_raw_eigen_logs | Full `eigen_metrics_logs/` directories for main 160M width-sweep runs | Rebuilding `layer_metrics.csv`, main scaling points, and main beta tables from raw logs |
| submitted_dion_rank_sweep_logs | Full `eigen_metrics_logs/` directories for AdamW and Dion rank fractions | Rebuilding Dion TAIL rank-sweep processed CSVs from raw logs |
| submitted_matched_loss_logs | Full `eigen_metrics_logs/` directories for AdamW 6K/12K and Dion comparison runs | Rebuilding matched-loss processed CSVs from raw logs |
| submitted_350m_tail_logs | Full `eigen_metrics_logs/` directories for GPT2-350M TAIL-token runs | Rebuilding 350M TAIL processed CSVs from raw logs |
| submitted_architecture_ablation_logs | Full 12-head and 6-head raw logs across optimizers | Rebuilding architecture-vs-optimizer beta values from raw logs |
| layer_metrics.csv.gz | Normalized per-layer/per-step metrics for main parser path | Recomputing scaling-point CSVs without reparsing raw logs |

When hosting externally, record the URL, file size, checksum, and date here.

## Placeholder table

| Artifact | URL | Size | SHA256 | Notes |
|---|---|---:|---|---|
| submitted_main_160m_raw_eigen_logs | TBD | TBD | TBD | External raw-log bundle |
| submitted_dion_rank_sweep_logs | TBD | TBD | TBD | External raw-log bundle |
| submitted_matched_loss_logs | TBD | TBD | TBD | External raw-log bundle |
| submitted_350m_tail_logs | TBD | TBD | TBD | External raw-log bundle |
| submitted_architecture_ablation_logs | TBD | TBD | TBD | External raw-log bundle |
