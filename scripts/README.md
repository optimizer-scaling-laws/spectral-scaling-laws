# Scripts

Executable entrypoints live here. Prefer the Makefile targets for common workflows; use the script paths below when you want a focused artifact family.

| Goal | Command / entrypoint | Output |
|---|---|---|
| Regenerate all committed figures | `make figures` | All PDFs in `results/figures/` |
| Regenerate all figures directly | `bash scripts/reproduce/reproduce_main_results_from_processed.sh results/processed results/figures` | All committed PDF figures |
| Rebuild main 160M processed CSVs from raw logs | `bash scripts/reproduce/reproduce_main_results_from_logs.sh results/processed/run_metadata.csv results/processed` | Main global and HEAD/MID/TAIL scaling CSVs and beta tables |
| Reproduce Dion rank sweep | `bash scripts/reproduce/reproduce_dion_rank_sweep.sh results/processed results/figures` | `dion_tail_*_rank_sweep.pdf` |
| Reproduce matched-loss figures | `bash scripts/reproduce/reproduce_matched_loss.sh results/processed results/figures` | Matched-loss / extended-AdamW PDFs |
| Reproduce GPT2-350M TAIL figures | `bash scripts/reproduce/reproduce_350m_tail.sh results/processed results/figures` | 350M TAIL hard/soft PDFs |
| Reproduce architecture-vs-optimizer figure | `bash scripts/reproduce/reproduce_architecture_vs_optimizer.sh results/processed results/figures` | `architecture_vs_optimizer.pdf` |
| Launch paper-scale runs | See `docs/training.md` | Outputs under `outputs/` |
| Recompute token-frequency buckets | `bash scripts/preprocess/prepare_fineweb10b_token_buckets.sh` | Token-frequency artifact |
| Validate released token buckets | `python scripts/validation/audit_token_frequency_buckets.py --help` | Bucket-audit report |

For figure provenance, see `results/figure_manifest.csv`. For the reproduction ladder and raw-log boundary, see `docs/reproduction.md`.
