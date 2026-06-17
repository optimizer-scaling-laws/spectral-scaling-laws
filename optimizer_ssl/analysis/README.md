# `optimizer_ssl.analysis`

This package contains torch-free analysis utilities for converting paper-run
telemetry logs into processed CSVs and regenerating paper-style figures from
those processed artifacts.

Current modules:

- `log_schema.py`: maps legacy `SE`/`PR` log names to the public soft/hard-rank vocabulary.
- `eigen_log_parser.py`: parses global and HEAD/MID/TAIL layer logs.
- `rank_aggregation.py`: implements the legacy plotting aggregation rule.
- `scaling_fits.py`: fits log-log power laws and beta intervals over width points.
- `rank_scaling_figures.py`: generates global, bucket, Dion rank-sweep, and GPT2-350M TAIL rank-scaling figures.
- `matched_loss_figures.py`: generates the matched-loss / extended-AdamW breakdown figures.

The plotting utilities consume processed CSVs under `results/processed/`; raw
paper raw logs remain external artifacts except for tiny parser fixtures.
