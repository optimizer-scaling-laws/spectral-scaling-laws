# Results artifacts

`results/processed/token_frequencies.npy` is the preferred public token-frequency artifact used to assign GPT-2 vocabulary tokens into HEAD/MID/TAIL occurrence-mass buckets.

`results/processed/token_frequencies.pt` is retained for compatibility with earlier training scripts and submitted-run code paths.

Raw training logs, checkpoints, and full eigenmetric logs are intentionally not committed. Training scripts write new outputs under `outputs/`.

After the paper-analysis scripts are integrated, processed CSVs and regenerated figures will be added here.
