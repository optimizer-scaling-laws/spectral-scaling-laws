# Method

Pipeline:

1. Train GPT-style language models under matched architecture/data settings.
2. Log FFN activation covariance spectra during training.
3. Compute pooled spectral ranks from globally reduced covariance statistics; compute token-frequency bucket ranks in the configured reduction mode (`rank0_local` for the paper's configs, `distributed_covariance` for new global bucket telemetry).
4. Fit width-scaling laws to spectral metrics.
5. Compare realized representation capacity across optimizers.
