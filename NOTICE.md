# Notices

Optimizer-Induced Spectral Scaling Laws is released by Nandan Kumar Jha and Brandon Reagen.

This repository includes modified and vendored components derived from the Dion optimizer codebase under `third_party/dion/`.

See `third_party/dion/README.md` and `third_party/dion/NOTICE.md` for upstream documentation, citations, and third-party notices.

## Model and data-loading utilities

`optimizer_ssl/models/gpt_model.py` and `optimizer_ssl/models/gpt_utils.py` contain GPT model and data-loading utilities adapted from nanoGPT/llm.c-style training code and modified for the Optimizer-SSL experiments. The repository-level MIT license applies to the modifications in this release. Upstream attribution is preserved here so readers can distinguish the paper-specific spectral telemetry code from the base GPT training utilities.

## Vendored optimizer code

`third_party/dion/` contains vendored and lightly adapted Dion/Muon/NorMuon optimizer implementations. See `third_party/dion/README.md`, `third_party/dion/LICENSE`, and `third_party/dion/NOTICE.md` for upstream documentation and notices.
