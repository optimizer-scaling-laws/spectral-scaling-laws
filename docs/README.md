# Documentation

A guide to the docs in this directory, grouped by what you're trying to do.

**Understand the work**
| Doc | Covers |
|---|---|
| [paper_summary.md](paper_summary.md) | One-paragraph overview of the paper and its central claim |
| [method.md](method.md) | The measurement pipeline, end to end |
| [metrics.md](metrics.md) | Soft / hard spectral rank and the HEAD/MID/TAIL frequency buckets |

**Run it**
| Doc | Covers |
|---|---|
| [getting_started.md](getting_started.md) | Fresh clone → install → tests → tiny smoke run |
| [training.md](training.md) | Full launch matrix for every experiment grid |
| [data.md](data.md) | FineWeb10B preparation and token-frequency buckets |
| [diagnostic_api.md](diagnostic_api.md) | Running the spectral-rank diagnostic on your own model |

**Reproduce the results**
| Doc | Covers |
|---|---|
| [reproduction.md](reproduction.md) | Regenerating figures from processed CSVs and from raw logs |
| [analysis_inventory.md](analysis_inventory.md) | Figure ↔ script ↔ processed-CSV map |
| [results_format.md](results_format.md) | Schema of the processed CSV artifacts |
| [reproducibility.md](reproducibility.md) | Seeds, reduction modes, and interval semantics |

**Scope, compute, and provenance**
| Doc | Covers |
|---|---|
| [compute_budget.md](compute_budget.md) | Model sizes, run grids, and hardware |
| [release_audit.md](release_audit.md) | What reproduces from this repo vs. external artifacts |
| [optimizer_hyperparameters.md](optimizer_hyperparameters.md) | Optimizer hyperparameter surface used by the frozen configs |
| [optimizer_implementations.md](optimizer_implementations.md) | Vendored Dion / Muon / NorMuon attribution |
| [adding_new_optimizer.md](adding_new_optimizer.md) | Extending the registry with a new optimizer |

**Reference**
| Doc | Covers |
|---|---|
| [repo_map.md](repo_map.md) | Per-file guide to the repository |
| [troubleshooting.md](troubleshooting.md) | Common issues |
| [faq.md](faq.md) | Frequently asked questions |
