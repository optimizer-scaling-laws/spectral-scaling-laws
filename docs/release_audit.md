# Release audit notes

This page records the current public-release boundary for the repository. It is intended to make clear what can be reproduced directly from the committed files and what requires external raw logs.

## Reproduction tiers

| Tier | What is committed | What can be reproduced directly |
|---|---|---|
| Processed-CSV figure reproduction | Compact CSVs under `results/processed/` and plotting scripts under `scripts/analysis/` | All committed PDF figures under `results/figures/` |
| Main raw-log parser path | Sample logs, schema normalizer, parser, aggregation code, and `run_metadata.csv` manifest schema | Main 160M global and HEAD/MID/TAIL scaling CSVs, if the external raw logs are supplied |
| Special figure families | Processed CSVs plus full launch-config grids for Dion rank sweep, matched-loss, 350M TAIL, and architecture-vs-optimizer | Their committed PDFs; submitted raw logs remain external |
| Training launch examples | Configs and shell wrappers | New runs with the released code; submitted historical seeds/logs are not fully bundled |

## Figure coverage

The source of truth for figure provenance is:

```text
results/figure_manifest.csv
```

It lists each committed PDF figure, the processed inputs it consumes, the reproduction command, and whether raw-log reproduction is fully represented in this repository or requires external artifacts.

## Current release choices

- The repo tracks publication-quality PDF figures only; PNG previews are intentionally not committed.
- Empty placeholder directories were removed. New folders should be added only when they contain runnable code, data, docs, or examples.
- Full raw logs, checkpoints, and large intermediate per-layer/per-step CSVs are external artifacts.
- The architecture-vs-optimizer figure is regenerated from processed beta values, and the full 80-run 12-head/6-head launch-config grid is committed. Submitted raw logs remain external.
- Dion rank-sweep, matched-loss, GPT2-350M TAIL, and architecture-vs-optimizer figure families are reproducible from processed CSVs in this repo and now have full launch-config grid coverage. Rebuilding their processed CSVs from submitted logs still requires the external raw-log bundles.
- Historical submitted runs should use `seed=not_recorded` in metadata. Released launch configs include `seed: 1337` for future reproducibility.

## Before adding another result family

For each new result family, add or update:

1. processed CSVs under `results/processed/`,
2. one plotting entrypoint under `scripts/analysis/`,
3. one focused wrapper under `scripts/reproduce/`,
4. one row per PDF in `results/figure_manifest.csv`,
5. documentation in `docs/analysis_inventory.md` and `results/README.md`,
6. tests that verify required artifacts exist and regenerate from processed CSVs.
