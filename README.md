# Turning generic neural learners into pavement experts: A backbone-agnostic engineering-knowledge interface for rutting prediction

**EKI** embeds observation referencing, layer–material semantics, and constrained
load–temperature responses into MLP, GRU, LSTM, and Transformer learners. The same
module implementation is reused across backbones; each model learns its own weights.

## Main findings

| Backbone | Ordinary residual RMSE (mm) | EKI RMSE (mm) | RMSE reduction |
|---|---:|---:|---:|
| MLP | 0.31186 | 0.28504 | 8.60% |
| GRU | 0.33026 | 0.29049 | 12.04% |
| LSTM | 0.34299 | 0.29373 | 14.36% |
| Transformer | 0.30066 | 0.28854 | 4.03% |

These are structure-balanced results on 2,089 dated targets from 15 held-out structures.
Five seeds are averaged in millimetres before evaluation. The residual XGBoost reference
has RMSE 0.28947 mm and persistence 0.41324 mm on the same targets.

- Cross-backbone superiority over ordinary residual networks met P1 (Holm p=0.0339966).
- All 125,340 fixed-context EKI response checks satisfied the specified monotonicity.
- Semantic structure ablations support useful representation on the two tested backbones.
- Historical calendar temperatures improve multi-step prediction relative to recent-temperature continuation.

The complete evaluation also measures trade-offs: P2 did not establish noninferiority
to unconstrained counterparts on all four backbones (Holm p=0.273873); the independent
benefit of current temperature modulation is uncertain; missing-inspection performance
decreases; class-transfer and explanation-background results vary. See [Results](RESULTS.md).
The title expresses domain specialization; it does not claim performance equal to human experts.

## Files and downloads

- [All experiment arms and scenarios](all_arms_and_scenarios.csv): 2,387 summary records.
- [31 result tables](results/tables/) and [11 figures in PNG/PDF](figures/).
- [Frozen numerical inputs](data/), [target manifest](target_manifest.csv),
  [run matrix](run_matrix.csv), [protocol](protocol.json), and model source in this repository.
- [Complete current-study package](https://github.com/Zerojy1/Turning-generic-neural-learners-into-pavement-experts/releases/download/v1.0.0/eki-latest-experiment.zip):
  all 605 fitted models, inner candidates, learning curves, seed predictions, scenario
  outputs, calibration data, attribution coalitions, and temperature source records.
- [SHA-256 checksums](https://github.com/Zerojy1/Turning-generic-neural-learners-into-pavement-experts/releases/download/v1.0.0/SHA256SUMS.txt).

The large files are release assets rather than Git blobs. `FILE_MANIFEST.json` identifies
every distribution file, its checksum, and whether it is stored in Git or the complete asset.
All current arms, seeds and evaluation subsets are included.

## Reproduction

See [REPRODUCTION.md](REPRODUCTION.md). The package includes byte-identical numerical
artifacts and a documented packaging-only source adaptation ledger. No model was
retrained and no result was selected or changed during publication packaging.

## Evaluation scope

This is a single-site exploratory evaluation under structure-grouped validation.
External independent validation is not available. Temperature/load monotonicity is
conditional on fixed history and context; it is not causal identification or guaranteed
trajectory monotonicity. Prediction intervals use empirical inner-data calibration.
See [PROTOCOL.md](PROTOCOL.md) for assumptions, comparisons and validity boundaries.

![Cross-backbone results](figures/fig01_main.png)
