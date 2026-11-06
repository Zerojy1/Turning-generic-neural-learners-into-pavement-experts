# Contributions of the reusable modules

This supplement provides the complete four-backbone comparison underlying the manuscript's module matrix, soft-constraint comparison and monotonic XGBoost reference. The evaluation uses the same 2,089 targets from 15 held-out RIOHTrack structures, five-seed ensembles and structure-balanced metrics.

| Manuscript label | Stored arm | Component examined |
|---|---|---|
| Direct depth | A0 | Generic neural reference |
| Anchored | A2 | OA on the unconstrained backbone |
| EKI | A1 | Complete OA + LME + MLT design |
| No offset | NOANCHOR | Measured-depth output offset |
| Flat | A7 | Layer–material encoding |
| Free path | FREEPATH | Routed exposure pathway |
| No signs | A3 | Hard rate and temperature sign constraints |
| No heat | A6 | Current temperature modulation |
| Soft | SOFT | Fixed monotonicity penalty |
| Anchored XGBoost | A4 | Tree reference |
| Monotonic anchored XGBoost | TREE_MONO | Constrained tree reference |
| Persistence | A5 | No-change reference |

## Numerical results

- [Summary for all 39 configurations](tables/summary.csv) contains accuracy, violation counts and magnitudes, mean and worst coverage, MPIW and MIS95.
- [37 paired contrasts](tables/contrasts.csv) report skill and RMSE differences with descriptive 95% intervals.
- [585 structure-level rows](tables/by_structure.csv) support independent recomputation of balanced accuracy and skill.
- [Embedding coordinates](tables/embedding.csv) describe the 19 structures in the prespecified MLP fold-1, seed-42 representation.

Positive `delta_skill` favors `arm` over `ref`; negative `RMSE_difference_mm` favors `arm`. Violation rates and coverage in the CSVs are proportions, not percentages. `checks` and `decreases` in the summary refer to the five-seed ensemble. They must not be pooled with the individual-fit check counts in the main release.

The four-backbone module extension is descriptive on previously examined data and does not replace P1/P2. Its frozen source revision is `4d556de52111bb8d0ac1d42d56fabd915afcb169`. The completed local verification covered 425 new fits and 950 model replays. [Source and file hashes](SOURCE.json)

These files are the verified numerical tables. The added module checkpoints are not part of the three existing public release packages. [Reproduction coverage](../../REPRODUCTION.md)

[Interpretation of the modules](../../RESULTS.md#complementary-module-contributions) · [Method and comparisons](../../PROTOCOL.md#comparators-and-module-ablations)
