# Paper-to-code guide

Companion material for *Reusable engineering-knowledge modules for reliable neural rutting forecasting*.

## Method and terminology

| Paper content | Repository entry |
|---|---|
| Study overview and datasets (Section 2; Figs. 1–2) | [Protocol](PROTOCOL.md), [RIOHTrack inputs](data/), [LTPP cohorts and sources](supplements/ltpp/README.md) |
| EKI architecture and OA/LME/MLT (Section 3; Figs. 3–4) | [Module equations](PROTOCOL.md#architecture), [core model](models.py), [input transformations](data.py) |
| Comparison names and stored arm IDs | [Comparator mapping](PROTOCOL.md#comparators-and-module-ablations) |
| Training, metrics and inference (Section 4.1) | [Protocol](PROTOCOL.md), [evaluation code](evaluation.py), [statistical analysis](analysis.py) |

## Results and supporting tables

| Manuscript figure or table | Supporting files |
|---|---|
| Table 1 and Fig. 5 — one-step comparison | [Direct / anchored / EKI table](supplements/direct-depth/results/tables/three_level_main_table.csv), [all module and constrained-tree metrics](supplements/modules/tables/summary.csv) |
| Fig. 6 and Fig. S1 — depth and increment agreement | [Metrics](supplements/direct-depth/results/tables/one_step_summary.csv), saved predictions in the [verified releases](REPRODUCTION.md) |
| Fig. 7 — per-structure skill | [Structure results](supplements/modules/tables/by_structure.csv) |
| Fig. 8 — inspection trajectories | [Prediction and interval records](REPRODUCTION.md), [structure-level intervals](supplements/direct-depth/results/tables/interval_structures.csv) |
| Fig. 9 and Tables S1–S2 — module effects | [Summary](supplements/modules/tables/summary.csv), [paired contrasts](supplements/modules/tables/contrasts.csv), [arm definitions](supplements/modules/README.md) |
| Fig. 10 — LME embedding | [PCA coordinates](supplements/modules/tables/embedding.csv) |
| Fig. 11 — exposure responses | [Individual-fit audit](results/tables/response_audit.csv), [ensemble audit metrics](supplements/modules/tables/summary.csv) |
| Fig. 12 and Figs. S2–S3 — monitoring conditions | [Scenario summaries](supplements/direct-depth/results/tables/scenario_summary.csv), [persistence](results/tables/scenario_persistence.csv), [class transfer](results/tables/class_summary.csv) |
| Fig. 13 — LTPP local fitting and weight transfer | [Local metrics](supplements/ltpp/tables/local_metrics.csv), [transfer metrics](supplements/ltpp/tables/transfer_metrics.csv), [exposure checks](supplements/ltpp/tables/shape_backbones.csv) |
| Fig. 14 and Table 2 — cross-dataset comparison | [RIOHTrack](supplements/modules/tables/summary.csv), [LTPP](supplements/ltpp/tables/local_metrics.csv), [LTPP perturbations](supplements/ltpp/tables/shape_perturbations.csv) |
| Fig. S4 — prediction intervals | [Interval summaries](supplements/direct-depth/results/tables/interval_summary.csv), [per-structure coverage](supplements/direct-depth/results/tables/interval_structures.csv) |
| Fig. S5 — ablation intervals | [Original component contrasts](results/tables/descriptive_components_tree.csv), [complete four-backbone contrasts](supplements/modules/tables/contrasts.csv) |

Figure numbers here follow the manuscript. Files under `figures/` retain experiment-local names and do not necessarily share the manuscript numbering. The links above locate the supporting evidence; repository plot exports are not a replacement for the final manuscript artwork.

[Read the results](RESULTS.md) · [Download and reproduce](REPRODUCTION.md)
