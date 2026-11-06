# Results

## Forecast accuracy across neural backbones

EKI improves one-step accuracy across all four tested neural architectures. Metrics below give each of the 15 held-out RIOHTrack structures equal weight, using 2,089 dated targets and five-seed ensembles.

| Backbone | Direct depth | Anchored | EKI | RMSE reduction versus direct depth |
|---|---:|---:|---:|---:|
| MLP | 0.56436 | 0.31186 | **0.28504** | **49.49%** |
| GRU | 0.68053 | 0.33026 | **0.29049** | **57.31%** |
| LSTM | 0.68601 | 0.34299 | **0.29373** | **57.18%** |
| Transformer | 0.67366 | 0.30066 | **0.28854** | **57.17%** |

RMSE is in millimeters. EKI has lower MSE than its matched direct-depth baseline on 14 of 15 structures for every backbone. Direct-depth contrasts have Holm-adjusted reference p values of 0.03720–0.03960. The 2,164-target sensitivity analysis preserves the ordering. [Main table](supplements/direct-depth/results/tables/three_level_main_table.csv) · [Paired differences and intervals](supplements/direct-depth/results/tables/supplementary_primary_contrasts.csv)

Relative to anchoring alone, EKI reduces RMSE by 8.60%, 12.04%, 14.36% and 4.03% for MLP, GRU, LSTM and Transformer. P1 supports the joint four-backbone comparison (Holm-adjusted p = 0.034). Anchored XGBoost and monotonic anchored XGBoost achieve 0.28947 and 0.29032 mm, compared with persistence at 0.41324 mm. [Complete configuration metrics](supplements/modules/tables/summary.csv) · [P1 and P2](results/tables/primary_conjunctions.csv)

Depth agreement alone conceals differences in predicting the next change. Direct-depth neural models attain depth $R^2$ of 0.908–0.938 while their increment $R^2$ is negative. EKI reaches increment $R^2$ of 0.488–0.517, alongside depth $R^2$ of 0.983–0.984.

## Complementary module contributions

The four-backbone ablations separate accuracy, conditional response direction and interval calibration.

| Module or comparison | Finding |
|---|---|
| OA | Largest accuracy contribution; anchored versus direct-depth skill improves by 1.296–2.128 across backbones |
| OA within EKI | Retaining the offset improves skill by 0.134–0.211 over No offset |
| LME versus Flat | Positive skill estimates for all four backbones; the LSTM interval includes zero |
| MLT versus Free path | Accuracy differences have intervals spanning zero; MLT eliminates decreasing exposure responses |
| Temperature modulation | No heat contrasts span zero for every backbone |
| Hard constraints versus Soft | EKI has zero violations; Soft retains 0.1–1.1%, with skill intervals spanning zero |

LME skill differences [descriptive 95% interval] are 0.030 [0.005, 0.065], 0.041 [0.007, 0.082], 0.038 [−0.044, 0.114] and 0.036 [0.001, 0.080] for MLP, GRU, LSTM and Transformer. The structural embedding visualization uses PCA fitted on fold-1 training structures with the MLP seed-42 model. [All contrasts](supplements/modules/tables/contrasts.csv) · [Structure metrics](supplements/modules/tables/by_structure.csv) · [Embedding coordinates](supplements/modules/tables/embedding.csv)

The sign constraints have an architecture-dependent accuracy tradeoff. P2 does not establish non-inferiority to No signs jointly across the four backbones at the 0.03 skill margin (p = 0.274). These results distinguish the common response guarantee from the accuracy contribution of each component.

## Response consistency and interval calibration

The architectural response guarantee is confirmed by **zero violations in 125,340 individual-fit checks**. Ensemble violation rates are 4.1–15.4% for direct-depth models, 8.3–21.1% for anchored models and 5.5–20.5% for Free path. Anchoring improves accuracy but increases decreasing responses in every RIOHTrack backbone; the constrained pathway supplies the missing response property.

Monotonic anchored XGBoost also has zero violations, compared with 28.7% for the unconstrained anchored tree. Its RMSE changes from 0.28947 to 0.29032 mm. Both approaches provide consistent response direction, with different sensitivity to exposure. [Module and tree audits](supplements/modules/tables/summary.csv) · [Individual-fit EKI audit](results/tables/response_audit.csv)

EKI mean one-step coverage is 95.0–95.3%, with MPIW 1.20–1.27 mm and MIS95 1.62–1.70 mm. Direct-depth models have coverage of 84.4–90.5%, widths of 2.01–2.30 mm and MIS95 of 3.25–4.18 mm. Anchored models already attain near-95% mean coverage. Worst-structure EKI coverage is 81.0–90.5% at one step and 54.6–78.7% at four steps. [Interval summaries](supplements/direct-depth/results/tables/interval_summary.csv) · [Per-structure coverage](supplements/direct-depth/results/tables/interval_structures.csv)

## Monitoring conditions

| Condition | EKI result | Relevant comparison |
|---|---|---|
| Four steps, prior-year calendar temperatures | RMSE 0.424–0.449 mm | Recent-temperature continuation 0.470–0.493 mm; persistence approximately 0.543 mm; anchored XGBoost approximately 0.404 mm |
| One missing inspection | RMSE 0.434–0.437 mm | Anchored neural models 0.401–0.429 mm; anchored XGBoost approximately 0.396 mm |
| Late loading | RMSE 0.199–0.210 mm | Anchored XGBoost approximately 0.216 mm |
| Pavement-class holdout | LME benefit varies with class and backbone | The contribution of structural encoding depends on the configurations represented during fitting |

The temperature scenarios use the same supplied future loads and 1,624 common prediction origins at each horizon. Missing-inspection evaluation uses 2,059 targets and late-loading evaluation 494. [Scenario tables](supplements/direct-depth/results/tables/scenario_summary.csv) · [Persistence](results/tables/scenario_persistence.csv) · [Late-loading tables](supplements/direct-depth/results/tables/late_load_summary.csv) · [Class-holdout tables](results/tables/class_summary.csv)

## Independent LTPP evaluation

On 63 targets from 26 sections in seven held-out states, locally fitted EKI reduces RMSE by **29.0–39.1%** relative to direct-depth neural models. All four EKI backbones outperform persistence. Anchored models have slightly lower RMSE than EKI, by 0.0007–0.0166 mm. EKI retains zero violations in 756 ensemble checks, whereas anchored models decrease in 144 checks (19.0%). Local fitting preserves anchoring's accuracy benefit and MLT's response consistency without the extra accuracy beyond OA seen on RIOHTrack.

Frozen RIOHTrack weights are evaluated on a separate cohort of **45 targets from 12 sections**. All four EKI models have higher RMSE than persistence, and 41 intervals exceed the source training duration range. This evaluates weight transfer on a different cohort, rather than a paired effect of local retraining. [Full LTPP results, cohorts and data sources](supplements/ltpp/README.md)

## Interpretation and application conditions

The results support reuse of the three-module design across the tested neural architectures. OA contributes most of the accuracy gain, LME represents structural composition, and MLT enforces the direction of current-exposure effects. They need not improve the same metric. Additional accuracy beyond OA depends on the monitoring setting, as shown by missed inspections and locally fitted LTPP models.

The direct-depth neural baselines are weaker than persistence under the matched budget. Percentage improvements describe those fitted configurations. RIOHTrack structures share weather and loading, and the LTPP cohort has one geographic split. The fixed temperature hinge has no consistently separable accuracy benefit in the ablations. The learned terms and embeddings do not identify layer-specific deformation mechanisms.

EKI contains approximately 17k–72k parameters. Its measured inference time is higher than the anchored counterpart despite similar parameter counts. Attribution is more stable across seeds under a fixed background, but no general improvement is established when the background changes. [Computational cost](results/tables/cost_summary.csv) · [Attribution stability](results/tables/attribution_background_stability.csv) · [Inference scope](PROTOCOL.md#study-status-and-inferential-scope)
