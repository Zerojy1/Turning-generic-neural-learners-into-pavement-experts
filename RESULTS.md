# Results and engineering interpretation

## 1. Does the complete framework improve generic learners?

On the 2,089-target primary set, full EKI improves the direct-depth versions of all
four tested backbones. Models use five-seed ensembles under structure-grouped validation.
RMSE is the square root of the equally weighted mean of structure MSEs.

| Backbone | Direct depth A0 | Observation residual A2 | Full EKI A1 | Reduction: A0 → A1 |
|---|---:|---:|---:|---:|
| MLP | 0.56436 | 0.31186 | **0.28504** | **49.49%** |
| GRU | 0.68053 | 0.33026 | **0.29049** | **57.31%** |
| LSTM | 0.68601 | 0.34299 | **0.29373** | **57.18%** |
| Transformer | 0.67366 | 0.30066 | **0.28854** | **57.17%** |

The four A1-versus-A0 comparisons meet the frozen supplementary criterion, with
Holm-adjusted reference p values of 0.03720–0.03960. Each backbone has lower error on
14/15 structures. The 2,164-target sensitivity set preserves the ordering.

Direct baselines receive the same available observation, exposure and structure information.
They are weaker than persistence in this budget; these results establish gains for the
tested configurations rather than the best attainable performance of all direct models.
The complete comparison does not identify each module's separate causal contribution.

[Main table](supplements/direct-depth/results/tables/three_level_main_table.csv) · [All one-step metrics](supplements/direct-depth/results/tables/one_step_summary.csv) · [Paired contrasts](supplements/direct-depth/results/tables/supplementary_primary_contrasts.csv)

## 2. What remains after observation referencing?

A2 already incorporates the observation-reference part of the framework. A1 reduces
its RMSE by 8.60% (MLP), 12.04% (GRU), 14.36% (LSTM) and 4.03% (Transformer).
This separates the complete-framework result from the added value of semantic structure
and constrained response organization. The A0-to-A2 and A2-to-A1 differences are sequential
comparisons, not an additive causal decomposition.

P1, the four-backbone superiority conjunction over A2, passes the original Holm procedure
(adjusted p=0.0339966). P2, four-backbone noninferiority to unconstrained A3 at a −0.03
skill margin, does not pass (adjusted p=0.273873). The direct-depth comparison is a separate
supplementary family; it does not replace either original test.

Residual XGBoost has RMSE 0.28947 mm and persistence 0.41324 mm. EKI's point estimates
are close to the tree's; similarity is not a statistical equivalence result.

[Original test results](results/tables/primary_conjunctions.csv) · [Per-backbone tests](results/tables/primary_backbone_contrasts.csv)

## 3. Which engineering properties and components are supported?

- **Conditional response consistency:** EKI has zero decreases in 125,340 fixed-context
  load/temperature perturbation checks. A3's load violations range from 5.44% to 13.11%
  and temperature violations from 2.72% to 5.62%; magnitudes are reported with frequencies.
- **Structure representation:** semantic versus flat encoding has positive descriptive
  skill contrasts for MLP (0.03015; 95% interval 0.00537–0.06471) and GRU (0.04102;
  0.00709–0.08234), the two tested ablation backbones.
- **Temperature modulation:** the corresponding component intervals include zero;
  its independent predictive benefit is unresolved in this dataset.

The signed context term and nonnegative response are learned model terms, not directly
measured permanent/recoverable deformation states. The shape guarantee is conditional
on fixed context, not causal identification or guaranteed temporal trajectory monotonicity.

[Response audit](results/tables/response_audit.csv) · [Component contrasts](results/tables/descriptive_components_tree.csv)

## 4. How do the gains change under engineering scenarios?

| Scenario | Finding | Scope |
|---|---|---|
| Historical-calendar temperature, four steps | EKI RMSE 0.42393–0.44902 mm; A2 0.46758–0.54476 mm; A0 1.02984–1.25494 mm | 1,624 common targets, 15 structures |
| Same four-step comparison with XGBoost | XGBoost RMSE 0.40363 mm is lower than all four EKI estimates | Same common target set |
| One missing inspection | EKI RMSE 0.43399–0.43647 mm is higher than each corresponding A2; XGBoost is 0.39603 mm | 2,059 targets |
| Late-load subset | EKI point estimates improve on A0, A2 and XGBoost | 494 targets; retrospective stratum |
| Whole-class holdout | Transfer differences vary by class and backbone | Internal category transfer |

Historical-calendar temperature improves EKI four-step results relative to recent-temperature
continuation; A2 and XGBoost also benefit. All these temperature scenarios use identical
supplied future loads. LOYO temperature is retrospective, and inference scenarios reuse
the fitted models. They do not establish unknown-load forecasting performance.

[Scenario tables](supplements/direct-depth/results/tables/scenario_summary.csv) · [Late-load results](supplements/direct-depth/results/tables/late_load_summary.csv) · [Class transfer](results/tables/class_summary.csv)

## 5. Intervals, explanations and resources

EKI one-step mean structure coverage is 94.96%–95.30%, with MPIW 1.201–1.273 mm.
At four steps under historical-calendar temperature, mean coverage is 92.28%–94.56%,
but worst-structure coverage is only 54.63%–78.70%. Coverage must be read with interval
width, MIS95 and the per-structure distribution. Calibration and selection share inner
data, so finite-sample coverage is not guaranteed. Missing-inspection MIS also favours A2
over EKI for each backbone.

Fixed-background attribution is more stable across seeds for all four EKI backbones;
changing backgrounds does not yield a general improvement. These explanations describe
model responses rather than causal physical effects. EKI has compact parameter counts,
but measured inference overhead is positive relative to A2, so lower deployment cost
or faster inference is not established.

[Intervals](supplements/direct-depth/results/tables/interval_summary.csv) · [Per-structure coverage](supplements/direct-depth/results/tables/interval_structures.csv) · [Attribution stability](results/tables/attribution_background_stability.csv) · [Measured cost](results/tables/cost_summary.csv)

All numerical tables, seeds, scenarios and adverse comparisons are retained. See
[Protocol](PROTOCOL.md#study-status-and-inferential-scope) for interpretation and
[Reproduction](REPRODUCTION.md) for artifact verification.
