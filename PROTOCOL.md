# Engineering-knowledge interface

## Architecture

EKI connects three engineering modules to a neural encoder of the preceding six exposure intervals. MLP, GRU, LSTM and Transformer encoders each produce a 64-dimensional history representation. This representation, the latest measured depth, the peak-to-latest gap, interval duration and a 16-dimensional structural representation form the context $c_t$.

$$\widehat y_t=y_{t-1}+b(c_t)+G_t,\qquad G_t=a(c_t)\sum_{j=1}^{d_t}\Delta N_j h(T_j).$$

Current-interval load and temperature enter only through $G_t$. The signed correction $b(c_t)$ permits measured depth decreases between inspections. The learned correction and exposure terms are model components, not separately measured deformation mechanisms.

### Observation anchoring (OA)

The latest measured depth $y_{t-1}$ is added directly to the output. Context includes that depth and the peak-to-latest gap $p_{t-1}-y_{t-1}$. The network estimates an adjustment to the measured starting condition. A zero adjustment reproduces persistence.

### Layer–material encoding (LME)

A shared 19–32–16 mapping encodes each active material–layer slot using its identity and transformed thickness. Within each layer, encoded slots are averaged with normalized thickness weights. Four pooled layer vectors and five thickness totals are projected to a 16-dimensional representation. Thickness totals retain absolute material amounts, while pooling represents relative composition. Empty layers contribute zero vectors. RIOHTrack and locally fitted LTPP models use their respective material vocabularies with the same encoding operations.

### Monotonic load–temperature pathway (MLT)

$$h(T)=1+\operatorname{softplus}(\theta_\gamma)\frac{\max(0,T-25)}{\sigma},\qquad
a(c_t)=0.01\operatorname{softplus}(r(c_t)).$$

Here $\sigma=\max(\mathrm{SD}(T),1\,{}^\circ\mathrm C)$ is estimated from fitting data and held fixed during inference. Daily loads are nonnegative and expressed in units of 10,000 applications. Interval load totals are distributed uniformly across days. The temperature hinge at 25 °C is a fixed modeling choice, not an estimated universal material threshold.

With context, interval duration, preprocessing and fitted parameters fixed, increasing any daily load or temperature cannot lower the forecast. Equal-weight ensembles preserve this property. Changes in the inspection origin change the context, so successive forecasts are not required to be monotonic in time. The response guarantee follows from exposure routing and coefficient signs; removing the observation offset alone does not remove it.

[Core implementation](models.py) · [Prepared inputs](data/) · [Machine-readable configuration](protocol.json)

## Comparators and module ablations

The manuscript names below map to the stored arm identifiers, which remain unchanged for reproducibility.

| Manuscript name | Stored arm | Definition |
|---|---|---|
| Direct-depth neural model / generic backbone | A0 | Unconstrained depth prediction with the available observation, structure and exposure inputs |
| Anchored model / observation-referenced generic model | A2 | Latest measured depth plus an unconstrained learned adjustment |
| EKI | A1 | OA, LME and MLT with the selected history backbone |
| No signs | A3 | Same routed pathway with unconstrained rate and temperature coefficients |
| No heat | A6 | No current-interval temperature modulation; historical temperature remains available |
| Flat | A7 | Flat structural encoding in place of LME |
| No offset | NOANCHOR | EKI with the measured-depth output offset removed; measured context remains available |
| Free path | FREEPATH | Anchoring and LME with an unconstrained readout receiving current exposure summaries |
| Soft | SOFT | Routed signed pathway trained with a fixed monotonicity penalty |
| Anchored XGBoost | A4 | Tree prediction of change from the latest measured depth |
| Monotonic anchored XGBoost | TREE_MONO | Positive constraints on the two current-exposure summary features |
| Persistence | A5 | Repeat the latest measured depth |

The six neural ablations are evaluated on all four backbones. Direct-depth and anchored baselines receive the same source information and training budget. The [module supplement](supplements/modules/README.md) provides all 39 configuration summaries and paired comparisons. Changing a complete component may change several connected operations, so sequential contrasts are not additive causal shares.

## Data and validation

RIOHTrack contains 2,753 rut-depth observations from 19 structures. Five outer folds hold out three structures each; STR4, STR5, STR9 and STR14 remain in training for material coverage. Inner structure splits determine preprocessing, learning rate and checkpoint. Selected configurations are refitted on all outer-training structures. Seeds 42–46 are equally averaged in millimeters. The primary set contains 2,089 dated targets; 2,164 targets are used for the archived-interval sensitivity analysis.

Daily pavement temperature represents 4 cm depth. Sensor coverage below 80% triggers a flagged archive proxy. Source flags distinguish provenance and do not enter the models. Original sensor workbooks are not duplicated; prepared inputs and source hashes are supplied. Upstream dataset access terms continue to apply.

Local LTPP fitting uses 177 targets from 64 sections in 26 states, partitioned into 80 training, 34 validation and 63 test targets. States are disjoint between partitions. A separate frozen-weight cohort contains 45 targets from 12 sections, excluded from local fitting and testing. [Cohorts, source fields and results](supplements/ltpp/README.md)

## Accuracy and statistical inference

RMSE is the square root of the mean of structure-level MSEs, and skill is

$$\mathrm{Skill}=1-\frac{\operatorname{mean}_s(\mathrm{MSE}_{f,s})}{\operatorname{mean}_s(\mathrm{MSE}_{\mathrm{persistence},s})}.$$

LTPP uses road sections as the averaging groups. MAE, depth and increment $R^2$, wins against persistence and per-group errors complement RMSE and skill. Depth $R^2$ pools observations; increment $R^2$ uses group-balanced errors and within-group increment variance. Their distinct denominators are retained.

P1 tests EKI superiority to anchoring alone jointly across the four backbones. P2 tests non-inferiority to No signs with a −0.03 skill margin. Each uses 10,000 paired structure bootstraps, the maximum one-sided boundary-centered reference p value across backbones, and Holm correction across the two conjunctions. EKI versus direct-depth comparisons form a separate four-test Holm family. Other module and scenario intervals are descriptive. [Conjunctions](results/tables/primary_conjunctions.csv) · [Backbone contrasts](results/tables/primary_backbone_contrasts.csv) · [Direct-depth contrasts](supplements/direct-depth/results/tables/supplementary_primary_contrasts.csv)

## Response checks and prediction intervals

Exposure checks increase daily load by 10% or 50%, or daily temperature by 5 °C, holding other inputs fixed. A decrease exceeding `1e-6 × max(1, abs(baseline prediction in mm))` counts as a violation. Individual-fit and ensemble counts are reported separately. Violation rates count checks without the structure weighting used for accuracy.

Empirical 95% intervals use the weighted 0.95 quantile of absolute inner-validation ensemble residuals, separately by model, scenario and horizon. Evaluation reports mean and worst-group coverage, mean prediction interval width (MPIW) and mean interval score (MIS95). Calibration and model selection share validation data, so coverage is assessed empirically.

## Monitoring conditions

Multistep predictions update the latest depth and running maximum with the five-model ensemble prediction. Future measured depths are excluded. Historical-calendar, recent-temperature continuation and realized-temperature scenarios use identical supplied future loads. Historical-calendar temperatures use records available before the prediction origin; other-year temperature sensitivity is retrospective. Comparisons use the designated common target sets.

A missing inspection merges adjacent intervals and removes the omitted depth from both the anchor and peak history. Late-loading evaluation uses observations after 75% of eventual loading as a retrospective stratum. Class holdouts evaluate internal structural transfer. Scenarios reuse fitted models. [Scenario results](RESULTS.md#monitoring-conditions)

## Study status and inferential scope

RIOHTrack is a single-site exploratory evaluation on previously examined data. The direct-depth comparison and complete module extension were specified after earlier study results, with their configurations fixed before their new fits. They retain their separate statistical roles. Shared weather, loading and overlapping training sets limit independence between held-out structures.

LTPP supplies independent road records with local fitting and one geographic split. The design is reused across architectures and datasets; its learned parameters remain application-specific. Conditional monotonicity describes model behavior at fixed context and is distinct from identifying a physical mechanism or a causal exposure effect. [Complete results and application conditions](RESULTS.md)

## Related methodological foundations

The manuscript discusses local referencing alongside [NLinear](https://doi.org/10.1609/aaai.v37i9.26317) and [RevIN](https://openreview.net/forum?id=cGDAkQo1C0p), and shape constraints alongside [monotonicity regularization](https://proceedings.mlr.press/v180/monteiro22a.html), [constrained monotonic networks](https://proceedings.mlr.press/v202/runje23a.html) and [monotonic XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html). EKI combines observation anchoring, pavement composition and routed exposure response within a shared engineering design.
