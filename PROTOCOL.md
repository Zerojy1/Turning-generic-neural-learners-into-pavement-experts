# Method and evaluation

## Modular engineering-knowledge embedding

EKI predicts rutting depth from an observation reference, a signed context adjustment
and an exposure-driven response:

$$\widehat y_t=y_{t-1}+b_\theta(c_t)+G_t,\qquad
G_t=\sum_j\Delta N_j\,a_\theta(c_t)\,h_\theta(T_j).$$

Context combines six historical exposure steps, latest depth, peak–latest discrepancy,
layer–material structure representation and known interval duration. Current-interval
load and temperature are excluded from context. The rate is nonnegative; temperature
modulation is nonnegative and nondecreasing. Thus, at fixed context and duration, the
prediction is nondecreasing in current exposure. The terms are learned response
components, not separately identified physical deformation states.

## Comparison design

| Arm | Prediction | Purpose |
|---|---|---|
| A0 | Direct depth from the available inputs | Generic learner reference |
| A2 | Latest depth plus a free learned increment | Observation-reference bridge |
| A1 | Latest depth plus context and constrained exposure response | Complete EKI |
| A3 | EKI routing with unconstrained rate/temperature coefficients | Shape-constraint comparison |
| A4 / A5 | Residual XGBoost / persistence | Competitive and no-change references |
| A6 / A7 | Remove current temperature modulation / use flat structure | Component comparisons for MLP and GRU |

A0 and A2 share inputs, backbone, head dimensions, initialization rules, optimizer,
learning-rate candidates and training budget. Their output reference/target parameterization
differs. A1 changes the combined representation and response design. Models are fitted
independently with inner selection; shared budget does not imply globally optimal tuning.
Historical temperature remains available in A6. A3 preserves exposure routing while
relaxing coefficient sign constraints.

## Data and validation

RIOHTrack contains 19 structures and 2,753 observations, yielding 2,734 transitions.
Five outer structure-held-out folds evaluate 15 structures; STR4/5/9/14 remain training
anchors. Inner structure splits determine preprocessing, learning rate and epoch selection;
selected configurations are refitted on outer training structures. Seeds 42–46 are
equally averaged in millimetres. The main evaluation uses 2,089 dated targets and the
sensitivity analysis 2,164 targets.

Daily internal temperature represents 4 cm depth. Coverage below 80% uses a flagged
archive proxy. The mean-temperature threshold proxy is distinct from a 10-minute
degree-hour integral. Load is distributed uniformly within each observation interval.
Original sensor workbooks are not duplicated; numerical inputs and provenance hashes
are supplied. Repository access does not grant a new licence to upstream datasets.

## Metrics and statistical comparisons

The structure-balanced skill is

`skill = 1 - mean_s(MSE_model,s) / mean_s(MSE_persistence,s)`.

Balanced RMSE is the square root of mean structure MSE. MAE, increment R², structure
wins, coverage, MPIW and MIS95 supplement the main metric. Bootstrap resampling uses
10,000 paired structure draws; seeds and individual observations are not independent units.

- **Original P1:** four-backbone A1-versus-A2 superiority conjunction.
- **Original P2:** four-backbone A1-versus-A3 noninferiority conjunction, margin −0.03 skill.
  Each conjunction uses the maximum one-sided boundary-centred bootstrap reference p;
  Holm correction is applied across P1/P2.
- **Direct-depth supplementary comparison:** four A1-versus-A0 contrasts, Holm-adjusted
  across the four backbones. All must have positive direction and adjusted p ≤ 0.05.
  This is a separate family and does not replace P1/P2.

Percentile 95% intervals and other comparisons are descriptive. Reference p values are
not distribution-free exact tests. Full configuration is in [protocol.json](protocol.json),
[execution_addendum.md](execution_addendum.md) and the
[direct-depth protocol](supplements/direct-depth/protocol.json).

## Inference scenarios and uncertainty

Actual-temperature, historical-calendar and recent-temperature rollout scenarios use
the same supplied future loads. Strict historical-calendar inputs only access supporting
records available before the forecast origin; origins without support remain unavailable.
LOYO uses other years and is retrospective. All scenario comparisons use their designated
common target sets. Missing-inspection tests reconstruct exposures; late-load analysis
is retrospective. Class holdouts evaluate internal category transfer.

Fitted models are reused for inference. Empirical intervals use structure-balanced
absolute residual quantiles of selected inner ensembles, separately by scenario and
horizon. Model selection and calibration share inner data, so coverage is not guaranteed.

## Study status and inferential scope

The original RIOHTrack evaluation is a single-site exploratory study. The direct-depth comparison was specified
and frozen after the initial EKI results were available, but before its new direct-baseline
fits; it is not an independent confirmation of the original study. Numerical configurations
and comparison rules were frozen for each experimental block. The presentation emphasizes
the complete framework, while the original tests and the supplementary statistical family
retain their distinct roles.

Shared loading/climate and overlapping training sets limit independence. A separate LTPP evaluation uses independent road sections for locally fitted method replication and a distinct frozen-weight transfer assessment. No evaluation against human experts was performed. The title refers
to domain specialization, not demonstrated equivalence to experts or replacement of
inspection equipment. Formula-based consistency is separate from mechanism identification.

## LTPP replication

The locally fitted study uses 64 sections in 26 states. Training, validation and testing use disjoint states. The 12 sections in the frozen-weight study are excluded throughout local fitting and testing. All results are descriptive. [Cohort, source fields, training and perturbation definitions](supplements/ltpp/).
