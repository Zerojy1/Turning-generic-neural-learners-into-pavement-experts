# Results and engineering interpretation

## Cross-backbone prediction
EKI improves one-step prediction over information-matched ordinary residual networks
on all four tested backbones. P1 passes the specified conjunction and Holm procedure.
The full one-step table contains all 2,164 targets as a sensitivity analysis as well as
the 2,089 dated primary targets. Similar point estimates to XGBoost do not establish equivalence.

## Response structure and accuracy trade-offs
EKI has zero decreases in 125,340 fixed-context load/temperature perturbation checks.
Unconstrained load-perturbation violations range from 5.44% to 13.11%, and temperature
violations from 2.72% to 5.62%. Response magnitudes are reported alongside frequencies.
P2 does not establish the four-backbone noninferiority claim at the -0.03 skill margin.
The signed and nonnegative output terms are model components, not measured physical states.

## Component contributions
Semantic versus flat structure representations show positive descriptive contrasts for
MLP and GRU. Confidence intervals for current temperature modulation include zero.
These ablations do not establish independent benefits for every engineering input.

## Engineering scenarios
Historical calendar temperature improves four-step results relative to recent temperature
continuation for EKI; ordinary residual networks and XGBoost also benefit. XGBoost remains
stronger in the four-step comparison. Following one omitted inspection, all four EKI
RMSE point estimates are worse than their ordinary residual counterparts and the tree.
Late-load point estimates are favourable, while whole-class transfer results are mixed.
Every task, arm, structure and forecast horizon is retained in the supplied tables.

## Uncertainty, explanations and resources
Read average and worst-structure coverage together with width and MIS95; the worst
structures can be substantially below nominal coverage. Empirical intervals have no
finite-sample coverage guarantee. Fixed-background attribution is more stable across
seeds for all four EKI backbones, but changing backgrounds gives no general improvement.
The modules have compact parameter counts; measured inference overhead is positive,
so the results do not establish faster or lower-cost deployment than ordinary residual networks.

## Evidence navigation
- `one_step_summary.csv`, `primary_backbone_contrasts.csv`, `primary_conjunctions.csv`:
  prediction and primary tests.
- The full `results/tables/` directory supplies all component, scenario, interval,
  transfer, response, attribution, cost, and independent verification tables.
- `all_arms_and_scenarios.csv` is a convenience aggregation; `record_type`, `scope`,
  `scenario`, `horizon`, and target counts identify the applicable aggregation.
