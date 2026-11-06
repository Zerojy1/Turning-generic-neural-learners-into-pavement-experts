# Independent highway evaluation on LTPP

The same OA, LME and MLT design was fitted locally with four neural backbones using 64 LTPP sections in 26 states. Training, validation and test sets contained 80, 34 and 63 targets, respectively, with entire states held out. Validation states determined the learning rate and checkpoint. Predictions were averaged over five seeds. [Split membership](tables/membership.csv)

| Backbone | Direct depth A0 | Anchored (OA) A2 | EKI (OA + LME + MLT) A1 | RMSE reduction vs A0 |
|---|---:|---:|---:|---:|
| MLP | 1.7114 | 1.0249 | 1.0415 | 39.1% |
| GRU | 1.6007 | 1.0066 | 1.0123 | 36.8% |
| LSTM | 1.6061 | 1.0073 | 1.0080 | 37.2% |
| Transformer | 1.5262 | 1.0676 | 1.0835 | 29.0% |

RMSE (mm) gives equal weight to the 26 test sections in seven states. Persistence scored 1.1383 mm and anchored XGBoost 1.0058 mm. A1 improved on A0 for all four backbones; A2 scored 0.0007–0.0166 mm below A1. [Metrics](tables/local_metrics.csv) · [Paired differences](tables/local_comparisons.csv) · [Section results](tables/local_sections.csv)

## Load and temperature response

With inspection history, structure and interval duration fixed, each valid day's load or temperature was increased. A decrease greater than `1e-6 × max(1, |baseline prediction in mm|)` counted as a violation.

| Five-seed ensemble | Load +10% | Load +50% | Temperature +5 °C | Total decreases |
|---|---:|---:|---:|---:|
| Direct A0 | 0/252 | 0/252 | 13/252 | 13/756 (1.7%) |
| Anchored (OA) A2 | 7/252 | 6/252 | 131/252 | 144/756 (19.0%) |
| EKI (OA + LME + MLT) A1 | 0/252 | 0/252 | 0/252 | **0/756** |

For A2, warming produced decreases on 32/63 MLP, 0/63 GRU, 58/63 LSTM and 41/63 Transformer targets. These decreases averaged 0.0384 mm. EKI's mean warming response was +0.0028 mm. [Response magnitudes](tables/shape_perturbations.csv) · [Backbones](tables/shape_backbones.csv) · [Individual-seed audits](tables/shape_seeds.csv)

## Frozen-weight transfer

On 45 targets from 12 other LTPP sections, frozen RIOHTrack EKI weights improved on A0 for three backbones and on A2 for two. All four had higher RMSE than persistence (0.8768 mm) and anchored XGBoost (0.8094 mm). In 41/45 cases the inspection interval exceeded the maximum seen in source training. [Metrics](tables/transfer_metrics.csv) · [Bias and model terms](tables/transfer_bias.csv) · [Training ranges](tables/transfer_ranges.csv)

## Data

Rutting depth averages the paired wheelpath measurements from a 1.8 m straightedge reference. Each section uses one construction period and survey device. Annual lane kESALs are apportioned by day. Near-surface pavement temperature is used where available, otherwise linked air temperature is flagged as a proxy. Local fits use LTPP material codes; frozen-weight transfer uses explicit source-material mappings. All comparisons assume realized interval exposures. [FHWA LTPP SDR 40](https://infopave.fhwa.dot.gov/Data/StandardDataRelease/) · [Tables, fields and hashes](tables/data_sources.csv) · [Models and replay](../../REPRODUCTION.md)
