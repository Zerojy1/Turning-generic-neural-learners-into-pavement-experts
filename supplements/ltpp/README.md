# Engineering knowledge on LTPP road sections

Can a pavement knowledge interface benefit several neural architectures when each is trained on a new road dataset? We fitted the same EKI prediction rule on FHWA LTPP data, with a layer vocabulary defined from LTPP material codes. In a geographic holdout, full EKI improved on direct-depth prediction and persistence for all four backbones. The shape audit shows what the constrained exposure pathway adds beyond observation referencing.

## Local training and geographic testing

The sample contains 177 eligible one-step targets from 64 sections in 26 US states. All sections of a state stayed together. Training comprised 80 targets from 28 sections in 13 states. Validation comprised 34 targets from 10 sections in 6 states. The remaining 63 targets from 26 sections in 7 states formed the test set. Three learning rates and checkpoint epochs were compared using validation states. Each selected model was refitted on training and validation states. Five seeds (42–46) were equally averaged. No RIOHTrack weights were loaded for these fits.

| Backbone | Direct depth A0 | Observation residual A2 | Full EKI A1 | RMSE reduction vs A0 |
|---|---:|---:|---:|---:|
| MLP | 1.7114 | 1.0249 | 1.0415 | 39.1% |
| GRU | 1.6007 | 1.0066 | 1.0123 | 36.8% |
| LSTM | 1.6061 | 1.0073 | 1.0080 | 37.2% |
| Transformer | 1.5262 | 1.0676 | 1.0835 | 29.0% |

RMSE is in mm, computed as the square root of the mean of section MSEs. Full EKI had lower MSE than A0 on 20–22 of the 26 sections per backbone. Persistence scored 1.1383 mm and residual XGBoost 1.0058 mm. Compared with A2, A1 had slightly higher RMSE on all four backbones (0.0007–0.0166 mm). These are descriptive differences from one geographic split. [All metrics](tables/local_metrics.csv) · [Paired comparisons](tables/local_comparisons.csv) · [Individual sections](tables/local_sections.csv) · [Split membership](tables/membership.csv)

## Conditional response audit

With history, structure, duration and model parameters fixed, we increased the load on each valid day by 10% or 50%, or increased all valid daily temperature inputs by 5 °C. Each perturbed input went through the fitted preprocessor. A decrease exceeding `1e-6 × max(1, |baseline prediction in mm|)` counted as a violation. Each column comprises 63 targets × 4 backbones. The same targets recur across perturbations.

| Five-seed ensemble | Load +10% | Load +50% | Temperature +5 °C | Total decreases |
|---|---:|---:|---:|---:|
| Direct A0 | 0/252 | 0/252 | 13/252 | 13/756 (1.7%) |
| Observation residual A2 | 7/252 | 6/252 | 131/252 | 144/756 (19.0%) |
| Full EKI A1 | 0/252 | 0/252 | 0/252 | **0/756** |

For warming alone, A2 decreases occurred in 32/63 MLP, 0/63 GRU, 58/63 LSTM and 41/63 Transformer checks. Among the 131 decreases the mean magnitude was 0.0384 mm and the maximum was 0.1515 mm. A1 had an average warming response of +0.0028 mm and also had zero decreases across 3,780 individual-seed checks. Direction and magnitude describe different aspects of the response. This audit evaluates conditional model behavior, not a causal effect of temperature. [Ensemble counts and magnitudes](tables/shape_perturbations.csv) · [By backbone](tables/shape_backbones.csv) · [Individual seeds pooled by arm](tables/shape_seeds.csv)

## Transfer of frozen RIOHTrack weights

A separate test evaluated frozen RIOHTrack models on 45 targets from 12 LTPP sections in 10 states. These 12 sections were excluded from the local-training study above. The median interval was 367 days versus 13 days on the RIOHTrack evaluation set. Directly transferring EKI weights improved RMSE over A0 for three of four backbones and over A2 for two. All four frozen EKI models scored worse than persistence (0.8768 mm) and residual XGBoost (0.8094 mm). GRU and LSTM overprediction was concentrated in the signed context term. Of 45 targets, 41 exceeded the source training duration maximum, but only one exceeded the numerical range of the interval load input. The source used load repetitions and LTPP uses lane ESALs. [Transfer metrics](tables/transfer_metrics.csv) · [Error and response components](tables/transfer_bias.csv) · [Training ranges](tables/transfer_ranges.csv)

## Data and interpretation

The section measurements are from the US Federal Highway Administration, [LTPP InfoPave Standard Data Release 40](https://infopave.fhwa.dot.gov/Data/StandardDataRelease/). The endpoint averages paired left/right 1.8 m straightedge rut-depth summaries. Each section contributes a single construction period and device, with no target spanning a recorded intervention. Annual lane kESAL is apportioned by day. Near-surface measured or interpolated pavement temperature is used where supported, otherwise a linked weather-station air-temperature proxy is flagged. The local models use native layer–material slots. The frozen-weight transfer uses explicit source-material mapping scenarios. Both studies condition on realized interval exposures. Sample limitations, measurement differences and the state split constrain generalization. [Source tables and original file hashes](tables/data_sources.csv)

## Reproduce

[Download the LTPP release](https://github.com/Zerojy1/Turning-generic-neural-learners-into-pavement-experts/releases/tag/v1.2.0). Extract the reviewer package and run `python replay.py` from its root with NumPy, pandas, PyTorch, SciPy, scikit-learn and XGBoost available. It verifies the package manifest, replays the 65 locally fitted models, rebuilds predictions for each perturbation and checks the tables. See the package README for the input and output layout.

