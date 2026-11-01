# Evaluation protocol

The experiment uses 19 structures, five structure-held-out folds, four training anchors,
and seeds 42–46. The primary analysis has 2,089 dated targets; 2,164 targets form the
sensitivity analysis. Candidate selection and preprocessing use inner training structures.
The selected candidate is refitted on outer training structures. Neural fits use three
learning rates and at most 220 epochs; the selected epoch is determined internally.

Current load and daily temperature enter A1 exclusively through the constrained output
pathway. Historical exposure, observation memory, structure, and duration determine
context. A3 changes the signs allowed for the rate and temperature coefficients while
preserving the pathway. A6 removes current temperature modulation, not historical
temperature. A7 replaces semantic structure with a capacity-matched flat representation.

All temperature rollout scenarios use identical supplied interval loads. Strict historical
calendar temperature uses years, support dates, and availability dates before the origin.
Origins without support remain unavailable. Leave-target-year-out temperature may use
later records and is a retrospective sensitivity analysis. Neither scenario claims fully
unknown-future-load prediction. Model fitting is grouped by structure, not chronologically.

The main statistic is one minus the ratio of structure-balanced model and persistence
MSE. P1 combines four A1-versus-A2 superiority comparisons; P2 combines four A1-versus-A3
noninferiority comparisons with a -0.03 skill margin. Each conjunction uses the maximum
one-sided reference p; Holm correction is applied across the two conjunctions. Other
comparisons and percentile intervals are descriptive. Shared site conditions and training
overlap limit independence. This is a single-site exploratory evaluation, not independent
external confirmation.

Intervals use structure-balanced inner selected-model ensemble errors by scenario and
horizon. Selection and calibration share inner data; empirical coverage is not guaranteed.
Conditional shape checks keep context and duration fixed and report response magnitudes
alongside violations. Attribution explains increments and is not a causal decomposition.
