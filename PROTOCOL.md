# Method and evaluation

The framework predicts depth relative to the latest observation. Context contains a
six-step historical exposure encoding, latest/peak observation memory, a semantic
layer–material representation and known interval duration. Current load and temperature
are excluded from that context. A signed context term is combined with a nonnegative
rate multiplying a sum of daily loads and monotone temperature modulation.

Data comprise 19 structures and 2,753 observations. The main five-fold study tests 15
structures; four anchors stay in training. The model inputs contain 2,734 transitions.
There are 2,089 dated primary evaluation targets and 2,164 sensitivity targets. Internal
temperature is represented at 4 cm depth; daily availability below 80% uses a flagged
archive proxy. The day-mean threshold proxy is not a 10-minute degree-hour integral.

The 605 logical jobs comprise 325 main-comparison, 100 component, and 180 class-transfer
fits. They are not 605 independent statistical samples. Seeds 42–46 are equally averaged.
P1 tests four-backbone improvement over ordinary residuals; P2 tests four-backbone
noninferiority to unconstrained counterparts. Models share information and approximately
matched module capacity. A1/A2 compares the combined embedding design, not a single
isolated physical effect. A1/A3 changes coefficient sign constraints while keeping routing.

The score is `1 - mean(structure MSE_model) / mean(structure MSE_persistence)`.
Balanced RMSE is the square root of mean structure MSE, not the mean of RMSEs.
The bootstrap uses 10,000 paired structure resamples; seeds and individual targets are
not independent bootstrap units. Other comparisons are descriptive. Details and all
thresholds are in `protocol.json` and `execution_addendum.md`.

Daily load is assumed uniform within each observation interval. All rollout temperature
scenarios use identical supplied actual loads; no fully unknown-load forecast is claimed.
Strict historical calendar temperature only accesses available pre-origin years and records.
LOYO temperature is retrospective. Missing-inspection evaluation reconstructs exposures;
late-load evaluation is retrospective. Class holdouts test internal category transfer.

The inputs are the numerical data used by the completed experiment. Original sensor
workbooks and raw 10-minute records are not duplicated in this current-study distribution.
Data provenance hashes accompany the daily arrays. Public availability of this repository
does not by itself grant a new licence to upstream third-party source datasets.

This is a single-site exploratory study. Shared loading, climate and overlapping training
sets limit independence. No independent external validation or expert-performance comparison
has been completed. Formula-based response consistency is distinct from measured mechanism.
