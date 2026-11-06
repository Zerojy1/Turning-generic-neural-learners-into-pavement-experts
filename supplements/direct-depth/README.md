# Direct-depth, anchored and EKI comparison

This evaluation tests the complete EKI framework against information-matched generic
direct-depth learners, with anchored networks as an intermediate reference.
It covers MLP, GRU, LSTM and Transformer backbones under the same grouped validation.

RMSE in millimetres, 2,089 dated targets and five-seed ensembles:

| Backbone | Direct depth A0 | Anchored (OA) A2 | EKI A1 | Reduction: A0 → A1 |
|---|---:|---:|---:|---:|
| MLP | 0.56436 | 0.31186 | **0.28504** | **49.49%** |
| GRU | 0.68053 | 0.33026 | **0.29049** | **57.31%** |
| LSTM | 0.68601 | 0.34299 | **0.29373** | **57.18%** |
| Transformer | 0.67366 | 0.30066 | **0.28854** | **57.17%** |

The four supplementary A1-versus-A0 comparisons pass the specified Holm criterion;
the original P1/P2 tests retain their separate interpretation. Consult the
[complete evidence discussion](../../RESULTS.md) and
[evaluation protocol](../../PROTOCOL.md) for scope and all adverse comparisons.

## Browse the evidence

- [Main comparison](results/tables/three_level_main_table.csv)
- [All one-step metrics and sensitivity](results/tables/one_step_summary.csv)
- [Scenario results](results/tables/scenario_summary.csv)
- [Intervals](results/tables/interval_summary.csv) and [per-structure coverage](results/tables/interval_structures.csv)
- [Chinese scientific interpretation](REPORT_ZH.md)
- [Figures](figures/) and [full result index](RESULT_INDEX.md)
- [Independent replay verification](checks/independent_supplement_verification.json)

## Download and reproduce

[v1.1.0](https://github.com/Zerojy1/Turning-generic-neural-learners-into-pavement-experts/releases/tag/v1.1.0) supplies the full reproduction ZIP (about 179 MB,
2,649 files), a smaller results ZIP and SHA-256 checksums. The full ZIP includes all
100 direct-depth fits, inner candidates, inputs, saved predictions and calibration records.
[v1.0.0](https://github.com/Zerojy1/Turning-generic-neural-learners-into-pavement-experts/releases/tag/v1.0.0) supplies the original EKI weights.

Execute from the extracted release, following [reproduction instructions](../../REPRODUCTION.md).
The Git mirror supports inspection and navigation; release manifests bind the original
executable snapshots. The [machine-readable protocol](protocol.json) and frozen hashes
remain available without changing their experimental definitions.
