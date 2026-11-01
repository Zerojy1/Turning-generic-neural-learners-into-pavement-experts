# Matched direct-depth comparison

This folder supplements the original EKI distribution without changing its results.
A0 is direct depth; A2 adds the latest observation to a learned residual; A1 is full EKI.
All baselines receive observation memory, structure, temperature and load information.

| Backbone | Direct depth A0 | Ordinary residual A2 | Full EKI A1 | A1 vs A0 reduction | A1 vs A2 reduction |
|---|---:|---:|---:|---:|---:|
| MLP | 0.56436 | 0.31186 | 0.28504 | 49.49% | 8.60% |
| GRU | 0.68053 | 0.33026 | 0.29049 | 57.31% | 12.04% |
| LSTM | 0.68601 | 0.34299 | 0.29373 | 57.18% | 14.36% |
| Transformer | 0.67366 | 0.30066 | 0.28854 | 57.17% | 4.03% |

The main scope is 2,089 dated targets from 15 held-out structures, using five seeds
averaged in millimetres. Full-target sensitivity (2,164 targets), all temperature
scenarios, one-missing-inspection results, late-load results and empirical intervals
are included in results/tables/. The baseline weakness and all adverse comparisons
are discussed in REPORT_ZH.md. Original P1 passed/P2 failed remains unchanged.

## Download and reproduce

Download `eki_direct_supplement_reproduction.zip` and `SHA256SUMS.txt` from
[v1.1.0](https://github.com/Zerojy1/Turning-generic-neural-learners-into-pavement-experts/releases/tag/v1.1.0).
The full package is about 179 MB and contains 2,649 files, including 100 final fits,
their inner candidates, numerical inputs, all predictions and calibration records.
`eki_direct_supplement_results.zip` is a smaller report/table/figure bundle.

Follow [REPRODUCTION_SUPPLEMENT.md](REPRODUCTION_SUPPLEMENT.md) inside the full extraction.
The source mirrored here is for inspection; execute from the full ZIP, which includes
data/, runs/, evaluation/ and base_reference/. Original EKI weights are in v1.0.0.
Check the ZIP SHA-256 and every extracted file against reproduction_manifest.json.

- [All result tables](results/tables/)
- [Reviewed figures](figures/)
- [Chinese explanation and manuscript guidance](REPORT_ZH.md)
- [Independent replay evidence](checks/independent_supplement_verification.json)
- [Windows member-hash verification](Windows_verification.json)
- [Frozen design](DESIGN.md) and [protocol](protocol.json)

Source freeze: `239199ecf24890a41ef481301827ac6f226aea46`.
This is a further frozen single-site exploratory comparison, not external confirmation.
The four-step calendar figure uses a common subset and different y-axis scales across
panels; numerical comparisons should use the tables. All future loads are given.
