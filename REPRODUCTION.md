# Reproduction and artifact verification

Use the versioned release packages as the executable snapshots. The main branch contains
updated explanatory documentation; release manifests describe their own immutable
snapshots, not every subsequent documentation edit on this branch.

## 1. Download the relevant package

| Evaluation | Required asset | Release |
|---|---|---|
| EKI, components, transfer and attribution | `eki-latest-experiment.zip` | [v1.0.0](https://github.com/Zerojy1/Turning-generic-neural-learners-into-pavement-experts/releases/tag/v1.0.0) |
| Matched direct-depth and three-level comparison | `eki_direct_supplement_reproduction.zip` | [v1.1.0](https://github.com/Zerojy1/Turning-generic-neural-learners-into-pavement-experts/releases/tag/v1.1.0) |
| Quick report/table/figure access | `eki_direct_supplement_results.zip` | [v1.1.0](https://github.com/Zerojy1/Turning-generic-neural-learners-into-pavement-experts/releases/tag/v1.1.0) |

Download the matching `SHA256SUMS.txt` and verify the archive before extraction. The
direct-depth package contains all 100 new final fits, their inner candidates, inputs,
prediction records and calibration artifacts. Original EKI weights remain in v1.0.0.

## 2. Check file integrity

For the extracted v1.0.0 package:

```bash
python verify_distribution.py --mode full
```

For a source-only integrity check, use the exact v1.0.0 source snapshot and `--mode git`,
rather than the evolving documentation branch. In v1.1.0, every file's SHA-256 and byte
count is listed in `reproduction_manifest.json`. The supplied
[Windows verification](supplements/direct-depth/Windows_verification.json) checks all 2,648 manifest members
and archive CRCs; the archive also contains the manifest itself.

## 3. Replay fitted models

Use a writable copy and the recorded Linux/Python/CUDA environment. From the v1.0.0 extraction:

```bash
python verify_results.py
```

This independently reloads 605 models and checks metrics, selection, recursive updates,
temperature provenance, shape responses and attribution additivity.

From the v1.1.0 extraction:

```bash
python verify_supplement.py
```

This reloads all 100 direct-depth models, recomputes primary metrics, bootstrap/Holm
comparisons and 360 calibration groups, and checks that reused EKI/residual results are
unchanged. Hardware differences may cause small floating-point differences.

## 4. Recompute tables or refit

The EKI snapshot supports `analysis.py`, `finish_analysis.py` and `plot_results.py`.
The direct-depth snapshot supports `supplement_analysis.py` and
`supplement_evaluation.py`; saved per-group inference records are included. Do not mix
artifacts across snapshots or replace supplied results with selectively rerun estimates.

Fresh training is optional: use a separate clean working copy with the exact frozen
inputs and configurations, without previous model or queue completion records. Use
`train_pipeline.py` for the EKI matrix or `queue_train.py main` for the direct-depth
matrix. This is not needed to validate the released fitted predictions.

## Provenance

The release packages retain training/analysis manifests, source hashes, environments,
split membership, learning curves and verification records. Current documentation
reorganization changes no numerical inputs, weights, predictions, tests or release assets.
See [Protocol](PROTOCOL.md) for evaluation scope and [Results](RESULTS.md) for all outcomes.
