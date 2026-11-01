# Reproducing and verifying the current experiment

## Download
Download `eki-latest-experiment.zip` and `SHA256SUMS.txt` from the
[release](https://github.com/Zerojy1/Turning-generic-neural-learners-into-pavement-experts/releases/tag/v1.0.0). Verify the ZIP hash before extraction.
The ZIP contains the repository files plus the large artifacts omitted from Git.

## Integrity check (standard Python only)
After extraction, run:
```bash
python verify_distribution.py --mode full
```
For a Git checkout without release artifacts use `--mode git`. Every included file is
size- and SHA-256-checked. The manifest distinguishes files stored in Git and the asset.

## Numerical replay
Use Linux with the library versions recorded in `environment.json` and CUDA for the
original neural replay script. Run in a working copy so the released artifacts stay intact:
```bash
python verify_results.py
```
This reloads all 605 final models, checks split membership and internal model selection,
recomputes main metrics and both primary tests, and verifies recursive state updates,
temperature provenance, response checks and attribution additivity. Floating-point
hardware differences may produce small variations. The original replay record is in
`checks/independent_verification.json`.

## Recompute summaries
The complete asset supports `python analysis.py`, `python finish_analysis.py`, and
`python plot_results.py` in a writable copy. Do not mix results from different snapshots.

## Fresh training
Create a NEW working copy of the Git source with the supplied input data and required
calendar records from the complete asset, but without `runs/`, `evaluation/`, `attribution/`,
or queue completion records. Use `python train_pipeline.py` to fit the fixed matrix.
Do not launch it in the delivered result directory. This publication task did not retrain models.

## Distribution provenance
`release_provenance.json` records source and packaged hashes. Packaging changes affect
documentation, non-computational metadata, local reference paths, and provenance guards.
Prediction formulas, weights, training choices, numeric input arrays and output files are
unchanged. `training_freeze.json` and `analysis_freeze.json` bind the distribution files;
they are explicitly packaging-time snapshots, not a new prospective registration.
