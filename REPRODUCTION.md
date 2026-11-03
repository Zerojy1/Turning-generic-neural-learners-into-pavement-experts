# Reproduction

Download a ZIP and its `SHA256SUMS.txt` from the same release. Verify the archive checksum, then run the commands from the extracted directory.

| Study | Package | Replay |
|---|---|---|
| [RIOHTrack EKI](https://github.com/Zerojy1/Turning-generic-neural-learners-into-pavement-experts/releases/tag/v1.0.0) | `eki-latest-experiment.zip` | `python verify_distribution.py --mode full`, then `python verify_results.py` |
| [RIOHTrack direct-depth comparison](https://github.com/Zerojy1/Turning-generic-neural-learners-into-pavement-experts/releases/tag/v1.1.0) | `eki_direct_supplement_reproduction.zip` | `python verify_supplement.py` |
| [LTPP local replication and shape audit](https://github.com/Zerojy1/Turning-generic-neural-learners-into-pavement-experts/releases/tag/v1.2.0) | `eki_ltpp_reviewer.zip` | `python replay.py` |

The LTPP replay checks 65 locally fitted models on 63 targets, including 2,268 five-seed ensemble perturbation checks. To refit from the prepared inputs, run `python fit_local.py NEW_DIRECTORY` in the extracted LTPP package. [Environment](environment.json) · [Evaluation protocol](PROTOCOL.md) · [LTPP data sources](supplements/ltpp/tables/data_sources.csv)

