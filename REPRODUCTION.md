# Data and reproduction

## Verified model packages

Download an archive and its checksum file from the same release. Verify the archive checksum, then execute the replay command from the extracted directory. The packages retain the executable snapshots, numerical inputs, fitted models and verification records.

| Evaluation | Package | Replay |
|---|---|---|
| [RIOHTrack EKI](https://github.com/Zerojy1/Turning-generic-neural-learners-into-pavement-experts/releases/tag/v1.0.0) | `eki-latest-experiment.zip` | `python verify_distribution.py --mode full`, then `python verify_results.py` |
| [Matched direct-depth models](https://github.com/Zerojy1/Turning-generic-neural-learners-into-pavement-experts/releases/tag/v1.1.0) | `eki_direct_supplement_reproduction.zip` | `python verify_supplement.py` |
| [LTPP local fitting and response checks](https://github.com/Zerojy1/Turning-generic-neural-learners-into-pavement-experts/releases/tag/v1.2.0) | `eki_ltpp_reviewer.zip` | `python replay.py` |

The LTPP replay verifies 65 locally fitted models on 63 test targets and 2,268 ensemble perturbation checks across the three neural arms. The EKI subset contains 756 checks. To fit from the prepared LTPP inputs, run `python fit_local.py NEW_DIRECTORY` within the extracted package.

## Complete module tables

The [module supplement](supplements/modules/README.md) adds the completed four-backbone ablations and constrained-tree results. It includes configuration summaries, paired intervals, per-structure errors and embedding coordinates with SHA-256 hashes. These tables can be used to reproduce the reported structure-balanced point metrics. The additional module checkpoints remain in the verified local study archive and are not included in the three releases above; no full checkpoint replay for that extension is claimed by these public table files.

## Source data

RIOHTrack uses the [2016–2021 rutting observation dataset](https://doi.org/10.57760/sciencedb.j00001.01195). Prepared numerical inputs and source hashes are under [data/](data/). LTPP records come from [FHWA InfoPave Standard Data Release 40](https://infopave.fhwa.dot.gov/Data/StandardDataRelease/), with original tables and field mappings listed in [data_sources.csv](supplements/ltpp/tables/data_sources.csv). Upstream dataset terms continue to govern the source records.

## Repository and release versions

This branch provides current documentation and browsable tables. Release-specific manifests bind their original executable snapshots, so replay should start from the corresponding release extraction. The current documentation manifest covers presentation files, and the module supplement has its own source manifest. Updating the paper-facing documentation does not change model weights, training inputs or stored predictions.

[Environment](environment.json) · [Protocol](PROTOCOL.md) · [Paper-to-code guide](MANUSCRIPT_GUIDE.md) · [Release provenance](release_provenance.json)
