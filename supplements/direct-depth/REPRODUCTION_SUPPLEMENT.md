# Direct-depth supplement: reproduction

This supplement adds 100 matched A0 models to the unchanged EKI distribution v1.0.0. It contains the actual frozen source, model weights (including inner candidates), numerical data, original reused predictions, learning curves, all scenario predictions and calibration records. The full original EKI weights remain in the base v1.0.0 release. The original P1 passed and P2 failed; both remain unchanged.

1. Verify the release ZIP against SHA256SUMS.txt. Check every member against reproduction_manifest.json (SHA-256 and byte counts); retain the manifest.
2. Extract to a new directory. Python packages and GPU context are listed in delivery_environment.json. Paths in runtime code resolve relative to that directory; provenance records retain the original machine's history.
3. Without refitting, run `python verify_supplement.py` on a CUDA-enabled environment to replay all 100 new models and recompute the primary metrics, paired bootstrap, Holm adjustment and 360 calibration groups. This writes verification logs in the extracted copy.
4. `python supplement_analysis.py` recomputes the three-level primary and sensitivity tables from saved predictions. `python supplement_evaluation.py` aggregates the saved scenario groups; inference results and inner calibration predictions are retained under evaluation/.
5. For a clean full refit, keep the original extraction intact and use a separate copy containing frozen root code/configuration, data/, base_reference/ and provenance checks. Omit runs/, evaluation/, results/ and queue state in that new copy, then run `python queue_train.py main`. This is optional and computationally expensive; it is not needed to verify the delivered results. Do not use outer performance to modify any settings.

The source freeze is 239199ecf24890a41ef481301827ac6f226aea46. training_freeze.json and analysis_freeze.json are byte-level guards. post_review_v1.py only enriches documentation after computation; it does not change numerical analysis or figures. checks/documentation_change_log.json records this distinction.

The new comparisons are further frozen exploratory comparisons on previously examined single-site data. A1 vs A0 measures the entire formulation, A2 vs A0 bridges the observation reference, and A1 vs A2 measures the additional EKI design. None establishes independent external confirmation.
