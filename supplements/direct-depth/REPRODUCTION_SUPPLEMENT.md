# Reproducing the direct-depth comparison

Follow the repository's [reproduction guide](../../REPRODUCTION.md) using the complete
v1.1.0 release extraction. It includes the exact source, 100 fitted direct models and
their inner candidates, numerical inputs, reused original predictions, all inference
scenarios, calibration records and byte-level manifests. Full original EKI weights
are supplied by v1.0.0.

The main verification command in a CUDA-enabled environment is:

```bash
python verify_supplement.py
```

The check independently replays 100 models and recomputes metrics, paired bootstrap,
Holm adjustment and 360 calibration groups. Run in a working copy so immutable released
artifacts remain available. Frozen configuration is in protocol.json; evaluation scope
is described in the repository [Protocol](../../PROTOCOL.md).
