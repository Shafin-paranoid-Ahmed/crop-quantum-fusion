# Quantum fusion and improvement experiments

**Problem.** Classify crop disease and its severity stage (25 crop-disease-stage classes) from leaf images and IoT environmental-sensor readings.

**This repository.** Hybrid quantum-classical fusion (E3): original E3-4, early-recall loss rescue, the 4/8-dimensional matched bottleneck study, warm starts, residual and pair-targeted variants, re-uploading, and IoT diagnostics. Paired classical controls stay with each study.

**Result.** Best recorded E3 Macro-F1 across the improvement attempts: 0.968655, below the classical E2-matched-8 reference of 0.973356. This is a controlled evaluation with negative results, not evidence of predictive quantum advantage.

| Code | Model |
| --- | --- |
| E0 | Image-only classifier (MobileNetV3-Large) |
| E1 | IoT-sensor-only MLP |
| E2 | Classical fusion of frozen E0/E1 embeddings |
| E3 | Hybrid fusion with a variational quantum circuit (PennyLane) |

## Start here

Read [RUN_COMMANDS.md](RUN_COMMANDS.md) for environment setup, safe verification,
artifact restoration and examples. Read [provenance/VERIFICATION.md](provenance/VERIFICATION.md)
for checks actually run and their limits.

- `e3_quantum_fusion/`, `e3_early_rescue/`, `e3_bottleneck_sweep/`, `e3_beat_e2_attack/`: experiments and paired controls.
- `e2_multimodal_fusion/scripts/`: frozen loader/training utilities used by these experiments.
- `docs/by_stage/`: experiment chronology, diagnostics, attempted improvements and claim revision.

## Project family

- [crop-multimodal-baselines](https://github.com/Shafin-paranoid-Ahmed/crop-multimodal-baselines): classical baselines and frozen embeddings
- [crop-quantum-fusion](https://github.com/Shafin-paranoid-Ahmed/crop-quantum-fusion): hybrid quantum-classical fusion experiments (this repository)
- [crop-corruption-robustness](https://github.com/Shafin-paranoid-Ahmed/crop-corruption-robustness): test-time corruption robustness study

The baselines repository builds the frozen embeddings, kept as a local bundle (`.artifacts/embeddings-v1.zip`) that is not published. Quantum fusion consumes those embeddings, while robustness evaluates frozen models from both projects.
Each repository includes versioned copies of the small helpers it needs. No submodule
or implicit sibling-directory lookup is required. [COMPANIONS.json](COMPANIONS.json)
lists the family; [ARTIFACTS.json](ARTIFACTS.json) records exact input hashes.

## Data and protocol

The fixed legacy split manifests contain 46,713 / 5,193 / 5,767 train/validation/test
rows and 25 combined crop-disease-stage classes. Legacy training/validation labels
are documented as pseudo/generated. These results must not be conflated with later
trusted-label HDC or QCNN protocols. Stage 0 is healthy; stages 1/2/3 are early/mid/severe.

Raw images, sensor files and regenerated corrupted inputs are not distributed here.
Provide an authorized dataset separately. Split CSVs, saved metrics, selected model
weights and predictions are included; frozen embeddings are in the local bundle.
Path fields in result files use `WORKSPACE` for the machine-specific directory the runs used; they are not runtime dependencies.

## Reproducibility

[provenance/MANIFEST.json](provenance/MANIFEST.json) holds SHA-256 hashes of the frozen code, splits and
result artifacts, and [provenance/NOTES.md](provenance/NOTES.md) records implementation notes. Historical experiment
runbooks preserve the original chronology and may contain superseded status text;
the root runbook and validation record take precedence. Canonical artifacts remain
frozen. Update shared helper copies only deliberately, with hashes and parity checks.

Related, separate research: [QCNN](https://github.com/Shafin-paranoid-Ahmed/qcnn) and
[QEMCMC](https://github.com/Shafin-paranoid-Ahmed/qemcmc). They use different protocols and are not part of this benchmark.

## Contributors

Shafin Ahmed ([@Shafin-paranoid-Ahmed](https://github.com/Shafin-paranoid-Ahmed)), with contributions from [@guardianbot1](https://github.com/guardianbot1) and [@thinkiamparanoid2](https://github.com/thinkiamparanoid2).
