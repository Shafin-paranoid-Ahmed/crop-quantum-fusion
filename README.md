# Quantum fusion and improvement experiments

Original E3-4, early-recall loss rescue, the 4/8-dimensional matched bottleneck study, warm starts, residual and pair-targeted variants, re-uploading, and experiment-specific IoT diagnostics. Paired classical controls stay with each study. Best recorded E3 attack Macro-F1: 0.968655, below the original E2-matched-8 reference of 0.973356. This is a controlled evaluation with negative results, not evidence of predictive quantum advantage.

This is an independent, private research snapshot extracted on 2026-10-12.
It does not require the original thesis checkout or a sibling repository for code imports.

## Start here

Read [RUN_COMMANDS.md](RUN_COMMANDS.md) for environment setup, safe verification,
artifact downloads and examples. Read [provenance/VERIFICATION.md](provenance/VERIFICATION.md)
for checks actually run and their limits.

- `e3_quantum_fusion/`, `e3_early_rescue/`, `e3_bottleneck_sweep/`, `e3_beat_e2_attack/`: experiments and paired controls.
- `e2_multimodal_fusion/scripts/`: frozen loader/training utilities used by these experiments.
- `docs/by_stage/`: experiment chronology, diagnostics, attempted improvements and claim revision.

## Companion repositories

- [crop-multimodal-baselines](https://github.com/Shafin-paranoid-Ahmed/crop-multimodal-baselines)
- [crop-quantum-fusion](https://github.com/Shafin-paranoid-Ahmed/crop-quantum-fusion)
- [crop-corruption-robustness](https://github.com/Shafin-paranoid-Ahmed/crop-corruption-robustness)

The baselines repository owns the frozen embedding release. Quantum fusion consumes
those embeddings, while robustness evaluates frozen models from both projects.
Each repository includes versioned copies of the small helpers it needs. No submodule
or implicit sibling-directory lookup is required. [COMPANIONS.json](COMPANIONS.json)
records the pinned companion commits; [ARTIFACTS.json](ARTIFACTS.json) records exact input hashes.

## Data and protocol

The fixed legacy split manifests contain 46,713 / 5,193 / 5,767 train/validation/test
rows and 25 combined crop-disease-stage classes. Legacy training/validation labels
are documented as pseudo/generated. These results must not be conflated with later
trusted-label HDC or QCNN protocols. Stage 0 is healthy; stages 1/2/3 are early/mid/severe.

Raw images, sensor files and regenerated corrupted inputs are not distributed here.
Provide an authorized dataset separately. Split CSVs, saved metrics, selected model
weights and predictions are retained; frozen embeddings are a private release download.
Historical absolute paths inside result files are provenance, not runtime dependencies.

## Provenance and maintenance

Source: `yes-yes-company/THESIS`, commit `da7ad651d7e3eaebfa2216950a5f2dd0d0640a1a`.
See [NOTICE.md](NOTICE.md), [provenance/SOURCE_MANIFEST.json](provenance/SOURCE_MANIFEST.json),
and [provenance/CHANGES.md](provenance/CHANGES.md). Historical experiment runbooks preserve
the original chronology and may contain old machine paths or superseded status text;
the root runbook and verification record take precedence. Canonical artifacts remain
frozen. Update shared helper copies only deliberately, with source hashes and parity checks.

Related, separate research: [QCNN](https://github.com/Shafin-paranoid-Ahmed/qcnn) and
[QEMCMC](https://github.com/Shafin-paranoid-Ahmed/qemcmc). They use different protocols and are not part
of this benchmark. No new experiments were run as part of extraction.
