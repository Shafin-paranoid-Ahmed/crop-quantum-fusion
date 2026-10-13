# Validation record

## Fresh install (2026-10-13)

A new virtual environment was created with the pinned `requirements.txt` versions plus
PennyLane 0.42.3 (torch 2.11.0 CPU build) on **Python 3.12** (Windows 11, CPU only; the documented
target is Python 3.10), with PYTHONPATH cleared:

- `scripts/verify.py` and `scripts/verify.py --quantum`: PASS (manifest hashes, split counts, syntax; E3-8, residual, E3-4, re-uploading and deep-control checkpoint forward/backward checks).

The large-embedding `--artifacts` mode was not rerun.

## Checks run (2026-10-12, existing environment)

- E3-4, E3-8, residual and re-uploading checkpoints pass synthetic forward/backward checks.
- The re-uploading paired deep-classical checkpoint loads and differentiates.
- Train/validation/test embedding hashes, 1280+64 dimensions, row ordering and class labels match the approved manifests.
- Five experiment entrypoints passed --help.

Scientific CSV/JSON/NPZ/checkpoint artifacts are hash-checked; see [MANIFEST.json](MANIFEST.json).
Python syntax and fixed split counts passed. All local Markdown file targets resolve.
A targeted credential-pattern scan found no matches; caches and environment/session files are excluded.

Fresh local clones passed `scripts/verify.py`. Baseline inputs were restored from the local
ZIP bundle; quantum and robustness inputs were restored from a fresh baseline clone. All
restored hashes, dimensions, sample ordering and labels passed. Canonical E2 predictions and
all 361 robustness Macro-F1 values were re-verified from the clones. Quantum clone checks also
cover original E3-4, E3-8, residual, re-uploading and the paired deep-classical checkpoint.

Environment: Python 3.10.20, torch 2.11.0+cu128, torchvision 0.26.0+cu128, scikit-learn 1.7.2
and PennyLane 0.42.3. Synthetic quantum checks used CPU default.qubit; they are not a simulator
performance benchmark.

## Not performed

No full retraining, raw-data transfer, full corruption inference matrix or new scientific
experiment was performed.

## Distribution

Large embedding arrays (`ARTIFACTS.json`, `published: false`) are kept as a local checksummed
bundle and are not distributed with the repository.
