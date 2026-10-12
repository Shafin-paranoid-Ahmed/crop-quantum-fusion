# Current standalone runbook

Run from this repository root in native Windows PowerShell. The commands below
are examples for manual use; extraction does not launch training or regenerate data.

## Environment

Create a Python 3.10 environment and select an appropriate PyTorch wheel for your machine.
The source environment used torch 2.11.0+cu128 and torchvision 0.26.0+cu128;
the portable requirements omit the CUDA suffix. The original pins are preserved in
`provenance/source-requirements.txt`.

```powershell
py -3.10 -m venv .venv
$PY = Join-Path $PWD '.venv/Scripts/python.exe'
& $PY -m pip install -r requirements-quantum.txt
& $PY scripts/verify.py --quantum
```

## Frozen embedding inputs

The Git checkout includes the small checkpoints and CSV evidence. Large NPZ inputs
are excluded from Git. Sign in with GitHub CLI (`gh auth login`) using an account with
access to the private baseline repository, then run:

```powershell
& $PY scripts/fetch_artifacts.py --download
& $PY scripts/verify.py --artifacts --quantum
```

Alternatively, copy verified inputs from a baseline checkout that already has them:

```powershell
& $PY scripts/fetch_artifacts.py --from-directory ../crop-multimodal-baselines
```

The helper refuses to overwrite differing artifacts and validates SHA-256.
It copies only paths listed in ARTIFACTS.json, without extracting arbitrary archive paths.
Raw datasets are separate inputs and are not included in this release.

## Reproduction examples

Use new run names and output directories when rerunning. Do not overwrite archived
results or manually promote a new model to canonical without an explicit decision.
The validation split selects checkpoints; the test set is not a tuning target.

The downloaded embeddings support head experiments without raw image/sensor data.
All input utilities and warm-start checkpoints are local to this checkout.

```powershell
& $PY e3_quantum_fusion/scripts/train_e3.py --help
& $PY e3_early_rescue/scripts/train_rescue.py --help
& $PY e3_bottleneck_sweep/scripts/train_bottleneck.py --help
& $PY e3_beat_e2_attack/scripts/train_warmstart_e3.py --help
& $PY e3_bottleneck_sweep/scripts/train_reupload.py --help
& $PY scripts/verify.py --quantum --artifacts
```

Preserve paired controls and parameter counts when changing architecture or loss.
Re-uploading and residual studies are separate variants, not interchangeable E3 checkpoints.
The dedicated IoT robustness scripts are historical experiment diagnostics; the unified
cross-model benchmark is maintained by crop-corruption-robustness.
