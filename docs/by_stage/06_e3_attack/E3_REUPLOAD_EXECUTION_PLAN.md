# E3 Reupload Execution Plan

Last updated: 2026-07-01

This records the remaining high-value E3 attack: `E3-8-reupload` versus `E2-matched-8-deep`.

## Goal

Test whether repeated quantum feature encoding closes the remaining E3/E2 gap better than a parameter-matched deeper classical post-bottleneck block.

Primary comparison:

```text
E3-8-reupload:       [AngleEmbedding(x) + VQC layers] x 2
E2-matched-8-deep:   [Linear(8,8,bias=False) + tanh] x 2
```

With `rounds=2` and `layers_per_round=4`:

```text
E3 post-bottleneck params = 2 * 4 * 8 * 2 = 128
E2 post-bottleneck params = 2 * 8 * 8 = 128
```

Both models warm-start `projection.*` and `readout.*` from:

```text
e3_bottleneck_sweep/checkpoints/best_e2_matched_8.pt
```

## Implementation

New script:

```text
e3_bottleneck_sweep/scripts/train_reupload.py
```

Run commands:

```text
e3_bottleneck_sweep/RUN_COMMANDS_REUPLOAD.md
```

## Speed Mode

The trainer supports `--eval-every` and `--train-val-batches` to reduce expensive validation passes during training. Full final validation/test metrics are still computed after training. Use `e3_8_reupload_r2_s42_fastdiag` first when speed matters.

## Hardware Note

`lightning.gpu` is not installed in `.venv-e1`, so the quantum model uses `lightning.qubit` on CPU. Hardware utilization is improved by controlling CPU threads and, when needed, running independent seed processes in parallel. The classical `e2_deep` comparator can use CUDA automatically.

## First Run Pair

```powershell
$PY = "WORKSPACE\.venv-e1\Scripts\python.exe"
& $PY -u e3_bottleneck_sweep\scripts\train_reupload.py --model e2_deep --run-name e2_matched_8_deep_r2_s42 --seed 42 --rounds 2 --batch-size 512 --middle-only-epochs 15 --joint-epochs 60 --patience 20 --middle-lr 0.005 --joint-middle-lr 0.001 --joint-head-lr 0.0001 --progress-batches 25

$env:OMP_NUM_THREADS="12"
$env:MKL_NUM_THREADS="12"
& $PY -u e3_bottleneck_sweep\scripts\train_reupload.py --model e3_reupload --run-name e3_8_reupload_r2_s42 --seed 42 --rounds 2 --layers-per-round 4 --sim lightning.qubit --batch-size 128 --middle-only-epochs 15 --joint-epochs 60 --patience 20 --middle-lr 0.005 --joint-middle-lr 0.001 --joint-head-lr 0.0001 --progress-batches 25 --torch-threads 12 --interop-threads 1 --eval-every 2 --train-val-batches 16
```

## Decision Rule

A useful win requires the E3 re-upload run to beat the original E2-matched-8 Macro F1 bar (`0.973356`) and preferably the paired `E2-matched-8-deep` comparator. Improvements only in ECE or early recall do not count as beating E2.