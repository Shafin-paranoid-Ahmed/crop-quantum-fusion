# E3 Reupload Run Commands

Last updated: 2026-07-01

Use the project venv explicitly:

```powershell
cd "G:\FOR THESIS\MAIN"
$PY = "G:\FOR THESIS\MAIN\.venv-e1\Scripts\python.exe"
```

## Hardware Note

`lightning.gpu` is not installed in `.venv-e1`, so the E3 re-upload quantum model uses `lightning.qubit` on CPU. One quantum training process may not saturate the whole machine. To use more hardware, run independent seeds/variants in parallel with controlled CPU thread counts.

The classical `e2_deep` comparator uses CUDA automatically when available unless `--cpu` is passed.

## Smoke Checks

Classical comparator smoke:

```powershell
& $PY -u e3_bottleneck_sweep\scripts\train_reupload.py --model e2_deep --run-name smoke_e2_deep_reupload --rounds 2 --batch-size 512 --middle-only-epochs 1 --joint-epochs 1 --patience 2 --max-train-batches 2 --max-eval-batches 2 --progress-batches 1
```

Quantum re-upload smoke:

```powershell
$env:OMP_NUM_THREADS="8"
$env:MKL_NUM_THREADS="8"
& $PY -u e3_bottleneck_sweep\scripts\train_reupload.py --model e3_reupload --run-name smoke_e3_8_reupload --rounds 2 --layers-per-round 4 --sim lightning.qubit --batch-size 64 --middle-only-epochs 1 --joint-epochs 1 --patience 2 --max-train-batches 2 --max-eval-batches 2 --progress-batches 1 --torch-threads 8 --interop-threads 1
```


## Faster Diagnostic Quantum Run

Use this first if the full quantum run is too slow. It keeps the same architecture but validates every 3 epochs on 8 validation batches during training, then performs full final validation/test evaluation at the end.

```powershell
$env:OMP_NUM_THREADS="12"
$env:MKL_NUM_THREADS="12"
& $PY -u e3_bottleneck_sweep\scripts\train_reupload.py --model e3_reupload --run-name e3_8_reupload_r2_s42_fastdiag --seed 42 --rounds 2 --layers-per-round 4 --sim lightning.qubit --batch-size 256 --middle-only-epochs 5 --joint-epochs 25 --patience 8 --middle-lr 0.005 --joint-middle-lr 0.001 --joint-head-lr 0.0001 --progress-batches 20 --torch-threads 12 --interop-threads 1 --eval-every 3 --train-val-batches 8
```

If this looks promising, run the full command afterward.
## First Full Pair

Run the faster classical comparator first:

```powershell
& $PY -u e3_bottleneck_sweep\scripts\train_reupload.py --model e2_deep --run-name e2_matched_8_deep_r2_s42 --seed 42 --rounds 2 --batch-size 512 --middle-only-epochs 15 --joint-epochs 60 --patience 20 --middle-lr 0.005 --joint-middle-lr 0.001 --joint-head-lr 0.0001 --progress-batches 25
```

Run the quantum re-upload model:

```powershell
$env:OMP_NUM_THREADS="12"
$env:MKL_NUM_THREADS="12"
& $PY -u e3_bottleneck_sweep\scripts\train_reupload.py --model e3_reupload --run-name e3_8_reupload_r2_s42 --seed 42 --rounds 2 --layers-per-round 4 --sim lightning.qubit --batch-size 128 --middle-only-epochs 15 --joint-epochs 60 --patience 20 --middle-lr 0.005 --joint-middle-lr 0.001 --joint-head-lr 0.0001 --progress-batches 25 --torch-threads 12 --interop-threads 1 --eval-every 2 --train-val-batches 16
```

## Background Quantum Run

```powershell
$env:OMP_NUM_THREADS="12"
$env:MKL_NUM_THREADS="12"
$out="G:\FOR THESIS\MAIN\e3_bottleneck_sweep\e3_8_reupload_r2_s42_stdout.log"
$err="G:\FOR THESIS\MAIN\e3_bottleneck_sweep\e3_8_reupload_r2_s42_stderr.log"
Remove-Item -LiteralPath $out -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $err -Force -ErrorAction SilentlyContinue
$p = Start-Process -FilePath $PY -ArgumentList @("-u","e3_bottleneck_sweep\scripts\train_reupload.py","--model","e3_reupload","--run-name","e3_8_reupload_r2_s42","--seed","42","--rounds","2","--layers-per-round","4","--sim","lightning.qubit","--batch-size","128","--middle-only-epochs","15","--joint-epochs","60","--patience","20","--middle-lr","0.005","--joint-middle-lr","0.001","--joint-head-lr","0.0001","--progress-batches","25","--torch-threads","12","--interop-threads","1") -WorkingDirectory "G:\FOR THESIS\MAIN" -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
$p.Id
```

Monitor:

```powershell
Get-Content -LiteralPath "G:\FOR THESIS\MAIN\e3_bottleneck_sweep\e3_8_reupload_r2_s42_stdout.log" -Wait -Tail 30
Get-Content -LiteralPath "G:\FOR THESIS\MAIN\e3_bottleneck_sweep\e3_8_reupload_r2_s42_stderr.log" -Wait -Tail 30
```

## Parallel Hardware Utilization Option

If one quantum run underutilizes the machine, run independent seeds in parallel. Use fewer threads per process to avoid oversubscription.

```powershell
$PY = "G:\FOR THESIS\MAIN\.venv-e1\Scripts\python.exe"
$seeds = @(42,13,37)
foreach ($seed in $seeds) {
  $env:OMP_NUM_THREADS="4"
  $env:MKL_NUM_THREADS="4"
  $run = "e3_8_reupload_r2_s$seed"
  $out = "G:\FOR THESIS\MAIN\e3_bottleneck_sweep\${run}_stdout.log"
  $err = "G:\FOR THESIS\MAIN\e3_bottleneck_sweep\${run}_stderr.log"
  Remove-Item -LiteralPath $out -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath $err -Force -ErrorAction SilentlyContinue
  Start-Process -FilePath $PY -ArgumentList @("-u","e3_bottleneck_sweep\scripts\train_reupload.py","--model","e3_reupload","--run-name",$run,"--seed",$seed,"--rounds","2","--layers-per-round","4","--sim","lightning.qubit","--batch-size","128","--middle-only-epochs","15","--joint-epochs","60","--patience","20","--middle-lr","0.005","--joint-middle-lr","0.001","--joint-head-lr","0.0001","--progress-batches","25","--torch-threads","4","--interop-threads","1") -WorkingDirectory "G:\FOR THESIS\MAIN" -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden
}
```

Only use the parallel option when you are intentionally running multiple seeds. Do not compare incomplete runs.

## Outputs

Each run writes:

```text
e3_bottleneck_sweep/results/<run_name>_metrics.json
e3_bottleneck_sweep/results/<run_name>_summary.csv
e3_bottleneck_sweep/results/<run_name>_history.csv
e3_bottleneck_sweep/results/<run_name>_<best_macro|best_mid>_test_predictions.csv
e3_bottleneck_sweep/results/confusion_matrices/<run_name>_<best_macro|best_mid>_test_confusion_matrix.csv
e3_bottleneck_sweep/checkpoints/best_<run_name>_<macro|mid>.pt
```