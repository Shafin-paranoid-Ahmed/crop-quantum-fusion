"""Train E3-8 data re-uploading VQC against a matched deep E2 comparator.

This is the remaining advanced E3 attack from docs/by_stage/06_e3_attack:
E3-8-reupload vs E2-matched-8-deep. Both models reuse the same frozen E2
embedding interface and warm-start projection/readout from best_e2_matched_8.pt.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn

SWEEP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SWEEP_ROOT.parent
E2_SCRIPTS = PROJECT_ROOT / "e2_multimodal_fusion" / "scripts"
E3_ROOT = PROJECT_ROOT / "e3_quantum_fusion"
for _p in (PROJECT_ROOT, SWEEP_ROOT / "scripts", E2_SCRIPTS, E3_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.thesis_eval import metrics_dict, per_class_metrics, save_confusion_matrix, save_predictions  # noqa: E402
from train_bottleneck import E2MatchedK, FusionProjectionK  # noqa: E402
from train_e2_fusion import load_embedding_manifest, load_split, make_loader, predict, set_seed  # noqa: E402

FUSED_DIM = 1344
HIDDEN = 128
BOTTLENECK = 8
NUM_CLASSES = 25
ANGLE_SCALE = 3.141592653589793
FEATURE_DIR = PROJECT_ROOT / "e2_multimodal_fusion" / "features"
SOURCE_CHECKPOINT = SWEEP_ROOT / "checkpoints" / "best_e2_matched_8.pt"
RESULTS_DIR = SWEEP_ROOT / "results"
CHECKPOINT_DIR = SWEEP_ROOT / "checkpoints"
E2_MACRO_F1_BAR = 0.973356


class ReuploadVQCLayer(nn.Module):
    """8D -> 8D VQC with repeated input re-uploading inside one circuit."""

    def __init__(self, n_qubits=BOTTLENECK, rounds=2, layers_per_round=4, sim="lightning.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.rounds = rounds
        self.layers_per_round = layers_per_round
        self.sim = sim

        dev = qml.device(sim, wires=n_qubits)
        diff_method = "backprop" if sim == "default.qubit" else "adjoint"

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            for r in range(rounds):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
                for layer in range(layers_per_round):
                    for q in range(n_qubits):
                        qml.RX(weights[r, layer, q, 0], wires=q)
                        qml.RY(weights[r, layer, q, 1], wires=q)
                    for q in range(n_qubits):
                        qml.CNOT(wires=[q, (q + 1) % n_qubits])
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

        weight_shapes = {"weights": (rounds, layers_per_round, n_qubits, 2)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        return self.qlayer(x)


class E3ReuploadK(nn.Module):
    def __init__(self, rounds=2, layers_per_round=4, num_classes=NUM_CLASSES, sim="lightning.qubit"):
        super().__init__()
        self.bottleneck = BOTTLENECK
        self.reupload_rounds = rounds
        self.vqc_layers_per_round = layers_per_round
        self.projection = FusionProjectionK(BOTTLENECK, fused_dim=FUSED_DIM, hidden=HIDDEN)
        self.vqc = ReuploadVQCLayer(
            n_qubits=BOTTLENECK,
            rounds=rounds,
            layers_per_round=layers_per_round,
            sim=sim,
        )
        self.readout = nn.Linear(BOTTLENECK, num_classes)

    def forward(self, x):
        z = ANGLE_SCALE * torch.tanh(self.projection(x))
        z = self.vqc(z)
        return self.readout(z)

    def post_bottleneck_parameters(self):
        return self.vqc.parameters()


class E2MatchedKDeep(nn.Module):
    def __init__(self, depth=2, num_classes=NUM_CLASSES):
        super().__init__()
        self.bottleneck = BOTTLENECK
        self.depth = depth
        self.projection = FusionProjectionK(BOTTLENECK, fused_dim=FUSED_DIM, hidden=HIDDEN)
        self.middle = nn.ModuleList([nn.Linear(BOTTLENECK, BOTTLENECK, bias=False) for _ in range(depth)])
        self.readout = nn.Linear(BOTTLENECK, num_classes)

    def forward(self, x):
        z = torch.tanh(self.projection(x))
        for layer in self.middle:
            z = torch.tanh(layer(z))
        return self.readout(z)

    def post_bottleneck_parameters(self):
        for layer in self.middle:
            yield from layer.parameters()


def parse_args():
    parser = argparse.ArgumentParser(description="Train E3 re-upload VQC or E2 deep matched comparator")
    parser.add_argument("--model", choices=["e3_reupload", "e2_deep"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--layers-per-round", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--middle-only-epochs", type=int, default=15)
    parser.add_argument("--joint-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--middle-lr", type=float, default=5e-3)
    parser.add_argument("--joint-middle-lr", type=float, default=1e-3)
    parser.add_argument("--joint-head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--sim", default="lightning.qubit")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--progress-batches", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--interop-threads", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from latest per-epoch state for this run name")
    parser.add_argument("--final-eval-only", action="store_true", help="Skip training and export metrics from existing best checkpoints")
    return parser.parse_args()


def count_param_split(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    post = sum(p.numel() for p in model.post_bottleneck_parameters() if p.requires_grad)
    return total, post


def set_requires_grad(module, enabled):
    for param in module.parameters():
        param.requires_grad = enabled


def middle_parameters(model):
    if hasattr(model, "vqc"):
        return list(model.vqc.parameters())
    params = []
    for layer in model.middle:
        params.extend(list(layer.parameters()))
    return params


def load_warm_projection_readout(model, num_classes):
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing warm-start checkpoint: {SOURCE_CHECKPOINT}")
    e2 = E2MatchedK(BOTTLENECK, num_classes=num_classes)
    e2.load_state_dict(torch.load(SOURCE_CHECKPOINT, map_location="cpu"))
    model.projection.load_state_dict(e2.projection.state_dict())
    model.readout.load_state_dict(e2.readout.state_dict())


def train_one_epoch_limited(model, loader, criterion, optimizer, device, max_batches=None, progress_batches=25):
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch_idx, (x, y) in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(y)
        total_count += len(y)
        if progress_batches and batch_idx % progress_batches == 0:
            print(f"  batch {batch_idx} loss={loss.item():.4f}", flush=True)
    return total_loss / max(total_count, 1)


def predict_limited(model, loader, criterion, device, max_batches=None):
    if max_batches is None:
        return predict(model, loader, criterion, device)
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_targets = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader, start=1):
            if batch_idx > max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            total_loss += float(loss.item()) * len(y)
            total_count += len(y)
            all_targets.append(y.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    return {
        "loss": total_loss / max(total_count, 1),
        "targets": np.concatenate(all_targets),
        "preds": np.concatenate(all_preds),
        "probs": np.concatenate(all_probs),
    }


def make_optimizer(model, args, phase):
    if phase == "middle_only":
        return torch.optim.Adam(middle_parameters(model), lr=args.middle_lr, weight_decay=args.weight_decay)
    return torch.optim.Adam(
        [
            {"params": middle_parameters(model), "lr": args.joint_middle_lr},
            {"params": list(model.projection.parameters()) + list(model.readout.parameters()), "lr": args.joint_head_lr},
        ],
        weight_decay=args.weight_decay,
    )


def set_phase_requires_grad(model, phase):
    if phase == "middle_only":
        set_requires_grad(model.projection, False)
        set_requires_grad(model.readout, False)
        for p in middle_parameters(model):
            p.requires_grad = True
        model.projection.eval()
        model.readout.eval()
    else:
        set_requires_grad(model, True)


def metric_float(value):
    return float("nan") if value is None else float(value)


def build_model(args, num_classes):
    if args.model == "e3_reupload":
        return E3ReuploadK(
            rounds=args.rounds,
            layers_per_round=args.layers_per_round,
            num_classes=num_classes,
            sim=args.sim,
        )
    return E2MatchedKDeep(depth=args.rounds, num_classes=num_classes)


def atomic_torch_save(payload, path):
    # Windows ACLs on this checkpoint folder deny atomic rename for .pt files.
    # Direct torch.save is the established working write path in this repo.
    torch.save(payload, path)


def atomic_history_write(history, path):
    # Direct write is used because this Windows folder ACL denies atomic rename.
    pd.DataFrame(history).to_csv(path, index=False)


def completed_phase_epochs(history):
    completed = defaultdict(int)
    for row in history:
        completed[str(row["phase"])] = max(completed[str(row["phase"])] , int(row["phase_epoch"]))
    return completed


def save_latest_state(path, model, optimizer, scheduler, args, phase, global_epoch, history, best_macro, best_mid, best_macro_epoch, best_mid_epoch, stale):
    atomic_torch_save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "phase": phase,
            "global_epoch": global_epoch,
            "history": history,
            "best_macro": best_macro,
            "best_mid": best_mid,
            "best_macro_epoch": best_macro_epoch,
            "best_mid_epoch": best_mid_epoch,
            "stale": stale,
            "args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    if args.interop_threads is not None:
        torch.set_num_interop_threads(args.interop_threads)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_embedding_manifest(FEATURE_DIR)
    classes = manifest["classes"]
    if manifest["image_embedding_dim"] != 1280 or manifest["iot_embedding_dim"] != 64:
        raise RuntimeError("Embedding manifest dimensions do not match the E2/E3 interface")
    if len(classes) != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} classes, manifest has {len(classes)}")

    x_train, y_train, train_meta = load_split(FEATURE_DIR, "train")
    x_val, y_val, val_meta = load_split(FEATURE_DIR, "val")
    x_test, y_test, test_meta = load_split(FEATURE_DIR, "test")

    class_counts = np.bincount(y_train, minlength=len(classes))
    class_weights = len(y_train) / (len(classes) * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    if args.sim == "lightning.gpu" and not torch.cuda.is_available():
        raise RuntimeError("--sim lightning.gpu requested but CUDA is not available")
    if args.model == "e3_reupload":
        device = torch.device("cuda" if args.sim == "lightning.gpu" and not args.cpu else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader = make_loader(x_train, y_train, args.batch_size, True, args.num_workers)
    val_loader = make_loader(x_val, y_val, args.batch_size, False, args.num_workers)
    test_loader = make_loader(x_test, y_test, args.batch_size, False, args.num_workers)

    model = build_model(args, len(classes))
    load_warm_projection_readout(model, len(classes))
    model = model.to(device)

    total_params, post_params = count_param_split(model)
    expected_e3_post = args.rounds * args.layers_per_round * BOTTLENECK * 2
    expected_e2_post = args.rounds * BOTTLENECK * BOTTLENECK
    if args.model == "e3_reupload" and post_params != expected_e3_post:
        raise RuntimeError(f"Unexpected E3 post-bottleneck params {post_params} != {expected_e3_post}")
    if args.model == "e2_deep" and post_params != expected_e2_post:
        raise RuntimeError(f"Unexpected E2 post-bottleneck params {post_params} != {expected_e2_post}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    checkpoint_macro_path = CHECKPOINT_DIR / f"best_{args.run_name}_macro.pt"
    checkpoint_mid_path = CHECKPOINT_DIR / f"best_{args.run_name}_mid.pt"

    history_path = RESULTS_DIR / f"{args.run_name}_history.csv"
    latest_state_path = CHECKPOINT_DIR / f"latest_{args.run_name}_state.pt"

    history = []
    best_macro = -1.0
    best_mid = -1.0
    best_macro_epoch = 0
    best_mid_epoch = 0
    stale = 0
    global_epoch = 0
    resume_state = None

    if args.resume or args.final_eval_only:
        if latest_state_path.exists():
            resume_state = torch.load(latest_state_path, map_location=device)
            model.load_state_dict(resume_state["model_state"])
            history = list(resume_state.get("history", []))
            best_macro = float(resume_state.get("best_macro", best_macro))
            best_mid = float(resume_state.get("best_mid", best_mid))
            best_macro_epoch = int(resume_state.get("best_macro_epoch", best_macro_epoch))
            best_mid_epoch = int(resume_state.get("best_mid_epoch", best_mid_epoch))
            stale = int(resume_state.get("stale", stale))
            global_epoch = int(resume_state.get("global_epoch", global_epoch))
            print(f"Loaded resume state: {latest_state_path} at global_epoch={global_epoch}", flush=True)
        elif args.resume:
            print(f"--resume requested but no latest state exists at {latest_state_path}; starting from warm start", flush=True)
        elif args.final_eval_only:
            print(f"--final-eval-only without latest state; using existing best checkpoints only", flush=True)

    print(f"Model: {args.model}  run={args.run_name}  device={device}", flush=True)
    print(f"Rounds={args.rounds} layers_per_round={args.layers_per_round if args.model == 'e3_reupload' else 'n/a'}", flush=True)
    print(f"Rows train/val/test: {len(y_train)}/{len(y_val)}/{len(y_test)}", flush=True)
    print(f"Trainable params total={total_params} post_bottleneck={post_params}", flush=True)
    print(f"Warm-start projection/readout: {SOURCE_CHECKPOINT}", flush=True)
    print(f"Latest resume state: {latest_state_path}", flush=True)
    print(f"Torch threads={torch.get_num_threads()} interop={torch.get_num_interop_threads()} OMP={os.environ.get('OMP_NUM_THREADS')} MKL={os.environ.get('MKL_NUM_THREADS')}", flush=True)

    if not args.final_eval_only:
        phases = [("middle_only", args.middle_only_epochs), ("joint", args.joint_epochs)]
        for phase, phase_epochs in phases:
            if phase_epochs <= 0:
                continue
            completed = completed_phase_epochs(history)
            start_local_epoch = completed[phase] + 1
            if start_local_epoch > phase_epochs:
                print(f"Skipping completed phase={phase} epochs={phase_epochs}", flush=True)
                continue
            set_phase_requires_grad(model, phase)
            optimizer = make_optimizer(model, args, phase)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=args.scheduler_factor,
                patience=args.scheduler_patience,
            )
            if resume_state is not None and resume_state.get("phase") == phase and resume_state.get("optimizer_state") is not None:
                optimizer.load_state_dict(resume_state["optimizer_state"])
                if resume_state.get("scheduler_state") is not None:
                    scheduler.load_state_dict(resume_state["scheduler_state"])
                print(f"Restored optimizer/scheduler for phase={phase}", flush=True)
            print(f"Starting phase={phase} epochs={phase_epochs} from local_epoch={start_local_epoch}", flush=True)
            for local_epoch in range(start_local_epoch, phase_epochs + 1):
                global_epoch += 1
                train_loss = train_one_epoch_limited(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    max_batches=args.max_train_batches,
                    progress_batches=args.progress_batches,
                )
                val_output = predict_limited(model, val_loader, criterion, device, max_batches=args.max_eval_batches)
                val_metrics = metrics_dict(val_output, classes)
                val_mid = val_metrics["per_stage_recall"]["2"]
                val_batches = args.max_eval_batches if args.max_eval_batches is not None else len(val_loader)
                row = {
                    "epoch": global_epoch,
                    "phase": phase,
                    "phase_epoch": local_epoch,
                    "train_loss": float(train_loss),
                    "lr_middle": float(optimizer.param_groups[0]["lr"]),
                    "lr_head": float(optimizer.param_groups[-1]["lr"]),
                    "val_batches": val_batches,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_macro_recall": val_metrics["macro_recall"],
                    "val_early_stage_recall": val_metrics["early_stage_recall"],
                    "val_mid_stage_recall": val_mid,
                    "val_severe_stage_recall": val_metrics["per_stage_recall"]["3"],
                    "val_ece": val_metrics["ece"],
                }
                history.append(row)
                print(
                    "epoch %03d phase=%s train_loss=%.4f val_loss=%.4f val_acc=%.4f val_macro_f1=%.4f val_mid=%.4f val_severe=%.4f val_ece=%.5f"
                    % (
                        global_epoch,
                        phase,
                        train_loss,
                        val_metrics["loss"],
                        val_metrics["accuracy"],
                        val_metrics["macro_f1"],
                        metric_float(val_mid),
                        metric_float(val_metrics["per_stage_recall"]["3"]),
                        val_metrics["ece"],
                    ),
                    flush=True,
                )
                scheduler.step(val_metrics["macro_f1"])

                improved = False
                if val_metrics["macro_f1"] > best_macro:
                    best_macro = val_metrics["macro_f1"]
                    best_macro_epoch = global_epoch
                    atomic_torch_save(model.state_dict(), checkpoint_macro_path)
                    improved = True
                if val_mid is not None and val_mid > best_mid:
                    best_mid = val_mid
                    best_mid_epoch = global_epoch
                    atomic_torch_save(model.state_dict(), checkpoint_mid_path)
                    improved = True
                stale = 0 if improved else stale + 1
                atomic_history_write(history, history_path)
                save_latest_state(
                    latest_state_path,
                    model,
                    optimizer,
                    scheduler,
                    args,
                    phase,
                    global_epoch,
                    history,
                    best_macro,
                    best_mid,
                    best_macro_epoch,
                    best_mid_epoch,
                    stale,
                )
                if stale >= args.patience:
                    print(f"early stopping at global epoch {global_epoch}", flush=True)
                    break
            resume_state = None
            if stale >= args.patience:
                break

    if history:
        atomic_history_write(history, history_path)
    if not checkpoint_macro_path.exists():
        raise FileNotFoundError(f"Missing best macro checkpoint: {checkpoint_macro_path}")
    if not checkpoint_mid_path.exists():
        checkpoint_mid_path.write_bytes(checkpoint_macro_path.read_bytes())
        best_mid_epoch = best_macro_epoch
    metrics_by_checkpoint = {}
    for checkpoint_name, checkpoint_path, best_epoch in [
        ("best_macro", checkpoint_macro_path, best_macro_epoch),
        ("best_mid", checkpoint_mid_path, best_mid_epoch),
    ]:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        val_output = predict(model, val_loader, criterion, device)
        test_output = predict(model, test_loader, criterion, device)
        val_metrics = metrics_dict(val_output, classes)
        test_metrics = metrics_dict(test_output, classes)
        test_metrics["per_class"] = per_class_metrics(test_output, classes)
        save_predictions(RESULTS_DIR / f"{args.run_name}_{checkpoint_name}_val_predictions.csv", val_meta, val_output, classes)
        save_predictions(RESULTS_DIR / f"{args.run_name}_{checkpoint_name}_test_predictions.csv", test_meta, test_output, classes)
        save_confusion_matrix(
            RESULTS_DIR / "confusion_matrices" / f"{args.run_name}_{checkpoint_name}_test_confusion_matrix.csv",
            test_output,
            classes,
        )
        metrics_by_checkpoint[checkpoint_name] = {
            "best_epoch": best_epoch,
            "val": val_metrics,
            "test": test_metrics,
            "beats_e2_macro_f1_bar": bool(test_metrics["macro_f1"] > E2_MACRO_F1_BAR),
        }

    if args.model == "e3_reupload":
        architecture = (
            "Linear(1344,128)-BN-ReLU-Dropout(0.1)-Linear(128,8)-tanh*pi-"
            f"[AngleEmbedding(x)+VQC({args.layers_per_round} layers)]x{args.rounds}-Linear(8,25)"
        )
        qubits = BOTTLENECK
        post_match_note = f"E3 post params = rounds*layers*qubits*2 = {post_params}; match E2 with depth={args.rounds} only when layers_per_round=4."
        sim = args.sim
    else:
        architecture = (
            "Linear(1344,128)-BN-ReLU-Dropout(0.1)-Linear(128,8)-tanh-"
            f"[Linear(8,8,bias=False)-tanh]x{args.rounds}-Linear(8,25)"
        )
        qubits = None
        post_match_note = f"E2 post params = depth*8*8 = {post_params}; matches E3 rounds={args.rounds}, layers_per_round=4."
        sim = None

    metrics = {
        "experiment_group": "e3_reupload_attack",
        "seed": args.seed,
        "run_name": args.run_name,
        "model": args.model,
        "architecture": architecture,
        "qubits": qubits,
        "bottleneck_dim": BOTTLENECK,
        "reupload_rounds": args.rounds if args.model == "e3_reupload" else None,
        "vqc_layers_per_round": args.layers_per_round if args.model == "e3_reupload" else None,
        "classical_deep_depth": args.rounds if args.model == "e2_deep" else None,
        "device": str(device),
        "sim": sim,
        "input_dim": x_train.shape[1],
        "image_embedding_dim": manifest["image_embedding_dim"],
        "iot_embedding_dim": manifest["iot_embedding_dim"],
        "num_classes": len(classes),
        "batch_size": args.batch_size,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "middle_only_epochs_requested": args.middle_only_epochs,
        "joint_epochs_requested": args.joint_epochs,
        "middle_lr": args.middle_lr,
        "joint_middle_lr": args.joint_middle_lr,
        "joint_head_lr": args.joint_head_lr,
        "weight_decay": args.weight_decay,
        "optimizer": "Adam",
        "loss": "weighted_cross_entropy",
        "scheduler": "ReduceLROnPlateau",
        "scheduler_monitor": "validation_macro_f1",
        "train_rows": len(y_train),
        "val_rows": len(y_val),
        "test_rows": len(y_test),
        "total_trainable_params": total_params,
        "post_bottleneck_trainable_params": post_params,
        "post_bottleneck_match_note": post_match_note,
        "warm_start_checkpoint": str(SOURCE_CHECKPOINT),
        "e2_macro_f1_bar": E2_MACRO_F1_BAR,
        "e0_checkpoint": manifest["e0_checkpoint"],
        "e1_checkpoint": manifest["e1_checkpoint"],
        "checkpoints": metrics_by_checkpoint,
        "classes": classes,
    }

    with open(RESULTS_DIR / f"{args.run_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary_rows = []
    for checkpoint_name, data in metrics_by_checkpoint.items():
        test = data["test"]
        summary_rows.append({
            "run_name": args.run_name,
            "model": args.model,
            "checkpoint": checkpoint_name,
            "seed": args.seed,
            "post_bottleneck_params": post_params,
            "best_epoch": data["best_epoch"],
            "accuracy": test["accuracy"],
            "macro_f1": test["macro_f1"],
            "macro_recall": test["macro_recall"],
            "early_stage_recall": test["early_stage_recall"],
            "mid_stage_recall": test["per_stage_recall"]["2"],
            "severe_stage_recall": test["per_stage_recall"]["3"],
            "ece": test["ece"],
            "beats_e2_macro_f1_bar": data["beats_e2_macro_f1_bar"],
        })
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / f"{args.run_name}_summary.csv", index=False)

    best = metrics_by_checkpoint["best_macro"]["test"]
    print(f"saved metrics to {RESULTS_DIR / f'{args.run_name}_metrics.json'}", flush=True)
    print(
        "best_macro_epoch=%d test_macro_f1=%.6f test_acc=%.6f test_mid=%.6f beats_e2=%s"
        % (
            best_macro_epoch,
            best["macro_f1"],
            best["accuracy"],
            best["per_stage_recall"]["2"],
            best["macro_f1"] > E2_MACRO_F1_BAR,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()




