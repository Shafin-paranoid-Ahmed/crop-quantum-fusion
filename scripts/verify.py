"""Verify the frozen extraction, saved metrics and model wiring without training."""
import argparse
import ast
import csv
import hashlib
import importlib.abc
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class NoQuantum(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('pennylane'):
            raise ImportError('Quantum dependencies deliberately blocked by classical smoke check')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', action='store_true', help='Require and verify the large embedding files')
    parser.add_argument('--quantum', action='store_true', help='Also load and differentiate the quantum models')
    args = parser.parse_args()
    if not args.quantum:
        sys.meta_path.insert(0, NoQuantum())
    manifest = json.loads((ROOT / 'provenance/SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    checked, skipped = 0, 0
    for item in manifest['files']:
        p = ROOT / item['path']
        if item['storage'] == 'release' and not args.artifacts:
            skipped += 1
            continue
        assert p.is_file(), f'Missing source artifact: {p}'
        assert digest(p) == item['extracted_sha256'], f'Changed source artifact: {p}'
        checked += 1
    print(f'Source manifest: {checked} hashes verified; {skipped} release inputs not required', flush=True)
    for split, count in [('train', 46713), ('val', 5193), ('test', 5767)]:
        with (ROOT / 'Data' / f'{split}.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == count, (split, len(rows))
        assert len({row['image'] for row in rows}) == count
    for p in ROOT.rglob('*.py'):
        if any(part in ('.venv', '.git', '.cache') for part in p.parts):
            continue
        ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
    print('Frozen split counts and Python syntax verified', flush=True)

    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import f1_score, accuracy_score
    torch.set_num_threads(2)
    from common.models import E1MLP
    from common.thesis_eval import metrics_dict
    from e2_multimodal_fusion.scripts.train_e2_fusion import E2FusionHead

    metrics_checked = 0
    for p in sorted(ROOT.glob('*/canonical/*_metrics.json')):
        prediction = p.with_name(p.name.replace('_metrics.json', '_test_predictions.csv'))
        if not prediction.exists():
            continue
        rows = pd.read_csv(prediction)
        target = json.loads(p.read_text())['test']
        assert abs(f1_score(rows.label, rows.pred_label, average='macro') - target['macro_f1']) < 1e-10, p
        assert abs(accuracy_score(rows.label, rows.pred_label) - target['accuracy']) < 1e-10, p
        metrics_checked += 1
    if metrics_checked:
        print(f'Canonical prediction files: {metrics_checked} Macro-F1/accuracy pairs recomputed', flush=True)

    x = torch.linspace(-1, 1, 2 * 1344).reshape(2, 1344)
    e2 = E2FusionHead().eval()
    path = ROOT / 'e2_multimodal_fusion/canonical/best_e2_bs512_fusionmlp.pt'
    if path.exists():
        e2.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    assert e2(x).shape == (2, 25)
    e2(x).square().mean().backward()
    e1_path = ROOT / 'e1_iot_only/canonical/best_e1_bs512_mlp128x64.pt'
    if e1_path.exists():
        e1 = E1MLP(54, 25).eval()
        e1.load_state_dict(torch.load(e1_path, map_location='cpu', weights_only=True))
        assert e1(torch.zeros(2, 54)).shape == (2, 25)
        assert e1.embedding(torch.zeros(2, 54)).shape == (2, 64)
        from torchvision.models import mobilenet_v3_large
        e0 = mobilenet_v3_large(weights=None)
        e0.classifier = torch.nn.Sequential(torch.nn.Linear(e0.classifier[0].in_features, 1280), torch.nn.Hardswish(), torch.nn.Dropout(0.2, inplace=True), torch.nn.Linear(1280, 25))
        e0.load_state_dict(torch.load(ROOT / 'e0_image_only/canonical/best_e0_bs768_mobilenetv3large.pth', map_location='cpu', weights_only=True))
        e0.eval()
        with torch.no_grad():
            assert e0(torch.zeros(1, 3, 224, 224)).shape == (1, 25)
        print('Canonical E0/E1/E2 checkpoint loading and synthetic forwards verified', flush=True)

    if (ROOT / 'e2_matched').exists():
        matched = load('e2_matched/scripts/train_matched.py', 'baseline_matched')
        for k in (4, 8):
            if k == 4:
                legacy = load('e2_matched/legacy_four.py', 'legacy_four')
                model = legacy.E2MatchedHead().eval()
            else:
                model = matched.E2MatchedK(k).eval()
            model.load_state_dict(torch.load(ROOT / f'e2_matched/checkpoints/best_e2_matched_{k}.pt', map_location='cpu', weights_only=True))
            assert matched.count_param_split(model) == ((173073, 16) if k == 4 else (173737, 64))
            model(x).square().mean().backward()
        print('Classical matched-4/8 checkpoints, parameter counts and backward pass verified', flush=True)

    if (ROOT / 'corruption_study').exists():
        evaluator = load('corruption_study/eval/run_corruption_eval.py', 'robustness_eval')
        load('corruption_study/gen/regen_embeddings.py', 'regen_embeddings')
        model = evaluator.E2MatchedK(8).eval()
        model.load_state_dict(torch.load(evaluator.E2_8_CHECKPOINT, map_location='cpu', weights_only=True))
        assert model(x).shape == (2, 25)
        summary = load('corruption_study/eval/summarize_results.py', 'summarize_results')
        table = summary.load_metrics(ROOT / 'corruption_study/results')
        predictions_checked = 0
        for p in sorted((ROOT / 'corruption_study/results').glob('*__metrics.json')):
            pred = p.parent / 'predictions' / p.name.replace('__metrics.json', '__predictions.csv')
            if pred.exists():
                frame = pd.read_csv(pred)
                target = json.loads(p.read_text())
                assert abs(f1_score(frame.y_true, frame.y_pred, average='macro') - target['macro_f1']) < 1e-10, p
                predictions_checked += 1
        assert len(table) == 361, len(table)
        assert predictions_checked == len(table), (predictions_checked, len(table))
        print(f'Robustness: {len(table)} conditions loaded; {predictions_checked} Macro-F1 values recomputed', flush=True)

    if args.artifacts:
        artifacts = json.loads((ROOT / 'ARTIFACTS.json').read_text())
        classes = json.loads((ROOT / 'e2_multimodal_fusion/features/e2_embedding_manifest.json').read_text())['classes']
        for item in artifacts['files']:
            assert digest(ROOT / item['path']) == item['sha256']
            with np.load(ROOT / item['path'], allow_pickle=False) as data:
                split = Path(item['path']).name.split('_')[0]
                with (ROOT / f'Data/{split}.csv').open(newline='', encoding='utf-8-sig') as stream:
                    rows = list(csv.DictReader(stream))
                assert data['image_embedding'].shape == (len(rows), 1280)
                assert data['iot_embedding'].shape == (len(rows), 64)
                assert data['image_id'].astype(str).tolist() == [r['image'] for r in rows]
                assert [classes[int(i)] for i in data['y']] == [r['label'] for r in rows]
        print('Embedding hashes, dimensions, sample ordering and labels verified', flush=True)
        # Reproduce canonical E2 predictions from the original frozen TEST inputs.
        if path.exists():
            with np.load(ROOT / 'e2_multimodal_fusion/features/test_embeddings.npz', allow_pickle=False) as data:
                inputs = torch.from_numpy(np.concatenate([data['image_embedding'], data['iot_embedding']], axis=1).astype(np.float32))
            with torch.no_grad():
                predicted = e2(inputs).argmax(1).numpy()
            saved = pd.read_csv(ROOT / 'e2_multimodal_fusion/canonical/e2_bs512_fusionmlp_test_predictions.csv')
            assert [classes[int(i)] for i in predicted] == saved.pred_label.tolist()
            print('All 5,767 canonical E2 TEST predictions reproduced', flush=True)

    if args.quantum:
        if (ROOT / 'common/quantum_models.py').exists():
            from common.quantum_models import E3VQCK, E3VQCKResidual
        else:
            sys.path.insert(0, str(ROOT / 'e3_bottleneck_sweep/scripts'))
            sys.path.insert(0, str(ROOT / 'e3_beat_e2_attack/scripts'))
            from train_bottleneck import E3VQCK
            from train_warmstart_e3 import E3VQCKResidual
        for residual in (False, True):
            if residual:
                model = E3VQCKResidual(8, 4, sim='default.qubit', alpha_bounded=True).eval()
                path = ROOT / 'e3_beat_e2_attack/checkpoints/best_e3_8_warm_residual_bounded_pair_diag_s42_val_macro_f1.pt'
            else:
                model = E3VQCK(8, 4, sim='default.qubit').eval()
                path = ROOT / 'e3_bottleneck_sweep/checkpoints/best_e3_8_vqc.pt'
            state = torch.load(path, map_location='cpu', weights_only=True)
            model.load_state_dict(state.get('state_dict', state))
            output = model(x)
            assert output.shape == (2, 25) and torch.isfinite(output).all()
            output.square().mean().backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        print('Quantum-8 and residual checkpoint forward/backward checks passed', flush=True)
        if (ROOT / 'e3_quantum_fusion/models.py').exists():
            original = load('e3_quantum_fusion/models.py', 'original_four')
            four = original.E3HybridHead(sim='default.qubit').eval()
            four.load_state_dict(torch.load(ROOT / 'e3_quantum_fusion/checkpoints/best_e3_4_vqc.pt', map_location='cpu', weights_only=True))
            four(x).square().mean().backward()
            assert original.count_param_split(four) == (173073, 16)
            reupload = load('e3_bottleneck_sweep/scripts/train_reupload.py', 'reupload')
            for cls, checkpoint in [(reupload.E3ReuploadK, 'best_e3_8_reupload_r2_s42_macro.pt'), (reupload.E2MatchedKDeep, 'best_e2_matched_8_deep_r2_s42_macro.pt')]:
                model = cls(sim='default.qubit') if cls is reupload.E3ReuploadK else cls()
                model.load_state_dict(torch.load(ROOT / 'e3_bottleneck_sweep/checkpoints' / checkpoint, map_location='cpu', weights_only=True))
                model.eval()
                output = model(x)
                assert output.shape == (2, 25) and torch.isfinite(output).all()
                output.square().mean().backward()
            print('Original E3-4 and re-uploading/deep-control checkpoint smokes passed', flush=True)
    elif not (ROOT / 'e3_quantum_fusion').exists():
        assert not any(name.startswith('pennylane') for name in sys.modules)
        print('Classical checks passed with PennyLane imports blocked', flush=True)
    # Imported project modules must live here, never in the parent thesis checkout.
    for name, module in list(sys.modules.items()):
        location = str(getattr(module, '__file__', '')).replace('\\', '/').lower()
        if '/for thesis/main/' in location and '/.venv' not in location and not location.startswith(str(ROOT).replace('\\', '/').lower() + '/'):
            raise AssertionError(f'Parent checkout dependency: {name}: {location}')
    print('PASS: standalone verification; no training or dataset generation', flush=True)

if __name__ == '__main__':
    main()
