# Extraction verification — 2026-10-12

Local standalone checks passed on 2026-10-12. See the dated publication update below for current distribution status.

- E3-4, E3-8, residual and re-uploading checkpoints pass synthetic forward/backward checks.
- The re-uploading paired deep-classical checkpoint loads and differentiates.
- Train/validation/test embedding hashes, 1280+64 dimensions, row ordering and class labels match the approved manifests.
- Five experiment entrypoints passed --help.

All copied source-file hashes were checked against the current parent checkout.
Scientific CSV/JSON/NPZ/checkpoint artifacts remain byte-identical. The extraction
manifest lists deliberate code/documentation adaptations. Python syntax and fixed
split counts passed. All local Markdown file targets resolve. A targeted credential
pattern scan found no matches; caches and environment/session files are excluded.

Checks used the existing Python 3.10.20 research interpreter, torch 2.11.0+cu128,
torchvision 0.26.0+cu128, scikit-learn 1.7.2 and PennyLane 0.42.3. Source imports were
restricted to this checkout; the original thesis code is not a runtime dependency.
No fresh dependency installation, full retraining, raw-data transfer, full corruption
inference matrix or new scientific experiment was performed. Synthetic quantum checks
used CPU default.qubit; they are not a simulator performance benchmark.

Git preserves original file bytes, including historical line endings. Whitespace
checks cover newly written root documentation/scripts rather than rewriting frozen
research CSVs. Large embeddings are checksummed local artifacts and a local archive, excluded from Git.

Fresh local clones passed standalone verification. Baseline inputs were restored
from the local ZIP bundle; quantum and robustness inputs were restored from that
fresh baseline clone. All restored hashes, dimensions, sample ordering and labels
passed. Canonical E2 predictions and all 361 robustness Macro-F1 values were again
verified from the clones. Quantum clone checks also cover original E3-4, E3-8,
residual, re-uploading and the paired deep-classical checkpoint. Git worktrees were
clean after checks. No original source-code imports or implicit neighboring paths
were required. The machine's global Git checkout hook adds a TokenSave ignore entry;
the repository now includes that ignore rule so fresh checkouts remain clean.

Historical publication status on 2026-10-12: LOCAL ONLY. No GitHub repository, release or Git remote exists
for this extraction. Automatic approval review blocked the proposed artifact upload;
the user explicitly chose to keep everything local. No upload was attempted after
that choice. Future publication requires a new user instruction.


## Private GitHub publication authorization — 2026-10-13

The user subsequently authorized pushing the three committed repositories to their
personal GitHub account as private repositories. This supersedes the earlier local-only
restriction for committed files. It does not authorize uploading the ignored embedding
arrays or archive: they remain local, with ARTIFACTS.json published=false.
Companion links now point to the private repositories. Existing commit authorship is
preserved; this documentation commit uses the same configured user identity without
AI co-author attribution. No scientific code, result or model artifact was changed.
The 2026-10-12 runtime checks remain historical; publication checks verify remote
visibility, branch/tag hashes and clean local worktrees.
