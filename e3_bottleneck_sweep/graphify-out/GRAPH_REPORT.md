# Graph Report - e3_bottleneck_sweep  (2026-10-02)

## Corpus Check
- 16 files · ~30,915 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 66 file(s) not represented in the graph (top: .csv 56, .log 6, (none) 2)

## Summary
- 144 nodes · 284 edges · 9 communities (8 shown, 1 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 5 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `da7ad651`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- evaluate_iot_robustness_stability.py
- evaluate_iot_robustness.py
- train_bottleneck.py
- train_reupload.py
- FusionProjectionK
- compare_bottlenecks.py
- E3 Reupload Run Commands
- pathlib
- E3 Bottleneck Sweep

## God Nodes (most connected - your core abstractions)
1. `main()` - 14 edges
2. `main()` - 13 edges
3. `E2MatchedK` - 12 edges
4. `main()` - 11 edges
5. `FusionProjectionK` - 10 edges
6. `E3VQCK` - 10 edges
7. `E3 Reupload Run Commands` - 8 edges
8. `corrupt_iot()` - 6 edges
9. `main()` - 6 edges
10. `corrupt_iot()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `load_models()` --uses--> `E2MatchedK`  [INFERRED]
  e3_bottleneck_sweep/iot_robustness_stability/evaluate_iot_robustness_stability.py → e3_bottleneck_sweep/scripts/train_bottleneck.py
- `load_models()` --uses--> `E3VQCK`  [INFERRED]
  e3_bottleneck_sweep/iot_robustness_stability/evaluate_iot_robustness_stability.py → e3_bottleneck_sweep/scripts/train_bottleneck.py
- `load_models()` --calls--> `E2MatchedK`  [EXTRACTED]
  e3_bottleneck_sweep/scripts/evaluate_iot_robustness.py → e3_bottleneck_sweep/scripts/train_bottleneck.py
- `load_models()` --calls--> `E3VQCK`  [EXTRACTED]
  e3_bottleneck_sweep/scripts/evaluate_iot_robustness.py → e3_bottleneck_sweep/scripts/train_bottleneck.py
- `main()` --calls--> `count_param_split()`  [EXTRACTED]
  e3_bottleneck_sweep/scripts/evaluate_iot_robustness.py → e3_bottleneck_sweep/scripts/train_bottleneck.py

## Import Cycles
- None detected.

## Communities (9 total, 1 thin omitted)

### Community 0 - "evaluate_iot_robustness_stability.py"
Cohesion: 0.15
Nodes (24): aggregate_mean_std(), apply_contradictory_iot(), apply_gaussian(), apply_missing(), apply_shuffle_iot(), audit_or_raise(), compute_metrics(), corrupt_iot() (+16 more)

### Community 1 - "evaluate_iot_robustness.py"
Cohesion: 0.15
Nodes (20): apply_contradictory_iot(), apply_gaussian(), apply_missing(), apply_shuffle_iot(), compute_metrics(), condition_drop_summary(), corrupt_iot(), expected_calibration_error_10() (+12 more)

### Community 2 - "train_bottleneck.py"
Cohesion: 0.14
Nodes (7): main(), count_param_split(), default_vqc_layers_for_match(), E2MatchedK, E3VQCK, main(), parse_args()

### Community 3 - "train_reupload.py"
Cohesion: 0.18
Nodes (16): atomic_history_write(), atomic_torch_save(), build_model(), completed_phase_epochs(), count_param_split(), load_warm_projection_readout(), main(), make_optimizer() (+8 more)

### Community 4 - "FusionProjectionK"
Cohesion: 0.15
Nodes (4): FusionProjectionK, E2MatchedKDeep, E3ReuploadK, ReuploadVQCLayer

### Community 5 - "compare_bottlenecks.py"
Cohesion: 0.31
Nodes (6): by_label(), delta(), fmt(), load(), main(), row()

### Community 6 - "E3 Reupload Run Commands"
Cohesion: 0.22
Nodes (8): Background Quantum Run, E3 Reupload Run Commands, Faster Diagnostic Quantum Run, First Full Pair, Hardware Note, Outputs, Parallel Hardware Utilization Option, Smoke Checks

### Community 8 - "E3 Bottleneck Sweep"
Cohesion: 0.50
Nodes (3): Commands, E3 Bottleneck Sweep, Models

## Knowledge Gaps
- **9 isolated node(s):** `Models`, `Commands`, `Hardware Note`, `Smoke Checks`, `Faster Diagnostic Quantum Run` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 36 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `E2MatchedK` connect `train_bottleneck.py` to `evaluate_iot_robustness_stability.py`, `evaluate_iot_robustness.py`, `train_reupload.py`, `FusionProjectionK`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Why does `FusionProjectionK` connect `FusionProjectionK` to `train_bottleneck.py`, `train_reupload.py`?**
  _High betweenness centrality (0.053) - this node is a cross-community bridge._
- **Why does `ReuploadVQCLayer` connect `FusionProjectionK` to `train_reupload.py`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `FusionProjectionK` (e.g. with `E2MatchedKDeep` and `E3ReuploadK`) actually correct?**
  _`FusionProjectionK` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Models`, `Commands`, `Hardware Note` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `evaluate_iot_robustness_stability.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1452991452991453 - nodes in this community are weakly interconnected._
- **Should `train_bottleneck.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1422924901185771 - nodes in this community are weakly interconnected._