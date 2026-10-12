# Graph Report - e3_beat_e2_attack  (2026-10-02)

## Corpus Check
- 34 files · ~17,287 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 183 file(s) not represented in the graph (top: .csv 149, .pt 30, .log 4)

## Summary
- 57 nodes · 92 edges · 7 communities (4 shown, 3 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `da7ad651`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- diagnose_mid_stage_errors.py
- main
- E2MatchedKResidual
- .__init__
- evaluate_and_save
- README.md

## God Nodes (most connected - your core abstractions)
1. `main()` - 15 edges
2. `E2MatchedKResidual` - 6 edges
3. `E3VQCKResidual` - 6 edges
4. `evaluate_and_save()` - 6 edges
5. `main()` - 4 edges
6. `PairTargetedLoss` - 4 edges
7. `build_model()` - 4 edges
8. `alpha_parameter_init()` - 3 edges
9. `middle_module()` - 3 edges
10. `middle_parameters()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `PairTargetedLoss`  [EXTRACTED]
  e3_beat_e2_attack/scripts/train_warmstart_e3.py → e3_beat_e2_attack/scripts/train_warmstart_e3.py  _Bridges community 4 → community 2_
- `main()` --calls--> `build_model()`  [EXTRACTED]
  e3_beat_e2_attack/scripts/train_warmstart_e3.py → e3_beat_e2_attack/scripts/train_warmstart_e3.py  _Bridges community 3 → community 2_
- `main()` --calls--> `predict_limited()`  [EXTRACTED]
  e3_beat_e2_attack/scripts/train_warmstart_e3.py → e3_beat_e2_attack/scripts/train_warmstart_e3.py  _Bridges community 5 → community 2_

## Import Cycles
- None detected.

## Communities (7 total, 3 thin omitted)

### Community 0 - "diagnose_mid_stage_errors.py"
Cohesion: 0.21
Nodes (4): display_path(), main(), parse_args(), stage_name()

### Community 2 - "main"
Cohesion: 0.22
Nodes (10): append_history(), main(), metric_float(), middle_module(), middle_parameters(), parse_args(), save_checkpoint(), set_requires_grad() (+2 more)

### Community 3 - "E2MatchedKResidual"
Cohesion: 0.31
Nodes (3): build_model(), E2MatchedKResidual, E3VQCKResidual

### Community 5 - "evaluate_and_save"
Cohesion: 0.40
Nodes (5): alpha_raw_value(), alpha_value(), evaluate_and_save(), load_checkpoint_state(), predict_limited()

## Knowledge Gaps
- **1 isolated node(s):** `E3 Beat E2 Attack`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 19 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `E3VQCKResidual` connect `E2MatchedKResidual` to `train_warmstart_e3.py`, `.__init__`?**
  _High betweenness centrality (0.111) - this node is a cross-community bridge._
- **Why does `E2MatchedKResidual` connect `E2MatchedKResidual` to `train_warmstart_e3.py`, `.__init__`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Why does `PairTargetedLoss` connect `.__init__` to `train_warmstart_e3.py`, `main`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **What connects `E3 Beat E2 Attack` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._