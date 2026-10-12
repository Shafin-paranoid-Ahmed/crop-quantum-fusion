# Graph Report - e3_quantum_fusion  (2026-10-02)

## Corpus Check
- 7 files · ~2,898 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 14 file(s) not represented in the graph (top: .csv 8, (none) 2, .pt 2)

## Summary
- 54 nodes · 83 edges · 6 communities (2 shown, 4 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `da7ad651`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- E2MatchedHead
- train_e3.py
- E3HybridHead
- E3 Quantum Fusion
- VQCLayer

## God Nodes (most connected - your core abstractions)
1. `E3HybridHead` - 10 edges
2. `E2MatchedHead` - 9 edges
3. `count_param_split()` - 7 edges
4. `VQCLayer` - 7 edges
5. `E3 Quantum Fusion` - 7 edges
6. `FusionProjection` - 6 edges
7. `main()` - 5 edges
8. `main()` - 4 edges
9. `parse_args()` - 2 edges
10. `E3 fusion heads (Spec B) for the controlled E2-vs-E3 comparison. STUB: layer…` - 1 edges

## Surprising Connections (you probably didn't know these)
- `main()` --uses--> `E2MatchedHead`  [INFERRED]
  e3_quantum_fusion/scripts/smoke_test.py → e3_quantum_fusion/models.py
- `E3HybridHead` --uses--> `VQCLayer`  [INFERRED]
  e3_quantum_fusion/models.py → e3_quantum_fusion/quantum_layer.py
- `main()` --calls--> `E2MatchedHead`  [EXTRACTED]
  e3_quantum_fusion/scripts/train_e3.py → e3_quantum_fusion/models.py
- `main()` --uses--> `E3HybridHead`  [INFERRED]
  e3_quantum_fusion/scripts/smoke_test.py → e3_quantum_fusion/models.py
- `main()` --calls--> `E3HybridHead`  [EXTRACTED]
  e3_quantum_fusion/scripts/train_e3.py → e3_quantum_fusion/models.py

## Import Cycles
- None detected.

## Communities (6 total, 4 thin omitted)

### Community 3 - "E3HybridHead"
Cohesion: 0.28
Nodes (3): count_param_split(), E3HybridHead, main()

### Community 4 - "E3 Quantum Fusion"
Cohesion: 0.25
Nodes (7): Completed Runs, Design (locked), E3 Quantum Fusion, Inputs (reuse E2 caches, do not recompute), Layout, Scoring, VQC (locked first pass)

## Knowledge Gaps
- **6 isolated node(s):** `Design (locked)`, `VQC (locked first pass)`, `Inputs (reuse E2 caches, do not recompute)`, `Layout`, `Completed Runs` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 28 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `E3HybridHead` connect `E3HybridHead` to `E2MatchedHead`, `train_e3.py`, `models.py`, `VQCLayer`?**
  _High betweenness centrality (0.175) - this node is a cross-community bridge._
- **Why does `VQCLayer` connect `VQCLayer` to `E2MatchedHead`, `models.py`, `E3HybridHead`?**
  _High betweenness centrality (0.163) - this node is a cross-community bridge._
- **Why does `E2MatchedHead` connect `E2MatchedHead` to `train_e3.py`, `models.py`, `E3HybridHead`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `E3HybridHead` (e.g. with `VQCLayer` and `main()`) actually correct?**
  _`E3HybridHead` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Design (locked)`, `VQC (locked first pass)`, `Inputs (reuse E2 caches, do not recompute)` to the rest of the system?**
  _6 weakly-connected nodes found - possible documentation gaps or missing edges._