# Graph Report - e3_early_rescue  (2026-10-02)

## Corpus Check
- 7 files · ~3,433 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 22 file(s) not represented in the graph (top: .csv 16, .pt 4, (none) 2)

## Summary
- 35 nodes · 43 edges · 5 communities (3 shown, 2 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `da7ad651`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- compare_rescue.py
- E3 Early-Detection Rescue
- FocalLoss
- main

## God Nodes (most connected - your core abstractions)
1. `E3 Early-Detection Rescue` - 6 edges
2. `main()` - 5 edges
3. `FocalLoss` - 4 edges
4. `load_metrics()` - 3 edges
5. `main()` - 3 edges
6. `make_criterion()` - 3 edges
7. `read_metrics()` - 2 edges
8. `fmt()` - 2 edges
9. `parse_args()` - 2 edges
10. `train_one_epoch()` - 2 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `make_criterion()`  [EXTRACTED]
  e3_early_rescue/scripts/train_rescue.py → e3_early_rescue/scripts/train_rescue.py  _Bridges community 3 → community 4_

## Import Cycles
- None detected.

## Communities (5 total, 2 thin omitted)

### Community 1 - "compare_rescue.py"
Cohesion: 0.36
Nodes (4): fmt(), load_metrics(), main(), read_metrics()

### Community 2 - "E3 Early-Detection Rescue"
Cohesion: 0.29
Nodes (6): Compare Runs, E3 Early-Detection Rescue, Loss Variants, Notes, Recommended Manual Runs, What This Folder Runs

### Community 4 - "main"
Cohesion: 0.50
Nodes (4): main(), metric_value(), parse_args(), train_one_epoch()

## Knowledge Gaps
- **5 isolated node(s):** `What This Folder Runs`, `Loss Variants`, `Recommended Manual Runs`, `Compare Runs`, `Notes`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 19 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `FocalLoss` connect `FocalLoss` to `train_rescue.py`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **Why does `main()` connect `main` to `train_rescue.py`, `FocalLoss`?**
  _High betweenness centrality (0.005) - this node is a cross-community bridge._
- **What connects `What This Folder Runs`, `Loss Variants`, `Recommended Manual Runs` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._