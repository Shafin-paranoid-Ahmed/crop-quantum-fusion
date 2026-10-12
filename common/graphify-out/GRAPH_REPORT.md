# Graph Report - common  (2026-10-02)

## Corpus Check
- 3 files · ~869 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 31 nodes · 34 edges · 7 communities (1 shown, 6 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `da7ad651`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- E1MLP
- metrics_dict
- expected_calibration_error
- classification_metrics_from_labels
- save_predictions

## God Nodes (most connected - your core abstractions)
1. `E1MLP` - 5 edges
2. `metrics_dict()` - 5 edges
3. `expected_calibration_error()` - 3 edges
4. `expected_calibration_error_from_confidence()` - 3 edges
5. `stage_recall()` - 3 edges
6. `label_to_stage()` - 2 edges
7. `classification_metrics_from_labels()` - 2 edges
8. `multiclass_roc_auc()` - 2 edges
9. `save_predictions()` - 2 edges
10. `Shared thesis infrastructure (evaluation metrics and models). Importable from…` - 1 edges

## Surprising Connections (you probably didn't know these)
- `metrics_dict()` --calls--> `expected_calibration_error()`  [EXTRACTED]
  common/thesis_eval.py → common/thesis_eval.py  _Bridges community 3 → community 2_

## Import Cycles
- None detected.

## Communities (7 total, 6 thin omitted)

### Community 2 - "metrics_dict"
Cohesion: 0.40
Nodes (4): label_to_stage(), metrics_dict(), multiclass_roc_auc(), stage_recall()

## Knowledge Gaps
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `metrics_dict()` connect `metrics_dict` to `thesis_eval.py`, `expected_calibration_error`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Why does `expected_calibration_error_from_confidence()` connect `expected_calibration_error` to `thesis_eval.py`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Why does `classification_metrics_from_labels()` connect `classification_metrics_from_labels` to `thesis_eval.py`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._