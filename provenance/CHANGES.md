# Extraction changes

- New repository entrypoints, dependency records, attribution, source manifests, artifact manifest/fetcher and standalone verifier. Raw datasets, caches, credentials, runtime logs, unrelated tracks and smoke/benchmark-only outputs excluded.

- Small required helpers are frozen copies; metrics, splits, checkpoint tensors, prediction CSVs and embedding arrays are preserved byte-for-byte. Missing historical Markdown links resolve to source-commit permalinks when their targets exist in the parent repository.

- Large frozen embedding arrays are local-only and distributed once through the private baseline embeddings-v1 release. Each consumer checks its required file hashes.

- E3 experiment code and paired controls retained. E2 training/loading helpers copied locally. Baseline NPZs and all retained warm-start/improvement checkpoints copied with hashes. No scientific training change.
