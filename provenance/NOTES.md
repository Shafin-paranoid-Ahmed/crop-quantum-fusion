# Implementation notes

- Small shared helpers are versioned copies; metrics, splits, checkpoint tensors, prediction CSVs and embedding arrays are kept byte-for-byte as produced by the recorded runs.

- Large frozen embedding arrays are kept in a local bundle (`.artifacts/embeddings-v1.zip`); no release is published. Each consumer checks its required file hashes.

- E3 experiment code and paired controls are included; E2 training/loading helpers are local copies. Baseline NPZs and all retained warm-start/improvement checkpoints are included with hashes. No scientific training change.
