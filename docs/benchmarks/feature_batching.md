# Front-end throughput: per-sample loop vs `extract_batch`

Wall-clock to featurize **128** windows on this host (best of 5 trials). `extract_batch` runs the log-mel front-end as a single batched `MelSpectrogram` call instead of one call per window.

| path | time (ms) | per-window (ms) |
|---|---|---|
| per-sample `extract` loop | 42.909 | 0.3352 |
| `extract_batch` | 13.561 | 0.1059 |

**Speedup: 3.16×** building 128 windows. ✅ parity (max |Δ| 0.00e+00 ≤ 1e-05).

Host throughput only — the device extractor streams one window at a time and is unchanged. The win compounds across every dataset / robustness / A-B build (and the autonomy loop's own test suite).
